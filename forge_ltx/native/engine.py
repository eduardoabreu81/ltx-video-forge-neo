"""LTX-2.x diffusion engine for Forge Neo: text-to-video with audio.

Video [1, 128, F, H/32, W/32] and audio [1, 8, T, 16] latents travel through Forge's sampler packed into one flat
tensor [1, 1, 1, N], as ComfyUI does (comfy.utils.pack_latents). The script sets up a generation with prepare(), which
returns the packed shape; the transformer unpacks it every step (forge_dit.py); decode_first_stage decodes both
streams, keeps the frames and the waveform for the script, and hands Forge the first frame.
"""

import contextlib

import torch
from backend import memory_management
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.patcher.clip import CLIP
from backend.patcher.unet import UnetPatcher
from backend.patcher.vae import VAE

from ..contracts import keyframe_defaults, raise_pending_error
from . import guides
from .model import LTXAV, TEXT_ENCODER
from .streams import Generation, stream_shapes
from .text_engine import LTXTextEngine

# ComfyUI's estimate of the text encoder's working memory: per token, half of it in 16-bit (lt.memory_estimation_function)
TEXT_MEMORY_PER_TOKEN = 3 * 1024 * 1024
MIN_TEXT_TOKENS = 642
# ComfyUI's estimate for the audio VAE and the vocoder (sd.py, "LTX Audio"), per latent element in fp32
AUDIO_MEMORY_PER_ELEMENT = 87000 * 4
# tiled video decode, in latent units: ComfyUI's LTX tiled decode uses 512-pixel tiles with 64 pixels of overlap
DECODE_TILE = 16
DECODE_TILE_OVERLAP = 2


def encode_memory(pixel_shape, dtype) -> int:
    """ComfyUI's working memory estimate for the LTX video encoder, for pixels [B, 3, F, H, W]."""
    return int(80 * max(pixel_shape[2], 7) * pixel_shape[3] * pixel_shape[4] * memory_management.dtype_size(dtype))


def _video_vae_dtype() -> torch.dtype:
    # ComfyUI runs the LTX video VAE in bf16 or fp32
    return memory_management.vae_dtype(allowed_dtypes=[torch.bfloat16, torch.float32])


def video_vae_patcher(model, dtype) -> VAE:
    """Forge's VAE object around the LTX video VAE, with what its tiled decode (Never OOM, and the fallback after an
    out-of-memory error) needs to know about the LTX latent: 8x in time after the first frame, 32x in space (ComfyUI
    sd.py, lightricks ltxv)."""
    vae = VAE(model=model, dtype=dtype, is_wan=True)
    vae.latent_channels = 128
    vae.upscale_ratio = (lambda a: max(0, a * 8 - 7), 32, 32)
    vae.upscale_index_formula = (8, 32, 32)
    vae.downscale_ratio = (lambda a: max(0, (a + 7) // 8), 32, 32)
    vae.downscale_index_formula = (8, 32, 32)
    vae.memory_used_decode = lambda shape, dtype: model.decode_memory(shape, dtype, vae.device)
    return vae


@contextlib.contextmanager
def _peak_memory(stage: str, device: torch.device, reserved: int):
    """Logs the working memory a stage really took on a CUDA device, next to what was reserved for it."""
    if getattr(device, "type", None) != "cuda":
        yield
        return
    torch.cuda.synchronize(device)
    before = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    yield
    peak = torch.cuda.max_memory_allocated(device) - before
    print(f"[LTX] {stage}: {peak / 2**30:.2f} GiB working memory (reserved {reserved / 2**30:.2f} GiB)")


class LTXEngine(ForgeDiffusionEngine):
    matched_guesses = [LTXAV]

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)

        text_encoder = huggingface_components["text_encoder"]
        clip = CLIP(model_dict={TEXT_ENCODER: text_encoder}, tokenizer_dict={TEXT_ENCODER: text_encoder.ltx_tokenizer})

        pair = huggingface_components["vae"]
        vae = video_vae_patcher(pair.video, _video_vae_dtype())
        # the audio VAE and the vocoder in fp32, as ComfyUI keeps them
        self.audio_vae = VAE(model=pair.audio, dtype=torch.float32, is_wan=True)

        unet = UnetPatcher.from_model(model=huggingface_components["transformer"], diffusers_scheduler=None,
                                      k_predictor=self._get_predictor(), config=estimated_config)

        self.text_processing_engine_ltx = LTXTextEngine(getattr(clip.cond_stage_model, TEXT_ENCODER), text_encoder.ltx_tokenizer,
                                                        text_encoder.text_embedding_projection)

        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

        self.is_ltx = True
        self.ltx_version = estimated_config.unet_config.get("version", "2.5")
        self.generation: Generation | None = None
        # first / last frame settings for the next generation (set_keyframes), and the img2img input image Forge
        # hands over through encode_first_stage, (1, H, W, 3) in [0, 1]
        self.first_frame: torch.Tensor | None = None
        self.last_frame: torch.Tensor | None = None
        self.keyframe_strength, self.keyframe_compression = keyframe_defaults(self.ltx_version)

    def set_keyframes(self, last_frame: torch.Tensor | None = None, strength: float | None = None, compression: int | None = None) -> None:
        """Called by the script for every generation, before Forge's img2img init calls encode_first_stage."""
        default_strength, default_compression = keyframe_defaults(self.ltx_version)
        self.first_frame = None
        self.last_frame = last_frame
        self.keyframe_strength = default_strength if strength is None else float(strength)
        self.keyframe_compression = default_compression if compression is None else int(compression)

    def keyframes(self) -> list[guides.Keyframe]:
        """The conditioning images of the next generation: the first frame in place alone, appended guides at the first
        and the last frame together (the ComfyUI image-to-video and first-last-frame templates)."""
        strength = self.keyframe_strength
        if self.first_frame is not None and self.last_frame is None:
            return [guides.Keyframe(self.first_frame, 0, strength, in_place=True)]
        return [guides.Keyframe(image, frame, strength) for image, frame in ((self.first_frame, 0), (self.last_frame, -1))
                if image is not None]

    @torch.inference_mode()
    def _encode_keyframes(self, keyframes: list[guides.Keyframe]) -> list[torch.Tensor]:
        # each image on its own, one latent frame [1, 128, 1, H/32, W/32]
        video_vae = self.forge_objects.vae
        largest = max(keyframes, key=lambda k: k.image.shape[1] * k.image.shape[2]).image
        memory_management.load_models_gpu([video_vae.patcher], memory_required=encode_memory((1, 3, 1, *largest.shape[1:3]), video_vae.vae_dtype))
        latents = []
        for keyframe in keyframes:
            image = guides.compress(keyframe.image[0].float(), self.keyframe_compression)
            pixels = image.movedim(-1, 0)[None, :, None].mul(2.0).sub(1.0)  # [1, 3, 1, H, W] in [-1, 1]
            latent = video_vae.first_stage_model.encode(pixels.to(video_vae.device, video_vae.vae_dtype), device=video_vae.device)
            latents.append(latent.float().cpu())
        return latents

    def prepare(self, frames: int, width: int, height: int, fps: float, seed: int) -> tuple[int, ...]:
        """Start a generation: remember its shapes and encode its keyframes; returns the shape of the packed latent
        (without the batch)."""
        keyframes = self.keyframes()
        for keyframe in keyframes:
            if tuple(keyframe.image.shape[1:3]) != (height, width):
                raise RuntimeError(f"[LTX] a keyframe is {keyframe.image.shape[2]}x{keyframe.image.shape[1]}, not the output size {width}x{height}")
        guide_frames = sum(1 for k in keyframes if not k.in_place)
        shapes = stream_shapes(frames, width, height, fps, guide_frames)
        conditioning = None
        if keyframes:
            video_shape = (1, shapes.video[1], shapes.video[2] - guide_frames, *shapes.video[3:])
            conditioning = guides.build(video_shape, frames, keyframes, self._encode_keyframes(keyframes))
        self.generation = Generation(shapes=shapes, seed=seed, fps=float(fps), frames=frames, conditioning=conditioning)
        self.forge_objects.unet.model.diffusion_model.generation = self.generation
        return (1, 1, shapes.size)

    def release_generation(self) -> None:
        """Drop the decoded frames and waveform, and the keyframes, once the script has written them."""
        self.generation = None
        self.first_frame = self.last_frame = None
        self.forge_objects.unet.model.diffusion_model.generation = None

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: list[str]):
        raise_pending_error()
        tokens = max([MIN_TEXT_TOKENS] + [len(self.text_processing_engine_ltx.tokenize(text)) + 1 for text in prompt])
        memory_management.load_models_gpu([self.forge_objects.clip.patcher], memory_required=tokens * TEXT_MEMORY_PER_TOKEN)
        return self.text_processing_engine_ltx(prompt)

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt: str) -> tuple[int, int]:
        token_count = len(self.text_processing_engine_ltx.tokenize(prompt))
        return token_count, max(1024, token_count)

    @torch.inference_mode()
    def encode_first_stage(self, x: torch.Tensor):
        # Forge's img2img init hands over the input image, already resized to the output size: it becomes the first
        # frame, encoded in prepare(); the placeholder latent is replaced by the packed one before sampling
        raise_pending_error()
        self.first_frame = x[:1].float().mul(0.5).add(0.5).clamp(0.0, 1.0).movedim(1, -1).cpu()
        return torch.zeros((1, 1, 1, 1), device=x.device)

    def _decode_video(self, video: torch.Tensor) -> torch.Tensor:
        """Frames [T, 3, H, W] in [0, 1] from the video latent. The whole clip at once, streaming through time; Never
        OOM's "always tiled" VAE switch, or running out of memory twice, takes Forge's own tiled decode instead."""
        video_vae = self.forge_objects.vae
        if memory_management.VAE_ALWAYS_TILED:
            return self._decode_tiled(video)
        reserve = video_vae.first_stage_model.decode_memory(video.shape, video_vae.vae_dtype, video_vae.device)
        for attempt in range(2):
            try:
                memory_management.load_models_gpu([video_vae.patcher], memory_required=reserve)
                with _peak_memory("video decode", video_vae.device, reserve):
                    pixels = video_vae.first_stage_model.decode(video.to(video_vae.device, video_vae.vae_dtype))  # [1, 3, T, H, W] in [-1, 1]
                return pixels[0].float().add(1.0).mul(0.5).clamp(0.0, 1.0).cpu().movedim(1, 0)
            except Exception as e:
                if not memory_management.is_oom(e):
                    raise
                pixels = None
                memory_management.soft_empty_cache(force=True)
                if attempt == 0:
                    # the estimate fell short: the DiT and the text encoder leave VRAM (they reload from their files)
                    print(f"[LTX] video decode ran out of memory with {reserve / 2**30:.2f} GiB reserved: retrying with the other models unloaded")
                    memory_management.unload_all_models()
                    reserve *= 2
        print("[LTX] video decode ran out of memory again: decoding in tiles")
        return self._decode_tiled(video)

    def _decode_tiled(self, video: torch.Tensor) -> torch.Tensor:
        video_vae = self.forge_objects.vae
        print(f"[LTX] video decode in tiles of {DECODE_TILE * 32} pixels")
        frames = video_vae.decode_tiled(video, tile_x=DECODE_TILE, tile_y=DECODE_TILE, overlap=DECODE_TILE_OVERLAP)  # [1, T, H, W, 3] in [0, 1]
        return frames[0].float().clamp(0.0, 1.0).cpu().movedim(-1, 1)

    @torch.inference_mode()
    def decode_first_stage(self, x: torch.Tensor):
        generation = self.generation
        if generation is None or x.shape[-1] != generation.shapes.size:
            raise RuntimeError("[LTX] the latent does not belong to the current LTX generation")
        video, audio = generation.shapes.unpack(x[:1].float())
        if generation.conditioning is not None and generation.conditioning.guide_frames:
            # the appended guide frames are not part of the clip (ComfyUI LTXVCropGuides)
            video = video[:, :, :video.shape[2] - generation.conditioning.guide_frames]

        generation.decoded = self._decode_video(video)


        memory_management.load_models_gpu([self.audio_vae.patcher], memory_required=audio.shape[2] * audio.shape[3] * AUDIO_MEMORY_PER_ELEMENT)
        audio_model = self.audio_vae.first_stage_model
        generation.waveform = audio_model.decode(audio.to(self.audio_vae.device, torch.float32))[0].float().cpu()
        generation.sample_rate = audio_model.output_sample_rate

        return generation.decoded[:1].mul(2.0).sub(1.0)
