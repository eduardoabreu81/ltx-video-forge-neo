"""LTX-2.x diffusion engine for Forge Neo: text-to-video with audio.

Video [1, 128, F, H/32, W/32] and audio [1, 8, T, 16] latents travel through Forge's sampler packed into one flat
tensor [1, 1, 1, N], as ComfyUI does (comfy.utils.pack_latents). The script sets up a generation with prepare(), which
returns the packed shape; the transformer unpacks it every step (forge_dit.py); decode_first_stage decodes both
streams, keeps the frames and the waveform for the script, and hands Forge the first frame.

Two stages (ComfyUI's LTX templates): the first generation runs at half the output size; refine_start() upscales its
video latent x2 with the latent upscaler (upsampler.py) and sets up the second generation at the output size, which
the script samples from there on the refining sigmas.
"""

import contextlib
import logging
import os

import torch
from backend import memory_management
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.patcher.base import ModelPatcher
from backend.patcher.clip import CLIP
from backend.patcher.unet import UnetPatcher
from backend.patcher.vae import VAE

from ..contracts import keyframe_defaults, raise_pending_error
from . import guides, upsampler
from .model import LTXAV, TEXT_ENCODER
from .streams import Generation, stream_shapes
from .text_engine import LTXTextEngine

# ComfyUI's estimate of the text encoder's working memory: per token, half of it in 16-bit (lt.memory_estimation_function)
TEXT_MEMORY_PER_TOKEN = 3 * 1024 * 1024
MIN_TEXT_TOKENS = 642
# ComfyUI's estimate for the audio VAE and the vocoder (sd.py, "LTX Audio"), per latent element in fp32
AUDIO_MEMORY_PER_ELEMENT = 87000 * 4
# tiled video decode, in latent units: ComfyUI's LTX tiled decode uses 512-pixel tiles with 64 pixels of overlap;
# smaller tiles when even those do not fit the card
DECODE_TILE = 16
DECODE_TILES = (16, 12, 8, 6)
DECODE_TILE_OVERLAP = 2
# ComfyUI's estimate for the latent upscaler, per input latent element (LTXVLatentUpsampler)
UPSCALE_MEMORY_PER_ELEMENT = 3000
# LTX_DEBUG_MEMORY=1 in Forge's environment prints the working memory each stage really took (CUDA only)
DEBUG_MEMORY = bool(os.environ.get("LTX_DEBUG_MEMORY"))

logger = logging.getLogger("forge_ltx")


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
    # Forge's tiled decode reserves memory for the shape it is given; with a tile set, for one tile
    vae.ltx_tile = None
    vae.memory_used_decode = lambda shape, dtype: model.decode_memory(_tile_shape(shape, vae.ltx_tile), dtype, vae.device,
                                                                      tiled=vae.ltx_tile is not None)
    return vae


def _tile_shape(shape, tile):
    if tile is None:
        return tuple(shape)
    side = tile + 2 * DECODE_TILE_OVERLAP
    return (*shape[:3], min(shape[3], side), min(shape[4], side))


def decode_plan(model, shape, dtype, room: int) -> int | None:
    """None when the whole clip fits in room bytes of working memory, else the largest tile (in latent units) that does."""
    if model.decode_memory(shape, dtype, None) <= room:
        return None
    for tile in DECODE_TILES:
        if model.decode_memory(_tile_shape(shape, tile), dtype, None, tiled=True) <= room:
            return tile
    return DECODE_TILES[-1]


@contextlib.contextmanager
def _peak_memory(stage: str, device: torch.device, reserved: int):
    """With LTX_DEBUG_MEMORY, logs the working memory a stage really took on a CUDA device, next to what was reserved."""
    if not DEBUG_MEMORY or getattr(device, "type", None) != "cuda":
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
        # the latent upscaler of two-stage generations, (path, patcher) of the last one used
        self._upscaler: tuple[str, ModelPatcher] | None = None

    def set_keyframes(self, last_frame: torch.Tensor | None = None, strength: float | None = None, compression: int | None = None) -> None:
        """Called by the script for every generation, before Forge's img2img init calls encode_first_stage."""
        default_strength, default_compression = keyframe_defaults(self.ltx_version)
        self.first_frame = None
        self.last_frame = last_frame
        self.keyframe_strength = default_strength if strength is None else float(strength)
        self.keyframe_compression = default_compression if compression is None else int(compression)

    def keyframes(self, refine: bool = False) -> list[guides.Keyframe]:
        """The conditioning images of the next generation: the first frame in place alone, appended guides at the first
        and the last frame together (the ComfyUI image-to-video and first-last-frame templates). The second stage of
        the image-to-video template holds the first frame at full strength."""
        strength = self.keyframe_strength
        if self.first_frame is not None and self.last_frame is None:
            return [guides.Keyframe(self.first_frame, 0, 1.0 if refine else strength, in_place=True)]
        return [guides.Keyframe(image, frame, strength) for image, frame in ((self.first_frame, 0), (self.last_frame, -1))
                if image is not None]

    @torch.inference_mode()
    def _encode_keyframes(self, keyframes: list[guides.Keyframe], height: int, width: int) -> list[torch.Tensor]:
        # each image on its own, one latent frame [1, 128, 1, H/32, W/32]; compressed at its own size, then fitted to
        # the generation's (LTXVPreprocess, then the conditioning node's resize)
        video_vae = self.forge_objects.vae
        memory_management.load_models_gpu([video_vae.patcher], memory_required=encode_memory((1, 3, 1, height, width), video_vae.vae_dtype))
        latents = []
        for keyframe in keyframes:
            image = guides.compress(keyframe.image[0].float(), self.keyframe_compression)
            image = guides.fit(image[None], height, width)[0]
            pixels = image.movedim(-1, 0)[None, :, None].mul(2.0).sub(1.0)  # [1, 3, 1, H, W] in [-1, 1]
            latent = video_vae.first_stage_model.encode(pixels.to(video_vae.device, video_vae.vae_dtype), device=video_vae.device)
            latents.append(latent.float().cpu())
        return latents

    def prepare(self, frames: int, width: int, height: int, fps: float, seed: int, refine: bool = False) -> tuple[int, ...]:
        """Start a generation of width x height (the output size, or half of it for the first of two stages): remember
        its shapes and encode its keyframes; returns the shape of the packed latent (without the batch)."""
        keyframes = self.keyframes(refine)
        guide_frames = sum(1 for k in keyframes if not k.in_place)
        shapes = stream_shapes(frames, width, height, fps, guide_frames)
        conditioning = None
        if keyframes:
            video_shape = (1, shapes.video[1], shapes.video[2] - guide_frames, *shapes.video[3:])
            conditioning = guides.build(video_shape, frames, keyframes, self._encode_keyframes(keyframes, height, width))
        self.generation = Generation(shapes=shapes, seed=seed, fps=float(fps), frames=frames, conditioning=conditioning)
        self.forge_objects.unet.model.diffusion_model.generation = self.generation
        return (1, 1, shapes.size)

    def _upscaler_patcher(self, path: str) -> ModelPatcher:
        if self._upscaler is not None and self._upscaler[0] == path:
            return self._upscaler[1]
        from safetensors import safe_open
        from safetensors.torch import load_file

        with safe_open(path, framework="pt") as f:
            config = upsampler.config_from_metadata(f.metadata())
        if not upsampler.is_spatial_x2(config):
            raise RuntimeError("[LTX] two stages need a spatial x2 latent upscaler (ltx-*-spatial-upscaler-x2)")
        state_dict = load_file(path)
        if not upsampler.is_upsampler(state_dict):
            raise RuntimeError(f"[LTX] {os.path.basename(path)} is not an LTX latent upscaler")
        self._upscaler = None
        model = upsampler.LatentUpsampler.from_config(config)
        model.load_state_dict(state_dict, strict=True)
        model = model.to(_video_vae_dtype()).eval().requires_grad_(False)
        patcher = ModelPatcher(model, load_device=memory_management.vae_device(), offload_device=memory_management.vae_offload_device())
        self._upscaler = (path, patcher)
        return patcher

    @torch.inference_mode()
    def upscale(self, video: torch.Tensor, path: str) -> torch.Tensor:
        """The normalized video latent [1, 128, F, H, W] at twice the size, through the latent upscaler, which works
        on un-normalized latents (ComfyUI LTXVLatentUpsampler)."""
        patcher = self._upscaler_patcher(path)
        memory_management.load_models_gpu([patcher], memory_required=video.numel() * UPSCALE_MEMORY_PER_ELEMENT)
        model = patcher.model
        statistics = self.forge_objects.vae.first_stage_model.per_channel_statistics
        x = statistics.un_normalize(video.to(patcher.load_device, next(model.parameters()).dtype))
        return statistics.normalize(model(x)).float().cpu()

    @torch.inference_mode()
    def refine_start(self, samples: torch.Tensor, width: int, height: int, upscaler_path: str) -> torch.Tensor:
        """After the first stage: its video latent upscaled x2 and the second generation set up at width x height, with
        the keyframes encoded again at that size; returns the packed latent the second stage starts from (the audio
        latent carries over)."""
        generation = self.generation
        video, audio = generation.shapes.unpack(samples[:1].float().cpu())
        if generation.conditioning is not None and generation.conditioning.guide_frames:
            video = video[:, :, :video.shape[2] - generation.conditioning.guide_frames]
        video = self.upscale(video, upscaler_path)
        self.prepare(generation.frames, width, height, generation.fps, generation.seed, refine=True)
        refined = self.generation
        if refined.conditioning is not None:
            video = refined.conditioning.place(video)
        if tuple(video.shape) != refined.shapes.video:
            raise RuntimeError(f"[LTX] the upscaled latent is {tuple(video.shape)}, not {refined.shapes.video}")
        return refined.shapes.pack(video, audio)

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

    def _decode_room(self) -> int | None:
        """Working memory the video decode can have on its device once every other model has left; None off CUDA."""
        video_vae = self.forge_objects.vae
        if getattr(video_vae.device, "type", None) != "cuda":
            return None
        total = memory_management.get_total_memory(video_vae.device)
        return int(total - video_vae.patcher.model_size() - memory_management.minimum_inference_memory())

    def _decode_video(self, video: torch.Tensor) -> torch.Tensor:
        """Frames [T, 3, H, W] in [0, 1] from the video latent.

        The whole clip at once, streaming through time, with its working memory reserved (Forge moves the DiT and the
        text encoder out as far as needed). When even the whole card cannot hold that, or Never OOM's VAE switch is on,
        Forge's own tiled decode runs with the largest tile that fits. Running out of memory anyway (the estimate is
        calibrated on one card) unloads the other models and tries again, then goes tiled: all of it quietly."""
        video_vae = self.forge_objects.vae
        model, dtype = video_vae.first_stage_model, video_vae.vae_dtype
        room = self._decode_room()
        tile = None if room is None else decode_plan(model, video.shape, dtype, room)
        if memory_management.VAE_ALWAYS_TILED:
            tile = tile or DECODE_TILE
        if tile is not None:
            return self._decode_tiled(video, tile)

        reserve = model.decode_memory(video.shape, dtype, video_vae.device)
        for attempt in range(2):
            try:
                memory_management.load_models_gpu([video_vae.patcher], memory_required=reserve)
                with _peak_memory("video decode", video_vae.device, reserve):
                    pixels = model.decode(video.to(video_vae.device, dtype))  # [1, 3, T, H, W] in [-1, 1]
                return pixels[0].float().add(1.0).mul(0.5).clamp(0.0, 1.0).cpu().movedim(1, 0)
            except Exception as e:
                if not memory_management.is_oom(e):
                    raise
                pixels = None
                memory_management.soft_empty_cache(force=True)
                logger.debug("[LTX] video decode out of memory with %.2f GiB reserved (attempt %d)", reserve / 2**30, attempt + 1)
                if attempt == 0:
                    memory_management.unload_all_models()
                    reserve = max(reserve * 2, room or 0)
        return self._decode_tiled(video, DECODE_TILES[-1] if room is not None else DECODE_TILE)

    def _decode_tiled(self, video: torch.Tensor, tile: int = DECODE_TILE) -> torch.Tensor:
        video_vae = self.forge_objects.vae
        logger.debug("[LTX] video decode in tiles of %d pixels", tile * 32)
        video_vae.ltx_tile = tile
        try:
            with _peak_memory(f"video decode in {tile * 32}-pixel tiles", video_vae.device,
                              video_vae.memory_used_decode(video.shape, video_vae.vae_dtype)):
                frames = video_vae.decode_tiled(video, tile_x=tile, tile_y=tile, overlap=DECODE_TILE_OVERLAP)  # [1, T, H, W, 3] in [0, 1]
        finally:
            video_vae.ltx_tile = None
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
