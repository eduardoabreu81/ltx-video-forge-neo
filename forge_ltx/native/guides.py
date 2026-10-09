"""First and last frames as LTX conditioning: image-to-video and first-last-frame-to-video.

As the ComfyUI LTX templates do it:
- image-to-video (LTXVImgToVideoInplace): the encoded image replaces the first latent frame;
- first and last frame (LTXVAddGuide at index 0 and -1): each encoded image is appended after the clip as a guide
  frame with its own pixel coordinates (keyframe_idxs), and dropped again before decoding (LTXVCropGuides).
Either way a denoise mask holds 1 - strength on the conditioned frames: the sampler feeds them in mixed with their
clean latent and the model sees them at a timestep of sigma * mask (KSamplerX0Inpaint and LTXAV.process_timestep),
which mix() and timesteps() reproduce inside the DiT's forward pass.

Images go through LTXVPreprocess first: one H.264 frame at a given CRF, a light compression that keeps the video from
copying the picture's clean detail (and freezing on it).
"""

import io
from dataclasses import dataclass

import numpy as np
import torch

from .streams import SPATIAL, TEMPORAL, latent_frames
from .transformer import SymmetricPatchifier, latent_to_pixel_coords

SCALE_FACTORS = (TEMPORAL, SPATIAL, SPATIAL)


def compress(image: torch.Tensor, crf: int) -> torch.Tensor:
    """ComfyUI's LTXVPreprocess: (H, W, 3) float in [0, 1] through one H.264 frame at this CRF (0 = untouched)."""
    if crf <= 0:
        return image
    import av

    height, width = image.shape[0] // 2 * 2, image.shape[1] // 2 * 2
    array = (image[:height, :width] * 255.0).byte().cpu().numpy()
    with io.BytesIO() as buffer:
        container = av.open(buffer, "w", format="mp4")
        try:
            stream = container.add_stream("libx264", rate=1, options={"crf": str(int(crf)), "preset": "veryfast"})
            stream.height, stream.width = height, width
            frame = av.VideoFrame.from_ndarray(array, format="rgb24").reformat(format="yuv420p")
            container.mux(stream.encode(frame))
            container.mux(stream.encode())
        finally:
            container.close()
        data = buffer.getvalue()
    with io.BytesIO(data) as buffer:
        container = av.open(buffer)
        try:
            stream = next(s for s in container.streams if s.type == "video")
            decoded = next(container.decode(stream)).to_ndarray(format="rgb24")
        finally:
            container.close()
    return torch.from_numpy(np.ascontiguousarray(decoded)).to(image.dtype) / 255.0


def fit(image: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """(1, H, W, 3) at another size, as ComfyUI's LTX conditioning nodes resize their image (common_upscale, bilinear,
    center crop): the first stage of a two-stage generation works at half the output size."""
    if tuple(image.shape[1:3]) == (height, width):
        return image
    x = image.movedim(-1, 1)
    old_h, old_w = x.shape[2], x.shape[3]
    old_aspect, new_aspect = old_w / old_h, width / height
    left = round((old_w - old_w * (new_aspect / old_aspect)) / 2) if old_aspect > new_aspect else 0
    top = round((old_h - old_h * (old_aspect / new_aspect)) / 2) if old_aspect < new_aspect else 0
    x = x[:, :, top:old_h - top, left:old_w - left]
    x = torch.nn.functional.interpolate(x.float(), size=(height, width), mode="bilinear")
    return x.movedim(1, -1).to(image.dtype)


@dataclass
class Keyframe:
    """One conditioning image: (1, H, W, 3) float in [0, 1] at the output size, the pixel frame it stands for (0 or
    -1 for the last), and how strongly it holds."""
    image: torch.Tensor
    frame: int
    strength: float
    in_place: bool = False


@dataclass
class Conditioning:
    """What the DiT needs on every step, for latents [1, 128, F + G, H, W] with G appended guide frames."""
    latent: torch.Tensor          # clean latents of the conditioned frames, zeros elsewhere
    mask: torch.Tensor            # [1, 1, F + G, H, W]: 1 generates, 1 - strength holds a conditioned frame
    keyframe_idxs: torch.Tensor | None   # [1, 3, G * H * W, 2] pixel coordinates of the guide tokens
    guide_entries: list | None    # per guide, for the guide attention strength (ComfyUI guide_attention_entries)
    guide_frames: int             # G, cropped before decoding

    def to(self, device, dtype) -> "Conditioning":
        return Conditioning(self.latent.to(device, dtype), self.mask.to(device, dtype),
                            None if self.keyframe_idxs is None else self.keyframe_idxs.to(device),
                            self.guide_entries, self.guide_frames)

    def mix(self, video: torch.Tensor) -> torch.Tensor:
        """The sampler's latent with the conditioned frames pulled to their clean latent (KSamplerX0Inpaint)."""
        return video * self.mask + self.latent * (1.0 - self.mask)

    def place(self, video: torch.Tensor) -> torch.Tensor:
        """A start latent with the clean latents in the conditioned frames, as ComfyUI's LTXVImgToVideoInplace and
        LTXVAddGuide leave them in the latent a sampler starts from; video holds the clip's frames, without guides."""
        guides = self.latent.shape[2] - video.shape[2]
        if guides:
            video = torch.cat([video, video.new_zeros((*video.shape[:2], guides, *video.shape[3:]))], dim=2)
        held = (self.mask < 1.0).to(video.dtype)
        return video * (1.0 - held) + self.latent.to(video) * held

    def timesteps(self, sigma: torch.Tensor) -> torch.Tensor:
        """Per-token video timesteps [B, tokens]: sigma * mask (LTXAV.process_timestep)."""
        per_frame = sigma.reshape(-1, 1, 1, 1, 1) * self.mask
        return per_frame.flatten(2).reshape(sigma.shape[0], -1)

    def model_kwargs(self, batch: int) -> dict:
        kwargs = {"denoise_mask": self.mask.expand(batch, -1, -1, -1, -1)}
        if self.keyframe_idxs is not None:
            kwargs["keyframe_idxs"] = self.keyframe_idxs.expand(batch, -1, -1, -1)
            kwargs["guide_attention_entries"] = self.guide_entries
        return kwargs


def resolve_frame(frame: int, frames: int) -> int:
    """A pixel frame index, negative ones counted from the end (ComfyUI LTXVAddGuide)."""
    return frame if frame >= 0 else max(frames + frame, 0)


def build(video_shape: tuple[int, ...], frames: int, keyframes: list[Keyframe], latents: list[torch.Tensor]) -> Conditioning:
    """The conditioning for a clip of frames pixel frames with video latent shape [1, 128, F, H, W], from the
    keyframes and their encoded latents [1, 128, 1, H, W]."""
    _, channels, f, h, w = video_shape
    if f != latent_frames(frames):
        raise ValueError("[LTX] the video latent does not match the frame count")
    guides = [(k, z) for k, z in zip(keyframes, latents) if not k.in_place]
    total = f + len(guides)
    latent = torch.zeros((1, channels, total, h, w), dtype=torch.float32)
    mask = torch.ones((1, 1, total, h, w), dtype=torch.float32)
    for keyframe, z in zip(keyframes, latents):
        if keyframe.in_place:
            latent[:, :, :1] = z
            mask[:, :, :1] = max(0.0, 1.0 - keyframe.strength)

    coords, entries = [], []
    patchifier = SymmetricPatchifier(1, start_end=True)
    for i, (keyframe, z) in enumerate(guides):
        latent[:, :, f + i:f + i + 1] = z
        mask[:, :, f + i:f + i + 1] = max(0.0, 1.0 - keyframe.strength)
        pixel = latent_to_pixel_coords(patchifier.latent_coords(1, h, w, 1, torch.device("cpu")), SCALE_FACTORS, causal_fix=True)
        pixel[:, 0] += resolve_frame(keyframe.frame, frames)
        coords.append(pixel)
        entries.append({"pre_filter_count": h * w, "strength": keyframe.strength, "pixel_mask": None, "latent_shape": [1, h, w]})
    keyframe_idxs = torch.cat(coords, dim=2) if coords else None
    return Conditioning(latent, mask, keyframe_idxs, entries or None, len(guides))


def describe(keyframes: list[Keyframe]) -> str:
    names = {0: "first", -1: "last"}
    return " and ".join(f"{names.get(k.frame, f'frame {k.frame}')} frame" for k in keyframes)
