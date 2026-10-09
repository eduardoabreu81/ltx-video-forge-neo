"""The packed latent of an LTX generation: video [1, 128, F, H/32, W/32] and audio [1, 8, T, 16] in one tensor.

Forge's sampler works on one tensor; ComfyUI packs the two streams the same way (comfy.utils.pack_latents). Both
streams share the sigma, so no rescaling is needed between them.
"""

import math
from dataclasses import dataclass, field

import torch

LATENT_CHANNELS = 128
AUDIO_CHANNELS = 8
AUDIO_FREQUENCY_BINS = 16
AUDIO_LATENTS_PER_SECOND = 25  # 16 kHz, hop 160, 4x downsampling
SPATIAL = 32
TEMPORAL = 8


def latent_frames(frames: int) -> int:
    return (frames - 1) // TEMPORAL + 1


def audio_latents(frames: int, fps: float) -> int:
    # ComfyUI AudioVAE.num_of_latents_from_frames
    return round(frames / fps * AUDIO_LATENTS_PER_SECOND)


@dataclass(frozen=True)
class StreamShapes:
    video: tuple[int, ...]
    audio: tuple[int, ...]

    @property
    def video_size(self) -> int:
        return math.prod(self.video[1:])

    @property
    def size(self) -> int:
        return self.video_size + math.prod(self.audio[1:])

    def pack(self, video: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        # [B, 1, 1, N]: Forge's sampler reads the latent's dims 2 and 3 as an (H, W) area, so it must be 4D
        flat = torch.cat([video.reshape(video.shape[0], -1), audio.reshape(audio.shape[0], -1)], dim=-1)
        return flat.reshape(flat.shape[0], 1, 1, -1)

    def unpack(self, packed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = packed.shape[0]
        flat = packed.reshape(b, -1)
        n = self.video_size
        return flat[:, :n].reshape(b, *self.video[1:]), flat[:, n:].reshape(b, *self.audio[1:])


def stream_shapes(frames: int, width: int, height: int, fps: float, guide_frames: int = 0) -> StreamShapes:
    """The two latents of a clip; guide_frames latent frames are appended to the video for keyframe guides."""
    return StreamShapes(video=(1, LATENT_CHANNELS, latent_frames(frames) + guide_frames, height // SPATIAL, width // SPATIAL),
                        audio=(1, AUDIO_CHANNELS, audio_latents(frames, fps), AUDIO_FREQUENCY_BINS))


def preview_frame(packed: torch.Tensor, shapes: StreamShapes | None) -> torch.Tensor:
    """The middle video frame [1, 128, H/32, W/32] of a packed latent, for Forge's RGB live preview; anything that is
    not the current generation's packed latent comes back unchanged."""
    if shapes is None or packed.shape[-1] != shapes.size:
        return packed
    video, _ = shapes.unpack(packed[:1])
    return video[:, :, video.shape[2] // 2]


@dataclass
class Generation:
    """What the transformer and the decoder need for the current request; set by the engine's prepare()."""
    shapes: StreamShapes
    seed: int
    fps: float
    frames: int
    # the DiT runs the text through its connectors once per prompt; filled on the first step
    contexts: dict = field(default_factory=dict)
    # first / last frame conditioning (guides.Conditioning), None for text-to-video
    conditioning: object = None
    decoded: torch.Tensor | None = None   # [T, 3, H, W] float in [0, 1], after decoding
    waveform: torch.Tensor | None = None  # [2, samples] float in [-1, 1], after decoding
    sample_rate: int = 0
