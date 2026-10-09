"""The LTX latent upscaler of the two-stage pipeline, ported from ComfyUI comfy/ldm/lightricks/latent_upsampler.py.

A small conv net on the video latent: 4 ResBlocks, a 2x spatial pixel shuffle (or a learned rational resampler for
x1.5), 4 more ResBlocks. It works on un-normalized latents, so the caller goes through the video VAE's per-channel
statistics on the way in and out (ComfyUI LTXVLatentUpsampler). The published files carry their settings as JSON in
the "config" metadata; LTX-2.0, 2.3 and 2.5 publish the same x2 layout.
"""

import json

import torch
import torch.nn as nn
import torch.nn.functional as F

RATIONAL = {0.75: (3, 4), 1.5: (3, 2), 2.0: (2, 1), 4.0: (4, 1)}
#: a key only the LTX latent upscalers have (ComfyUI LatentUpscaleModelLoader)
MARKER = "post_upsample_res_blocks.0.conv2.bias"


def is_upsampler(keys) -> bool:
    return MARKER in keys


def pixel_shuffle_2d(x: torch.Tensor, factor: int) -> torch.Tensor:
    # b (c p1 p2) h w -> b c (h p1) (w p2)
    b, c, h, w = x.shape
    x = x.reshape(b, c // (factor * factor), factor, factor, h, w)
    return x.permute(0, 1, 4, 2, 5, 3).reshape(b, c // (factor * factor), h * factor, w * factor)


def pixel_shuffle_3d(x: torch.Tensor, factors=(2, 2, 2)) -> torch.Tensor:
    # b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)
    p1, p2, p3 = factors
    b, c, d, h, w = x.shape
    x = x.reshape(b, c // (p1 * p2 * p3), p1, p2, p3, d, h, w)
    return x.permute(0, 1, 5, 2, 6, 3, 7, 4).reshape(b, c // (p1 * p2 * p3), d * p1, h * p2, w * p3)


def pixel_shuffle_1d(x: torch.Tensor, factor: int) -> torch.Tensor:
    # b (c p1) f h w -> b c (f p1) h w
    b, c, f, h, w = x.shape
    x = x.reshape(b, c // factor, factor, f, h, w)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(b, c // factor, f * factor, h, w)


def _frames_to_batch(x: torch.Tensor) -> torch.Tensor:
    # b c f h w -> (b f) c h w
    b, c, f, h, w = x.shape
    return x.permute(0, 2, 1, 3, 4).reshape(b * f, c, h, w)


def _batch_to_frames(x: torch.Tensor, b: int) -> torch.Tensor:
    # (b f) c h w -> b c f h w
    bf, c, h, w = x.shape
    return x.reshape(b, bf // b, c, h, w).permute(0, 2, 1, 3, 4)


class Shuffle(nn.Module):
    def __init__(self, dims: int, factor: int = 2):
        super().__init__()
        self.dims, self.factor = dims, factor

    def forward(self, x):
        if self.dims == 3:
            return pixel_shuffle_3d(x)
        if self.dims == 2:
            return pixel_shuffle_2d(x, self.factor)
        return pixel_shuffle_1d(x, self.factor)


class BlurDownsample(nn.Module):
    """Anti-aliased spatial downsampling by an integer stride with a fixed binomial kernel, per frame."""

    def __init__(self, stride: int):
        super().__init__()
        self.stride = stride
        k = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0])
        k2d = k[:, None] @ k[None, :]
        # saved in the files (LTX-2.0's upscaler carries it)
        self.register_buffer("kernel", (k2d / k2d.sum()).float()[None, None, :, :])

    def forward(self, x):
        if self.stride == 1:
            return x
        weight = self.kernel.to(x).expand(x.shape[1], 1, 5, 5)
        return F.conv2d(x, weight=weight, bias=None, stride=self.stride, padding=2, groups=x.shape[1])


class SpatialRationalResampler(nn.Module):
    """Up by num with a pixel shuffle, then down by den with the blur: scales H and W by num / den, per frame."""

    def __init__(self, mid_channels: int, scale: float):
        super().__init__()
        self.num, self.den = RATIONAL[float(scale)]
        self.conv = nn.Conv2d(mid_channels, (self.num ** 2) * mid_channels, kernel_size=3, padding=1)
        self.blur_down = BlurDownsample(self.den)

    def forward(self, x):
        b = x.shape[0]
        x = self.blur_down(pixel_shuffle_2d(self.conv(_frames_to_batch(x)), self.num))
        return _batch_to_frames(x, b)


class ResBlock(nn.Module):
    def __init__(self, channels: int, dims: int = 3):
        super().__init__()
        conv = nn.Conv2d if dims == 2 else nn.Conv3d
        self.conv1 = conv(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(32, channels)
        self.conv2 = conv(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, channels)
        self.activation = nn.SiLU()

    def forward(self, x):
        residual = x
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.activation(x + residual)


class LatentUpsampler(nn.Module):
    """Video latents [B, C, F, H, W] (un-normalized) -> upscaled latents."""

    def __init__(self, in_channels: int = 128, mid_channels: int = 512, num_blocks_per_stage: int = 4, dims: int = 3,
                 spatial_upsample: bool = True, temporal_upsample: bool = False, spatial_scale: float = 2.0,
                 rational_resampler: bool = False):
        super().__init__()
        self.dims = dims
        self.spatial_upsample, self.temporal_upsample = spatial_upsample, temporal_upsample
        self.spatial_scale = float(spatial_scale)
        conv = nn.Conv2d if dims == 2 else nn.Conv3d

        self.initial_conv = conv(in_channels, mid_channels, kernel_size=3, padding=1)
        self.initial_norm = nn.GroupNorm(32, mid_channels)
        self.initial_activation = nn.SiLU()
        self.res_blocks = nn.ModuleList([ResBlock(mid_channels, dims=dims) for _ in range(num_blocks_per_stage)])

        if spatial_upsample and temporal_upsample:
            self.upsampler = nn.Sequential(nn.Conv3d(mid_channels, 8 * mid_channels, kernel_size=3, padding=1), Shuffle(3))
        elif spatial_upsample and rational_resampler:
            self.upsampler = SpatialRationalResampler(mid_channels, self.spatial_scale)
        elif spatial_upsample:
            self.upsampler = nn.Sequential(nn.Conv2d(mid_channels, 4 * mid_channels, kernel_size=3, padding=1), Shuffle(2))
        elif temporal_upsample:
            self.upsampler = nn.Sequential(nn.Conv3d(mid_channels, 2 * mid_channels, kernel_size=3, padding=1), Shuffle(1))
        else:
            raise ValueError("[LTX] a latent upscaler scales space, time or both")

        self.post_upsample_res_blocks = nn.ModuleList([ResBlock(mid_channels, dims=dims) for _ in range(num_blocks_per_stage)])
        self.final_conv = conv(mid_channels, in_channels, kernel_size=3, padding=1)

    @classmethod
    def from_config(cls, config: dict) -> "LatentUpsampler":
        # ComfyUI's defaults for keys a config leaves out
        return cls(in_channels=config.get("in_channels", 4), mid_channels=config.get("mid_channels", 128),
                   num_blocks_per_stage=config.get("num_blocks_per_stage", 4), dims=config.get("dims", 2),
                   spatial_upsample=config.get("spatial_upsample", True), temporal_upsample=config.get("temporal_upsample", False),
                   spatial_scale=config.get("spatial_scale", 2.0), rational_resampler=config.get("rational_resampler", False))

    def _stem(self, x):
        x = self.initial_activation(self.initial_norm(self.initial_conv(x)))
        for block in self.res_blocks:
            x = block(x)
        return x

    def _head(self, x):
        for block in self.post_upsample_res_blocks:
            x = block(x)
        return self.final_conv(x)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        b = latent.shape[0]
        if self.dims == 2:
            x = self._stem(_frames_to_batch(latent))
            return _batch_to_frames(self._head(self.upsampler(x)), b)
        x = self._stem(latent)
        if self.temporal_upsample:
            x = self.upsampler(x)[:, :, 1:]
        elif isinstance(self.upsampler, SpatialRationalResampler):
            x = self.upsampler(x)
        else:
            x = _batch_to_frames(self.upsampler(_frames_to_batch(x)), b)
        return self._head(x)


def config_from_metadata(metadata: dict | None) -> dict:
    if not metadata or "config" not in metadata:
        raise ValueError("[LTX] this latent upscaler file has no config in its metadata; use the files LTX publishes")
    return json.loads(metadata["config"])


def is_spatial_x2(config: dict) -> bool:
    """The upscaler the two-stage pipeline takes: twice the width and height, the same frames."""
    return (config.get("spatial_upsample", True) and not config.get("temporal_upsample", False)
            and float(config.get("spatial_scale", 2.0)) == 2.0)
