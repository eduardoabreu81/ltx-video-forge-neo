"""The LTX-2.x video VAE (the convolutional one), ported from ComfyUI comfy/ldm/lightricks/vae
(causal_video_autoencoder.py, causal_conv3d.py, conv_nd_factory.py, pixel_norm.py).

The encoder and decoder stream the clip through time in chunks sized from the device memory: causal 3D convolutions
keep the frames they still need between chunks, and residual paths that run ahead of their main path are added back
once the main path catches up (add_exchange_cache). The cache lives on the modules, keyed by thread, and is cleared
after every call. Conv3d, GroupNorm and LayerNorm are plain torch.nn names that Forge Neo's loader swaps for its own
operations; padding other than zeros is applied here, because Forge Neo's NVIDIA Conv3d path ignores padding_mode.
"""

import math
import threading

import torch
from backend import memory_management
from einops import rearrange
from torch import nn

from .dit import TimestepEmbedder

# region helpers


def _cat(tensors, dim):
    tensors = [t for t in tensors if t is not None and t.shape[dim] > 0]
    if len(tensors) > 1:
        return torch.cat(tensors, dim)
    return tensors[0] if tensors else None


def _split2(tensor, split_point, dim=2):
    return torch.split(tensor, [split_point, tensor.shape[dim] - split_point], dim=dim)


def add_exchange_cache(dest, cache_in, new_input, dim=2):
    """Adds a residual that runs ahead of its main path: the part that lines up with dest goes in now, the rest is
    returned as the cache for the next chunk."""
    if dest is not None:
        if cache_in is not None:
            overlap = min(dest.shape[dim], cache_in.shape[dim])
            lead_dest, dest = _split2(dest, overlap, dim=dim)
            lead_cache, cache_in = _split2(cache_in, overlap, dim=dim)
            lead_dest.add_(lead_cache)
        body, new_input = _split2(new_input, dest.shape[dim], dim)
        dest.add_(body)
    return _cat([cache_in, new_input], dim=dim)


def mark_ended(module):
    """The next chunk is the last one: causal convolutions pad its end instead of caching it."""
    tid = threading.get_ident()
    for m in module.modules():
        if isinstance(m, CausalConv3d):
            m.temporal_cache_state[tid] = (m.temporal_cache_state.get(tid, (None, False))[0], True)


def clear_caches(module):
    tid = threading.get_ident()
    for m in module.modules():
        if hasattr(m, "temporal_cache_state"):
            m.temporal_cache_state.pop(tid, None)


MIN_VRAM_FOR_CHUNK_SCALING = 6 * 1024**3
MAX_VRAM_FOR_CHUNK_SCALING = 24 * 1024**3
MIN_CHUNK_SIZE = 32 * 1024**2
MAX_CHUNK_SIZE = 128 * 1024**2
# working memory of the streaming decoder in 16-bit, a base plus a share per latent position, fitted to session 1 on
# an A40: 2.04 GiB for 768x512x49 (2688 positions), 7.34 to 7.55 GiB for 1280x704x121 (14080)
DECODE_BASE_MEMORY = int(0.74 * 1024**3)
DECODE_MEMORY_PER_POSITION = 507 * 1024
# a tile of Forge's tiled decode costs more per position than the whole clip: 6.46 GiB measured for a 20x20 latent
# tile of 16 frames, 1.7 times the fit above
DECODE_TILE_FACTOR = 1.75
# room on top of the measurements (owner: 1.25x)
DECODE_MARGIN = 1.25


def max_chunk_size(device) -> int:
    total = memory_management.get_total_memory(device)
    if total <= MIN_VRAM_FOR_CHUNK_SCALING:
        return MIN_CHUNK_SIZE
    if total >= MAX_VRAM_FOR_CHUNK_SCALING:
        return MAX_CHUNK_SIZE
    share = (total - MIN_VRAM_FOR_CHUNK_SCALING) / (MAX_VRAM_FOR_CHUNK_SCALING - MIN_VRAM_FOR_CHUNK_SCALING)
    return int(MIN_CHUNK_SIZE + share * (MAX_CHUNK_SIZE - MIN_CHUNK_SIZE))


def patchify(x, patch_size_hw, patch_size_t=1):
    if patch_size_hw == 1 and patch_size_t == 1:
        return x
    return rearrange(x, "b c (f p) (h q) (w r) -> b (c p r q) f h w", p=patch_size_t, q=patch_size_hw, r=patch_size_hw)


def unpatchify(x, patch_size_hw, patch_size_t=1):
    if patch_size_hw == 1 and patch_size_t == 1:
        return x
    return rearrange(x, "b (c p r q) f h w -> b c (f p) (h q) (w r)", p=patch_size_t, q=patch_size_hw, r=patch_size_hw)


class PixelNorm(nn.Module):
    def __init__(self, dim=1, eps=1e-8):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x**2, dim=self.dim, keepdim=True) + self.eps)


class LayerNorm(nn.Module):
    def __init__(self, dim, eps, elementwise_affine=True):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x):
        return rearrange(self.norm(rearrange(x, "b c d h w -> b d h w c")), "b d h w c -> b c d h w")


def _norm(norm_layer, channels, groups, eps):
    if norm_layer == "group_norm":
        return nn.GroupNorm(num_groups=groups, num_channels=channels, eps=eps, affine=True)
    if norm_layer == "pixel_norm":
        return PixelNorm()
    if norm_layer == "layer_norm":
        return LayerNorm(channels, eps=eps, elementwise_affine=True)
    raise ValueError(f"[LTX] unknown norm layer {norm_layer}")


# endregion

# region convolutions


class CausalConv3d(nn.Module):
    """3D convolution, causal (or centered) in time, streaming: the last frames of a chunk are cached for the next."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1, spatial_padding_mode="zeros", **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_stride = stride if isinstance(stride, int) else stride[0]
        kernel = (kernel_size, kernel_size, kernel_size)
        self.time_kernel_size = kernel[0]
        self.spatial_pad = (kernel[1] // 2, kernel[2] // 2)
        self.spatial_padding_mode = spatial_padding_mode
        own_padding = spatial_padding_mode != "zeros"
        self.conv = nn.Conv3d(in_channels, out_channels, kernel, stride=stride, dilation=(dilation, 1, 1),
                              padding=(0, 0, 0) if own_padding else (0, *self.spatial_pad), groups=groups)
        self.temporal_cache_state = {}

    def _spatial(self, x):
        if self.spatial_padding_mode == "zeros":
            return x
        h, w = self.spatial_pad
        return torch.nn.functional.pad(x, (w, w, h, h, 0, 0), mode=self.spatial_padding_mode)

    def _empty_output(self, x):
        h = (x.shape[3] + 2 * self.spatial_pad[0] - self.conv.kernel_size[1]) // self.conv.stride[1] + 1
        w = (x.shape[4] + 2 * self.spatial_pad[1] - self.conv.kernel_size[2]) // self.conv.stride[2] + 1
        return x.new_empty((x.shape[0], self.out_channels, 0, h, w))

    def forward(self, x, causal: bool = True):
        tid = threading.get_ident()
        cached, is_end = self.temporal_cache_state.get(tid, (None, False))
        if cached is None:
            padding = self.time_kernel_size - 1
            if not causal:
                padding //= 2
            if x.shape[2] == 0:
                return self._empty_output(x)
            cached = x[:, :, :1].repeat((1, 1, padding, 1, 1))
        pieces = [cached, x]
        if is_end and not causal:
            pieces.append(x[:, :, -1:].repeat((1, 1, (self.time_kernel_size - 1) // 2, 1, 1)))
        input_length = sum(p.shape[2] for p in pieces)
        cache_length = (self.time_kernel_size - self.time_stride) + ((input_length - self.time_kernel_size) % self.time_stride)

        needs_caching = not is_end
        if needs_caching and cache_length == 0:
            self.temporal_cache_state[tid] = (x[:, :, :0], False)
            needs_caching = False
        if needs_caching and x.shape[2] >= cache_length:
            needs_caching = False
            self.temporal_cache_state[tid] = (x[:, :, -cache_length:], False)

        x = torch.cat(pieces, dim=2)
        del pieces, cached
        if needs_caching:
            self.temporal_cache_state[tid] = (x[:, :, -cache_length:], False)
        elif is_end:
            self.temporal_cache_state[tid] = (None, True)
        return self.conv(self._spatial(x)) if x.shape[2] >= self.time_kernel_size else self._empty_output(x)

    @property
    def weight(self):
        return self.conv.weight


def make_conv(dims, in_channels, out_channels, kernel_size, stride=1, padding=0, causal=False, spatial_padding_mode="zeros", bias=True):
    if dims != 3:
        raise NotImplementedError(f"[LTX] {dims}D video VAE convolutions are not supported")
    if causal:
        return CausalConv3d(in_channels, out_channels, kernel_size, stride=stride, spatial_padding_mode=spatial_padding_mode)
    return nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias, padding_mode=spatial_padding_mode)


def make_linear(dims, in_channels, out_channels, bias=True):
    return nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=bias)


# endregion

# region blocks


class ResnetBlock3D(nn.Module):
    def __init__(self, dims, in_channels, out_channels=None, dropout=0.0, groups=32, eps=1e-6, norm_layer="group_norm",
                 inject_noise=False, timestep_conditioning=False, spatial_padding_mode="zeros"):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.inject_noise = inject_noise

        self.norm1 = _norm(norm_layer, in_channels, groups, eps)
        self.non_linearity = nn.SiLU()
        self.conv1 = make_conv(dims, in_channels, out_channels, 3, padding=1, causal=True, spatial_padding_mode=spatial_padding_mode)
        if inject_noise:
            self.per_channel_scale1 = nn.Parameter(torch.zeros((in_channels, 1, 1)))
        self.norm2 = _norm(norm_layer, out_channels, groups, eps)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = make_conv(dims, out_channels, out_channels, 3, padding=1, causal=True, spatial_padding_mode=spatial_padding_mode)
        if inject_noise:
            self.per_channel_scale2 = nn.Parameter(torch.zeros((in_channels, 1, 1)))
        self.conv_shortcut = make_linear(dims, in_channels, out_channels) if in_channels != out_channels else nn.Identity()
        self.norm3 = LayerNorm(in_channels, eps=eps, elementwise_affine=True) if in_channels != out_channels else nn.Identity()

        self.timestep_conditioning = timestep_conditioning
        if timestep_conditioning:
            self.scale_shift_table = nn.Parameter(torch.randn(4, in_channels) / in_channels**0.5)
        self.temporal_cache_state = {}

    @staticmethod
    def _spatial_noise(hidden_states, per_channel_scale):
        noise = torch.randn(hidden_states.shape[-2:], device=hidden_states.device, dtype=hidden_states.dtype)[None]
        return hidden_states + (noise * per_channel_scale)[None, :, None, ...]

    def forward(self, input_tensor, causal: bool = True, timestep=None):
        batch = input_tensor.shape[0]
        hidden = self.norm1(input_tensor)
        if self.timestep_conditioning:
            ada = self.scale_shift_table[None, ..., None, None, None].to(device=hidden.device, dtype=hidden.dtype) + timestep.reshape(
                batch, 4, -1, timestep.shape[-3], timestep.shape[-2], timestep.shape[-1])
            shift1, scale1, shift2, scale2 = ada.unbind(dim=1)
            hidden = hidden * (1 + scale1) + shift1
        hidden = self.conv1(self.non_linearity(hidden), causal=causal)
        if self.inject_noise:
            hidden = self._spatial_noise(hidden, self.per_channel_scale1.to(device=hidden.device, dtype=hidden.dtype))
        hidden = self.norm2(hidden)
        if self.timestep_conditioning:
            hidden = hidden * (1 + scale2) + shift2
        hidden = self.conv2(self.dropout(self.non_linearity(hidden)), causal=causal)
        if self.inject_noise:
            hidden = self._spatial_noise(hidden, self.per_channel_scale2.to(device=hidden.device, dtype=hidden.dtype))

        shortcut = self.conv_shortcut(self.norm3(input_tensor))
        tid = threading.get_ident()
        self.temporal_cache_state[tid] = add_exchange_cache(hidden, self.temporal_cache_state.get(tid), shortcut, dim=2)
        return hidden


class UNetMidBlock3D(nn.Module):
    def __init__(self, dims, in_channels, dropout=0.0, num_layers=1, resnet_eps=1e-6, resnet_groups=32, norm_layer="group_norm",
                 inject_noise=False, timestep_conditioning=False, spatial_padding_mode="zeros"):
        super().__init__()
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        self.timestep_conditioning = timestep_conditioning
        if timestep_conditioning:
            self.time_embedder = TimestepEmbedder(in_channels * 4)
        self.res_blocks = nn.ModuleList([
            ResnetBlock3D(dims, in_channels, in_channels, eps=resnet_eps, groups=resnet_groups, dropout=dropout, norm_layer=norm_layer,
                          inject_noise=inject_noise, timestep_conditioning=timestep_conditioning, spatial_padding_mode=spatial_padding_mode)
            for _ in range(num_layers)
        ])

    def forward(self, hidden_states, causal: bool = True, timestep=None):
        embed = None
        if self.timestep_conditioning:
            batch = hidden_states.shape[0]
            embed = self.time_embedder(timestep.flatten(), hidden_states.dtype)
            embed = embed.view(batch, embed.shape[-1], 1, 1, 1)
        for resnet in self.res_blocks:
            hidden_states = resnet(hidden_states, causal=causal, timestep=embed)
        return hidden_states


class SpaceToDepthDownsample(nn.Module):
    def __init__(self, dims, in_channels, out_channels, stride, spatial_padding_mode):
        super().__init__()
        self.stride = stride
        self.group_size = in_channels * math.prod(stride) // out_channels
        self.conv = make_conv(dims, in_channels, out_channels // math.prod(stride), 3, causal=True, spatial_padding_mode=spatial_padding_mode)
        self.temporal_cache_state = {}

    def _rearrange(self, x):
        p1, p2, p3 = self.stride
        return rearrange(x, "b c (d p1) (h p2) (w p3) -> b (c p1 p2 p3) d h w", p1=p1, p2=p2, p3=p3)

    def forward(self, x, causal: bool = True):
        tid = threading.get_ident()
        cached, pad_first, cached_x, cached_input = self.temporal_cache_state.get(tid, (None, True, None, None))
        if cached_input is not None:
            x = _cat([cached_input, x], dim=2)
            cached_input = None
        if self.stride[0] == 2 and pad_first:
            x = torch.cat([x[:, :, :1], x], dim=2)  # the first frame duplicated as padding
            pad_first = False
        if x.shape[2] < self.stride[0]:
            self.temporal_cache_state[tid] = (cached, pad_first, cached_x, x)
            return None

        # skip connection: space to depth, averaged per group
        x_in = rearrange(self._rearrange(x), "b (c g) d h w -> b c g d h w", g=self.group_size).mean(dim=2)

        x = self.conv(x, causal=causal)
        if self.stride[0] == 2 and x.shape[2] == 1:
            if cached_x is not None:
                x = _cat([cached_x, x], dim=2)
                cached_x = None
            else:
                cached_x, x = x, None
        if x is not None:
            x = self._rearrange(x)

        cached = add_exchange_cache(x, cached, x_in, dim=2)
        self.temporal_cache_state[tid] = (cached, pad_first, cached_x, cached_input)
        return x


class DepthToSpaceUpsample(nn.Module):
    def __init__(self, dims, in_channels, stride, residual=False, out_channels_reduction_factor=1, spatial_padding_mode="zeros"):
        super().__init__()
        self.stride = stride
        self.out_channels = math.prod(stride) * in_channels // out_channels_reduction_factor
        self.conv = make_conv(dims, in_channels, self.out_channels, 3, causal=True, spatial_padding_mode=spatial_padding_mode)
        self.residual = residual
        self.out_channels_reduction_factor = out_channels_reduction_factor
        self.temporal_cache_state = {}

    def _rearrange(self, x):
        p1, p2, p3 = self.stride
        return rearrange(x, "b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)", p1=p1, p2=p2, p3=p3)

    def forward(self, x, causal: bool = True, timestep=None):
        tid = threading.get_ident()
        cached, drop_first_conv, drop_first_res = self.temporal_cache_state.get(tid, (None, True, True))
        y = self._rearrange(self.conv(x, causal=causal))
        if self.stride[0] == 2 and y.shape[2] > 0 and drop_first_conv:
            y = y[:, :, 1:]
            drop_first_conv = False
        if self.residual:
            x_in = self._rearrange(x).repeat(1, math.prod(self.stride) // self.out_channels_reduction_factor, 1, 1, 1)
            if self.stride[0] == 2 and x_in.shape[2] > 0 and drop_first_res:
                x_in = x_in[:, :, 1:]
                drop_first_res = False
            if y.shape[2] == 0:
                y = None
            cached = add_exchange_cache(y, cached, x_in, dim=2)
            self.temporal_cache_state[tid] = (cached, drop_first_conv, drop_first_res)
        else:
            self.temporal_cache_state[tid] = (None, drop_first_conv, False)
        return y


# endregion

# region encoder and decoder


class Encoder(nn.Module):
    def __init__(self, dims=3, in_channels=3, out_channels=3, blocks=(("res_x", 1),), base_channels=128, norm_num_groups=32,
                 patch_size=1, norm_layer="group_norm", latent_log_var="per_channel", spatial_padding_mode="zeros"):
        super().__init__()
        self.patch_size = patch_size
        self.latent_log_var = latent_log_var
        in_channels = in_channels * patch_size**2
        channel = base_channels
        self.conv_in = make_conv(dims, in_channels, channel, 3, padding=1, causal=True, spatial_padding_mode=spatial_padding_mode)

        strides = {"compress_time": (2, 1, 1), "compress_space": (1, 2, 2), "compress_all": (2, 2, 2), "compress_all_x_y": (2, 2, 2),
                   "compress_all_res": (2, 2, 2), "compress_space_res": (1, 2, 2), "compress_time_res": (2, 1, 1)}
        self.down_blocks = nn.ModuleList()
        for name, params in blocks:
            input_channel = channel
            if isinstance(params, int):
                params = {"num_layers": params}
            if name == "res_x":
                block = UNetMidBlock3D(dims, input_channel, num_layers=params["num_layers"], resnet_eps=1e-6, resnet_groups=norm_num_groups,
                                       norm_layer=norm_layer, spatial_padding_mode=spatial_padding_mode)
            elif name == "res_x_y":
                channel = params.get("multiplier", 2) * channel
                block = ResnetBlock3D(dims, input_channel, channel, eps=1e-6, groups=norm_num_groups, norm_layer=norm_layer,
                                      spatial_padding_mode=spatial_padding_mode)
            elif name in ("compress_time", "compress_space", "compress_all"):
                block = make_conv(dims, input_channel, channel, 3, stride=strides[name], causal=True, spatial_padding_mode=spatial_padding_mode)
            elif name == "compress_all_x_y":
                channel = params.get("multiplier", 2) * channel
                block = make_conv(dims, input_channel, channel, 3, stride=strides[name], causal=True, spatial_padding_mode=spatial_padding_mode)
            elif name in ("compress_all_res", "compress_space_res", "compress_time_res"):
                channel = params.get("multiplier", 2) * channel
                block = SpaceToDepthDownsample(dims, input_channel, channel, strides[name], spatial_padding_mode)
            else:
                raise ValueError(f"[LTX] unknown encoder block {name}")
            self.down_blocks.append(block)

        self.conv_norm_out = _norm(norm_layer, channel, norm_num_groups, 1e-6)
        self.conv_act = nn.SiLU()
        conv_out_channels = out_channels
        if latent_log_var == "per_channel":
            conv_out_channels *= 2
        elif latent_log_var in ("uniform", "constant"):
            conv_out_channels += 1
        elif latent_log_var != "none":
            raise ValueError(f"[LTX] invalid latent_log_var {latent_log_var}")
        self.conv_out = make_conv(dims, channel, conv_out_channels, 3, padding=1, causal=True, spatial_padding_mode=spatial_padding_mode)

    def _forward_chunk(self, sample):
        sample = self.conv_in(sample)
        for block in self.down_blocks:
            sample = block(sample)
            if sample is None or sample.shape[2] == 0:
                return None
        sample = self.conv_out(self.conv_act(self.conv_norm_out(sample)))
        if sample is None or sample.shape[2] == 0:
            return None
        if self.latent_log_var == "uniform":
            last = sample[:, -1:]
            sample = torch.cat([sample, last.repeat(1, sample.shape[1] - 2, *([1] * (sample.dim() - 2)))], dim=1)
        elif self.latent_log_var == "constant":
            sample = sample[:, :-1]
            sample = torch.cat([sample, torch.full_like(sample, -30.0)], dim=1)
        return sample

    def forward(self, sample, device=None):
        try:
            chunk_bytes = max_chunk_size(sample.device if device is None else device) * 2  # the encoder needs less than the decoder
            frame_bytes = int(sample[:, :, :1].numel() * sample.element_size() * (self.conv_in.out_channels / self.conv_in.in_channels))
            chunks = [sample[:, :, :1]]
            if sample.shape[2] > 1:
                chunk_t = max(2, chunk_bytes // frame_bytes)
                chunk_t = 2 if chunk_t < 4 else 4 if chunk_t < 8 else (chunk_t // 8) * 8
                chunks += list(torch.split(sample[:, :, 1:], chunk_t, dim=2))
            outputs = []
            for i, chunk in enumerate(chunks):
                if i == len(chunks) - 1:
                    mark_ended(self)
                out = self._forward_chunk(patchify(chunk, patch_size_hw=self.patch_size, patch_size_t=1).to(device=device))
                if out is not None:
                    outputs.append(out)
            return _cat(outputs, dim=2)
        finally:
            clear_caches(self)


class Decoder(nn.Module):
    def __init__(self, dims, in_channels=3, out_channels=3, blocks=(("res_x", 1),), base_channels=128, layers_per_block=2, norm_num_groups=32,
                 patch_size=1, norm_layer="group_norm", causal=True, timestep_conditioning=False, spatial_padding_mode="zeros"):
        super().__init__()
        self.patch_size = patch_size
        out_channels = out_channels * patch_size**2
        self.causal = causal

        channel = base_channels
        for name, params in blocks:
            params = params if isinstance(params, dict) else {}
            if name == "res_x_y":
                channel = params.get("in_channels", channel * params.get("multiplier", 2))
            if name in ("compress_all", "compress_space", "compress_time"):
                channel = channel * params.get("multiplier", 1)

        self.conv_in = make_conv(dims, in_channels, channel, 3, padding=1, causal=True, spatial_padding_mode=spatial_padding_mode)

        strides = {"compress_time": (2, 1, 1), "compress_space": (1, 2, 2), "compress_all": (2, 2, 2)}
        self.up_blocks = nn.ModuleList()
        for name, params in reversed(list(blocks)):
            input_channel = channel
            if isinstance(params, int):
                params = {"num_layers": params}
            if name == "res_x":
                block = UNetMidBlock3D(dims, input_channel, num_layers=params["num_layers"], resnet_eps=1e-6, resnet_groups=norm_num_groups,
                                       norm_layer=norm_layer, inject_noise=params.get("inject_noise", False),
                                       timestep_conditioning=timestep_conditioning, spatial_padding_mode=spatial_padding_mode)
            elif name == "res_x_y":
                channel = params.get("out_channels", channel // params.get("multiplier", 2))
                block = ResnetBlock3D(dims, input_channel, channel, eps=1e-6, groups=norm_num_groups, norm_layer=norm_layer,
                                      inject_noise=params.get("inject_noise", False), spatial_padding_mode=spatial_padding_mode)
            elif name in strides:
                channel = channel // params.get("multiplier", 1)
                block = DepthToSpaceUpsample(dims, input_channel, strides[name], residual=params.get("residual", False) if name == "compress_all" else False,
                                             out_channels_reduction_factor=params.get("multiplier", 1), spatial_padding_mode=spatial_padding_mode)
            else:
                raise ValueError(f"[LTX] unknown decoder block {name}")
            self.up_blocks.append(block)

        self.conv_norm_out = _norm(norm_layer, channel, norm_num_groups, 1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = make_conv(dims, channel, out_channels, 3, padding=1, causal=True, spatial_padding_mode=spatial_padding_mode)

        # output scale: channels, (t, h, w) factors and the frames the causal start drops
        ts, hs, ws, to = 1, 1, 1, 0
        for block in self.up_blocks:
            if isinstance(block, DepthToSpaceUpsample):
                ts, hs, ws = ts * block.stride[0], hs * block.stride[1], ws * block.stride[2]
                if block.stride[0] > 1:
                    to = to * block.stride[0] + 1
        self._output_scale = (out_channels // (patch_size**2), (ts, hs * patch_size, ws * patch_size), to)

        self.timestep_conditioning = timestep_conditioning
        if timestep_conditioning:
            self.timestep_scale_multiplier = nn.Parameter(torch.tensor(1000.0, dtype=torch.float32))
            self.last_time_embedder = TimestepEmbedder(channel * 2)
            self.last_scale_shift_table = nn.Parameter(torch.empty(2, channel))

    def decode_output_shape(self, input_shape):
        c, (ts, hs, ws), to = self._output_scale
        return (input_shape[0], c, input_shape[2] * ts - to, input_shape[3] * hs, input_shape[4] * ws)

    def _run_up(self, idx, sample_ref, ended, shift_scale, scaled_timestep, output, offset, chunk_bytes):
        sample = sample_ref[0]
        sample_ref[0] = None
        if idx >= len(self.up_blocks):
            sample = self.conv_norm_out(sample)
            if shift_scale is not None:
                shift, scale = shift_scale
                sample = sample * (1 + scale) + shift
            sample = self.conv_act(sample)
            if ended:
                mark_ended(self.conv_out)
            sample = self.conv_out(sample, causal=self.causal)
            if sample is not None and sample.shape[2] > 0:
                sample = unpatchify(sample, patch_size_hw=self.patch_size, patch_size_t=1)
                frames = sample.shape[2]
                output[:, :, offset[0]:offset[0] + frames].copy_(sample)
                offset[0] += frames
            return

        block = self.up_blocks[idx]
        if ended:
            mark_ended(block)
        if self.timestep_conditioning and isinstance(block, UNetMidBlock3D):
            sample = block(sample, causal=self.causal, timestep=scaled_timestep)
        else:
            sample = block(sample, causal=self.causal)
        if sample is None or sample.shape[2] == 0:
            return

        chunks = (sample.numel() * sample.element_size() + chunk_bytes - 1) // chunk_bytes
        if chunks == 1:
            next_ref = [sample]
            del sample  # the callee can free it as soon as it is done
            self._run_up(idx + 1, next_ref, ended, shift_scale, scaled_timestep, output, offset, chunk_bytes)
            return
        parts = torch.chunk(sample, chunks=chunks, dim=2)
        for i, part in enumerate(parts):
            self._run_up(idx + 1, [part], ended and i == len(parts) - 1, shift_scale, scaled_timestep, output, offset, chunk_bytes)

    def forward(self, sample, timestep=None, output_buffer=None):
        try:
            batch = sample.shape[0]
            mark_ended(self.conv_in)
            sample = self.conv_in(sample, causal=self.causal)

            shift_scale = scaled_timestep = None
            if self.timestep_conditioning:
                scaled_timestep = timestep * self.timestep_scale_multiplier.to(dtype=sample.dtype, device=sample.device)
                embedded = self.last_time_embedder(scaled_timestep.flatten(), sample.dtype).view(batch, -1, 1, 1, 1)
                ada = self.last_scale_shift_table[None, ..., None, None, None].to(device=sample.device, dtype=sample.dtype) + embedded.reshape(
                    batch, 2, -1, embedded.shape[-3], embedded.shape[-2], embedded.shape[-1])
                shift_scale = ada.unbind(dim=1)

            if output_buffer is None:
                output_buffer = torch.empty(self.decode_output_shape(sample.shape), dtype=sample.dtype, device=memory_management.intermediate_device())
            self._run_up(0, [sample], True, shift_scale, scaled_timestep, output_buffer, [0], max_chunk_size(sample.device))
            return output_buffer
        finally:
            clear_caches(self)


# endregion


#: statistics LTX-2.0's VAE also carries; ComfyUI loads without them and so does the extension
UNUSED_KEYS = ("per_channel_statistics.channel", "per_channel_statistics.mean-of-stds", "per_channel_statistics.mean-of-stds_over_std-of-means")


class LatentStatistics(nn.Module):
    """Per-channel latent mean and std (ComfyUI "processor"); the DiT works on normalized latents."""

    def __init__(self):
        super().__init__()
        self.register_buffer("std-of-means", torch.empty(128))
        self.register_buffer("mean-of-means", torch.empty(128))

    def un_normalize(self, x):
        return x * self.get_buffer("std-of-means").view(1, -1, 1, 1, 1).to(x) + self.get_buffer("mean-of-means").view(1, -1, 1, 1, 1).to(x)

    def normalize(self, x):
        return (x - self.get_buffer("mean-of-means").view(1, -1, 1, 1, 1).to(x)) / self.get_buffer("std-of-means").view(1, -1, 1, 1, 1).to(x)


class VideoVAE(nn.Module):
    """Pixels [B, 3, F, H, W] in [-1, 1] <-> normalized latents [B, 128, 1 + (F - 1) / 8, H / 32, W / 32]."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.timestep_conditioning = config.get("timestep_conditioning", False)
        self.decode_noise_scale = config.get("decode_noise_scale", 0.025)
        self.decode_timestep = config.get("decode_timestep", 0.05)
        latent_log_var = config.get("latent_log_var", "per_channel" if config.get("double_z", True) else "none")
        blocks = config.get("blocks")

        self.encoder = Encoder(dims=config["dims"], in_channels=config.get("in_channels", 3), out_channels=config["latent_channels"],
                               blocks=config.get("encoder_blocks", blocks), patch_size=config.get("patch_size", 1), latent_log_var=latent_log_var,
                               norm_layer=config.get("norm_layer", "group_norm"), spatial_padding_mode=config.get("spatial_padding_mode", "zeros"),
                               base_channels=config.get("encoder_base_channels", 128))
        # ComfyUI's decoder default is reflect padding: LTX-2.0's VAE config does not name one
        self.decoder = Decoder(dims=config["dims"], in_channels=config["latent_channels"], out_channels=config.get("out_channels", 3),
                               blocks=config.get("decoder_blocks", blocks), base_channels=config.get("decoder_base_channels", 128),
                               patch_size=config.get("patch_size", 1), norm_layer=config.get("norm_layer", "group_norm"),
                               causal=config.get("causal_decoder", False), timestep_conditioning=self.timestep_conditioning,
                               spatial_padding_mode=config.get("spatial_padding_mode", "reflect"))
        self.per_channel_statistics = LatentStatistics()

    def encode(self, x, device=None):
        x = x[:, :, :max(1, 1 + ((x.shape[2] - 1) // 8) * 8)]
        means, _ = torch.chunk(self.encoder(x, device=device), 2, dim=1)
        return self.per_channel_statistics.normalize(means)

    def decode_output_shape(self, input_shape):
        return self.decoder.decode_output_shape(input_shape)

    def decode_memory(self, latent_shape, dtype, device, tiled: bool = False) -> int:
        """Working memory the streaming decoder needs on device for a latent (or one tile of it) of this shape, the
        weights aside.

        ComfyUI reserves 1200 * F * H * W * 512 bytes per element, enough to decode the whole clip at once (17 GiB for
        five seconds of 720p); the decoder here keeps one temporal chunk per stage instead."""
        latent_positions = math.prod(latent_shape[2:]) * latent_shape[0]
        scale = memory_management.dtype_size(dtype) / 2 * DECODE_MARGIN * (DECODE_TILE_FACTOR if tiled else 1.0)
        return int((DECODE_BASE_MEMORY + latent_positions * DECODE_MEMORY_PER_POSITION) * scale)

    def decode(self, x, output_buffer=None, generator=None):
        if self.timestep_conditioning:
            noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype) if generator is not None else torch.randn_like(x)
            x = noise * self.decode_noise_scale + (1.0 - self.decode_noise_scale) * x
        return self.decoder(self.per_channel_statistics.un_normalize(x), timestep=self.decode_timestep, output_buffer=output_buffer)
