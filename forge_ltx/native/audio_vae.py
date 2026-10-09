"""The LTX-2.x audio VAE, ported from ComfyUI comfy/ldm/lightricks/vae (causal_audio_autoencoder.py, audio_vae.py).

A causal 2D autoencoder over the stereo log-mel spectrogram (time x 64 mel bins, causal in time) and a vocoder that
turns the decoded spectrogram into a waveform (vocoder.py). Latents are [B, 8, T, 16], 25 per second at 16 kHz,
normalized per channel in the DiT's token layout. Conv2d and GroupNorm are plain torch.nn names that Forge Neo's
loader swaps for its own operations.

Encoding a waveform (for audio inputs) needs its log-mel spectrogram; that front end comes with the features that use it.
"""

import json

import torch
import torch.nn.functional as F
from torch import nn

from .transformer import AudioPatchifier
from .video_vae import PixelNorm
from .vocoder import Vocoder, VocoderWithBWE

LATENT_DOWNSAMPLE_FACTOR = 4

# causality axis -> the (left, right, top, bottom) padding of a k x k causal convolution, and of the stride-2 downsample
_NONE, _WIDTH, _HEIGHT, _WIDTH_COMPAT = None, "width", "height", "width-compatibility"


def _axis(value):
    value = None if value is None or str(value).lower() == "none" else str(value).lower()
    if value not in (_NONE, _WIDTH, _HEIGHT, _WIDTH_COMPAT):
        raise ValueError(f"[LTX] invalid causality axis {value}")
    return value


def _normalize(channels, norm_type):
    if norm_type == "group":
        return nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)
    if norm_type == "pixel":
        return PixelNorm(dim=1, eps=1e-6)
    raise ValueError(f"[LTX] invalid normalization {norm_type}")


class CausalConv2d(nn.Module):
    """2D convolution padded on one side of the causal axis, so outputs never see later time steps."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True, causality_axis=_HEIGHT):
        super().__init__()
        kernel = nn.modules.utils._pair(kernel_size)
        dilation = nn.modules.utils._pair(dilation)
        pad_h, pad_w = (kernel[0] - 1) * dilation[0], (kernel[1] - 1) * dilation[1]
        if causality_axis is _NONE:
            self.padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
        elif causality_axis in (_WIDTH, _WIDTH_COMPAT):
            self.padding = (pad_w, 0, pad_h // 2, pad_h - pad_h // 2)
        else:
            self.padding = (pad_w // 2, pad_w - pad_w // 2, pad_h, 0)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride=stride, padding=0, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        return self.conv(F.pad(x, self.padding))


def _conv(in_channels, out_channels, kernel_size, causality_axis):
    return CausalConv2d(in_channels, out_channels, kernel_size, causality_axis=causality_axis)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv, causality_axis=_HEIGHT):
        super().__init__()
        self.with_conv = with_conv
        self.causality_axis = causality_axis
        if with_conv:
            self.conv = _conv(in_channels, in_channels, 3, causality_axis)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
            # drop the first step on the causal axis, undoing the encoder's padding (length stays 1 + 2n)
            if self.causality_axis == _HEIGHT:
                x = x[:, :, 1:, :]
            elif self.causality_axis == _WIDTH:
                x = x[:, :, :, 1:]
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv, causality_axis=_WIDTH):
        super().__init__()
        self.with_conv = with_conv
        self.causality_axis = causality_axis
        if causality_axis is not _NONE and not with_conv:
            raise ValueError("[LTX] causal downsampling needs a convolution")
        if with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if not self.with_conv:
            return F.avg_pool2d(x, kernel_size=2, stride=2)
        pad = {_NONE: (0, 1, 0, 1), _WIDTH: (2, 0, 0, 1), _HEIGHT: (0, 1, 2, 0), _WIDTH_COMPAT: (1, 0, 0, 1)}[self.causality_axis]
        return self.conv(F.pad(x, pad, mode="constant", value=0))


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False, dropout, norm_type="group", causality_axis=_HEIGHT):
        super().__init__()
        if causality_axis is not _NONE and norm_type == "group":
            raise ValueError("[LTX] a causal ResnetBlock cannot use GroupNorm")
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.norm1 = _normalize(in_channels, norm_type)
        self.non_linearity = nn.SiLU()
        self.conv1 = _conv(in_channels, out_channels, 3, causality_axis)
        self.norm2 = _normalize(out_channels, norm_type)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = _conv(out_channels, out_channels, 3, causality_axis)
        if in_channels != out_channels:
            if conv_shortcut:
                self.conv_shortcut = _conv(in_channels, out_channels, 3, causality_axis)
            else:
                self.nin_shortcut = _conv(in_channels, out_channels, 1, causality_axis)

    def forward(self, x):
        h = self.conv1(self.non_linearity(self.norm1(x)))
        h = self.conv2(self.dropout(self.non_linearity(self.norm2(h))))
        if self.in_channels != self.out_channels:
            x = self.conv_shortcut(x) if self.use_conv_shortcut else self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels, norm_type="group"):
        super().__init__()
        self.norm = _normalize(in_channels, norm_type)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        h = self.norm(x)
        q, k, v = self.q(h), self.k(h), self.v(h)
        b, c, hh, ww = q.shape
        weights = torch.softmax(torch.bmm(q.reshape(b, c, hh * ww).permute(0, 2, 1), k.reshape(b, c, hh * ww)) * (int(c) ** -0.5), dim=2)
        h = torch.bmm(v.reshape(b, c, hh * ww), weights.permute(0, 2, 1)).reshape(b, c, hh, ww)
        return x + self.proj_out(h)


def _attention(in_channels, attn_type, norm_type):
    if attn_type == "vanilla":
        return AttnBlock(in_channels, norm_type=norm_type)
    if attn_type == "none":
        return nn.Identity()
    raise NotImplementedError(f"[LTX] attention type {attn_type} is not supported")


class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks, attn_resolutions, dropout=0.0, resamp_with_conv=True,
                 in_channels, resolution, z_channels, double_z=True, attn_type="vanilla", mid_block_add_attention=True, norm_type="group",
                 causality_axis=_WIDTH, **_):
        super().__init__()
        axis = _axis(causality_axis)
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        self.conv_in = _conv(in_channels, ch, 3, axis)
        self.non_linearity = nn.SiLU()

        res = resolution
        in_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for level in range(self.num_resolutions):
            blocks, attns = nn.ModuleList(), nn.ModuleList()
            block_in, block_out = ch * in_mult[level], ch * ch_mult[level]
            for _ in range(num_res_blocks):
                blocks.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout, norm_type=norm_type, causality_axis=axis))
                block_in = block_out
                if res in attn_resolutions:
                    attns.append(_attention(block_in, attn_type, norm_type))
            down = nn.Module()
            down.block, down.attn = blocks, attns
            if level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv, causality_axis=axis)
                res //= 2
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout, norm_type=norm_type, causality_axis=axis)
        self.mid.attn_1 = _attention(block_in, attn_type, norm_type) if mid_block_add_attention else nn.Identity()
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout, norm_type=norm_type, causality_axis=axis)
        self.norm_out = _normalize(block_in, norm_type)
        self.conv_out = _conv(block_in, 2 * z_channels if double_z else z_channels, 3, axis)

    def forward(self, x):
        h = self.conv_in(x)
        for level in range(self.num_resolutions):
            for i in range(self.num_res_blocks):
                h = self.down[level].block[i](h)
                if len(self.down[level].attn) > 0:
                    h = self.down[level].attn[i](h)
            if level != self.num_resolutions - 1:
                h = self.down[level].downsample(h)
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(h)))
        return self.conv_out(self.non_linearity(self.norm_out(h)))


class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks, attn_resolutions, dropout=0.0, resamp_with_conv=True,
                 in_channels, resolution, z_channels, tanh_out=False, attn_type="vanilla", mid_block_add_attention=True, norm_type="group",
                 causality_axis=_WIDTH, **_):
        super().__init__()
        axis = _axis(causality_axis)
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.out_ch = out_ch
        self.z_channels = z_channels
        self.tanh_out = tanh_out

        block_in = ch * ch_mult[-1]
        res = resolution // 2 ** (self.num_resolutions - 1)
        self.conv_in = _conv(z_channels, block_in, 3, axis)
        self.non_linearity = nn.SiLU()
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout, norm_type=norm_type, causality_axis=axis)
        self.mid.attn_1 = _attention(block_in, attn_type, norm_type) if mid_block_add_attention else nn.Identity()
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout, norm_type=norm_type, causality_axis=axis)

        self.up = nn.ModuleList()
        for level in reversed(range(self.num_resolutions)):
            blocks, attns = nn.ModuleList(), nn.ModuleList()
            block_out = ch * ch_mult[level]
            for _ in range(num_res_blocks + 1):
                blocks.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout, norm_type=norm_type, causality_axis=axis))
                block_in = block_out
                if res in attn_resolutions:
                    attns.append(_attention(block_in, attn_type, norm_type))
            up = nn.Module()
            up.block, up.attn = blocks, attns
            if level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv, causality_axis=axis)
                res *= 2
            self.up.insert(0, up)

        self.norm_out = _normalize(block_in, norm_type)
        self.conv_out = _conv(block_in, out_ch, 3, axis)

    @staticmethod
    def _fit(decoded, target_shape):
        # crop, then pad with zeros, to the exact target length and bins
        _, channels, time, freq = target_shape
        decoded = decoded[:, :channels, :min(decoded.shape[2], time), :min(decoded.shape[3], freq)]
        pad_t, pad_f = time - decoded.shape[2], freq - decoded.shape[3]
        if pad_t > 0 or pad_f > 0:
            decoded = F.pad(decoded, (0, max(pad_f, 0), 0, max(pad_t, 0)))
        return decoded[:, :channels, :time, :freq]

    def forward(self, z, target_shape):
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(self.conv_in(z))))
        for level in reversed(range(self.num_resolutions)):
            for i in range(self.num_res_blocks + 1):
                h = self.up[level].block[i](h)
                if len(self.up[level].attn) > 0:
                    h = self.up[level].attn[i](h)
            if level != 0:
                h = self.up[level].upsample(h)
        h = self.conv_out(self.non_linearity(self.norm_out(h)))
        if self.tanh_out:
            h = torch.tanh(h)
        return self._fit(h, target_shape)


class LatentStatistics(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("std-of-means", torch.empty(128))
        self.register_buffer("mean-of-means", torch.empty(128))

    def un_normalize(self, x):
        return x * self.get_buffer("std-of-means").to(x) + self.get_buffer("mean-of-means").to(x)

    def normalize(self, x):
        return (x - self.get_buffer("mean-of-means").to(x)) / self.get_buffer("std-of-means").to(x)


class CausalAudioAutoencoder(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        params = config.get("model", {}).get("params", {})
        self.sampling_rate = params.get("sampling_rate", config.get("sampling_rate", 16000))
        encoder_config = params.get("encoder", params.get("ddconfig", {}))
        decoder_config = params.get("decoder", encoder_config)
        stft = config.get("preprocessing", {}).get("stft", {})
        self.mel_bins = encoder_config.get("mel_bins", 64)
        self.mel_hop_length = stft.get("hop_length", 160)
        self.n_fft = stft.get("filter_length", 1024)
        self.causality_axis = _axis(encoder_config.get("causality_axis", _HEIGHT))
        self.is_causal = self.causality_axis == _HEIGHT
        self.encoder = Encoder(**encoder_config)
        self.decoder = Decoder(**decoder_config)
        self.per_channel_statistics = LatentStatistics()


def components(metadata: dict) -> tuple[dict, dict]:
    """(audio VAE config, vocoder config) from a file's "config" metadata."""
    raw = (metadata or {}).get("config")
    if not raw:
        raise ValueError("[LTX] the audio VAE file has no config metadata")
    config = json.loads(raw) if isinstance(raw, str) else raw
    if config.get("audio_vae") is None or config.get("vocoder") is None:
        raise ValueError("[LTX] the audio VAE config misses the autoencoder or the vocoder")
    return config["audio_vae"], config["vocoder"]


class AudioVAE(nn.Module):
    """Normalized audio latents [B, 8, T, 16] -> stereo waveform at output_sample_rate. State dict keys follow the
    published files once their "audio_vae." prefix is renamed "autoencoder." (as ComfyUI loads them)."""

    def __init__(self, autoencoder_config: dict, vocoder_config: dict):
        super().__init__()
        self.autoencoder = CausalAudioAutoencoder(autoencoder_config)
        self.vocoder = VocoderWithBWE(vocoder_config) if "bwe" in vocoder_config else Vocoder(vocoder_config)
        self.patchifier = AudioPatchifier(sample_rate=self.autoencoder.sampling_rate, hop_length=self.autoencoder.mel_hop_length,
                                          audio_latent_downsample_factor=LATENT_DOWNSAMPLE_FACTOR, is_causal=self.autoencoder.is_causal,
                                          start_end=False)

    def _statistics(self, latents, fn):
        channels, freq = latents.shape[1], latents.shape[3]
        tokens, _ = self.patchifier.patchify(latents)
        return self.patchifier.unpatchify(fn(tokens), channels=channels, freq=freq)

    def normalize(self, latents):
        return self._statistics(latents, self.autoencoder.per_channel_statistics.normalize)

    def denormalize(self, latents):
        return self._statistics(latents, self.autoencoder.per_channel_statistics.un_normalize)

    def target_shape(self, latents_shape):
        batch, _, time, _ = latents_shape
        length = time * LATENT_DOWNSAMPLE_FACTOR
        if self.autoencoder.causality_axis is not _NONE:
            length -= LATENT_DOWNSAMPLE_FACTOR - 1
        return batch, self.autoencoder.decoder.out_ch, length, self.autoencoder.mel_bins

    def encode_mel(self, mel):
        """A log-mel spectrogram [B, 2, T, 64] to normalized latents (the posterior mode)."""
        mean, _ = torch.chunk(self.autoencoder.encoder(mel), 2, dim=1)
        return self.normalize(mean)

    def decode(self, latents):
        mel = self.autoencoder.decoder(self.denormalize(latents), target_shape=self.target_shape(latents.shape))
        channels = self.autoencoder.decoder.out_ch
        mel = mel.transpose(2, 3)
        if channels == 1:
            mel = mel.squeeze(1)
        elif channels != 2:
            raise ValueError(f"[LTX] unsupported audio channel count {channels}")
        return self.vocoder(mel)

    @property
    def sample_rate(self) -> int:
        return int(self.autoencoder.sampling_rate)

    @property
    def latents_per_second(self) -> float:
        return self.sample_rate / self.autoencoder.mel_hop_length / LATENT_DOWNSAMPLE_FACTOR

    @property
    def output_sample_rate(self) -> int:
        rate = getattr(self.vocoder, "output_sample_rate", None)
        if rate is not None:
            return int(rate)
        return int(self.sample_rate * self.vocoder.upsample_factor / self.autoencoder.mel_hop_length)
