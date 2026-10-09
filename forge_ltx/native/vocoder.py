"""LTX-2.x vocoders, ported from ComfyUI comfy/ldm/lightricks/vocoders/vocoder.py.

LTX-2.0: HiFi-GAN (resblock "1") from the 64-bin stereo mel at 16 kHz to 24 kHz audio. LTX-2.3 and 2.5: BigVGAN v2
(anti-aliased AMP blocks with SnakeBeta) to 16 kHz, then a bandwidth extension stage to 48 kHz: a causal mel of the
16 kHz audio drives a second BigVGAN whose output is added to the resampled audio.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

LRELU_SLOPE = 0.1


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


# region anti-aliased resampling (kaiser-sinc filters, from NVIDIA BigVGAN)


def _sinc(x):
    return torch.where(x == 0, torch.tensor(1.0, device=x.device, dtype=x.dtype), torch.sin(math.pi * x) / math.pi / x)


def kaiser_sinc_filter1d(cutoff, half_width, kernel_size):
    even = kernel_size % 2 == 0
    half = kernel_size // 2
    a = 2.285 * (half - 1) * math.pi * 4 * half_width + 7.95
    beta = 0.1102 * (a - 8.7) if a > 50.0 else 0.5842 * (a - 21) ** 0.4 + 0.07886 * (a - 21.0) if a >= 21.0 else 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)
    time = torch.arange(-half, half) + 0.5 if even else torch.arange(kernel_size) - half
    if cutoff == 0:
        return torch.zeros_like(time).view(1, 1, kernel_size)
    filter_ = 2 * cutoff * window * _sinc(2 * cutoff * time)
    return (filter_ / filter_.sum()).view(1, 1, kernel_size)


class LowPassFilter1d(nn.Module):
    def __init__(self, cutoff=0.5, half_width=0.6, stride=1, padding=True, padding_mode="replicate", kernel_size=12):
        super().__init__()
        if not 0.0 <= cutoff <= 0.5:
            raise ValueError("[LTX] the low-pass cutoff must be within [0, 0.5]")
        even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        self.register_buffer("filter", kaiser_sinc_filter1d(cutoff, half_width, kernel_size))

    def forward(self, x):
        channels = x.shape[1]
        if self.padding:
            x = F.pad(x, (self.pad_left, self.pad_right), mode=self.padding_mode)
        return F.conv1d(x, self.filter.expand(channels, -1, -1).to(dtype=x.dtype, device=x.device), stride=self.stride, groups=channels)


class UpSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None, persistent=True, window_type="kaiser"):
        super().__init__()
        self.ratio = ratio
        self.stride = ratio
        if window_type == "hann":
            # torchaudio-style Hann-windowed sinc, with replicate padding at the edges
            rolloff, lowpass_filter_width = 0.99, 6
            width = math.ceil(lowpass_filter_width / rolloff)
            self.kernel_size = 2 * width * ratio + 1
            self.pad = width
            self.pad_left = 2 * width * ratio
            self.pad_right = self.kernel_size - ratio
            t = (torch.arange(self.kernel_size) / ratio - width) * rolloff
            window = torch.cos(t.clamp(-lowpass_filter_width, lowpass_filter_width) * math.pi / lowpass_filter_width / 2) ** 2
            filter_ = (torch.sinc(t) * window * rolloff / ratio).view(1, 1, -1)
        else:
            self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
            self.pad = self.kernel_size // ratio - 1
            self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
            self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
            filter_ = kaiser_sinc_filter1d(cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=self.kernel_size)
        self.register_buffer("filter", filter_, persistent=persistent)

    def forward(self, x):
        channels = x.shape[1]
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        x = self.ratio * F.conv_transpose1d(x, self.filter.expand(channels, -1, -1).to(dtype=x.dtype, device=x.device), stride=self.stride, groups=channels)
        return x[..., self.pad_left:-self.pad_right]


class DownSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.lowpass = LowPassFilter1d(cutoff=0.5 / ratio, half_width=0.6 / ratio, stride=ratio, kernel_size=kernel_size)

    def forward(self, x):
        return self.lowpass(x)


class Activation1d(nn.Module):
    def __init__(self, activation, up_ratio=2, down_ratio=2, up_kernel_size=12, down_kernel_size=12):
        super().__init__()
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    def forward(self, x):
        return self.downsample(self.act(self.upsample(x)))


# endregion

# region activations and residual blocks


class Snake(nn.Module):
    def __init__(self, in_features, alpha=1.0, alpha_logscale=True):
        super().__init__()
        self.alpha_logscale = alpha_logscale
        self.alpha = nn.Parameter(torch.zeros(in_features) if alpha_logscale else torch.ones(in_features) * alpha)
        self.eps = 1e-9

    def forward(self, x):
        a = self.alpha.unsqueeze(0).unsqueeze(-1).to(dtype=x.dtype, device=x.device)
        if self.alpha_logscale:
            a = torch.exp(a)
        return x + (1.0 / (a + self.eps)) * torch.sin(x * a).pow(2)


class SnakeBeta(nn.Module):
    def __init__(self, in_features, alpha=1.0, alpha_logscale=True):
        super().__init__()
        self.alpha_logscale = alpha_logscale
        self.alpha = nn.Parameter(torch.zeros(in_features) if alpha_logscale else torch.ones(in_features) * alpha)
        self.beta = nn.Parameter(torch.zeros(in_features) if alpha_logscale else torch.ones(in_features) * alpha)
        self.eps = 1e-9

    def forward(self, x):
        a = self.alpha.unsqueeze(0).unsqueeze(-1).to(dtype=x.dtype, device=x.device)
        b = self.beta.unsqueeze(0).unsqueeze(-1).to(dtype=x.dtype, device=x.device)
        if self.alpha_logscale:
            a, b = torch.exp(a), torch.exp(b)
        return x + (1.0 / (b + self.eps)) * torch.sin(x * a).pow(2)


def _convs(channels, kernel_size, dilations):
    return nn.ModuleList([nn.Conv1d(channels, channels, kernel_size, 1, dilation=d, padding=get_padding(kernel_size, d)) for d in dilations])


class AMPBlock1(nn.Module):
    """BigVGAN v2 anti-aliased multi-periodicity block."""

    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), activation="snake"):
        super().__init__()
        act = SnakeBeta if activation == "snakebeta" else Snake
        self.convs1 = _convs(channels, kernel_size, dilation[:3])
        self.convs2 = _convs(channels, kernel_size, (1, 1, 1))
        self.acts1 = nn.ModuleList([Activation1d(act(channels)) for _ in range(3)])
        self.acts2 = nn.ModuleList([Activation1d(act(channels)) for _ in range(3)])

    def forward(self, x):
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, self.acts1, self.acts2):
            x = x + c2(a2(c1(a1(x))))
        return x


class ResBlock1(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = _convs(channels, kernel_size, dilation[:3])
        self.convs2 = _convs(channels, kernel_size, (1, 1, 1))

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            x = c2(F.leaky_relu(c1(F.leaky_relu(x, LRELU_SLOPE)), LRELU_SLOPE)) + x
        return x


class ResBlock2(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super().__init__()
        self.convs = _convs(channels, kernel_size, dilation[:2])

    def forward(self, x):
        for c in self.convs:
            x = c(F.leaky_relu(x, LRELU_SLOPE)) + x
        return x


# endregion


class Vocoder(nn.Module):
    """HiFi-GAN (resblock "1" / "2") or BigVGAN v2 (resblock "AMP1"): mel [B, 2, mels, T] (or [B, mels, T]) -> audio."""

    def __init__(self, config: dict):
        super().__init__()
        kernel_sizes = config.get("resblock_kernel_sizes", [3, 7, 11])
        rates = config.get("upsample_rates", [5, 4, 2, 2, 2])
        up_kernels = config.get("upsample_kernel_sizes", [16, 16, 8, 4, 4])
        dilations = config.get("resblock_dilation_sizes", [[1, 3, 5], [1, 3, 5], [1, 3, 5]])
        initial = config.get("upsample_initial_channel", 1024)
        stereo = config.get("stereo", True)
        activation = config.get("activation", "snake")

        self.output_sample_rate = config.get("output_sample_rate")
        self.resblock = config.get("resblock", "1")
        self.use_tanh_at_final = config.get("use_tanh_at_final", True)
        self.apply_final_activation = config.get("apply_final_activation", True)
        self.num_kernels = len(kernel_sizes)
        self.num_upsamples = len(rates)

        self.conv_pre = nn.Conv1d(128 if stereo else 64, initial, 7, 1, padding=3)
        block = {"1": ResBlock1, "2": ResBlock2, "AMP1": AMPBlock1}.get(self.resblock)
        if block is None:
            raise ValueError(f"[LTX] unknown vocoder resblock {self.resblock}")
        self.ups = nn.ModuleList([nn.ConvTranspose1d(initial // 2**i, initial // 2 ** (i + 1), k, u, padding=(k - u) // 2)
                                  for i, (u, k) in enumerate(zip(rates, up_kernels))])
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = initial // 2 ** (i + 1)
            for k, d in zip(kernel_sizes, dilations):
                self.resblocks.append(block(ch, k, d, activation=activation) if self.resblock == "AMP1" else block(ch, k, d))
        if self.resblock == "AMP1":
            self.act_post = Activation1d((SnakeBeta if activation == "snakebeta" else Snake)(ch))
        else:
            self.act_post = nn.LeakyReLU()
        self.conv_post = nn.Conv1d(ch, 2 if stereo else 1, 7, 1, padding=3, bias=config.get("use_bias_at_final", True))
        self.upsample_factor = int(np.prod([up.stride[0] for up in self.ups]))

    def forward(self, x):
        if x.dim() == 4:  # stereo: the two channels' mels side by side
            x = torch.cat((x[:, 0], x[:, 1]), dim=1)
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            if self.resblock != "AMP1":
                x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                out = self.resblocks[i * self.num_kernels + j](x)
                xs = out if xs is None else xs + out
            x = xs / self.num_kernels
        x = self.conv_post(self.act_post(x))
        if self.apply_final_activation:
            x = torch.tanh(x) if self.use_tanh_at_final else torch.clamp(x, -1, 1)
        return x


class _STFTFn(nn.Module):
    """Causal STFT as a convolution with the DFT x Hann bases stored in the checkpoint (bit-exact with training)."""

    def __init__(self, filter_length, hop_length, win_length):
        super().__init__()
        self.hop_length = hop_length
        self.win_length = win_length
        n_freqs = filter_length // 2 + 1
        self.register_buffer("forward_basis", torch.zeros(n_freqs * 2, 1, filter_length))
        self.register_buffer("inverse_basis", torch.zeros(n_freqs * 2, 1, filter_length))

    def forward(self, y):
        if y.dim() == 2:
            y = y.unsqueeze(1)
        y = F.pad(y, (max(0, self.win_length - self.hop_length), 0))  # left only: no lookahead
        spec = F.conv1d(y, self.forward_basis.to(dtype=y.dtype, device=y.device), stride=self.hop_length, padding=0)
        n_freqs = spec.shape[1] // 2
        real, imag = spec[:, :n_freqs], spec[:, n_freqs:]
        return torch.sqrt(real**2 + imag**2), torch.atan2(imag.float(), real.float()).to(real.dtype)


class MelSTFT(nn.Module):
    def __init__(self, filter_length, hop_length, win_length, n_mel_channels, **_):
        super().__init__()
        self.stft_fn = _STFTFn(filter_length, hop_length, win_length)
        self.register_buffer("mel_basis", torch.zeros(n_mel_channels, filter_length // 2 + 1))

    def log_mel(self, y):
        magnitude, _ = self.stft_fn(y)
        mel = torch.matmul(self.mel_basis.to(dtype=magnitude.dtype, device=y.device), magnitude)
        return torch.log(torch.clamp(mel, min=1e-5))


class VocoderWithBWE(nn.Module):
    """Base vocoder, then bandwidth extension to the output rate (16 kHz -> 48 kHz for LTX-2.3 and 2.5)."""

    def __init__(self, config: dict):
        super().__init__()
        bwe = config["bwe"]
        self.vocoder = Vocoder(config["vocoder"])
        self.bwe_generator = Vocoder({**bwe, "apply_final_activation": False})
        self.input_sample_rate = bwe["input_sampling_rate"]
        self.output_sample_rate = bwe["output_sampling_rate"]
        self.hop_length = bwe["hop_length"]
        self.mel_stft = MelSTFT(bwe["n_fft"], bwe["hop_length"], bwe["n_fft"], bwe["num_mels"])
        self.resampler = UpSample1d(ratio=self.output_sample_rate // self.input_sample_rate, persistent=False, window_type="hann")

    def forward(self, mel_spec):
        x = self.vocoder(mel_spec)
        length = x.shape[2]
        out_length = length * self.output_sample_rate // self.input_sample_rate
        if length % self.hop_length:
            x = F.pad(x, (0, self.hop_length - length % self.hop_length))
        batch, channels, _ = x.shape
        mel = self.mel_stft.log_mel(x.reshape(batch * channels, -1))
        residual = self.bwe_generator(mel.reshape(batch, channels, mel.shape[1], mel.shape[2]))
        skip = self.resampler(x)
        if residual.shape != skip.shape:
            raise RuntimeError(f"[LTX] bandwidth extension shapes differ: {tuple(residual.shape)} != {tuple(skip.shape)}")
        return torch.clamp(residual + skip, -1, 1)[..., :out_length]
