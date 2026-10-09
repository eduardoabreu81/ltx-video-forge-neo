"""LTX-2.x DiT layers, ported from ComfyUI comfy/ldm/lightricks (model.py, av_model.py, embeddings_connector.py).

Inference only: the training branches, ComfyUI's block-replace patches and weight prefetching are left out. Linear,
RMSNorm and LayerNorm are plain torch.nn names, which Forge Neo's loader swaps for its own operations while it builds
the model; the fused kernels (RMS + AdaLN, RoPE) come from comfy-kitchen through Forge Neo.
"""

import functools
import math

import numpy as np
import torch
import torch.nn as nn
from backend.attention import attention_function
from backend.quant_ops import ck

# 6 base AdaLN parameters (shift/scale/gate for self-attention and MLP), +3 for the text cross-attention query
ADALN_BASE_PARAMS_COUNT = 6
ADALN_CROSS_ATTN_PARAMS_COUNT = 9


def rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.nn.functional.rms_norm(x, (x.shape[-1],), eps=eps)


# region timestep and text embeddings


def timestep_embedding(timesteps: torch.Tensor, dim: int = 256, max_period: int = 10000) -> torch.Tensor:
    # sinusoidal, cos first (flip_sin_to_cos=True, downscale_freq_shift=0)
    half = dim // 2
    exponent = -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half
    emb = timesteps[:, None].float() * torch.exp(exponent)[None, :]
    emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)
    if dim % 2:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=True)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, bias=True)

    def forward(self, sample):
        return self.linear_2(torch.nn.functional.silu(self.linear_1(sample)))


class TimestepEmbedder(nn.Module):
    # ComfyUI PixArtAlphaCombinedTimestepSizeEmbeddings without the unused size conditions
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.timestep_embedder = TimestepEmbedding(256, embedding_dim)

    def forward(self, timestep, hidden_dtype):
        return self.timestep_embedder(timestep_embedding(timestep).to(dtype=hidden_dtype))


class AdaLayerNormSingle(nn.Module):
    """adaLN-single (PixArt-Alpha): the timestep embedding and `coefficient` modulation vectors from it."""

    def __init__(self, embedding_dim: int, embedding_coefficient: int = 6):
        super().__init__()
        self.emb = TimestepEmbedder(embedding_dim)
        self.linear = nn.Linear(embedding_dim, embedding_coefficient * embedding_dim, bias=True)

    def forward(self, timestep, hidden_dtype):
        embedded = self.emb(timestep, hidden_dtype)
        return self.linear(torch.nn.functional.silu(embedded)), embedded


class PixArtAlphaTextProjection(nn.Module):
    """LTX-2.0: caption projection after the embeddings connector."""

    def __init__(self, in_features: int, hidden_size: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_features, hidden_size, bias=True)
        self.linear_2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, caption):
        return self.linear_2(torch.nn.functional.gelu(self.linear_1(caption), approximate="tanh"))


class NormSingleLinearTextProjection(nn.Module):
    """Caption projection before the connector (caption_projection_first_linear), RMS-normalized input."""

    def __init__(self, in_features: int, hidden_size: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_features, hidden_size, bias=True)
        self.hidden_size = hidden_size
        self.in_features = in_features

    def forward(self, caption):
        return self.linear_1(rms_norm(caption) * (self.hidden_size / self.in_features) ** 0.5)


def compute_prompt_timestep(adaln: AdaLayerNormSingle | None, timestep_scaled, batch_size, hidden_dtype):
    """One prompt timestep per sample for the cross-attention AdaLN: the max across tokens, broadcast over the text."""
    if adaln is None:
        return None
    ts_input = timestep_scaled.max(dim=1, keepdim=True).values.flatten() if timestep_scaled.dim() > 1 else timestep_scaled.flatten()
    prompt_ts, _ = adaln(ts_input, hidden_dtype)
    return prompt_ts.view(batch_size, 1, prompt_ts.shape[-1])


# endregion

# region rotary embeddings


def _log_base(x, base):
    return np.log(x) / np.log(base)


@functools.lru_cache(maxsize=5)
def freq_grid_float64(theta: float, max_pos_count: int, inner_dim: int, _=None) -> torch.Tensor:
    # frequencies_precision "float64", which every published LTX-2.x file sets
    pow_indices = np.power(theta, np.linspace(_log_base(1, theta), _log_base(theta, theta), inner_dim // (2 * max_pos_count), dtype=np.float64))
    return torch.tensor(pow_indices * math.pi / 2, dtype=torch.float32)


def freq_grid_float32(theta: float, max_pos_count: int, inner_dim: int, device=None) -> torch.Tensor:
    indices = theta ** torch.linspace(math.log(1, theta), math.log(theta, theta), inner_dim // (2 * max_pos_count), device=device, dtype=torch.float32)
    return indices.to(dtype=torch.float32) * math.pi / 2


def generate_freqs(indices, indices_grid, max_pos, use_middle_indices_grid):
    if use_middle_indices_grid:
        assert len(indices_grid.shape) == 4 and indices_grid.shape[-1] == 2
        indices_grid = (indices_grid[..., 0] + indices_grid[..., 1]) / 2.0
    elif len(indices_grid.shape) == 4:
        indices_grid = indices_grid[..., 0]

    fractional = torch.stack([indices_grid[:, i] / max_pos[i] for i in range(indices_grid.shape[1])], dim=-1)
    indices = indices.to(device=fractional.device)
    return (indices * (fractional.unsqueeze(-1) * 2 - 1)).transpose(-1, -2).flatten(2)


def freqs_cis_matrix(freqs, pad_size, split_mode, num_attention_heads, out_dtype):
    cos_freq = freqs.cos().to(out_dtype)
    sin_freq = freqs.sin().to(out_dtype)
    if pad_size:
        matrix_pad = pad_size if split_mode else pad_size // 2
        cos_freq = torch.cat((torch.ones_like(cos_freq[:, :, :matrix_pad]), cos_freq), dim=-1)
        sin_freq = torch.cat((torch.zeros_like(sin_freq[:, :, :matrix_pad]), sin_freq), dim=-1)

    batch, tokens, half = cos_freq.shape
    cos_freq = cos_freq.reshape(batch, tokens, num_attention_heads, half // num_attention_heads)
    sin_freq = sin_freq.reshape(batch, tokens, num_attention_heads, half // num_attention_heads)
    rotation = torch.stack((cos_freq, -sin_freq, sin_freq, cos_freq), dim=-1)
    return rotation.reshape(*rotation.shape[:-1], 2, 2), split_mode


def apply_rotary_emb(x, freqs_cis):
    rotation, split = freqs_cis
    shape = x.shape
    x = x.reshape(x.shape[0], x.shape[1], rotation.shape[2], -1)
    out = ck.apply_rope_split_half1(x, rotation) if split else ck.apply_rope1(x, rotation)
    return out.reshape(shape)


def apply_rotary_emb_qk(q, k, freqs_cis):
    rotation, split = freqs_cis
    q_shape, k_shape = q.shape, k.shape
    q = q.reshape(q.shape[0], q.shape[1], rotation.shape[2], -1)
    k = k.reshape(k.shape[0], k.shape[1], rotation.shape[2], -1)
    q, k = ck.apply_rope_split_half(q, k, rotation) if split else ck.apply_rope(q, k, rotation)
    return q.reshape(q_shape), k.reshape(k_shape)


# endregion

# region attention


class GuideAttentionMask:
    """The two per-group masks for guide self-attention: noisy queries and tracked guide queries attend with a
    log-space bias on each other, so a guide strength below 1 weakens the guide."""

    __slots__ = ("guide_start", "tracked_count", "noisy_mask", "tracked_mask")

    def __init__(self, total_tokens, guide_start, tracked_count, tracked_weights):
        finfo = torch.finfo(tracked_weights.dtype)
        positive = tracked_weights > 0
        log_w = torch.full_like(tracked_weights, finfo.min)
        log_w[positive] = torch.log(tracked_weights[positive].clamp(min=finfo.tiny))

        self.guide_start = guide_start
        self.tracked_count = tracked_count
        self.noisy_mask = torch.zeros((1, 1, 1, total_tokens), device=tracked_weights.device, dtype=tracked_weights.dtype)
        self.noisy_mask[:, :, :, guide_start:guide_start + tracked_count] = log_w.view(1, 1, 1, -1)
        self.tracked_mask = torch.zeros((1, 1, tracked_count, total_tokens), device=tracked_weights.device, dtype=tracked_weights.dtype)
        self.tracked_mask[:, :, :, :guide_start] = log_w.view(1, 1, -1, 1)


def _attention_with_guide_mask(q, k, v, heads, guide_mask: GuideAttentionMask):
    # queries split into noisy and tracked-guide groups, each with its own small mask, instead of a dense T x T mask
    start = guide_mask.guide_start
    end = start + guide_mask.tracked_count
    out = torch.empty_like(q)
    if start > 0:
        out[:, :start, :] = attention_function(q[:, :start, :], k, v, heads, mask=guide_mask.noisy_mask)
    out[:, start:end, :] = attention_function(q[:, start:end, :], k, v, heads, mask=guide_mask.tracked_mask)
    if end < q.shape[1]:
        out[:, end:, :] = attention_function(q[:, end:, :], k, v, heads)
    return out


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, apply_gated_attention=False):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = query_dim if context_dim is None else context_dim
        self.heads = heads
        self.dim_head = dim_head

        # these norms span all heads, so the per-head RMS + RoPE kernel does not apply
        self.q_norm = nn.RMSNorm(inner_dim, eps=1e-5)
        self.k_norm = nn.RMSNorm(inner_dim, eps=1e-5)
        self.to_q = nn.Linear(query_dim, inner_dim, bias=True)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=True)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=True)
        self.to_gate_logits = nn.Linear(query_dim, heads, bias=True) if apply_gated_attention else None
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(0.0))

    def forward(self, x, context=None, mask=None, pe=None, k_pe=None, transformer_options={}):
        self_attn = context is None
        q = self.to_q(x)
        context = x if context is None else context
        k = self.to_k(context)
        v = self.to_v(context)

        # Spatio-Temporal Guidance: the flagged self-attention layers pass the values through
        if self_attn and transformer_options.get("stg_skip_self_attn", False):
            out = v
        else:
            q = self.q_norm(q)
            k = self.k_norm(k)
            if pe is not None:
                if k_pe is None and q.shape == k.shape:
                    q, k = apply_rotary_emb_qk(q, k, pe)
                else:
                    q = apply_rotary_emb(q, pe)
                    k = apply_rotary_emb(k, pe if k_pe is None else k_pe)

            if isinstance(mask, GuideAttentionMask):
                out = _attention_with_guide_mask(q, k, v, self.heads, mask)
            else:
                out = attention_function(q, k, v, self.heads, mask=mask)

        if self.to_gate_logits is not None:
            batch, tokens, _ = out.shape
            gates = 2.0 * torch.sigmoid(self.to_gate_logits(x))  # zero-init -> identity
            out = (out.view(batch, tokens, self.heads, self.dim_head) * gates.unsqueeze(-1)).view(batch, tokens, self.heads * self.dim_head)

        return self.to_out(out)


class GELU_approx(nn.Module):
    def __init__(self, dim_in, dim_out, bias=True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out, mult=4, ff_bias=True):
        super().__init__()
        inner_dim = int(dim * mult)
        # keys follow ComfyUI: net.0.proj, net.2
        self.net = nn.Sequential(GELU_approx(dim, inner_dim, bias=ff_bias), nn.Dropout(0.0), nn.Linear(inner_dim, dim_out, bias=ff_bias))

    def forward(self, x):
        return self.net[2](torch.nn.functional.gelu(self.net[0].proj(x), approximate="tanh"))


def apply_cross_attention_adaln(x, context, attn, q_shift, q_scale, q_gate, prompt_scale_shift_table, prompt_timestep, attention_mask=None, transformer_options={}):
    """Text cross-attention with AdaLN modulation of the query and of the text keys/values (LTX-2.3 and later)."""
    batch = x.shape[0]
    shift_kv, scale_kv = (prompt_scale_shift_table[None, None].to(device=x.device, dtype=x.dtype)
                          + prompt_timestep.reshape(batch, prompt_timestep.shape[1], 2, -1)).unbind(dim=2)
    attn_input = ck.rms_adaln(x, q_scale, q_shift)
    encoder_hidden_states = context * (1 + scale_kv) + shift_kv
    return attn(attn_input, context=encoder_hidden_states, mask=attention_mask, transformer_options=transformer_options) * q_gate


# endregion

# region transformer block


class CompressedTimestep:
    """Video timestep embeddings kept per frame (all patches of a frame share them) and expanded on use."""

    __slots__ = ("data", "batch_size", "num_frames", "patches_per_frame", "feature_dim")

    def __init__(self, tensor: torch.Tensor, patches_per_frame: int | None, per_frame: bool = False):
        self.batch_size, n, self.feature_dim = tensor.shape
        if per_frame:
            self.patches_per_frame = patches_per_frame
            self.num_frames = n
            self.data = tensor
        elif patches_per_frame is not None and n >= patches_per_frame and n % patches_per_frame == 0:
            self.patches_per_frame = patches_per_frame
            self.num_frames = n // patches_per_frame
            self.data = tensor.view(self.batch_size, self.num_frames, patches_per_frame, self.feature_dim)[:, :, 0, :].contiguous()
        else:
            self.patches_per_frame = 1
            self.num_frames = n
            self.data = tensor

    def expand(self):
        if self.patches_per_frame == 1:
            return self.data
        expanded = self.data.unsqueeze(2).expand(self.batch_size, self.num_frames, self.patches_per_frame, self.feature_dim)
        return expanded.reshape(self.batch_size, -1, self.feature_dim)

    def expand_for_computation(self, scale_shift_table: torch.Tensor, batch_size: int, indices: slice = slice(None, None)):
        params = scale_shift_table.shape[0]
        table = scale_shift_table[indices].unsqueeze(0).unsqueeze(0).to(device=self.data.device, dtype=self.data.dtype)
        if self.patches_per_frame == 1:
            reshaped = self.data.reshape(batch_size, self.data.shape[1], params, self.feature_dim // params)[:, :, indices, :]
            return (table + reshaped).unbind(dim=2)
        frame = self.data.reshape(batch_size, self.num_frames, params, -1)[:, :, indices, :]
        return tuple(value.unsqueeze(2).expand(batch_size, self.num_frames, self.patches_per_frame, -1).reshape(batch_size, -1, value.shape[-1])
                     for value in (table + frame).unbind(dim=2))


def ada_values(scale_shift_table, batch_size, timestep, indices: slice = slice(None, None)):
    if isinstance(timestep, CompressedTimestep):
        return timestep.expand_for_computation(scale_shift_table, batch_size, indices)
    params = scale_shift_table.shape[0]
    return (scale_shift_table[indices].unsqueeze(0).unsqueeze(0).to(device=timestep.device, dtype=timestep.dtype)
            + timestep.reshape(batch_size, timestep.shape[1], params, -1)[:, :, indices, :]).unbind(dim=2)


class BasicAVTransformerBlock(nn.Module):
    def __init__(self, v_dim, a_dim, v_heads, a_heads, vd_head, ad_head, v_context_dim=None, a_context_dim=None,
                 apply_gated_attention=False, cross_attention_adaln=False, ff_bias=True, audio_ff_bias=True):
        super().__init__()
        self.cross_attention_adaln = cross_attention_adaln
        gated = apply_gated_attention

        self.attn1 = CrossAttention(v_dim, heads=v_heads, dim_head=vd_head, apply_gated_attention=gated)
        self.audio_attn1 = CrossAttention(a_dim, heads=a_heads, dim_head=ad_head, apply_gated_attention=gated)
        self.attn2 = CrossAttention(v_dim, context_dim=v_context_dim, heads=v_heads, dim_head=vd_head, apply_gated_attention=gated)
        self.audio_attn2 = CrossAttention(a_dim, context_dim=a_context_dim, heads=a_heads, dim_head=ad_head, apply_gated_attention=gated)
        # Q video, K/V audio; and the reverse
        self.audio_to_video_attn = CrossAttention(v_dim, context_dim=a_dim, heads=a_heads, dim_head=ad_head, apply_gated_attention=gated)
        self.video_to_audio_attn = CrossAttention(a_dim, context_dim=v_dim, heads=a_heads, dim_head=ad_head, apply_gated_attention=gated)
        self.ff = FeedForward(v_dim, dim_out=v_dim, ff_bias=ff_bias)
        self.audio_ff = FeedForward(a_dim, dim_out=a_dim, ff_bias=audio_ff_bias)

        params = ADALN_CROSS_ATTN_PARAMS_COUNT if cross_attention_adaln else ADALN_BASE_PARAMS_COUNT
        self.scale_shift_table = nn.Parameter(torch.empty(params, v_dim))
        self.audio_scale_shift_table = nn.Parameter(torch.empty(params, a_dim))
        if cross_attention_adaln:
            self.prompt_scale_shift_table = nn.Parameter(torch.empty(2, v_dim))
            self.audio_prompt_scale_shift_table = nn.Parameter(torch.empty(2, a_dim))
        self.scale_shift_table_a2v_ca_audio = nn.Parameter(torch.empty(5, a_dim))
        self.scale_shift_table_a2v_ca_video = nn.Parameter(torch.empty(5, v_dim))

    def _text_cross_attention(self, x, context, attn, table, prompt_table, timestep, prompt_timestep, attention_mask, transformer_options):
        if self.cross_attention_adaln:
            shift_q, scale_q, gate = ada_values(table, x.shape[0], timestep, slice(6, 9))
            return apply_cross_attention_adaln(x, context, attn, shift_q, scale_q, gate, prompt_table, prompt_timestep, attention_mask, transformer_options)
        return attn(rms_norm(x), context=context, mask=attention_mask, transformer_options=transformer_options)

    def forward(self, x, v_context=None, a_context=None, attention_mask=None, v_timestep=None, a_timestep=None,
                v_pe=None, a_pe=None, v_cross_pe=None, a_cross_pe=None, v_cross_scale_shift_timestep=None,
                a_cross_scale_shift_timestep=None, v_cross_gate_timestep=None, a_cross_gate_timestep=None,
                transformer_options=None, self_attention_mask=None, v_prompt_timestep=None, a_prompt_timestep=None):
        transformer_options = transformer_options or {}
        run_vx = transformer_options.get("run_vx", True)
        run_ax = transformer_options.get("run_ax", True)
        vx, ax = x
        run_ax = run_ax and ax.numel() > 0
        run_a2v = run_vx and transformer_options.get("a2v_cross_attn", True) and ax.numel() > 0
        run_v2a = run_ax and transformer_options.get("v2a_cross_attn", True)

        if run_vx:
            shift, scale = ada_values(self.scale_shift_table, vx.shape[0], v_timestep, slice(0, 2))
            out = self.attn1(ck.rms_adaln(vx, scale, shift), pe=v_pe, mask=self_attention_mask, transformer_options=transformer_options)
            vx.addcmul_(out, ada_values(self.scale_shift_table, vx.shape[0], v_timestep, slice(2, 3))[0])
            vx.add_(self._text_cross_attention(vx, v_context, self.attn2, self.scale_shift_table, getattr(self, "prompt_scale_shift_table", None),
                                               v_timestep, v_prompt_timestep, attention_mask, transformer_options))

        if run_ax:
            shift, scale = ada_values(self.audio_scale_shift_table, ax.shape[0], a_timestep, slice(0, 2))
            out = self.audio_attn1(rms_norm(ax) * (1 + scale) + shift, pe=a_pe, transformer_options=transformer_options)
            ax.addcmul_(out, ada_values(self.audio_scale_shift_table, ax.shape[0], a_timestep, slice(2, 3))[0])
            ax.add_(self._text_cross_attention(ax, a_context, self.audio_attn2, self.audio_scale_shift_table, getattr(self, "audio_prompt_scale_shift_table", None),
                                               a_timestep, a_prompt_timestep, attention_mask, transformer_options))

        if run_a2v or run_v2a:
            ax_norm = rms_norm(ax)
            if run_a2v:
                a_scale, a_shift = ada_values(self.scale_shift_table_a2v_ca_audio[:4, :], ax.shape[0], a_cross_scale_shift_timestep)[:2]
                v_scale, v_shift = ada_values(self.scale_shift_table_a2v_ca_video[:4, :], vx.shape[0], v_cross_scale_shift_timestep)[:2]
                out = self.audio_to_video_attn(ck.rms_adaln(vx, v_scale, v_shift), context=ax_norm * (1 + a_scale) + a_shift,
                                               pe=v_cross_pe, k_pe=a_cross_pe, transformer_options=transformer_options)
                vx.addcmul_(out, ada_values(self.scale_shift_table_a2v_ca_video[4:, :], vx.shape[0], v_cross_gate_timestep)[0])
            if run_v2a:
                a_scale, a_shift = ada_values(self.scale_shift_table_a2v_ca_audio[:4, :], ax.shape[0], a_cross_scale_shift_timestep)[2:4]
                v_scale, v_shift = ada_values(self.scale_shift_table_a2v_ca_video[:4, :], vx.shape[0], v_cross_scale_shift_timestep)[2:4]
                out = self.video_to_audio_attn(ax_norm * (1 + a_scale) + a_shift, context=ck.rms_adaln(vx, v_scale, v_shift),
                                               pe=a_cross_pe, k_pe=v_cross_pe, transformer_options=transformer_options)
                ax.addcmul_(out, ada_values(self.scale_shift_table_a2v_ca_audio[4:, :], ax.shape[0], a_cross_gate_timestep)[0])

        if run_vx:
            shift, scale = ada_values(self.scale_shift_table, vx.shape[0], v_timestep, slice(3, 5))
            out = self.ff(ck.rms_adaln(vx, scale, shift))
            vx.addcmul_(out, ada_values(self.scale_shift_table, vx.shape[0], v_timestep, slice(5, 6))[0])

        if run_ax:
            shift, scale = ada_values(self.audio_scale_shift_table, ax.shape[0], a_timestep, slice(3, 5))
            out = self.audio_ff(rms_norm(ax) * (1 + scale) + shift)
            ax.addcmul_(out, ada_values(self.audio_scale_shift_table, ax.shape[0], a_timestep, slice(5, 6))[0])

        return vx, ax


# endregion

# region text embeddings connector


class BasicTransformerBlock1D(nn.Module):
    def __init__(self, dim, n_heads, d_head, apply_gated_attention=False, ff_bias=True):
        super().__init__()
        self.attn1 = CrossAttention(dim, heads=n_heads, dim_head=d_head, apply_gated_attention=apply_gated_attention)
        self.ff = FeedForward(dim, dim_out=dim, ff_bias=ff_bias)

    def forward(self, hidden_states, attention_mask=None, pe=None):
        hidden_states = self.attn1(rms_norm(hidden_states).squeeze(1), mask=attention_mask, pe=pe) + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        hidden_states = self.ff(rms_norm(hidden_states)) + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        return hidden_states


class Embeddings1DConnector(nn.Module):
    """Refines the projected text before the DiT: learnable registers fill the prompt up to 1024 tokens, then a few
    1D transformer blocks with RoPE over the token index."""

    def __init__(self, attention_head_dim=128, num_attention_heads=30, num_layers=2, positional_embedding_theta=10000.0,
                 positional_embedding_max_pos=(4096,), num_learnable_registers=128, apply_gated_attention=False,
                 connector_ff_bias=True, split_rope=False):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.inner_dim = num_attention_heads * attention_head_dim
        self.positional_embedding_theta = positional_embedding_theta
        self.positional_embedding_max_pos = list(positional_embedding_max_pos)
        self.split_rope = split_rope
        self.transformer_1d_blocks = nn.ModuleList([
            BasicTransformerBlock1D(self.inner_dim, num_attention_heads, attention_head_dim, apply_gated_attention=apply_gated_attention, ff_bias=connector_ff_bias)
            for _ in range(num_layers)
        ])
        self.num_learnable_registers = num_learnable_registers
        if num_learnable_registers:
            self.learnable_registers = nn.Parameter(torch.empty(num_learnable_registers, self.inner_dim))

    def freqs_cis(self, indices_grid, out_dtype):
        # double-precision RoPE grid, as ComfyUI builds both LTX connectors
        fractional = torch.stack([indices_grid[:, 0] / self.positional_embedding_max_pos[0]], dim=-1)
        indices = freq_grid_float64(self.positional_embedding_theta, indices_grid.shape[1], self.inner_dim).to(device=fractional.device)
        freqs = (indices * (fractional.unsqueeze(-1) * 2 - 1)).transpose(-1, -2).flatten(2)
        pad = self.inner_dim // 2 - freqs.shape[-1] if self.split_rope else self.inner_dim % 2
        return freqs_cis_matrix(freqs, pad, self.split_rope, self.num_attention_heads, out_dtype)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None):
        if self.num_learnable_registers:
            copies = math.ceil(max(1024, hidden_states.shape[1]) / self.num_learnable_registers)
            registers = torch.tile(self.learnable_registers.to(hidden_states), (copies, 1))
            hidden_states = torch.cat((hidden_states, registers[hidden_states.shape[1]:].unsqueeze(0).repeat(hidden_states.shape[0], 1, 1)), dim=1)
            if attention_mask is not None:
                attention_mask = torch.zeros([1, 1, 1, hidden_states.shape[1]], dtype=attention_mask.dtype, device=attention_mask.device)

        indices_grid = torch.arange(hidden_states.shape[1], dtype=torch.float32, device=hidden_states.device)[None, None, :]
        pe = self.freqs_cis(indices_grid, hidden_states.dtype)
        for block in self.transformer_1d_blocks:
            hidden_states = block(hidden_states, attention_mask=attention_mask, pe=pe)
        return rms_norm(hidden_states), attention_mask


# endregion
