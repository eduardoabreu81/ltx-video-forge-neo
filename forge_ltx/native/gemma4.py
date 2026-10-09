"""Gemma 4 12B (gemma4_unified), the text encoder of LTX-2.5, text path only.

Ported from ComfyUI comfy/text_encoders/gemma4.py (Gemma4Attention, TransformerBlockGemma4, Gemma4Transformer),
keeping what one prefill pass over a prompt needs: no KV cache, no KV sharing and no per-layer inputs (the 12B has
neither), no vision or audio embedders (LTX encodes text). The blocks are built on Forge Neo's operations (the loader
swaps Linear, RMSNorm and Embedding), its MLP and attention, and comfy-kitchen's split-half RoPE.

Against Gemma 3: five sliding layers (head 256, 8 K/V heads, local RoPE) then one global layer (head 512, one K/V head,
no v_proj: V is the raw K projection, RoPE on a quarter of the pairs). V is RMS-normalized without a weight, the norms
use the weight itself (not 1 + w), the attention scale is 1 (the q/k norms carry it) and every layer output is
multiplied by its layer_scalar.
"""

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
from backend.nn.llm.llama import MLP, BaseLlama, attention_function
from backend.quant_ops import ck


@dataclass
class Gemma4_12BConfig:
    vocab_size: int = 262144
    hidden_size: int = 3840
    intermediate_size: int = 15360
    num_hidden_layers: int = 48
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    num_global_key_value_heads: int = 1
    rms_norm_eps: float = 1e-6
    head_dim: int = 256
    global_head_dim: int = 512
    partial_rotary_factor: float = 0.25
    transformer_type: str = "gemma4"
    final_norm: bool = True
    # not dataclass fields: the global layers use the first theta, the sliding ones the second
    rope_theta = [1000000.0, 10000.0]
    sliding_attention = [1024, 1024, 1024, 1024, 1024, False]
    mlp_activation = "gelu_pytorch_tanh"


# the published files keep these beside the language model; LTX never runs them
VISION_PREFIXES = ("vision_model.", "multi_modal_projector.", "audio_projector.")


class Gemma4Attention(nn.Module):
    def __init__(self, config: Gemma4_12BConfig, head_dim: int, num_kv_heads: int, k_eq_v: bool):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.eps = config.rms_norm_eps

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, num_kv_heads * head_dim, bias=False)
        # k_eq_v: V reuses the K projection, there is no v_proj weight
        self.v_proj = None if k_eq_v else nn.Linear(config.hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * head_dim, config.hidden_size, bias=False)
        self.q_norm = nn.RMSNorm(head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None, freqs_cis: torch.Tensor, sliding_window: int | None):
        batch, seq, _ = x.shape
        xq = self.q_norm(self.q_proj(x).view(batch, seq, self.num_heads, self.head_dim)).transpose(1, 2)
        xk = self.k_proj(x).view(batch, seq, self.num_kv_heads, self.head_dim)
        xv = xk if self.v_proj is None else self.v_proj(x).view(batch, seq, self.num_kv_heads, self.head_dim)
        xk = self.k_norm(xk).transpose(1, 2)
        xv = torch.nn.functional.rms_norm(xv, (self.head_dim,), eps=self.eps).transpose(1, 2)

        xq = ck.apply_rope_split_half1(xq, freqs_cis)
        xk = ck.apply_rope_split_half1(xk, freqs_cis)

        # as ComfyUI, to match the reference numerics: K/V heads are expanded before a masked sliding window, and
        # passed grouped otherwise
        expand_kv = self.num_heads != self.num_kv_heads and sliding_window is not None and xk.shape[2] >= sliding_window
        if expand_kv:
            xk = xk.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            xv = xv.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        gqa = {"enable_gqa": True} if self.num_heads != self.num_kv_heads and not expand_kv else {}
        out = attention_function(xq, xk, xv, self.num_heads, mask=mask, skip_reshape=True, scale=1.0, **gqa)
        return self.o_proj(out)


class Gemma4Block(nn.Module):
    def __init__(self, config: Gemma4_12BConfig, index: int):
        super().__init__()
        self.sliding_attention = config.sliding_attention[index % len(config.sliding_attention)]
        if self.sliding_attention:
            self.self_attn = Gemma4Attention(config, config.head_dim, config.num_key_value_heads, k_eq_v=False)
        else:
            self.self_attn = Gemma4Attention(config, config.global_head_dim, config.num_global_key_value_heads, k_eq_v=True)
        self.mlp = MLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.register_buffer("layer_scalar", torch.ones(1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None, freqs_cis: list[torch.Tensor]):
        window = None
        if self.sliding_attention:
            window = self.sliding_attention
            if x.shape[1] > window:
                far = torch.ones(x.shape[1], x.shape[1], dtype=torch.bool, device=x.device).tril_(-window)
                sliding = torch.zeros(x.shape[1], x.shape[1], dtype=x.dtype, device=x.device).masked_fill_(far, torch.finfo(x.dtype).min)
                mask = sliding if mask is None else mask + sliding
            freqs = freqs_cis[1]
        else:
            freqs = freqs_cis[0]

        x = x + self.post_attention_layernorm(self.self_attn(self.input_layernorm(x), mask, freqs, window))
        x = x + self.post_feedforward_layernorm(self.mlp(self.pre_feedforward_layernorm(x)))
        return x * self.layer_scalar.to(x)


def _inv_freq(config: Gemma4_12BConfig) -> tuple[torch.Tensor, torch.Tensor]:
    # computed on the CPU in float32, as the reference does; the global layers rotate only the first quarter of their
    # pairs, the remaining frequencies are zero (identity)
    angles = int(config.partial_rotary_factor * config.global_head_dim // 2)
    unrotated = config.global_head_dim // 2 - angles
    global_inv = 1.0 / (config.rope_theta[0] ** (torch.arange(0, 2 * angles, 2).float() / config.global_head_dim))
    global_inv = torch.cat([global_inv, torch.zeros(unrotated)])
    sliding_inv = 1.0 / (config.rope_theta[1] ** (torch.arange(0, config.head_dim, 2).float() / config.head_dim))
    return global_inv, sliding_inv


def _rotations(inv_freq: torch.Tensor, position_ids: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Per-pair 2x2 rotation matrices [B, 1, S, d/2, 2, 2] for comfy-kitchen's split-half RoPE."""
    freqs = (inv_freq.to(position_ids.device)[None, :, None].expand(position_ids.shape[0], -1, 1) @ position_ids[:, None, :].float()).transpose(1, 2)
    cos, sin = freqs.cos(), freqs.sin()
    matrix = torch.stack((torch.stack((cos, -sin), dim=-1), torch.stack((sin, cos), dim=-1)), dim=-2)
    return matrix.unsqueeze(1).to(dtype)


class Gemma4Transformer(nn.Module):
    def __init__(self, config: Gemma4_12BConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Gemma4Block(config, index=i) for i in range(config.num_hidden_layers)])
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        global_inv, sliding_inv = _inv_freq(config)
        self.register_buffer("_global_inv_freq", global_inv, persistent=False)
        self.register_buffer("_sliding_inv_freq", sliding_inv, persistent=False)

    def forward(self, x, attention_mask=None, embeds=None, num_tokens=None, intermediate_output=None, final_layer_norm_intermediate=True, dtype=None, embeds_info=(), **kwargs):
        x = embeds if embeds is not None else self.embed_tokens(x).to(dtype=dtype)
        # the reference scales the embedding by a Python scalar, at full precision
        x = x * self.config.hidden_size**0.5

        seq = x.shape[1]
        position_ids = torch.arange(seq, device=x.device).unsqueeze(0)
        freqs_cis = [_rotations(self._global_inv_freq, position_ids, x.dtype), _rotations(self._sliding_inv_freq, position_ids, x.dtype)]

        low = torch.finfo(x.dtype).min
        mask = None
        if attention_mask is not None:
            mask = 1.0 - attention_mask.to(x.dtype).reshape((attention_mask.shape[0], 1, -1, attention_mask.shape[-1])).expand(attention_mask.shape[0], 1, seq, attention_mask.shape[-1])
            mask = mask.masked_fill(mask.to(torch.bool), low)
        if seq > 1:
            causal = torch.zeros(seq, seq, dtype=x.dtype, device=x.device).masked_fill_(torch.ones(seq, seq, dtype=torch.bool, device=x.device).triu_(1), low)
            mask = causal if mask is None else mask + causal

        if intermediate_output == "all":
            hidden = []
            intermediate_output = None
        else:
            hidden = None
            if intermediate_output is not None and intermediate_output < 0:
                intermediate_output = len(self.layers) + intermediate_output

        intermediate = None
        for i, layer in enumerate(self.layers):
            if hidden is not None:
                hidden.append(x.unsqueeze(1).clone())
            x = layer(x, mask, freqs_cis)
            if i == intermediate_output:
                intermediate = x.clone()

        x = self.norm(x)
        if hidden is not None:
            hidden.append(x.unsqueeze(1).clone())
            intermediate = torch.cat(hidden, dim=1)
        if intermediate is not None and final_layer_norm_intermediate:
            intermediate = self.norm(intermediate)
        return x, intermediate


class Gemma4_12B(BaseLlama, nn.Module):
    def __init__(self, config_dict: dict | None = None, config: Gemma4_12BConfig | None = None):
        super().__init__()
        config = config or Gemma4_12BConfig()
        for key, value in asdict(config).items():
            if config_dict and key in config_dict:
                assert value == config_dict[key], f"[LTX] Gemma 4 12B expects {key}={value}, got {config_dict[key]}"

        self.num_layers = config.num_hidden_layers
        self.model = Gemma4Transformer(config)
