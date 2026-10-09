"""Gemma 3 12B, the text encoder of LTX-2.0 and LTX-2.3, on Forge Neo's Llama2_.

Forge Neo's Llama2_ already runs the Gemma 3 block (sliding window, a global and a local RoPE) but ships only the
Gemma 2 2B size; the 12B config comes from ComfyUI comfy/text_encoders/llama.py Gemma3_12B_Config. LTX reads the
text path only, so the SigLIP tower and its projector, which the published files also hold, are not built.
"""

from dataclasses import asdict, dataclass

import torch.nn as nn
from backend.nn.llm.llama import BaseLlama, Llama2_


@dataclass
class Gemma3_12BConfig:
    vocab_size: int = 262208
    hidden_size: int = 3840
    intermediate_size: int = 15360
    num_hidden_layers: int = 48
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-6
    transformer_type: str = "gemma3"
    final_norm: bool = True
    # not dataclass fields, as in ComfyUI: the global layers use the first theta and scale, the sliding ones the second
    rope_theta = [1000000.0, 10000.0]
    rope_scale = [8.0, 1.0]
    head_dim = 256
    rms_norm_add = True
    mlp_activation = "gelu_pytorch_tanh"
    qkv_bias = False
    rope_dims = None
    q_norm = "gemma3"
    k_norm = "gemma3"
    sliding_attention = [1024, 1024, 1024, 1024, 1024, False]


# the published files keep these beside the language model; LTX never runs them
VISION_PREFIXES = ("vision_model.", "multi_modal_projector.")


class Gemma3_12B(BaseLlama, nn.Module):
    def __init__(self, config_dict: dict | None = None, config: Gemma3_12BConfig | None = None):
        super().__init__()
        config = config or Gemma3_12BConfig()
        for key, value in asdict(config).items():
            if config_dict and key in config_dict:
                assert value == config_dict[key], f"[LTX] Gemma 3 12B expects {key}={value}, got {config_dict[key]}"

        self.num_layers = config.num_hidden_layers
        self.model = Llama2_(config)
