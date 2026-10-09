"""Builds tests/fixtures/gemma4_tiny_reference.pt: a tiny Gemma 4 unified text model from transformers, its random
weights, a left-padded prompt and the hidden state of every layer, as the reference for forge_ltx.native.gemma4.

Forge Neo pins transformers 4.57, which has no Gemma 4; run this with a newer transformers in a separate environment:

    D:/ltx-forge-dev/ref-venv/Scripts/python.exe tools/make_gemma4_reference.py
"""

from pathlib import Path

import torch
from transformers.models.gemma4_unified import (
    Gemma4UnifiedTextConfig,
    Gemma4UnifiedTextModel,
)

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "gemma4_tiny_reference.pt"

# the layout of the LTX-2.5 text encoder (gemma_config in its header), scaled down: five sliding layers then one
# global layer with a single K/V head shared by K and V, RoPE on a quarter of the global pairs
CONFIG = dict(
    vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=6, num_attention_heads=4,
    num_key_value_heads=2, num_global_key_value_heads=1, head_dim=16, global_head_dim=32, sliding_window=8,
    layer_types=["sliding_attention"] * 5 + ["full_attention"], attention_k_eq_v=True, hidden_size_per_layer_input=0,
    num_kv_shared_layers=0, rms_norm_eps=1e-6, hidden_activation="gelu_pytorch_tanh",
    rope_parameters={"full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"},
                     "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}},
)
PAD = 4  # left padding, as LTX pads every prompt to 1024 tokens
TOKENS = 20


def main():
    torch.manual_seed(0)
    config = Gemma4UnifiedTextConfig(**CONFIG)
    config._attn_implementation = "eager"
    model = Gemma4UnifiedTextModel(config).eval()
    with torch.no_grad():
        # random norms and layer scalars too, so a weight read as 1 + w or a skipped scalar shows
        for name, param in model.named_parameters():
            param.copy_(torch.randn_like(param) * (0.3 if param.ndim == 1 else 0.05) + (1.0 if param.ndim == 1 else 0.0))
        for layer in model.layers:
            layer.layer_scalar.copy_(torch.rand(1) + 0.5)

    ids = torch.randint(3, config.vocab_size, (1, TOKENS))
    ids[0, :PAD] = 0
    ids[0, PAD] = 2  # BOS
    mask = torch.ones_like(ids)
    mask[0, :PAD] = 0
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "config": CONFIG,
        "state_dict": {k: v.clone() for k, v in model.state_dict().items()},
        "input_ids": ids,
        "attention_mask": mask,
        "hidden_states": torch.stack(out.hidden_states, dim=1),  # [B, layers + 1, T, D]
    }, OUT)
    print(f"{len(out.hidden_states)} hidden states, {sum(p.numel() for p in model.parameters())} parameters -> {OUT}")


if __name__ == "__main__":
    main()
