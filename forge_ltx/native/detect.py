"""Which LTX-2.x a DiT is and the LTXAVModel settings it needs, from its tensor names and shapes (and its metadata).

ComfyUI comfy/model_detection.py reads a few shapes and then the "transformer" section of the file's "config"
metadata. Community repacks (GGUF and others) often drop the metadata, so each version also has the settings its
official files carry; the tensors decide what they can (gated attention, cross-attention AdaLN, keyframe embedding,
feed-forward bias, connector depth), the metadata the rest.
"""

import json

PREFIX = "model.diffusion_model."

# the "transformer" metadata of the official files, reduced to what LTXAVModel reads
_COMMON = {
    "caption_channels": 3840, "cross_attention_dim": 4096, "audio_cross_attention_dim": 2048, "attention_head_dim": 128,
    "audio_attention_head_dim": 64, "num_attention_heads": 32, "audio_num_attention_heads": 32, "in_channels": 128,
    "num_layers": 48, "causal_temporal_positioning": True, "use_middle_indices_grid": True, "rope_type": "split",
    "frequencies_precision": "float64", "av_ca_timestep_scale_multiplier": 1000.0, "timestep_scale_multiplier": 1000,
    "positional_embedding_max_pos": [20, 2048, 2048], "audio_positional_embedding_max_pos": [20],
    "positional_embedding_theta": 10000.0, "connector_attention_head_dim": 128,
}
PRESETS = {
    "2.0": {**_COMMON, "connector_num_attention_heads": 30, "connector_num_layers": 2},
    "2.3": {**_COMMON, "apply_gated_attention": True, "caption_proj_before_connector": True, "caption_projection_first_linear": False,
            "cross_attention_adaln": True, "connector_apply_gated_attention": True, "connector_num_attention_heads": 32,
            "connector_num_layers": 8, "audio_connector_attention_head_dim": 64, "audio_connector_num_attention_heads": 32},
}
PRESETS["2.5"] = {**PRESETS["2.3"], "ff_bias": False, "use_keyframes_abs_pos_embedding": True}

# metadata keys LTXAVModel takes; the rest of the diffusers-style config is ignored
MODEL_KEYS = set(PRESETS["2.5"]) | {"audio_ff_bias", "use_prompt_adaln_single", "vae_scale_factors"}


def is_ltx(keys, prefix: str = PREFIX) -> bool:
    return f"{prefix}adaln_single.emb.timestep_embedder.linear_1.bias" in keys and f"{prefix}audio_adaln_single.linear.weight" in keys


def version(keys, prefix: str = PREFIX) -> str:
    if f"{prefix}keyframes_abs_pos_embedding" in keys:
        return "2.5"
    if f"{prefix}transformer_blocks.0.prompt_scale_shift_table" in keys:
        return "2.3"
    return "2.0"


def _count(keys, prefix: str) -> int:
    return len({k[len(prefix):].split(".", 1)[0] for k in keys if k.startswith(prefix)})


def dit_config(shapes: dict, metadata: dict | None = None, prefix: str = PREFIX) -> dict:
    """shapes: tensor name -> shape (a state dict's or a header's). Returns LTXAVModel keyword arguments plus "version"."""
    keys = set(shapes)
    found = version(keys, prefix)
    config = dict(PRESETS[found])

    raw = (metadata or {}).get("config")
    if raw:
        transformer = (json.loads(raw) if isinstance(raw, str) else raw).get("transformer", {})
        config.update({k: v for k, v in transformer.items() if k in MODEL_KEYS})

    # input widths may be packed (NVFP4 halves them, and mixed files keep some layers unpacked), so the cross-attention
    # width comes from the metadata or the preset; output widths never are
    block = f"{prefix}transformer_blocks.0."
    config.update(
        num_layers=_count(keys, f"{prefix}transformer_blocks."),
        attention_head_dim=shapes[f"{block}attn2.to_k.weight"][0] // config["num_attention_heads"],
        apply_gated_attention=f"{block}attn1.to_gate_logits.weight" in keys,
        cross_attention_adaln=f"{block}prompt_scale_shift_table" in keys,
        use_keyframes_abs_pos_embedding=f"{prefix}keyframes_abs_pos_embedding" in keys,
        ff_bias=f"{block}ff.net.0.proj.bias" in keys,
        audio_ff_bias=f"{block}audio_ff.net.0.proj.bias" in keys,
        connector_num_layers=_count(keys, f"{prefix}video_embeddings_connector.transformer_1d_blocks."),
        connector_apply_gated_attention=f"{prefix}video_embeddings_connector.transformer_1d_blocks.0.attn1.to_gate_logits.weight" in keys,
        caption_proj_before_connector=f"{prefix}caption_projection.linear_1.weight" not in keys,
    )
    config["version"] = found
    return config
