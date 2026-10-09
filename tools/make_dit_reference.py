"""Builds tests/fixtures/ltx_dit_tiny_reference.pt with ComfyUI's own LTXAVModel: tiny models in the layout of
LTX-2.0, 2.3 and 2.5, their random weights, inputs and outputs, as the reference for forge_ltx.native.transformer.

Needs a ComfyUI checkout and its requirements in a separate environment (CPU is enough):

    D:/ltx-forge-dev/ref-venv/Scripts/python.exe tools/make_dit_reference.py [--comfyui D:/ltx-forge-dev/comfyui-ref]
"""

import argparse
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "ltx_dit_tiny_reference.pt"

# the settings every published file carries in its metadata, scaled down: video 4 heads x 8, audio 4 heads x 4
COMMON = dict(
    in_channels=8, audio_in_channels=128, num_attention_heads=4, attention_head_dim=8, audio_num_attention_heads=4,
    audio_attention_head_dim=4, cross_attention_dim=32, audio_cross_attention_dim=16, num_layers=2,
    causal_temporal_positioning=True, use_middle_indices_grid=True, rope_type="split", frequencies_precision="float64",
    av_ca_timestep_scale_multiplier=1000.0, timestep_scale_multiplier=1000, positional_embedding_max_pos=[20, 2048, 2048],
    audio_positional_embedding_max_pos=[20], positional_embedding_theta=10000.0,
)
VARIANTS = {
    "2.0": dict(COMMON, caption_channels=24, connector_attention_head_dim=8, connector_num_attention_heads=3, connector_num_layers=2),
    "2.3": dict(COMMON, caption_channels=24, apply_gated_attention=True, caption_proj_before_connector=True,
                caption_projection_first_linear=False, cross_attention_adaln=True, connector_apply_gated_attention=True,
                connector_attention_head_dim=8, connector_num_attention_heads=4, connector_num_layers=3,
                audio_connector_attention_head_dim=4, audio_connector_num_attention_heads=4),
}
VARIANTS["2.5"] = dict(VARIANTS["2.3"], ff_bias=False, use_keyframes_abs_pos_embedding=True)

F, H, W, T_AUDIO, T_TEXT = 3, 2, 3, 7, 5


def scenarios(torch, variant):
    """(name, kwargs for forward) with fresh inputs; the guide scenario appends one keyframe latent frame."""
    g = torch.Generator().manual_seed(1)
    vx = torch.randn(1, 8, F, H, W, generator=g)
    ax = torch.randn(1, 8, T_AUDIO, 16, generator=g)
    sigma = torch.tensor([0.7])
    tokens = F * H * W

    first_frame_clean = torch.full((1, tokens), 0.7)
    first_frame_clean[:, :H * W] = 0.0  # image-to-video in place: the first latent frame is not denoised
    i2v_mask = torch.ones(1, 1, F, H, W)
    i2v_mask[:, :, 0] = 0.0

    guide = torch.randn(1, 8, 1, H, W, generator=g)
    guide_mask = torch.ones(1, 1, F + 1, H, W)
    guide_mask[:, :, -1] = 0.3  # 1 - strength
    ys, xs = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    start = torch.stack([torch.full_like(ys, 16), ys * 32, xs * 32]).reshape(3, -1).float()  # pixel frame 16
    keyframe_idxs = torch.stack([start, start + torch.tensor([1.0, 32.0, 32.0])[:, None]], dim=-1)[None]
    guide_t = torch.cat([torch.full((1, tokens), 0.7), torch.full((1, H * W), 0.7 * 0.3)], dim=1)

    return [
        ("t2v", dict(x=[vx, ax], timestep=sigma)),
        ("i2v", dict(x=[vx, ax], timestep=(first_frame_clean, sigma), denoise_mask=i2v_mask)),
        ("guide", dict(x=[torch.cat([vx, guide], dim=2), ax], timestep=(guide_t, sigma), denoise_mask=guide_mask,
                       keyframe_idxs=keyframe_idxs,
                       guide_attention_entries=[{"pre_filter_count": H * W, "strength": 0.7, "pixel_mask": None, "latent_shape": [1, H, W]}])),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfyui", default="D:/ltx-forge-dev/comfyui-ref")
    args = parser.parse_args()
    sys.path.insert(0, args.comfyui)
    sys.argv = [sys.argv[0], "--cpu"]
    import comfy.options

    comfy.options.enable_args_parsing()
    import comfy.ops
    import torch
    from comfy.ldm.lightricks.av_model import LTXAVModel

    result = {}
    for name, config in VARIANTS.items():
        torch.manual_seed(0)
        model = LTXAVModel(**config, dtype=torch.float32, device="cpu", operations=comfy.ops.disable_weight_init).eval()
        with torch.no_grad():
            for key, param in model.named_parameters():
                scale = 0.3 if "scale_shift_table" in key or key.endswith("norm.weight") else 0.1
                param.copy_(torch.randn_like(param) * scale + (1.0 if key.endswith("norm.weight") else 0.0))

        text_dim = config["cross_attention_dim"] + config["audio_cross_attention_dim"] if config.get("caption_proj_before_connector") else config["caption_channels"]
        text = torch.randn(1, T_TEXT, text_dim, generator=torch.Generator().manual_seed(2))
        with torch.no_grad():
            context = model.preprocess_text_embeds(text.clone(), unprocessed=True)
            outputs = {}
            for scenario, kwargs in scenarios(torch, name):
                inputs = {k: ([t.clone() for t in v] if k == "x" else v) for k, v in kwargs.items()}
                out = model(context=context.clone(), attention_mask=None, frame_rate=24, transformer_options={}, **inputs)
                outputs[scenario] = [t.clone() for t in out] if isinstance(out, list) else out.clone()
        result[name] = {"config": config, "state_dict": {k: v.clone() for k, v in model.state_dict().items()},
                        "text": text, "context": context, "outputs": outputs}
        print(f"{name}: {sum(p.numel() for p in model.parameters())} parameters, context {tuple(context.shape)}")

    torch.save(result, OUT)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
