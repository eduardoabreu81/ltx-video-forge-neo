"""Builds tests/fixtures/ltx_upsampler_tiny_reference.pt with ComfyUI's own LTX latent upscaler and image resize: tiny
upscalers in the published x2 layout and the other ComfyUI variants (rational x1.5, 2D convolutions, temporal x2), and
common_upscale's bilinear center crop as LTXVImgToVideoInplace uses it, as the reference for
forge_ltx.native.upsampler and guides.fit.

    D:/ltx-forge-dev/ref-venv/Scripts/python.exe tools/make_upsampler_reference.py [--comfyui D:/ltx-forge-dev/comfyui-ref]
"""

import argparse
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "ltx_upsampler_tiny_reference.pt"

_COMMON = {"_class_name": "LatentUpsampler", "in_channels": 8, "mid_channels": 32, "num_blocks_per_stage": 1}
CONFIGS = {
    # the layout LTX-2.0, 2.3 and 2.5 publish, scaled down
    "x2": {**_COMMON, "dims": 3, "spatial_upsample": True, "temporal_upsample": False, "spatial_scale": 2.0, "rational_resampler": False},
    "x1.5": {**_COMMON, "dims": 3, "spatial_upsample": True, "temporal_upsample": False, "spatial_scale": 1.5, "rational_resampler": True},
    "x2-2d": {**_COMMON, "dims": 2, "spatial_upsample": True, "temporal_upsample": False, "spatial_scale": 2.0, "rational_resampler": False},
    "temporal": {**_COMMON, "dims": 3, "spatial_upsample": False, "temporal_upsample": True, "spatial_scale": 2.0, "rational_resampler": False},
}
# (source H, W) -> (target H, W) for the resize
RESIZES = [((40, 64), (20, 32)), ((48, 80), (32, 32)), ((70, 30), (32, 48))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfyui", default="D:/ltx-forge-dev/comfyui-ref")
    args = parser.parse_args()
    sys.path.insert(0, args.comfyui)
    sys.argv = [sys.argv[0], "--cpu"]
    import comfy.options

    comfy.options.enable_args_parsing()
    import comfy.ops
    import comfy.utils
    import torch
    from comfy.ldm.lightricks.latent_upsampler import LatentUpsampler

    result = {"upsamplers": {}, "resizes": []}
    latent = torch.randn(1, 8, 3, 6, 10, generator=torch.Generator().manual_seed(1))
    for name, config in CONFIGS.items():
        model = LatentUpsampler.from_config(config, operations=comfy.ops.disable_weight_init).eval()
        g = torch.Generator().manual_seed(0)
        with torch.no_grad():
            for tensor in model.state_dict(keep_vars=True).values():
                tensor.copy_((torch.randn(tensor.shape, generator=g) * 0.1).half().float())
            output = model(latent.clone())
        state = {k: v.half() for k, v in model.state_dict().items()}
        result["upsamplers"][name] = {"config": config, "state_dict": state, "latent": latent, "output": output}
        print(f"{name}: {tuple(latent.shape)} -> {tuple(output.shape)}")

    g = torch.Generator().manual_seed(2)
    for (h, w), (th, tw) in RESIZES:
        image = torch.rand(1, h, w, 3, generator=g)
        out = comfy.utils.common_upscale(image.movedim(-1, 1), tw, th, "bilinear", "center").movedim(1, -1)
        result["resizes"].append({"image": image, "height": th, "width": tw, "output": out})

    torch.save(result, OUT)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
