"""Read the safetensors headers of the LTX-2.x files on the Hugging Face Hub, without downloading the weights.

Writes tests/fixtures/ltx_headers.json.gz: for each file, its size, metadata and every tensor's dtype and shape.
The CPU tests build the modules on the meta device and compare them with these headers.

Gated repositories (Lightricks/*) need a token from an account that accepted the license (`hf auth login`).

    python tools/probe_headers.py [--only 2.5|2.3|2.0] [--out PATH]
"""

import argparse
import gzip
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi

FILES = {
    "2.5": [
        ("Lightricks/LTX-2.5", "diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"),
        ("Lightricks/LTX-2.5", "diffusion_models/ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors"),
        ("Lightricks/LTX-2.5", "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"),
        ("Lightricks/LTX-2.5", "diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors"),
        ("Lightricks/LTX-2.5", "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"),
        ("Lightricks/LTX-2.5", "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"),
        ("Lightricks/LTX-2.5", "vae/ltx-2.5-video-vae-conv-bf16.safetensors"),
        ("Lightricks/LTX-2.5", "vae/ltx-2.5-video-vae-bf16.safetensors"),
        ("Lightricks/LTX-2.5", "vae/ltx-2.5-audio-vae-bf16.safetensors"),
        ("Lightricks/LTX-2.5", "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"),
        ("Lightricks/LTX-2.5", "model_patches/ltx-2.5-duration-head-bf16.safetensors"),
        ("Lightricks/LTX-2.5", "loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors"),
    ],
    "2.3": [
        ("Lightricks/LTX-2.3", "ltx-2.3-22b-distilled-1.1.safetensors"),
        ("Lightricks/LTX-2.3", "ltx-2.3-22b-dev.safetensors"),
        ("Lightricks/LTX-2.3", "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"),
        ("Lightricks/LTX-2.3", "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"),
    ],
    "2.0": [
        ("Lightricks/LTX-2", "ltx-2-19b-distilled-fp8.safetensors"),
        ("Lightricks/LTX-2", "ltx-2-19b-distilled.safetensors"),
        ("Lightricks/LTX-2", "ltx-2-19b-dev-fp4.safetensors"),
        ("Lightricks/LTX-2", "ltx-2-19b-distilled-lora-384.safetensors"),
        ("Comfy-Org/ltx-2", "split_files/text_encoders/gemma_3_12B_it.safetensors"),
        ("Comfy-Org/ltx-2", "split_files/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors"),
        ("Comfy-Org/ltx-2", "split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors"),
    ],
}

# ungated copies of gated repositories; entries read from them are provisional
MIRRORS = {"Lightricks/LTX-2.5": "comfyicu/LTX-2.5"}

DEFAULT_OUT =Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "ltx_headers.json.gz"


def probe(api, repo, filename):
    sizes = {s.path: s.size for s in api.get_paths_info(repo, [filename])}
    info = api.parse_safetensors_file_metadata(repo, filename)
    return {
        "repo": repo,
        "size": sizes.get(filename),
        "metadata": info.metadata or {},
        "tensors": {name: [t.dtype, list(t.shape)] for name, t in sorted(info.tensors.items())},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=sorted(FILES))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--mirror", action="store_const", const=MIRRORS,
                        help="read gated repositories from an ungated mirror (provisional, until a token is set)")
    args = parser.parse_args()

    result = {}
    if args.out.exists():
        with gzip.open(args.out, "rt", encoding="utf-8") as f:
            result = json.load(f)

    api = HfApi()
    failed = []
    for version, files in FILES.items():
        if args.only and version != args.only:
            continue
        for repo, filename in files:
            name = filename.rsplit("/", 1)[-1]
            if args.mirror and repo in args.mirror:
                repo = args.mirror[repo]
            try:
                entry = probe(api, repo, filename)
            except Exception as error:  # gated, missing or renamed: report and go on
                failed.append(name)
                print(f"FAIL {version} {name}: {type(error).__name__}: {str(error).splitlines()[0]}")
                continue
            entry["version"] = version
            result[name] = entry
            print(f"ok   {version} {name}: {len(entry['tensors'])} tensors")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.out, "wt", encoding="utf-8") as f:
        json.dump(result, f, separators=(",", ":"), sort_keys=True)
    print(f"{len(result)} files in {args.out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
