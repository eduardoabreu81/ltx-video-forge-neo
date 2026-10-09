"""Puts a Forge Neo checkout on the import path for the CPU tests, in CPU mode.

LTX_FORGE_DIR points at it (default D:/ltx-forge-dev/forge-neo). Without it the tests that need Forge's backend skip.
"""

import gzip
import json
import os
import sys
from pathlib import Path

FORGE_DIR = Path(os.environ.get("LTX_FORGE_DIR", "D:/ltx-forge-dev/forge-neo"))
FIXTURES = Path(__file__).with_name("fixtures")
# quantization side tensors next to a quantized weight
AUX = (".comfy_quant", ".weight_scale", ".weight_scale_2", ".input_scale", ".weight_s_channel", ".weight_s_rel")


def setup() -> bool:
    if not (FORGE_DIR / "backend").is_dir():
        return False
    for path in (FORGE_DIR / "modules_forge" / "packages", FORGE_DIR):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    # backend.args reads the command line with parse_known_args, modules.shared (which backend.loader imports) with a
    # strict parse_args: both are imported here, once, with Forge's own command line
    argv = sys.argv
    sys.argv = [argv[0], "--cpu"]
    try:
        import backend.memory_management  # noqa: F401
        import modules.shared  # noqa: F401
    except Exception:
        return False
    finally:
        sys.argv = argv
    return True


FORGE = setup()


def headers() -> dict:
    """file name -> {"tensors": {name: [dtype, shape]}, "metadata": ..., "version": ...} (tools/probe_headers.py)."""
    with gzip.open(FIXTURES / "ltx_headers.json.gz", "rt", encoding="utf-8") as f:
        return json.load(f)


def layout_problems(module, tensors: dict, skip_prefixes=()) -> list[str]:
    """Missing, unexpected and mismatched keys of a module against a file's tensors (quantization aside)."""
    import torch

    # U8 holds embedded files (tokenizer, configs) and NVFP4-packed weights; only the weights belong to the module
    file_keys = {k: v[1] for k, v in tensors.items()
                 if not k.endswith(AUX) and not k.startswith(skip_prefixes) and (v[0] != "U8" or k.endswith(".weight"))}
    # quantized: a comfy_quant marker, or a packed integer / fp8 weight (newer NVFP4 files describe the format in
    # their metadata instead of per layer)
    quantized = {k[: -len(".comfy_quant")] for k in tensors if k.endswith(".comfy_quant")}
    quantized |= {k[: -len(".weight")] for k, v in tensors.items() if k.endswith(".weight") and v[0] in ("U8", "I8", "F8_E4M3", "F8_E5M2")}
    mine = {k: tuple(v.shape) for k, v in module.state_dict().items() if isinstance(v, torch.Tensor)}
    problems = [f"missing {k}" for k in sorted(set(mine) - set(file_keys))]
    problems += [f"unexpected {k}" for k in sorted(set(file_keys) - set(mine))]
    for k in sorted(set(mine) & set(file_keys)):
        shape = tuple(file_keys[k])
        # quantized weights may be packed along the input axis; the output axis always matches
        same = shape[0] == mine[k][0] if k.rsplit(".", 1)[0] in quantized and k.endswith(".weight") else shape == mine[k]
        if not same:
            problems.append(f"shape {k}: {mine[k]} != {shape}")
    return problems
