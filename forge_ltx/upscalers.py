"""The latent upscalers two-stage generations use, from Forge Neo's models/latent_upscale_models (ComfyUI's folder name).

"Automatic" takes the spatial x2 upscaler LTX publishes for the checkpoint's version, told apart by the file name
(the three versions publish the same layout).
"""

import re
from pathlib import Path

from .contracts import LTXError

FOLDER = "latent_upscale_models"
AUTOMATIC = "Automatic"
SUFFIXES = (".safetensors",)
DOWNLOADS = {"2.5": "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors (Lightricks/LTX-2.5)",
             "2.3": "ltx-2.3-spatial-upscaler-x2-1.1.safetensors (Lightricks/LTX-2.3)",
             "2.0": "ltx-2-spatial-upscaler-x2-1.0.safetensors (Lightricks/LTX-2)"}


def folder() -> Path:
    from modules import paths
    return Path(paths.models_path) / FOLDER


def names(root: Path | None = None) -> list[str]:
    root = root or folder()
    if not root.is_dir():
        return []
    return sorted(str(p.relative_to(root)).replace("\\", "/") for p in root.rglob("*") if p.suffix.lower() in SUFFIXES)


def version_of(name: str) -> str | None:
    """The LTX version a file name speaks for: ltx-2.5-..., ltx-2.3-..., ltx-2-... (2.0)."""
    match = re.search(r"ltx-?2(?:\.(\d))?[-_]", Path(name).name.lower())
    if match is None:
        return None
    return f"2.{match.group(1)}" if match.group(1) else "2.0"


def is_x2(name: str) -> bool:
    return "x2" in Path(name).name.lower() and "temporal" not in Path(name).name.lower()


def check(path: Path) -> None:
    """A clear error for a file that is not a spatial x2 LTX latent upscaler, read from its header only."""
    from safetensors import safe_open

    from .native import upsampler
    try:
        with safe_open(str(path), framework="pt") as f:
            keys, metadata = set(f.keys()), f.metadata()
        config = upsampler.config_from_metadata(metadata)
    except Exception as error:
        raise LTXError(f"{path.name} is not an LTX latent upscaler: {error}") from None
    if not upsampler.is_upsampler(keys):
        raise LTXError(f"{path.name} is not an LTX latent upscaler.")
    if not upsampler.is_spatial_x2(config):
        raise LTXError(f"{path.name} does not upscale space x2; two stages need a spatial x2 upscaler.")


def resolve(value, version: str, root: Path | None = None) -> Path:
    """The upscaler file for a request: the panel's choice, or for Automatic the x2 one of the checkpoint's version."""
    root = root or folder()
    if value in (None, "", AUTOMATIC):
        found = [n for n in names(root) if is_x2(n) and version_of(n) == version]
        if not found:
            raise LTXError(f"Two stages need the LTX-{version} latent upscaler in models/{FOLDER}: "
                           f"{DOWNLOADS.get(version, DOWNLOADS['2.5'])}.")
        path = root / found[-1]
    else:
        path = Path(value) if Path(value).is_absolute() else root / value
        if not path.is_file():
            raise LTXError(f"The latent upscaler {value} is not in models/{FOLDER}. Refresh the list.")
        other = version_of(path.name)
        if other is not None and other != version:
            print(f"[LTX] {path.name} looks like an LTX-{other} upscaler; the checkpoint is LTX-{version}")
    check(path)
    return path.resolve()
