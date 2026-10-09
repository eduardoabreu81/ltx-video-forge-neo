"""Checks that the installed Forge Neo has every piece the native LTX backend builds on, before anything is patched."""

import importlib
import importlib.metadata
import inspect

# what the backend imports or patches, by module; an older Forge Neo misses some of these
REQUIRED = {
    "backend.attention": ("attention_function",),
    "backend.memory_management": ("load_model_gpu", "load_models_gpu", "vae_dtype", "get_total_memory", "intermediate_device"),
    "backend.operations": ("using_forge_operations",),
    "backend.quant_ops": ("ck",),
    "backend.state_dict": ("load_state_dict",),
    "backend.loader": ("load_huggingface_component", "replace_state_dict", "possible_models"),
    "backend.nn.krea": ("SingleStreamDiT",),
    "backend.nn.llm.llama": ("Gemma2_2B", "Llama2_", "BaseLlama", "MLP", "attention_function"),
    "backend.text_processing._comfy": ("SDClipModel", "SDTokenizer", "INF"),
    "backend.text_processing.emphasis": ("EmphasisNone", "uses_emphasis"),
    "backend.patcher.clip": ("CLIP",),
    "backend.patcher.unet": ("UnetPatcher",),
    "backend.patcher.vae": ("VAE",),
    "backend.diffusion_engine.base": ("ForgeDiffusionEngine", "ForgeObjects"),
    "huggingface_guess.detection": ("detect_unet_config",),
    "huggingface_guess.model_list": ("models", "BASE", "ModelType"),
    "huggingface_guess.latent": ("LatentFormat",),
    "modules_forge.presets": ("PresetArch", "SAMPLERS", "SCHEDULERS", "STEPS", "CFG", "SHIFT", "register"),
    "modules.sd_schedulers": ("Scheduler", "all_schedulers", "schedulers", "schedulers_map"),
    "transformers.convert_slow_tokenizer": ("GemmaConverter", "generate_merges", "import_protobuf"),
}

# the DiT and the text encoder go through these branches of the loader
LOADER_BRANCHES = ("Krea2Transformer2DModel", "Gemma2Model")

# the fused kernels of the DiT and the Gemma encoders
KITCHEN_KERNELS = ("rms_adaln", "apply_rope", "apply_rope1", "apply_rope_split_half", "apply_rope_split_half1")
MIN_COMFY_KITCHEN = (0, 2, 37)  # Forge Neo 2.30


def _version_tuple(version: str) -> tuple[int, ...]:
    parts = []
    for p in version.split("."):
        digits = "".join(c for c in p if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def check() -> list[str]:
    problems: list[str] = []

    for module_name, names in REQUIRED.items():
        try:
            module = importlib.import_module(module_name)
        except Exception as e:
            problems.append(f"{module_name} could not be imported ({type(e).__name__}: {e})")
            continue
        missing = [n for n in names if not hasattr(module, n)]
        if missing:
            problems.append(f"{module_name} is missing {', '.join(missing)}")

    if not problems:
        from backend import loader
        from backend.quant_ops import ck

        source = inspect.getsource(loader.load_huggingface_component)
        missing = [b for b in LOADER_BRANCHES if f'"{b}"' not in source]
        if missing:
            problems.append(f"backend.loader has no {', '.join(missing)} branch")
        missing = [k for k in KITCHEN_KERNELS if not hasattr(ck, k)]
        if missing:
            problems.append(f"comfy_kitchen has no {', '.join(missing)}")

    try:
        version = importlib.metadata.version("comfy-kitchen")
        if _version_tuple(version) < MIN_COMFY_KITCHEN:
            problems.append(f"comfy-kitchen {version} is older than {'.'.join(map(str, MIN_COMFY_KITCHEN))}")
    except importlib.metadata.PackageNotFoundError:
        problems.append("comfy-kitchen is not installed")

    return problems


def report(problems: list[str]) -> str:
    lines = [
        "[LTX] This Forge Neo is too old for the extension, so nothing was changed and LTX is disabled.",
        "[LTX] Update Forge Neo (git pull on the neo branch) and restart. Details:",
    ]
    lines += [f"[LTX]   - {p}" for p in problems]
    message = "\n".join(lines)
    print(message)
    return message
