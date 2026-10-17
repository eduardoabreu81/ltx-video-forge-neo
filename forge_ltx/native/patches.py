"""Runtime hooks into Forge Neo's loader.

Every hook hands anything that is not LTX-2.x to the original Forge Neo function untouched.
"""

import contextlib
import functools
import logging

import torch
from backend import loader, memory_management
from backend.nn import krea
from backend.nn.llm import llama
from backend.operations import using_forge_operations
from backend.state_dict import load_state_dict
from huggingface_guess import detection, model_list
from transformers.modeling_utils import no_init_weights

from . import filebacked, gemma3, gemma4, model, projection, release, text_engine, vae
from .engine import LTXEngine
from .forge_dit import ForgeLTXAV

logger = logging.getLogger("forge_ltx")

GEMMA_KEYS = ("model.layers.0.post_feedforward_layernorm.weight", "model.layers.47.self_attn.q_norm.weight")
MISSING_TE = ("LTX needs its Gemma text encoder: select gemma_3_12B_it (LTX-2.0 / 2.3) or "
              "gemma4-12b-with-proj-ltx-2.5 (LTX-2.5) under VAE / Text Encoder")
MISSING_PROJECTION = ("The text projection is missing: LTX-2.5 needs gemma4-12b-with-proj-ltx-2.5 (the file with "
                      "\"with-proj\"); LTX-2.0 and 2.3 checkpoints carry it themselves")
MISSING_TOKENIZER = "This Gemma file has no tokenizer inside; use the files LTX publishes (Lightricks or Comfy-Org)"
# Gemma parts LTX never runs, and files embedded in the published text encoders
UNUSED_TE_PREFIXES = gemma4.VISION_PREFIXES + gemma3.VISION_PREFIXES + ("hf_asset__",)
TOKENIZERS = ("tokenizer_json", "spiece_model")

_applied = False


@contextlib.contextmanager
def _swapped(module, name: str, value):
    # the loader imports these classes inside the function, so a temporary swap routes one call to our class
    original = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, original)


def is_gemma(asd: dict) -> bool:
    return all(k in asd for k in GEMMA_KEYS)


def is_gemma4(sd: dict) -> bool:
    # ComfyUI sd.py: Gemma 4 12B drops v_proj on its global layers
    return "model.layers.5.self_attn.v_proj.weight" not in sd


def _replace(sd: dict, prefixes: tuple[str, ...], asd: dict, prefix: str = "") -> None:
    for k in [k for k in sd if k.startswith(prefixes)]:
        del sd[k]
    for k, v in asd.items():
        sd[prefix + k] = v


def _hook_detection() -> None:
    original = detection.detect_unet_config

    def detect_unet_config(state_dict, key_prefix, *args, **kwargs):
        config = model.detect_config(state_dict, key_prefix)
        if config is not None:
            return config
        return original(state_dict, key_prefix, *args, **kwargs)

    detection.detect_unet_config = detect_unet_config


def _hook_replace_state_dict() -> None:
    original = loader.replace_state_dict

    def replace_state_dict(sd: dict, asd: dict, guess, path):
        if not isinstance(guess, model.LTXAV):
            return original(sd, asd, guess, path)
        if is_gemma(asd):
            # ahead of Forge Neo's own Gemma 2 rule, which asserts the Gemma 2 layout
            _replace(sd, (model.TEXT_PREFIX,), asd, model.TEXT_PREFIX)
        elif vae.is_diffusion_vae(asd):
            raise ValueError(vae.DIFFUSION_VAE)
        elif vae.is_video_vae(asd):
            # as the 2.0 / 2.3 checkpoints hold it; LTXAV.process_vae_state_dict takes it from there
            _replace(sd, ("vae.",), {k: v for k, v in asd.items() if not k.startswith(("audio.", "video."))}, "vae.")
            for k in [k for k in asd if k.startswith(("audio.", "video."))]:
                logger.warning(f"[LTX] ignoring {k} in the video VAE file")
        elif vae.is_audio_vae(asd):
            _replace(sd, ("audio_vae.", "vocoder.", "vae.audio."), asd)
        else:
            return original(sd, asd, guess, path)
        return sd

    loader.replace_state_dict = replace_state_dict


def _load_vae(state_dict: dict) -> vae.AutoencoderLTXAV:
    if not isinstance(state_dict, dict):
        raise ValueError(vae.MISSING_VAE)
    video_version, audio_version = vae.state_dict_versions(state_dict)
    unused = {"video." + k for k in vae.video_vae.UNUSED_KEYS}
    state_dict = {k: v for k, v in state_dict.items() if k not in unused}
    with no_init_weights():
        with using_forge_operations(device=memory_management.cpu, dtype=torch.float32, extra_dtype="vae"):
            pair = vae.AutoencoderLTXAV(video_version, audio_version)
    load_state_dict(pair, state_dict, log_name="LTX VAE")
    return pair


def _file_backed(name: str, module, storages: set[int]) -> None:
    # see filebacked.py: weights that return from VRAM become file views again instead of anonymous copies
    try:
        backed, total = filebacked.keep_file_backed(module, storages)
    except Exception as e:
        # the model works without it, with Forge's own copies in system RAM
        logger.warning(f"[LTX] {name}: weights will be copied to system RAM when they leave VRAM: {e}")
        return
    print(f"[LTX] {name}: {backed / 2**30:.1f} of {total / 2**30:.1f} GiB stay file-backed in system RAM")


def _text_encoder_factory(sd: dict):
    """The class Forge Neo's Gemma branch instantiates, for this state dict: our Gemma with the text projection."""
    spec = projection.detect({k: tuple(v.shape) for k, v in sd.items() if k.startswith(projection.PREFIX)})
    if spec is None:
        raise ValueError(MISSING_PROJECTION)
    gemma = gemma4.Gemma4_12B if is_gemma4(sd) else gemma3.Gemma3_12B

    def build(config_dict):
        te = gemma()
        te.text_embedding_projection = projection.build(spec)
        return te

    return build


def _load_text_encoder(guess, component_name, lib_name, repo_path, state_dict, original):
    if not isinstance(state_dict, dict) or not is_gemma(state_dict):
        raise ValueError(MISSING_TE)
    blob = next(((name, state_dict.pop(name)) for name in TOKENIZERS if name in state_dict), None)
    if blob is None:
        raise ValueError(MISSING_TOKENIZER)
    for k in [k for k in state_dict if k.startswith(UNUSED_TE_PREFIXES)]:
        del state_dict[k]
    tokenizer = text_engine.gemma4_tokenizer(blob[1]) if blob[0] == "tokenizer_json" else text_engine.gemma3_tokenizer(blob[1])

    # Forge Neo's Gemma 2 branch: dtype, quantization (Comfy int8, fp8, GGUF) and device handling
    storages = filebacked.file_storages(state_dict)
    with _swapped(llama, "Gemma2_2B", _text_encoder_factory(state_dict)):
        te = original(guess, component_name, lib_name, "Gemma2Model", repo_path, state_dict)
    _file_backed("text encoder", te, storages)
    te.ltx_tokenizer = tokenizer
    print(f"[LTX] text encoder: {'Gemma 4' if isinstance(te, gemma4.Gemma4_12B) else 'Gemma 3'} 12B, "
          f"{type(te.text_embedding_projection).__name__}")
    return te


def _hook_components() -> None:
    original = loader.load_huggingface_component

    # wraps leaves __wrapped__ behind, which inspect.getsource follows: the compat checks of the other extensions
    # that read Forge's loader source still reach it through this hook
    @functools.wraps(original)
    def load_huggingface_component(guess, component_name, lib_name, cls_name, repo_path, state_dict):
        if not isinstance(guess, model.LTXAV):
            return original(guess, component_name, lib_name, cls_name, repo_path, state_dict)
        if cls_name == "AutoencoderLTXAV":
            return _load_vae(state_dict)
        if cls_name == "LTXAVTransformer3DModel":
            # the Krea 2 branch is a generic DiT load: dtype, quantization and device handling
            storages = filebacked.file_storages(state_dict)
            with _swapped(krea, "SingleStreamDiT", ForgeLTXAV):
                dit = original(guess, component_name, lib_name, "Krea2Transformer2DModel", repo_path, state_dict)
            _file_backed("diffusion model", dit, storages)
            print(f"[LTX] LTX-{guess.unet_config.get('version', '?')} diffusion model, {len(dit.transformer_blocks)} blocks")
            return dit
        if cls_name == "LTXGemmaTextEncoder":
            return _load_text_encoder(guess, component_name, lib_name, repo_path, state_dict, original)
        return original(guess, component_name, lib_name, cls_name, repo_path, state_dict)

    loader.load_huggingface_component = load_huggingface_component


def _hook_unload() -> None:
    # see release.py: a replaced LTX model is emptied before Forge detaches it to system RAM
    original = memory_management.unload_all_models
    types = release.ltx_module_types()

    def unload_all_models(*args, **kwargs):
        try:
            from modules import sd_models
            if release.release_replaced(memory_management.current_loaded_models, sd_models.model_data.sd_model, types):
                print("[LTX] released the replaced LTX model without copying it to system RAM")
        except Exception as e:
            logger.warning(f"[LTX] could not release the replaced LTX model early: {e}")
        return original(*args, **kwargs)

    memory_management.unload_all_models = unload_all_models

    from modules import processing, sd_models
    original_manage = processing.manage_model_and_prompt_cache

    def manage_model_and_prompt_cache(p):
        if release.reload_instead_of_unload(processing.need_global_unload, sd_models.model_data.sd_model):
            # an emptied hash makes forge_model_reload discard LTX (see above) and load it again from disk
            print("[LTX] settings changed: reloading LTX from disk instead of moving it to system RAM")
            sd_models.model_data.forge_hash = ""
        return original_manage(p)

    processing.manage_model_and_prompt_cache = manage_model_and_prompt_cache


def apply() -> None:
    global _applied
    if _applied:
        return

    _hook_detection()
    if model.LTXAV not in model_list.models:
        model_list.models.append(model.LTXAV)
    if LTXEngine not in loader.possible_models:
        loader.possible_models = (*loader.possible_models, LTXEngine)
    _hook_replace_state_dict()
    _hook_components()
    try:
        _hook_unload()
    except Exception as e:
        # LTX still works: a replaced model goes through system RAM as any other
        logger.warning(f"[LTX] could not hook model unloading: {e}")

    from . import presets, schedules

    try:
        schedules.register()
    except Exception as e:
        logger.warning(f"[LTX] could not add the LTX schedules: {e}")
    try:
        presets.register()
    except Exception as e:
        # the model still works without the preset: Euler a, LTX Distilled, 8 steps, CFG 1
        logger.warning(f"[LTX] could not add the ltx UI preset: {e}")

    _applied = True
    print(f"[LTX] native backend enabled (torch {torch.__version__})")
