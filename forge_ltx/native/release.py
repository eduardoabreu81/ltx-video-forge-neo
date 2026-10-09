"""Dropping a replaced LTX model without copying it back to system RAM first.

When another checkpoint is selected, Forge Neo's forge_model_reload clears the active model and calls
unload_all_models, which detaches every loaded model to its offload device: the LTX DiT (about 21 GB with the int8
files) moves from VRAM to system RAM, where the Gemma text encoder (15 to 25 GB) still is, only to be freed right
after. A replaced LTX model is discarded anyway, so its tensors are emptied first and the detach has nothing to copy.

A settings change (Diffusion in Low Bits, the selected modules) makes Forge unload every model while LTX stays
active: the DiT and the text encoder would all land in system RAM. Reloading LTX from disk instead goes through the
release above and costs a model load.
"""

import torch
import torch.nn as nn


def ltx_module_types() -> tuple[type, ...]:
    from .audio_vae import AudioVAE
    from .gemma3 import Gemma3_12B
    from .gemma4 import Gemma4_12B
    from .transformer import LTXAVModel
    from .vae import AutoencoderLTXAV
    from .video_vae import VideoVAE
    return (LTXAVModel, Gemma3_12B, Gemma4_12B, AutoencoderLTXAV, VideoVAE, AudioVAE)


def is_ltx_module(module, types: tuple[type, ...]) -> bool:
    return isinstance(module, nn.Module) and any(isinstance(m, types) for m in module.modules())


def empty_tensors(module: nn.Module) -> None:
    """Replace every parameter and buffer with an empty tensor, freeing the storage wherever it lives."""
    for m in module.modules():
        for name, tensor in list(m._parameters.items()):
            if tensor is not None:
                m._parameters[name] = nn.Parameter(torch.empty(0), requires_grad=False)
        for name, tensor in list(m._buffers.items()):
            if tensor is not None:
                m._buffers[name] = torch.empty(0)


def reload_instead_of_unload(need_global_unload: bool, active_model) -> bool:
    """Whether Forge's global unload with LTX active should become a reload from disk."""
    return bool(need_global_unload) and getattr(active_model, "is_ltx", False)


def release_replaced(loaded_models, active_model, types: tuple[type, ...]) -> int:
    """Empty the LTX models Forge still lists as loaded once LTX is no longer the active model; returns how many."""
    if getattr(active_model, "is_ltx", False):
        return 0
    released = 0
    for loaded in list(loaded_models):
        patcher = getattr(loaded, "model", None)
        module = getattr(patcher, "model", None)
        if not is_ltx_module(module, types):
            continue
        # the original weights a LoRA patch kept would be copied back on detach
        backup = getattr(patcher, "backup", None)
        if isinstance(backup, dict):
            backup.clear()
        empty_tensors(module)
        released += 1
    return released
