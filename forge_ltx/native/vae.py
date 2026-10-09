"""The LTX video and audio VAEs as one Forge component, with the settings of each published layout.

The official files carry their VAE settings in metadata, but Forge Neo's loader keeps only the checkpoint's metadata
and drops that of the extra files, so the settings come from vae_configs.json (the metadata of the LTX-2.0 and
LTX-2.3 checkpoints; LTX-2.5's files carry the same as 2.3) and the layout is told apart by its tensors.
"""

import json
import os
from types import SimpleNamespace

from torch import nn

from . import audio_vae, video_vae

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vae_configs.json"), encoding="utf-8") as f:
    CONFIGS = json.load(f)

MISSING_VAE = ("LTX needs its video and audio VAEs: an LTX-2.0 or 2.3 checkpoint has them inside; for LTX-2.5 "
               "select ltx-2.5-video-vae-conv and ltx-2.5-audio-vae under VAE / Text Encoder")
DIFFUSION_VAE = ("The LTX-2.5 diffusion video decoder (ltx-2.5-video-vae-bf16) is not supported yet: select "
                 "ltx-2.5-video-vae-conv-bf16 instead")


def is_video_vae(sd: dict, prefix: str = "") -> bool:
    return f"{prefix}decoder.up_blocks.0.res_blocks.0.conv1.conv.weight" in sd and f"{prefix}encoder.conv_in.conv.weight" in sd


def is_diffusion_vae(sd: dict, prefix: str = "") -> bool:
    return f"{prefix}decoder.conv_in_x_t.weight" in sd


def is_audio_vae(sd: dict, prefix: str = "") -> bool:
    return f"{prefix}audio_vae.decoder.conv_in.conv.weight" in sd and any(k.startswith(f"{prefix}vocoder.") for k in sd)


def video_layout(sd: dict, prefix: str = "") -> str:
    """"2.0" or "2.3" (also LTX-2.5's conv VAE), by the decoder's depth."""
    blocks = {k[len(prefix):].split(".")[2] for k in sd if k.startswith(f"{prefix}decoder.up_blocks.")}
    return "2.0" if len(blocks) == 7 else "2.3"


def audio_layout(sd: dict, prefix: str = "") -> str:
    """"2.0" (HiFi-GAN) or "2.3" (BigVGAN with bandwidth extension, also LTX-2.5)."""
    return "2.3" if f"{prefix}vocoder.vocoder.conv_pre.weight" in sd else "2.0"


class AutoencoderLTXAV(nn.Module):
    def __init__(self, video_version: str, audio_version: str):
        super().__init__()
        self.video = video_vae.VideoVAE(CONFIGS["video"][video_version])
        audio = CONFIGS["audio"][audio_version]
        self.audio = audio_vae.AudioVAE(audio["audio_vae"], audio["vocoder"])
        # what Forge's VAE patcher reads from a model
        self.video.config = SimpleNamespace(z_dim=128, latent_channels=128)
        self.audio.config = SimpleNamespace(z_dim=8, latent_channels=8)
        self.versions = (video_version, audio_version)


def state_dict_versions(sd: dict) -> tuple[str, str]:
    """The layouts of a pair state dict (keys video.* and audio.autoencoder.* / audio.vocoder.*)."""
    if not any(k.startswith("video.") for k in sd) or not any(k.startswith("audio.") for k in sd):
        raise ValueError(MISSING_VAE)
    if is_diffusion_vae(sd, "video."):
        raise ValueError(DIFFUSION_VAE)
    audio = "2.3" if "audio.vocoder.vocoder.conv_pre.weight" in sd else "2.0"
    return video_layout(sd, "video."), audio
