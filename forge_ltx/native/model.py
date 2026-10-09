"""LTX-2.x registration for huggingface_guess, ported from ComfyUI (latent_formats.LTXAV, supported_models.LTXAV).

Where the parts sit in Forge Neo's state dict, whatever file they came from (see patches.py):
- the DiT under model.diffusion_model.
- the Gemma text encoder, with the text projection, under text_encoders.gemma.
- the video VAE under vae.video., the audio VAE under vae.audio.autoencoder. and its vocoder under vae.audio.vocoder.
LTX-2.0 and 2.3 checkpoints carry the VAEs (vae., audio_vae., vocoder.) and the text projection at the top level;
LTX-2.5 ships each part as its own file.
"""

import json
import os

import torch
from huggingface_guess.latent import LatentFormat
from huggingface_guess.model_list import BASE, ModelType

from . import detect

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "huggingface", "LTX-2")
TEXT_ENCODER = "gemma"
TEXT_PREFIX = f"text_encoders.{TEXT_ENCODER}."
PROJECTION = "text_embedding_projection."

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "latent_rgb.json"), encoding="utf-8") as f:
    _RGB = json.load(f)


class LTXAVLatent(LatentFormat):
    # the video stream; normalization lives inside the VAE, so latents pass through unchanged
    def __init__(self):
        self.latent_channels = 128
        self.scale_factor = 1.0
        self.latent_rgb_factors = _RGB["factors"]
        self.latent_rgb_factors_bias = _RGB["bias"]

    def latent_rgb_factors_reshape(self, sample):
        # Forge's RGB live preview reshapes the latent first: the packed video+audio latent becomes the middle video
        # frame of the generation being sampled
        from modules import shared

        from .streams import preview_frame
        generation = getattr(shared.sd_model, "generation", None)
        return preview_frame(sample, generation.shapes if generation is not None else None)


class LTXAV(BASE):
    # absolute path: the loader joins it with its own huggingface folder, and os.path.join keeps an absolute second part
    huggingface_repo = CONFIG_DIR

    unet_config = {"image_model": "ltxav"}

    # flow matching with the sigma itself as the timestep (the DiT scales it by 1000); no shift: the LTX schedules
    # give their own sigmas
    sampling_settings = {"shift": 1.0, "multiplier": 1.0}

    # ComfyUI's 0.077 under its 0.01 MB/element estimate; Forge Neo's KModel uses 0.02
    memory_usage_factor = 0.0385
    supported_inference_dtypes = [torch.bfloat16, torch.float32]

    unet_extra_config = {}
    latent_format = LTXAVLatent

    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    unet_target = "transformer"

    def clip_target(self, state_dict={}):
        return {TEXT_ENCODER: "text_encoder"}

    def model_type(self, state_dict):
        return ModelType.FLOW

    def process_vae_state_dict(self, state_dict):
        # the 2.0 / 2.3 layout (also what patches.py makes of the 2.5 files) into one VAE pair
        for key in list(state_dict):
            if key.startswith("vae.") and not key.startswith(("vae.video.", "vae.audio.")):
                state_dict["vae.video." + key[len("vae."):]] = state_dict.pop(key)
            elif key.startswith("audio_vae."):
                state_dict["vae.audio.autoencoder." + key[len("audio_vae."):]] = state_dict.pop(key)
            elif key.startswith("vocoder."):
                state_dict["vae.audio.vocoder." + key[len("vocoder."):]] = state_dict.pop(key)
        return state_dict

    def process_clip_state_dict(self, state_dict):
        # LTX-2.0 and 2.3 keep the text projection in the checkpoint; an LTX-2.5 text encoder brings its own
        if not any(k.startswith(TEXT_PREFIX + PROJECTION) for k in state_dict):
            for key in [k for k in state_dict if k.startswith(PROJECTION)]:
                state_dict[TEXT_PREFIX + key] = state_dict.pop(key)
        return super().process_clip_state_dict(state_dict)


def detect_config(state_dict: dict, key_prefix: str) -> dict | None:
    """The LTXAVModel settings for a DiT state dict, or None when it is not LTX-2.x."""
    if not detect.is_ltx(state_dict, key_prefix):
        return None
    shapes = {k: tuple(v.shape) for k, v in state_dict.items() if k.startswith(key_prefix)}
    return {"image_model": "ltxav", **detect.dit_config(shapes, None, prefix=key_prefix)}
