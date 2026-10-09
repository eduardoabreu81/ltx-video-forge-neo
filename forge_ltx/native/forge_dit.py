"""The LTX DiT as Forge Neo's sampler calls it: one packed video+audio latent per batch item.

Forge hands the model the packed latent [B, 1, 1, N] of the current generation (engine.prepare), the sigma as the
timestep and the projected text (text_engine) as the context. The text goes through the DiT's embeddings connectors on
every call, which ComfyUI runs once per prompt in extra_conds: Forge rebuilds the batched context every step, and the
connectors cost under 1% of a step.

With first / last frame conditioning (guides.py) the conditioned frames go in mixed with their clean latent and at
their own timesteps, and the prediction is turned back into a velocity for the latent Forge's sampler holds, so that
its x - sigma * v is ComfyUI's masked denoised result. The mix is linear, so CFG over the two halves of a batch gives
the same result as ComfyUI mixing after CFG.

On-the-fly LoRAs on INT8 layers run as low-rank terms next to the INT8 matmul (lora.py, from the H3 extension).
"""

import torch

from ..contracts import raise_pending_error
from . import lora
from .transformer import LTXAVModel


class ForgeLTXAV(LTXAVModel):
    # the current request (streams.Generation), set by the engine for every generation
    generation = None

    def forward(self, x, timestep, context, transformer_options={}, **kwargs):
        if isinstance(x, (list, tuple)):
            return super().forward(x, timestep, context, transformer_options=transformer_options, **kwargs)
        generation = self.generation
        if generation is None:
            raise_pending_error()
            raise RuntimeError("[LTX] the LTX settings were not applied to this generation; check the console for an "
                               "earlier LTX message")
        if x.shape[-1] != generation.shapes.size:
            raise RuntimeError("[LTX] the latent does not belong to the current LTX generation")
        with lora.low_rank(self):
            return self._forward_packed(x, timestep, context, generation, transformer_options)

    def _forward_packed(self, x, timestep, context, generation, transformer_options):
        shapes = generation.shapes
        context = self.preprocess_text_embeds(context.to(x.dtype))
        video, audio = shapes.unpack(x)
        conditioning = generation.conditioning
        if conditioning is None:
            out = super().forward([video, audio], timestep, context, attention_mask=None, frame_rate=generation.fps,
                                  transformer_options=transformer_options)
            return shapes.pack(*out)

        conditioning = conditioning.to(video.device, torch.float32)
        sigma = timestep.float().reshape(-1)
        mixed = conditioning.mix(video.float())
        v, a = super().forward([mixed.to(video.dtype), audio], (conditioning.timesteps(sigma), timestep), context,
                               attention_mask=None, frame_rate=generation.fps, transformer_options=transformer_options,
                               **conditioning.model_kwargs(video.shape[0]))
        s = sigma.reshape(-1, 1, 1, 1, 1).clamp(min=1e-6)
        denoised = conditioning.mix(mixed - s * v.float())
        v = (video.float() - denoised) / s
        return shapes.pack(v.to(video.dtype), a)
