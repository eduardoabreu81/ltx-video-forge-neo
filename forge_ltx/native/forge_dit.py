"""The LTX DiT as Forge Neo's sampler calls it: one packed video+audio latent per batch item.

Forge hands the model the packed latent [B, 1, 1, N] of the current generation (engine.prepare), the sigma as the
timestep and the projected text (text_engine) as the context. The text goes through the DiT's embeddings connectors on
every call, which ComfyUI runs once per prompt in extra_conds: Forge rebuilds the batched context every step, and the
connectors cost under 1% of a step.
"""

from ..contracts import raise_pending_error
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
        shapes = generation.shapes
        if x.shape[-1] != shapes.size:
            raise RuntimeError("[LTX] the latent does not belong to the current LTX generation")

        context = self.preprocess_text_embeds(context.to(x.dtype))
        video, audio = shapes.unpack(x)
        out = super().forward([video, audio], timestep, context, attention_mask=None, frame_rate=generation.fps,
                              transformer_options=transformer_options)
        return shapes.pack(*out)
