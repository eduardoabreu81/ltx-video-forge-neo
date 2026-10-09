"""The text projection between Gemma and the LTX DiT, ported from ComfyUI comfy/text_encoders/lt.py (LTXAVTEModel).

LTX conditions on the hidden states of every Gemma layer at once: the embedding and the 48 layer outputs, the last
one after the final norm, 49 x 3840 values per token. The prompt is left-padded, so only the trailing real tokens are
kept. Two forms, told apart by the weights:

- single linear (LTX-2.0, weights in the checkpoint): each token is rescaled across layers and features to its range,
  then one Linear to 3840; the DiT's caption projections take it from there.
- dual linear (LTX-2.3 in the checkpoint, LTX-2.5 in the text encoder file): each layer's hidden state is
  RMS-normalized, then one Linear for the video stream and one for the audio stream, concatenated.

In both cases the DiT's embeddings connectors run on the result (ComfyUI's "unprocessed_ltxav_embeds").
"""

import math

import torch
import torch.nn as nn

PREFIX = "text_embedding_projection."


def detect(shapes: dict[str, list[int]], prefix: str = PREFIX) -> dict | None:
    """The projection a state dict (or a header: name -> shape) holds, or None. ComfyUI lt.sd_detect."""
    video, audio = shapes.get(f"{prefix}video_aggregate_embed.weight"), shapes.get(f"{prefix}audio_aggregate_embed.weight")
    if video is not None and audio is not None:
        return {"kind": "dual_linear", "in_dim": video[1], "video_dim": video[0], "audio_dim": audio[0],
                "video_bias": f"{prefix}video_aggregate_embed.bias" in shapes, "audio_bias": f"{prefix}audio_aggregate_embed.bias" in shapes}
    single = shapes.get(f"{prefix}aggregate_embed.weight")
    if single is not None:
        return {"kind": "single_linear", "in_dim": single[1], "video_dim": single[0], "video_bias": f"{prefix}aggregate_embed.bias" in shapes}
    return None


class SingleLinearProjection(nn.Module):
    def __init__(self, in_dim: int, video_dim: int, video_bias: bool = False, **_):
        super().__init__()
        self.aggregate_embed = nn.Linear(in_dim, video_dim, bias=video_bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # hidden: [B, layers, T, D] -> [B, T, D, layers]
        x = hidden.movedim(1, -1)
        x = 8.0 * (x - x.mean(dim=(1, 2), keepdim=True)) / (x.amax(dim=(1, 2), keepdim=True) - x.amin(dim=(1, 2), keepdim=True) + 1e-6)
        return self.aggregate_embed(x.reshape(x.shape[0], x.shape[1], -1))


class DualLinearProjection(nn.Module):
    def __init__(self, in_dim: int, video_dim: int, audio_dim: int, video_bias: bool = True, audio_bias: bool = True, **_):
        super().__init__()
        self.audio_aggregate_embed = nn.Linear(in_dim, audio_dim, bias=audio_bias)
        self.video_aggregate_embed = nn.Linear(in_dim, video_dim, bias=video_bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        source_dim = hidden.shape[-1]
        x = hidden.movedim(1, -1)  # [B, T, D, layers]
        x = (x * torch.rsqrt(torch.mean(x**2, dim=2, keepdim=True) + 1e-6)).flatten(start_dim=2)
        video = self.video_aggregate_embed(x * math.sqrt(self.video_aggregate_embed.out_features / source_dim))
        audio = self.audio_aggregate_embed(x * math.sqrt(self.audio_aggregate_embed.out_features / source_dim))
        return torch.cat((video, audio), dim=-1)


def build(spec: dict) -> nn.Module:
    return {"single_linear": SingleLinearProjection, "dual_linear": DualLinearProjection}[spec["kind"]](**spec)


def project(projection: nn.Module, hidden: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    """hidden: [B, layers, T, D] from Gemma with layer="all"; keeps the trailing real tokens of the left-padded prompt."""
    if attention_mask is not None:
        hidden = hidden[:, :, -int(attention_mask.sum().item()):]
    weight = next(projection.parameters())
    return projection(hidden.to(dtype=weight.dtype if weight.is_floating_point() else hidden.dtype))
