"""The LTX-2.x audio-video DiT (LTXAV), ported from ComfyUI comfy/ldm/lightricks (LTXBaseModel, LTXVModel and
LTXAVModel flattened into one class, symmetric_patchifier.py). See dit.py for the layers.

The model takes the video latent [B, 128, F, H, W] and the audio latent [B, 8, T, 16] as a list, the sigma (or a
(video, audio) pair of per-token timesteps) and the context from preprocess_text_embeds. Keyframe guides are video
tokens appended after the noisy ones with their own pixel coordinates (keyframe_idxs); a denoise mask below zero
drops tokens from the sequence. Left out for now: ID-LoRA reference audio and block-replace patches.
"""

import math

import torch
import torch.nn as nn
from einops import rearrange

from .dit import (
    ADALN_BASE_PARAMS_COUNT,
    ADALN_CROSS_ATTN_PARAMS_COUNT,
    AdaLayerNormSingle,
    BasicAVTransformerBlock,
    CompressedTimestep,
    Embeddings1DConnector,
    GuideAttentionMask,
    NormSingleLinearTextProjection,
    PixArtAlphaTextProjection,
    compute_prompt_timestep,
    freq_grid_float32,
    freq_grid_float64,
    freqs_cis_matrix,
    generate_freqs,
)

# region patchifiers


def latent_to_pixel_coords(latent_coords, scale_factors, causal_fix=False):
    """Latent corner coordinates [B, 3, N(, 2)] to pixel coordinates; causal_fix gives the first frame scale 1."""
    shape = [1] * latent_coords.ndim
    shape[1] = -1
    pixel = latent_coords * torch.tensor(scale_factors, device=latent_coords.device).view(*shape)
    if causal_fix:
        pixel[:, 0, ...] = (pixel[:, 0, ...] + 1 - scale_factors[0]).clamp(min=0)
    return pixel


class SymmetricPatchifier:
    def __init__(self, patch_size: int = 1, start_end: bool = True):
        self.patch_size = (1, patch_size, patch_size)
        self.start_end = start_end

    def latent_coords(self, frames, height, width, batch, device):
        grid = torch.meshgrid(torch.arange(0, frames, self.patch_size[0], device=device),
                              torch.arange(0, height, self.patch_size[1], device=device),
                              torch.arange(0, width, self.patch_size[2], device=device), indexing="ij")
        start = torch.stack(grid, dim=0)
        end = start + torch.tensor(self.patch_size, device=device, dtype=start.dtype)[:, None, None, None]
        start = rearrange(start.unsqueeze(0).repeat(batch, 1, 1, 1, 1), "b c f h w -> b c (f h w)")
        if not self.start_end:
            return start
        end = rearrange(end.unsqueeze(0).repeat(batch, 1, 1, 1, 1), "b c f h w -> b c (f h w)")
        return torch.stack((start, end), dim=-1)

    def patchify(self, latents):
        b, _, f, h, w = latents.shape
        coords = self.latent_coords(f, h, w, b, latents.device)
        p1, p2, p3 = self.patch_size
        return rearrange(latents, "b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)", p1=p1, p2=p2, p3=p3), coords

    def unpatchify(self, latents, output_height, output_width, output_num_frames, out_channels):
        return rearrange(latents, "b (f h w) (c p q) -> b c f (h p) (w q)", f=output_num_frames,
                         h=output_height // self.patch_size[1], w=output_width // self.patch_size[2],
                         p=self.patch_size[1], q=self.patch_size[2])


class AudioPatchifier:
    """Audio latents [B, C, T, F] to tokens [B, T, C*F], with each token's start and end time in seconds."""

    def __init__(self, sample_rate=16000, hop_length=160, audio_latent_downsample_factor=4, is_causal=True, start_end=True, shift=0):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.audio_latent_downsample_factor = audio_latent_downsample_factor
        self.is_causal = is_causal
        self.start_end = start_end
        self.shift = shift

    def latent_time_in_sec(self, start, end, dtype, device):
        mel = torch.arange(start, end, dtype=dtype, device=device) * self.audio_latent_downsample_factor
        if self.is_causal:
            mel = (mel + 1 - self.audio_latent_downsample_factor).clip(min=0)
        return mel * self.hop_length / self.sample_rate

    def patchify(self, audio_latents):
        b, _, t, _ = audio_latents.shape
        tokens = rearrange(audio_latents, "b c t f -> b t (c f)")
        start = self.latent_time_in_sec(self.shift, t + self.shift, torch.float32, audio_latents.device).unsqueeze(0).expand(b, -1).unsqueeze(1)
        if not self.start_end:
            return tokens, start
        end = self.latent_time_in_sec(self.shift + 1, t + self.shift + 1, torch.float32, audio_latents.device).unsqueeze(0).expand(b, -1).unsqueeze(1)
        return tokens, torch.stack([start, end], dim=-1)

    def unpatchify(self, audio_latents, channels, freq):
        return rearrange(audio_latents, "b t (c f) -> b c t f", c=channels, f=freq)


# endregion


class LTXAVModel(nn.Module):
    def __init__(self, in_channels=128, audio_in_channels=128, cross_attention_dim=4096, audio_cross_attention_dim=2048,
                 attention_head_dim=128, audio_attention_head_dim=64, num_attention_heads=32, audio_num_attention_heads=32,
                 caption_channels=3840, num_layers=48, positional_embedding_theta=10000.0,
                 positional_embedding_max_pos=(20, 2048, 2048), audio_positional_embedding_max_pos=(20,),
                 causal_temporal_positioning=False, vae_scale_factors=(8, 32, 32), use_middle_indices_grid=False,
                 timestep_scale_multiplier=1000.0, av_ca_timestep_scale_multiplier=1.0, apply_gated_attention=False,
                 caption_proj_before_connector=False, caption_projection_first_linear=True, cross_attention_adaln=False,
                 ff_bias=True, audio_ff_bias=True, use_prompt_adaln_single=True, use_keyframes_abs_pos_embedding=False,
                 rope_type=None, frequencies_precision="float32", connector_attention_head_dim=128,
                 connector_num_attention_heads=30, connector_num_layers=2, connector_apply_gated_attention=False,
                 audio_connector_attention_head_dim=None, audio_connector_num_attention_heads=None, dtype=None, device=None, **kwargs):
        super().__init__()
        self.dtype = dtype
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.cross_attention_dim = cross_attention_dim
        self.audio_cross_attention_dim = audio_cross_attention_dim
        self.num_attention_heads = num_attention_heads
        self.audio_num_attention_heads = audio_num_attention_heads
        self.caption_channels = caption_channels
        self.positional_embedding_max_pos = list(positional_embedding_max_pos)
        self.audio_positional_embedding_max_pos = list(audio_positional_embedding_max_pos)
        self.causal_temporal_positioning = causal_temporal_positioning
        self.vae_scale_factors = tuple(vae_scale_factors)
        self.use_middle_indices_grid = use_middle_indices_grid
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.av_ca_timestep_scale_multiplier = av_ca_timestep_scale_multiplier
        self.caption_proj_before_connector = caption_proj_before_connector
        self.cross_attention_adaln = cross_attention_adaln
        # ComfyUI's defaults differ on purpose: interleaved RoPE in the DiT, split in the connectors
        self.split_rope = rope_type == "split"
        self.positional_embedding_theta = positional_embedding_theta
        self.freq_grid = freq_grid_float64 if frequencies_precision == "float64" else freq_grid_float32

        self.inner_dim = num_attention_heads * attention_head_dim
        self.audio_inner_dim = audio_num_attention_heads * audio_attention_head_dim
        self.num_audio_channels = 8
        self.audio_frequency_bins = 16

        coefficient = ADALN_CROSS_ATTN_PARAMS_COUNT if cross_attention_adaln else ADALN_BASE_PARAMS_COUNT
        prompt_adaln = cross_attention_adaln and use_prompt_adaln_single

        # video
        self.patchify_proj = nn.Linear(in_channels, self.inner_dim, bias=True)
        self.keyframes_abs_pos_embedding = nn.Parameter(torch.zeros(1, self.inner_dim)) if use_keyframes_abs_pos_embedding else None
        self.adaln_single = AdaLayerNormSingle(self.inner_dim, embedding_coefficient=coefficient)
        self.prompt_adaln_single = AdaLayerNormSingle(self.inner_dim, embedding_coefficient=2) if prompt_adaln else None

        # audio
        self.audio_patchify_proj = nn.Linear(audio_in_channels, self.audio_inner_dim, bias=True)
        self.audio_adaln_single = AdaLayerNormSingle(self.audio_inner_dim, embedding_coefficient=coefficient)
        self.audio_prompt_adaln_single = AdaLayerNormSingle(self.audio_inner_dim, embedding_coefficient=2) if prompt_adaln else None

        # audio-video cross-attention modulation
        self.av_ca_video_scale_shift_adaln_single = AdaLayerNormSingle(self.inner_dim, embedding_coefficient=4)
        self.av_ca_a2v_gate_adaln_single = AdaLayerNormSingle(self.inner_dim, embedding_coefficient=1)
        self.av_ca_audio_scale_shift_adaln_single = AdaLayerNormSingle(self.audio_inner_dim, embedding_coefficient=4)
        self.av_ca_v2a_gate_adaln_single = AdaLayerNormSingle(self.audio_inner_dim, embedding_coefficient=1)

        # text: LTX-2.0 projects after the connectors, later versions before them (or not at all)
        if caption_proj_before_connector:
            if caption_projection_first_linear:
                self.caption_projection = NormSingleLinearTextProjection(caption_channels, self.inner_dim)
                self.audio_caption_projection = NormSingleLinearTextProjection(caption_channels, self.audio_inner_dim)
            else:
                self.caption_projection = self.audio_caption_projection = None
        else:
            self.caption_projection = PixArtAlphaTextProjection(caption_channels, self.inner_dim)
            self.audio_caption_projection = PixArtAlphaTextProjection(caption_channels, self.audio_inner_dim)

        connector_split = (rope_type or "split") == "split"
        self.audio_embeddings_connector = Embeddings1DConnector(
            attention_head_dim=audio_connector_attention_head_dim or connector_attention_head_dim,
            num_attention_heads=audio_connector_num_attention_heads or connector_num_attention_heads,
            num_layers=connector_num_layers, split_rope=connector_split, apply_gated_attention=connector_apply_gated_attention)
        self.video_embeddings_connector = Embeddings1DConnector(
            attention_head_dim=connector_attention_head_dim, num_attention_heads=connector_num_attention_heads,
            num_layers=connector_num_layers, split_rope=connector_split, apply_gated_attention=connector_apply_gated_attention)

        self.transformer_blocks = nn.ModuleList([
            BasicAVTransformerBlock(self.inner_dim, self.audio_inner_dim, num_attention_heads, audio_num_attention_heads,
                                    attention_head_dim, audio_attention_head_dim, v_context_dim=cross_attention_dim,
                                    a_context_dim=audio_cross_attention_dim, apply_gated_attention=apply_gated_attention,
                                    cross_attention_adaln=cross_attention_adaln, ff_bias=ff_bias, audio_ff_bias=audio_ff_bias)
            for _ in range(num_layers)
        ])

        # output
        self.scale_shift_table = nn.Parameter(torch.empty(2, self.inner_dim))
        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, self.out_channels)
        self.audio_scale_shift_table = nn.Parameter(torch.empty(2, self.audio_inner_dim))
        self.audio_norm_out = nn.LayerNorm(self.audio_inner_dim, elementwise_affine=False, eps=1e-6)
        self.audio_proj_out = nn.Linear(self.audio_inner_dim, audio_in_channels)

        self.patchifier = SymmetricPatchifier(1, start_end=True)
        self.a_patchifier = AudioPatchifier(start_end=True)

    # region text

    def preprocess_text_embeds(self, context: torch.Tensor) -> torch.Tensor:
        """The text encoder output (projection.project) through the embeddings connectors; run once per prompt."""
        if context.shape[-1] == self.cross_attention_dim + self.audio_cross_attention_dim:
            context_vid = context[:, :, :self.cross_attention_dim]
            context_audio = context[:, :, self.cross_attention_dim:]
        else:
            context_vid = context_audio = context
        if self.caption_proj_before_connector and self.caption_projection is not None:
            context_vid = self.caption_projection(context_vid)
            context_audio = self.audio_caption_projection(context_audio)
        return torch.cat((self.video_embeddings_connector(context_vid)[0], self.audio_embeddings_connector(context_audio)[0]), dim=-1)

    def _prepare_context(self, context, batch_size, vx, ax):
        video_dim, audio_dim = vx.shape[-1], ax.shape[-1]
        v_dim = video_dim if self.caption_proj_before_connector else self.caption_channels
        a_dim = audio_dim if self.caption_proj_before_connector else self.caption_channels
        v_context, a_context = torch.split(context, [v_dim, a_dim], dim=-1)
        if not self.caption_proj_before_connector:
            v_context = self.caption_projection(v_context)
            a_context = self.audio_caption_projection(a_context)
        return v_context.view(batch_size, -1, video_dim), a_context.view(batch_size, -1, audio_dim)

    # endregion

    # region input

    def tokens_per_latent_frame(self, orig_shape):
        return (orig_shape[3] // self.patchifier.patch_size[1]) * (orig_shape[4] // self.patchifier.patch_size[2])

    def keyframes_abs_pos_mask(self, pixel_coords, orig_shape, grid_mask, num_guide_tokens, generated_keyframes):
        """Tokens that encode a single standalone pixel frame (the first latent frame, generated keyframe slots)."""
        temporal_start = pixel_coords[:, 0]
        if temporal_start.ndim == 3:
            temporal_start = temporal_start[..., 0]
        mask = temporal_start == 0
        if num_guide_tokens > 0:
            mask[:, -num_guide_tokens:] = False
        if generated_keyframes is not None:
            tokens_per_frame = self.tokens_per_latent_frame(orig_shape)
            if generated_keyframes["tokens_per_frame"] != tokens_per_frame:
                raise ValueError("[LTX] the generated keyframes were recorded at another latent size")
            first = generated_keyframes["first_latent_frame"] * tokens_per_frame
            slots = torch.zeros(orig_shape[2] * tokens_per_frame, dtype=torch.bool, device=mask.device)
            slots[first:first + generated_keyframes["num_keyframes"] * tokens_per_frame] = True
            if grid_mask is not None:
                slots = slots[grid_mask]
            mask = mask | slots
        return mask

    def _process_video_input(self, vx, keyframe_idxs, denoise_mask, **kwargs):
        args = {"orig_shape": list(vx.shape)}
        vx, latent_coords = self.patchifier.patchify(vx)
        pixel_coords = latent_to_pixel_coords(latent_coords, self.vae_scale_factors, causal_fix=self.causal_temporal_positioning)

        grid_mask = None
        if keyframe_idxs is not None and keyframe_idxs.shape[2] > 0:
            tokens_per_frame = self.tokens_per_latent_frame(args["orig_shape"])
            if keyframe_idxs.shape[2] % tokens_per_frame:
                raise ValueError("[LTX] the keyframe guides were recorded at another latent size")
            args["orig_patchified_shape"] = list(vx.shape)
            grid_mask = ~torch.any(self.patchifier.patchify(denoise_mask)[0] < 0, dim=-1)[0]
            args["grid_mask"] = grid_mask
            vx = vx[:, grid_mask, :]
            pixel_coords = pixel_coords[:, :, grid_mask, ...]
            kf_grid_mask = grid_mask[-keyframe_idxs.shape[2]:]

            # per-guide surviving token counts; guides are appended in order and partition the keyframe grid mask
            entries = kwargs.get("guide_attention_entries")
            if entries:
                if sum(e["pre_filter_count"] for e in entries) != len(kf_grid_mask):
                    raise ValueError("[LTX] the guide token counts do not match the keyframe grid")
                resolved, offset = [], 0
                for entry in entries:
                    count = entry["pre_filter_count"]
                    resolved.append({**entry, "surviving_count": int(kf_grid_mask[offset:offset + count].sum().item())})
                    offset += count
                args["resolved_guide_entries"] = resolved

            keyframe_idxs = keyframe_idxs[..., kf_grid_mask, :]
            if keyframe_idxs.shape[2] > 0:
                pixel_coords[:, :, -keyframe_idxs.shape[2]:, :] = keyframe_idxs
            args["num_guide_tokens"] = keyframe_idxs.shape[2]

        vx = self.patchify_proj(vx)
        if self.keyframes_abs_pos_embedding is not None:
            mask = self.keyframes_abs_pos_mask(pixel_coords, args["orig_shape"], grid_mask, args.get("num_guide_tokens", 0), kwargs.get("generated_keyframes"))
            vx = vx + mask.unsqueeze(-1).to(vx.dtype) * self.keyframes_abs_pos_embedding.to(device=vx.device, dtype=vx.dtype)
        return vx, pixel_coords, args

    def _process_input(self, vx, ax, keyframe_idxs, denoise_mask, **kwargs):
        has_spatial_mask = False
        if denoise_mask is not None:
            for frame in range(denoise_mask.shape[2]):
                values = denoise_mask[0, 0, frame]
                if values.numel() > 0 and values.min() != values.max():
                    has_spatial_mask = True
                    break

        vx, v_pixel_coords, args = self._process_video_input(vx, keyframe_idxs, denoise_mask, **kwargs)
        args["has_spatial_mask"] = has_spatial_mask
        if kwargs.get("ref_audio") is not None:
            raise NotImplementedError("[LTX] reference audio (ID-LoRA) is not supported yet")
        ax, a_latent_coords = self.a_patchifier.patchify(ax)
        ax = self.audio_patchify_proj(ax)
        return vx, ax, v_pixel_coords, a_latent_coords, args

    # endregion

    # region timesteps, positions and masks

    def _prepare_timestep(self, timestep, a_timestep, batch_size, hidden_dtype, grid_mask=None, orig_shape=None, has_spatial_mask=None, **kwargs):
        v_patches_per_frame = None
        if not has_spatial_mask and orig_shape is not None and len(orig_shape) == 5:
            v_patches_per_frame = orig_shape[3] * orig_shape[4]

        timestep_scaled = (timestep[:, grid_mask] if grid_mask is not None else timestep) * self.timestep_scale_multiplier

        # patches of a frame share their timestep when there is no spatial mask: one row per frame
        per_frame = v_patches_per_frame is not None and (timestep.numel() // batch_size) % v_patches_per_frame == 0
        if per_frame:
            rows = timestep.reshape(batch_size, -1, v_patches_per_frame)[:, :, 0]
            if grid_mask is not None:
                rows = rows[:, grid_mask[::v_patches_per_frame]]
            ts_input = rows * self.timestep_scale_multiplier
        else:
            ts_input = timestep_scaled

        v_timestep, v_embedded = self.adaln_single(ts_input.flatten(), hidden_dtype)
        v_timestep = CompressedTimestep(v_timestep.view(batch_size, -1, v_timestep.shape[-1]), v_patches_per_frame, per_frame=per_frame)
        v_embedded = CompressedTimestep(v_embedded.view(batch_size, -1, v_embedded.shape[-1]), v_patches_per_frame, per_frame=per_frame)
        v_prompt_timestep = compute_prompt_timestep(self.prompt_adaln_single, timestep_scaled, batch_size, hidden_dtype)

        a_timestep_scaled = a_timestep * self.timestep_scale_multiplier
        a_flat = a_timestep_scaled.flatten()
        v_flat = timestep_scaled.flatten()
        av_ca_factor = self.av_ca_timestep_scale_multiplier / self.timestep_scale_multiplier

        a_ss, _ = self.av_ca_audio_scale_shift_adaln_single(a_flat, hidden_dtype)
        v_ss, _ = self.av_ca_video_scale_shift_adaln_single(v_flat, hidden_dtype)
        a2v_gate, _ = self.av_ca_a2v_gate_adaln_single(a_timestep_scaled.max().expand_as(v_flat) * av_ca_factor, hidden_dtype)
        v2a_gate, _ = self.av_ca_v2a_gate_adaln_single(timestep_scaled.max().expand_as(a_flat) * av_ca_factor, hidden_dtype)
        cross = [
            a_ss.view(batch_size, -1, a_ss.shape[-1]),
            CompressedTimestep(v_ss.view(batch_size, -1, v_ss.shape[-1]), v_patches_per_frame),
            CompressedTimestep(a2v_gate.view(batch_size, -1, a2v_gate.shape[-1]), v_patches_per_frame),
            v2a_gate.view(batch_size, -1, v2a_gate.shape[-1]),
        ]

        a_out, a_embedded = self.audio_adaln_single(a_flat, hidden_dtype)
        a_out = a_out.view(batch_size, -1, a_out.shape[-1])
        a_embedded = a_embedded.view(batch_size, -1, a_embedded.shape[-1])
        a_prompt_timestep = compute_prompt_timestep(self.audio_prompt_adaln_single, a_timestep_scaled, batch_size, hidden_dtype)
        return (v_timestep, a_out, cross, v_prompt_timestep, a_prompt_timestep), (v_embedded, a_embedded)

    def _freqs_cis(self, indices_grid, dim, out_dtype, max_pos, use_middle_indices_grid, num_attention_heads):
        indices = self.freq_grid(self.positional_embedding_theta, indices_grid.shape[1], dim, indices_grid.device)
        freqs = generate_freqs(indices, indices_grid, max_pos, use_middle_indices_grid)
        pad = dim // 2 - freqs.shape[-1] if self.split_rope else dim % (2 * indices_grid.shape[1])
        return freqs_cis_matrix(freqs, pad, self.split_rope, num_attention_heads, out_dtype)

    def _prepare_positional_embeddings(self, v_pixel_coords, a_latent_coords, frame_rate, x_dtype):
        coords = v_pixel_coords.to(torch.float32)
        coords[:, 0] = coords[:, 0] * (1.0 / frame_rate)
        v_pe = self._freqs_cis(coords, self.inner_dim, x_dtype, self.positional_embedding_max_pos, self.use_middle_indices_grid, self.num_attention_heads)
        a_pe = self._freqs_cis(a_latent_coords, self.audio_inner_dim, x_dtype, self.audio_positional_embedding_max_pos, self.use_middle_indices_grid, self.audio_num_attention_heads)

        # the temporal middle of each token, for the audio-video cross-attention
        max_pos = [max(self.positional_embedding_max_pos[0], self.audio_positional_embedding_max_pos[0])]
        v_cross = self._freqs_cis(coords[:, 0:1, :], self.audio_cross_attention_dim, x_dtype, max_pos, True, self.audio_num_attention_heads)
        a_cross = self._freqs_cis(a_latent_coords[:, 0:1, :], self.audio_cross_attention_dim, x_dtype, max_pos, True, self.audio_num_attention_heads)
        return (v_pe, v_cross), (a_pe, a_cross)

    @staticmethod
    def _prepare_attention_mask(attention_mask, x_dtype):
        if attention_mask is not None and not torch.is_floating_point(attention_mask):
            attention_mask = (attention_mask - 1).to(x_dtype).reshape((attention_mask.shape[0], 1, -1, attention_mask.shape[-1])) * torch.finfo(x_dtype).max
        return attention_mask

    def _guide_self_attention_mask(self, vx, num_guide_tokens=0, resolved_guide_entries=None, **kwargs):
        """Log-space bias that weakens (strength < 1) or strengthens noisy <-> guide attention, per guide."""
        if num_guide_tokens == 0 or not resolved_guide_entries:
            return None
        if not any(e["strength"] != 1.0 or e.get("pixel_mask") is not None for e in resolved_guide_entries):
            return None

        total, device, dtype = vx.shape[1], vx.device, vx.dtype
        weights = []
        for entry in resolved_guide_entries:
            surviving = entry["surviving_count"]
            if surviving == 0:
                continue
            pixel_mask, latent_shape = entry.get("pixel_mask"), entry.get("latent_shape")
            if pixel_mask is not None and latent_shape is not None:
                per_token = self._downsample_mask_to_latent(pixel_mask.to(device=device, dtype=dtype), *latent_shape)[:1]
                weights.append(per_token[:, :min(per_token.shape[1], surviving)] * entry["strength"])
            else:
                weights.append(torch.full((1, surviving), entry["strength"], device=device, dtype=dtype))
        if not weights:
            return None
        tracked = torch.cat(weights, dim=1)
        if (tracked == 1.0).all():
            return None
        return GuideAttentionMask(total, total - num_guide_tokens, tracked.shape[1], tracked)

    @staticmethod
    def _downsample_mask_to_latent(mask, f_lat, h_lat, w_lat):
        """A pixel mask [B, 1, F, H, W] in [0, 1] to per-token latent weights [B, F_lat * H_lat * W_lat]."""
        b, f_pix = mask.shape[0], mask.shape[2]
        spatial = torch.nn.functional.interpolate(rearrange(mask, "b 1 f h w -> (b f) 1 h w"), size=(h_lat, w_lat), mode="area")
        spatial = rearrange(spatial, "(b f) 1 h w -> b 1 f h w", b=b)
        first = spatial[:, :, :1]
        if f_pix > 1 and f_lat > 1:
            t = (f_pix - 1) // (f_lat - 1)
            if t < 1:
                rest = torch.nn.functional.interpolate(rearrange(spatial[:, :, 1:], "b 1 f h w -> (b h w) 1 f"), size=f_lat - 1, mode="nearest")
                rest = rearrange(rest, "(b h w) 1 f -> b 1 f h w", b=b, h=h_lat, w=w_lat)
            else:
                rest = rearrange(spatial[:, :, 1:1 + (f_lat - 1) * t], "b 1 (f t) h w -> b 1 f t h w", t=t).mean(dim=3)
            latent = torch.cat([first, rest], dim=2)
        elif f_lat > 1:
            latent = first.expand(-1, -1, f_lat, -1, -1)
        else:
            latent = first
        return rearrange(latent, "b 1 f h w -> b (f h w)")

    # endregion

    # region output

    def _process_output(self, vx, ax, v_embedded, a_embedded, keyframe_idxs, orig_shape=None, grid_mask=None, orig_patchified_shape=None, **kwargs):
        if isinstance(v_embedded, CompressedTimestep):
            v_embedded = v_embedded.expand()
        shift, scale = (self.scale_shift_table[None, None].to(device=vx.device, dtype=vx.dtype) + v_embedded[:, :, None]).unbind(dim=2)
        vx = self.proj_out(self.norm_out(vx) * (1 + scale) + shift)
        if keyframe_idxs is not None and keyframe_idxs.shape[2] > 0:
            full = torch.zeros(orig_patchified_shape, dtype=vx.dtype, device=vx.device)
            full[:, grid_mask, :] = vx
            vx = full
        vx = self.patchifier.unpatchify(vx, output_height=orig_shape[3], output_width=orig_shape[4], output_num_frames=orig_shape[2],
                                        out_channels=orig_shape[1] // math.prod(self.patchifier.patch_size))

        a_shift, a_scale = (self.audio_scale_shift_table[None, None].to(device=a_embedded.device, dtype=a_embedded.dtype) + a_embedded[:, :, None]).unbind(dim=2)
        ax = self.audio_proj_out(self.audio_norm_out(ax) * (1 + a_scale) + a_shift)
        ax = self.a_patchifier.unpatchify(ax, channels=self.num_audio_channels, freq=self.audio_frequency_bins)
        return vx, ax

    # endregion

    def forward(self, x, timestep, context, attention_mask=None, frame_rate=25, transformer_options={}, keyframe_idxs=None, denoise_mask=None, **kwargs):
        """x: [video] or [video, audio] latents; timestep: the sigma per sample, or (video, audio) per-token timesteps.
        Returns [video, audio], or the video alone when there is no audio."""
        if isinstance(timestep, (tuple, list)) and len(timestep) == 2:
            timestep, a_timestep = timestep
        else:
            a_timestep = timestep

        vx = x[0] if isinstance(x, (list, tuple)) else x
        has_audio = isinstance(x, (list, tuple)) and len(x) > 1
        ax = x[1] if has_audio else torch.zeros((vx.shape[0], self.num_audio_channels, 0, self.audio_frequency_bins), device=vx.device, dtype=vx.dtype)
        batch_size, input_dtype = vx.shape[0], vx.dtype
        options = {**transformer_options, **kwargs}

        vx, ax, v_coords, a_coords, args = self._process_input(vx, ax, keyframe_idxs, denoise_mask, **options)
        options.update(args)
        timesteps, embedded = self._prepare_timestep(timestep, a_timestep, batch_size, input_dtype, **options)
        v_context, a_context = self._prepare_context(context, batch_size, vx, ax)
        attention_mask = self._prepare_attention_mask(attention_mask, input_dtype)
        (v_pe, v_cross_pe), (a_pe, a_cross_pe) = self._prepare_positional_embeddings(v_coords, a_coords, frame_rate, input_dtype)
        self_attention_mask = self._guide_self_attention_mask(vx, **options)

        v_timestep, a_ts, cross, v_prompt_timestep, a_prompt_timestep = timesteps
        stg_blocks = transformer_options.get("stg_self_attn_blocks", ())
        for i, block in enumerate(self.transformer_blocks):
            block_options = {**transformer_options, "stg_skip_self_attn": True} if i in stg_blocks else transformer_options
            vx, ax = block((vx, ax), v_context=v_context, a_context=a_context, attention_mask=attention_mask,
                           v_timestep=v_timestep, a_timestep=a_ts, v_pe=v_pe, a_pe=a_pe, v_cross_pe=v_cross_pe, a_cross_pe=a_cross_pe,
                           v_cross_scale_shift_timestep=cross[1], a_cross_scale_shift_timestep=cross[0],
                           v_cross_gate_timestep=cross[2], a_cross_gate_timestep=cross[3], transformer_options=block_options,
                           self_attention_mask=self_attention_mask, v_prompt_timestep=v_prompt_timestep, a_prompt_timestep=a_prompt_timestep)

        vx, ax = self._process_output(vx, ax, embedded[0], embedded[1], keyframe_idxs, **options)
        return [vx, ax] if ax.numel() > 0 else vx
