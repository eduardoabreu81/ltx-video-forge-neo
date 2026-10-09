"""Builds tests/fixtures/ltx_vae_tiny_reference.pt with ComfyUI's own LTX VAEs: tiny video VAEs in the LTX-2.0 and
LTX-2.5 layouts (encode and decode, whole and in forced small chunks) and tiny audio VAEs with the HiFi-GAN (2.0) and
BigVGAN + bandwidth extension (2.3/2.5) vocoders, as the reference for forge_ltx.native.video_vae / audio_vae.

    D:/ltx-forge-dev/ref-venv/Scripts/python.exe tools/make_vae_reference.py [--comfyui D:/ltx-forge-dev/comfyui-ref]
"""

import argparse
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "ltx_vae_tiny_reference.pt"

_VIDEO_COMMON = {"_class_name": "CausalVideoAutoencoder", "dims": 3, "in_channels": 3, "out_channels": 3, "latent_channels": 128,
                 "norm_layer": "pixel_norm", "patch_size": 4, "latent_log_var": "uniform", "causal_decoder": False,
                 "timestep_conditioning": False, "encoder_base_channels": 4, "decoder_base_channels": 4}
VIDEO = {
    # LTX-2.0: residual decoder upsampling, no padding mode named (decoder falls back to reflect)
    "2.0": {**_VIDEO_COMMON,
            "encoder_blocks": [["res_x", {"num_layers": 1}], ["compress_space_res", {"multiplier": 2}], ["res_x", {"num_layers": 1}],
                               ["compress_time_res", {"multiplier": 2}], ["res_x", {"num_layers": 1}], ["compress_all_res", {"multiplier": 2}],
                               ["res_x", {"num_layers": 1}], ["compress_all_res", {"multiplier": 2}], ["res_x", {"num_layers": 1}]],
            "decoder_blocks": [["res_x", {"num_layers": 1, "inject_noise": False}], ["compress_all", {"residual": True, "multiplier": 2}],
                               ["res_x", {"num_layers": 1, "inject_noise": False}], ["compress_all", {"residual": True, "multiplier": 2}],
                               ["res_x", {"num_layers": 1, "inject_noise": False}], ["compress_all", {"residual": True, "multiplier": 2}],
                               ["res_x", {"num_layers": 1, "inject_noise": False}]]},
    # LTX-2.3 / 2.5 conv VAE
    "2.5": {**_VIDEO_COMMON, "spatial_padding_mode": "zeros",
            "encoder_blocks": [["res_x", {"num_layers": 1}], ["compress_space_res", {"multiplier": 2}], ["res_x", {"num_layers": 1}],
                               ["compress_time_res", {"multiplier": 2}], ["res_x", {"num_layers": 1}], ["compress_all_res", {"multiplier": 2}],
                               ["res_x", {"num_layers": 1}], ["compress_all_res", {"multiplier": 1}], ["res_x", {"num_layers": 1}]],
            "decoder_blocks": [["res_x", {"num_layers": 1}], ["compress_space", {"multiplier": 2}], ["res_x", {"num_layers": 1}],
                               ["compress_time", {"multiplier": 2}], ["res_x", {"num_layers": 1}], ["compress_all", {"multiplier": 1}],
                               ["res_x", {"num_layers": 1}], ["compress_all", {"multiplier": 2}], ["res_x", {"num_layers": 1}]]},
}

_AUDIO_AE = {"model": {"params": {"ddconfig": {"double_z": True, "mel_bins": 64, "z_channels": 8, "resolution": 256, "downsample_time": False,
                                               "in_channels": 2, "out_ch": 2, "ch": 8, "ch_mult": [1, 2, 4], "num_res_blocks": 1,
                                               "attn_resolutions": [], "dropout": 0.0, "mid_block_add_attention": False, "norm_type": "pixel",
                                               "causality_axis": "height"}, "sampling_rate": 16000}},
             "preprocessing": {"stft": {"filter_length": 1024, "hop_length": 160}}}
AUDIO = {
    "2.0": {"audio_vae": _AUDIO_AE, "vocoder": {"resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5]], "stereo": True, "upsample_rates": [6, 5, 2, 2, 2],
                                                "resblock_kernel_sizes": [3, 7], "upsample_kernel_sizes": [16, 15, 8, 4, 4], "resblock": "1",
                                                "upsample_initial_channel": 32}},
    "2.5": {"audio_vae": _AUDIO_AE, "vocoder": {
        "vocoder": {"upsample_initial_channel": 64, "resblock": "AMP1", "upsample_rates": [5, 2, 2, 2, 2, 2], "resblock_kernel_sizes": [3, 7],
                    "upsample_kernel_sizes": [11, 4, 4, 4, 4, 4], "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5]], "stereo": True,
                    "use_tanh_at_final": False, "activation": "snakebeta", "use_bias_at_final": False},
        "bwe": {"upsample_initial_channel": 32, "resblock": "AMP1", "upsample_rates": [6, 5, 2, 2, 2], "resblock_kernel_sizes": [3, 7],
                "upsample_kernel_sizes": [12, 11, 4, 4, 4], "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5]], "stereo": True,
                "use_tanh_at_final": False, "activation": "snakebeta", "use_bias_at_final": False, "apply_final_activation": False,
                "input_sampling_rate": 16000, "output_sampling_rate": 48000, "hop_length": 80, "n_fft": 512, "win_size": 512, "num_mels": 64}}},
}


def randomize(torch, module):
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for name, tensor in module.state_dict(keep_vars=True).items():  # persistent state only
            if "filter" in name:  # the anti-aliasing filters are fixed by construction; only rounded like the rest
                tensor.copy_(tensor.half().float())
                continue
            values = torch.randn(tensor.shape, generator=g) * 0.2
            if "std-of-means" in name:
                values = values.abs() + 0.5
            elif name.endswith((".alpha", ".beta")):
                values = values * 0.5
            tensor.copy_(values.half().float())  # exact in fp16, so the fixture stores half the bytes


def half_state(module):
    return {k: v.half() if v.is_floating_point() else v.clone() for k, v in module.state_dict().items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfyui", default="D:/ltx-forge-dev/comfyui-ref")
    args = parser.parse_args()
    sys.path.insert(0, args.comfyui)
    sys.argv = [sys.argv[0], "--cpu"]
    import comfy.options

    comfy.options.enable_args_parsing()
    import comfy.ldm.lightricks.vae.causal_video_autoencoder as cva
    import torch
    from comfy.ldm.lightricks.vae.audio_vae import AudioVAE
    from comfy.ldm.mmaudio.vae.distributions import DiagonalGaussianDistribution

    result = {"video": {}, "audio": {}}
    pixels = torch.rand(1, 3, 9, 64, 64, generator=torch.Generator().manual_seed(1)) * 2 - 1
    for name, config in VIDEO.items():
        vae = cva.VideoVAE(config=config).eval()
        randomize(torch, vae)
        out = {}
        original = cva.get_max_chunk_size
        with torch.no_grad():
            for chunking, size in (("whole", None), ("chunked", 1)):
                cva.get_max_chunk_size = original if size is None else (lambda device, size=size: size)
                latent = vae.encode(pixels.clone())
                out[chunking] = {"latent": latent, "decoded": vae.decode(latent.clone())}
        cva.get_max_chunk_size = original
        result["video"][name] = {"config": config, "state_dict": half_state(vae), "pixels": pixels, **out}
        print(f"video {name}: latent {tuple(out['whole']['latent'].shape)}, decoded {tuple(out['whole']['decoded'].shape)}")

    latents = torch.randn(1, 8, 13, 16, generator=torch.Generator().manual_seed(2))
    mel = torch.randn(1, 2, 49, 64, generator=torch.Generator().manual_seed(3))
    for name, config in AUDIO.items():
        vae = AudioVAE(metadata={"config": json.dumps(config)}).eval()
        randomize(torch, vae)
        with torch.no_grad():
            waveform = vae.decode(latents.clone())
            encoded = vae.normalizer.normalize(DiagonalGaussianDistribution(vae.autoencoder.encode(mel.clone())).mode())
        result["audio"][name] = {"config": config, "state_dict": half_state(vae), "latents": latents,
                                 "waveform": waveform, "mel": mel, "encoded": encoded, "output_sample_rate": vae.output_sample_rate}
        print(f"audio {name}: waveform {tuple(waveform.shape)} at {vae.output_sample_rate} Hz, encoded {tuple(encoded.shape)}")

    torch.save(result, OUT)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
