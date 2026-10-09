"""The LTX-2.x video and audio VAEs on CPU: every published layout (2.0 and 2.3 inside the checkpoint, 2.5 as
separate files), and encode / decode against ComfyUI's own VAEs on tiny models (tools/make_vae_reference.py)."""

import json
import unittest
from unittest import mock

from forge_env import FIXTURES, FORGE, headers, layout_problems

if FORGE:
    import torch
    from backend.operations import using_forge_operations

    from forge_ltx.native import audio_vae, video_vae

# file -> (prefix of the video VAE tensors, prefix of the audio VAE tensors); None when the file has no such part
FILES = {
    "ltx-2-19b-distilled.safetensors": ("vae.", "audio_vae."),
    "ltx-2.3-22b-distilled-1.1.safetensors": ("vae.", "audio_vae."),
    "ltx-2.5-video-vae-conv-bf16.safetensors": ("", None),
    "ltx-2.5-audio-vae-bf16.safetensors": (None, "audio_vae."),
}


def part(tensors, prefix, rename=None):
    out = {}
    for k, v in tensors.items():
        if k.startswith(prefix):
            out[(rename or "") + k[len(prefix):]] = v
    return out


@unittest.skipUnless(FORGE, "needs a Forge Neo checkout (LTX_FORGE_DIR)")
class LayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.headers = headers()

    def test_video_vae_of_every_version(self):
        for name, (prefix, _) in FILES.items():
            if prefix is None:
                continue
            with self.subTest(name):
                entry = self.headers[name]
                config = json.loads(entry["metadata"]["config"])["vae"]
                with using_forge_operations(device=torch.device("meta"), dtype=torch.bfloat16):
                    model = video_vae.VideoVAE(config)
                tensors = {k: v for k, v in part(entry["tensors"], prefix).items() if k not in video_vae.UNUSED_KEYS}
                self.assertEqual(layout_problems(model, tensors), [])

    def test_audio_vae_and_vocoder_of_every_version(self):
        for name, (_, prefix) in FILES.items():
            if prefix is None:
                continue
            with self.subTest(name):
                entry = self.headers[name]
                with using_forge_operations(device=torch.device("meta"), dtype=torch.bfloat16):
                    model = audio_vae.AudioVAE(*audio_vae.components(entry["metadata"]))
                tensors = {**part(entry["tensors"], prefix, "autoencoder."), **part(entry["tensors"], "vocoder.", "vocoder.")}
                self.assertEqual(layout_problems(model, tensors), [])

    def test_output_rates(self):
        expected = {"ltx-2-19b-distilled.safetensors": 24000, "ltx-2.3-22b-distilled-1.1.safetensors": 48000, "ltx-2.5-audio-vae-bf16.safetensors": 48000}
        for name, rate in expected.items():
            with self.subTest(name):
                with using_forge_operations(device=torch.device("meta"), dtype=torch.bfloat16):
                    model = audio_vae.AudioVAE(*audio_vae.components(self.headers[name]["metadata"]))
                self.assertEqual(model.output_sample_rate, rate)
                self.assertEqual(model.latents_per_second, 25.0)


@unittest.skipUnless(FORGE, "needs a Forge Neo checkout (LTX_FORGE_DIR)")
class ForwardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = torch.load(FIXTURES / "ltx_vae_tiny_reference.pt", weights_only=False)

    @staticmethod
    def load(model, state):
        model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()}, strict=True)
        return model.eval()

    def test_video_matches_comfyui(self):
        for version, stored in self.reference["video"].items():
            with using_forge_operations(device=torch.device("cpu"), dtype=torch.float32):
                model = video_vae.VideoVAE(stored["config"])
            self.load(model, stored["state_dict"])
            for chunking, size in (("whole", None), ("chunked", 1)):
                with self.subTest(version=version, chunking=chunking), torch.no_grad():
                    patch = mock.patch.object(video_vae, "max_chunk_size", lambda device: size) if size else mock.patch.object(video_vae, "max_chunk_size", video_vae.max_chunk_size)
                    with patch:
                        latent = model.encode(stored["pixels"].clone())
                        decoded = model.decode(stored[chunking]["latent"].clone())
                    torch.testing.assert_close(latent, stored[chunking]["latent"], rtol=1e-4, atol=1e-4)
                    torch.testing.assert_close(decoded, stored[chunking]["decoded"], rtol=1e-4, atol=1e-4)

    def test_chunked_video_equals_whole(self):
        # the streaming caches must give the same clip as one pass
        for version, stored in self.reference["video"].items():
            with self.subTest(version=version):
                torch.testing.assert_close(stored["chunked"]["latent"], stored["whole"]["latent"], rtol=1e-4, atol=1e-4)
                torch.testing.assert_close(stored["chunked"]["decoded"], stored["whole"]["decoded"], rtol=1e-4, atol=1e-4)

    def test_audio_matches_comfyui(self):
        for version, stored in self.reference["audio"].items():
            with self.subTest(version=version), torch.no_grad():
                config = stored["config"]
                with using_forge_operations(device=torch.device("cpu"), dtype=torch.float32):
                    model = audio_vae.AudioVAE(config["audio_vae"], config["vocoder"])
                self.load(model, stored["state_dict"])
                self.assertEqual(model.output_sample_rate, stored["output_sample_rate"])
                torch.testing.assert_close(model.decode(stored["latents"].clone()), stored["waveform"], rtol=1e-4, atol=1e-5)
                torch.testing.assert_close(model.encode_mel(stored["mel"].clone()), stored["encoded"], rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
