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

    def test_tiled_decode_through_forge(self):
        # Never OOM's tiled decode goes through Forge's own VAE object: one tile has to give the plain decode, more
        # tiles the same clip shape (the tiny model's random weights make the seams differ)
        import types

        from forge_ltx.native import engine
        stored = self.reference["video"]["2.5"]
        with using_forge_operations(device=torch.device("cpu"), dtype=torch.float32):
            model = video_vae.VideoVAE(stored["config"])
        self.load(model, stored["state_dict"])
        model.config = types.SimpleNamespace(z_dim=128, latent_channels=128)  # as vae.AutoencoderLTXAV sets it
        vae = engine.video_vae_patcher(model, torch.float32)
        latent = torch.randn(1, 128, 3, 4, 6)
        with torch.inference_mode():  # as decode_first_stage runs
            plain = model.decode(latent.clone())[0].add(1.0).mul(0.5).clamp(0.0, 1.0).movedim(1, 0)
            owner = types.SimpleNamespace(forge_objects=types.SimpleNamespace(vae=vae))
            with mock.patch.object(engine, "DECODE_TILE_OVERLAP", 1):
                whole = engine.LTXEngine._decode_tiled(owner, latent.clone(), 8)
                tiled = engine.LTXEngine._decode_tiled(owner, latent.clone(), 3)
        self.assertEqual(tuple(whole.shape), (17, 3, 128, 192))
        torch.testing.assert_close(whole, plain, rtol=1e-4, atol=1e-4)
        self.assertEqual(tiled.shape, whole.shape)
        self.assertTrue(torch.isfinite(tiled).all())

    def test_decode_plan_fits_the_card(self):
        # the whole clip when the card holds it, else the largest tile that fits; 1280x704x121 needs about 8.5 GiB
        from forge_ltx.native import engine
        with using_forge_operations(device=torch.device("meta"), dtype=torch.bfloat16):
            model = video_vae.VideoVAE(self.reference["video"]["2.5"]["config"])
        shape, gib = (1, 128, 16, 22, 40), 1024**3
        self.assertIsNone(engine.decode_plan(model, shape, torch.bfloat16, 44 * gib))   # A40 / 48 GB
        self.assertIsNone(engine.decode_plan(model, shape, torch.bfloat16, 13 * gib))   # 16 GB card
        self.assertEqual(engine.decode_plan(model, shape, torch.bfloat16, 9 * gib), 16)
        self.assertEqual(engine.decode_plan(model, shape, torch.bfloat16, 6 * gib), 12)  # 8 GB card
        self.assertEqual(engine.decode_plan(model, shape, torch.bfloat16, 4 * gib), 6)
        for tile in engine.DECODE_TILES:
            self.assertLessEqual(model.decode_memory(engine._tile_shape(shape, tile), torch.bfloat16, None, tiled=True),
                                 model.decode_memory(shape, torch.bfloat16, None))
        # the session 1 measurements stay below their reserves
        measured = {(1, 128, 7, 16, 24): 2.04, (1, 128, 16, 22, 40): 7.55}
        for latent, used in measured.items():
            self.assertGreater(model.decode_memory(latent, torch.bfloat16, None) / gib, used * 1.05)
        self.assertGreater(model.decode_memory((1, 128, 16, 20, 20), torch.bfloat16, None, tiled=True) / gib, 6.46 * 1.05)

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
