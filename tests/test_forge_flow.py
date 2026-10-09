"""The Forge side of LTX on CPU: where each published file lands in Forge Neo's state dict, the component checks the
panel and the request run before loading, the packed latent, the schedules, and the DiT as Forge's sampler calls it.

Files are stood in for by header-only copies of the published headers (tests/fixtures/ltx_headers.json.gz): the
readers never touch tensor data.
"""

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

from forge_env import FIXTURES, FORGE, headers

if FORGE:
    import torch
    from backend.operations import using_forge_operations

    from forge_ltx.contracts import GenerationRequest, LTXError
    from forge_ltx.models import inspect_model, resolve_components
    from forge_ltx.native import detect, model, patches, schedules, vae
    from forge_ltx.native.forge_dit import ForgeLTXAV
    from forge_ltx.native.streams import Generation, stream_shapes

DIT_20 = "ltx-2-19b-distilled.safetensors"
DIT_23 = "ltx-2.3-22b-distilled-1.1.safetensors"
DIT_25 = "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
GEMMA3 = "gemma_3_12B_it_fp8_scaled.safetensors"
GEMMA4 = "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"
VIDEO_25 = "ltx-2.5-video-vae-conv-bf16.safetensors"
DIFFUSION_VAE_25 = "ltx-2.5-video-vae-bf16.safetensors"
AUDIO_25 = "ltx-2.5-audio-vae-bf16.safetensors"


def fake_tensors(entry):
    # name -> a stand-in with the published shape, enough for the key and shape rules
    return {k: torch.empty(v[1], device="meta") for k, v in entry["tensors"].items()}


@unittest.skipUnless(FORGE, "needs a Forge Neo checkout (LTX_FORGE_DIR)")
class StateDictTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.headers = headers()
        cls.guess = model.LTXAV({"image_model": "ltxav"})

    def split(self, sd):
        # Forge Neo split_state_dict, after the extra files: the VAE pair, then the text encoder
        sd = self.guess.process_vae_state_dict(sd)
        vae_sd = {k[len("vae."):]: v for k, v in sd.items() if k.startswith("vae.")}
        clip = self.guess.process_clip_state_dict(sd)
        te = {k[len("gemma."):]: v for k, v in clip.items() if k.startswith("gemma.")}
        return vae_sd, te

    def replace(self, sd, name):
        original = lambda *a: self.fail(f"{name} reached Forge Neo's own rules")  # noqa: E731
        patches.loader.replace_state_dict = original
        try:
            patches._hook_replace_state_dict()
            return patches.loader.replace_state_dict(sd, fake_tensors(self.headers[name]), self.guess, name)
        finally:
            patches.loader.replace_state_dict = original

    def test_checkpoint_with_everything_inside(self):
        for name, version in ((DIT_20, "2.0"), (DIT_23, "2.3")):
            with self.subTest(name):
                sd = self.replace(fake_tensors(self.headers[name]), GEMMA3)
                vae_sd, te = self.split(sd)
                self.assertEqual(vae.state_dict_versions(vae_sd), (version, version))
                self.assertIn("text_embedding_projection.video_aggregate_embed.weight" if version == "2.3"
                              else "text_embedding_projection.aggregate_embed.weight", te)
                self.assertIn("model.layers.47.self_attn.q_norm.weight", te)
                self.assertFalse(patches.is_gemma4(te))

    def test_ltx25_from_separate_files(self):
        sd = fake_tensors(self.headers[DIT_25])
        for name in (GEMMA4, VIDEO_25, AUDIO_25):
            sd = self.replace(sd, name)
        vae_sd, te = self.split(sd)
        self.assertEqual(vae.state_dict_versions(vae_sd), ("2.3", "2.3"))
        self.assertTrue(patches.is_gemma4(te))
        self.assertIn("tokenizer_json", te)
        self.assertIn("text_embedding_projection.audio_aggregate_embed.weight", te)
        self.assertTrue(any(k.startswith("model.diffusion_model.") for k in sd))

    def test_pair_loads_every_layout(self):
        for video, audio in (("2.0", "2.0"), ("2.3", "2.3")):
            with self.subTest(video=video), using_forge_operations(device=torch.device("meta"), dtype=torch.float32):
                pair = vae.AutoencoderLTXAV(video, audio)
            self.assertEqual(pair.video.config.z_dim, 128)

    def test_diffusion_vae_is_refused(self):
        with self.assertRaisesRegex(ValueError, "not supported yet"):
            self.replace(fake_tensors(self.headers[DIT_25]), DIFFUSION_VAE_25)

    def test_detection_of_every_dit(self):
        for name, version in ((DIT_20, "2.0"), (DIT_23, "2.3"), (DIT_25, "2.5")):
            with self.subTest(name):
                config = model.detect_config(fake_tensors(self.headers[name]), detect.PREFIX)
                self.assertEqual((config["image_model"], config["version"]), ("ltxav", version))
        self.assertIsNone(model.detect_config(fake_tensors(self.headers[GEMMA4]), detect.PREFIX))


@unittest.skipUnless(FORGE, "needs a Forge Neo checkout (LTX_FORGE_DIR)")
class ComponentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        cls.paths = {}
        for name, entry in headers().items():
            header = {k: {"dtype": v[0], "shape": v[1], "data_offsets": [0, 0]} for k, v in entry["tensors"].items()}
            header["__metadata__"] = {k: v for k, v in entry["metadata"].items() if k != "license"}
            data = json.dumps(header).encode()
            path = Path(cls.dir.name) / name
            path.write_bytes(struct.pack("<Q", len(data)) + data)
            cls.paths[name] = str(path)

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def resolve(self, dit, *modules):
        return resolve_components(self.paths[dit], [self.paths[m] for m in modules])

    def test_roles_and_variants(self):
        expected = {DIT_20: ("dit", "2.0", True), DIT_25: ("dit", "2.5", False), GEMMA3: ("text_encoder", "gemma3", False),
                    GEMMA4: ("text_encoder", "gemma4", True), VIDEO_25: ("video_vae", "conv", False),
                    DIFFUSION_VAE_25: ("video_vae", "diffusion", False), AUDIO_25: ("audio_vae", "", False)}
        for name, (role, variant, bundled) in expected.items():
            with self.subTest(name):
                item = inspect_model(self.paths[name])
                self.assertEqual((item.role, item.variant, item.bundled), (role, variant, bundled))

    def test_valid_combinations(self):
        self.assertEqual(self.resolve(DIT_20, GEMMA3).version, "2.0")
        self.assertEqual(self.resolve(DIT_23, GEMMA3).version, "2.3")
        self.assertEqual(self.resolve(DIT_25, GEMMA4, VIDEO_25, AUDIO_25).version, "2.5")
        # the 2.3 checkpoint's own VAEs share their layout with LTX-2.5's files
        self.assertEqual(self.resolve(DIT_23, GEMMA3, VIDEO_25, AUDIO_25).version, "2.3")

    def test_clear_errors(self):
        cases = [((DIT_25, GEMMA3, VIDEO_25, AUDIO_25), "Gemma 4"),
                 ((DIT_23, GEMMA4), "Gemma 3"),
                 ((DIT_25, GEMMA4, AUDIO_25), "video VAE"),
                 ((DIT_25, GEMMA4, DIFFUSION_VAE_25, AUDIO_25), "not supported yet"),
                 ((DIT_20, GEMMA3, VIDEO_25), "does not fit LTX-2.0"),
                 ((DIT_20, GEMMA3, AUDIO_25), "does not fit LTX-2.0")]
        for args, message in cases:
            with self.subTest(args=args), self.assertRaisesRegex(LTXError, message):
                self.resolve(*args)


@unittest.skipUnless(FORGE, "needs a Forge Neo checkout (LTX_FORGE_DIR)")
class RequestTests(unittest.TestCase):
    def test_frames_and_sizes(self):
        request = GenerationRequest(width=1280, height=704, frames=121, fps=24)
        self.assertAlmostEqual(request.duration, 121 / 24)
        for kwargs in ({"frames": 120}, {"width": 1000}, {"fps": 0}):
            with self.subTest(kwargs), self.assertRaises(LTXError):
                GenerationRequest(**{"width": 768, "height": 512, **kwargs})
        self.assertEqual(GenerationRequest(output="Still image").frames, 1)

    def test_packed_latent_round_trip(self):
        shapes = stream_shapes(121, 768, 512, 24)
        self.assertEqual(shapes.video, (1, 128, 16, 16, 24))
        self.assertEqual(shapes.audio, (1, 8, 126, 16))
        video, audio = torch.randn(shapes.video), torch.randn(shapes.audio)
        packed = shapes.pack(video, audio)
        self.assertEqual(tuple(packed.shape), (1, 1, 1, shapes.size))
        v, a = shapes.unpack(packed)
        self.assertTrue(torch.equal(v, video) and torch.equal(a, audio))


@unittest.skipUnless(FORGE, "needs a Forge Neo checkout (LTX_FORGE_DIR)")
class ScheduleTests(unittest.TestCase):
    def test_distilled(self):
        self.assertEqual(schedules.distilled_sigmas(8).tolist(), [float(torch.tensor(s)) for s in schedules.DISTILLED])
        for steps in (4, 12, 20):
            sigmas = schedules.distilled_sigmas(steps)
            self.assertEqual(len(sigmas), steps + 1)
            self.assertEqual((float(sigmas[0]), float(sigmas[-1])), (1.0, 0.0))
            self.assertTrue(bool((sigmas[:-1] >= sigmas[1:]).all()))

    def test_dev_ends_at_terminal(self):
        sigmas = schedules.dev_sigmas(30, tokens=16 * 16 * 24)
        self.assertEqual(len(sigmas), 31)
        self.assertAlmostEqual(float(sigmas[-2]), schedules.TERMINAL, places=5)
        self.assertEqual(float(sigmas[-1]), 0.0)


@unittest.skipUnless(FORGE, "needs a Forge Neo checkout (LTX_FORGE_DIR)")
class ForgeDiTTests(unittest.TestCase):
    def test_packed_call_matches_comfyui(self):
        # the tiny DiTs of tools/make_dit_reference.py, called as Forge's sampler calls them: packed latent and the
        # projected text (the connectors run inside)
        reference = torch.load(FIXTURES / "ltx_dit_tiny_reference.pt", weights_only=False)
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
        from make_dit_reference import T_AUDIO, F, H, W, scenarios

        for variant, stored in reference.items():
            with self.subTest(variant=variant), torch.no_grad():
                with using_forge_operations(device=torch.device("cpu"), dtype=torch.float32):
                    dit = ForgeLTXAV(**stored["config"]).eval()
                dit.load_state_dict(stored["state_dict"], strict=True)
                kwargs = dict(scenarios(torch, variant))["t2v"]
                vx, ax = kwargs["x"]
                from forge_ltx.native.streams import StreamShapes
                shapes = StreamShapes(video=tuple(vx.shape), audio=tuple(ax.shape))
                dit.generation = Generation(shapes=shapes, seed=0, fps=24.0, frames=(F - 1) * 8 + 1)
                out = dit(shapes.pack(vx.clone(), ax.clone()), kwargs["timestep"], stored["text"].clone(), transformer_options={})
                expected = shapes.pack(*stored["outputs"]["t2v"])
                torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-5)
                self.assertEqual((H, W, T_AUDIO), (vx.shape[3], vx.shape[4], ax.shape[2]))


if __name__ == "__main__":
    unittest.main()
