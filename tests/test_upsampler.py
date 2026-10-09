"""Two-stage generation on CPU: the latent upscaler against the published headers and ComfyUI's own LatentUpsampler,
the keyframe resize against ComfyUI's common_upscale (tools/make_upsampler_reference.py), the start latent of the second
stage, its sigmas, the size rule and how the upscaler file is found."""

import json
import tempfile
import unittest
from pathlib import Path

from forge_env import FIXTURES, FORGE, headers, layout_problems

from forge_ltx import upscalers
from forge_ltx.contracts import GenerationRequest, LTXError

if FORGE:
    import torch

    from forge_ltx.native import guides, schedules, upsampler

UPSCALERS = {"ltx-2-spatial-upscaler-x2-1.0.safetensors": "2.0", "ltx-2.3-spatial-upscaler-x2-1.1.safetensors": "2.3",
             "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors": "2.5"}


@unittest.skipUnless(FORGE, "needs a Forge Neo checkout (LTX_FORGE_DIR)")
class UpsamplerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = torch.load(FIXTURES / "ltx_upsampler_tiny_reference.pt", weights_only=False)

    def test_published_layouts(self):
        entries = headers()
        for name in UPSCALERS:
            with self.subTest(name):
                entry = entries[name]
                config = upsampler.config_from_metadata(entry["metadata"])
                self.assertTrue(upsampler.is_spatial_x2(config))
                self.assertTrue(upsampler.is_upsampler(entry["tensors"]))
                with torch.device("meta"):
                    model = upsampler.LatentUpsampler.from_config(config)
                self.assertEqual(layout_problems(model, entry["tensors"]), [])

    def test_against_comfyui(self):
        for name, case in self.reference["upsamplers"].items():
            with self.subTest(name):
                model = upsampler.LatentUpsampler.from_config(case["config"]).eval()
                model.load_state_dict({k: v.float() for k, v in case["state_dict"].items()}, strict=True)
                with torch.no_grad():
                    out = model(case["latent"].clone())
                self.assertEqual(out.shape, case["output"].shape)
                torch.testing.assert_close(out, case["output"], rtol=1e-4, atol=1e-5)

    def test_not_x2(self):
        self.assertFalse(upsampler.is_spatial_x2({"spatial_upsample": True, "spatial_scale": 1.5, "rational_resampler": True}))
        self.assertFalse(upsampler.is_spatial_x2({"spatial_upsample": False, "temporal_upsample": True}))
        with self.assertRaises(ValueError):
            upsampler.config_from_metadata({})


@unittest.skipUnless(FORGE, "needs a Forge Neo checkout (LTX_FORGE_DIR)")
class SecondStageTests(unittest.TestCase):
    def test_keyframe_resize_matches_comfyui(self):
        for case in torch.load(FIXTURES / "ltx_upsampler_tiny_reference.pt", weights_only=False)["resizes"]:
            out = guides.fit(case["image"], case["height"], case["width"])
            torch.testing.assert_close(out, case["output"])
        image = torch.rand(1, 8, 8, 3)
        self.assertIs(guides.fit(image, 8, 8), image)

    def test_start_latent_holds_the_conditioned_frames(self):
        f, h, w = 3, 2, 4
        clean = [torch.randn(1, 4, 1, h, w) for _ in range(2)]
        # in place: the first frame of the clip
        conditioning = guides.build((1, 4, f, h, w), 17, [guides.Keyframe(torch.zeros(1, 64, 128, 3), 0, 1.0, in_place=True)], clean[:1])
        video = torch.randn(1, 4, f, h, w)
        start = conditioning.place(video)
        torch.testing.assert_close(start[:, :, :1], clean[0])
        torch.testing.assert_close(start[:, :, 1:], video[:, :, 1:])
        # appended guides: the clip unchanged, the guide frames after it
        keyframes = [guides.Keyframe(torch.zeros(1, 64, 128, 3), 0, 0.7), guides.Keyframe(torch.zeros(1, 64, 128, 3), -1, 0.7)]
        conditioning = guides.build((1, 4, f, h, w), 17, keyframes, clean)
        start = conditioning.place(video)
        self.assertEqual(start.shape[2], f + 2)
        torch.testing.assert_close(start[:, :, :f], video)
        torch.testing.assert_close(start[:, :, f:f + 1], clean[0])
        torch.testing.assert_close(start[:, :, f + 1:], clean[1])

    def test_refine_sigmas(self):
        # ComfyUI's templates: LTX-2.3 and 2.5 refine from 0.85, LTX-2.0 from 0.909375; three steps each
        self.assertEqual(schedules.refine_sigmas("2.5"), (0.85, 0.725, 0.4219, 0.0))
        self.assertEqual(schedules.refine_sigmas("2.3"), (0.85, 0.725, 0.4219, 0.0))
        self.assertEqual(schedules.refine_sigmas("2.0"), (0.909375, 0.725, 0.421875, 0.0))
        sigmas = schedules.refine_override("2.5")(7)
        self.assertEqual(sigmas.tolist(), torch.tensor(schedules.refine_sigmas("2.5")).tolist())


class RequestTests(unittest.TestCase):
    def test_two_stage_sizes(self):
        request = GenerationRequest(width=1280, height=704, two_stage=True)
        self.assertEqual(request.first_stage_size, (640, 352))
        self.assertEqual(GenerationRequest(width=1280, height=736).first_stage_size, (1280, 736))
        with self.assertRaises(LTXError):
            GenerationRequest(width=1280, height=736, two_stage=True)  # 368 is not a multiple of 32
        with self.assertRaises(LTXError):
            GenerationRequest(width=64, height=64, two_stage=True)


class UpscalerFileTests(unittest.TestCase):
    def test_version_from_name(self):
        for name, version in UPSCALERS.items():
            self.assertEqual(upscalers.version_of(name), version)
        self.assertIsNone(upscalers.version_of("RealESRGAN_x2.safetensors"))
        self.assertFalse(upscalers.is_x2("ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors"))

    @unittest.skipUnless(FORGE, "needs a Forge Neo checkout (LTX_FORGE_DIR)")
    def test_resolve(self):
        import torch
        from safetensors.torch import save_file

        config = {"in_channels": 128, "mid_channels": 1024, "spatial_upsample": True, "spatial_scale": 2.0}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(LTXError):
                upscalers.resolve(upscalers.AUTOMATIC, "2.5", root)
            tensors = {upsampler.MARKER: torch.zeros(1)}
            save_file(tensors, str(root / "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"), {"config": json.dumps(config)})
            save_file(tensors, str(root / "ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors"),
                      {"config": json.dumps({**config, "spatial_scale": 1.5, "rational_resampler": True})})
            save_file({"weight": torch.zeros(1)}, str(root / "other.safetensors"))
            self.assertEqual(upscalers.resolve(None, "2.5", root).name, "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors")
            with self.assertRaises(LTXError):
                upscalers.resolve(upscalers.AUTOMATIC, "2.3", root)  # only an x1.5 one for 2.3
            with self.assertRaises(LTXError):
                upscalers.resolve("ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors", "2.3", root)
            with self.assertRaises(LTXError):
                upscalers.resolve("other.safetensors", "2.5", root)
            with self.assertRaises(LTXError):
                upscalers.resolve("missing.safetensors", "2.5", root)
            self.assertEqual(upscalers.names(root), ["ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors",
                                                     "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors", "other.safetensors"])


if __name__ == "__main__":
    unittest.main()
