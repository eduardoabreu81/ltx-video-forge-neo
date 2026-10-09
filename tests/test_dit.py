"""The LTX-2.x DiT on CPU: version and settings from the published headers, every published layout, and the forward
pass against ComfyUI's own LTXAVModel on tiny models (tools/make_dit_reference.py)."""

import unittest

from forge_env import FIXTURES, FORGE, headers, layout_problems

if FORGE:
    import torch
    from backend.operations import using_forge_operations

    from forge_ltx.native import detect
    from forge_ltx.native.transformer import LTXAVModel

DIT_FILES = {
    "ltx-2-19b-distilled.safetensors": "2.0",
    "ltx-2-19b-distilled-fp8.safetensors": "2.0",
    "ltx-2-19b-dev-fp4.safetensors": "2.0",
    "ltx-2.3-22b-distilled-1.1.safetensors": "2.3",
    "ltx-2.3-22b-dev.safetensors": "2.3",
    "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors": "2.5",
    "ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors": "2.5",
    "ltx-2.5-22b-distilled-transformer-bf16.safetensors": "2.5",
    "ltx-2.5-22b-distilled-transformer-nvfp4.safetensors": "2.5",
}


def dit_tensors(entry):
    return {k[len(detect.PREFIX):]: v for k, v in entry["tensors"].items() if k.startswith(detect.PREFIX)}


@unittest.skipUnless(FORGE, "needs a Forge Neo checkout (LTX_FORGE_DIR)")
class LayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.headers = headers()

    def config(self, name):
        entry = self.headers[name]
        return detect.dit_config({k: v[1] for k, v in entry["tensors"].items()}, entry["metadata"])

    def test_version_of_every_published_dit(self):
        for name, expected in DIT_FILES.items():
            with self.subTest(name):
                self.assertTrue(detect.is_ltx(self.headers[name]["tensors"]))
                self.assertEqual(self.config(name)["version"], expected)

    def test_settings_without_metadata_match_the_metadata(self):
        # community repacks often drop the metadata: the presets and the tensors must give the same model
        for name in DIT_FILES:
            with self.subTest(name):
                entry = self.headers[name]
                bare = detect.dit_config({k: v[1] for k, v in entry["tensors"].items()}, None)
                self.assertEqual(bare, self.config(name))

    def test_every_published_dit_loads_into_the_model(self):
        for name in DIT_FILES:
            with self.subTest(name):
                config = self.config(name)
                with using_forge_operations(device=torch.device("meta"), dtype=torch.bfloat16):
                    model = LTXAVModel(**config)
                self.assertEqual(layout_problems(model, dit_tensors(self.headers[name])), [])


@unittest.skipUnless(FORGE, "needs a Forge Neo checkout (LTX_FORGE_DIR)")
class ForwardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = torch.load(FIXTURES / "ltx_dit_tiny_reference.pt", weights_only=False)

    def build(self, variant):
        stored = self.reference[variant]
        with using_forge_operations(device=torch.device("cpu"), dtype=torch.float32):
            model = LTXAVModel(**stored["config"]).eval()
        model.load_state_dict(stored["state_dict"], strict=True)
        return model, stored

    def test_matches_comfyui(self):
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
        from make_dit_reference import scenarios

        for variant in ("2.0", "2.3", "2.5"):
            model, stored = self.build(variant)
            with torch.no_grad():
                context = model.preprocess_text_embeds(stored["text"].clone())
            with self.subTest(variant=variant, part="context"):
                torch.testing.assert_close(context, stored["context"], rtol=1e-4, atol=1e-5)
            for scenario, kwargs in scenarios(torch, variant):
                with self.subTest(variant=variant, scenario=scenario), torch.no_grad():
                    inputs = {k: ([t.clone() for t in v] if k == "x" else v) for k, v in kwargs.items()}
                    out = model(context=stored["context"].clone(), attention_mask=None, frame_rate=24, transformer_options={}, **inputs)
                    expected = stored["outputs"][scenario]
                    self.assertEqual(len(out), len(expected))
                    for mine, theirs in zip(out, expected):
                        torch.testing.assert_close(mine, theirs, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
