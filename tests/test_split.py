"""Forge Neo's own split of a checkpoint and the selected modules into components (backend.loader.split_state_dict),
run for real with the extension's hooks; only reading the files is stood in for, by the published headers.

LTX-2.0 and 2.3 ship one checkpoint with the DiT, both VAEs, the vocoder and the text projection (only Gemma comes
apart); LTX-2.5 ships every part as its own file. Every tensor LTX needs has to reach its component, whatever the
layout, and the version has to come out right.
"""

import unittest
from unittest import mock

from forge_env import FORGE, headers

if FORGE:
    import torch
    from backend import loader

    from forge_ltx.native import model, patches, vae

DTYPES = {"F32": "float32", "F16": "float16", "BF16": "bfloat16", "F8_E4M3": "float8_e4m3fn", "F8_E5M2": "float8_e5m2",
          "I8": "int8", "U8": "uint8", "I32": "int32", "I64": "int64"}

# (checkpoint, modules, version, VAE layouts, text encoder)
CASES = [
    ("ltx-2-19b-distilled.safetensors", ["gemma_3_12B_it.safetensors"], "2.0", ("2.0", "2.0"), "gemma3"),
    ("ltx-2-19b-distilled-fp8.safetensors", ["gemma_3_12B_it_fp8_scaled.safetensors"], "2.0", ("2.0", "2.0"), "gemma3"),
    ("ltx-2.3-22b-distilled-1.1.safetensors", ["gemma_3_12B_it_fp8_scaled.safetensors"], "2.3", ("2.3", "2.3"), "gemma3"),
    ("ltx-2.3-22b-dev.safetensors", ["gemma_3_12B_it.safetensors"], "2.3", ("2.3", "2.3"), "gemma3"),
    ("ltx-2.5-22b-distilled-transformer-bf16.safetensors",
     ["gemma4-12b-with-proj-ltx-2.5-bf16.safetensors", "ltx-2.5-video-vae-conv-bf16.safetensors", "ltx-2.5-audio-vae-bf16.safetensors"],
     "2.5", ("2.3", "2.3"), "gemma4"),
    ("ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
     ["gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors", "ltx-2.5-audio-vae-bf16.safetensors", "ltx-2.5-video-vae-conv-bf16.safetensors"],
     "2.5", ("2.3", "2.3"), "gemma4"),
]


def fake_file(entry):
    return {k: torch.empty(v[1], dtype=getattr(torch, DTYPES.get(v[0], "float32")), device="meta") for k, v in entry["tensors"].items()}


@unittest.skipUnless(FORGE, "needs a Forge Neo checkout (LTX_FORGE_DIR)")
class SplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.headers = headers()
        # the extension's hooks, as patches.apply() installs them, without the UI parts
        cls.saved = (loader.replace_state_dict, patches.detection.detect_unet_config, list(patches.model_list.models))
        patches._hook_detection()
        patches._hook_replace_state_dict()
        if model.LTXAV not in patches.model_list.models:
            patches.model_list.models.append(model.LTXAV)

    @classmethod
    def tearDownClass(cls):
        loader.replace_state_dict, patches.detection.detect_unet_config, models = cls.saved
        patches.model_list.models[:] = models

    def load(self, path, return_metadata=False, **kwargs):
        entry = self.headers[path]
        sd = fake_file(entry)
        return (sd, dict(entry["metadata"] or {})) if return_metadata else sd

    def split(self, checkpoint, modules):
        with mock.patch.object(loader, "load_torch_file", side_effect=self.load):
            return loader.split_state_dict(checkpoint, modules)

    def test_every_layout(self):
        for checkpoint, modules, version, vae_layouts, te in CASES:
            with self.subTest(checkpoint):
                state_dicts, guess = self.split(checkpoint, modules)
                self.assertIsInstance(guess, model.LTXAV)
                self.assertEqual(guess.unet_config["version"], version)

                files = [self.headers[name]["tensors"] for name in [checkpoint, *modules]]
                # Forge adds per-layer quantization entries for the fp8 files; every DiT tensor has to be there
                dit_keys = {k[len("model.diffusion_model."):] for k in files[0] if k.startswith("model.diffusion_model.")}
                self.assertEqual(dit_keys - set(state_dicts["transformer"]), set())
                self.assertFalse(any(k.startswith(("vae", "audio_vae", "vocoder", "text_")) for k in state_dicts["transformer"]))

                pair = state_dicts["vae"]
                self.assertEqual(vae.state_dict_versions(pair), vae_layouts)
                expected_vae = sum(1 for f in files for k in f if k.startswith(("vae.", "audio_vae.", "vocoder.")))
                expected_vae += sum(len(f) for name, f in zip([checkpoint, *modules], files) if "video-vae" in name)
                self.assertEqual(len(pair), expected_vae)

                text = state_dicts["text_encoder"]
                self.assertTrue(patches.is_gemma(text))
                self.assertEqual(patches.is_gemma4(text), te == "gemma4")
                projection = [k for k in text if k.startswith(model.PROJECTION)]
                self.assertTrue(projection, "the text projection must reach the text encoder")
                # LTX-2.0 / 2.3: the projection comes from the checkpoint; LTX-2.5: from the text encoder file
                source = files[0] if version != "2.5" else files[1]
                self.assertEqual(sorted(projection), sorted(k for k in source if k.startswith(model.PROJECTION)))

    def test_checkpoint_without_its_text_encoder(self):
        # the VAE / Text Encoder selection empty: the split works, the component check stops the request first
        state_dicts, guess = self.split("ltx-2.3-22b-distilled-1.1.safetensors", [])
        self.assertEqual(guess.unet_config["version"], "2.3")
        self.assertFalse(patches.is_gemma(state_dicts["text_encoder"]))
        self.assertEqual(vae.state_dict_versions(state_dicts["vae"]), ("2.3", "2.3"))


if __name__ == "__main__":
    unittest.main()
