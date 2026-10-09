"""Request rules that need neither Forge nor a model: dev or distilled by file name, the keyframe settings and the
console hint about CFG."""

import unittest

from forge_env import FORGE

from forge_ltx.contracts import LTXError, keyframe_defaults
from forge_ltx.models import is_dev_name


class DevNameTests(unittest.TestCase):
    def test_published_names(self):
        dev = ["ltx-2-19b-dev.safetensors", "ltx-2-19b-dev-fp8.safetensors", "ltx-2.3-22b-dev.safetensors",
               "ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors", "ltx-2-3-22b-dev-Q4_K_M.gguf"]
        distilled = ["ltx-2-19b-distilled.safetensors", "ltx-2.3-22b-distilled-1.1.safetensors",
                     "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors", "LTX-2.3-distilled-Q8_0.gguf",
                     "ltx-2.3-22b-dev-distilled-merge.safetensors", "developer-cut.safetensors"]
        for name in dev:
            self.assertTrue(is_dev_name(name), name)
        for name in distilled:
            self.assertFalse(is_dev_name(name), name)


class KeyframeDefaultTests(unittest.TestCase):
    def test_recipes_by_version(self):
        self.assertEqual(keyframe_defaults("2.0"), (1.0, 33))
        self.assertEqual(keyframe_defaults("2.3"), (0.7, 18))
        self.assertEqual(keyframe_defaults("2.5"), (0.7, 18))
        self.assertEqual(keyframe_defaults(None), (0.7, 18))


@unittest.skipUnless(FORGE, "needs a Forge Neo checkout (LTX_FORGE_DIR)")
class SettingTests(unittest.TestCase):
    def test_keyframe_settings(self):
        from forge_ltx.integration import keyframe_settings
        self.assertEqual(keyframe_settings(None, None), (None, None))
        self.assertEqual(keyframe_settings(0.7, 18.0), (0.7, 18))
        with self.assertRaises(LTXError):
            keyframe_settings(1.5, 18)
        with self.assertRaises(LTXError):
            keyframe_settings(0.7, 101)

    def test_guidance_note(self):
        from forge_ltx.integration import guidance_note
        self.assertIsNone(guidance_note(False, 1.0, "LTX Distilled"))
        self.assertIn("CFG 1", guidance_note(False, 4.0, "LTX Distilled"))
        self.assertIsNone(guidance_note(True, 4.0, "LTX"))
        self.assertIsNone(guidance_note(True, 1.0, "LTX Distilled"))  # dev with the distilled LoRA
        self.assertIn("distilled LoRA", guidance_note(True, 1.0, "LTX"))


if __name__ == "__main__":
    unittest.main()
