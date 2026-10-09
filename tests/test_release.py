"""Dropping a replaced LTX model: its tensors are emptied before Forge detaches it, other models are left alone."""

import types
import unittest

from forge_env import FIXTURES, FORGE

if FORGE:
    import torch
    from backend.operations import using_forge_operations

    from forge_ltx.native import release
    from forge_ltx.native.forge_dit import ForgeLTXAV
    from forge_ltx.native.video_vae import VideoVAE


@unittest.skipUnless(FORGE, "needs a Forge Neo checkout (LTX_FORGE_DIR)")
class ReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dit_config = torch.load(FIXTURES / "ltx_dit_tiny_reference.pt", weights_only=False)["2.3"]["config"]
        cls.vae_config = torch.load(FIXTURES / "ltx_vae_tiny_reference.pt", weights_only=False)["video"]["2.5"]["config"]

    def setUp(self):
        self.types = release.ltx_module_types()
        with using_forge_operations(device=torch.device("cpu"), dtype=torch.float32):
            # Forge wraps the DiT in a KModel; the LTX module sits below the patcher's model
            self.dit = torch.nn.Module()
            self.dit.diffusion_model = ForgeLTXAV(**self.dit_config)
            self.vae = VideoVAE(self.vae_config)
        self.other = torch.nn.Linear(8, 8)
        self.loaded = [types.SimpleNamespace(model=types.SimpleNamespace(model=m, backup={"w": torch.ones(4)}))
                       for m in (self.dit, self.vae, self.other)]

    def sizes(self, module):
        return sum(t.numel() for t in list(module.parameters()) + list(module.buffers()))

    def test_replaced_ltx_is_emptied_and_other_models_kept(self):
        other = self.sizes(self.other)
        self.assertEqual(release.release_replaced(self.loaded, None, self.types), 2)
        self.assertEqual((self.sizes(self.dit), self.sizes(self.vae)), (0, 0))
        self.assertEqual(self.sizes(self.other), other)
        self.assertEqual([bool(m.model.backup) for m in self.loaded], [False, False, True])
        # Forge's detach still works on the emptied model
        self.dit.to("cpu")

    def test_active_ltx_is_left_alone(self):
        before = self.sizes(self.dit)
        self.assertEqual(release.release_replaced(self.loaded, types.SimpleNamespace(is_ltx=True), self.types), 0)
        self.assertEqual(self.sizes(self.dit), before)

    def test_another_active_model_releases_ltx(self):
        self.assertEqual(release.release_replaced(self.loaded, types.SimpleNamespace(), self.types), 2)


@unittest.skipUnless(FORGE, "needs a Forge Neo checkout (LTX_FORGE_DIR)")
class ReloadDecisionTests(unittest.TestCase):
    def test_global_unload_with_ltx_active_reloads_from_disk(self):
        ltx, other = types.SimpleNamespace(is_ltx=True), types.SimpleNamespace()
        self.assertTrue(release.reload_instead_of_unload(True, ltx))
        self.assertFalse(release.reload_instead_of_unload(False, ltx))
        self.assertFalse(release.reload_instead_of_unload(True, other))
        self.assertFalse(release.reload_instead_of_unload(True, None))


if __name__ == "__main__":
    unittest.main()
