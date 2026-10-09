"""First and last frames on CPU: the conditioning ComfyUI's LTX nodes build (LTXVImgToVideoInplace, LTXVAddGuide), the
DiT as Forge's sampler calls it against ComfyUI's own LTXAVModel under KSamplerX0Inpaint (tools/make_dit_reference.py),
and the engine's choice between an in-place first frame and appended guides."""

import sys
import types
import unittest
from pathlib import Path

from forge_env import FIXTURES, FORGE

if FORGE:
    import torch
    from backend.operations import using_forge_operations

    from forge_ltx.native import guides
    from forge_ltx.native.engine import LTXEngine
    from forge_ltx.native.forge_dit import ForgeLTXAV
    from forge_ltx.native.streams import Generation, StreamShapes

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    from make_dit_reference import F, H, W, scenarios

# pixel frames of a clip with F latent frames, and the pixel frame the reference guide stands for
FRAMES = (F - 1) * 8 + 1
GUIDE_FRAME = 16


def image():
    return torch.rand(1, H * 32, W * 32, 3)


@unittest.skipUnless(FORGE, "needs a Forge Neo checkout (LTX_FORGE_DIR)")
class ConditioningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenarios = dict(scenarios(torch, "2.5"))

    def test_in_place_first_frame(self):
        expected = self.scenarios["i2v"]
        latent = torch.randn(1, 8, 1, H, W)
        cond = guides.build((1, 8, F, H, W), FRAMES, [guides.Keyframe(image(), 0, 1.0, in_place=True)], [latent])
        self.assertEqual(cond.guide_frames, 0)
        self.assertIsNone(cond.keyframe_idxs)
        torch.testing.assert_close(cond.mask, expected["denoise_mask"])
        torch.testing.assert_close(cond.latent[:, :, :1], latent)
        torch.testing.assert_close(cond.timesteps(torch.tensor([0.7])), expected["timestep"][0])

    def test_appended_guide(self):
        expected = self.scenarios["guide"]
        cond = guides.build((1, 8, F, H, W), FRAMES, [guides.Keyframe(image(), GUIDE_FRAME, 0.7)], [torch.randn(1, 8, 1, H, W)])
        self.assertEqual(cond.guide_frames, 1)
        torch.testing.assert_close(cond.keyframe_idxs.float(), expected["keyframe_idxs"])
        torch.testing.assert_close(cond.mask, expected["denoise_mask"])
        torch.testing.assert_close(cond.timesteps(torch.tensor([0.7])), expected["timestep"][0])
        self.assertEqual(cond.guide_entries, expected["guide_attention_entries"])

    def test_last_frame_counts_from_the_end(self):
        cond = guides.build((1, 8, F, H, W), FRAMES, [guides.Keyframe(image(), 0, 0.7), guides.Keyframe(image(), -1, 0.7)],
                            [torch.zeros(1, 8, 1, H, W)] * 2)
        self.assertEqual(cond.guide_frames, 2)
        starts = cond.keyframe_idxs[0, 0, :, 0].unique().tolist()
        self.assertEqual(starts, [0, FRAMES - 1])
        self.assertTrue((cond.mask[:, :, :F] == 1).all() and torch.allclose(cond.mask[:, :, F:], torch.tensor(0.3)))

    def test_compression_keeps_the_picture(self):
        picture = torch.rand(64, 96, 3)
        self.assertIs(guides.compress(picture, 0), picture)
        compressed = guides.compress(picture, 18)
        self.assertEqual(compressed.shape, picture.shape)
        self.assertLess((compressed - picture).abs().mean().item(), 0.25)
        smooth = torch.linspace(0, 1, 96).expand(64, 96)[..., None].expand(64, 96, 3).contiguous()
        self.assertLess((guides.compress(smooth, 18) - smooth).abs().mean().item(), 0.02)


@unittest.skipUnless(FORGE, "needs a Forge Neo checkout (LTX_FORGE_DIR)")
class ForwardTests(unittest.TestCase):
    """Forge's sampler computes x - sigma * v from what the DiT returns; with conditioning that must be ComfyUI's
    masked result: the model run on the mixed latent, its denoised latent held to the clean one where the mask is."""

    @classmethod
    def setUpClass(cls):
        cls.stored = torch.load(FIXTURES / "ltx_dit_tiny_reference.pt", weights_only=False)["2.5"]
        with using_forge_operations(device=torch.device("cpu"), dtype=torch.float32):
            cls.model = ForgeLTXAV(**cls.stored["config"]).eval()
        cls.model.load_state_dict(cls.stored["state_dict"], strict=True)
        cls.scenarios = dict(scenarios(torch, "2.5"))

    def run_forge(self, video, audio, conditioning):
        shapes = StreamShapes(video=tuple(video.shape), audio=tuple(audio.shape))
        self.model.generation = Generation(shapes=shapes, seed=0, fps=24.0, frames=FRAMES, conditioning=conditioning)
        try:
            with torch.no_grad():
                out = self.model(shapes.pack(video, audio), torch.tensor([0.7]), self.stored["text"].clone())
        finally:
            self.model.generation = None
        return shapes.unpack(out)

    def test_guide_frame(self):
        vx, ax = self.scenarios["guide"]["x"]
        guide = vx[:, :, -1:]
        cond = guides.build((1, 8, F, H, W), FRAMES, [guides.Keyframe(image(), GUIDE_FRAME, 0.7)], [guide])
        # the sampler's latent already holds the clean guide, so the mixed input is the reference's own input
        v, a = self.run_forge(vx, ax, cond)
        v_ref, a_ref = self.stored["outputs"]["guide"]
        torch.testing.assert_close(v[:, :, :F], v_ref[:, :, :F], rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(v[:, :, F:], 0.3 * v_ref[:, :, F:], rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(a, a_ref, rtol=1e-4, atol=1e-5)

    def test_in_place_first_frame(self):
        vx, ax = self.scenarios["i2v"]["x"]
        cond = guides.build((1, 8, F, H, W), FRAMES, [guides.Keyframe(image(), 0, 1.0, in_place=True)], [vx[:, :, :1]])
        noisy = vx.clone()
        noisy[:, :, :1] = torch.randn(1, 8, 1, H, W)  # whatever the sampler holds there, the clean frame goes in
        v, a = self.run_forge(noisy, ax, cond)
        v_ref, a_ref = self.stored["outputs"]["i2v"]
        torch.testing.assert_close(v[:, :, 1:], v_ref[:, :, 1:], rtol=1e-4, atol=1e-5)
        # x - sigma * v lands on the clean frame
        torch.testing.assert_close(noisy[:, :, :1] - 0.7 * v[:, :, :1], vx[:, :, :1], rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(a, a_ref, rtol=1e-4, atol=1e-5)

    def test_classifier_free_guidance_batch(self):
        vx, ax = self.scenarios["guide"]["x"]
        cond = guides.build((1, 8, F, H, W), FRAMES, [guides.Keyframe(image(), GUIDE_FRAME, 0.7)], [vx[:, :, -1:]])
        shapes = StreamShapes(video=tuple(vx.shape), audio=tuple(ax.shape))
        self.model.generation = Generation(shapes=shapes, seed=0, fps=24.0, frames=FRAMES, conditioning=cond)
        try:
            with torch.no_grad():
                packed = shapes.pack(vx, ax)
                text = self.stored["text"]
                one = self.model(packed, torch.tensor([0.7]), text.clone())
                two = self.model(packed.repeat(2, 1, 1, 1), torch.tensor([0.7, 0.7]), text.repeat(2, 1, 1))
        finally:
            self.model.generation = None
        torch.testing.assert_close(two[:1], one, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(two[1:], one, rtol=1e-4, atol=1e-5)


@unittest.skipUnless(FORGE, "needs a Forge Neo checkout (LTX_FORGE_DIR)")
class KeyframeChoiceTests(unittest.TestCase):
    def engine(self, first=None, last=None):
        return types.SimpleNamespace(first_frame=first, last_frame=last, keyframe_strength=0.7)

    def test_first_frame_alone_goes_in_place(self):
        (keyframe,) = LTXEngine.keyframes(self.engine(first=image()))
        self.assertEqual((keyframe.frame, keyframe.in_place), (0, True))

    def test_first_and_last_frame_are_guides(self):
        frames = LTXEngine.keyframes(self.engine(first=image(), last=image()))
        self.assertEqual([(k.frame, k.in_place) for k in frames], [(0, False), (-1, False)])

    def test_last_frame_alone_is_a_guide(self):
        (keyframe,) = LTXEngine.keyframes(self.engine(last=image()))
        self.assertEqual((keyframe.frame, keyframe.in_place), (-1, False))

    def test_text_to_video_has_none(self):
        self.assertEqual(LTXEngine.keyframes(self.engine()), [])


if __name__ == "__main__":
    unittest.main()
