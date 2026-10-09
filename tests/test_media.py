"""The MP4 export on CPU with the FFmpeg on PATH: raw frames through FFmpeg's standard input, the audio cut or padded to
the clip, cancellation and bad input."""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from forge_ltx.contracts import LTXError
from forge_ltx.media import ExportCancelled, export_video, frame_array

FFMPEG = shutil.which("ffmpeg")


def clip(frames=9, height=64, width=96):
    # flat colours that survive yuv420p, one per frame
    values = np.linspace(20, 230, frames).astype(np.uint8)
    array = np.zeros((frames, height, width, 3), np.uint8)
    array[..., 0] = values[:, None, None]
    array[..., 1] = 128
    array[..., 2] = 255 - values[:, None, None]
    return array


@unittest.skipUnless(FFMPEG, "needs ffmpeg on PATH")
class ExportTests(unittest.TestCase):
    def test_frames_and_audio(self):
        import av

        frames = clip()
        audio = np.sin(np.linspace(0, 200, 8000, dtype=np.float32))[None].repeat(2, 0) * 0.5
        with tempfile.TemporaryDirectory() as tmp:
            out = export_video(frames, audio, Path(tmp) / "a.mp4", fps=24, sample_rate=48000, infotext="test")
            with av.open(out) as container:
                decoded = np.stack([f.to_ndarray(format="rgb24") for f in container.decode(video=0)])
            with av.open(out) as container:
                stream = container.streams.audio[0]
                samples = sum(f.samples for f in container.decode(audio=0))
                rate = stream.sample_rate
            self.assertEqual(list(Path(tmp).iterdir()), [Path(out)])  # no scratch left behind
        self.assertEqual(decoded.shape, frames.shape)
        self.assertLess(np.abs(decoded.astype(int) - frames.astype(int)).mean(), 3.0)
        # padded to the clip's 9 / 24 s
        self.assertAlmostEqual(samples / rate, 9 / 24, delta=0.03)

    def test_pil_frames_and_no_audio(self):
        import av
        from PIL import Image

        frames = [Image.fromarray(f) for f in clip(frames=3)]
        with tempfile.TemporaryDirectory() as tmp:
            out = export_video(frames, None, Path(tmp) / "b.mp4", fps=24)
            with av.open(out) as container:
                self.assertEqual(len(container.streams.audio), 0)
                self.assertEqual(sum(1 for _ in container.decode(video=0)), 3)

    def test_cancelled(self):
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ExportCancelled):
                export_video(clip(), None, Path(tmp) / "c.mp4", fps=24, cancelled=lambda: calls.append(1) or len(calls) > 3)
            self.assertEqual(list(Path(tmp).iterdir()), [])


class FrameTests(unittest.TestCase):
    def test_bad_frames(self):
        with self.assertRaises(LTXError):
            frame_array(np.zeros((2, 63, 96, 3), np.uint8))  # odd height
        with self.assertRaises(LTXError):
            frame_array(np.zeros((2, 64, 96, 3), np.float32))
        with self.assertRaises(LTXError):
            frame_array(np.zeros((0, 64, 96, 3), np.uint8))
        with self.assertRaises(LTXError):
            frame_array([])


if __name__ == "__main__":
    unittest.main()
