"""Atomic local exports using Forge's FFmpeg or imageio's bundled executable.

Video frames go to FFmpeg as raw RGB through its standard input: writing them as PNG files first took longer than the
whole sampling of a 5-second clip.
"""

import os
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np
from PIL.PngImagePlugin import PngInfo

from .contracts import LTXError


class ExportCancelled(LTXError):
    pass


def find_ffmpeg(configured=""):
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_file():
            return str(candidate.resolve())
        raise LTXError("The LTX FFmpeg executable does not exist. Correct it in Settings.")
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        raise LTXError("FFmpeg is missing. Install FFmpeg or imageio-ffmpeg, or set the LTX FFmpeg executable in Settings.") from None


def _write_wave(audio, path, samples, sample_rate):
    if hasattr(audio, "detach"):
        audio = audio.detach().float().cpu().numpy()
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[None, :]
    if audio.ndim != 2 or audio.shape[0] not in (1, 2) or not audio.shape[1]:
        raise LTXError("LTX audio must have shape [channels, samples] with one or two channels.")
    if not np.isfinite(audio).all():
        raise LTXError("LTX audio must contain finite samples.")
    aligned = np.zeros((audio.shape[0], samples), dtype=np.float32)
    count = min(samples, audio.shape[1])
    aligned[:, :count] = audio[:, :count]
    pcm = (np.clip(aligned, -1, 1).T * 32767).astype("<i2")
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(pcm.shape[1])
        writer.setsampwidth(2)
        writer.setframerate(int(sample_rate))
        writer.writeframes(pcm.tobytes())


def frame_array(frames) -> np.ndarray:
    """[T, H, W, 3] uint8 from PIL images or such an array; H and W even, as yuv420p needs."""
    if isinstance(frames, np.ndarray):
        array = frames
    else:
        sizes = {frame.size for frame in frames}
        if len(sizes) > 1:
            raise LTXError("LTX frames must have equal, even dimensions.")
        array = np.stack([np.asarray(frame.convert("RGB")) for frame in frames]) if frames else np.zeros((0, 2, 2, 3), np.uint8)
    if array.ndim != 4 or array.shape[-1] != 3 or array.dtype != np.uint8:
        raise LTXError("LTX frames must be RGB images.")
    if not len(array):
        raise LTXError("The LTX backend returned no frames.")
    if array.shape[1] % 2 or array.shape[2] % 2:
        raise LTXError("LTX frames must have equal, even dimensions.")
    return np.ascontiguousarray(array)


def export_video(frames, audio, output, *, fps, sample_rate=0, ffmpeg="", infotext="", cancelled=lambda: False):
    """frames: [T, H, W, 3] uint8 or PIL images; audio: [channels, samples] at sample_rate, cut or padded to the clip's
    length, or None."""
    array = frame_array(frames)
    count, height, width = array.shape[:3]
    target = Path(output).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    executable = find_ffmpeg(ffmpeg)
    with tempfile.TemporaryDirectory(prefix=".ltx-export-", dir=target.parent) as scratch:
        scratch = Path(scratch)
        encoded = scratch / "result.mp4"
        command = [executable, "-hide_banner", "-loglevel", "error", "-y",
                   "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-framerate", str(fps), "-i", "pipe:0"]
        if audio is not None:
            _write_wave(audio, scratch / "audio.wav", round(count / fps * sample_rate), sample_rate)
            command += ["-i", str(scratch / "audio.wav"), "-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-b:a", "192k"]
        else:
            command += ["-an"]
        command += ["-frames:v", str(count), "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", "-metadata", "comment=" + infotext, str(encoded)]
        errors = scratch / "ffmpeg.log"
        try:
            with open(errors, "wb") as log:
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log,
                                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                try:
                    for frame in array:
                        if cancelled():
                            raise ExportCancelled("LTX export cancelled.")
                        process.stdin.write(frame.tobytes())
                    process.stdin.close()
                    process.wait(timeout=600)
                except BaseException:
                    process.kill()
                    process.wait()
                    try:
                        process.stdin.close()
                    except OSError:
                        pass
                    raise
        except BrokenPipeError:
            pass  # FFmpeg stopped reading: its own error is in the log
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise LTXError(f"LTX video export failed: {exc}") from exc
        if process.returncode or not encoded.is_file():
            message = errors.read_text(encoding="utf-8", errors="replace")[-2000:] if errors.is_file() else ""
            raise LTXError("LTX video export failed: " + message)
        if cancelled():
            raise ExportCancelled("LTX export cancelled.")
        os.replace(encoded, target)
    return str(target)


def export_still(image, output, infotext):
    target = Path(output).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    info = PngInfo()
    info.add_text("parameters", infotext)
    with tempfile.TemporaryDirectory(prefix=".ltx-image-", dir=target.parent) as scratch:
        candidate = Path(scratch) / "result.png"
        image.save(candidate, format="PNG", pnginfo=info)
        os.replace(candidate, target)
    return str(target)
