"""Generation contracts independent of Forge, CUDA and model libraries."""

import math
from dataclasses import dataclass

# frames follow 8n + 1 (the VAE compresses time 8x after the first frame); sizes are multiples of 32
FRAME_STEP = 8
MIN_FRAMES = 1
MAX_FRAMES = 1001
DEFAULT_FRAMES = 121
STILL_FRAMES = 1
SIZE_STEP = 32
MIN_SIZE = 64
# the frame rate the DiT is told (ComfyUI LTXVConditioning); the templates use 24, LTX-2.5 goes up to 50
DEFAULT_FPS = 24
MIN_FPS = 1
MAX_FPS = 60


class LTXError(RuntimeError):
    """An actionable configuration or generation failure."""


# Forge logs and swallows exceptions raised in script callbacks; a rejected LTX request is kept here and raised
# again from the model, which Forge calls directly
_pending_error = None


def set_pending_error(error):
    global _pending_error
    _pending_error = error


def raise_pending_error():
    if _pending_error is not None:
        raise _pending_error


def align_frames(value) -> int:
    """The nearest valid frame count at or above value."""
    value = int(value)
    return max(MIN_FRAMES, math.ceil((value - MIN_FRAMES) / FRAME_STEP) * FRAME_STEP + MIN_FRAMES)


def integer(value, label) -> int:
    try:
        result = int(value)
        if float(value) != result:
            raise ValueError
        return result
    except (TypeError, ValueError, OverflowError):
        raise LTXError(f"{label} must be an integer.") from None


@dataclass
class GenerationRequest:
    """What LTX adds to Forge's own generation settings: length, frame rate, output kind and audio."""
    width: int = 768
    height: int = 512
    frames: int = DEFAULT_FRAMES
    fps: int = DEFAULT_FPS
    output: str = "Video"
    include_audio: bool = True

    def __post_init__(self):
        if self.output not in ("Video", "Still image"):
            raise LTXError("LTX output must be Video or Still image.")
        self.width = integer(self.width, "Width")
        self.height = integer(self.height, "Height")
        if min(self.width, self.height) < MIN_SIZE or self.width % SIZE_STEP or self.height % SIZE_STEP:
            raise LTXError(f"LTX width and height must be multiples of {SIZE_STEP}, at least {MIN_SIZE}.")
        self.fps = integer(self.fps, "FPS")
        if not MIN_FPS <= self.fps <= MAX_FPS:
            raise LTXError(f"LTX FPS must be between {MIN_FPS} and {MAX_FPS}.")
        if self.output == "Still image":
            self.frames = STILL_FRAMES
            self.include_audio = False
        else:
            self.frames = integer(self.frames, "Frames")
            if not MIN_FRAMES <= self.frames <= MAX_FRAMES or (self.frames - MIN_FRAMES) % FRAME_STEP:
                raise LTXError(f"LTX Frames must follow 8n + 1, from 1 to {MAX_FRAMES} (for example 97 or 121).")

    @property
    def duration(self) -> float:
        return self.frames / self.fps
