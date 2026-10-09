"""Read only safetensors headers; never deserialize a checkpoint to discover it."""

import json
import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .contracts import LTXError

MAX_HEADER_BYTES = 32 * 1024 * 1024
ROLE_LABELS = {"dit": "diffusion model", "text_encoder": "text encoder",
               "video_vae": "video VAE", "audio_vae": "audio VAE"}


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise LTXError(f"Duplicate key in safetensors header: {key}")
        result[key] = value
    return result


@lru_cache(maxsize=32)
def _read_header(path, size, mtime):
    try:
        with open(path, "rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise LTXError("Invalid safetensors header: missing length.")
            length = struct.unpack("<Q", prefix)[0]
            if not 2 <= length <= MAX_HEADER_BYTES or length > size - 8:
                raise LTXError("Invalid or oversized safetensors header.")
            header = json.loads(stream.read(length), object_pairs_hook=_unique_pairs)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LTXError(f"Cannot read safetensors header: {exc}") from exc
    if not isinstance(header, dict):
        raise LTXError("Invalid safetensors header object.")
    for name, tensor in header.items():
        if name == "__metadata__":
            if not isinstance(tensor, dict):
                raise LTXError("Invalid safetensors metadata.")
            continue
        if not isinstance(tensor, dict) or not isinstance(tensor.get("shape"), list):
            raise LTXError(f"Invalid tensor in safetensors header: {name}")
        if not all(isinstance(n, int) and n >= 0 for n in tensor["shape"]):
            raise LTXError(f"Invalid tensor shape in safetensors header: {name}")
    return header


GGUF_MAGIC = b"GGUF"
MAX_GGUF_ITEMS = 1_000_000
MAX_GGUF_STRING = 1024 * 1024
# ggml tensor types (ggml.h); Forge Neo dequantizes F32/F16/BF16, Q4_0..Q8_0 and the K-quants, not the IQ family
GGML_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K",
              11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
              19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32",
              27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16"}
GGUF_PLAIN = {"F32", "F16", "BF16"}
GGUF_FORGE = GGUF_PLAIN | {"Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K"}
# GGUF metadata value types: fixed-size scalars by struct format, 8 = string, 9 = array
GGUF_SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


class _GGUFStream:
    def __init__(self, stream):
        self.stream = stream

    def unpack(self, fmt):
        size = struct.calcsize("<" + fmt)
        data = self.stream.read(size)
        if len(data) != size:
            raise LTXError("Invalid GGUF header: truncated.")
        return struct.unpack("<" + fmt, data)[0]

    def count(self):
        value = self.unpack("Q")
        if value > MAX_GGUF_ITEMS:
            raise LTXError("Invalid or oversized GGUF header.")
        return value

    def string(self):
        length = self.unpack("Q")
        if length > MAX_GGUF_STRING:
            raise LTXError("Invalid GGUF header: oversized string.")
        data = self.stream.read(length)
        if len(data) != length:
            raise LTXError("Invalid GGUF header: truncated.")
        return data.decode("utf-8")

    def value(self, kind):
        if kind in GGUF_SCALARS:
            return self.unpack(GGUF_SCALARS[kind])
        if kind == 8:
            return self.string()
        if kind == 9:
            item_kind = self.unpack("I")
            return [self.value(item_kind) for _ in range(self.count())]
        raise LTXError(f"Invalid GGUF metadata type {kind}.")


@lru_cache(maxsize=32)
def _read_gguf_header(path, size, mtime):
    """The tensor table of a GGUF file in the shape of a safetensors header: name -> {"dtype", "shape"}."""
    try:
        with open(path, "rb") as raw:
            if raw.read(4) != GGUF_MAGIC:
                raise LTXError("Invalid GGUF file: missing magic.")
            stream = _GGUFStream(raw)
            if stream.unpack("I") not in (2, 3):
                raise LTXError("Unsupported GGUF version.")
            tensor_count, kv_count = stream.count(), stream.count()
            metadata = {}
            for _ in range(kv_count):
                key = stream.string()
                metadata[key] = stream.value(stream.unpack("I"))
            header = {}
            for _ in range(tensor_count):
                name = stream.string()
                dims = [stream.unpack("Q") for _ in range(stream.unpack("I"))]
                kind, _offset = stream.unpack("I"), stream.unpack("Q")
                if name in header:
                    raise LTXError(f"Duplicate tensor in GGUF header: {name}")
                # ggml lists dimensions innermost first; ComfyUI-GGUF stores reshaped tensors' real shape apart
                shape = metadata.get(f"comfy.gguf.orig_shape.{name}") or dims[::-1]
                header[name] = {"dtype": GGML_TYPES.get(kind, f"type{kind}"), "shape": [int(n) for n in shape]}
    except (OSError, UnicodeError, RecursionError) as exc:
        raise LTXError(f"Cannot read GGUF header: {exc}") from exc
    header["__metadata__"] = {k: v for k, v in metadata.items() if not k.startswith("comfy.gguf.orig_shape.")}
    return header


def read_header(path):
    path = Path(path).resolve()
    try:
        stat = path.stat()
    except OSError as exc:
        raise LTXError(f"Model file is unavailable: {path.name}") from exc
    reader = _read_gguf_header if path.suffix.lower() == ".gguf" else _read_header
    return reader(str(path), stat.st_size, stat.st_mtime_ns)


@dataclass(frozen=True)
class ModelInfo:
    path: Path
    role: str
    quantization: str
    # dit: "2.0" / "2.3" / "2.5"; text encoder: "gemma3" / "gemma4"; video VAE: "conv" / "diffusion"
    variant: str
    # an LTX-2.0 / 2.3 checkpoint carries the VAEs and the text projection; a text encoder may carry its projection
    bundled: bool = False


def _role(keys):
    from .native import detect

    prefixed = detect.is_ltx(keys)
    if prefixed or detect.is_ltx(keys, prefix=""):
        return "dit"
    if "model.layers.0.post_feedforward_layernorm.weight" in keys and "model.layers.47.self_attn.q_norm.weight" in keys:
        return "text_encoder"
    if "decoder.conv_in_x_t.weight" in keys or ("decoder.up_blocks.0.res_blocks.0.conv1.conv.weight" in keys and "encoder.conv_in.conv.weight" in keys):
        return "video_vae"
    if "audio_vae.decoder.conv_in.conv.weight" in keys:
        return "audio_vae"
    return None


# ComfyUI quantization formats as the files name them, and the short label the Components panel shows
QUANT_LABELS = {"int8_tensorwise": "int8", "asym_w4a8_int8": "w4a8", "w6a8_int8": "w6a8", "convrot_w4a4": "int4",
                "float8_e4m3fn": "fp8", "float8_e5m2": "fp8", "mxfp8": "mxfp8", "nvfp4": "nvfp4"}


def _quantization(header, keys):
    raw = header.get("__metadata__", {}).get("_quantization_metadata")
    if raw:
        try:
            layers = json.loads(raw).get("layers", {})
            labels = sorted({QUANT_LABELS.get(v.get("format"), v.get("format")) for v in layers.values() if isinstance(v, dict)} - {None})
            if labels:
                return " + ".join(labels)
        except (ValueError, AttributeError):
            pass
    dtypes = {v.get("dtype") for k, v in header.items() if k != "__metadata__"}
    if "U8" in dtypes and any(k.endswith("weight_scale_2") for k in keys):
        return "nvfp4"
    if any(k.endswith(".comfy_quant") for k in keys):
        return "int8" if "I8" in dtypes else "fp8" if {"F8_E4M3", "F8_E5M2"} & dtypes else "quantized"
    if {"F8_E4M3", "F8_E5M2"} & dtypes:
        return "fp8"
    return "bf16" if "BF16" in dtypes else "plain"


def _gguf_quantization(path, header):
    types = {v["dtype"] for k, v in header.items() if k != "__metadata__"}
    unsupported = sorted(types - GGUF_FORGE)
    if unsupported:
        raise LTXError(f"{path.name} uses GGUF types Forge Neo cannot dequantize ({', '.join(unsupported)}). "
                       "Use a Q2_K to Q8_0 build.")
    counts = {}
    for k, v in header.items():
        if k != "__metadata__" and v["dtype"] not in GGUF_PLAIN:
            counts[v["dtype"]] = counts.get(v["dtype"], 0) + 1
    return f"gguf {max(counts, key=counts.get)}" if counts else "gguf"


def inspect_model(path):
    path = Path(path).resolve()
    suffix = path.suffix.lower()
    if suffix not in (".safetensors", ".gguf"):
        return None
    header = read_header(path)
    keys = set(header) - {"__metadata__"}
    role = _role(keys)
    if role is None:
        return None
    variant, bundled = "", False
    if role == "dit":
        from .native import detect

        prefix = detect.PREFIX if detect.is_ltx(keys) else ""
        variant = detect.version(keys, prefix)
        bundled = "vae.encoder.conv_in.conv.weight" in keys and "audio_vae.decoder.conv_in.conv.weight" in keys
    elif role == "text_encoder":
        variant = "gemma3" if "model.layers.5.self_attn.v_proj.weight" in keys else "gemma4"
        bundled = any(k.startswith("text_embedding_projection.") for k in keys)
    elif role == "video_vae":
        variant = "diffusion" if "decoder.conv_in_x_t.weight" in keys else "conv"
    quantization = _gguf_quantization(path, header) if suffix == ".gguf" else _quantization(header, keys)
    return ModelInfo(path, role, quantization, variant, bundled)


@dataclass(frozen=True)
class Components:
    dit: ModelInfo
    text_encoder: ModelInfo
    video_vae: ModelInfo | None
    audio_vae: ModelInfo | None

    @property
    def models(self):
        return tuple(m for m in (self.dit, self.text_encoder, self.video_vae, self.audio_vae) if m is not None)

    @property
    def version(self) -> str:
        return self.dit.variant


def resolve_components(dit_path, module_paths):
    """The LTX checkpoint and the modules selected under VAE / Text Encoder, checked before Forge loads them."""
    dit = inspect_model(dit_path)
    if dit is None or dit.role != "dit":
        raise LTXError("The selected checkpoint is not a recognized LTX-2.x diffusion model.")
    resolved = {}
    for path in dict.fromkeys(str(Path(p).resolve()) for p in module_paths):
        item = inspect_model(path)
        if item is None or item.role == "dit":
            raise LTXError(f"Unrecognized LTX component: {Path(path).name}. Select only the LTX text encoder and VAEs.")
        if item.role in resolved:
            raise LTXError(f"More than one LTX {ROLE_LABELS[item.role]} is selected.")
        resolved[item.role] = item

    text_encoder = resolved.get("text_encoder")
    if dit.variant == "2.5":
        if text_encoder is None or text_encoder.variant != "gemma4":
            raise LTXError("LTX-2.5 needs its Gemma 4 text encoder: select gemma4-12b-with-proj-ltx-2.5 under VAE / Text Encoder.")
        if not text_encoder.bundled:
            raise LTXError(f"{text_encoder.path.name} has no text projection: select gemma4-12b-with-proj-ltx-2.5 (\"with-proj\").")
    elif text_encoder is None or text_encoder.variant != "gemma3":
        raise LTXError(f"LTX-{dit.variant} needs the Gemma 3 12B text encoder: select gemma_3_12B_it (for example "
                       "gemma_3_12B_it_fp8_scaled) under VAE / Text Encoder.")

    for role in ("video_vae", "audio_vae"):
        if role not in resolved and not dit.bundled:
            raise LTXError(f"Select the LTX {ROLE_LABELS[role]} under VAE / Text Encoder"
                           + (": ltx-2.5-video-vae-conv-bf16." if role == "video_vae" else ": ltx-2.5-audio-vae-bf16."))
    video = resolved.get("video_vae")
    if video is not None and video.variant == "diffusion":
        raise LTXError("The LTX-2.5 diffusion video decoder (ltx-2.5-video-vae-bf16) is not supported yet: select "
                       "ltx-2.5-video-vae-conv-bf16 instead.")
    return Components(dit=dit, text_encoder=text_encoder, video_vae=video, audio_vae=resolved.get("audio_vae"))
