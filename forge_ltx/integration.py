"""LTX inside Forge's own txt2img flow, through script callbacks.

Forge loads the model (native/patches.py registers it), samples and decodes as usual. The callbacks here turn the
Frames control into LTX's length, give the sampler the packed video+audio noise, and write the MP4 with sound.
"""

import json
import uuid
from pathlib import Path

from .contracts import GenerationRequest, LTXError, set_pending_error
from .media import export_still, export_video, find_ffmpeg
from .models import inspect_model, resolve_components


def checkpoint_info(value):
    # the same lookup Forge applies to a checkpoint override: alias, then title substring, with or without the hash
    from modules import sd_models
    return sd_models.get_closet_checkpoint_match(value)


def select_ltx(p):
    """The selected checkpoint when it is an LTX diffusion model; read from its header, before Forge loads it."""
    from modules import shared
    overrides = getattr(p, "override_settings", {})
    info = checkpoint_info(overrides.get("sd_model_checkpoint", shared.opts.sd_model_checkpoint))
    if info is None:
        return None
    item = inspect_model(info.filename)
    return info if item is not None and item.role == "dit" else None


def module_paths(values):
    from modules_forge import main_entry
    result = []
    for value in values or []:
        path = main_entry.module_list.get(value, value)
        if not Path(path).is_file():
            raise LTXError(f"Selected LTX component is unavailable: {Path(value).name}. Refresh the model list.")
        result.append(str(Path(path).resolve()))
    return result


def validate_processing(p):
    from modules import processing
    if isinstance(p, processing.StableDiffusionProcessingImg2Img):
        raise LTXError("LTX image-to-video is not available yet: use txt2img.")
    if getattr(p, "n_iter", 1) != 1:
        raise LTXError("LTX currently supports Batch Count = 1. Frames sets the video length.")
    if getattr(p, "enable_hr", False) or getattr(p, "txt2img_upscale", False):
        raise LTXError("Disable Hires. fix for LTX generation.")
    if getattr(p, "restore_faces", False):
        raise LTXError("Disable Restore faces for LTX generation.")
    if (getattr(p, "subseed_strength", 0) > 0 or getattr(p, "seed_resize_from_w", -1) > 0
            or getattr(p, "seed_resize_from_h", -1) > 0):
        raise LTXError("Disable variation seed and seed resize for LTX.")
    if getattr(p, "script_args", ()) and p.script_args[0] not in (0, None, "None"):
        raise LTXError("Select Script: None for LTX generation. Script combinations are not validated yet.")


def before_process(p, output="Video", include_audio=True, fps=None):
    """Before Forge loads the model: check the request and turn Frames into a single LTX generation."""
    p.ltx_request = None
    set_pending_error(None)
    try:
        _before_process(p, output, include_audio, fps)
    except LTXError as error:
        set_pending_error(error)
        raise


def script_before_process(p, output="Video", include_audio=True, fps=None, *_):
    """before_process as Forge's script runner calls it: a rejected request prints one line instead of the traceback
    Forge logs for any exception; the error stays pending and stops the generation when the model is first called."""
    try:
        before_process(p, output, include_audio, fps)
    except LTXError as error:
        print(f"[LTX] {error}")


def _before_process(p, output, include_audio, fps):
    from modules import shared

    from .contracts import DEFAULT_FPS
    info = select_ltx(p)
    if info is None:
        return
    validate_processing(p)
    overrides = getattr(p, "override_settings", {})
    components = resolve_components(info.filename, module_paths(overrides.get("forge_additional_modules", shared.opts.forge_additional_modules)))
    request = GenerationRequest(width=p.width, height=p.height, frames=p.batch_size, fps=fps or DEFAULT_FPS,
                                output=output, include_audio=include_audio)
    if request.output == "Video":
        find_ffmpeg(getattr(shared.opts, "ltx_ffmpeg_path", ""))
    p.ltx_request = request
    p.ltx_version = components.version
    p.batch_size = 1
    audio = "with audio" if request.include_audio else "without audio"
    print(f"[LTX] LTX-{components.version} {request.output.lower()}: {request.frames} frames at {request.width}x{request.height}, "
          f"{request.fps} FPS ({request.duration:.2f} s), {audio}")


def process(p):
    request = getattr(p, "ltx_request", None)
    if request is None:
        return
    if not getattr(p.sd_model, "is_ltx", False):
        raise LTXError("Forge did not load the LTX model. Check the VAE / Text Encoder selection and the console.")
    from .native import schedules
    from .native.streams import SPATIAL, latent_frames
    if schedules.is_dev(getattr(p, "scheduler", None)):
        tokens = latent_frames(request.frames) * (request.height // SPATIAL) * (request.width // SPATIAL)
        p.sampler_noise_scheduler_override = schedules.dev_override(tokens)
    p.extra_generation_params.update({"LTX Version": getattr(p, "ltx_version", ""), "LTX Frames": request.frames,
                                      "LTX FPS": request.fps, "LTX Audio": request.include_audio, "LTX Output": request.output})


def before_sampling(p, noise):
    import torch
    from modules import rng
    request = getattr(p, "ltx_request", None)
    if request is None:
        return
    shape = p.sd_model.prepare(request.frames, request.width, request.height, request.fps, int(p.seeds[0]))
    # Forge made p.rng for an image latent; the samplers that add noise on the way (ancestral, SDE) draw from it too,
    # so it has to give the packed shape
    p.rng = rng.ImageRNG(shape, p.seeds, subseeds=p.subseeds, subseed_strength=p.subseed_strength,
                         seed_resize_from_h=p.seed_resize_from_h, seed_resize_from_w=p.seed_resize_from_w)
    p.modified_noise = p.rng.next().to(device=noise.device, dtype=noise.dtype)
    if getattr(p, "init_latent", None) is not None:
        p.init_latent = torch.zeros_like(p.modified_noise)


def postprocess(p, processed):
    """After Forge's own outputs: write the MP4 with sound next to them."""
    request = getattr(p, "ltx_request", None)
    if request is None:
        return
    engine = p.sd_model
    generation = getattr(engine, "generation", None)
    from modules import shared
    try:
        if generation is None or generation.decoded is None:
            return
        if shared.state.interrupted or shared.state.skipped:
            processed.comments += "LTX video not written: the generation was interrupted\n"
            return
        _write_output(p, processed, request, generation)
    finally:
        engine.release_generation()


def _write_output(p, processed, request, generation):
    import numpy as np
    import torch
    from modules import shared
    from PIL import Image

    pixels = generation.decoded.clamp(0, 1).mul(255).round().to(torch.uint8)
    frames = [Image.fromarray(np.moveaxis(frame.numpy(), 0, 2)) for frame in pixels]
    infotext = processed.infotexts[0] if processed.infotexts else processed.info
    directory = Path(p.outpath_samples or shared.opts.outdir_samples or "outputs/ltx").resolve()
    stem = f"ltx-{generation.seed}-{uuid.uuid4().hex[:12]}"
    if request.output == "Still image":
        output = export_still(frames[0], directory / f"{stem}.png", infotext)
    else:
        audio = generation.waveform if request.include_audio else None
        output = export_video(frames, audio, directory / f"{stem}.mp4", fps=request.fps, sample_rate=generation.sample_rate,
                              ffmpeg=getattr(shared.opts, "ltx_ffmpeg_path", ""), infotext=infotext,
                              cancelled=lambda: shared.state.interrupted)
    sidecar = dict(prompt=p.prompt, negative_prompt=p.negative_prompt, seed=generation.seed, width=request.width,
                   height=request.height, frames=request.frames, fps=request.fps, steps=p.steps, sampler=p.sampler_name,
                   scheduler=p.scheduler, cfg=p.cfg_scale, include_audio=request.include_audio, output=request.output,
                   version=getattr(p, "ltx_version", ""), infotext=infotext)
    Path(output).with_suffix(".json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    processed.video_path = output
    processed.comments += f"LTX {request.output.lower()} saved to {output}\n"
