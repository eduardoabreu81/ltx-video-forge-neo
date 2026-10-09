"""LTX inside Forge's own txt2img and img2img flows, through script callbacks.

Forge loads the model (native/patches.py registers it), samples and decodes as usual. The callbacks here turn the
Frames control into LTX's length, hand the engine its first / last frames (keyframes.py), give the sampler the packed
video+audio noise, run the second stage of a two-stage generation, and write the MP4 with sound.
"""

import json
import uuid
from pathlib import Path

from . import keyframes, upscalers
from .contracts import GenerationRequest, LTXError, set_pending_error
from .media import export_still, export_video, find_ffmpeg
from .models import inspect_model, resolve_components

# img2img Resize mode "Just resize (latent upscale)": it would interpolate the placeholder latent
LATENT_UPSCALE = 3


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
    if getattr(p, "n_iter", 1) != 1:
        raise LTXError("LTX currently supports Batch Count = 1. Frames sets the video length.")
    if getattr(p, "enable_hr", False) or getattr(p, "txt2img_upscale", False):
        raise LTXError("Disable Hires. fix for LTX generation.")
    if getattr(p, "restore_faces", False):
        raise LTXError("Disable Restore faces for LTX generation.")
    if getattr(p, "image_mask", None) is not None:
        raise LTXError("LTX inpainting is not available. Use the regular img2img image input as the first frame.")
    if (getattr(p, "subseed_strength", 0) > 0 or getattr(p, "seed_resize_from_w", -1) > 0
            or getattr(p, "seed_resize_from_h", -1) > 0):
        raise LTXError("Disable variation seed and seed resize for LTX.")
    if getattr(p, "script_args", ()) and p.script_args[0] not in (0, None, "None"):
        raise LTXError("Select Script: None for LTX generation. Script combinations are not validated yet.")


def keyframe_settings(strength, compression) -> tuple[float | None, int | None]:
    """The panel's keyframe strength and compression; None (an API call that leaves them out) is the version's default."""
    if strength is not None:
        strength = float(strength)
        if not 0.0 <= strength <= 1.0:
            raise LTXError("LTX keyframe strength must be between 0 and 1.")
    if compression is not None:
        compression = int(compression)
        if not 0 <= compression <= 100:
            raise LTXError("LTX keyframe compression must be between 0 and 100.")
    return strength, compression


def before_process(p, output="Video", include_audio=True, fps=None, strength=None, compression=None, two_stage=False,
                   upscaler=None):
    """Before Forge loads the model: check the request and turn Frames into a single LTX generation."""
    p.ltx_request = None
    p.ltx_last_frame = None
    p.ltx_upscaler = None
    set_pending_error(None)
    try:
        _before_process(p, output, include_audio, fps, strength, compression, two_stage, upscaler)
    except LTXError as error:
        set_pending_error(error)
        raise


def script_before_process(p, output="Video", include_audio=True, fps=None, strength=None, compression=None, two_stage=False,
                          upscaler=None, *_):
    """before_process as Forge's script runner calls it: a rejected request prints one line instead of the traceback
    Forge logs for any exception; the error stays pending and stops the generation when the model is first called."""
    try:
        before_process(p, output, include_audio, fps, strength, compression, two_stage, upscaler)
    except LTXError as error:
        print(f"[LTX] {error}")


def is_distilled_schedule(scheduler) -> bool:
    from .native import schedules
    return scheduler in (schedules.DISTILLED_NAME, schedules.DISTILLED_LABEL)


def guidance_note(dev: bool, cfg: float, scheduler) -> str | None:
    """A console hint when CFG does not suit the checkpoint: distilled ones are made for CFG 1, a dev one needs CFG
    (or the distilled LoRA, which goes with the LTX Distilled schedule)."""
    if not dev and cfg != 1.0:
        return f"distilled checkpoints are made for CFG 1: CFG {cfg:g} applies guidance twice and runs two passes per step"
    if dev and cfg == 1.0 and not is_distilled_schedule(scheduler):
        return "a dev checkpoint at CFG 1 needs the distilled LoRA (with LTX Distilled); without it use CFG 3 to 4"
    return None


def _before_process(p, output, include_audio, fps, strength, compression, two_stage, upscaler):
    from modules import processing, shared

    from .contracts import DEFAULT_FPS
    info = select_ltx(p)
    if info is None:
        return
    validate_processing(p)
    is_img2img = isinstance(p, processing.StableDiffusionProcessingImg2Img)
    if is_img2img:
        if not getattr(p, "init_images", None):
            raise LTXError("Add an input image for LTX in img2img: it becomes the first frame.")
        if getattr(p, "resize_mode", 0) == LATENT_UPSCALE:
            raise LTXError("LTX does not take the latent upscale resize mode. Choose another Resize mode.")
    overrides = getattr(p, "override_settings", {})
    components = resolve_components(info.filename, module_paths(overrides.get("forge_additional_modules", shared.opts.forge_additional_modules)))
    request = GenerationRequest(width=p.width, height=p.height, frames=p.batch_size, fps=fps or DEFAULT_FPS,
                                output=output, include_audio=include_audio, two_stage=bool(two_stage))
    if request.two_stage:
        p.ltx_upscaler = str(upscalers.resolve(upscaler, components.version))
    last = keyframes.last_frame(p, request.width, request.height)
    if last is not None and request.output != "Video":
        raise LTXError("A last frame (ImageStitch Integrated) needs Output: Video.")
    strength, compression = keyframe_settings(strength, compression)
    if request.output == "Video":
        find_ffmpeg(getattr(shared.opts, "ltx_ffmpeg_path", ""))
    p.ltx_request = request
    p.ltx_version = components.version
    p.ltx_dev = components.dit.is_dev
    p.ltx_first_frame = is_img2img
    p.ltx_last_frame = last
    p.ltx_keyframe_settings = (strength, compression)
    p.batch_size = 1
    if is_img2img:
        # LTX generates the whole clip from noise; the input image conditions it as the first frame
        p.denoising_strength = 1.0
    audio = "with audio" if request.include_audio else "without audio"
    frames = " and ".join(name for name, used in (("first", is_img2img), ("last", last is not None)) if used)
    conditioning = f", {frames} frame" if frames else ""
    stages = f", two stages from {request.first_stage_size[0]}x{request.first_stage_size[1]}" if request.two_stage else ""
    print(f"[LTX] LTX-{components.version} {'dev' if p.ltx_dev else 'distilled'} {request.output.lower()}: "
          f"{request.frames} frames at {request.width}x{request.height}, {request.fps} FPS ({request.duration:.2f} s), "
          f"{audio}{conditioning}{stages}")
    note = guidance_note(p.ltx_dev, getattr(p, "cfg_scale", 1.0), getattr(p, "scheduler", None))
    if note:
        print(f"[LTX] {note}")


def process(p):
    request = getattr(p, "ltx_request", None)
    if request is None:
        return
    if not getattr(p.sd_model, "is_ltx", False):
        raise LTXError("Forge did not load the LTX model. Check the VAE / Text Encoder selection and the console.")
    from .native import schedules
    from .native.streams import SPATIAL, latent_frames
    if schedules.is_dev(getattr(p, "scheduler", None)):
        width, height = request.first_stage_size
        tokens = latent_frames(request.frames) * (height // SPATIAL) * (width // SPATIAL)
        p.sampler_noise_scheduler_override = schedules.dev_override(tokens)
    engine = p.sd_model
    last = getattr(p, "ltx_last_frame", None)
    strength, compression = getattr(p, "ltx_keyframe_settings", (None, None))
    engine.set_keyframes(keyframes.to_tensor(last) if last is not None else None, strength, compression)
    p.extra_generation_params.update({"LTX Version": getattr(p, "ltx_version", ""), "LTX Frames": request.frames,
                                      "LTX FPS": request.fps, "LTX Audio": request.include_audio, "LTX Output": request.output})
    if request.two_stage:
        p.extra_generation_params.update({"LTX Two stages": True, "LTX Upscaler": Path(p.ltx_upscaler).name})
    first = bool(getattr(p, "ltx_first_frame", False))
    if first or last is not None:
        p.extra_generation_params.update({"LTX First frame": first, "LTX Last frame": last is not None,
                                          "LTX Keyframe strength": engine.keyframe_strength,
                                          "LTX Keyframe compression": engine.keyframe_compression})


def before_sampling(p, noise):
    import torch
    from modules import rng
    request = getattr(p, "ltx_request", None)
    if request is None:
        return
    if getattr(p, "ltx_first_frame", False) and p.sd_model.first_frame is None:
        error = LTXError("The img2img input image did not reach LTX. Set Settings > VAE > VAE for Encoding to Full, "
                         "and check the console for an earlier error.")
        set_pending_error(error)
        raise error
    width, height = request.first_stage_size
    shape = p.sd_model.prepare(request.frames, width, height, request.fps, int(p.seeds[0]))
    # Forge made p.rng for an image latent; the samplers that add noise on the way (ancestral, SDE) draw from it too,
    # so it has to give the packed shape
    p.rng = rng.ImageRNG(shape, p.seeds, subseeds=p.subseeds, subseed_strength=p.subseed_strength,
                         seed_resize_from_h=p.seed_resize_from_h, seed_resize_from_w=p.seed_resize_from_w)
    p.modified_noise = p.rng.next().to(device=noise.device, dtype=noise.dtype)
    if getattr(p, "init_latent", None) is not None:
        # img2img samples from init_latent at full denoise; the packed start is pure noise
        p.init_latent = torch.zeros_like(p.modified_noise)


def post_sample(p, samples):
    """After Forge's sampling: for two stages, the latent refined at the output size takes the first stage's place."""
    request = getattr(p, "ltx_request", None)
    if request is None or not request.two_stage:
        return samples
    from modules import shared
    if shared.state.interrupted or shared.state.skipped:
        return samples
    return refine(p, samples)


def refine(p, samples):
    """The second stage: the first stage's latent upscaled x2, new noise (from the seed + 1, as Forge's sampler draws
    it), and the version's refining sigmas through Forge's own img2img sampling, with the same sampler, prompt and
    CFG."""
    from modules import rng

    from .native import schedules
    request = p.ltx_request
    version = getattr(p, "ltx_version", "2.5")
    start = p.sd_model.refine_start(samples, request.width, request.height, p.ltx_upscaler)
    seeds = [int(seed) + 1 for seed in p.seeds]
    # the samplers that add noise on the way draw from p.rng too
    p.rng = rng.ImageRNG(tuple(start.shape[1:]), seeds, subseeds=p.subseeds, subseed_strength=p.subseed_strength,
                         seed_resize_from_h=p.seed_resize_from_h, seed_resize_from_w=p.seed_resize_from_w)
    noise = p.rng.next()
    start = start.to(device=noise.device, dtype=noise.dtype)
    saved = p.sampler_noise_scheduler_override, p.denoising_strength
    p.sampler_noise_scheduler_override = schedules.refine_override(version)
    # with full denoise, Forge's img2img sampling runs the whole schedule it is given
    p.denoising_strength = 1.0
    try:
        return p.sampler.sample_img2img(p, start, noise, p.c, p.uc, steps=len(schedules.refine_sigmas(version)) - 1,
                                        image_conditioning=p.txt2img_image_conditioning(start))
    finally:
        p.sampler_noise_scheduler_override, p.denoising_strength = saved


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
                   version=getattr(p, "ltx_version", ""), two_stage=request.two_stage,
                   upscaler=Path(p.ltx_upscaler).name if request.two_stage else None,
                   first_frame=bool(getattr(p, "ltx_first_frame", False)),
                   last_frame=getattr(p, "ltx_last_frame", None) is not None, infotext=infotext)
    Path(output).with_suffix(".json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    processed.video_path = output
    processed.comments += f"LTX {request.output.lower()} saved to {output}\n"
