"""Small LTX accordion and native-control events for Forge's Gradio interface (the H3 extension's pattern)."""

import html
import logging

import gradio as gr

from .contracts import DEFAULT_FPS, DEFAULT_FRAMES, MAX_FPS, MIN_FPS, LTXError
from .integration import checkpoint_info
from .models import ROLE_LABELS, inspect_model
from .ui_state import duration_text, frame_view, preset_frame_view

COMPONENTS = {}
NATIVE_CONTROLS = ("batch_size", "batch_count", "sampling", "scheduler", "cfg_scale", "steps")
PANELS = []
logger = logging.getLogger("forge_ltx")

# the distilled template of the ComfyUI LTX-2.5 workflows (presets.py)
DEFAULT_SAMPLER = "Euler a"
DEFAULT_SCHEDULER = "LTX Distilled"
DEFAULT_STEPS = 8


def _in_blocks():
    # Forge also creates components outside the UI build (on page load); events can only be bound inside it
    return gr.context.Context.root_block is not None


def capture(component, **kwargs):
    elem_id = getattr(component, "elem_id", None)
    if elem_id:
        COMPONENTS[elem_id] = component
        if elem_id in ("txt2img_batch_size", "img2img_batch_size"):
            key = elem_id.replace("batch_size", "ltx_duration")
            if key not in COMPONENTS:
                COMPONENTS[key] = gr.Markdown(value="", visible=False, elem_id=key)
        if elem_id in ("setting_sd_model_checkpoint", "setting_sd_modules", "forge_ui_preset"):
            bind_all(elem_id)


class Panel:
    def __init__(self, is_img2img):
        self.tab = "img2img" if is_img2img else "txt2img"
        self.saved = gr.State({"active": False})
        with gr.Accordion("LTX Video", open=False, visible=False, elem_id=f"{self.tab}_ltx_panel") as self.accordion:
            self.output = gr.Radio(["Video", "Still image"], value="Video", label="Output", elem_id=f"{self.tab}_ltx_output")
            with gr.Row():
                self.audio = gr.Checkbox(value=True, label="Include generated audio", elem_id=f"{self.tab}_ltx_audio")
                self.fps = gr.Slider(minimum=MIN_FPS, maximum=MAX_FPS, step=1, value=DEFAULT_FPS, label="FPS",
                                     elem_id=f"{self.tab}_ltx_fps")
            self.status = gr.Markdown("Select the LTX text encoder (and, for LTX-2.5, both VAEs) in VAE / Text Encoder.")
            with gr.Accordion("Components", open=False):
                self.summary = gr.Markdown("")
        self.bound = False
        self.attempts = []
        PANELS.append(self)

    @property
    def inputs(self):
        # the order integration.script_before_process takes
        return [self.output, self.audio, self.fps]

    @property
    def needed(self):
        ids = [f"{self.tab}_{name}" for name in NATIVE_CONTROLS]
        return ids + ["setting_sd_model_checkpoint", "setting_sd_modules", f"{self.tab}_ltx_duration"]

    def bind(self, trigger="?"):
        if self.bound:
            return
        if not _in_blocks():
            self.attempts.append(f"{trigger}: outside Blocks")
            return
        needed = self.needed
        if any(name not in COMPONENTS for name in needed):
            self.attempts.append(f"{trigger}: waiting for {', '.join(n for n in needed if n not in COMPONENTS)}")
            return
        native = [COMPONENTS[f"{self.tab}_{name}"] for name in NATIVE_CONTROLS]
        checkpoint = COMPONENTS["setting_sd_model_checkpoint"]
        modules = COMPONENTS["setting_sd_modules"]
        preset = COMPONENTS.get("forge_ui_preset")
        duration = COMPONENTS[f"{self.tab}_ltx_duration"]
        defaults = [{key: getattr(c, key) for key in ("minimum", "maximum", "step", "label", "visible", "choices", "interactive")
                     if hasattr(c, key)} for c in native]
        preset_input = [preset] if preset is not None else []
        inputs = [checkpoint, self.output, self.fps, modules, self.saved] + native + preset_input
        outputs = [self.accordion, self.audio, self.fps, duration, self.status, self.summary, self.saved] + native + preset_input

        def update(value, output, fps, module_values, saved, *values):
            preset_value = values[-1] if preset is not None else None
            values = values[:len(native)]
            info = checkpoint_info(value)
            error = ""
            active = False
            if info:
                try:
                    item = inspect_model(info.filename)
                    active = item is not None and item.role == "dit"
                except LTXError as exc:
                    error = str(exc)
            saved = dict(saved or {"active": False})
            entering = active and not saved.get("active")
            leaving = not active and saved.get("active")
            if entering:
                saved["native"] = [dict(config, value=value) for config, value in zip(defaults, values)]
                if preset_value is not None:
                    from modules_forge.presets import is_video
                    saved["native"][0].update(preset_frame_view(is_video(preset_value), values[0]))
            updates = [gr.update() for _ in native]
            frames = values[0]
            if active:
                frames = DEFAULT_FRAMES if entering else frames
                view = frame_view(True, output, frames)
                updates[0] = gr.update(**view)
                frames = view["value"]
                updates[1] = gr.update(value=1, visible=False)
                if entering:
                    updates[2] = gr.update(value=DEFAULT_SAMPLER)
                    updates[3] = gr.update(value=DEFAULT_SCHEDULER)
                    updates[4] = gr.update(value=1.0)
                    updates[5] = gr.update(value=DEFAULT_STEPS)
            elif leaving:
                updates = [gr.update(**config) for config in saved.get("native", [])]
                saved.pop("native", None)
            saved["active"] = active
            summary = ""
            if active:
                try:
                    from .integration import module_paths
                    from .models import resolve_components
                    components = resolve_components(info.filename, module_paths(module_values))
                    lines = [f"**Version:** LTX-{components.version}"]
                    lines += [f"**{ROLE_LABELS[m.role][:1].upper() + ROLE_LABELS[m.role][1:]}:** {html.escape(m.path.name)} ({m.quantization})"
                              for m in components.models]
                    if components.dit.bundled:
                        lines.append("**VAEs:** inside the checkpoint")
                    summary = "  \n".join(lines)
                    error = ""
                except LTXError as exc:
                    error = str(exc)
            video = output == "Video"
            status = html.escape(error) if error else ("LTX makes video and audio together; the checkbox keeps or drops the audio in the MP4."
                                                       if video else "Still image is a one-frame LTX generation.")
            return [gr.update(visible=active), gr.update(visible=active and video), gr.update(visible=active and video),
                    gr.update(value=duration_text(frames, fps) if active else "", visible=active and video), status, summary,
                    saved] + updates + ([gr.update()] if preset is not None else [])

        for event in (checkpoint.change, self.output.change, modules.change):
            event(update, inputs=inputs, outputs=outputs, queue=False, show_progress=False)
        # frame and FPS changes update the duration without touching the other native controls
        for source in (native[0], self.fps):
            source.change(lambda n, fps: gr.update(value=duration_text(n, fps)), inputs=[native[0], self.fps], outputs=[duration],
                          queue=False, show_progress=False)
        gr.context.Context.root_block.load(update, inputs=inputs, outputs=outputs, queue=False, show_progress=False)
        self.bound = True
        logger.info("LTX native controls connected for %s.", self.tab)


def bind_all(trigger="ui_tabs"):
    for panel in PANELS:
        panel.bind(trigger)
    return []


def check_bindings(*_args):
    # Forge also builds panel instances it never renders; a tab is fine as long as one of its panels is bound
    for tab in dict.fromkeys(panel.tab for panel in PANELS):
        panels = [panel for panel in PANELS if panel.tab == tab]
        if any(panel.bound for panel in panels):
            continue
        missing = sorted({name for panel in panels for name in panel.needed if name not in COMPONENTS})
        attempts = [attempt for panel in panels for attempt in panel.attempts]
        logger.error("LTX native controls were not found for %s (missing: %s; attempts: %s); restart Forge after "
                     "updating it.", tab, ", ".join(missing) or "none", "; ".join(attempts) or "none")


def reset():
    COMPONENTS.clear()
    PANELS.clear()
