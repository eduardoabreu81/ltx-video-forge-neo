"""The LTX schedules, added to Forge Neo's Schedule type list.

"LTX Distilled": the sigmas the distilled checkpoints are trained on, from the ComfyUI templates (ManualSigmas, 8
steps). At 8 steps they are exact; other step counts follow the same curve, interpolated over the step index.

"LTX": ComfyUI's LTXVScheduler for the dev checkpoints: a uniform grid shifted by the latent's token count (shift
0.95 at 1024 tokens up to 2.05 at 4096, extrapolated linearly), stretched so the last sigma before 0 is 0.1.
"""

import math

import torch

DISTILLED = (1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0)

DISTILLED_NAME, DISTILLED_LABEL = "ltx_distilled", "LTX Distilled"
DEV_NAME, DEV_LABEL = "ltx", "LTX"

MAX_SHIFT, BASE_SHIFT = 2.05, 0.95
TERMINAL = 0.1
DEFAULT_TOKENS = 4096


def distilled_sigmas(steps: int) -> torch.Tensor:
    if steps == len(DISTILLED) - 1:
        return torch.tensor(DISTILLED, dtype=torch.float32)
    position = torch.linspace(0, len(DISTILLED) - 1, steps + 1, dtype=torch.float64)
    low = position.floor().long().clamp(max=len(DISTILLED) - 2)
    frac = position - low
    table = torch.tensor(DISTILLED, dtype=torch.float64)
    return (table[low] * (1 - frac) + table[low + 1] * frac).float()


def dev_sigmas(steps: int, tokens: int = DEFAULT_TOKENS, stretch: bool = True, terminal: float = TERMINAL) -> torch.Tensor:
    sigmas = torch.linspace(1.0, 0.0, steps + 1)
    slope = (MAX_SHIFT - BASE_SHIFT) / (4096 - 1024)
    shift = tokens * slope + (BASE_SHIFT - slope * 1024)
    sigmas = torch.where(sigmas != 0, math.exp(shift) / (math.exp(shift) + (1 / sigmas - 1)), 0)
    if stretch:
        nonzero = sigmas != 0
        one_minus = 1.0 - sigmas[nonzero]
        sigmas[nonzero] = 1.0 - one_minus / (one_minus[-1] / (1.0 - terminal))
    return sigmas


def _distilled(n, sigma_min=None, sigma_max=None, device="cpu", **kwargs):
    return distilled_sigmas(n).to(device)


def _dev(n, sigma_min=None, sigma_max=None, device="cpu", **kwargs):
    # without the request's latent size (see override): the ComfyUI node's default of 4096 tokens
    return dev_sigmas(n).to(device)


SCHEDULES = {DISTILLED_LABEL: (DISTILLED_NAME, _distilled), DEV_LABEL: (DEV_NAME, _dev)}


def register() -> None:
    from modules import sd_schedulers, shared

    for label, (name, function) in SCHEDULES.items():
        if label in sd_schedulers.schedulers_map:
            continue
        entry = sd_schedulers.Scheduler(name, label, function)
        sd_schedulers.all_schedulers.append(entry)
        if label not in shared.opts.hide_schedulers:
            sd_schedulers.schedulers.append(entry)
            for key in (name, label):
                sd_schedulers.schedulers_map[key] = entry


def is_dev(scheduler) -> bool:
    return scheduler in (DEV_NAME, DEV_LABEL)


def dev_override(tokens: int):
    """A sampler_noise_scheduler_override that gives LTX its shift for the request's latent token count."""
    def schedule(steps: int) -> torch.Tensor:
        return dev_sigmas(steps, tokens)

    return schedule
