"""
Task adapters: how to turn one dataloader batch into a `(gt, recon)` pair.

This is the only part of the sweep that knows what a task IS. Everything else
-- metrics, aggregation, CSV -- is task agnostic, so adding a task means adding
one function here and nothing else.

An adapter has the signature

    adapter(net, batch, cfg, device, sigma, generator) -> (gt, recon)

both complex or both real, shape `(B, C, H, W)`, on `device`.

`sigma` is PINNED by the caller rather than sampled per batch. Evaluation wants
a fixed operating point (or a sweep of them); sampling would average over the
training noise distribution and hide the shape of the degradation.

Each adapter mirrors the corresponding validation loop in `training/` -- that
is deliberate, and the reason to change them together: if this drifts from
`training/recon.py`'s val block, the sweep stops measuring what training
reported.
"""

from __future__ import annotations

import torch

from operators import FFT2D, Identity, Mask, Sense
from operators.noise import awgn
from physics.mask import get_mask_cached as get_mask
from training.common import prepare_measurement


def _call_net(net, x, E, sigma):
    """Uniform model call. Unet variants take real-valued stacked input."""
    if net.__class__.__name__ in ("Unet", "NormUnet"):
        out = net(torch.view_as_real(x))
        return torch.view_as_complex(out.contiguous())
    out = net(x, E=E, sigma=sigma)
    return out[0] if isinstance(out, tuple) else out


# ---------------------------------------------------------------------------
def recon(net, batch, cfg, device, sigma, generator=None):
    """CS-MRI reconstruction. Mirrors the val block of `training/recon.py`."""
    kspace, smaps, image, _organ_mask = (b.to(device, non_blocking=True)
                                         for b in batch)
    mri = cfg["mri"]

    mask = get_mask(image, R=mri["R"], acs_lines=mri["acs_lines"],
                    mode=mri.get("mask_dist", "uniform"),
                    offset=mri.get("mask_offset", 0))

    y, sigma_n, extra = prepare_measurement(
        image=image, kspace=kspace, mask=mask, smaps=smaps,
        kspace_type=mri.get("kspace_type", "simulated"),
        noise_std=sigma,                        # a number: pinned, not sampled
        noise_dist=cfg["training"].get("noise_dist", "uniform"),
        whiten_kspace=mri.get("whiten_kspace", False),
        generator=generator,
    )

    E = Mask(mask) @ FFT2D() @ Sense(extra["smaps"])
    out = _call_net(net, y, E, sigma_n)

    if mri.get("whiten_kspace", False) and "Zinv" in extra:
        out = extra["Zinv"] * out
    return image, out


def denoiser(net, batch, cfg, device, sigma, generator=None):
    """Gaussian denoising. Mirrors the val block of `training/denoiser.py`.

    A loader may hand back either a clean image (noise added here) or a
    pre-noised `(noisy, gt, sigma)` triple; the second form's own sigma wins,
    because the pair was generated together and re-noising would be wrong.
    """
    if isinstance(batch, (list, tuple)):
        noisy, gt, sigma_n = (b.to(device, non_blocking=True) for b in batch)
    else:
        gt = batch.to(device, non_blocking=True)
        noisy, sigma_n = awgn(gt, [sigma, sigma],
                              dist=cfg["training"].get("noise_dist", "uniform"))

    return gt, _call_net(net, noisy, Identity(), sigma_n)


REGISTRY = {
    "recon": recon,
    "denoiser": denoiser,
}


def build_adapter(task):
    if task not in REGISTRY:
        raise ValueError(
            f"no evaluation adapter for task {task!r}; registered: "
            f"{sorted(REGISTRY)}. Add one to evaluation/tasks.py.")
    return REGISTRY[task]


def default_sigma(cfg):
    """The noise level to evaluate at when none is given.

    `val_noise_std` if the config pins one -- that is what training validated
    against, so the sweep reproduces the number on the val curve. Otherwise the
    midpoint of the training range, which is what a pinned validation would
    have used anyway.
    """
    t = cfg.get("training", {})
    if t.get("val_noise_std") is not None:
        return float(t["val_noise_std"])
    std = t.get("noise_std", 0.0)
    if isinstance(std, (list, tuple)):
        return float(sum(std) / len(std))
    return float(std)
