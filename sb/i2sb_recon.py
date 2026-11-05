"""
sb/i2sb_recon.py -- I2SB for accelerated MRI RECONSTRUCTION.

    x0 = target: the fully-sampled coil-combined image (complex, (B, 1, H, W))
    x1 = bridge start, built from the measurement y alone:
             "adjoint"   x1 = E^H y, the zero-filled SENSE adjoint (the default; I2SB's
                         restoration setup -- the degraded image IS the prior endpoint)

The bridge is the ordinary sb/base.py one, run on COMPLEX tensors. Nothing in forward_sample or
reverse_sample is real-specific except the noise: `torch.randn_like` on a complex tensor draws
E|eps|^2 = 1, i.e. std 1/sqrt(2) per real component. `tau` / std_sb are therefore the COMPLEX
(total) std, the same convention as the measurement sigma everywhere in this repo
(operators/noise.py; models/cg_init.py converts by 1/sqrt(2) where a real std is needed).

Regressor contract (every net trained by training/i2sb_recon.py):

    x0_hat, extra = net(x_t, sigma=std_fwd, x1=x1, y=y, E=E, sigma_n=sigma_n)

`sigma` is the bridge position (std_fwd at the step), `sigma_n` the measurement noise std. A net
is free to ignore what it does not use: SBReconUnet (models/sb_recon_unet.py) reads x_t and x1
only; the unrolled nets will add data consistency through (y, E).
"""

import torch

from physics.object_mask import otsu_threshold
from sb.base import forward_std, reverse_sample


X1_SOURCES = ("adjoint",)
TAU_SCALES = ("none", "x1_rms")


def slice_scale(x1):
    """(B,) intensity of each zero-filled image: the RMS of |x1| over its Otsu foreground.

    Taken from x1 = E^H y alone, so it exists at test time. The foreground keeps the background
    out (an edge slice with less anatomy is not scored dimmer for its area), and it is a power, so
    it tracks the noise power the bridge has to match."""
    a = x1.abs().float()
    a = a.reshape(a.shape[0], 1, *a.shape[-2:]) if a.dim() == 4 else a
    fg = a > otsu_threshold(a)
    n = fg.flatten(1).sum(1).clamp_min(1)
    return ((a.pow(2) * fg).flatten(1).sum(1) / n).sqrt()


def bridge_noise_scale(x1, tau_scale="none", tau_ref=None):
    """Per-sample temperature multiplier kappa, or None for the absolute bridge.

    "x1_rms": kappa_i = slice_scale(x1_i) / tau_ref, so sample i's bridge runs at
    tau_i = tau * kappa_i -- the injected noise follows the slice's own intensity. With tau_ref
    the MEDIAN slice scale of the training data (scripts/calibrate_recon_bridge.py measures it),
    tau keeps its meaning: the absolute tau of a typical slice. The network's time conditioning
    stays on the reference schedule; on the Brownian bridge tau moves only the noise."""
    if tau_scale in (None, "none"):
        return None
    if tau_scale != "x1_rms":
        raise ValueError(f"tau_scale {tau_scale!r} not in {TAU_SCALES}")
    if not tau_ref or float(tau_ref) <= 0:
        raise ValueError("tau_scale='x1_rms' needs tau_ref > 0: the median slice scale of the "
                         "training data, from scripts/calibrate_recon_bridge.py")
    return slice_scale(x1) / float(tau_ref)


def bridge_start(y, E, source="adjoint"):
    """The prior endpoint x1 from the measurement. Deterministic in y, so it is the same at
    train and test time."""
    if source == "adjoint":
        return E.adjoint(y)
    raise ValueError(f"x1_source {source!r} not supported; use one of {X1_SOURCES}")


def predict_x0_recon(net, x_t, sigma, x1=None, y=None, E=None, sigma_n=None, guide=None):
    """Run a recon bridge regressor; returns the complex x0 estimate. `guide` (side
    information, guided regressors only) is passed only when there is one."""
    kw = {} if guide is None else {"guide": guide}
    out, _ = net(x_t, sigma=sigma, x1=x1, y=y, E=E, sigma_n=sigma_n, **kw)
    return out


@torch.no_grad()
def i2sb_recon_sample(net, y, E, sched, sigma_n=None, x1=None, x1_source="adjoint", nfe=None,
                      deterministic=False, posterior="ddpm", log_count=1, verbose=True,
                      guide=None, noise_scale=None):
    """Reconstruct x0 from (y, E) along the reverse bridge.

    `x1` defaults to bridge_start(y, E, x1_source); pass it to reuse one already built. y, E,
    sigma_n and the guide (side information) are held fixed along the whole trajectory.
    `noise_scale` is the per-sample temperature multiplier (bridge_noise_scale). Returns
    (recon, xs, pred_x0s), as sb.base.reverse_sample (no clip_denoise: a complex image has no
    [-1, 1] box)."""
    device = sched.std_fwd.device
    x1 = bridge_start(y, E, x1_source) if x1 is None else x1
    x1 = x1.to(device)

    def pred_x0_fn(x_t, step):
        step_t = torch.full((x_t.shape[0],), step, device=device, dtype=torch.long)
        sigma = forward_std(sched, step_t, xdim=x_t.shape[1:])
        return predict_x0_recon(net, x_t, sigma, x1=x1, y=y, E=E, sigma_n=sigma_n,
                                guide=guide)

    return reverse_sample(sched, pred_x0_fn, x1, nfe=nfe, deterministic=deterministic,
                          posterior=posterior, clip_denoise=False, log_count=log_count,
                          verbose=verbose, noise_scale=noise_scale)
