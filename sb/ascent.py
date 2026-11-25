"""
A trained I2SB recon regressor run as an ImMAP-style ASCENT, with any noise schedule and step rule.

The equivalence (2026-11-25). Given x1, the Brownian bridge state is a variance-exploding diffusion
in the coordinate

    z = (x_t - mu_x1 x1) / mu_x0 = x0 + sigma_eff eps,     sigma_eff = std_sb / mu_x0
                                                          (~ 2 tau sqrt(t / (1 - t)))

(the Brownian bridge is a time-changed Brownian motion), so an x1-conditioned regressor
E[x0 | x_t, x1] is a Gaussian denoiser of p(x0 | x1) at level sigma_eff, and I2SB's DDPM step is
ImMAP's ascent step (DualCDL/code/csa.py)

    z <- z + h (m - z) + gamma eps,    gamma^2 = sigma^2 (c^2 - (1 - h)^2),    c = sigma_next / sigma

with h = 1 - c^2. Here the step rule is ImMAP's zeta-rule, h = 1 - c sqrt(1 - zeta),
gamma = sigma c sqrt(zeta):  zeta = 1 - c^2 per step ("ddpm") is I2SB's sampler; zeta = 0 is the
deterministic (DDIM-like) ascent; zeta = 1 re-noises the prediction (h = 1). The schedule is any
descending set of grid steps; `schedule_steps` builds the bridge's own (uniform t) or geometric /
linear sigma_eff schedules between the same endpoints, so a swap changes ONLY the spacing.

The net is queried exactly as the bridge sampler queries it: x_t = mu_x0 z + mu_x1 x1 on a grid
step, conditioned on that step's std_fwd, x1, y, E, the guide. The CG data-consistency step and the
hard support act on every prediction, as in sb/i2sb_recon.py::i2sb_recon_sample. The run starts
at the top step (t = 1, z = x1: the bridge's own start) and returns the state at step 0, mapped
back to x_t (masked by the support), as the bridge sampler does.
"""

from __future__ import annotations

import numpy as np
import torch

from sb.base import bridge_coeffs, forward_std, n_steps, space_indices
from sb.i2sb_recon import cg_data_consistency, predict_x0_recon

SCHEDULES = ("bridge", "geometric", "linear")


def sigma_eff(sched):
    """(N,) the VE noise level of each grid step: std_sb / mu_x0."""
    mu0, _, ssb = bridge_coeffs(sched)
    return ssb / mu0


def schedule_steps(sched, kind="bridge", nfe=20, sigma_hi=None, sigma_lo=None):
    """Ascending grid steps [0, ..., N-1], as reverse_sample's space_indices(N, nfe + 1).

    "bridge": uniform t (the I2SB sampler's own). "geometric" / "linear": the top step (t = 1,
    the one-shot start), then nfe - 1 evaluations spaced geometrically / linearly in sigma_eff
    from `sigma_hi` to `sigma_lo` -- by default the bridge schedule's own second and last
    evaluations, so only the spacing differs -- then step 0 (the returned state)."""
    N = n_steps(sched)
    bridge = space_indices(N, nfe + 1)
    if kind == "bridge":
        return bridge
    if kind not in SCHEDULES:
        raise ValueError(f"schedule {kind!r} not in {SCHEDULES}")
    sig = sigma_eff(sched).detach().cpu().numpy()
    hi = float(sig[bridge[-2]]) if sigma_hi is None else float(sigma_hi)
    lo = float(sig[bridge[1]]) if sigma_lo is None else float(sigma_lo)
    k = nfe - 1
    targets = (np.geomspace(hi, lo, k) if kind == "geometric" else np.linspace(hi, lo, k))
    idx = []
    for s in targets:                                     # nearest grid step, strictly descending
        n = int(np.argmin(np.abs(sig[1:N - 1] - s))) + 1
        if idx and n >= idx[-1]:
            n = idx[-1] - 1
        if n < 1:
            raise ValueError(f"{kind} schedule: {nfe} evaluations do not fit between sigma "
                             f"{hi:g} and {lo:g} on this grid")
        idx.append(n)
    return sorted({0, N - 1, *idx})


@torch.no_grad()
def ascent_sample(net, y, E, sched, x1, steps, zeta="ddpm", sigma_n=None, guide=None,
                  noise_scale=None, dc_iters=0, dc_lam=None, support=None):
    """The ascent over the ascending grid `steps` (module docstring) -> recon (x_t at step 0).

    zeta: "ddpm" (1 - c^2 per step: the I2SB sampler) or a number in [0, 1]. noise_scale: the
    per-sample temperature kappa, multiplying the injected noise (as the bridge sampler)."""
    if zeta != "ddpm" and not 0.0 <= float(zeta) <= 1.0:
        raise ValueError(f"zeta must be 'ddpm' or in [0, 1], got {zeta!r}")
    device = sched.std_fwd.device
    x1 = x1.to(device)
    mu0, mu1, _ = bridge_coeffs(sched)
    sig = sigma_eff(sched)
    sup = None if support is None else support.to(device=device, dtype=x1.real.dtype)
    kappa = 1.0
    if noise_scale is not None:
        kappa = torch.as_tensor(noise_scale, device=device, dtype=x1.real.dtype).reshape(
            -1, *([1] * (x1.dim() - 1)))
    steps = sorted(set(int(s) for s in steps))
    if steps[0] != 0:
        raise ValueError("steps must include 0 (the returned state)")
    z = x1.clone()                                        # the top step's state is x1: z = x1
    for n, p in zip(steps[1:][::-1], steps[:-1][::-1]):
        x_t = mu0[n] * z + mu1[n] * x1
        step_t = torch.full((x1.shape[0],), n, device=device, dtype=torch.long)
        sigma = forward_std(sched, step_t, xdim=x1.shape[1:])
        m = predict_x0_recon(net, x_t, sigma, x1=x1, y=y, E=E, sigma_n=sigma_n, guide=guide)
        if dc_iters:
            m = cg_data_consistency(m, y, E, dc_lam, dc_iters)
        if sup is not None:
            m = m * sup
        c = sig[p] / sig[n]
        zt = (1.0 - c * c) if zeta == "ddpm" else torch.as_tensor(float(zeta), device=device)
        h = 1.0 - c * torch.sqrt(1.0 - zt)
        z = z + h * (m - z)
        if p > 0:                                         # the last step adds no noise
            z = z + kappa * sig[n] * c * torch.sqrt(zt) * torch.randn_like(z)
    recon = mu0[0] * z + mu1[0] * x1
    return recon if sup is None else recon * sup
