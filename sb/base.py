"""
sb/base.py -- Schrodinger-bridge schedule and the shared algorithm helpers.

The whole bridge is ONE tensor: `std_fwd` (sigma over the discrete steps), the "diffusion time"
the network conditions on. Everything else -- the interpolation weights mu_x0 / mu_x1 and the
bridge-noise std std_sb -- is derived from it on demand (`bridge_coeffs`), so a schedule is
literally a labelled 1-D tensor. This replaces the old physics/bbridge.py, whose BridgeSchedule
stored five redundant tensors.

Two ways to build one:

    brownian(tau, n)              the constant Brownian bridge:  std_fwd(t) = 2 * tau * sqrt(t).
                                  `tau` is the TEMPERATURE -- the peak injected-noise std
                                  (max std_sb = tau at t = 1/2). The mean is a plain linear
                                  interpolation mu_x0 = 1 - t, mu_x1 = t.
    from_betas(betas)             any schedule at all:  std_fwd = sqrt(cumsum(betas)).

The FAITHFUL I2SB paper schedule is NOT the Brownian one (its betas are a mirrored quadratic, so
std_fwd is not proportional to sqrt(t)). Reproduce it EXACTLY with

    from_betas(i2sb_betas(n, beta_max))

and `build_schedule(kind=...)` is the single config-facing switch between the two: "brownian"
(default) or "i2sb".

Helpers shared by training and every sampler:
    forward_std      sigma at a step (fed to the network)
    bridge_coeffs    (mu_x0, mu_x1, std_sb) vectors derived from std_fwd
    forward_sample   training input: the noisy convex combination q(x_t | x0, x1)
    predict_x0       run the (any) regressor on the bridge state, return the endpoint estimate
    reverse_sample   the shared reverse loop (image OR latent), ddpm / interpolant posterior
"""

from collections import namedtuple

import numpy as np
import torch
from tqdm import tqdm

from operators import Identity
from utils.tensor import unsqueeze_xdim


# The schedule IS a single tensor: std_fwd (sigma) over the discrete bridge steps.
BridgeSchedule = namedtuple("BridgeSchedule", ["std_fwd"])


# ---------------------------------------------------------------------------
# schedule construction
# ---------------------------------------------------------------------------
def _schedule(std_fwd, device):
    return BridgeSchedule(std_fwd=torch.as_tensor(
        np.ascontiguousarray(std_fwd, dtype=np.float64), dtype=torch.float32, device=device))


def brownian(tau, n=1000, device="cpu"):
    """Constant Brownian bridge:  std_fwd(t) = 2 * tau * sqrt(t)  with t = k/n, k = 1..n, so

        max std_fwd = 2 * tau,   max std_sb = tau  (at t = 1/2),   mean = (1 - t) * x0 + t * x1.

    `tau` is an ABSOLUTE std -- scale it with the data intensity. Numerically identical to the old
    build_bridge(bridge_type="brownian", shape="constant") (uniform betas = (2*tau)**2 / n)."""
    t = np.arange(1, int(n) + 1, dtype=np.float64) / int(n)
    return _schedule(2.0 * float(tau) * np.sqrt(t), device)


def from_betas(betas, device="cpu"):
    """Arbitrary schedule from a 1-D `betas` array:  std_fwd = sqrt(cumsum(betas))."""
    betas = np.ascontiguousarray(betas, dtype=np.float64)
    return _schedule(np.sqrt(np.cumsum(betas)), device)


def i2sb_betas(n=1000, beta_max=0.3):
    """The faithful I2SB paper betas: a quadratic ramp (linspace in sqrt-space, squared) with
    linear_start = 1e-4 and linear_end = beta_max / n, first half mirrored. Feed to `from_betas`
    for EXACT paper parity (this is the only knob that is not the closed-form Brownian bridge)."""
    n = int(n)
    b = np.linspace((1e-4) ** 0.5, (beta_max / n) ** 0.5, n, dtype=np.float64) ** 2
    return np.concatenate([b[: n // 2], np.flip(b[: n // 2])])


def build_schedule(kind="brownian", tau=0.19, n_points=1000, beta_max=0.3, device="cpu"):
    """The single config-facing dispatcher.

        kind="brownian"  -> brownian(tau, n_points)                 the constant t(1-t) bridge (default)
        kind="i2sb"      -> from_betas(i2sb_betas(n_points, beta_max))   the FAITHFUL paper schedule
                            (tau is ignored; beta_max sets the peak diffusivity)

    Noise-matched defaults: brownian tau=0.19 and i2sb beta_max=0.3 inject essentially the same
    peak std_sb (0.190 vs 0.188), so the schedule SHAPE is the only difference when comparing them.
    For anything else, build your own betas and call from_betas(betas)."""
    if kind == "brownian":
        return brownian(tau=tau, n=n_points, device=device)
    if kind == "i2sb":
        return from_betas(i2sb_betas(n_points, beta_max), device=device)
    raise ValueError(f"unknown schedule kind {kind!r} (use 'brownian' or 'i2sb')")


def n_steps(sched):
    """Number of discrete bridge steps."""
    return sched.std_fwd.shape[0]


def space_indices(num_steps, count):
    """Evenly spaced integer indices in [0, num_steps-1] inclusive (I2SB util). Sub-samples the
    full grid down to `count` NFE checkpoints."""
    assert count <= num_steps
    frac_stride = 1 if count <= 1 else (num_steps - 1) / (count - 1)
    cur, taken = 0.0, []
    for _ in range(count):
        taken.append(round(cur))
        cur += frac_stride
    return taken


# ---------------------------------------------------------------------------
# derived bridge statistics (everything comes from std_fwd)
# ---------------------------------------------------------------------------
def forward_std(sched, step, xdim=None):
    """sigma at `step` (the noise level the network conditions on). If `xdim` is given and `step`
    is a (B,) index tensor, broadcast to (B, 1, ...) for elementwise use."""
    s = sched.std_fwd[step]
    return s if xdim is None else unsqueeze_xdim(s, xdim)


def bridge_coeffs(sched):
    """Derive (mu_x0, mu_x1, std_sb), each a length-n vector, from std_fwd alone.

    With var_fwd = std_fwd**2 and total = var_fwd[-1], the backward variance is

        var_bwd[k] = total - var_fwd[k-1]        (var_fwd[-1] := 0),

    and the Gaussian product N(x0, var_fwd) * N(x1, var_bwd) gives the bridge mean weights and
    variance:

        mu_x0 = var_bwd / den,  mu_x1 = var_fwd / den,  std_sb = sqrt(var_fwd * var_bwd / den),
        den = var_fwd + var_bwd            (mu_x0 + mu_x1 = 1).

    This reproduces the classic I2SB std_bwd = sqrt(flip(cumsum(flip(betas)))) construction
    exactly (to float32), for any schedule -- Brownian or i2sb."""
    var_fwd = sched.std_fwd ** 2
    total = var_fwd[-1]
    var_fwd_prev = torch.cat([var_fwd.new_zeros(1), var_fwd[:-1]])   # var_fwd shifted right, 0 at k=0
    var_bwd = total - var_fwd_prev
    den = var_fwd + var_bwd
    return var_bwd / den, var_fwd / den, (var_fwd * var_bwd / den).sqrt()


# ---------------------------------------------------------------------------
# forward bridge (training)
# ---------------------------------------------------------------------------
def forward_sample(sched, step, x0, x1, deterministic=False):
    """Training input: draw x_t on the x0 <-> x1 bridge at `step` (the "noisy convex combination"
    the network regresses from). `step` is a (B,) long index tensor.

        x_t = mu_x0(step) * x0 + mu_x1(step) * x1 (+ std_sb(step) * noise)      (mu_x0 + mu_x1 = 1)

    `deterministic` drops the bridge noise (the OT-ODE limit)."""
    assert x0.shape == x1.shape
    xdim = x0.shape[1:]
    mu_x0, mu_x1, std_sb = bridge_coeffs(sched)
    x_t = unsqueeze_xdim(mu_x0[step], xdim) * x0 + unsqueeze_xdim(mu_x1[step], xdim) * x1
    if not deterministic:
        x_t = x_t + unsqueeze_xdim(std_sb[step], xdim) * torch.randn_like(x_t)
    return x_t.detach()


# ---------------------------------------------------------------------------
# regressor call (shared by training and sampling)
# ---------------------------------------------------------------------------
def predict_x0(net, xt, sigma, cond=None, target_channels=1):
    """Run the regressor on the bridge state `xt` (+ optional conditioning) and return its endpoint
    (x0) estimate for the target channel(s). Net-agnostic: works for any denoiser with the
    (input, E, sigma) signature (CDLNet, GroupCDL, ...).

    Conditioning is concatenated onto xt (the net sees C = target_channels + n_cond in/out); we
    keep the first `target_channels` channels as the estimate. E is passed BY KEYWORD: CDLNet.forward
    is (y, E, sigma) but GroupCDL.forward is (y, sigma, E), so a positional Identity() would collide
    with GroupCDL's sigma -- the keyword works for both."""
    net_in = xt if (cond is None or cond.shape[1] == 0) else torch.cat([xt, cond], dim=1)
    out, _ = net(net_in, E=Identity(), sigma=sigma)
    return out[:, :target_channels]


# ---------------------------------------------------------------------------
# reverse bridge (sampling) -- the shared loop, image OR latent
# ---------------------------------------------------------------------------
@torch.no_grad()
def reverse_sample(sched, pred_x0_fn, x1, nfe=None, deterministic=False, posterior="ddpm",
                   clip_denoise=False, log_count=1, verbose=True):
    """Shared reverse loop for every SB variant (image or latent -- it only touches tensors).

    Walks the discrete bridge from the prior end (x1, t = 1) down to the target end (t = 0),
    calling `pred_x0_fn(x_t, step) -> x0_hat` at each visited step and updating the state x_t with
    the chosen posterior (sigma_n = std_fwd[n], sigma_p = std_fwd[n_prev], n_prev < n):

      posterior="ddpm"        (default; I2SB's recursive "moving average" of the EVOLVING state and
                               the endpoint estimate)
          a = (sigma_n**2 - sigma_p**2) / sigma_n**2,   b = sigma_p**2 / sigma_n**2      # a + b = 1
          x_t <- a * x0_hat + b * x_t
          x_t <- x_t + sqrt(sigma_p**2 * (sigma_n**2 - sigma_p**2) / sigma_n**2) * eps    # unless
                                                             # deterministic or the final step

      posterior="interpolant" (the "standard average": a convex combination of the FIXED prior x1
                               and x0_hat, re-placed on the bridge at the new step -- this is just
                               forward_sample(n_prev, x0_hat, x1), so it ignores the running state
                               except through x0_hat)
          x_t <- mu_x0(n_prev) * x0_hat + mu_x1(n_prev) * x1
          x_t <- x_t + std_sb(n_prev) * eps                                              # unless
                                                             # deterministic or the final step

    `nfe` = number of function evaluations (default interval - 1, i.e. every step). `log_count`
    intermediate states are recorded for diagnostics.

    Returns
    -------
    recon    : (B, ...)               final x0 estimate (the state at step 0)
    xs       : (B, log_count, ...)    logged reverse trajectory (newest-first)
    pred_x0s : (B, log_count, ...)    logged per-step x0 predictions (newest-first)
    """
    if posterior not in ("ddpm", "interpolant"):
        raise ValueError(f"posterior {posterior!r} must be 'ddpm' or 'interpolant'")
    device = sched.std_fwd.device
    interval = n_steps(sched)
    nfe = nfe or interval - 1
    assert 0 < nfe < interval, f"need 0 < nfe < {interval}, got {nfe}"

    steps = space_indices(interval, nfe + 1)               # ascending checkpoints 0 .. interval-1
    log_count = min(len(steps) - 1, log_count)
    log_steps = {steps[i] for i in space_indices(len(steps) - 1, log_count)}

    x1 = x1.detach().to(device)
    x_t = x1.clone()                                       # start AT the prior (t = 1)
    mu_x0, mu_x1, std_sb = bridge_coeffs(sched)            # only the interpolant posterior uses these

    xs, pred_x0s = [], []
    pairs = list(zip(steps[1:][::-1], steps[:-1][::-1]))   # (n, n_prev) walking downward
    if verbose:
        pairs = tqdm(pairs, desc=f"SB reverse [{posterior}]", total=len(pairs))

    for n, n_prev in pairs:                                # n_prev < n
        x0_hat = pred_x0_fn(x_t, n)
        if clip_denoise:
            x0_hat = x0_hat.clamp(-1.0, 1.0)

        if posterior == "ddpm":                            # blend the CURRENT state with x0_hat
            sn2, sp2 = sched.std_fwd[n] ** 2, sched.std_fwd[n_prev] ** 2
            var_step = sn2 - sp2                           # variance the forward process added p -> n
            a = var_step / sn2                             # weight on x0_hat (-> 1 as n_prev -> 0)
            b = sp2 / sn2                                  # weight on x_t     (a + b = 1)
            x_t = a * x0_hat + b * x_t
            if not deterministic and n_prev > 0:
                x_t = x_t + (sp2 * var_step / sn2).sqrt() * torch.randn_like(x_t)
        else:                                              # convex combo of the FIXED prior x1 and x0_hat
            x_t = mu_x0[n_prev] * x0_hat + mu_x1[n_prev] * x1
            if not deterministic and n_prev > 0:
                x_t = x_t + std_sb[n_prev] * torch.randn_like(x_t)

        if n_prev in log_steps:
            xs.append(x_t.detach().cpu())
            pred_x0s.append(x0_hat.detach().cpu())

    newest_first = lambda z: torch.flip(torch.stack(z, dim=1), dims=(1,))
    return x_t, newest_first(xs), newest_first(pred_x0s)   # recon = state at step 0
