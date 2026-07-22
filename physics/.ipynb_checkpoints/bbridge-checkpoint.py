"""
Brownian-bridge noise schedules for image-to-image Schrodinger bridges (I2SB).

The schedule design lives here so the algorithm functions in diffusion/i2sb.py stay
schedule-agnostic. Build a beta schedule however you like, turn it into the derived bridge
statistics with `bridge_schedule`, and hand the resulting `BridgeSchedule` to those functions.
The paper's symmetric quadratic schedule is provided as `symmetric_bridge_schedule`, but ANY
1-D `betas` array works -- that is the knob for redesigning the schedule for training.

A `BridgeSchedule` is a plain namedtuple (data only, no methods): the forward / backward /
bridge stds and the Gaussian-product interpolation coefficients, all as float32 tensors on
`device`. Everything is derived from `betas` via the cumulative-variance identities (Eq 11).
"""

from collections import namedtuple

import numpy as np
import torch


BridgeSchedule = namedtuple(
    "BridgeSchedule",
    ["betas", "std_fwd", "std_bwd", "std_sb", "mu_x0", "mu_x1", "device"],
)


# ---------------------------------------------------------------------------
# schedule design (betas)
# ---------------------------------------------------------------------------
def make_beta_schedule(n_timestep=1000, linear_start=1e-4, linear_end=2e-2):
    """Quadratic beta ramp (NVIDIA i2sb make_beta_schedule): linspace in sqrt-space, squared."""
    betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return betas.numpy()


def build_symmetric_betas(interval=1000, beta_max=0.3):
    """The paper's symmetric schedule: a quadratic ramp with linear_end = beta_max / interval,
    take the first half, mirror it. Returns a length-`interval` numpy array."""
    betas = make_beta_schedule(n_timestep=interval, linear_end=beta_max / interval)
    return np.concatenate([betas[: interval // 2], np.flip(betas[: interval // 2])])


# ---------------------------------------------------------------------------
# core bridge coefficient math
# ---------------------------------------------------------------------------
def compute_gaussian_product_coef(sigma1, sigma2):
    """Coefficients of N(x | x0, s1^2) * N(x | x1, s2^2) = N(x | c1 x0 + c2 x1, var).
    Works elementwise on numpy arrays or torch tensors."""
    denom = sigma1 ** 2 + sigma2 ** 2
    coef1 = sigma2 ** 2 / denom
    coef2 = sigma1 ** 2 / denom
    var = (sigma1 ** 2 * sigma2 ** 2) / denom
    return coef1, coef2, var


# ---------------------------------------------------------------------------
# schedule -> derived bridge statistics
# ---------------------------------------------------------------------------
def bridge_schedule(betas, device="cpu"):
    """Turn a 1-D `betas` schedule into the derived bridge statistics (Eq 11).

    This is the schedule-design entry point: pass any betas -- build_symmetric_betas, a plain
    linear ramp, or your own array -- and get a BridgeSchedule for diffusion/i2sb.py.
    """
    betas = np.ascontiguousarray(betas, dtype=np.float64)
    std_fwd = np.sqrt(np.cumsum(betas))
    std_bwd = np.sqrt(np.flip(np.cumsum(np.flip(betas))))
    mu_x0, mu_x1, var = compute_gaussian_product_coef(std_fwd, std_bwd)
    std_sb = np.sqrt(var)

    to_t = lambda a: torch.as_tensor(np.ascontiguousarray(a, dtype=np.float64),
                                     dtype=torch.float32, device=device)
    return BridgeSchedule(
        betas=to_t(betas), std_fwd=to_t(std_fwd), std_bwd=to_t(std_bwd),
        std_sb=to_t(std_sb), mu_x0=to_t(mu_x0), mu_x1=to_t(mu_x1), device=device,
    )


def symmetric_bridge_schedule(interval=1000, beta_max=0.3, device="cpu"):
    """Convenience: the paper's symmetric schedule -> BridgeSchedule."""
    return bridge_schedule(build_symmetric_betas(interval, beta_max), device)


def brownian_bridge_schedule(tau, n=1000, device="cpu", shape="constant"):
    """Bridge schedule parameterized by the peak bridge-noise std `tau` and the number of
    points `n` -- the natural single knob (see the module notes).

    The total diffusivity is set to sum(betas) = (2*tau)**2, so:
        max(std_sb)  = tau        (peak stochastic noise injected mid-bridge)
        max(std_fwd) = 2*tau      (the sigma ceiling fed to CDLNet)
    `tau` is an ABSOLUTE std, so scale it with your data intensity (per-contrast scales).

    shape:
        'constant'  -- pure Brownian bridge, beta_k = (2*tau)**2 / n  (simplest; recommended)
        'symmetric' -- the paper's mirrored-quadratic profile, rescaled to the same total
    """
    n = int(n)
    total = (2.0 * float(tau)) ** 2
    if shape == "constant":
        betas = np.full(n, total / n, dtype=np.float64)
    elif shape == "symmetric":
        prof = np.asarray(build_symmetric_betas(n, beta_max=0.3), dtype=np.float64)  # paper curvature
        betas = prof * (total / prof.sum())
    else:
        raise ValueError(f"unknown shape {shape!r} (use 'constant' or 'symmetric')")
    return bridge_schedule(betas, device)


def build_bridge(bridge_type="brownian", n_points=1000, device="cpu",
                 tau=0.19, shape="constant", beta_max=0.3):
    """Config-facing dispatcher: select a bridge schedule by `bridge_type`.

    'brownian' -- tau-parameterized Brownian bridge (peak std_sb = tau); `shape` picks
                  'constant' (the pure t(1-t) bridge) or 'symmetric' (paper profile, rescaled).
    'i2sb'     -- the FAITHFUL I2SB paper schedule (symmetric quadratic betas set by `beta_max`,
                  NOT rescaled). This is the true-I2SB baseline. `n_points` is the paper's interval.

    Both use `n_points` as the number of discrete steps. Note the defaults are noise-matched:
    brownian tau=0.19 has peak std_sb=0.19, and i2sb beta_max=0.3 has peak std_sb=0.188 -- so the
    two bridges inject essentially the same peak noise, isolating the schedule *shape* as the
    only variable when you compare them.
    """
    if bridge_type == "brownian":
        return brownian_bridge_schedule(tau=tau, n=n_points, device=device, shape=shape)
    if bridge_type == "i2sb":
        return symmetric_bridge_schedule(interval=n_points, beta_max=beta_max, device=device)
    raise ValueError(f"unknown bridge_type {bridge_type!r} (use 'brownian' or 'i2sb')")


def n_steps(sched):
    """Number of discrete bridge steps in a schedule."""
    return sched.betas.shape[0]


# ---------------------------------------------------------------------------
# time-grid discretization for sampling
# ---------------------------------------------------------------------------
def space_indices(num_steps, count):
    """Evenly spaced integer indices in [0, num_steps-1] inclusive (I2SB util).
    Sub-samples the full grid down to `count` NFE checkpoints."""
    assert count <= num_steps
    frac_stride = 1 if count <= 1 else (num_steps - 1) / (count - 1)
    cur, taken = 0.0, []
    for _ in range(count):
        taken.append(round(cur))
        cur += frac_stride
    return taken
