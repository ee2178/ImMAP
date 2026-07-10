"""
Image-to-Image Schrodinger Bridge (I2SB) -- algorithm functions.

Structured like the ImMAP algorithms in diffusion/immap.py: a set of importable functions
rather than a stateful class. Each takes a `sched` -- a physics.bbridge.BridgeSchedule -- plus
tensors, so schedule design is fully decoupled (see physics/bbridge.py) and you can swap
schedules for training without touching this file.

Endpoints share the 1-channel target (T1ce) domain:
    x0 = target contrast (what we synthesize)     x1 = prior contrast (T1; sampling starts here)
Multi-contrast inputs (FLAIR/T1/T2) enter as conditioning channels via `cdlnet_pred`, not as
bridge endpoints. Real-valued throughout (BraTS magnitude images).

Parameterization: with "x0" the network output IS the x0 estimate (natural for a denoiser like
CDLNet); with "eps" it is the scaled residual of Eq 12 and x0 is recovered via compute_pred_x0.
"""

import torch
from tqdm import tqdm

from operators import Identity
from utils.tensor import unsqueeze_xdim
from physics.bbridge import compute_gaussian_product_coef, space_indices, n_steps


# ---------------------------------------------------------------------------
# forward bridge (training)
# ---------------------------------------------------------------------------
def get_std_fwd(sched, step, xdim=None):
    """Forward std at `step`; broadcast to (B, 1, ...) if `xdim` given. Fed to CDLNet as sigma."""
    std_fwd = sched.std_fwd[step]
    return std_fwd if xdim is None else unsqueeze_xdim(std_fwd, xdim)


def q_sample(sched, step, x0, x1, ot_ode=False):
    """Sample q(x_t | x0, x1), the bridge interpolant (Eq 11). step: (B,) long."""
    assert x0.shape == x1.shape
    batch, *xdim = x0.shape
    mu_x0 = unsqueeze_xdim(sched.mu_x0[step], xdim)
    mu_x1 = unsqueeze_xdim(sched.mu_x1[step], xdim)
    std_sb = unsqueeze_xdim(sched.std_sb[step], xdim)
    xt = mu_x0 * x0 + mu_x1 * x1
    if not ot_ode:
        xt = xt + std_sb * torch.randn_like(xt)
    return xt.detach()


def compute_label(sched, step, x0, xt):
    """Eq 12: the scaled-residual target (eps parameterization)."""
    std_fwd = get_std_fwd(sched, step, xdim=x0.shape[1:])
    return ((xt - x0) / std_fwd).detach()


def compute_pred_x0(sched, step, xt, net_out, clip_denoise=False):
    """Inverse of Eq 12: recover x0 from an eps-parameterized network output."""
    std_fwd = get_std_fwd(sched, step, xdim=xt.shape[1:])
    pred_x0 = xt - std_fwd * net_out
    if clip_denoise:
        pred_x0.clamp_(-1.0, 1.0)
    return pred_x0


# ---------------------------------------------------------------------------
# denoiser call (shared by training and sampling)
# ---------------------------------------------------------------------------
def cdlnet_pred(net, xt, std_fwd_step, cond=None, target_channels=1):
    """Run the real-valued CDLNet on bridge state `xt` (+ optional conditioning), returning its
    image-domain estimate for the target channel(s).

    Conditioning is concatenated onto xt (CDLNet sees C = target_channels + n_cond in/out); we
    keep the first `target_channels` as the T1ce prediction. Requires CDLNet adaptive=True and
    C = target_channels + n_cond.
    """
    net_in = xt if (cond is None or cond.shape[1] == 0) else torch.cat([xt, cond], dim=1)
    x_hat, _ = net(net_in, Identity(), sigma=std_fwd_step)
    return x_hat[:, :target_channels]


# ---------------------------------------------------------------------------
# reverse bridge (sampling)
# ---------------------------------------------------------------------------
def p_posterior(sched, nprev, n, x_n, x0, ot_ode=False):
    """Sample p(x_{nprev} | x_n, x0), the DDPM posterior (Eq 4)."""
    assert nprev < n
    std_n = sched.std_fwd[n]
    std_nprev = sched.std_fwd[nprev]
    std_delta = (std_n ** 2 - std_nprev ** 2).sqrt()
    mu_x0, mu_xn, var = compute_gaussian_product_coef(std_nprev, std_delta)
    xt_prev = mu_x0 * x0 + mu_xn * x_n
    if not ot_ode and nprev > 0:
        xt_prev = xt_prev + var.sqrt() * torch.randn_like(xt_prev)
    return xt_prev


@torch.no_grad()
def ddpm_sampling(sched, steps, pred_x0_fn, x1, ot_ode=False, log_steps=None, verbose=True):
    """Reverse sampling x1 -> x0. `steps`: ascending index list starting at 0.
    `pred_x0_fn(xt, step)` returns the predicted x0 at integer `step`."""
    xt = x1.detach().to(sched.device)
    xs, pred_x0s = [], []
    log_steps = log_steps or steps
    assert steps[0] == log_steps[0] == 0
    steps = steps[::-1]
    pair_steps = zip(steps[1:], steps[:-1])
    if verbose:
        pair_steps = tqdm(pair_steps, desc="I2SB sampling", total=len(steps) - 1)
    for prev_step, step in pair_steps:
        assert prev_step < step, f"{prev_step=}, {step=}"
        pred_x0 = pred_x0_fn(xt, step)
        xt = p_posterior(sched, prev_step, step, xt, pred_x0, ot_ode=ot_ode)
        if prev_step in log_steps:
            pred_x0s.append(pred_x0.detach().cpu())
            xs.append(xt.detach().cpu())
    stack_bwd = lambda z: torch.flip(torch.stack(z, dim=1), dims=(1,))
    return stack_bwd(xs), stack_bwd(pred_x0s)


@torch.no_grad()
def i2sb_sample(net, x1, sched, cond=None, nfe=None, ot_ode=False, parameterization="x0",
                clip_denoise=False, target_channels=1, log_count=1, verbose=True):
    """Synthesize x0 (T1ce) from the prior x1 (T1) + optional conditioning, given a schedule.

    Returns
    -------
    recon    : (B, target_channels, H, W)               final x0 estimate (t=0)
    xs       : (B, log_count, target_channels, H, W)    reverse trajectory (newest-first)
    pred_x0s : (B, log_count, target_channels, H, W)    per-step x0 predictions
    """
    device = sched.device
    interval = n_steps(sched)
    nfe = nfe or interval - 1
    assert 0 < nfe < interval

    steps = space_indices(interval, nfe + 1)
    log_count = min(len(steps) - 1, log_count)
    log_steps = [steps[i] for i in space_indices(len(steps) - 1, log_count)]
    assert log_steps[0] == 0

    x1 = x1.to(device)
    if cond is not None:
        cond = cond.to(device)

    def pred_x0_fn(xt, step):
        step_t = torch.full((xt.shape[0],), step, device=device, dtype=torch.long)
        std_fwd = get_std_fwd(sched, step_t, xdim=xt.shape[1:])
        out = cdlnet_pred(net, xt, std_fwd, cond=cond, target_channels=target_channels)
        if parameterization == "x0":
            return out.clamp(-1.0, 1.0) if clip_denoise else out
        return compute_pred_x0(sched, step_t, xt, out, clip_denoise=clip_denoise)

    xs, pred_x0s = ddpm_sampling(sched, steps, pred_x0_fn, x1, ot_ode=ot_ode,
                                 log_steps=log_steps, verbose=verbose)
    recon = xs[:, 0].to(device)
    return recon, xs, pred_x0s
