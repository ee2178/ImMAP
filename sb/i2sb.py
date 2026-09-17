"""
sb/i2sb.py -- image-domain I2SB sampling.

    x0 = target contrast (T1ce; what we synthesize)     x1 = prior contrast (sampling starts here)

Multi-contrast inputs (FLAIR / T1 / T2) enter as conditioning channels via `predict_x0`, not as
bridge endpoints. The reverse loop itself lives in sb.base.reverse_sample; here we only build the
x0-predictor and hand it over. Real-valued throughout (BraTS magnitude images).
"""

import torch

from sb.base import reverse_sample, predict_x0, forward_std


@torch.no_grad()
def i2sb_sample(net, x1, sched, cond=None, nfe=None, deterministic=False, posterior="ddpm",
                clip_denoise=False, target_channels=1, log_count=1, verbose=True, guide=None,
                dc=None):
    """Synthesize x0 (T1ce) from the prior x1 (+ optional conditioning) along the reverse bridge.

    `posterior` selects the update rule ("ddpm" = I2SB recursive posterior, "interpolant" = convex
    combination of x1 and x0_hat). Returns (recon, xs, pred_x0s); see sb.base.reverse_sample.

    `guide` is held FIXED along the whole reverse trajectory: it is side information about the
    patient, not part of the bridge state, so every step sees the same guides.
    """
    device = sched.std_fwd.device
    x1 = x1.to(device)
    cond = None if cond is None else cond.to(device)
    guide = None if guide is None else guide.to(device)

    def pred_x0_fn(x_t, step):
        step_t = torch.full((x_t.shape[0],), step, device=device, dtype=torch.long)
        sigma = forward_std(sched, step_t, xdim=x_t.shape[1:])
        # x1 and cond stay fixed along the trajectory, like the guide; with learned DC the
        # operator is re-bound at every step because its bridge coefficients change.
        return predict_x0(net, x_t, sigma, cond=cond, target_channels=target_channels,
                          guide=guide, dc=dc, x1=x1)

    return reverse_sample(sched, pred_x0_fn, x1, nfe=nfe, deterministic=deterministic,
                          posterior=posterior, clip_denoise=clip_denoise,
                          log_count=log_count, verbose=verbose)
