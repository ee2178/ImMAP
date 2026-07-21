"""
sb/immap_sb.py -- ImMAP-SB: self-paced annealed Tweedie ascent (ImMAP counterpart to i2sb_sample).

STATUS: WORK IN PROGRESS. This is relocated as-is from the old diffusion/i2sb.py so it is not lost
during the sb/ refactor; it has only been mechanically ported to the sb.base helpers (and pinned to
the x0 parameterization, now the only one). The algorithm itself is still being iterated on and is
NOT part of the cleanup's polish scope -- expect it to change.

It anneals toward argmax_x log p(x | y) with an I2SB regressor r(x_t, t, y) ~ E[x0 | x_t, y]. Unlike
i2sb_sample's fixed discrete reverse schedule, the bridge level t is SELF-ESTIMATED each step and the
loop runs until t < t_L (cf. the self-estimated sigma_t in diffusion/immap.py:immap2).
"""

import torch
from tqdm import tqdm

from sb.base import n_steps, forward_std, predict_x0, bridge_coeffs
from utils.tensor import unsqueeze_xdim


@torch.no_grad()
def immap_sb(net, y, sched, cond=None, tau=0.19, beta=0.05, t_L=0.01, h_0=0.01,
             init_t=0.99, monotone=False, warm_start=False, warm_start_noise=True,
             target_channels=1, max_iter=1000, log_every=None, verbose=True):
    """Per iteration (the provided algorithm):

        x_hat0 = r(x_k, t, y)                                  # predicted endpoint
        t      = ||x_hat0 - x_k|| / ||x_hat0 - y||             # self-estimated bridge level
        mu     = (1 - t) x_hat0 + t y                          # predicted bridge mean
        h_k    = h_0 k / (1 + h_0 (k - 1))
        g^2    = t(1-t) tau ((1 - beta h_k)^2 - (1 - h_k)^2)   # noise injection
        x_{k+1}= x_k + h_k (mu - x_k) + g * eps                # Tweedie ascent

    Geometry is the t(1-t) (Brownian) bridge, so this is most consistent with a net trained on the
    constant Brownian schedule; a net trained on the i2sb schedule still works (the regressor is just
    a denoiser), with a mild geometric mismatch in the annealing.

    `sched` is used only to map the continuous t to the sigma the net was conditioned on (nearest
    discrete step). Pass the SAME schedule the network was trained on.

    Caveats
    -------
    * The literal start x_1 = y is a STATIONARY fixed point (t=1 -> mu=y, g=0 -> no update). Two
      escapes: (a) `init_t` (< 1) seeds the first t but leaves x at y, so a tiny h_0 step can still
      let noise bounce t back to 1; (b) `warm_start=True` (recommended) jumps x directly to the
      bridge interpolant (1-init_t) r(y,t=1) + init_t y, so the self-estimated t equals init_t and
      x never sits at y. `warm_start_noise` adds std_sb(init_t) so the regressor input is
      in-distribution. `monotone=True` additionally forces t non-increasing.

    Returns
    -------
    x_star : (B, target_channels, H, W)   annealed-ascent estimate
    ts     : list[float]                  max self-estimated t per iteration (diagnostic)
    xs     : list[Tensor]                 logged iterates on cpu if log_every else [x_star]
    """
    device = sched.std_fwd.device
    y = y.to(device)
    if cond is not None:
        cond = cond.to(device)
    N = n_steps(sched)
    _, _, std_sb = bridge_coeffs(sched)                           # bridge-noise std per step

    x = y.clone()                                                 # x_1 <- y
    t = torch.full((x.shape[0],), float(init_t), device=device)   # seed < 1 to bootstrap

    if warm_start:
        # Escape the y-endpoint fixed point by doing the FIRST reverse step I2SB-style: take the
        # (trusted) endpoint regression x_hat0 = r(y, t=1) and jump straight to the bridge
        # interpolant at init_t. Then the self-estimated t is ~init_t (< 1), not 1.
        step1 = torch.full((x.shape[0],), N - 1, device=device, dtype=torch.long)   # t = 1 endpoint
        sig1 = forward_std(sched, step1, xdim=y.shape[1:])
        x0_init = predict_x0(net, y, sig1, cond=cond, target_channels=target_channels)
        tb0 = unsqueeze_xdim(t, y.shape[1:])
        x = (1.0 - tb0) * x0_init + tb0 * y                       # x <- bridge mean at init_t
        if warm_start_noise:
            step_it = (t.clamp(0.0, 1.0) * (N - 1)).round().long()
            x = x + unsqueeze_xdim(std_sb[step_it], y.shape[1:]) * torch.randn_like(x)
    k = 1
    ts, xs = [], []
    pbar = tqdm(total=max_iter, desc="ImMAP-SB", disable=not verbose)

    while float(t.max()) > t_L and k <= max_iter:
        # x_hat0 = r(x_k, t, y): map continuous t -> nearest discrete step -> sigma conditioning
        step = (t.clamp(0.0, 1.0) * (N - 1)).round().long()
        sigma = forward_std(sched, step, xdim=x.shape[1:])
        x_hat0 = predict_x0(net, x, sigma, cond=cond, target_channels=target_channels)

        # self-estimated bridge level (first iter keeps the init_t seed)
        num = (x_hat0 - x).flatten(1).norm(dim=1)
        den = (x_hat0 - y).flatten(1).norm(dim=1).clamp_min(1e-8)
        t_est = (num / den).clamp(0.0, 1.0)
        # Remove this safeguard for testing.
        # if k > 1:
        #     t = torch.minimum(t_est, t) if monotone else t_est

        ts.append(float(t.max()))
        tb = unsqueeze_xdim(t, x.shape[1:])

        mu = (1.0 - tb) * x_hat0 + tb * y                         # predicted bridge mean
        h_k = h_0 * k / (1.0 + h_0 * (k - 1))
        inj = (1.0 - beta * h_k) ** 2 - (1.0 - h_k) ** 2
        gamma = (tb * (1.0 - tb) * tau * max(inj, 0.0)).sqrt()
        x = x + h_k * (mu - x) + gamma * torch.randn_like(x)      # Tweedie ascent

        if log_every and (k % log_every == 0):
            xs.append(x.detach().cpu())
        k += 1
        pbar.update(1)
        pbar.set_postfix(t=f"{float(t.max()):.4f}")

    pbar.close()
    if not xs:
        xs = [x.detach().cpu()]
    return x, ts, xs
