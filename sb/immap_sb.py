"""
sb/immap_sb.py -- ImMAP-SB: I2SB sampling with a learned data-consistency prox.

Structurally ImMAP (denoise, then a data prox, at every step of an annealed schedule) on a
Schrodinger bridge. (The earlier self-paced variant lives in sb/immap_sb_ascent.py.)

At every reverse step the trained regressor's endpoint estimate x_hat = D(z_t, t) ~ E[x0 | z_t]
is replaced, before the ordinary posterior update, by

    x_tilde = argmin_x  1/2 ||x - x_hat||^2 / gamma_t  +  1/2 ||M (y - A(x))||^2 / sigma_A^2

with y = T1 (the bridge's own start, x1), A the frozen learned operator CT1 -> T1
(models/forward_ops.py) and M an optional fidelity mask. Under p(x0 | z_t) ~ N(x_hat, gamma_t I)
and A linearized, x_tilde is E[x0 | z_t, y]: the posterior mean given the measurement too.

Solved by Gauss-Newton: linearize A at x_bar (J = dA/dx there), then CG on

    (lam_t I + J^T M J) delta = J^T M (y - A(x_bar)) - lam_t (x_bar - x_hat),   lam_t = sigma_A^2/gamma_t

and x_bar <- x_bar + delta, `gn_iters` times (x_bar starts at x_hat, so the second term vanishes on
the first pass). lam_t I + J^T M J is symmetric positive definite for lam_t > 0 by construction.

    gamma_t = c * sigma_eff(t)^2,     sigma_eff = std_sb / mu_x0

the uncertainty of x0 implied by the bridge state. So lam_t is LARGE near t = 0 (the denoiser is
trusted, the prox ~ identity) and SMALL toward t = 1 (the data term dominates). c = 0 disables the
prox exactly, reproducing sb.i2sb.i2sb_sample bit for bit.

The noise schedule belongs entirely to sb.base.reverse_sample: the prox only replaces x_hat. It
cannot stop the sampler reaching sigma = 0 at t = 0.

Plug-and-play: the regressor is used as trained and A is frozen; nothing here trains.
"""

import torch

from operators.learned import LinearizedOperator
from sb.base import bridge_coeffs, forward_std, n_steps, predict_x0, reverse_sample
from solvers.cg import batched_cg


class ImMAPProx:
    """The learned data-consistency prox, bound to one bridge schedule.

    A          frozen ForwardOp (CT1 -> T1), eval mode
    sigma_A    A's residual std on held-out data (the ladder checkpoint stores it)
    c          prior-variance scale: gamma_t = c * sigma_eff(t)^2. 0 disables the prox.
    t_max      apply the prox only for t = step / (n - 1) <= t_max
    cg_iters   CG iterations per Gauss-Newton pass;  cg_tol  their relative-residual stop
    gn_iters   Gauss-Newton re-linearizations (1 = linearize once, at x_hat)
    """

    def __init__(self, sched, A, sigma_A, c=1.0, t_max=1.0, cg_iters=10, gn_iters=1, cg_tol=1e-4):
        if not sigma_A > 0:
            raise ValueError(f"sigma_A must be > 0, got {sigma_A}")
        if c < 0:
            raise ValueError(f"c must be >= 0, got {c}")
        self.A, self.sigma_A = A, float(sigma_A)
        self.c, self.t_max = float(c), float(t_max)
        self.cg_iters, self.gn_iters, self.cg_tol = int(cg_iters), int(gn_iters), float(cg_tol)
        mu0, _, std_sb = bridge_coeffs(sched)
        self.sigma_eff = (std_sb / mu0.clamp_min(1e-12)).to(sched.std_fwd.device)
        self.n = n_steps(sched)

    def active(self, step):
        return self.c > 0 and step / max(self.n - 1, 1) <= self.t_max

    def lam(self, step):
        """lam_t = sigma_A^2 / (c * sigma_eff(t)^2)."""
        gamma = self.c * float(self.sigma_eff[step]) ** 2
        return self.sigma_A ** 2 / max(gamma, 1e-30)

    @torch.no_grad()
    def __call__(self, x_hat, y, step, cond=None, mask=None):
        """-> (x_tilde, stats). `step` is the reverse loop's integer step index."""
        if not self.active(step):
            return x_hat, {"step": int(step), "active": False}
        if x_hat.is_complex():
            raise TypeError("ImMAPProx expects a REAL x_hat (use magnitude_output for complex nets)")
        lam = self.lam(step)
        M = 1.0 if mask is None else mask
        x_bar = x_hat
        res0 = None
        for _ in range(self.gn_iters):
            J = LinearizedOperator(self.A, x_bar, cond)
            r = M * (y - J.value)
            if res0 is None:
                res0 = _rms(r, mask)
            b = J.adjoint(r) - lam * (x_bar - x_hat)
            delta = batched_cg(lambda v: lam * v + J.adjoint(M * J.forward(v)), b,
                               tol=self.cg_tol, max_iter=self.cg_iters)
            x_bar = x_bar + delta
        res1 = _rms(M * (y - self.A(x_bar, cond)), mask)
        return x_bar, {"step": int(step), "active": True, "lam": lam,
                       "res_before": res0, "res_after": res1,
                       "delta_rms": _rms(x_bar - x_hat, mask)}


def _rms(t, mask=None):
    if mask is None:
        return float(t.pow(2).mean().sqrt())
    return float((t.pow(2) * mask).sum().div(mask.sum().clamp_min(1)).sqrt())


@torch.no_grad()
def immap_sb(net, x1, sched, prox, cond=None, a_cond=None, mask=None, nfe=None,
                     deterministic=False, posterior="ddpm", clip_denoise=False, target_channels=1,
                     log_count=1, verbose=True, guide=None):
    """sb.i2sb.i2sb_sample with the data prox between the regressor and the posterior update.

    x1 is T1: the bridge's start AND the measurement y. `cond` goes to the regressor (its own
    conditioning channels), `a_cond` to A (its side information; None for a CT1-only A). `mask`
    is the fidelity region M, or None for the whole frame.

    Returns (recon, xs, pred_x0s, stats); `pred_x0s` logs x_tilde (the prox output), and `stats`
    has one dict per visited step.
    """
    if target_channels != 1:
        raise ValueError("the DC prox is defined for a single target channel")
    device = sched.std_fwd.device
    x1 = x1.to(device)
    cond = None if cond is None else cond.to(device)
    a_cond = None if a_cond is None else a_cond.to(device)
    mask = None if mask is None else mask.to(device)
    guide = None if guide is None else guide.to(device)
    stats = []

    def pred_x0_fn(x_t, step):
        step_t = torch.full((x_t.shape[0],), step, device=device, dtype=torch.long)
        sigma = forward_std(sched, step_t, xdim=x_t.shape[1:])
        x_hat = predict_x0(net, x_t, sigma, cond=cond, target_channels=target_channels,
                           guide=guide)
        x_tilde, st = prox(x_hat, x1, step, cond=a_cond, mask=mask)
        stats.append(st)
        return x_tilde

    recon, xs, pred_x0s = reverse_sample(sched, pred_x0_fn, x1, nfe=nfe,
                                         deterministic=deterministic, posterior=posterior,
                                         clip_denoise=clip_denoise, log_count=log_count,
                                         verbose=verbose)
    return recon, xs, pred_x0s, stats
