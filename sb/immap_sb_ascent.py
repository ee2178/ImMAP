"""
sb/immap_sb_ascent.py -- ImMAP-SB: the Kadkhodaie-Simoncelli / ImMAP annealed ascent ON THE BRIDGE.

Derivation: the "ImMAP-SB Bridge Annealing" note (debiased coordinates; why the earlier ascent
stalled; the bridge-derived noise injection). In short:

THE DEBIASED STATE. With x1 known, the bridge state x_t = mu0 x0 + mu1 x1 + std_sb eps is
equivalent to

    u = (x_t - mu1 x1) / mu0 = x0 + sigma eps,      sigma = std_sb / mu0,

so every bridge time is plain Gaussian denoising of x0 at level sigma, and the regressor
D(x_t, t) ~ E[x0 | u, x1] is a conditional denoiser with an exact Tweedie residual. The bridge
time and the noise level are one variable:

    mu1(sigma) = sigma^2 / (sigma^2 + V),      std_sb = mu0 * sigma,      V = std_fwd[-1]^2

(V = 4 tau^2 for the Brownian bridge; the relation holds for any schedule up to the
one-step offset in sb.base.bridge_coeffs's discrete var_bwd).

THE ITERATION is ImMAP's rule in u, re-placed on the bridge at the new time:

    v_k      = D(x_k, t_k)                                   (or the DC prox of it, `prox`)
    h_k      = h_0 k / (1 + h_0 (k - 1))
    sigma'   = (1 - beta h_k) sigma_k,   t' = mu1(sigma')    the ImMAP contraction
    u_bar    = (1 - h_k) u_k + h_k v_k
    x_{k+1}  = (1 - t') u_bar + t' x1 + gamma_k eps
    gamma_k  = std_sb(t') sqrt(1 - rho_k^2),   rho_k = (1 - h_k) / (1 - beta h_k)

gamma_k tops the noise carried over from x_k (rho_k std_sb(t') in x) up to exactly std_sb(t'),
so with an accurate D every iterate is a sample of the bridge marginal at t'. It is applied in x
as the convex combination c_x x_k + c_v v_k + c_1 x1 (u is never formed: it blows up at t -> 1).

WHY THE EARLIER VERSION STALLED (kept here so nobody reintroduces it): it relaxed x toward the
bridge mean at the CURRENT t, mu_t = (1-t) x0_hat + t x1. On a noise-free iterate on the bridge
with a perfect denoiser, mu_t = x and its estimate t = |x0_hat - x| / |x0_hat - x1| returns t
again -- the step removes noise but never moves along the bridge. And x1 itself (t = 1, zero
noise) is a fixed point of any bridge-consistent estimator, so the loop must START at t_0 < 1,
which is KS's finite sigma_0, not a hack.

I2SB IS A SPECIAL CASE. With the sigma ladder prescribed by the training schedule, r = sigma'/
sigma, h = 1 - r^2 and beta = 1/(1 + r), this is exactly sb.base.reverse_sample(posterior=
"ddpm") (rho = r). tests/test_immap_sb_ascent.py checks it.

`sigma_mode` picks how sigma_k is obtained:
    "open"       sigma_k from the recursion alone (the 'eero' ladder of diffusion/common.py)
    "lagged"     ImMAP's self-estimate sigma_hat^2 = mean |v_k - u_k|^2, u_k formed at t_k
    "quadratic"  the self-consistent t: solve |v - u(t)|^2 / N = sigma(t)^2 for t, taking the
                 root nearest the current t (the other root at x = x1 is the t = 1 fixed point)
The two self-estimates are biased (high at large sigma, where D shrinks toward the mean; they
include D's own error at small sigma), and `monotone` keeps sigma from rising.
"""

import torch
from tqdm import tqdm

from sb.base import bridge_coeffs, forward_std, n_steps, predict_x0
from utils.tensor import unsqueeze_xdim


SIGMA_MODES = ("open", "lagged", "quadratic")


def _per_sample_mean_sq(z):
    return z.abs().pow(2).flatten(1).mean(dim=1)


class _Bridge:
    """sigma <-> bridge time for one schedule, plus the nearest discrete step for the net."""

    def __init__(self, sched):
        self.sched = sched
        self.V = float(sched.std_fwd[-1]) ** 2
        _, mu1, _ = bridge_coeffs(sched)
        self.mu1_tab = mu1.contiguous()               # increasing in the step index
        self.n = n_steps(sched)

    def t_of_sigma(self, sigma):
        s2 = sigma ** 2
        return s2 / (s2 + self.V)

    def sigma_of_t(self, t):
        return (self.V * t / (1.0 - t).clamp_min(1e-12)).sqrt()

    def std_sb(self, t):
        return (1.0 - t) * self.sigma_of_t(t)

    def step(self, t):
        """Nearest discrete step whose mu1 matches t (the net is conditioned on a table value)."""
        hi = torch.searchsorted(self.mu1_tab, t.contiguous()).clamp(0, self.n - 1)
        lo = (hi - 1).clamp(0, self.n - 1)
        take_lo = (self.mu1_tab[lo] - t).abs() < (self.mu1_tab[hi] - t).abs()
        return torch.where(take_lo, lo, hi)


def _quadratic_t(x, v, x1, V, t_prev):
    """Self-consistent bridge time: with a = v - x1, b = x - x1, s = 1 - t, the condition
    |v - u|^2 / N = V t / (1 - t) is

        s^2 (|a|^2 + V N) - s (2 Re<a, b> + V N) + |b|^2 = 0.

    Returns the root nearest 1 - t_prev, as t, clamped to [0, 1)."""
    N = x[0].numel()
    a, b = (v - x1).flatten(1), (x - x1).flatten(1)
    A = a.abs().pow(2).sum(1) + V * N
    Bq = 2.0 * (a.conj() * b).real.sum(1) + V * N
    C = b.abs().pow(2).sum(1)
    disc = (Bq ** 2 - 4.0 * A * C).clamp_min(0.0).sqrt()
    r1, r2 = (Bq - disc) / (2.0 * A), (Bq + disc) / (2.0 * A)
    s_prev = 1.0 - t_prev
    s = torch.where((r1 - s_prev).abs() < (r2 - s_prev).abs(), r1, r2)
    return (1.0 - s).clamp(0.0, 1.0 - 1e-6)


def bridge_ascent_step(x, v, x1, t, t_next, h, beta, std_sb_next, eps=None):
    """One ImMAP step in u, re-placed on the bridge at t_next (all of it in x-coordinates):

        x' = c_x x + c_v v + c_1 x1 + gamma eps,
        c_x = (1 - h)(1 - t')/(1 - t),  c_v = h (1 - t'),  c_1 = t' - c_x t     (sum to 1)
        gamma = std_sb(t') sqrt(1 - rho^2),  rho = (1 - h)/(1 - beta h)

    t, t_next, std_sb_next are (B,) tensors; h, beta scalars; eps None for no noise."""
    xdim = x.shape[1:]
    bc = lambda s: unsqueeze_xdim(s, xdim)
    c_x = (1.0 - h) * (1.0 - t_next) / (1.0 - t)
    c_v = h * (1.0 - t_next)
    c_1 = t_next - c_x * t
    out = bc(c_x) * x + bc(c_v) * v + bc(c_1) * x1
    if eps is not None:
        rho = (1.0 - h) / max(1.0 - beta * h, 1e-12)
        out = out + bc(std_sb_next * max(1.0 - rho ** 2, 0.0) ** 0.5) * eps
    return out


@torch.no_grad()
def immap_sb_ascent(net, x1, sched, cond=None, guide=None, dc=None, prox=None, y=None,
                    a_cond=None, mask=None, t_0=0.99, beta=0.5, h_0=0.05, sigma_min=0.01,
                    sigma_mode="open", monotone=True, deterministic=False, init="regress",
                    target_channels=1, max_iter=1000, log_every=None, verbose=True):
    """Annealed ImMAP ascent on the bridge from the known endpoint x1 toward x0.

    net, cond, guide, dc   the I2SB regressor and what sb.base.predict_x0 forwards to it
    prox, y, a_cond, mask  optional data-consistency prox (sb.immap_sb.ImMAPProx), applied to v_k
                           at the nearest discrete step; y defaults to x1
    t_0                    starting bridge time (< 1); sigma_0 = sigma(t_0)
    beta, h_0              ImMAP's noise-retention ratio and initial step size
    sigma_min              stop once sigma_k <= sigma_min, then denoise once more
    sigma_mode, monotone   see the module docstring
    deterministic          no injected noise (gamma = 0) and no noise in the start
    init                   "regress": x^(0) drawn around D(x1, t ~ 1) ~ E[x0 | x1] at t_0;
                           "x1": x^(0) = x1 (+ std_sb(t_0) noise), i.e. no warm regression

    Returns (x0_hat, info): x0_hat the final denoise D(x_K, t_K); info has "x" (the last
    iterate), "sigma" / "t" (per-iteration lists of the batch max) and "xs" (logged iterates).
    """
    if sigma_mode not in SIGMA_MODES:
        raise ValueError(f"sigma_mode {sigma_mode!r} not in {SIGMA_MODES}")
    if not 0.0 < t_0 < 1.0:
        raise ValueError(f"t_0 must be in (0, 1), got {t_0}: t = 1 is x1 itself, a fixed point")
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"beta must be in [0, 1], got {beta}")
    if init not in ("regress", "x1"):
        raise ValueError(f"init must be 'regress' or 'x1', got {init!r}")

    device = sched.std_fwd.device
    br = _Bridge(sched)
    x1 = x1.to(device)
    y = x1 if y is None else y.to(device)
    cond = None if cond is None else cond.to(device)
    guide = None if guide is None else guide.to(device)
    B, xdim = x1.shape[0], x1.shape[1:]
    bc = lambda s: unsqueeze_xdim(s, xdim)                       # (B,) -> broadcastable

    def denoise(x, t):
        step = br.step(t)
        sigma = forward_std(sched, step, xdim=xdim)
        v = predict_x0(net, x, sigma, cond=cond, target_channels=target_channels,
                       guide=guide, dc=dc, x1=x1)
        if prox is not None:
            # ImMAPProx takes one integer step for the batch; use the noisiest sample's
            v, _ = prox(v, y, int(step.max()), cond=a_cond, mask=mask)
        return v

    def noise(std):
        return 0.0 if deterministic else bc(std) * torch.randn_like(x1)

    # ---- start at t_0 < 1 (KS's finite sigma_0) --------------------------------------------
    t = torch.full((B,), float(t_0), device=device)
    sig = br.sigma_of_t(t)
    if init == "regress":
        v0 = denoise(x1, torch.ones(B, device=device))          # D(x1, t ~ 1) ~ E[x0 | x1]
        x = bc(1.0 - t) * v0 + bc(t) * x1 + noise(br.std_sb(t))
    else:
        x = x1 + noise(br.std_sb(t))

    sigmas, ts, xs = [], [], []
    k = 1
    pbar = tqdm(total=max_iter, desc="ImMAP-SB", disable=not verbose)
    while float(sig.max()) > sigma_min and k <= max_iter:
        v = denoise(x, t)

        if sigma_mode == "lagged":
            u = (x - bc(t) * x1) / bc(1.0 - t)
            sig_hat = _per_sample_mean_sq(v - u).sqrt()
            sig = torch.minimum(sig_hat, sig) if monotone else sig_hat
        elif sigma_mode == "quadratic":
            t_hat = _quadratic_t(x, v, x1, br.V, t)
            t_hat = torch.minimum(t_hat, t) if monotone else t_hat
            # re-express the CURRENT iterate's coordinates at t_hat: the coefficients below
            # divide by 1 - t, which must be the t its u is formed at
            t, sig = t_hat, br.sigma_of_t(t_hat)
        sigmas.append(float(sig.max()))
        ts.append(float(t.max()))

        h = h_0 * k / (1.0 + h_0 * (k - 1))
        sig_next = (1.0 - beta * h) * sig
        t_next = br.t_of_sigma(sig_next)
        eps = None if deterministic else torch.randn_like(x1)
        x = bridge_ascent_step(x, v, x1, t, t_next, h, beta, br.std_sb(t_next), eps)
        t, sig = t_next, sig_next

        if log_every and k % log_every == 0:
            xs.append(x.detach().cpu())
        k += 1
        pbar.update(1)
        pbar.set_postfix(sigma=f"{float(sig.max()):.4f}", t=f"{float(t.max()):.4f}")
    pbar.close()

    x0_hat = denoise(x, t)                                       # one final denoise, as immap2
    return x0_hat, {"x": x, "sigma": sigmas, "t": ts, "xs": xs or [x.detach().cpu()],
                    "iters": k - 1}
