"""
sb/i2sb_recon.py -- I2SB for accelerated MRI RECONSTRUCTION.

    x0 = target: the fully-sampled coil-combined image (complex, (B, 1, H, W))
    x1 = bridge start, built from the measurement y alone:
             "adjoint"   x1 = E^H y, the zero-filled SENSE adjoint (the default; I2SB's
                         restoration setup -- the degraded image IS the prior endpoint)
             "guide"     x1 = the GUIDE contrast itself (the side scan, e.g. PD for a PDFS
                         target, coil-combined under the operator's maps, undersampled per
                         guide_R): I2SB-Inversion's setup (Wang et al., arXiv 2411.14269), a
                         contrast translation. Needs a paired loader; both contrasts are scaled
                         to fg p95 ~ 1 (per-contrast scale_fac), so the pair is comparable.
             "acs"       x1 = E^H (A y): only the CONTIGUOUS fully-sampled block of phase-encode
                         lines around the k-space centre, zero-padded -- an alias-free low-pass
                         image. The bridge then DEBLURS (adds structure, which the guide can
                         supply) instead of de-aliasing: a uniform-R zero-filled image is
                         coherently aliased, and the bridge needs a large tau just to cover that
                         floor. The outer lines are still in y, reaching the net through (y, E)
                         and the sampler's data consistency. Its width is the MASK's centre block
                         (center_frac / acs_lines): ~15 lines at knee R8 center_frac 0.04, so a
                         sharper start needs a wider centre in the mask, not a change here.

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

Sampling-time data consistency (I2SB-Recon, Wang et al., arXiv 2411.14269, Alg. 2): after every
x0 prediction and BEFORE the posterior step, x0_hat is replaced by

    x0_bar = argmin_x ||y - E x||^2 + lam ||x - x0_hat||^2
           = CG on (E^H E + lam I) x = E^H y + lam x0_hat,   warm-started at x0_hat

with a FIXED number of iterations (their code: lam = 7e-5, 7 iterations, every step including
the last). Training is untouched -- any trained regressor can be sampled with it. lam is
scale-free: both terms scale with intensity^2, and E (ortho FFT, unit-norm maps) has
||E^H E|| <= 1, as in their implementation. Small lam + few warm-started iterations is a partial
pull toward the data, not an exact solve: CG-SENSE at R8 amplifies the measurement noise if run
to convergence with lam -> 0, so `dc_iters` is the effective regularizer as much as lam.
"""

import torch

from physics.object_mask import otsu_threshold
from sb.base import _per_sample, forward_std, n_steps, reverse_sample, space_indices
from solvers.cg import batched_cg


X1_SOURCES = ("adjoint", "acs", "guide")
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


def _sampled_lines(y, E):
    """(N, W) bool: which phase-encode lines (last axis) are sampled, from E's own k-space mask
    (operators.Mask, OP_KIND "kspace_diag"), else from where y is non-zero. N = 1 for a mask
    shared by the batch, else the batch size."""
    m = None
    for op in getattr(E, "ops", [E]):
        if getattr(op, "OP_KIND", None) == "kspace_diag" and torch.is_tensor(op.mask):
            m = op.mask
            break
    if m is None or m.shape[-1] != y.shape[-1]:
        m = y
    m = m.abs() > 0 if torch.is_complex(m) or m.dtype != torch.bool else m
    while m.dim() < 3:
        m = m.unsqueeze(0)
    m = m.reshape(-1, *m.shape[-2:]) if m.dim() == 3 else m.reshape(m.shape[0], -1, m.shape[-1])
    return m.any(dim=-2)


def acs_lines_mask(y, E):
    """(N, 1, 1, W) bool: the contiguous run of sampled lines containing the centre line W // 2
    (where ifftc puts DC) -- the measurement's fully-sampled calibration block."""
    lines = _sampled_lines(y, E).cpu()
    W = lines.shape[-1]
    c = W // 2
    out = torch.zeros_like(lines)
    for i, row in enumerate(lines):
        if not bool(row[c]):
            raise ValueError("x1_source='acs': the centre line is not sampled, so the mask has "
                             "no calibration block to build a low-pass start from")
        lo, hi = c, c
        while lo > 0 and bool(row[lo - 1]):
            lo -= 1
        while hi < W - 1 and bool(row[hi + 1]):
            hi += 1
        out[i, lo:hi + 1] = True
    return out[:, None, None, :].to(y.device)


def bridge_start(y, E, source="adjoint", guide_image=None):
    """The prior endpoint x1. Deterministic in (y, guide), so it is the same at train and test
    time. "guide" needs `guide_image`: the side (other-contrast) scan's coil combination under
    E's maps, (B, 1, H, W) -- training/i2sb_recon.py::_measure builds it."""
    if source == "guide":
        if guide_image is None:
            raise ValueError("x1_source='guide' needs the guide image (a paired loader's side "
                             "scan, combined under the operator's maps)")
        if guide_image.shape != (y.shape[0], 1, *y.shape[-2:]):
            raise ValueError(f"guide image {tuple(guide_image.shape)} is not (B, 1, H, W) on "
                             f"the measured grid {tuple(y.shape[-2:])}")
        return guide_image
    if source == "adjoint":
        return E.adjoint(y)
    if source == "acs":
        return E.adjoint(y * acs_lines_mask(y, E))
    raise ValueError(f"x1_source {source!r} not supported; use one of {X1_SOURCES}")


def predict_x0_recon(net, x_t, sigma, x1=None, y=None, E=None, sigma_n=None, guide=None):
    """Run a recon bridge regressor; returns the complex x0 estimate. `guide` (side
    information, guided regressors only) is passed only when there is one."""
    kw = {} if guide is None else {"guide": guide}
    out, _ = net(x_t, sigma=sigma, x1=x1, y=y, E=E, sigma_n=sigma_n, **kw)
    return out


DC_LAM, DC_ITERS = 7e-5, 7          # I2SB-Inversion's sample.py defaults (cg_weight, cg_iter)


def cg_data_consistency(x0_hat, y, E, lam=DC_LAM, iters=DC_ITERS):
    """argmin_x ||y - E x||^2 + lam ||x - x0_hat||^2 by `iters` CG steps from x0_hat.

    Batched CG (per-slice step sizes; their reference code uses one global alpha for the
    batch), no early stop: the iteration count is part of the method."""
    if iters <= 0:
        return x0_hat
    lam = float(lam)
    b = E.adjoint(y) + lam * x0_hat
    return batched_cg(lambda v: E.adjoint(E.forward(v)) + lam * v, b, x0=x0_hat,
                      tol=0.0, max_iter=int(iters), stop="rhs")


@torch.no_grad()
def i2sb_recon_sample(net, y, E, sched, sigma_n=None, x1=None, x1_source="adjoint", nfe=None,
                      deterministic=False, posterior="ddpm", log_count=1, verbose=True,
                      guide=None, noise_scale=None, dc_iters=0, dc_lam=DC_LAM,
                      support=None, support_state=False, trace=None):
    """Reconstruct x0 from (y, E) along the reverse bridge.

    `x1` defaults to bridge_start(y, E, x1_source); pass it to reuse one already built. y, E,
    sigma_n and the guide (side information) are held fixed along the whole trajectory.
    `noise_scale` is the per-sample temperature multiplier (bridge_noise_scale). `dc_iters` > 0
    applies cg_data_consistency(x0_hat, y, E, dc_lam, dc_iters) to every prediction before the
    posterior step (I2SB-Recon; module doc); 0 is the plain sampler.

    `support` (B, 1, H, W), bool or 0/1: a HARD SUPPORT. Every x0 prediction is multiplied by
    it after the data-consistency step, and so is the returned recon (the background of both is
    exactly 0) -- and, with `support_state`, the bridge state
    too after every update (the trajectory's background cannot drift; the net then sees states
    unlike its training ones outside the support). online_support() builds one from the
    operator's own ESPIRiT eigenvalues. `trace` (a list): one dict per function evaluation is
    appended, {"step", "x_t" (the state the net saw), "x0_pre" (its prediction), "x0_dc" (after
    data consistency), "x0_post" (and the support)}, on the CPU. Returns
    (recon, xs, pred_x0s), as sb.base.reverse_sample (no clip_denoise: a complex image has no
    [-1, 1] box)."""
    device = sched.std_fwd.device
    x1 = bridge_start(y, E, x1_source) if x1 is None else x1
    x1 = x1.to(device)

    def pred_x0_fn(x_t, step):
        step_t = torch.full((x_t.shape[0],), step, device=device, dtype=torch.long)
        sigma = forward_std(sched, step_t, xdim=x_t.shape[1:])
        x0_hat = predict_x0_recon(net, x_t, sigma, x1=x1, y=y, E=E, sigma_n=sigma_n,
                                  guide=guide)
        x0_dc = cg_data_consistency(x0_hat, y, E, dc_lam, dc_iters)
        x0 = x0_dc if sup is None else x0_dc * sup
        if trace is not None:
            trace.append(dict(step=int(step), x_t=x_t.detach().cpu(),
                              x0_pre=x0_hat.detach().cpu(), x0_dc=x0_dc.detach().cpu(),
                              x0_post=x0.detach().cpu()))
        return x0

    sup = None if support is None else support.to(device=device, dtype=x1.real.dtype)
    post = (lambda x: x * sup) if (sup is not None and support_state) else None
    recon, xs, preds = reverse_sample(
        sched, pred_x0_fn, x1 if post is None else post(x1), nfe=nfe,
        deterministic=deterministic, posterior=posterior, clip_denoise=False,
        log_count=log_count, verbose=verbose, noise_scale=noise_scale, post_step=post)
    if sup is not None:
        # the last ddpm step still keeps std_fwd[0]^2 / std_fwd[n]^2 of the previous state
        # (std_fwd[0] > 0), so the recon is not exactly the last prediction: mask it too
        recon = recon * sup
    return recon, xs, preds


@torch.no_grad()
def bridge_inversion(net, x, sched, y=None, E=None, sigma_n=None, x1=None, guide=None,
                     n_steps_inv=None, verbose=False):
    """I2SB-Inversion's deterministic inversion (github.com/zhyjSIAT/I2SB-Inversion,
    i2sb/diffusion.py::ddpm_inverse_sampling, second loop + ddim_inversion): walk a
    reconstruction back up the bridge, ONE index at a time, with no noise and no data consistency,

        x_{i+1} = x_i - (s_{i+1}^2 - s_i^2) / (s_i^2 + 1e-8) * (x_i - x0_hat(x_i, i)),

    s = std_fwd, over the indices 0, 1, ..., n_steps_inv - 1 (their
    `space_indices(inver_step, inver_step)`; inver_step defaults to the full interval, so the
    state lands at the last index, which phase 3 starts from as if it were x1).

    The SIGN is their code's. Their paper's Eq. 20 reads with a plus (x_{i+1} - x0 grows like the
    bridge mean toward x1); the code moves each step toward the current prediction instead
    (comment there: "inversion direction = -eps_net"), so the returned state stays close to a
    reconstruction rather than becoming a guide-like image. The published numbers come from the
    code, so this is what the baseline does."""
    device = sched.std_fwd.device
    n = int(sched.std_fwd.shape[0]) if n_steps_inv is None else int(n_steps_inv)
    s2 = sched.std_fwd.to(device).double() ** 2
    x = x.to(device)
    for i in range(n - 1):
        step_t = torch.full((x.shape[0],), i, device=device, dtype=torch.long)
        sigma = forward_std(sched, step_t, xdim=x.shape[1:])
        x0_hat = predict_x0_recon(net, x, sigma, x1=x1, y=y, E=E, sigma_n=sigma_n, guide=guide)
        coef = float((s2[i + 1] - s2[i]) / (s2[i] + 1e-8))
        x = x - coef * (x - x0_hat)
        if verbose and (i + 1) % 100 == 0:
            print(f"  inversion {i + 1}/{n - 1}")
    return x


@torch.no_grad()
def i2sb_inversion_sample(net, y, E, sched, x1, sigma_n=None, nfe=80, dc_iters=DC_ITERS,
                          dc_lam=DC_LAM, n_steps_inv=None, guide=None, noise_scale=None,
                          posterior="ddpm", verbose=False):
    """The full I2SB-Inversion reconstruction (their sample.py, sampling_method="inversion"):

        1. I2SB-Recon: the stochastic reverse bridge from x1 (the guide contrast) with CG data
           consistency on every x0 prediction (nfe steps)                     -> recon_1
        2. inversion: bridge_inversion(recon_1), deterministic, no CG       -> b_hat
        3. I2SB-Recon again, from b_hat in place of x1, CG on every step     -> recon

    No final hard data consistency (theirs is commented out) and no support mask: the output is
    the last CG-corrected prediction, as reverse_sample's last step returns it. The net is called
    with x1 = the ORIGINAL start throughout (a net with cond_x1=False, theirs, ignores it).
    -> (recon, {"recon_1": ..., "b_hat": ...})."""
    def recon_from(start):
        def pred_x0_fn(x_t, step):
            step_t = torch.full((x_t.shape[0],), step, device=x_t.device, dtype=torch.long)
            sigma = forward_std(sched, step_t, xdim=x_t.shape[1:])
            x0_hat = predict_x0_recon(net, x_t, sigma, x1=x1, y=y, E=E, sigma_n=sigma_n,
                                      guide=guide)
            return cg_data_consistency(x0_hat, y, E, dc_lam, dc_iters)
        return reverse_sample(sched, pred_x0_fn, start, nfe=nfe, deterministic=False,
                              posterior=posterior, clip_denoise=False, log_count=1,
                              verbose=verbose, noise_scale=noise_scale)[0]

    x1 = x1.to(sched.std_fwd.device)
    recon_1 = recon_from(x1)
    b_hat = bridge_inversion(net, recon_1, sched, y=y, E=E, sigma_n=sigma_n, x1=x1, guide=guide,
                             n_steps_inv=n_steps_inv, verbose=verbose)
    recon = recon_from(b_hat)
    return recon, {"recon_1": recon_1, "b_hat": b_hat}


STEP_SIZES = ("i2sb", "ode")


@torch.no_grad()
def debiased_ascent(net, x1, sched, y=None, E=None, sigma_n=None, guide=None, nfe=None,
                    h="i2sb", start="bridge", start_step=None, u0=None, dc_iters=0,
                    dc_lam=DC_LAM, noise_scale=None, trace=None):
    """The ImMAP ascent run in the DEBIASED coordinate u = x0 + sigma eps -- no bridge
    reparameterisation of the steps; the bridge regressor is only how E[x0 | u, x1(, y)] is
    evaluated (notes/immap_i2sb/pmi_guided_ascent.tex, "Worked case").

    Coordinates. At a ladder step n the regressor is called on the bridge state that u
    corresponds to,

        x_n = (1 - m_n) u + m_n x1,   m_n = std_fwd[n]^2 / std_fwd[-1]^2,
        sigma_n^2 = std_fwd[-1]^2 m_n / (1 - m_n)              (the level of the noise in u)

    (the continuous t = sigma_f^2 / V; the map I2SB's ddpm posterior implies). u is held, never
    formed from x, so nothing divides by 1 - m.

    Step, from rung n to the next rung p (r = sigma_p / sigma_n):

        v = D(u)                     regressor at x_n (+ CG data step if dc_iters > 0)
        u <- (1 - h) u + h v + gamma z,     gamma^2 = sigma_p^2 - (1 - h)^2 sigma_n^2

    i.e. gradient ascent on log p_sigma(u | x1(, y)) with step size h sigma_n^2, and the injection
    that leaves u a noisy target at the next level exactly (for an exact denoiser). `h`:
        "i2sb"   h = 1 - r^2, the variance removed -- with start="bridge" this is
                 reverse_sample(posterior="ddpm") iterate for iterate (x_n = (1 - m_n) u_n + m_n x1)
        "ode"    h = 1 - r, gamma = 0: the probability-flow Euler step (deterministic)
        a float or callable(k, r) -> h, in [1 - r, 1]
    As in reverse_sample, nothing is injected on the step into the last rung (step 0).

    Start. "bridge": the ladder begins at the last step (sigma = inf, m = 1), the first call sees
    x1 itself and the first step (r = 0, h = 1) is the warm start E[x0 | x1(, y)] + sigma_1 z.
    "noise": begin at the finite rung `start_step` from u0 (default pure noise, sigma z) -- the
    ascent recovers x0 from random noise; as start_step -> the top it forgets u0.

    The ladder is reverse_sample's: space_indices(n, nfe + 1), walked downward (with "noise", the
    rungs above start_step dropped). Returns (x, v): the final state mapped to x at step 0 (what
    reverse_sample returns) and the last estimate v. `trace` (a list) gets one dict per step,
    {"step", "u", "x", "v"} after the update, on the CPU."""
    device = sched.std_fwd.device
    interval = n_steps(sched)
    nfe = nfe or interval - 1
    steps = space_indices(interval, nfe + 1)[::-1]                 # descending, top first
    if start == "noise":
        if start_step is None:
            raise ValueError("start='noise' needs start_step, the rung to start from")
        steps = [int(start_step)] + [s for s in steps if s < int(start_step)]
    elif start != "bridge":
        raise ValueError(f"start must be 'bridge' or 'noise', got {start!r}")
    total = sched.std_fwd[-1].double() ** 2
    m = (sched.std_fwd.double() ** 2 / total).to(device)

    def level(n):                                              # sigma_n; inf at the top
        return float("inf") if float(m[n]) >= 1.0 else float((total * m[n] / (1 - m[n])).sqrt())

    x1 = x1.to(device)
    kappa = 1.0 if noise_scale is None else _per_sample(noise_scale, x1.shape[0], x1.shape[1:],
                                                        x1.real)
    n0 = steps[0]
    if start == "noise":
        if not level(n0) < float("inf"):
            raise ValueError("start='noise' needs a finite start level (start_step below the top)")
        u = level(n0) * torch.randn_like(x1) if u0 is None else u0.to(device)
    else:
        u = None                                               # sigma = inf: no u yet
    v = None
    for k, (n, p) in enumerate(zip(steps[:-1], steps[1:])):
        x_n = x1 if u is None else (1 - float(m[n])) * u + float(m[n]) * x1
        step_t = torch.full((x1.shape[0],), n, device=device, dtype=torch.long)
        sigma = forward_std(sched, step_t, xdim=x1.shape[1:])
        v = predict_x0_recon(net, x_n, sigma, x1=x1, y=y, E=E, sigma_n=sigma_n, guide=guide)
        if dc_iters:
            v = cg_data_consistency(v, y, E, dc_lam, dc_iters)
        s_n, s_p = level(n), level(p)
        r = 0.0 if s_n == float("inf") else s_p / s_n
        if h == "i2sb":
            hk = 1.0 - r * r
        elif h == "ode":
            hk = 1.0 - r
        else:
            hk = float(h(k, r) if callable(h) else h)
            if not (1.0 - r - 1e-9 <= hk <= 1.0 + 1e-9):
                raise ValueError(f"step size h = {hk} outside [1 - r, 1] = [{1 - r:.4f}, 1] at "
                                 f"rung {n} -> {p}: the injection would need negative variance")
        if u is None:
            if hk != 1.0:
                raise ValueError("the first step from sigma = inf must be h = 1")
            u = torch.zeros_like(v)
        # noise left in u after the step: (1 - h) sigma_n (none at h = 1, also from sigma = inf);
        # the injection tops it up to sigma_p. "ode" injects nothing by construction (from
        # sigma = inf its first step lands on the estimate, as the probability-flow limit in x).
        left = 0.0 if hk == 1.0 else (1.0 - hk) * s_n
        g2 = 0.0 if (p == 0 or h == "ode") else max(s_p ** 2 - left ** 2, 0.0)
        u = (1.0 - hk) * u + hk * v
        if g2 > 0.0:
            u = u + kappa * g2 ** 0.5 * torch.randn_like(u)
        if trace is not None:
            trace.append(dict(step=int(p), u=u.detach().cpu(), v=v.detach().cpu(),
                              x=((1 - float(m[p])) * u + float(m[p]) * x1).detach().cpu()))
    x = (1 - float(m[steps[-1]])) * u + float(m[steps[-1]]) * x1
    return x, v


def online_support(y, mask, acs_lines, thresh=0.9, **kws):
    """(B, 1, H, W) bool: where the OPERATOR's own ESPIRiT top eigenvalue exceeds `thresh` -- the
    hard support of online maps estimated as the operator's are (the same measurement, ACS and
    kernel settings), whatever thresh_eig the operator itself uses (the configs keep 0, soft
    maps). Test-time computable: it reads only the measurement. One extra ESPIRiT call."""
    from physics.online_smaps import online_smaps
    kws = {k: v for k, v in kws.items() if k != "thresh_eig"}
    _, lam = online_smaps(y, mask, acs_lines=acs_lines, thresh_eig=0.0, return_lam=True, **kws)
    return (lam > float(thresh))[:, None]
