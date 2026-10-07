"""
sb/unsb.py -- Unpaired Neural Schrodinger Bridge (UNSB; Kim, Kwon, Kim & Ye, ICLR 2024).

    x1 = source / prior contrast (T1; the chain starts here)     x0 = target contrast (CT1)

Same orientation as the rest of sb/: the source sits at bridge position t = 1 (step n-1) and the
target at t = 0 (step 0). The paper runs its clock the other way (s = 0 source -> s = 1 target), so
s = 1 - t throughout; nothing below uses s except in comments.

WHAT IS THE SAME AS I2SB
    The sampler. At each stage the regressor predicts the target endpoint x0_hat from the current
    state, and the state moves to the next checkpoint with the Gaussian bridge posterior
        x_t' = a * x0_hat + b * x_t + noise,    a = (s_n^2 - s_p^2)/s_n^2,  b = s_p^2/s_n^2.
    That is EXACTLY `reverse_sample(posterior="ddpm")`. On the Brownian schedule it is the paper's
    Eq. 11 (a = (t_{i+1}-t_i)/(1-t_i), noise var = a(1-a)(1-t_i) tau_paper) with
        tau_paper = std_fwd[-1]**2 = (2 * tau)**2.
    `tests/test_unsb.py` pins both facts, so the regressor (SBUnet, CDLNet, GroupCDL, ...) and
    `predict_x0` are reused unchanged and conditioning enters through `cond` as for I2SB.

WHAT IS DIFFERENT: HOW THE REGRESSOR IS TRAINED
    I2SB manufactures x_t from a true pair (forward_sample) and regresses x0. UNSB has no pair, so
    x_t comes from the model's OWN chain, and x0_hat is trained by distribution matching:

        stage j ~ U{0..K-1};  x_t = chain(x1, j stages)  [no grad];  x0_hat = G(x_t, step)
        L_G = lambda_gan * LSGAN(D(x0_hat))                      realism   (the target marginal)
            + lambda_sb  * tau * ( mean|x_t - x0_hat|^2          transport (stay near the state)
                                   - (1 - t) * ET )              entropy   (do not collapse)
            + lambda_nce * PatchNCE(x1, x0_hat)                  structure (same anatomy)

    D is a time-conditioned PatchGAN; ET is a MINE estimate from a second critic E; PatchNCE ties
    learned features of the output to those of the SOURCE at the same location. The nets live in
    models/unsb_nets.py; this file only needs them as callables, so it imports without models/
    (which does not import on torch 1.12).

WHAT MATTERS, IN ORDER
    The discriminator is the core: it is the only term that defines the target contrast, and (with
    `cond_d=True`) the only one that can tie the output to T2 / FLAIR / the prior study. Everything
    else is optional and independently switchable:
        GAN only            lambda_sb = 0, nce = None            -- plain conditional GAN on the chain
        GAN + transport     E = None                             -- no entropy critic, no batch >= 2
        + entropy           E given (batch >= 2)
        + structure         nce = make_nce_fn(...)               -- PatchNCE to the source
    Caution: with no structure term the only thing tying the output to the source is the transport
    penalty (tau-scaled, weak) and the source's presence in the state; a strong D alone will happily
    paint a plausible CT1 that is not THIS patient's.

THE GENERATOR CONTRACT (so any regressor works)
    G is whatever `predict_x0(net, x_t, sigma, cond, ...)` accepts: net(cat[x_t, cond], E=,
    sigma=) -> (x0_hat, _). SBUnet, SBCDLNet, SBGroupCDL and SBGuidedGroupCDL all do. Two rules:
      * the net's schedule (kind/tau/n_points) must equal `sched`, because the SB nets recover the
        step from sigma (`net.assert_schedule_matches(sched)`); the chain only ever visits indices
        of `sched`, so that is exact.
      * the SB-CDL nets debias with mu_1 * x_1, so the SOURCE image must be one of the cond
        channels, at the net's `prior_idx`. That is not an I2SB quirk -- it stays correct here: on
        the Brownian schedule the chain state is x_t = t * x_1 + (1 - t) * (mix of predictions) +
        noise, so x_t - mu_1 x_1 is a mix of target predictions, which is what those nets model.
        (pinned in tests/test_unsb.py). A UNet needs neither; cond is free-form for it.
    Nothing here reads G's internals; PatchNCE has a G-independent extractor (PatchEncoder).

NOTATION
    K = `num_stages` G evaluations per sample. `steps` is the ascending list of K+1 bridge indices
    [0, ..., n-1]; stage j evaluates G at steps[K - j] (j = 0 is the source end) and the state then
    moves to steps[K - j - 1]. steps[0] is the target END POINT and is never evaluated: the output
    is the last x0_hat, not a state (unlike reverse_sample, whose final state is blended with the
    last prediction by a factor b = s_0^2/s_n^2, ~1/n).

DEVIATIONS FROM THE PAPER / OFFICIAL CODE (all deliberate, all small)
    * The entropy critic's negatives are the batch rolled by one (a different sample's pair), not a
      second rollout. Needs batch >= 2; saves a full chain.
    * ET is a proper log-mean-exp MINE estimate (official: logsumexp without -log N). Same gradient
      for G; the critic's offset regulariser then acts on log-mean-exp.
    * No AdaIN style vector z; the bridge noise is the only stochasticity.
    * No identity branch (nce_idt).
    * Squared errors are per-pixel MEANS, as in the official code, so lambda_sb is not comparable
      with a paper that sums.
"""

import math
from collections import namedtuple

import numpy as np
import torch

from sb.base import n_steps, forward_std, predict_x0, space_indices


# One training sample of the chain: the (detached) state x_t entering stage `stage` at bridge index
# `step`, and the grad-carrying endpoint prediction G(x_t).
UNSBState = namedtuple("UNSBState", ["x_t", "x_hat", "step", "stage"])


# ---------------------------------------------------------------------------
# the clock
# ---------------------------------------------------------------------------
def _official_positions(num_stages):
    """The official UNSB clock s_0 = 0 < ... < s_K = 1 (source -> target): the first jump is half
    the bridge, then ever finer toward the target. K = 5 gives [0, .5, .74, .86, .94, 1]."""
    if num_stages == 1:
        return np.array([0.0, 1.0])
    incs = np.array([0.0] + [1.0 / (i + 1) for i in range(num_stages - 1)])
    s = np.cumsum(incs)
    s = s / s[-1]
    s = 0.5 * s[-1] + 0.5 * s
    return np.concatenate([np.zeros(1), s])


def unsb_steps(sched, num_stages=5, grid="uniform"):
    """Ascending bridge indices [steps[0] = 0, ..., steps[K] = n - 1] for a K-stage chain.

        grid="uniform"   the paper's text; identical to the checkpoints reverse_sample(nfe=K) walks
        grid="official"  the released code's clock (see _official_positions)

    The position of a step is t_eff[k] = std_fwd[k]^2 / std_fwd[-1]^2 (= (k+1)/n for the Brownian
    schedule), so the official clock lands on the same bridge fractions under any schedule."""
    K = int(num_stages)
    n = n_steps(sched)
    if K < 1 or K + 1 > n:
        raise ValueError(f"need 1 <= num_stages <= {n - 1}, got {K}")
    if grid == "uniform":
        steps = space_indices(n, K + 1)
    elif grid == "official":
        var = sched.std_fwd.double() ** 2
        t_eff = (var / var[-1]).cpu()
        steps = [int((t_eff - (1.0 - s)).abs().argmin()) for s in _official_positions(K)[::-1]]
    else:
        raise ValueError(f"grid {grid!r} must be 'uniform' or 'official'")
    if len(set(steps)) != K + 1 or any(b <= a for a, b in zip(steps, steps[1:])):
        raise ValueError(f"num_stages={K} does not fit {n} bridge steps on grid {grid!r}: {steps}")
    return steps


def sample_stage(num_stages):
    """Uniform training stage in {0..K-1}. One scalar per batch, as in the official code."""
    return int(torch.randint(int(num_stages), (1,)).item())


# ---------------------------------------------------------------------------
# the bridge step (== reverse_sample's "ddpm" posterior)
# ---------------------------------------------------------------------------
def bridge_step_coeffs(sched, n, n_prev):
    """(a, b, std) of the move from step n down to n_prev < n:
        x_prev = a * x0_hat + b * x_n + std * eps,    a + b = 1.
    Paper Eq. 11 on the Brownian schedule."""
    sn2 = sched.std_fwd[n] ** 2
    sp2 = sched.std_fwd[n_prev] ** 2
    var_step = sn2 - sp2
    return var_step / sn2, sp2 / sn2, (sp2 * var_step / sn2).sqrt()


def bridge_step(sched, x_t, x0_hat, n, n_prev, noise_scale=1.0):
    """Move the state from step n to n_prev. `noise_scale` = 0 drops the bridge noise; there is
    never noise at the target end (n_prev == 0), as in reverse_sample."""
    a, b, std = bridge_step_coeffs(sched, n, n_prev)
    x = a * x0_hat + b * x_t
    if noise_scale and n_prev > 0:
        x = x + float(noise_scale) * std * torch.randn_like(x)
    return x


def make_g_fn(net, sched, x1, cond=None, target_channels=1, guide=None, dc=None):
    """`g_fn(x_t, step) -> x0_hat`: the regressor as the chain sees it. Conditioning, guide and
    learned DC stay fixed along the chain, exactly as in i2sb_sample."""
    device = sched.std_fwd.device

    def g_fn(x_t, step):
        step_t = torch.full((x_t.shape[0],), int(step), device=device, dtype=torch.long)
        sigma = forward_std(sched, step_t, xdim=x_t.shape[1:])
        return predict_x0(net, x_t, sigma, cond=cond, target_channels=target_channels,
                          guide=guide, dc=dc, x1=x1)

    return g_fn


# ---------------------------------------------------------------------------
# chain: training rollout and sampling
# ---------------------------------------------------------------------------
@torch.no_grad()
def unsb_rollout(sched, g_fn, x1, steps, stage, noise_scale=1.0):
    """Run `stage` bridge moves from the source and return (x_t, step): the state that ENTERS
    stage `stage` and its bridge index steps[K - stage]. stage = 0 returns x1 itself.

    This is the on-policy part: the state G is trained on is the one the current G produces."""
    K = len(steps) - 1
    if not 0 <= stage < K:
        raise ValueError(f"stage {stage} outside 0..{K - 1}")
    x_t = x1.detach().clone()
    for j in range(stage):
        n, n_prev = steps[K - j], steps[K - j - 1]
        x_t = bridge_step(sched, x_t, g_fn(x_t, n), n, n_prev, noise_scale=noise_scale)
    return x_t, steps[K - stage]


def unsb_forward(g_fn, sched, x1, steps, stage, noise_scale=1.0):
    """One training sample: no-grad rollout to `stage`, then ONE grad-carrying G call."""
    x_t, n = unsb_rollout(sched, g_fn, x1, steps, stage, noise_scale=noise_scale)
    return UNSBState(x_t=x_t, x_hat=g_fn(x_t, n), step=n, stage=stage)


@torch.no_grad()
def unsb_sample(net, x1, sched, cond=None, num_stages=5, nfe=None, grid="uniform",
                deterministic=False, target_channels=1, guide=None, dc=None):
    """Translate x1 (T1) into the target contrast.

    `num_stages` must equal the K the net was TRAINED with (the net only ever saw states from that
    chain); `nfe <= K` evaluates an evenly spaced subset of those stages, as the official code does.

    Returns (recon, xs, pred_x0s), the layout of reverse_sample:
        recon     the LAST x0_hat (what the discriminator saw in training)
        xs        (B, nfe, ...)  the state ENTERING each G call
        pred_x0s  (B, nfe, ...)  each G call's output
    both newest-first, i.e. index 0 is the target end (the last call) and the array ascends in
    bridge position -- the same ordering trap as reverse_sample (see the sb/ memory note)."""
    device = sched.std_fwd.device
    x1 = x1.to(device)
    cond = None if cond is None else cond.to(device)
    guide = None if guide is None else guide.to(device)

    K = int(num_stages)
    steps = unsb_steps(sched, K, grid)
    evals = steps[:0:-1]                                  # steps[K], ..., steps[1]: G's positions
    nfe = K if nfe is None else int(nfe)
    if not 1 <= nfe <= K:
        raise ValueError(f"need 1 <= nfe <= num_stages={K}, got {nfe}")
    chosen = [evals[j] for j in space_indices(K, nfe)]

    g_fn = make_g_fn(net, sched, x1, cond=cond, target_channels=target_channels,
                     guide=guide, dc=dc)
    noise_scale = 0.0 if deterministic else 1.0
    x_t, xs, hats = x1.clone(), [], []
    for i, n in enumerate(chosen):
        x_hat = g_fn(x_t, n)
        xs.append(x_t.detach().cpu())
        hats.append(x_hat.detach().cpu())
        if i + 1 < len(chosen):
            x_t = bridge_step(sched, x_t, x_hat, n, chosen[i + 1], noise_scale=noise_scale)

    newest_first = lambda z: torch.flip(torch.stack(z, dim=1), dims=(1,))
    return x_hat, newest_first(xs), newest_first(hats)


# ---------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------
def tau_sb(sched):
    """The paper's tau for this schedule: the bridge's TOTAL variance, std_fwd[-1]**2 = (2 tau)**2.
    The paper's 0.01 is tau = 0.05 here."""
    return float(sched.std_fwd[-1]) ** 2


def remaining_fraction(sched, step):
    """The paper's (1 - t_i): the share of the bridge variance still ahead of `step`."""
    var = sched.std_fwd ** 2
    return float(var[step] / var[-1])


def set_requires_grad(nets, flag):
    for net in ([nets] if isinstance(nets, torch.nn.Module) else nets):
        for p in net.parameters():
            p.requires_grad_(flag)


def lsgan_d_loss(d_real, d_fake):
    return 0.5 * (((d_real - 1.0) ** 2).mean() + (d_fake ** 2).mean())


def lsgan_g_loss(d_fake):
    return ((d_fake - 1.0) ** 2).mean()


def _step_vec(state, ref):
    return torch.full((ref.shape[0],), int(state.step), device=ref.device, dtype=torch.long)


def _with_cond(img, cond):
    return img if cond is None else torch.cat([img, cond], dim=1)


def d_loss(D, state, x0_real, cond=None, cond_real=None, cond_d=True):
    """Discriminator loss on the detached prediction vs a REAL target-contrast image.

    `cond_d` is THE conditioning toggle for the discriminator (G's own conditioning is separate, in
    `g_fn`). Build D with the same value: models.unsb_nets.build_critics(..., cond_d=cond_d).

    cond_d=False (or cond=None) -> D sees the image alone: it only learns what a CT1 looks like.
    cond_d=True                 -> D scores (image, cond) pairs. `cond_real` is the conditioning
                  that belongs to `x0_real` (defaults to `cond`, i.e. the same case). This is the
                  only place where "enhancement belongs where T2/FLAIR/the prior study say" can be
                  enforced, and it is only as good as the registration of cond against the real
                  CT1."""
    step = _step_vec(state, x0_real)
    if not cond_d:
        cond = cond_real = None
    fake = _with_cond(state.x_hat.detach(), cond)
    real = _with_cond(x0_real, cond if cond_real is None else cond_real)
    return lsgan_d_loss(D(real, step), D(fake, step))


def _critic_pairs(state):
    a = torch.cat([state.x_t, state.x_hat], dim=1)
    if a.shape[0] < 2:
        raise ValueError("the entropy critic draws its negatives from the rest of the batch, so "
                         "it needs batch size >= 2")
    return a, a.roll(1, dims=0)                           # b = a DIFFERENT sample's pair


def mine_et(E, a, b, step):
    """MINE estimate of H(a) = I(a, a): ET = E(a, a).mean() - log mean exp E(a, b).
    Returns (ET, log_z, joint) where joint = E(a, a).mean()."""
    joint = E(a, step, a).mean()
    marginal = E(a, step, b).flatten()
    log_z = torch.logsumexp(marginal, dim=0) - math.log(marginal.numel())
    return joint - log_z, log_z, joint


def e_loss(E, state):
    """Critic loss: maximise ET, with log_z**2 pinning the critic's free additive offset (the
    official code's `temp + temp**2`)."""
    a, b = _critic_pairs(state._replace(x_hat=state.x_hat.detach()))
    step = _step_vec(state, a)
    _, log_z, joint = mine_et(E, a, b, step)
    return -joint + log_z + log_z ** 2


def g_loss(D, E, state, sched, x1, cond=None, nce=None, cond_d=True,
           lambda_gan=1.0, lambda_sb=1.0, lambda_nce=1.0, tau=None):
    """Generator objective for one sample of the chain. Freeze D and E first
    (`set_requires_grad([D, E], False)`): gradients must reach G, not the critics.

    `cond_d` must equal the value D was built and trained with (see d_loss). `E=None` drops the
    entropy term (transport only); `lambda_sb=0` drops transport too. `nce` is `make_nce_fn(...)`'s
    closure, or None for no structure loss. Returns (total, terms), `terms` the detached parts."""
    zero = state.x_hat.new_zeros(())
    terms = {}

    adv = zero
    if lambda_gan > 0:
        d_cond = cond if cond_d else None
        adv = lsgan_g_loss(D(_with_cond(state.x_hat, d_cond), _step_vec(state, state.x_hat)))
    terms["adv"] = adv

    sb = zero
    transport = ((state.x_t - state.x_hat) ** 2).mean()
    terms["transport"] = transport
    if lambda_sb > 0:
        tau = tau_sb(sched) if tau is None else float(tau)
        entropy = zero
        if E is not None:
            a, b = _critic_pairs(state)
            entropy, _, _ = mine_et(E, a, b, _step_vec(state, a))
        sb = tau * (transport - remaining_fraction(sched, state.step) * entropy)
        terms["entropy"] = entropy
    terms["sb"] = sb

    nce_val = zero
    if lambda_nce > 0 and nce is not None:
        nce_val = nce(x1, state.x_hat)
    terms["nce"] = nce_val

    total = lambda_gan * adv + lambda_sb * sb + lambda_nce * nce_val
    return total, {k: v.detach() for k, v in terms.items()}


def make_nce_fn(extractor, pnce, sched, step=None, cond=None, mask=None):
    """`nce(x_src, x_hat) -> scalar`: features of the output (query) against features of the
    SOURCE (key) at the same locations.

    `extractor(x, cond, sigma) -> [feature maps]` is either
        models.unsb_nets.PatchEncoder   own weights, independent of G -- works with any regressor
        models.unsb_nets.FeatureTap     G's own encoder layers -- only nets that have them (SBUnet)
    and `pnce` is a PatchNCE built from `extractor.channels(...)`. The parameters of whichever
    extractor owns weights, and of `pnce`, are trained by this loss: add them to G's optimizer.
    Both passes run at one fixed bridge `step` (default: the source end, the analogue of the
    official time index 0; PatchEncoder ignores it). `mask` restricts the sampled locations to the
    brain."""
    n = n_steps(sched) - 1 if step is None else int(step)

    def nce(x_src, x_hat):
        step_t = torch.full((x_hat.shape[0],), n, device=sched.std_fwd.device, dtype=torch.long)
        sigma = forward_std(sched, step_t, xdim=x_hat.shape[1:])
        with torch.no_grad():                     # the key is detached in PatchNCE: skip its graph
            keys = extractor(x_src, cond, sigma)
        return pnce(extractor(x_hat, cond, sigma), keys, mask)

    return nce
