# Pairs baked in: the same 1-D toy, now with PAIRED data (x0, x1).
#   x0 ~ p (target, two modes -0.5, 2.0),  x1 = x0 - 1.5 + n,  n ~ N(0, SC^2)   (source = shifted, blurred)
# so the true conditional p(x0 | x1) ∝ p(x0) N(x1; x0 - 1.5, SC^2) is a per-sample Gaussian mixture.
# Every method below is the SAME annealed ImMAP / Kadkhodaie-Simoncelli ascent
#     u <- u + h (D_sigma(u) - u) + gamma z,   sigma <- (1 - beta h) sigma,
#     gamma^2 = [(1 - beta h)^2 - (1 - h)^2] sigma^2,   D_sigma(u) = u + sigma^2 grad log prior_sigma(u)
# run on a different prior and from a different start:
#   unpaired   prior p(x0)            start x1 + sigma0 z        (SDEdit with an ImMAP sampler)
#   paired     prior p(x0 | x1)       start E[x0|x1] + sigma0 z  (I2SB's warm start, ImMAP sampler)
#   paired, via PMI: log p(x0) + [log p(x0, x1) - log p(x0) - log q(x1)]  -- the same thing written as
#              "target prior + pointwise mutual information", checked to equal the conditional.
import numpy as np
from scipy.stats import wasserstein_distance

rng = np.random.default_rng(1)
S, SC, SHIFT = 0.35, 0.6, 1.5
W = np.array([.5, .5]); M = np.array([-0.5, 2.0])          # target p(x0)
N = 2000
k = rng.choice(2, N, p=W)
x0 = M[k] + S * rng.standard_normal(N)
x1 = x0 - SHIFT + SC * rng.standard_normal(N)


def mix_score(w, m, v, x, sig):
    """grad log of sum_k w_k N(x; m_k, v_k + sig^2); w, m, v are (N, K) or (K,)."""
    vv = v + sig ** 2
    lg = np.log(w) - 0.5 * (x[:, None] - m) ** 2 / vv - 0.5 * np.log(vv)
    r = np.exp(lg - lg.max(1, keepdims=True)); r /= r.sum(1, keepdims=True)
    return (r * (m - x[:, None]) / vv).sum(1)


def prior_uncond():
    return np.broadcast_to(W, (N, 2)), np.broadcast_to(M, (N, 2)), np.full((N, 2), S ** 2)


def prior_cond(x1):
    """p(x0 | x1): product of each target component with N(x0; x1 + 1.5, SC^2)."""
    c = x1[:, None] + SHIFT
    w = W * np.exp(-0.5 * (c - M) ** 2 / (S ** 2 + SC ** 2))
    w = w / w.sum(1, keepdims=True)
    v = np.full((N, 2), S ** 2 * SC ** 2 / (S ** 2 + SC ** 2))
    m = (M * SC ** 2 + c * S ** 2) / (S ** 2 + SC ** 2)
    return w, m, v


def ks_ascent(prior, start, sigma0, beta, h=0.1, sig_min=1e-3):
    w, m, v = prior
    u = start + sigma0 * rng.standard_normal(N)
    sig = sigma0
    while sig > sig_min:
        D = u + sig ** 2 * mix_score(w, m, v, u, sig)
        g2 = max(((1 - beta * h) ** 2 - (1 - h) ** 2) * sig ** 2, 0.0)
        u = u + h * (D - u) + np.sqrt(g2) * rng.standard_normal(N)
        sig = (1 - beta * h) * sig
    return u + sig ** 2 * mix_score(w, m, v, u, sig)


# sanity: PMI form == conditional (gradients in x0, at fixed x1)
g = np.linspace(-3, 4, N)
wq, mq, vq = prior_uncond()
pmi_grad = (mix_score(wq, mq, vq, g, 0.0)                         # grad log p(x0)
            + (x1 - (g - SHIFT)) / SC ** 2)                         # grad log p(x1|x0) = grad PMI
wc, mc, vc = prior_cond(x1)
print(f"PMI form vs conditional, max |grad diff| = "
      f"{np.max(np.abs(pmi_grad - mix_score(wc, mc, vc, g, 0.0))):.1e}\n")

post_mean = (wc * mc).sum(1)
truth_right = x0 > 0.75
print(f"{'method':44s} {'W1 to p(x0)':>11s} {'pair err':>9s} {'right-mode':>10s} "
      f"{'mode = true pair':>16s}")


def report(name, x):
    print(f"{name:44s} {wasserstein_distance(x, x0):11.3f} {np.median(np.abs(x - x0)):9.3f} "
          f"{np.mean(x > 0.75):10.1%} {np.mean((x > 0.75) == truth_right):16.1%}")


report("source x1 itself", x1)
report("oracle: E[x0 | x1]", post_mean)
for s0 in (0.3, 1.0, 3.0):
    for beta, lab in ((1.0, "MAP"), (0.5, "sample")):
        report(f"unpaired  p(x0),      start x1,  s0={s0:3.1f} {lab}",
               ks_ascent(prior_uncond(), x1, s0, beta))
for s0 in (0.3, 1.0, 3.0):
    for beta, lab in ((1.0, "MAP"), (0.5, "sample")):
        report(f"paired    p(x0|x1),   start E[x0|x1], s0={s0:3.1f} {lab}",
               ks_ascent(prior_cond(x1), post_mean, s0, beta))
# exact posterior samples for reference
r = rng.random(N) < wc[:, 1]
exact = np.where(r, mc[:, 1], mc[:, 0]) + np.sqrt(vc[:, 0]) * rng.standard_normal(N)
report("reference: exact samples of p(x0|x1)", exact)
print(f"\nfloor: W1 between two independent p(x0) samples ~ "
      f"{wasserstein_distance(M[rng.choice(2, N)] + S * rng.standard_normal(N), x0):.3f}")
