# Toy: recover a TARGET image from a SOURCE image by maximising log p(x) - gamma log q(x).
# 1-D, p and q Gaussian mixtures (analytic scores at any smoothing sigma). Target = source shifted
# by +1.5, so the "right answer" for a source point x is x + 1.5 (the 1-D optimal transport map).
import numpy as np
from scipy.stats import wasserstein_distance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
S = 0.35
Q = dict(w=np.array([.5, .5]), m=np.array([-2.0, 0.5]), s=S)        # source
P = dict(w=np.array([.5, .5]), m=np.array([-0.5, 2.0]), s=S)        # target = source + 1.5
SHIFT = 1.5


def sample(D, n):
    k = rng.choice(len(D["w"]), n, p=D["w"])
    return D["m"][k] + D["s"] * rng.standard_normal(n)


def _resp(D, x, sig):
    v = D["s"] ** 2 + sig ** 2
    lg = np.log(D["w"]) - 0.5 * (x[..., None] - D["m"]) ** 2 / v - 0.5 * np.log(2 * np.pi * v)
    mx = lg.max(-1, keepdims=True)
    r = np.exp(lg - mx)
    return r / r.sum(-1, keepdims=True), v, (mx[..., 0] + np.log(np.exp(lg - mx).sum(-1)))


def logpdf(D, x, sig=0.0):
    return _resp(D, x, sig)[2]


def score(D, x, sig=0.0):
    r, v, _ = _resp(D, x, sig)
    return (r * (D["m"] - x[..., None]) / v).sum(-1)


def dscore(D, x, sig=0.0):          # d^2/dx^2 log density
    r, v, _ = _resp(D, x, sig)
    a = (D["m"] - x[..., None]) / v
    return (r * (a ** 2 - 1 / v)).sum(-1) - (r * a).sum(-1) ** 2


x_src = sample(Q, 1000)
x_tgt = sample(P, 1000)
out = {}

# ---------------------------------------------------------------- A. point-wise ratio ascent
print("A. gradient ascent on log p(x) - gamma log q(x), from 1000 source samples")
for gamma in (0.0, 0.5, 1.0):
    x = x_src.copy()
    for _ in range(4000):
        x = x + 0.01 * (score(P, x) - gamma * score(Q, x))
        x = np.clip(x, -1e3, 1e3)
    out[f"A{gamma}"] = x
    div = np.mean(np.abs(x) > 50)
    err = np.median(np.abs(x - (x_src + SHIFT)))
    print(f"  gamma={gamma:3.1f}: diverged (|x|>50) {div:5.1%}   median |x - (x_src+1.5)| "
          f"{err:7.3f}   W1 to target {wasserstein_distance(x, x_tgt):8.3f}   "
          f"final modes ~ {np.round(np.unique(np.round(x[np.abs(x) < 50], 2)), 2)[:6]}")

# the measurement cancels in a ratio of POSTERIORS with the same likelihood
y, sn = 2.3, 0.5
g = np.linspace(-4, 4, 9)
lik = (y - g) / sn ** 2
print(f"  y cancels: |grad[log p(x|y) - log q(x|y)] - grad[log p - log q]| max = "
      f"{np.max(np.abs((score(P, g) + lik - score(Q, g) - lik) - (score(P, g) - score(Q, g)))):.1e}")

# ---------------------------------------------------------------- B. ADMM split, smoothed source
print("\nB. ADMM: x-step prox of -log p, z-step 'anti-prox' of gamma log q_sigma, x = z")
grid = np.linspace(-30, 30, 30001)
lp_grid = logpdf(P, grid)
for gamma, sig_q in ((1.0, 0.0), (1.0, 0.5), (0.5, 0.0)):
    lq_grid = logpdf(Q, grid, sig_q)
    curv = dscore(Q, grid, sig_q)
    for rho in (0.5 * gamma / max(sig_q, S) ** 2, 4.0 * gamma / max(sig_q, S) ** 2):
        x = x_src[:300].copy(); z = x.copy(); u = np.zeros_like(x)
        nonconvex = np.min(rho + gamma * curv) < 0      # z-subproblem Hessian rho + gamma d2 log q
        for _ in range(300):
            x = grid[np.argmin(-lp_grid[None] + 0.5 * rho * (grid[None] - (z - u)[:, None]) ** 2, 1)]
            v = x + u
            z = grid[np.argmin(gamma * lq_grid[None] + 0.5 * rho * (grid[None] - v[:, None]) ** 2, 1)]
            u = u + x - z
        edge = np.mean(np.abs(z) > 29)
        print(f"  gamma={gamma}, sigma_q={sig_q:3.1f}, rho*s_eff^2/gamma={rho * max(sig_q, S)**2 / gamma:3.1f}: "
              f"z-step nonconvex somewhere: {str(nonconvex):5s}  runs to grid edge {edge:5.1%}  "
              f"|x-z| {np.median(np.abs(x - z)):.1e}  median x {np.median(x):7.2f}")
    # where is the true optimum of -log p + gamma log q_sigma (if bounded)?
    obj = -lp_grid + gamma * lq_grid
    print(f"    objective min on [-30, 30] at x = {grid[np.argmin(obj)]:.2f} "
          f"(target modes -0.5, 2.0; edge = unbounded)")
    out[f"B{gamma}_{sig_q}"] = (grid, obj)

# ---------------------------------------------------------------- C. KL flow: q updated as rho_t
print("\nC. Wasserstein gradient flow of KL(rho_t || p): velocity s_p(x) - grad log rho_t(x), rho_t = KDE")
x = x_src.copy(); h = 0.12
hist_c = []
for it in range(3001):
    d = x[:, None] - x[None, :]
    k = np.exp(-0.5 * d ** 2 / h ** 2)
    s_rho = -(k * d).sum(1) / h ** 2 / k.sum(1)
    x = x + 0.004 * (score(P, x) - s_rho)
    if it % 500 == 0:
        hist_c.append((it, wasserstein_distance(x, x_tgt)))
out["C"] = x
print("  W1 to target over iterations:", ", ".join(f"{i}: {w:.3f}" for i, w in hist_c))
right = lambda z: np.mean(z > 0.75)                 # mass in the target's right mode (want 50%)
print(f"  mass in right target mode: {right(x):.1%} (target 50%)   median |x - (x_src+1.5)| "
      f"{np.median(np.abs(x - (x_src + SHIFT))):.3f}")
for k in ("A0.0", "A0.5"):
    print(f"    for comparison {k}: right-mode mass {right(out[k]):.1%}")

# C'. the same flow ANNEALED: both p and rho_t smoothed at a common sigma, sigma 2 -> 0
x = x_src.copy()
sig_sched = np.concatenate([np.geomspace(2.0, 0.02, 2500), np.full(500, 0.0)])
for sg in sig_sched:
    d = x[:, None] - x[None, :]
    hb = np.sqrt(h ** 2 + sg ** 2)
    k = np.exp(-0.5 * d ** 2 / hb ** 2)
    s_rho = -(k * d).sum(1) / hb ** 2 / k.sum(1)
    x = x + 0.004 * (score(P, x, sg) - s_rho)
out["C2"] = x
print(f"  annealed KL flow: W1 to target {wasserstein_distance(x, x_tgt):.3f}   right-mode mass "
      f"{right(x):.1%}   median |x - (x_src+1.5)| {np.median(np.abs(x - (x_src + SHIFT))):.3f}")

# ---------------------------------------------------------------- D. DDIB: encode with q, decode with p
print("\nD. DDIB: probability-flow ODE up with the source score, down with the target score")


def pf_ode(D, x, sigs):
    for a, b in zip(sigs[:-1], sigs[1:]):        # dx/dsigma = -sigma * score_sigma(x), RK4
        f = lambda xx, s: -s * score(D, xx, s)
        hh = b - a
        k1 = f(x, a); k2 = f(x + hh / 2 * k1, a + hh / 2); k3 = f(x + hh / 2 * k2, a + hh / 2)
        k4 = f(x + hh * k3, b)
        x = x + hh / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


sigs = np.geomspace(1e-3, 20.0, 600)
lat = pf_ode(Q, x_src.copy(), sigs)
x_d = pf_ode(P, lat, sigs[::-1])
out["D"] = x_d
print(f"  max |T(x) - (x+1.5)| = {np.max(np.abs(x_d - (x_src + SHIFT))):.4f}   "
      f"median {np.median(np.abs(x_d - (x_src + SHIFT))):.4f}   W1 to target "
      f"{wasserstein_distance(x_d, x_tgt):.3f}")
print(f"\n  reference: W1(source, target) = {wasserstein_distance(x_src, x_tgt):.3f}, "
      f"W1(target, fresh target samples) = {wasserstein_distance(sample(P, 1000), x_tgt):.3f}")

# ---------------------------------------------------------------- figure
xs = np.linspace(-5, 6, 1000)
fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
a = ax[0, 0]
a.plot(xs, np.exp(logpdf(Q, xs)), label="source q", c="tab:orange")
a.plot(xs, np.exp(logpdf(P, xs)), label="target p", c="tab:blue")
a2 = a.twinx()
for gm, ls in ((1.0, "-"), (0.5, "--")):
    a2.plot(xs, logpdf(P, xs) - gm * logpdf(Q, xs), c="k", ls=ls, lw=1,
            label=f"log p - {gm:g} log q")
a2.set_ylim(-30, 40); a2.legend(loc="upper right", fontsize=8)
a.legend(loc="upper left", fontsize=8); a.set_title("(a) densities and the ratio objective")
a = ax[0, 1]
bins = np.linspace(-5, 6, 120)
a.hist(x_tgt, bins, density=True, alpha=.25, color="tab:blue", label="target samples")
for gm, c in ((0.0, "tab:green"), (0.5, "tab:red")):
    a.hist(out[f"A{gm}"], bins, density=True, histtype="step", color=c, lw=1.5,
           label=f"ratio ascent, gamma={gm:g}")
a.set_title(f"(b) point-wise ascent (gamma=1: {np.mean(np.abs(out['A1.0'])>50):.0%} diverge, not shown)")
a.legend(fontsize=8)
a = ax[1, 0]
a.hist(x_tgt, bins, density=True, alpha=.25, color="tab:blue", label="target samples")
a.hist(out["C"], bins, density=True, histtype="step", color="k", lw=1.5, label="KL flow (q -> rho_t)")
a.hist(out["C2"], bins, density=True, histtype="step", color="tab:green", lw=1.5, label="KL flow, annealed sigma")
a.hist(out["D"], bins, density=True, histtype="step", color="tab:purple", lw=1.5, ls="--",
       label="DDIB (encode q, decode p)")
a.set_title("(c) distribution-level fixes"); a.legend(fontsize=8)
a = ax[1, 1]
o = np.argsort(x_src)
a.plot(x_src[o], (x_src + SHIFT)[o], c="tab:blue", lw=3, alpha=.4, label="true map x + 1.5")
a.plot(x_src[o], out["D"][o], c="tab:purple", label="DDIB")
a.plot(x_src[o], out["A0.5"][o], c="tab:red", label="ratio ascent, gamma=0.5")
a.plot(x_src[o], out["A0.0"][o], c="tab:green", label="target MAP only (gamma=0)")
a.set_xlabel("source x"); a.set_ylabel("output"); a.set_title("(d) per-sample maps")
a.legend(fontsize=8)
fig.tight_layout()
fig.savefig(__file__.replace(".py", ".png"), dpi=130)
print("\nfigure:", __file__.replace(".py", ".png"))
print(f"\nDDIB right-mode mass {np.mean(out['D'] > 0.75):.1%}; target samples {np.mean(x_tgt > 0.75):.1%}; "
      f"gamma=1 non-diverged right-mode {np.mean(out['A1.0'][np.abs(out['A1.0'])<50] > 0.75):.1%}")
