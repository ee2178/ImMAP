#!/usr/bin/env python3
"""
Does a CLASSICAL V-cycle remove aliasing, with nothing learned? One fully sampled slice.

    python scripts/classical_vcycle.py path/to/volume.h5 --R 8 --out results/classical_vcycle/brain_R8

Objective, per level, with smoothed total variation in place of the learned prior:

    F_l(x) = f_l(x) + lam / 2^l * TV_delta(x),   TV_delta = sum sqrt(|D x|^2 + delta^2) - delta

Coarse problem as in MGProx / MG-OPT:

    x_c* = argmin F_{l+1}(x) - <pi, x>,   pi = grad F_{l+1}(R x) - R (grad F_l(x) - pi_parent)
    x <- x + alpha P(x_c* - R x),         alpha by backtracking on the level's tilted objective

R and P are ideal k-space crop and zero-pad (P^H = 4 R per level). Two coarse data terms:

    crop      f_c = 1/2 ||M_c F S_c x - y_c||^2     the nets' rediscretized operator
    galerkin  f_c = f(P x) / 4                      costs a fine Gram. With this ideal P it only
                                                    reaches the central lines, so it nearly equals
                                                    crop: it does NOT test what the nets' sinc_vertex
                                                    Galerkin folds in through its transition band

Compared against gradient descent and Nesterov on the fine level alone, by WORK: a fine gradient
is 1, a coarse crop gradient 4^-l, a Galerkin one 1 at any level. A convergent solver only reaches
the objective's minimizer faster, so the minimizer is computed as well: what it holds is the
objective's doing, not the solver's. Ghost content is `beta` of scripts/coarse_diag.py without its
lattice filter: the residual projected on the zero-filled template with its true edges removed,
1 for the zero-filled image and 0 for the target.

Outputs: <out>_curves.png, <out>_images.png, <out>.txt.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from physics.mask import make_acc_mask, resolve_acs_lines  # noqa: E402
from show_coarse_problems import acs_maps, crop, fft2c, ifft2c  # noqa: E402


def restrict(x):
    H, W = x.shape[-2:]
    return ifft2c(crop(fft2c(x), 2)) / 2


def prolong(x):
    H, W = x.shape[-2:]
    out = np.zeros(x.shape[:-2] + (2 * H, 2 * W), complex)
    out[..., H - H // 2:H - H // 2 + H, W - W // 2:W - W // 2 + W] = fft2c(x)
    return ifft2c(out) * 2


def D(x):
    return np.stack([np.roll(x, -1, 0) - x, np.roll(x, -1, 1) - x])


def Dh(d):
    return (np.roll(d[0], 1, 0) - d[0]) + (np.roll(d[1], 1, 1) - d[1])


class Problem:
    def __init__(self, y, mask, maps, lam, delta, levels, mode):
        self.lam, self.delta, self.levels, self.mode = lam, delta, levels, mode
        self.y, self.m, self.s = [y], [mask], [maps]
        for l in range(1, levels):
            f = self.y[-1]
            self.y.append(crop(f, 2) / 2)
            w = self.m[-1].size
            self.m.append(self.m[-1][w // 2 - w // 4:w // 2 + w // 4])
            sc = restrict(self.s[-1])
            self.s.append(sc / (np.sqrt((np.abs(sc) ** 2).sum(0, keepdims=True)) + 1e-12))
        self.work = 0.0

    def _resid(self, x, l):
        return self.m[l] * fft2c(self.s[l] * x) - self.m[l] * self.y[l]

    def _grad_data(self, x, l):
        if self.mode == "galerkin" and l > 0:
            g = self._grad_data(prolong(x), l - 1)
            return restrict(g)
        self.work += 4.0 ** -l
        return (np.conj(self.s[l]) * ifft2c(self.m[l] * self._resid(x, l))).sum(0)

    def _f(self, x, l):
        if self.mode == "galerkin" and l > 0:
            return self._f(prolong(x), l - 1) / 4
        self.work += 4.0 ** -l
        return 0.5 * np.linalg.norm(self._resid(x, l)) ** 2

    def grad(self, x, l, pi=None):
        d = D(x)
        g = self._grad_data(x, l) + self.lam / 2 ** l * Dh(d / np.sqrt(np.abs(d) ** 2 + self.delta ** 2))
        return g if pi is None else g - pi

    def obj(self, x, l=0, pi=None):
        d = D(x)
        v = self._f(x, l) + self.lam / 2 ** l * (np.sqrt(np.abs(d) ** 2 + self.delta ** 2) - self.delta).sum()
        return v if pi is None else v - np.vdot(pi, x).real

    @property
    def L(self):
        return 1.0 + 8 * self.lam / self.delta


def smooth(p, x, l, pi, n, nesterov):
    step = 1.0 / (1.0 + 8 * p.lam / 2 ** l / p.delta)
    z, t = x.copy(), 1.0
    for _ in range(n):
        xn = z - step * p.grad(z, l, pi)
        if nesterov:
            tn = (1 + np.sqrt(1 + 4 * t * t)) / 2
            z = xn + (t - 1) / tn * (xn - x)
            t = tn
        else:
            z = xn
        x = xn
    return x


def vcycle(p, x, l, pi, iters, nesterov, log, cb):
    if l == len(iters) - 1:
        return smooth(p, x, l, pi, iters[l], nesterov)
    x = smooth(p, x, l, pi, iters[l] // 2, nesterov)
    if l == 0:
        cb(x)
    xc = restrict(x)
    pic = p.grad(xc, l + 1) - restrict(p.grad(x, l, pi))
    xc2 = vcycle(p, xc, l + 1, pic, iters, nesterov, log, cb)
    d = prolong(xc2 - xc)
    # backtracking on the level's tilted objective; alpha = 0 when no step helps
    f0, alpha = p.obj(x, l, pi), 1.0
    while alpha > 1e-3 and p.obj(x + alpha * d, l, pi) >= f0:
        alpha /= 2
    alpha = alpha if alpha > 1e-3 else 0.0
    xn = x + alpha * d
    log.append(dict(level=l, alpha=alpha, x_before=x if l == 0 else None, x_after=xn if l == 0 else None))
    if l == 0:
        cb(xn)
    return smooth(p, xn, l, pi, iters[l] // 2, nesterov)


class Score:
    def __init__(self, target, zf):
        self.t = np.abs(target)
        self.sup = ndi.binary_erosion(self.t > 0.1 * np.percentile(self.t, 99.5), iterations=2)
        hp = lambda a: a - ndi.uniform_filter1d(a, 7, axis=1)     # noqa: E731
        self.hp = hp
        z = np.abs(zf)
        z = z * (z[self.sup] @ self.t[self.sup]) / (z[self.sup] @ z[self.sup])
        H, T = hp(self.t), hp(z)
        self.G = (T - (T[self.sup] @ H[self.sup]) / (H[self.sup] @ H[self.sup]) * H)
        self.zf_scaled = z

    def nrmse(self, x):
        return np.linalg.norm((np.abs(x) - self.t)[self.sup]) / np.linalg.norm(self.t[self.sup])

    def beta(self, x):
        e = self.hp(np.abs(x) - self.t)
        return (e[self.sup] @ self.G[self.sup]) / (self.G[self.sup] @ self.G[self.sup])


def fine_only(p, x0, n, nesterov, score, every=2):
    step = 1.0 / p.L
    x, z, t = x0.copy(), x0.copy(), 1.0
    p.work = 0.0
    out = [(0.0, score.nrmse(x), score.beta(x), x)]
    for i in range(n):
        xn = z - step * p.grad(z, 0)
        if nesterov:
            tn = (1 + np.sqrt(1 + 4 * t * t)) / 2
            z = xn + (t - 1) / tn * (xn - x)
            t = tn
        else:
            z = xn
        x = xn
        if (i + 1) % every == 0:
            out.append((p.work, score.nrmse(x), score.beta(x), x))
    return out


def main(argv=None):
    a_ = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a_.add_argument("volume")
    a_.add_argument("--R", type=int, default=8)
    a_.add_argument("--slice", type=int, default=5)
    a_.add_argument("--sigma", type=float, default=0.015)
    a_.add_argument("--scale", type=float, default=2000.0)
    a_.add_argument("--lam", type=float, default=None, help="default: best of a small grid at the minimizer")
    a_.add_argument("--delta", type=float, default=0.02)
    a_.add_argument("--iters", type=int, nargs="+", default=[4, 4, 6], help="layers per level, fine to coarse")
    a_.add_argument("--cycles", type=int, default=6)
    a_.add_argument("--budget", type=float, default=30.0, help="work at which solvers are compared")
    a_.add_argument("--min-iters", type=int, default=600, help="Nesterov steps for the minimizer")
    a_.add_argument("--out", required=True)
    a = a_.parse_args(argv)

    with h5py.File(a.volume, "r") as f:
        k = f["kspace"][a.slice].astype(np.complex64) * a.scale
    C, H, W = k.shape
    cf = 0.32 / a.R
    acs = resolve_acs_lines(W, None, cf)
    mask = make_acc_mask(shape=(H, W), accel=a.R, acs_lines=None, mode="uniform", center_frac=cf,
                         adjust_accel=False).numpy().squeeze()
    mask = (mask if mask.ndim == 1 else mask[0]).astype(float)
    rng = np.random.default_rng(1234)
    y = k + a.sigma * (rng.standard_normal(k.shape) + 1j * rng.standard_normal(k.shape)) / np.sqrt(2)
    maps = acs_maps(k, acs)
    target = (np.conj(maps) * ifft2c(k)).sum(0)
    zf = (np.conj(maps) * ifft2c(mask * y)).sum(0)
    score = Score(target, zf)
    L = len(a.iters)
    lines = [f"R={a.R} slice {a.slice}: {int(mask.sum())} lines, iters {a.iters} x {a.cycles} cycles, "
             f"delta {a.delta}, sigma {a.sigma}"]

    x0 = score.zf_scaled * np.exp(1j * np.angle(zf))      # started at the right scale, as a net's preproc is
    t0 = time.time()
    if a.lam is None:
        best = None
        for lam in (0.005, 0.01, 0.02, 0.04):
            p = Problem(y, mask, maps, lam, a.delta, 1, "crop")
            x = fine_only(p, x0, 250, True, score, every=250)[-1][3]
            e = score.nrmse(x)
            lines.append(f"  lam {lam}: NRMSE {e:.3f} after 250 Nesterov steps")
            if best is None or e < best[0]:
                best = (e, lam)
        a.lam = best[1]
    lines.append(f"lam = {a.lam}")

    p = Problem(y, mask, maps, a.lam, a.delta, L, "crop")
    run = fine_only(p, x0, a.min_iters, True, score, every=a.min_iters)
    xstar = run[-1][3]
    Fstar = p.obj(xstar)
    print(f"minimizer done {time.time() - t0:.0f}s", flush=True)

    curves = {}
    finals = {}

    def at_budget(hist):
        return min(hist, key=lambda r: abs(r[0] - a.budget))

    for name, nest in (("fine GD", False), ("fine Nesterov", True)):
        p = Problem(y, mask, maps, a.lam, a.delta, 1, "crop")
        hist = fine_only(p, x0, int(1.5 * a.budget), nest, score, every=1)
        curves[name] = [(w, e, b, p.obj(x) - Fstar) for w, e, b, x in hist]
        finals[name] = at_budget(hist)[3]

    corr = []
    for mode in ("crop", "galerkin"):
        for nest in (False, True):
            name = f"V-cycle {mode}" + (" Nesterov" if nest else " GD")
            p = Problem(y, mask, maps, a.lam, a.delta, L, mode)
            p.work = 0.0
            x, hist, logs = x0.copy(), [], []

            def note(z):
                w = p.work
                hist.append((w, score.nrmse(z), score.beta(z), z))

            note(x)
            for _ in range(a.cycles):
                log = []
                x = vcycle(p, x, 0, None, a.iters, nest, log, note)
                note(x)
                logs.append(log)
            w = p.work
            curves[name] = []
            for wk, e, b, z in hist:
                curves[name].append((wk, e, b, p.obj(z) - Fstar))
            p.work = w
            finals[name] = hist[-1][3]
            for c, log in enumerate(logs):
                for r in log:
                    if r["level"] == 0:
                        corr.append((name, c + 1, r["alpha"], score.nrmse(r["x_before"]), score.nrmse(r["x_after"]),
                                     score.beta(r["x_before"]), score.beta(r["x_after"])))
                    else:
                        corr.append((name, c + 1, r["alpha"], None, None, None, None, r["level"]))
            print(f"{name} done {time.time() - t0:.0f}s, work {w:.1f}", flush=True)

    lines += ["", f"{'solver':<28}{'work':>8}{'NRMSE':>8}{'ghost':>8}{'F - F*':>12}"]
    lines.append(f"{'zero-filled, scaled':<28}{0:>8.1f}{score.nrmse(x0):>8.3f}{score.beta(x0):>8.3f}{'':>12}")
    for name, c in curves.items():
        r = c[-1] if name.startswith("V-cycle") else min(c, key=lambda r: abs(r[0] - a.budget))
        lines.append(f"{name:<28}{r[0]:>8.1f}{r[1]:>8.3f}{r[2]:>8.3f}{r[3]:>12.1f}")
    lines.append(f"{'minimizer':<28}{a.min_iters:>8.0f}{score.nrmse(xstar):>8.3f}{score.beta(xstar):>8.3f}{0:>12.1f}")
    lines += ["", "L1 -> L0 coarse correction, each cycle: what the step itself did",
              f"{'solver':<28}{'cycle':>6}{'alpha':>7}{'NRMSE':>16}{'ghost':>16}"]
    for r in corr:
        if len(r) == 7:
            lines.append(f"{r[0]:<28}{r[1]:>6}{r[2]:>7.2f}{r[3]:>8.3f}->{r[4]:.3f}{r[5]:>8.3f}->{r[6]:.3f}")
    alphas = [r[2] for r in corr if len(r) == 8]
    if alphas:
        lines.append(f"L2 -> L1 corrections: mean alpha {np.mean(alphas):.2f}, zero on {np.mean(np.array(alphas) == 0):.0%}")
    txt = "\n".join(lines)
    print(txt)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out + ".txt", "w") as f:
        f.write(txt + "\n")

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for name, c in curves.items():
        c = np.array([(r[0], r[1], r[2], max(r[3], 1e-3)) for r in c])
        style = "--" if name.startswith("fine") else "-"
        for ax, j in zip(axs, (1, 2, 3)):
            ax.plot(c[:, 0], c[:, j], style, label=name, lw=1.3)
    for ax, t in zip(axs, ("NRMSE", "ghost content", "F - F*")):
        ax.set_xlabel("work, fine gradients")
        ax.set_title(t, fontsize=10)
        ax.grid(alpha=0.3)
        ax.axvline(a.budget, color="gray", lw=0.7, ls=":")
    axs[0].axhline(score.nrmse(xstar), color="k", lw=0.8, ls=":")
    axs[1].axhline(score.beta(xstar), color="k", lw=0.8, ls=":")
    axs[2].set_yscale("log")
    axs[0].legend(fontsize=7)
    fig.suptitle("Dotted black: the minimizer. Dotted gray: the work budget", fontsize=9)
    fig.tight_layout()
    fig.savefig(a.out + "_curves.png", dpi=130)

    sel = [("fine Nesterov", finals["fine Nesterov"]), ("V-cycle crop", finals["V-cycle crop GD"]),
           ("V-cycle Galerkin", finals["V-cycle galerkin GD"]), ("minimizer", xstar)]
    rows = np.flatnonzero(np.abs(target).mean(1) > 0.15 * np.abs(target).mean(1).max())
    rs = slice(max(rows[0] - 8, 0), rows[-1] + 9)
    vmax = np.percentile(score.t, 99.5)
    fig, ax = plt.subplots(2, len(sel) + 1, figsize=(2.9 * (len(sel) + 1), 7.2))
    ax[0, 0].imshow(score.t[rs], cmap="gray", vmin=0, vmax=vmax)
    ax[0, 0].set_title("target", fontsize=9)
    ax[1, 0].imshow(np.abs(score.zf_scaled - score.t)[rs], cmap="inferno", vmin=0, vmax=vmax / 3)
    ax[1, 0].set_title(f"zero-filled error, NRMSE {score.nrmse(x0):.2f}", fontsize=9)
    for i, (n, x) in enumerate(sel, 1):
        ax[0, i].imshow(np.abs(x)[rs], cmap="gray", vmin=0, vmax=vmax)
        ax[0, i].set_title(n, fontsize=9)
        ax[1, i].imshow(np.abs(np.abs(x) - score.t)[rs], cmap="inferno", vmin=0, vmax=vmax / 3)
        ax[1, i].set_title(f"NRMSE {score.nrmse(x):.3f}, ghost {score.beta(x):.3f}", fontsize=9)
    for A in ax.ravel():
        A.axis("off")
    fig.tight_layout()
    fig.savefig(a.out + "_images.png", dpi=130)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
