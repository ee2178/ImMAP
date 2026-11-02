#!/usr/bin/env python3
"""
What the rediscretized coarse problems look like on one fully sampled slice.

    python scripts/show_coarse_problems.py path/to/volume.h5 --R 8 --slice 5 \
        --out results/coarse_problems/brain_R8.png

For each level l the problem operators/coarse.py poses: the central 1/2^l of k-space with the
same mask (mode uniform, center_frac 0.32 / R, adjust_accel off: exp18's cf32), the restricted
coil maps, and white noise added at the finest level. Per level, the mask, the level's target
(the coil-combined full data of that level), the zero-filled image and a Tikhonov CG-SENSE
solve: how far the physics alone gets, with no network. Levels past L2 are not in the
current nets, which stop there.

Coil maps come from a low-pass of the calibration lines, not online ESPIRiT, so this shows the
shape of the problems rather than what a trained net sees. Numpy for the linear algebra, the
repo's mask and restriction for everything that defines the problem.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from physics.mask import make_acc_mask, resolve_acs_lines  # noqa: E402


def _load_resample():
    # by path: `operators/__init__` is not needed for restriction
    spec = importlib.util.spec_from_file_location("resample", os.path.join(ROOT, "operators", "resample.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fft2c(x):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x, axes=(-2, -1)), norm="ortho"), axes=(-2, -1))


def ifft2c(k):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(k, axes=(-2, -1)), norm="ortho"), axes=(-2, -1))


def crop(k, f):
    H, W = k.shape[-2:]
    hc, wc = H // f, W // f
    r0, c0 = H // 2 - hc // 2, W // 2 - wc // 2
    return k[..., r0:r0 + hc, c0:c0 + wc]


def acs_maps(k, acs):
    """Unit-RSS maps from a Hann low-pass of the central `acs` lines."""
    W = k.shape[-1]
    win = np.zeros(W)
    lo = W // 2 - acs // 2
    win[lo:lo + acs] = np.hanning(acs + 2)[1:-1]
    H = k.shape[-2]
    wr = np.hanning(H)
    s = ifft2c(k * win * wr[:, None])
    return s / (np.sqrt((np.abs(s) ** 2).sum(0, keepdims=True)) + 1e-12)


def restrict_maps(s, l, restrict):
    t = torch.from_numpy(s[None].astype(np.complex64))
    for _ in range(l):
        t = restrict(t, filter="sinc_vertex")
    s = t[0].numpy()
    return s / (np.sqrt((np.abs(s) ** 2).sum(0, keepdims=True)) + 1e-12)


def cg_sense(y, m, s, lam, iters):
    """(E^H E + lam) x = E^H y, E = m F S."""
    def EH(k):
        return (np.conj(s) * ifft2c(m * k)).sum(0)

    def G(x):
        return EH(fft2c(s * x)) + lam * x

    b = EH(y)
    x = np.zeros_like(b)
    r = b.copy()
    p = r.copy()
    rs = np.vdot(r, r).real
    for _ in range(iters):
        Gp = G(p)
        a = rs / np.vdot(p, Gp).real
        x = x + a * p
        r = r - a * Gp
        new = np.vdot(r, r).real
        if new < 1e-10 * np.vdot(b, b).real:
            break
        p = r + new / rs * p
        rs = new
    return x, b


def nrmse(x, ref, sup):
    return np.linalg.norm((np.abs(x) - np.abs(ref))[sup]) / np.linalg.norm(np.abs(ref)[sup])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("volume")
    p.add_argument("--R", type=int, default=8)
    p.add_argument("--slice", type=int, default=5)
    p.add_argument("--levels", type=int, default=5)
    p.add_argument("--sigma", type=float, default=0.015)
    p.add_argument("--scale", type=float, default=2000.0, help="the loader's scale_fac")
    p.add_argument("--lam", type=float, default=1e-2)
    p.add_argument("--iters", type=int, default=60)
    p.add_argument("--current-levels", type=int, default=3, help="how many levels the nets have")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    restrict = _load_resample().restrict
    with h5py.File(a.volume, "r") as f:
        k = f["kspace"][a.slice].astype(np.complex64) * a.scale
    C, H, W = k.shape
    cf = 0.32 / a.R
    acs = resolve_acs_lines(W, None, cf)
    mask = make_acc_mask(shape=(H, W), accel=a.R, acs_lines=None, mode="uniform", center_frac=cf,
                         adjust_accel=False).numpy().squeeze()
    mask = mask if mask.ndim == 1 else mask[0]
    rng = np.random.default_rng(1234)
    noise = a.sigma * (rng.standard_normal(k.shape) + 1j * rng.standard_normal(k.shape)) / np.sqrt(2)
    y0 = k + noise
    maps0 = acs_maps(k, acs)

    # one crop for every level, from the fine image's support
    ref0 = (np.conj(maps0) * ifft2c(k)).sum(0)
    prof = np.abs(ref0).mean(1)
    rows = np.flatnonzero(prof > 0.15 * prof.max())
    r0, r1 = (max(rows[0] - 8, 0) / H, min(rows[-1] + 9, H) / H)

    fig, axs = plt.subplots(a.levels, 6, figsize=(12.5, 2.35 * a.levels),
                            gridspec_kw=dict(width_ratios=[0.8, 1, 1, 1, 1, 1], wspace=0.04, hspace=0.22))
    print(f"{'level':<6}{'size':>10}{'lines':>7}{'R_eff':>7}{'ACS share':>11}{'NRMSE zf':>10}{'NRMSE cg':>10}")
    for l in range(a.levels):
        f = 2 ** l
        yl, kl = crop(y0, f) / f, crop(k, f) / f
        ml = crop(mask[None], f)[0] if False else mask[W // 2 - (W // f) // 2:W // 2 - (W // f) // 2 + W // f]
        sl = restrict_maps(maps0, l, restrict)
        tgt = (np.conj(sl) * ifft2c(kl)).sum(0)
        x, zf = cg_sense(yl, ml, sl, a.lam, a.iters)
        sup = np.abs(tgt) > 0.1 * np.percentile(np.abs(tgt), 99.5)
        gain = np.vdot(np.abs(zf)[sup], np.abs(tgt)[sup]).real / np.vdot(np.abs(zf)[sup], np.abs(zf)[sup]).real
        zf_s = zf * gain
        e_zf, e_cg = nrmse(zf_s, tgt, sup), nrmse(x, tgt, sup)
        n = int(ml.sum())
        reff = ml.size / n
        print(f"L{l:<5}{tgt.shape[0]:>5}x{tgt.shape[1]:<4}{n:>7}{reff:>7.1f}{min(acs, ml.size) / n:>11.0%}{e_zf:>10.3f}{e_cg:>10.3f}")

        hl = tgt.shape[0]
        rs = slice(int(r0 * hl), max(int(r1 * hl), int(r0 * hl) + 1))
        vmax = np.percentile(np.abs(tgt), 99.5)
        ax = axs[l]
        # antialiased: at L0 a 1 px line is thinner than a screen pixel and nearest drops some
        ax[0].imshow(np.tile(ml[:, None].T, (max(ml.size // 3, 3), 1)), cmap="gray", vmin=0, vmax=1,
                     interpolation="antialiased", aspect="auto")
        status = "in current nets" if l < a.current_levels else "not in current nets"
        ax[0].set_ylabel(f"L{l}: {tgt.shape[0]}x{tgt.shape[1]}\nR_eff {reff:.1f}\n{status}", fontsize=9,
                         color="black" if l < a.current_levels else "tab:blue")
        ax[0].set_title(f"k-space lines ({n} of {ml.size})", fontsize=8)
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        panels = [(np.abs(tgt), "target", "gray", vmax),
                  (np.abs(zf_s), f"zero-filled, NRMSE {e_zf:.2f}", "gray", vmax),
                  (np.abs(np.abs(zf_s) - np.abs(tgt)), "|zero-filled error|", "inferno", vmax / 2),
                  (np.abs(x), f"CG-SENSE, NRMSE {e_cg:.2f}", "gray", vmax),
                  (np.abs(np.abs(x) - np.abs(tgt)), "|CG-SENSE error|", "inferno", vmax / 2)]
        for A, (im, title, cm, vm) in zip(ax[1:], panels):
            A.imshow(im[rs], cmap=cm, vmin=0, vmax=vm, interpolation="nearest")
            A.set_title(title, fontsize=8)
            A.axis("off")
    fig.suptitle(f"The coarse problems at R={a.R}  (cf32 mask, centre {acs} lines, sigma {a.sigma}; "
                 f"zero-filled rescaled to the target; errors at half the image window; no network)", fontsize=10)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=130, bbox_inches="tight", facecolor="white")
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
