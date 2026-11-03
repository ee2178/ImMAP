#!/usr/bin/env python3
"""
What does RAM's start look like on our setups? CG-SENSE at deepinv's defaults (models/cg_init.py),
on fully sampled volumes, under each sampling protocol and R.

    python scripts/cg_init_check.py path/to/volume.h5 --anatomy brain --out results/cg_init/brain
    python scripts/cg_init_check.py v1.h5 v2.h5 --slices 0 8 --R 8 12 --protocols cf32 measured ...

Protocols, as scripts/make_mg_recon_configs.py builds them:
    cf32      center_frac = 0.32 / R, adjust_accel off (exp18 and exp21)
    measured  center_frac = 0.04, adjust_accel on (the ISMRM abstract's runs)
Both: measured k-space times the loader's scale_fac, white noise of std --sigma on every sample
(the measurement_awgn type), online coil maps from the noisy ACS (ESPIRiT at the configs'
settings; --maps walsh if memory is short). The target is the SENSE combination of the fully
sampled k-space with those same maps, so map error is not scored. The stored maps that define
the training target are cluster-only.

For each setup, at deepinv's defaults (50 iterations, |r| < 1e-4 |b|, started at E^H y):
    mu        RAM's ridge weight 1 / gamma (fact = 1, its init)
    iters     CG iterations used, or 50 when the cap stopped it
    NRMSE     of |x| against |target| inside the anatomy, and `ghost`: the share of the
              zero-filled image's aliasing pattern left (1 for E^H y, 0 for the target;
              scripts/classical_vcycle.py::Score)
and the same two numbers as a function of the iteration count, run past the cap to 200 so
semi-convergence shows if it is there. Outputs: <out>.txt, <out>_curves.png, <out>_images.png.
"""

from __future__ import annotations

import argparse
import os
import sys

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from classical_vcycle import Score  # noqa: E402
from models.cg_init import DEEPINV_MAX_ITER, DEEPINV_TOL, CGInit, ram_gamma  # noqa: E402
from operators import FFT2D, Mask, Sense  # noqa: E402
from physics.mask import effective_accel, make_acc_mask  # noqa: E402
from physics.online_smaps import online_smaps  # noqa: E402
from solvers.cg import batched_cg  # noqa: E402

SCALE = {"brain": 2e3, "knee": 5e3}          # datasets/fastmri/loader.py::FASTMRI_PATHS
PROTOCOLS = {"cf32": lambda R: dict(center_frac=0.32 / R, adjust_accel=False),
             "measured": lambda R: dict(center_frac=0.04, adjust_accel=True)}
CURVE = (0, 1, 2, 3, 5, 8, 12, 20, 30, 50, 100, 200)


def setup(k, R, protocol, sigma, maps, rng):
    C, H, W = k.shape
    m = make_acc_mask(shape=(H, W), accel=R, acs_lines=None, mode="uniform", **PROTOCOLS[protocol](R))
    m = torch.as_tensor(m).float().reshape(-1, W)[:1].reshape(1, 1, 1, W)
    kt = torch.from_numpy(k)[None]
    n = sigma * torch.from_numpy((rng.standard_normal(k.shape) + 1j * rng.standard_normal(k.shape)) / np.sqrt(2))
    y = (m * (kt + n[None])).to(torch.complex64)
    kws = dict(kernel_size=4, thresh_eig=0.0, maxit=10) if maps == "espirit" else {}
    S = online_smaps(y, m, method=maps, **kws).to(torch.complex64)
    E = Mask(m) @ FFT2D() @ Sense(S)
    target = (S.conj() * FFT2D().adjoint(kt.to(torch.complex64))).sum(1, keepdim=True)
    return y, E, target, m


def cg_run(y, E, sigma, iters, tol, fact=1.0):
    """deepinv's solve with a call counter: (x, CG iterations used, mu)."""
    z = E.adjoint(y)
    mu = 1.0 / (ram_gamma(y, sigma).view(-1, 1, 1, 1) * fact)
    calls = [0]

    def H(v):
        calls[0] += 1
        return E.gram(v) + mu * v

    x = batched_cg(H, z + mu * z, x0=z, max_iter=iters, tol=tol, stop="rhs") if iters else z
    return x, max(calls[0] - 1, 0), float(mu.reshape(-1)[0])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("volumes", nargs="+")
    p.add_argument("--anatomy", default="brain", choices=sorted(SCALE))
    p.add_argument("--slices", type=int, nargs="+", default=[5])
    p.add_argument("--R", type=int, nargs="+", default=[8, 12])
    p.add_argument("--protocols", nargs="+", default=["cf32", "measured"], choices=sorted(PROTOCOLS))
    p.add_argument("--sigma", type=float, default=0.015, help="the configs' validation sigma")
    p.add_argument("--maps", default="espirit", choices=("espirit", "walsh"))
    p.add_argument("--facts", type=float, nargs="+", default=[1, 10, 100, 1000],
                   help="values of RAM's learned `fact` (mu = 1 / (gamma fact)); 1 is its init")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    torch.set_grad_enabled(False)

    lines = [f"{a.anatomy}, sigma {a.sigma}, {a.maps} maps; deepinv defaults: {DEEPINV_MAX_ITER} iterations, "
             f"tol {DEEPINV_TOL:g}, started at E^H y. NRMSE inside the anatomy; ghost 1 = E^H y, 0 = target.", "",
             f"{'volume':<28}{'sl':>3} {'protocol':<9}{'R':>3}{'R_eff':>6}{'ACS':>5}{'fact':>6}{'mu':>9}{'iters':>6}"
             f"{'ZF nrmse':>9}{'x0 nrmse':>9}{'x0 ghost':>9}{'@200 nrmse':>11}{'@200 ghost':>11}"]
    curves, panels = {}, []
    for path in a.volumes:
        vol = os.path.basename(path)[:-3]
        with h5py.File(path, "r") as f:
            ks = {s: f["kspace"][s].astype(np.complex64) * SCALE[a.anatomy] for s in a.slices}
        for s, k in ks.items():
            for prot in a.protocols:
                for R in a.R:
                    rng = np.random.default_rng(1234)
                    y, E, target, m = setup(k, R, prot, a.sigma, a.maps, rng)
                    sig = torch.tensor([a.sigma])
                    t = target[0, 0].numpy()
                    zf = E.adjoint(y)
                    score = Score(t, zf[0, 0].numpy())
                    for fact in a.facts:
                        init = CGInit()
                        init.fact.fill_(fact)
                        x0 = init(y, E, sig)
                        xr, used, mu = cg_run(y, E, sig, DEEPINV_MAX_ITER, DEEPINV_TOL, fact)
                        assert torch.allclose(x0, xr, atol=1e-5 * float(xr.abs().max())), "CGInit != the counted run"
                        curve = [(n, *(lambda v: (score.nrmse(v), score.beta(v)))(
                            cg_run(y, E, sig, n, 0.0, fact)[0][0, 0].numpy())) for n in CURVE]
                        key = f"{vol} s{s} {prot} R{R} fact {fact:g}"
                        curves[key] = curve
                        x200 = curve[-1]
                        lines.append(f"{vol:<28}{s:>3} {prot:<9}{R:>3}{float(effective_accel(m)):>6.1f}{_acs(m):>5}"
                                     f"{fact:>6g}{mu:>9.2e}{used:>6}{score.nrmse(score.zf_scaled):>9.3f}"
                                     f"{score.nrmse(x0[0, 0].numpy()):>9.3f}{score.beta(x0[0, 0].numpy()):>9.3f}"
                                     f"{x200[1]:>11.3f}{x200[2]:>11.3f}")
                        print(lines[-1], flush=True)
                        if fact in (a.facts[0], a.facts[-1]):
                            panels.append((key, t, score.zf_scaled, x0[0, 0].numpy(),
                                           cg_run(y, E, sig, 200, 0.0, fact)[0][0, 0].numpy(), score))
                    lines.append("")
    txt = "\n".join(lines)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out + ".txt", "w") as f:
        f.write(txt + "\n")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for key, c in curves.items():
        n = [max(v[0], 0.5) for v in c]
        ax[0].plot(n, [v[1] for v in c], marker=".", label=key)
        ax[1].plot(n, [v[2] for v in c], marker=".", label=key)
    for A, t in zip(ax, ("NRMSE", "ghost (1 = E^H y)")):
        A.axvline(DEEPINV_MAX_ITER, color="k", ls=":", lw=0.8)
        A.set_xscale("log")
        A.set_xlabel("CG iterations (0 plotted at 0.5; dotted: deepinv's cap)")
        A.set_title(t, fontsize=9)
        A.grid(alpha=0.3)
    ax[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(a.out + "_curves.png", dpi=130)

    fig, ax = plt.subplots(len(panels), 6, figsize=(15, 2.9 * len(panels)), squeeze=False)
    for i, (key, t, zf, x0, x200, score) in enumerate(panels):
        H = t.shape[0]
        rs = slice(H // 4, 3 * H // 4) if H > 1.5 * t.shape[1] else slice(None)     # brain's 2x readout
        vmax = np.percentile(np.abs(t), 99.5)
        for j, (name, img) in enumerate((("target", t), ("E^H y, scaled", zf), ("x0, deepinv defaults", x0),
                                         ("CG at 200", x200))):
            ax[i, j].imshow(np.abs(img)[rs], cmap="gray", vmin=0, vmax=vmax)
            ax[i, j].set_title(f"{key}\n{name}" if j == 0 else name, fontsize=7)
        for j, (name, img) in enumerate((("|x0| - target", x0), ("|E^H y| - target", zf)), start=4):
            ax[i, j].imshow(np.abs(np.abs(img) - np.abs(t))[rs], cmap="inferno", vmin=0, vmax=vmax / 4)
            ax[i, j].set_title(f"{name}, NRMSE {score.nrmse(img):.3f}", fontsize=7)
    for A in ax.ravel():
        A.axis("off")
    fig.tight_layout()
    fig.savefig(a.out + "_images.png", dpi=110)
    print("wrote", a.out)
    return 0


def _acs(m):
    """Length of the contiguous run of sampled lines through the centre."""
    line = m.reshape(-1).numpy() > 0
    c = line.size // 2
    lo, hi = c, c
    while lo > 0 and line[lo - 1]:
        lo -= 1
    while hi < line.size - 1 and line[hi + 1]:
        hi += 1
    return hi - lo + 1


if __name__ == "__main__":
    sys.exit(main())
