#!/usr/bin/env python3
"""
What an UNTRAINED net does to one fully sampled slice: a diagnostic of the structure itself.

    python scripts/init_net_diag.py path/to/volume.h5 --R 8 --out results/init_net/brain_R8

Builds the generator's real cells at their initial weights and runs them on measured k-space
with white noise, under the cf32 mask. With lam0 = 1e-3 the prior barely acts, so a random-init
net is close to data consistency plus coarse corrections, with no training to hide anything.

For each V-cycle of a multigrid net, the fine iterate at four stages: pre (after pre-smoothing),
+x (pre + alpha_x P(dx)), post (after post-smoothing), with NRMSE and ghost content (`beta`, as in
scripts/classical_vcycle.py: 1 for the zero-filled image, 0 for the target). Also the coarse output
of each level against that level's own target, and the share of dual coefficients on the clip.

Coil maps are a low-pass of the calibration lines, not online ESPIRiT. One slice and one seed per
net unless --seeds says otherwise. Outputs: <out>.txt, <out>_curves.png, <out>_images.png.
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
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from classical_vcycle import Score  # noqa: E402
from physics.mask import make_acc_mask, resolve_acs_lines  # noqa: E402
from show_coarse_problems import acs_maps, cg_sense, crop, ifft2c  # noqa: E402


def build_cells(names):
    import make_mg_recon_configs as gen
    from models.components import set_complex_mode
    from models.mg_lpds import MGLPDSNet
    set_complex_mode("planar")

    def make(name, seed):
        torch.manual_seed(seed)
        extra = {}
        cell = name
        if name.endswith("+a0"):             # the same net with every coarse correction off
            cell, extra = name[:-3], dict(alpha0=0.0)
        return MGLPDSNet(**dict(gen.MODELS[cell]["params"], **extra)).eval()

    return {n: make for n in names}


def clip_share(layer, z, sigma):
    lam = layer.prox.prox.threshold(z, sigma)
    return float((z.abs() >= 0.99 * lam).float().mean())


def run_net(net, y, E, sigma, score):
    """Forward with hooks: one row per stage of each top-level V-cycle, and the coarse outputs."""
    from models.mg_lpds import PDVCycle
    rows, coarse, hooks, cur = [], {}, [], {}

    def note(k, stage, x):
        x = x.detach()
        # the net works on x - mu; `mu` is added back by the post-process
        a = (x + cur["mu"])[0, 0].numpy()
        rows.append(dict(cycle=k, stage=stage, nrmse=score.nrmse(a), beta=score.beta(a)))

    if isinstance(getattr(net.net, "layers", [None])[0], PDVCycle):
        for k, v in enumerate(net.net.layers):
            def pre(m, i, o, k=k, v=v):
                (x, z), _ = o
                cur["x"], cur["z"] = x, z
                note(k, "pre", x)
                rows[-1]["clip"] = clip_share(v.lpdsA.first, z, cur["sigma"])

            def ax(m, i, o, k=k):
                cur["x_corr"] = cur["x"] + o
                note(k, "+x", cur["x_corr"])
                rows[-1]["rel"] = float(o.norm() / cur["x"].norm())

            def post(m, i, o, k=k):
                note(k, "post", o[0][0])

            def low(m, i, o, k=k):
                if k == len(net.net.layers) - 1:
                    coarse["L1"] = o[0][0].detach()[0, 0].numpy()

            hooks += [v.lpdsA.register_forward_hook(pre), v.alpha_x.register_forward_hook(ax),
                      v.lpdsB.register_forward_hook(post), v.mglayer.register_forward_hook(low)]

    from preprocessing.kspace import kspace_pre_process
    _, _, (mu, _) = kspace_pre_process(y, E, net.pad_stride)
    cur["mu"], cur["sigma"] = mu, sigma
    t0 = time.time()
    with torch.no_grad():
        out, _ = net(y, E=E, sigma=sigma)
    for h in hooks:
        h.remove()
    return out[0, 0].numpy(), rows, coarse, time.time() - t0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("volume")
    p.add_argument("--R", type=int, default=8)
    p.add_argument("--slice", type=int, default=5)
    p.add_argument("--sigma", type=float, default=0.015)
    p.add_argument("--scale", type=float, default=2000.0)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--nets", nargs="+", default=["lpdsnet", "mg81v6vtxAnestUcf32", "mg81v6vtxAnestUcf32+a0",
                                                 "mg81v6vtxAnestUlamcf32"])
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    from operators import FFT2D, Mask, Sense

    with h5py.File(a.volume, "r") as f:
        k = f["kspace"][a.slice].astype(np.complex64) * a.scale
    C, H, W = k.shape
    cf = 0.32 / a.R
    acs = resolve_acs_lines(W, None, cf)
    mask = make_acc_mask(shape=(H, W), accel=a.R, acs_lines=None, mode="uniform", center_frac=cf,
                         adjust_accel=False)
    mline = mask.numpy().squeeze()
    mline = (mline if mline.ndim == 1 else mline[0]).astype(float)
    rng = np.random.default_rng(1234)
    noisy = k + a.sigma * (rng.standard_normal(k.shape) + 1j * rng.standard_normal(k.shape)) / np.sqrt(2)
    maps = acs_maps(k, acs)
    target = (np.conj(maps) * ifft2c(k)).sum(0)
    zf = (np.conj(maps) * ifft2c(mline * noisy)).sum(0)
    score = Score(target, zf)
    cg, _ = cg_sense(mline * noisy, mline, maps, 1e-2, 60)

    y = torch.from_numpy((mline * noisy)[None].astype(np.complex64))
    S = torch.from_numpy(maps[None].astype(np.complex64))
    E = Mask(mask.float()) @ FFT2D() @ Sense(S)
    sigma = torch.tensor(a.sigma)

    lines = [f"R={a.R} slice {a.slice}, {int(mline.sum())} lines, untrained nets",
             f"zero-filled, scaled: NRMSE {score.nrmse(score.zf_scaled):.3f}  ghost 1.000",
             f"CG-SENSE, no network: NRMSE {score.nrmse(cg):.3f}  ghost {score.beta(cg):.3f}", ""]
    cells = build_cells(a.nets)
    results = {}
    for name, make in cells.items():
        for seed in range(a.seeds):
            net = make(name, seed)
            out, rows, coarse, dt = run_net(net, y, E, sigma, score)
            results[(name, seed)] = (out, rows, coarse)
            line = f"{name:<30} seed {seed}: NRMSE {score.nrmse(out):.3f}  ghost {score.beta(out):.3f}  {dt:.0f}s"
            lines.append(line)
            print(line, flush=True)
            for r in rows:
                extra = "".join(f"  {n} {r[n]:.3f}" for n in ("rel", "clip") if n in r)
                lines.append(f"    V{r['cycle'] + 1} {r['stage']:<5} NRMSE {r['nrmse']:.3f}  ghost {r['beta']:.3f}{extra}")
            del net
    txt = "\n".join(lines)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out + ".txt", "w") as f:
        f.write(txt + "\n")
    print(txt)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for (name, seed), (out, rows, _) in results.items():
        if rows:
            axs[0].plot([r["nrmse"] for r in rows], marker=".", label=f"{name} s{seed}")
            axs[1].plot([r["beta"] for r in rows], marker=".", label=f"{name} s{seed}")
    for ax, t, ref in zip(axs, ("NRMSE", "ghost"), (score.nrmse(cg), score.beta(cg))):
        ax.axhline(ref, color="k", ls=":", lw=0.8)
        ax.set_title(t + " by stage: pre, +x, post per V-cycle (dotted: CG-SENSE)", fontsize=9)
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(a.out + "_curves.png", dpi=130)

    rows_ = np.flatnonzero(np.abs(target).mean(1) > 0.15 * np.abs(target).mean(1).max())
    rs = slice(max(rows_[0] - 8, 0), rows_[-1] + 9)
    vmax = np.percentile(score.t, 99.5)
    sel = [(f"{n} s{s}", r[0]) for (n, s), r in results.items() if s == 0] + [("CG-SENSE", cg)]
    fig, ax = plt.subplots(2, len(sel) + 1, figsize=(2.9 * (len(sel) + 1), 7))
    ax[0, 0].imshow(score.t[rs], cmap="gray", vmin=0, vmax=vmax)
    ax[0, 0].set_title("target", fontsize=8)
    ax[1, 0].imshow(np.abs(score.zf_scaled - score.t)[rs], cmap="inferno", vmin=0, vmax=vmax / 3)
    ax[1, 0].set_title(f"zero-filled NRMSE {score.nrmse(score.zf_scaled):.2f}", fontsize=8)
    for i, (n, x) in enumerate(sel, 1):
        ax[0, i].imshow(np.abs(x)[rs], cmap="gray", vmin=0, vmax=vmax)
        ax[0, i].set_title(n.replace("mg81v6vtxAnestU", "mg "), fontsize=8)
        ax[1, i].imshow(np.abs(np.abs(x) - score.t)[rs], cmap="inferno", vmin=0, vmax=vmax / 3)
        ax[1, i].set_title(f"NRMSE {score.nrmse(x):.3f}, ghost {score.beta(x):.3f}", fontsize=8)
    for A in ax.ravel():
        A.axis("off")
    fig.tight_layout()
    fig.savefig(a.out + "_images.png", dpi=130)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
