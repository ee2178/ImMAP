#!/usr/bin/env python3
"""
How large are the analysis codes A^H x, compared with the clip radius lam? For choosing lam0.

    python scripts/code_stats.py --dumps trained_nets/mg_recon --volume path/to/volume.h5 \
        --out results/code_stats/init

Filters are the net's initial ones (random, ||A B|| = 1), so this describes what training starts
from, not a trained net. Measured per level on the images a level sees (restricted by sinc_vertex):

  CODES from the dumps' reference and zero-filled images, and from white noise of level sigma:
        |A^H x| percentiles inside the anatomy, per slice, and the share of codes inside the clip
        at each candidate lam, i.e. |A^H x| < lam. Slice-to-slice spread shows how much a single
        lam can mean across a volume.
  FORCE from a fully sampled volume: at the true image x, the data term pushes with
        E^H (E x - y) and the prior with B clip(A^H x, lam). At an optimum the two cancel, so the
        ratio ||prior|| / ||data|| at each lam says how large lam must be for the prior to matter.
        Coil maps are a low-pass of the calibration lines.
"""

from __future__ import annotations

import argparse
import glob
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
from physics.mask import make_acc_mask, resolve_acs_lines  # noqa: E402
from show_coarse_problems import acs_maps, fft2c, ifft2c  # noqa: E402

LAMS = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
LEVELS = 3


def first_layer(M, seed, P=7):
    from models.lpds import make_lpds_layer
    torch.manual_seed(seed)
    return make_lpds_layer(1, M, P=P, stride=2, is_complex=True, lam0=1e-3, tau0=0.5, theta0=0.0, degrees=1)


def pyramid(x, restrict):
    """[x at level 0, 1, 2], complex (H, W) numpy to complex tensors (1, 1, h, w)."""
    t = torch.from_numpy(x.astype(np.complex64))[None, None]
    out = [t]
    for _ in range(LEVELS - 1):
        out.append(restrict(out[-1], filter="sinc_vertex"))
    return out


def anatomy(m, code):
    """The organ mask on the code grid: a code counts if its pixel block touches the anatomy."""
    t = torch.from_numpy(m.astype(np.float32))[None, None]
    h, w = code.shape[-2:]
    t = torch.nn.functional.adaptive_max_pool2d(t, (h, w))
    return t[0, 0].numpy() > 0


def codes(layer, x, mask):
    with torch.no_grad():
        c = layer.analysis(x).abs().numpy()[0]
    return c[:, anatomy(mask, c)].ravel()


def code_table(dumps, anatomy_name, run, layer, restrict, sigma, lines, per_slice):
    files = sorted(glob.glob(os.path.join(dumps, anatomy_name, run, "eval_dump", "*.h5")))
    stats = {}
    rng = np.random.default_rng(0)
    for p in files:
        with h5py.File(p, "r") as f:
            ref, zf = f["reference"][:], f["zero_filled"][:]
            organ = f["organ_mask"][:] > 0
        vol = os.path.basename(p)[:-3]
        for j in range(ref.shape[0]):
            m = organ[j]
            for kind, img in (("target", ref[j]), ("zero-filled", zf[j])):
                x = img - img.mean()
                mp = m
                for l, t in enumerate(pyramid(x, restrict)):
                    mp_l = m[::2 ** l, ::2 ** l]
                    c = codes(layer, t, mp_l)
                    stats.setdefault((kind, l), []).append(
                        (vol, j, *np.percentile(c, [10, 50, 90, 99]), *[np.mean(c < v) for v in LAMS]))
            if j == 0:      # noise depends only on sigma and the filters
                n = sigma * (rng.standard_normal(ref[j].shape) + 1j * rng.standard_normal(ref[j].shape)) / np.sqrt(2)
                c = codes(layer, torch.from_numpy(n.astype(np.complex64))[None, None], np.ones_like(m))
                stats.setdefault(("noise", 0), []).append(
                    (vol, j, *np.percentile(c, [10, 50, 90, 99]), *[np.mean(c < v) for v in LAMS]))
    lines += ["", f"{anatomy_name}: {len(files)} volume(s), {sum(1 for r in stats[('target', 0)])} slices  "
              f"(median over slices; |code| percentiles, then share of codes inside the clip at lam)",
              f"{'input':<13}{'lvl':<4}{'p10':>8}{'p50':>8}{'p90':>8}{'p99':>8}  "
              + "".join(f"{'<' + format(v, 'g'):>8}" for v in LAMS)]
    for (kind, l), rows in sorted(stats.items(), key=lambda kv: (kv[0][0] != "noise", kv[0][0], kv[0][1])):
        a = np.array([r[2:] for r in rows], float)
        med = np.median(a, axis=0)
        lines.append(f"{kind:<13}L{l:<3}" + "".join(f"{v:>8.4f}" for v in med[:4]) + "  "
                     + "".join(f"{v:>8.0%}" for v in med[4:]))
    rows = stats[("target", 0)]
    p50 = np.array([r[3] for r in rows])
    lines.append(f"target L0 median |code| across slices: min {p50.min():.4f}, max {p50.max():.4f}, "
                 f"ratio {p50.max() / p50.min():.1f}")
    vols = sorted({r[0] for r in rows})
    for v in vols:
        q = np.array([r[3] for r in rows if r[0] == v])
        lines.append(f"    {v:<30} {q.min():.4f} to {q.max():.4f}  ratio {q.max() / q.min():.1f}")
    per_slice[anatomy_name] = rows
    return stats


def force_table(path, layer, R, sigma, scale, lines):
    with h5py.File(path, "r") as f:
        K = f["kspace"][:]
    N, C, H, W = K.shape
    cf = 0.32 / R
    acs = resolve_acs_lines(W, None, cf)
    mask = make_acc_mask(shape=(H, W), accel=R, acs_lines=None, mode="uniform", center_frac=cf,
                         adjust_accel=False).numpy().squeeze()
    mask = (mask if mask.ndim == 1 else mask[0]).astype(float)
    rng = np.random.default_rng(1234)
    ratios = {v: [] for v in LAMS}
    for j in range(N):
        k = K[j].astype(np.complex64) * scale
        S = acs_maps(k, acs)
        x = (np.conj(S) * ifft2c(k)).sum(0)
        n = sigma * (rng.standard_normal(k.shape) + 1j * rng.standard_normal(k.shape)) / np.sqrt(2)
        data = (np.conj(S) * ifft2c(mask * (fft2c(S * x) - k - n))).sum(0)
        t = torch.from_numpy((x - x.mean()).astype(np.complex64))[None, None]
        with torch.no_grad():
            a = layer.analysis(t)
            for v in LAMS:
                z = a * torch.clamp(v / a.abs().clamp_min(1e-12), max=1.0)
                ratios[v].append(float(layer.synthesis(z).norm()) / np.linalg.norm(data))
    lines += ["", f"FORCE, {os.path.basename(path)}, R={R}, {N} slices: ||B clip(A^H x, lam)|| / ||E^H (E x - y)|| "
              "at the true image (median, min to max over slices)"]
    for v in LAMS:
        r = np.array(ratios[v])
        lines.append(f"  lam {v:<7g} {np.median(r):>8.2f}   ({r.min():.2f} to {r.max():.2f})")
    return ratios


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dumps", default="trained_nets/mg_recon")
    p.add_argument("--volume", help="a fully sampled volume, for the force balance")
    p.add_argument("--M", type=int, default=81)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--R", type=int, default=8)
    p.add_argument("--sigma", type=float, default=0.015)
    p.add_argument("--scale", type=float, default=2000.0)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    from models.components import set_complex_mode
    from operators.resample import restrict
    set_complex_mode("planar")
    layer = first_layer(a.M, a.seed)
    lines = [f"Initial filters: M={a.M}, P=7, stride 2, seed {a.seed}, ||A B|| = 1. lam0 now = 1e-3. sigma = {a.sigma}"]
    per = {}
    for anat in ("brain", "knee"):
        code_table(a.dumps, anat, f"lpdsnet_R{a.R}", layer, restrict, a.sigma, lines, per)
    if a.volume:
        force_table(a.volume, layer, a.R, a.sigma, a.scale, lines)
    txt = "\n".join(lines)
    print(txt)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out + ".txt", "w") as f:
        f.write(txt + "\n")

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for A, (anat, rows) in zip(ax, per.items()):
        vols = sorted({r[0] for r in rows})
        for v in vols:
            q = [(r[1], r[3], r[5]) for r in rows if r[0] == v]
            A.plot([x[0] for x in q], [x[1] for x in q], marker=".", label=v.replace("file_brain_AXT2_", ""))
        A.set_yscale("log")
        A.set_title(f"{anat}: median |code| of the target by slice position", fontsize=9)
        A.set_xlabel("stack position")
        A.axhline(1e-3, color="k", ls=":", lw=0.8)
        A.grid(alpha=0.3)
        A.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(a.out + "_slices.png", dpi=130)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
