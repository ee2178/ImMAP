#!/usr/bin/env python3
"""
Control for the multigrid aliasing: does a net with the coarse correction forced to
alpha = 1 still alias, on slices where the free-alpha net does?

    python scripts/alpha_control.py                 # needs a GPU: dumps, then the figure
    python scripts/alpha_control.py --skip-dump     # figure only, from existing dumps

alpha = 1 forces the coarse level's change into the output, so a useless coarse solve
costs the loss directly; that net keeps good coarse images but scores worse. Compare it
with `mg81v6vtxBnest`: the same net with alpha free, so alpha is the only difference.
`mg81v6_R8` and `varnetmaps_R8` are there as the net that aliases and the net that does
not. Read each column at the known artifact against the reference, not by score alone:
a net can lose an artifact by being blurrier.

Every run is dumped with scripts/dump_eval.py on the SAME measurement (measured k-space
by default) into --out/dumps, so nothing under trained_nets/ is overwritten. Each image
is a patch around a known artifact: reference, zero-filled, every net with its rho and
strength there (scripts/alias_flag.py's definitions), and the same for each net's E.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from alias_flag import hp_pe, local_corr, slice_sigma, strength_map  # noqa: E402

RUNS = ["varnetmaps_R8", "mg81v6_R8", "mg81v6vtxBnest_R8", "mg81v6vtxBnestafix_R8"]
# knee R=8 slices where the local ranking found a line the reference lacks, in
# LPDSNet and/or MGLPDS (volume, slice, row, col); see results/alias_knee.
SITES = [("file1000052", 15, 189, 88), ("file1000264", 16, 458, 100), ("file1000033", 22, 502, 298)]
WIN = (25, 9)


def dump(a, volumes):
    cmd = [sys.executable, os.path.join(HERE, "dump_eval.py"), "--runs", a.runs, "--exact",
           "--only", *a.nets, "--volumes", *volumes, "--slices", a.slices,
           "--metrics", "psnr", "ssim", "nrmse", "--val-kspace-type", a.val_kspace,
           "--out", a.dumps]
    print("[control]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def load(a, net, vol, sl):
    path = os.path.join(a.dumps, net, "eval_dump", vol + ".h5")
    if not os.path.exists(path):
        return None
    with h5py.File(path, "r") as h:
        i = list(h["slice_index"][:]).index(sl)
        return dict(recon=np.abs(h["recon"][i]), ref=np.abs(h["reference"][i]),
                    zf=np.abs(h["zero_filled"][i]), organ=h["organ_mask"][i] > 0,
                    psnr=float(h["psnr_slice"][i]), kspace=str(h.attrs.get("kspace_type", "?")))


def figure(a, vol, sl, r0, c0):
    data = {n: load(a, n, vol, sl) for n in a.nets}
    missing = [n for n, d in data.items() if d is None]
    if missing:
        print(f"[control] {vol} s{sl}: no dump for {missing} -- left out", file=sys.stderr)
    data = {n: d for n, d in data.items() if d is not None}
    if not data:
        return
    d0 = next(iter(data.values()))
    ref, zf = d0["ref"], d0["zf"]
    inner = ndi.binary_erosion(d0["organ"], iterations=4)
    sigma = slice_sigma(ref, [d["recon"] for d in data.values()], inner)
    t = hp_pe(zf)
    h = a.patch_size // 2
    crop = (slice(max(r0 - h, 0), r0 + h), slice(max(c0 - h, 0), c0 + h))
    w = (slice(max(r0 - 12, 0), r0 + 13), slice(max(c0 - 4, 0), c0 + 5))
    vmax = max(np.percentile(x[crop], 99.5) for x in [ref, *(d["recon"] for d in data.values())]) or 1.0
    es = {n: hp_pe(d["recon"] - ref) for n, d in data.items()}
    emax = max(np.abs(e[crop]).max() for e in es.values()) or 1.0

    n = len(data) + 2
    fig, axs = plt.subplots(2, n, figsize=(2.5 * n, 5.4))
    axs[0, 0].imshow(ref[crop], cmap="gray", vmin=0, vmax=vmax, interpolation="nearest")
    axs[0, 0].set_title("reference", fontsize=8)
    axs[0, 1].imshow(zf[crop], cmap="gray", vmin=0, vmax=np.percentile(zf[crop], 99.5) or 1.0,
                     interpolation="nearest")
    axs[0, 1].set_title("zero-filled (own scale)", fontsize=8)
    tmax = np.abs(t[crop]).max() or 1.0
    axs[1, 1].imshow(t[crop], cmap="RdBu_r", vmin=-tmax, vmax=tmax, interpolation="nearest")
    axs[1, 1].set_title("T = HP(|ZF|) (own scale)", fontsize=8)
    axs[1, 0].axis("off")
    print(f"\n{vol} s{sl} ({r0},{c0}) kspace={d0['kspace']}")
    for k, (name, d) in enumerate(data.items()):
        rho = local_corr(es[name], t, WIN)[w]
        i = np.unravel_index(np.argmax(rho), rho.shape)
        st = strength_map(es[name], sigma, WIN)[w][i]
        print(f"  {name:24s} rho {rho[i]:+.2f}  strength {st:.1f}  slice PSNR {d['psnr']:.2f}")
        axs[0, k + 2].imshow(d["recon"][crop], cmap="gray", vmin=0, vmax=vmax, interpolation="nearest")
        axs[0, k + 2].set_title(f"{name}\nrho {rho[i]:.2f}, strength {st:.1f}", fontsize=8)
        axs[1, k + 2].imshow(es[name][crop], cmap="RdBu_r", vmin=-emax, vmax=emax, interpolation="nearest")
        axs[1, k + 2].set_title("E (shared scale)", fontsize=8)
    for ax in axs.ravel():
        ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle(f"{vol}  slice {sl}  (array row {r0}, col {c0})  {d0['kspace']}", fontsize=9)
    fig.tight_layout()
    path = os.path.join(a.out, f"control_{vol}_s{sl}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  wrote {path}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default="trained_nets/mg_recon/knee")
    p.add_argument("--nets", nargs="+", default=RUNS, help="run directory names")
    p.add_argument("--out", default="results/alpha_control")
    p.add_argument("--dumps", help="where the dumps are (default: --out/dumps)")
    p.add_argument("--slices", default="12:25", help="slice range dumped per volume")
    p.add_argument("--val-kspace", default="measurement_awgn",
                   choices=["simulated", "measurement_awgn"])
    p.add_argument("--patch-size", type=int, default=64)
    p.add_argument("--skip-dump", action="store_true", help="only draw, from existing dumps")
    a = p.parse_args(argv)
    a.dumps = a.dumps or os.path.join(a.out, "dumps")
    os.makedirs(a.out, exist_ok=True)
    if not a.skip_dump:
        dump(a, sorted({s[0] for s in SITES}))
    for vol, sl, r0, c0 in SITES:
        figure(a, vol, sl, r0, c0)


if __name__ == "__main__":
    sys.exit(main())
