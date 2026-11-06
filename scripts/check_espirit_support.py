#!/usr/bin/env python3
"""
Does the offline ESPIRiT support cut into the anatomy? Sweep thresh_eig on sampled PDFS slices.

    python scripts/check_espirit_support.py                              # train split, 12 volumes
    python scripts/check_espirit_support.py --thresh 0.95 0.9 0.85 0.8 0.7 --n-volumes 20
    python scripts/check_espirit_support.py --contrast pd                # the PD side scans

ESPIRiT's support is {lam > thresh_eig}, lam the top eigenvalue of the calibration operator,
in [0, 1] and scale-free -- so each slice runs ESPIRiT ONCE (physics/smaps.py::espirit,
return_lam=True, with the offline job's ACS / kernel / row-space settings) and every threshold
is applied to the same lam. The anatomy reference is the Otsu foreground of the full-k-space RSS
(the foreground the intensity statistics use). Per slice and threshold:

    cut        foreground pixels OUTSIDE the support / foreground pixels   (want ~0)
    cut_e      the same, weighted by RSS^2: the share of tissue SIGNAL the maps zero out
    over       support pixels outside the foreground / foreground pixels   (overestimate; cheap)
    sup        support fraction of the FOV

Reported: median, 90th percentile and max over slices per threshold -- the max is what decides,
since one slice with a clipped edge is a hole in that slice's ground truth. A panel (worst
slices at the current threshold: RSS with the support contour at each threshold, and lam) and
results.json go to --out. Needs a GPU for the 640-line knee grid.
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from datasets.fastmri.paired import PairedKneeDataset                 # noqa: E402
from operators.fourier import ifftc                                   # noqa: E402
from physics.object_mask import otsu_threshold                        # noqa: E402
from physics.smaps import espirit                                     # noqa: E402

METRICS = ("cut", "cut_e", "over", "sup")


def slice_metrics(rss, lam, thresholds):
    a = rss[None, None].float()
    fg = (a > otsu_threshold(a))[0, 0]
    nfg = fg.sum().clamp_min(1)
    e = rss.pow(2) * fg
    out = {}
    for t in thresholds:
        sup = lam > t
        out[t] = dict(cut=float((fg & ~sup).sum() / nfg),
                      cut_e=float((e * ~sup).sum() / e.sum().clamp_min(1e-30)),
                      over=float((sup & ~fg).sum() / nfg),
                      sup=float(sup.float().mean()))
    return out, fg


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/knee/guided/tgvn_R8_g1.json")
    ap.add_argument("--split", default="train", choices=("train", "val"))
    ap.add_argument("--contrast", default="pdfs", choices=("pdfs", "pd"))
    ap.add_argument("--n-volumes", type=int, default=12)
    ap.add_argument("--slices-per-volume", type=int, default=5,
                    help="evenly spaced over the WHOLE volume: edge slices are where it clips")
    ap.add_argument("--thresh", type=float, nargs="+", default=[0.95, 0.9, 0.85, 0.8, 0.7, 0.5])
    ap.add_argument("--current", type=float, default=0.95, help="threshold the panel ranks by")
    ap.add_argument("--acs", type=int, default=32)
    ap.add_argument("--kernel-size", type=int, default=8)
    ap.add_argument("--thresh-rowspace", type=float, default=0.05)
    ap.add_argument("--panel-slices", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/espirit_support")
    args = ap.parse_args()
    ts = sorted(set(args.thresh) | {args.current}, reverse=True)

    with open(args.config) as f:
        d = json.load(f)["data"][args.split]
    ds = PairedKneeDataset(d["split_csv"], d["kspace_roots"], scale_fac=1.0, target="rss")
    rng = np.random.default_rng(args.seed)
    pairs = list(ds.pairs)
    if args.n_volumes and args.n_volumes < len(pairs):
        pairs = [pairs[i] for i in sorted(rng.choice(len(pairs), args.n_volumes, replace=False))]
    print(f"{args.contrast} {args.split}: {len(pairs)} volumes x {args.slices_per_volume} slices, "
          f"ACS {args.acs}, kernel {args.kernel_size}, rowspace {args.thresh_rowspace}, "
          f"device {args.device}")

    rows, keep = [], []
    for vi, (pd_path, fs_path, n, _) in enumerate(pairs):
        path = fs_path if args.contrast == "pdfs" else pd_path
        sls = sorted(set(np.linspace(0, n - 1, args.slices_per_volume).round().astype(int)))
        with h5py.File(path, "r") as f:
            k = torch.from_numpy(f["kspace"][sls]).to(args.device, torch.complex64)
        for j, sl in enumerate(sls):
            kj = k[j:j + 1]
            with torch.no_grad():
                _, lam = espirit(kj, acs_size=(args.acs, args.acs), kernel_size=args.kernel_size,
                                 thresh_rowspace=args.thresh_rowspace, thresh_eig=ts[0],
                                 return_lam=True)
            rss = ifftc(kj)[0].abs().pow(2).sum(0).sqrt().cpu()
            lam = lam[0].float().cpu()
            m, fg = slice_metrics(rss, lam, ts)
            rows.append(dict(volume=os.path.basename(path), slice=int(sl), n_slices=int(n),
                             **{f"{k_}@{t:g}": v for t in ts for k_, v in m[t].items()}))
            keep.append((m[args.current]["cut_e"], os.path.basename(path), int(sl), rss, lam, fg))
        print(f"  [{vi + 1}/{len(pairs)}] {os.path.basename(path)}: slices {sls}  "
              f"max cut@{args.current:g} "
              f"{max(r[f'cut@{args.current:g}'] for r in rows[-len(sls):]):.4f}")

    print(f"\n{'thresh':>7} | {'cut  med / p90 / max':>24} | {'cut_e  med / p90 / max':>26} | "
          f"{'over  med / max':>16} | {'sup med':>7}")
    summary = {}
    for t in ts:
        col = lambda k_: np.array([r[f"{k_}@{t:g}"] for r in rows])          # noqa: E731
        c, ce, o, s = col("cut"), col("cut_e"), col("over"), col("sup")
        summary[f"{t:g}"] = {k_: dict(median=float(np.median(v)), p90=float(np.percentile(v, 90)),
                                       max=float(v.max()))
                             for k_, v in zip(METRICS, (c, ce, o, s))}
        print(f"{t:7.2f} | {np.median(c):7.4f} {np.percentile(c, 90):7.4f} {c.max():7.4f} | "
              f"{np.median(ce):8.5f} {np.percentile(ce, 90):8.5f} {ce.max():8.5f} | "
              f"{np.median(o):7.3f} {o.max():7.3f} | {np.median(s):7.3f}")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(dict(args=vars(args), summary=summary, slices=rows), f, indent=2)
    keep.sort(key=lambda r: -r[0])
    panel(os.path.join(args.out, "support.png"), keep[:args.panel_slices], ts, args.current)
    print(f"\nwrote {args.out}/results.json and support.png "
          f"(the {min(len(keep), args.panel_slices)} slices with the most signal cut at "
          f"{args.current:g})")


def panel(path, worst, ts, current):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not worst:
        return
    fig, ax = plt.subplots(len(worst), 2, figsize=(9, 4.6 * len(worst)), squeeze=False)
    cmap = plt.get_cmap("viridis")
    for i, (ce, name, sl, rss, lam, fg) in enumerate(worst):
        a = ax[i, 0]
        a.imshow(rss.sqrt(), cmap="gray")                     # sqrt: shows dim tissue at the rim
        a.contour(fg.numpy(), levels=[0.5], colors="r", linewidths=0.8, linestyles="dotted")
        for j, t in enumerate(ts):
            a.contour((lam > t).numpy(), levels=[0.5], colors=[cmap(j / max(len(ts) - 1, 1))],
                      linewidths=0.9)
        a.set_title(f"{name} sl {sl}, cut_e@{current:g} = {ce:.4f}\n"
                    f"sqrt RSS, red dots = Otsu fg; support at\n"
                    f"{', '.join(f'{t:g}' for t in ts)} (dark -> light)", fontsize=8)
        b = ax[i, 1]
        im = b.imshow(lam, cmap="magma", vmin=0, vmax=1)
        b.set_title("top eigenvalue lam", fontsize=8)
        fig.colorbar(im, ax=b, fraction=0.03)
        for x in (a, b):
            x.set_xticks([])
            x.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
