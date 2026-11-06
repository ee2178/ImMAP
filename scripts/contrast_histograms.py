#!/usr/bin/env python3
"""
Intensity histograms of the knee PD / PDFS pairs at RAW scale, to pick a per-contrast scale_fac.

    python scripts/contrast_histograms.py                         # train split, 20 volumes
    python scripts/contrast_histograms.py --split val --n-volumes 40 --slices-per-volume 8
    python scripts/contrast_histograms.py --percentiles 90 95 99 --out results/contrast_hist

The data come from a guided config (default config/knee/guided/tgvn_R8_g1.json): its split CSV,
k-space roots, offline PDFS maps and training slice range -- the slices the nets train on --
read with scale_fac = 1, so every number below is in raw fastMRI units. Per sampled slice:

    pdfs_sense   |offline SENSE image| of the PDFS slice: the training TARGET (target "sense")
    pdfs_rss     RSS of the PDFS k-space (what the target would be under target "rss")
    pd_rss       RSS of the PD k-space: the guide contrast (the net sees its SENSE combination
                 under the online maps, which has the same magnitude up to the map estimate)

Each image is split into foreground (Otsu on its own RSS, as sb/i2sb_recon.py::slice_scale) and
the whole FOV. Reported per contrast:

    pooled percentiles   over all sampled pixels (foreground, and whole image)
    per-slice p95        how much one slice's level varies across slices and within a volume
                         (min / median / max, and the max/min ratio): what a single fixed
                         scale_fac per contrast leaves unnormalized
    scale_fac @ pXX      1 / pooled pXX: the factor that puts the XX-th percentile at 1.0, for
                         --percentiles, from the foreground AND the whole-image percentile
    current              what the configs' scale_fac (5e3) maps those percentiles to

Outputs under --out: histograms.png (per contrast: pixel histogram with the percentiles marked;
per-slice p95 against slice index, one line per volume), slices.csv, summary.json.
"""

import argparse
import csv
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

CONTRASTS = ("pdfs_sense", "pdfs_rss", "pd_rss")
REGIONS = ("fg", "all")
SLICE_Q = (50, 90, 95, 99, 99.9, 100)


def rss(k):
    return ifftc(k).abs().pow(2).sum(dim=0).sqrt()                    # (H, W)


def foreground(mag_rss):
    a = mag_rss[None, None].float()
    return (a > otsu_threshold(a))[0, 0]


def sample_slices(n, lo, hi, k):
    lo, hi = min(lo, n - 1), n if hi is None else min(int(hi), n)
    hi = max(hi, lo + 1)
    return sorted(set(np.linspace(lo, hi - 1, num=min(k, hi - lo)).round().astype(int).tolist()))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/knee/guided/tgvn_R8_g1.json")
    ap.add_argument("--split", default="train", choices=("train", "val"))
    ap.add_argument("--n-volumes", type=int, default=20, help="random volumes (0: all)")
    ap.add_argument("--slices-per-volume", type=int, default=6,
                    help="evenly spaced over the config's training slice range")
    ap.add_argument("--percentiles", type=float, nargs="+", default=[90, 95, 99])
    ap.add_argument("--pixels-per-slice", type=int, default=20000,
                    help="random pixels kept per slice and region for the pooled histogram")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/contrast_hist")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    d = cfg["data"][args.split]
    cur = float(d.get("scale_fac", 1.0))
    sense = d.get("target") == "sense"
    ds = PairedKneeDataset(d["split_csv"], d["kspace_roots"], scale_fac=1.0,
                           target=d.get("target", "rss"), smap_root=d.get("smap_root"))
    rng = np.random.default_rng(args.seed)
    pairs = list(ds.pairs)
    if args.n_volumes and args.n_volumes < len(pairs):
        pairs = [pairs[i] for i in sorted(rng.choice(len(pairs), args.n_volumes, replace=False))]
    contrasts = CONTRASTS if sense else tuple(c for c in CONTRASTS if c != "pdfs_sense")
    print(f"{args.split}: {len(pairs)}/{len(ds.pairs)} volumes, <= {args.slices_per_volume} "
          f"slices each in [{d.get('start_slice', 0)}, {d.get('end_slice')}), raw scale "
          f"(configs use scale_fac {cur:g})")

    pooled = {(c, r): [] for c in contrasts for r in REGIONS}
    rows = []
    for vi, (pd_path, fs_path, n, map_path) in enumerate(pairs):
        sls = sample_slices(n, int(d.get("start_slice", 0)), d.get("end_slice"),
                            args.slices_per_volume)
        with h5py.File(fs_path, "r") as f:
            k_fs = torch.from_numpy(f["kspace"][sls])
        with h5py.File(pd_path, "r") as f:
            k_pd = torch.from_numpy(f["kspace"][sls])
        img = None
        if sense:
            with h5py.File(map_path, "r") as f:
                img = torch.from_numpy(f["image"][sls]).abs()
        for j, sl in enumerate(sls):
            r_fs, r_pd = rss(k_fs[j]), rss(k_pd[j])
            mags = {"pdfs_rss": (r_fs, foreground(r_fs)), "pd_rss": (r_pd, foreground(r_pd))}
            if sense:
                mags["pdfs_sense"] = (img[j].float(), foreground(r_fs))
            for c in contrasts:
                m, fg = mags[c]
                vals = {"fg": m[fg].numpy(), "all": m.flatten().numpy()}
                row = dict(volume=os.path.basename(fs_path if c != "pd_rss" else pd_path),
                           vol_idx=vi, slice=sl, contrast=c, fg_frac=float(fg.float().mean()))
                for r in REGIONS:
                    v = vals[r]
                    if v.size == 0:
                        continue
                    for q in SLICE_Q:
                        row[f"{r}_p{q:g}"] = float(np.percentile(v, q))
                    keep = rng.choice(v.size, min(v.size, args.pixels_per_slice), replace=False)
                    pooled[(c, r)].append(v[keep])
                rows.append(row)
        print(f"  [{vi + 1}/{len(pairs)}] {os.path.basename(fs_path)}: slices {sls}")

    pooled = {k: np.concatenate(v) for k, v in pooled.items() if v}
    summary = {"config": args.config, "split": args.split, "n_volumes": len(pairs),
               "n_slices": len(rows) // len(contrasts), "current_scale_fac": cur,
               "contrasts": {}}
    qs = list(args.percentiles)
    print(f"\n{'contrast':<12}{'region':<6}" + "".join(f"{'p' + format(q, 'g'):>11}" for q in qs)
          + f"{'max':>11}  |  scale_fac to put p -> 1: "
          + "  ".join(f"p{q:g}" for q in qs))
    for c in contrasts:
        ent = {}
        for r in REGIONS:
            v = pooled[(c, r)]
            pv = {f"p{q:g}": float(np.percentile(v, q)) for q in qs}
            ent[r] = dict(percentiles=pv, max=float(v.max()),
                          scale_fac={k: 1.0 / x for k, x in pv.items() if x > 0},
                          at_current_scale={k: x * cur for k, x in pv.items()})
            print(f"{c:<12}{r:<6}" + "".join(f"{x:11.3e}" for x in pv.values())
                  + f"{v.max():11.3e}  |  "
                  + "  ".join(f"{1 / x:9.4g}" for x in pv.values()))
        p95 = np.array([row["fg_p95"] for row in rows if row["contrast"] == c and "fg_p95" in row])
        within = []
        for vi in range(len(pairs)):
            v = [row["fg_p95"] for row in rows if row["contrast"] == c and row["vol_idx"] == vi
                 and "fg_p95" in row]
            if len(v) > 1:
                within.append(max(v) / min(v))
        ent["slice_fg_p95"] = dict(min=float(p95.min()), median=float(np.median(p95)),
                                   max=float(p95.max()), max_over_min=float(p95.max() / p95.min()),
                                   within_volume_max_over_min_median=(
                                       float(np.median(within)) if within else None))
        summary["contrasts"][c] = ent
    print("\nper-slice foreground p95 (raw):")
    for c in contrasts:
        s = summary["contrasts"][c]["slice_fg_p95"]
        print(f"  {c:<12} min {s['min']:.3e}  median {s['median']:.3e}  max {s['max']:.3e}  "
              f"(max/min {s['max_over_min']:.2f}, median within-volume max/min "
              f"{s['within_volume_max_over_min_median'] or float('nan'):.2f})")
    print(f"\nat the current scale_fac {cur:g}, the pooled foreground percentiles are:")
    for c in contrasts:
        a = summary["contrasts"][c]["fg"]["at_current_scale"]
        print(f"  {c:<12} " + "  ".join(f"{k} {x:.3f}" for k, x in a.items()))
    if "pdfs_sense" in summary["contrasts"] and "pd_rss" in summary["contrasts"]:
        for q in qs:
            k = f"p{q:g}"
            r = (summary["contrasts"]["pd_rss"]["fg"]["percentiles"][k]
                 / summary["contrasts"]["pdfs_sense"]["fg"]["percentiles"][k])
            print(f"  PD / PDFS foreground {k}: {r:.2f}x (one shared scale_fac leaves the guide "
                  f"this much brighter)")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    keys = sorted({k for row in rows for k in row}, key=lambda k: (k[0] != "v", k))
    with open(os.path.join(args.out, "slices.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    plot(os.path.join(args.out, "histograms.png"), contrasts, pooled, rows, qs, cur)
    print(f"\nwrote {args.out}/summary.json, slices.csv, histograms.png")


def plot(path, contrasts, pooled, rows, qs, cur):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(len(contrasts), 2, figsize=(13, 3.6 * len(contrasts)), squeeze=False)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, c in enumerate(contrasts):
        a = ax[i, 0]
        fg, al = pooled[(c, "fg")] * cur, pooled[(c, "all")] * cur
        hi = float(np.percentile(fg, 99.9)) * 1.3
        bins = np.linspace(0, hi, 200)
        a.hist(al, bins=bins, histtype="step", color="0.5", label="whole image", density=True)
        a.hist(fg, bins=bins, histtype="stepfilled", alpha=0.45, color=colors[0],
               label="foreground", density=True)
        a.set_yscale("log")
        for j, q in enumerate(qs):
            x = float(np.percentile(fg, q))
            a.axvline(x, color=colors[(j + 1) % len(colors)], ls="--",
                      label=f"fg p{q:g} = {x:.3f}")
        a.set_title(f"{c}  (x scale_fac {cur:g})")
        a.set_xlabel("|image|")
        a.legend(fontsize=8)
        b = ax[i, 1]
        for vi in sorted({r["vol_idx"] for r in rows}):
            pts = [(r["slice"], r["fg_p95"] * cur) for r in rows
                   if r["contrast"] == c and r["vol_idx"] == vi and "fg_p95" in r]
            if pts:
                xs, ys = zip(*pts)
                b.plot(xs, ys, "-o", ms=3, lw=1, alpha=0.7)
        b.set_title(f"{c}: per-slice foreground p95, one line per volume")
        b.set_xlabel("slice index")
        b.set_ylabel(f"p95 x {cur:g}")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
