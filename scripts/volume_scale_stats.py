#!/usr/bin/env python3
"""
Per-VOLUME intensity statistics of the knee PD / PDFS pairs, for per-volume normalization.

    python scripts/volume_scale_stats.py                        # train + val + test splits
    python scripts/volume_scale_stats.py --stride 2 --out config/knee/guided/volume_scales.json

Why per volume: across slices, the foreground p95 spreads ~2x (PDFS) / ~3x (PD) p10 -> p90, but
mostly BETWEEN volumes (76% / 86% of the log-variance; within a volume ~15% sd). Dividing each
volume by its own level removes the between-volume part, so one fixed scale (and an ABSOLUTE
bridge tau) means the same thing on every volume. The loader applies it:
datasets/fastmri/paired.py, `volume_scales` + `volume_stat` / `side_volume_stat`.

Per volume (every pair of the TGVN split CSVs, both contrasts), raw fastMRI units, pooled over
the volume's slices in [--start-slice, --end-slice) every --stride-th, foreground = Otsu on each
slice's own image:

    full_p{q}   q-th percentile of |RSS of the FULL k-space| over the foreground
    acs_p{q}    the same on the ACS-only image: the centre lines a measurement of this volume
                ACTUALLY contains (resolve_acs_lines(W, center_frac): 15 of 368 at 0.04), zero-
                padded, coil RSS. Blurred, so lower than full_p{q}, but available at TEST time.

FAIRNESS. Which statistic a volume is scaled by decides what a test-time pipeline would need:
the TARGET contrast's full_p95 comes from the fully-sampled scan (not available when
reconstructing), its acs_p95 from the measurement itself. The configs scale the target by
acs_p95 and the fully-sampled guide by full_p95 (the guide IS given fully sampled at guide_R 1).
The report gives corr(log acs, log full) over volumes: near 1 means the ACS statistic tracks the
true level and costs nothing.

Output (JSON): {"meta": {...}, "volumes": {"<file>.h5": {"contrast", "n_slices", "full_p95",
"acs_p95", ...}}}. Keyed by file name, which is unique across fastMRI train + val.
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

from datasets.fastmri.paired import read_pairs                        # noqa: E402
from operators.fourier import ifftc                                   # noqa: E402
from physics.mask import resolve_acs_lines                            # noqa: E402
from physics.object_mask import otsu_threshold                        # noqa: E402

QUANTILES = (90, 95, 99)


def volume_stats(path, start, end, stride, center_frac, pixels, rng):
    with h5py.File(path, "r") as f:
        n = f["kspace"].shape[0]
        lo, hi = min(start, n - 1), n if end is None else min(int(end), n)
        sls = list(range(lo, max(hi, lo + 1), stride))
        k = torch.from_numpy(f["kspace"][sls]).to(torch.complex64)        # (S, C, H, W)
    W = k.shape[-1]
    nc = resolve_acs_lines(W, None, center_frac)
    acs = torch.zeros(W, dtype=torch.bool)
    acs[W // 2 - nc // 2: W // 2 - nc // 2 + nc] = True
    pools = {"full": [], "acs": []}
    for j in range(k.shape[0]):
        for name, kk in (("full", k[j]), ("acs", k[j] * acs)):
            img = ifftc(kk).abs().pow(2).sum(0).sqrt()                      # (H, W)
            a = img[None, None]
            fg = (a > otsu_threshold(a))[0, 0]
            v = img[fg].numpy()
            if v.size:
                pools[name].append(v[rng.choice(v.size, min(v.size, pixels), replace=False)])
    out = dict(n_slices=len(sls), acs_lines=nc)
    for name, chunks in pools.items():
        v = np.concatenate(chunks) if chunks else np.zeros(1)
        for q in QUANTILES:
            out[f"{name}_p{q}"] = float(np.percentile(v, q))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", nargs="+", default=[
        f"config/knee/guided/splits/{s}.csv" for s in ("train", "val", "test")])
    ap.add_argument("--kspace-roots", nargs="+", default=[
        "../datasets/fastmri/knee/multicoil_train", "../datasets/fastmri/knee/multicoil_val"])
    ap.add_argument("--start-slice", type=int, default=0)
    ap.add_argument("--end-slice", type=int, default=None)
    ap.add_argument("--stride", type=int, default=2, help="every Nth slice of each volume")
    ap.add_argument("--center-frac", type=float, default=0.04,
                    help="the measurement mask's centre fraction (cfg mri.center_frac)")
    ap.add_argument("--pixels-per-slice", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="config/knee/guided/volume_scales.json")
    args = ap.parse_args()

    def find(name):
        for r in args.kspace_roots:
            p = os.path.join(r, name)
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"{name} is in none of {args.kspace_roots}")

    files = {}
    for csv_path in args.splits:
        for pd_name, pdfs_name in read_pairs(csv_path):
            files[pd_name], files[pdfs_name] = "pd", "pdfs"
    rng = np.random.default_rng(args.seed)
    vols = {}
    for i, (name, contrast) in enumerate(sorted(files.items())):
        try:
            st = volume_stats(find(name), args.start_slice, args.end_slice, args.stride,
                              args.center_frac, args.pixels_per_slice, rng)
        except FileNotFoundError as e:
            print(f"  SKIP {e}")
            continue
        vols[name] = dict(contrast=contrast, **st)
        print(f"  [{i + 1}/{len(files)}] {name} ({contrast}): full_p95 {st['full_p95']:.3e} "
              f"acs_p95 {st['acs_p95']:.3e}", flush=True)

    print()
    for c in ("pdfs", "pd"):
        full = np.array([v["full_p95"] for v in vols.values() if v["contrast"] == c])
        acs = np.array([v["acs_p95"] for v in vols.values() if v["contrast"] == c])
        if full.size < 2:
            continue
        r = float(np.corrcoef(np.log(full), np.log(acs))[0, 1])
        ratio = acs / full
        print(f"{c:<5} {full.size} volumes: full_p95 p10/p90 = "
              f"{np.percentile(full, 90) / np.percentile(full, 10):.2f}; "
              f"corr(log acs_p95, log full_p95) = {r:.3f}; acs/full ratio median "
              f"{np.median(ratio):.3f}, p10-p90 [{np.percentile(ratio, 10):.3f}, "
              f"{np.percentile(ratio, 90):.3f}]")

    meta = dict(splits=args.splits, start_slice=args.start_slice, end_slice=args.end_slice,
                stride=args.stride, center_frac=args.center_frac, quantiles=list(QUANTILES),
                foreground="otsu per slice on |image|", units="raw fastMRI k-space")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + ".partial"
    with open(tmp, "w") as f:
        json.dump(dict(meta=meta, volumes=vols), f, indent=1)
    os.replace(tmp, args.out)
    print(f"\nwrote {args.out} ({len(vols)} volumes)")


if __name__ == "__main__":
    main()
