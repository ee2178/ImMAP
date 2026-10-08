#!/usr/bin/env python3
"""
The matrix sizes and coil counts a fastMRI split actually contains.

    python scripts/fastmri_sizes.py --config config/brain/mg/lpdsnet_R12.json

fastMRI volumes come at a handful of sizes (brain is mostly 640x320 with some
768x396; knee is 640x368 / 640x372), and the cost of a network -- time, memory,
whether the image-domain embedding fires -- depends on which. This reads them
from the files the config's loader would serve and writes the census to
`cache/fastmri_sizes_<anatomy>.json`, so the answer is recorded once instead of
being remembered:

    {"anatomy": "brain", "sizes": [{"H": 640, "W": 320, "coils": 20,
                                    "volumes": 812, "slices": 4060,
                                    "by_split": {"train": {...}, "val": {...}}}]}

`volumes` is the weight that matters for training, which draws ONE slice per
volume per epoch; `slices` counts the slices inside the config's
[start_slice, end_slice) range. `scripts/time_net.py --sizes-from` times every
net on these sizes and weights the mean by `volumes`.

Headers only: no k-space is read, so this takes seconds. It uses the loader's
own file list (the acquisition filter, any `volumes` restriction), so the
census is of what a run on this config sees and nothing else.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def scan_split(cfg, split):
    """`{(H, W, coils): {"volumes": n, "slices": m}}` for one split."""
    import h5py

    from datasets.fastmri.loader import FastMRIDataset

    d = dict(cfg["data"][split])
    for k in ("name", "task", "batch_size"):
        d.pop(k, None)
    ds = FastMRIDataset(task="recon", **d)
    lo, hi = int(ds.start_slice), ds.end_slice
    out = {}
    for fname in ds.files:
        with h5py.File(os.path.join(ds.smap_root, fname), "r") as f:
            n = int(f["image"].shape[0])
            H, W = (int(v) for v in f["image"].shape[-2:])
            coils = int(f["smaps"].shape[-3])
        # the loader's rule: end_slice=None means "always start_slice"
        used = 1 if hi is None else max(min(int(hi), n) - min(lo, n), 0)
        c = out.setdefault((H, W, coils), dict(volumes=0, slices=0))
        c["volumes"] += 1
        c["slices"] += used
    return out


def census(cfg, splits):
    by_split = {s: scan_split(cfg, s) for s in splits}
    keys = sorted({k for d in by_split.values() for k in d})
    sizes = []
    for H, W, coils in keys:
        per = {s: by_split[s][(H, W, coils)] for s in splits if (H, W, coils) in by_split[s]}
        sizes.append(dict(H=H, W=W, coils=coils,
                          volumes=sum(v["volumes"] for v in per.values()),
                          slices=sum(v["slices"] for v in per.values()),
                          by_split=per))
    sizes.sort(key=lambda s: (-s["volumes"], s["H"], s["W"], s["coils"]))
    return sizes


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="any config with a fastMRI data block")
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--out", default=None,
                    help="default: cache/fastmri_sizes_<anatomy>.json")
    ap.add_argument("--pad-multiple", type=int, default=8,
                    help="report each size's embedded grid for this network stride "
                         "(8 = s=2 with three levels)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    splits = [s for s in args.splits if s in cfg.get("data", {})]
    if not splits:
        raise SystemExit(f"{args.config} has none of the data splits {args.splits}")
    anatomy = cfg["data"][splits[0]].get("anatomy", "unknown")
    sizes = census(cfg, splits)
    total = sum(s["volumes"] for s in sizes)

    from operators.truncate import embedded_size
    print(f"{anatomy}: {total} volumes, {len(sizes)} distinct (size, coils) "
          f"over {' + '.join(splits)}\n")
    print(f"  {'size':<10}{'coils':>5}{'volumes':>9}{'share':>7}{'slices':>8}   "
          f"embedded to a multiple of {args.pad_multiple}")
    for s in sizes:
        eh, ew = embedded_size((s["H"], s["W"]), args.pad_multiple)
        grid = f"{eh}x{ew}" + ("" if (eh, ew) == (s["H"], s["W"]) else "  (padded)")
        print(f"  {str(s['H']) + 'x' + str(s['W']):<10}{s['coils']:>5}{s['volumes']:>9}"
              f"{100 * s['volumes'] / total:>6.1f}%{s['slices']:>8}   {grid}")

    out = args.out or os.path.join("cache", f"fastmri_sizes_{anatomy}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump(dict(anatomy=anatomy, config=args.config, splits=splits, sizes=sizes),
                  f, indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
