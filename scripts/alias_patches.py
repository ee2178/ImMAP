#!/usr/bin/env python3
"""
Show other nets on the patches an earlier alias_check.py scan flagged.

    python scripts/dump_eval.py --runs trained_nets/mg_recon/knee --exact \
        --only mg81v6vtxAnestUgalcf32_R8 mg81v6vtxAnestUfoldcf32_R8 mg81v6vtxAnestUaz0cf32_R8 \
        --flags results/alias_slices_cf32_all/knee/slices_R8.csv \
        --val-kspace-type measurement_awgn --metrics psnr ssim nrmse --out trained_nets/mg_flagged/knee
    python scripts/alias_patches.py --flags results/alias_slices_cf32_all/knee/slices_R8.csv \
        --runs trained_nets/mg_recon_measured/knee trained_nets/mg_flagged/knee --R 8 \
        --methods lpdsnetcf32 ramcf32 mg81v6vtxAnestUcf32 \
            mg81v6vtxAnestUgalcf32 mg81v6vtxAnestUfoldcf32 mg81v6vtxAnestUaz0cf32 \
        --out results/alias_patches_exp19/knee

For nets that have not been scanned, for example ones still training. dump_eval.py --flags
runs them on the flagged slices only, and this reads the slices and patch locations from
--flags, so nothing is ranked again and no new cutoff is applied. Each flagged slice gets one
image from alias_check.py's render_patch: the reference, zero-filled and every method on the
same crop, each method with its rho and strength there.

--runs takes several trees, and a method is read from the first that has it, so the existing
dumps of the core nets are reused. Those trees need not hold the same slices. The sampling
mask is deterministic, so a slice's aliasing pattern is the same in all of them, but a dump's
noise draw depends on its volume list: a net dumped with --flags sees different noise of the
same level from one dumped whole. List a complete dump first: the reference and zero-filled
come from the first method.

Outputs under --out: patch_R<R>_<rank>_<volume>_s<slice>.png, patches_R<R>.csv (rho and
strength per method and patch) and patches_R<R>.txt, which also records each dump's k-space
type and training step: a net still training is not comparable with a finished one.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from argparse import Namespace

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alias_check import render_patch  # noqa: E402


def read_flags(path, top=0, every=False):
    """The flagged rows of a slices_R<R>.csv in rank order, the first `top` if set."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if rows and "flagged" not in rows[0]:
        raise SystemExit(f"{path} has no `flagged` column: pass alias_check.py --top-slices's slices_R<R>.csv")
    keep = sorted((r for r in rows if every or int(r["flagged"])), key=lambda r: int(r["rank"]))
    return keep[:top] if top else keep


def slice_pos(path, sl):
    """The stack position of source slice `sl` in a dump, or None."""
    with h5py.File(path, "r") as h:
        hit = np.flatnonzero(h["slice_index"][:] == sl)
    return int(hit[0]) if hit.size else None


def find_dumps(roots, methods, R):
    """{method: eval_dump dir}, each from the first tree that has it."""
    out = {}
    for m in methods:
        for root in roots:
            d = os.path.join(root, f"{m}_R{R}", "eval_dump")
            if os.path.isdir(d):
                out[m] = d
                break
    missing = [m for m in methods if m not in out]
    if missing:
        raise SystemExit(f"no dumps for {missing} under {roots}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--flags", required=True, help="slices_R<R>.csv of the earlier scan")
    p.add_argument("--runs", nargs="+", required=True,
                   help="anatomy dirs holding <method>_R<R>/eval_dump/, searched in order")
    p.add_argument("--R", type=int, required=True)
    p.add_argument("--methods", nargs="+", required=True, help="the first must hold zero_filled")
    p.add_argument("--top", type=int, default=0, help="only the top N flagged slices (0: all)")
    p.add_argument("--patch-size", type=int, default=48)
    p.add_argument("--win-rows", type=int, default=25, help="as the scan: window along readout (px)")
    p.add_argument("--win-cols", type=int, default=9, help="as the scan: window along phase-encode (px)")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    flags = read_flags(a.flags, a.top)
    dumps = find_dumps(a.runs, a.methods, a.R)
    os.makedirs(a.out, exist_ok=True)
    opts = Namespace(win_rows=a.win_rows, win_cols=a.win_cols, patch_size=a.patch_size)

    rows, skipped, info = [], [], {}
    for r in flags:
        vol, sl = r["volume"], int(r["slice"])
        pos, gone = {}, []
        for m, d in dumps.items():
            f = os.path.join(d, vol + ".h5")
            k = slice_pos(f, sl) if os.path.exists(f) else None
            if k is None:
                gone.append(m)
            else:
                pos[m] = k
                if m not in info:
                    with h5py.File(f, "r") as h:
                        info[m] = (h.attrs.get("step", "?"), str(h.attrs.get("kspace_type", "?")))
        if gone:
            skipped.append(f"{vol} s{sl}: no dump for {gone}")
            continue
        rk, row, col = int(r["rank"]), int(r["row"]), int(r["col"])
        rhos, stren = render_patch(dumps, a.methods, None, vol, pos, sl, row, col, opts,
                                   os.path.join(a.out, f"patch_R{a.R}_{rk:03d}_{vol}_s{sl}.png"))
        out = dict(rank=rk, volume=vol, slice=sl, row=row, col=col, first_scan_rho=r["rho"],
                   first_scan_best=r.get("best_method", ""))
        for m in a.methods:
            out[f"rho_{m}"], out[f"strength_{m}"] = round(rhos[m], 3), round(stren[m], 1)
        rows.append(out)

    lines = [f"R={a.R}: {len(rows)} patches from {a.flags}, {len(skipped)} skipped", "",
             f"{'method':<34}{'k-space':<20}{'step':>10}{'median rho':>12}"]
    for m in a.methods:
        med = float(np.median([x[f"rho_{m}"] for x in rows])) if rows else float("nan")
        step, kspace = info.get(m, ("?", "?"))
        lines.append(f"{m:<34}{kspace:<20}{step:>10}{med:>12.2f}")
    if len({v[1] for v in info.values()}) > 1:
        lines.append("WARNING: these dumps mix k-space types, so their images do not compare")
    lines += [f"skipped: {s}" for s in skipped]
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(a.out, f"patches_R{a.R}.txt"), "w") as f:
        f.write(txt + "\n")
    if rows:
        with open(os.path.join(a.out, f"patches_R{a.R}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
