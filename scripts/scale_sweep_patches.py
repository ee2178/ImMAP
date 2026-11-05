#!/usr/bin/env python3
"""
Look at what scripts/scale_sweep.py measured: one figure per flagged slice, the patch
around its flagged ghost for every net (rows) at every input scale c (columns).

    python scripts/scale_sweep_patches.py --sweep results/scale_sweep/knee_R8 \
        --flags results/alias_slices_cf32_all/knee/slices_R8.csv --top 12
    python scripts/scale_sweep_patches.py --sweep results/scale_sweep/knee_R8 \
        --flags results/alias_slices_cf32_all/knee/slices_R8.csv --by-drop mg81v6vtxAnestUcf32_R8

Reads only the sweep's dumps (<sweep>/<run>/c<c>/eval_dump/*.h5), its sweep.csv and the
flags CSV: no torch, no GPU, so it runs on a login node or a laptop after an rsync.

Each figure has two blocks, columns = c:
  top     |recon| / c (already divided back, so every panel is in data units), one shared
          grey scale per slice so brightness differences are real
  bottom  |recon| - |reference|, one shared symmetric colour scale per slice: a ghost
          the net left shows as a displaced copy of an edge
The first column holds the reference (top) and the zero-filled image on its own scale
(bottom). Each panel's title carries rho_at and NRMSE from sweep.csv. The dashed box is
alias_check.py's rho window at the flagged pixel.

Slice choice: --top N flagged slices by flag rank (default), or --by-drop RUN for the N
whose rho_at falls most for RUN from c = 1 to --drop-to (default 0.5), or explicit
--slices volume:slice ...

Outputs: <out>/scale_R<R>_<k>_<volume>_s<slice>.png (default out = <sweep>/patches).
"""

from __future__ import annotations

import argparse
import csv
import os

import h5py
import numpy as np


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def discover(sweep):
    """{run: {c: dump dir}} under the sweep's output directory."""
    out = {}
    for run in sorted(os.listdir(sweep)):
        d = os.path.join(sweep, run)
        if not os.path.isdir(d):
            continue
        for sub in os.listdir(d):
            dump = os.path.join(d, sub, "eval_dump")
            if sub.startswith("c") and os.path.isdir(dump):
                out.setdefault(run, {})[float(sub[1:])] = dump
    return {r: v for r, v in out.items() if v}


def read_slice(dump, vol, sl):
    with h5py.File(os.path.join(dump, vol + ".h5"), "r") as h:
        idx = np.flatnonzero(h["slice_index"][:] == sl)
        if not len(idx):
            return None
        j = int(idx[0])
        return {k: h[k][j] for k in ("recon", "reference", "zero_filled") if k in h}


def render(vol, sl, row, col, runs, dumps, scales, stats, a, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    ims, ref, zf = {}, None, None
    for r in runs:
        for c in scales:
            d = read_slice(dumps[r][c], vol, sl)
            if d is None:
                return False
            ims[r, c] = np.abs(d["recon"])
            if ref is None:
                ref, zf = np.abs(d["reference"]), np.abs(d["zero_filled"])
    H, W = ref.shape
    half = a.patch // 2
    y0, x0 = min(max(row - half, 0), max(H - a.patch, 0)), min(max(col - half, 0), max(W - a.patch, 0))
    crop = (slice(y0, y0 + a.patch), slice(x0, x0 + a.patch))
    vmax = np.percentile(ref[crop], 99.5) or 1.0
    errs = {k: v[crop] - ref[crop] for k, v in ims.items()}
    emax = np.percentile(np.abs(np.stack(list(errs.values()))), 99.5) or 1.0

    nr, nc = 2 * len(runs), len(scales) + 1
    fig, ax = plt.subplots(nr, nc, figsize=(1.9 * nc, 2.0 * nr), squeeze=False)
    for A in ax.ravel():
        A.axis("off")
    ax[0, 0].imshow(ref[crop], cmap="gray", vmin=0, vmax=vmax, interpolation="nearest")
    ax[0, 0].set_title("reference", fontsize=8)
    ax[len(runs), 0].imshow(zf[crop], cmap="gray", vmin=0,
                            vmax=np.percentile(zf[crop], 99.5) or 1.0, interpolation="nearest")
    ax[len(runs), 0].set_title("zero-filled\n(own scale)", fontsize=8)
    win = (a.win_rows, a.win_cols)
    box = (col - x0 - win[1] / 2, row - y0 - win[0] / 2)
    for i, r in enumerate(runs):
        short = r.replace("_R" + str(a.R), "")
        for j, c in enumerate(scales):
            st = stats.get((r, vol, sl, c), {})
            tag = f"rho {st.get('rho_at', float('nan')):.2f}  nrmse {st.get('nrmse', float('nan')):.3f}"
            A = ax[i, j + 1]
            A.imshow(ims[r, c][crop], cmap="gray", vmin=0, vmax=vmax, interpolation="nearest")
            # the net's name heads its row in the first scale column
            A.set_title((f"c = {c:g}\n" if i == 0 else "") + (f"{short}\n" if j == 0 else "\n")
                        + tag, fontsize=7)
            B = ax[len(runs) + i, j + 1]
            im = B.imshow(errs[r, c], cmap="RdBu_r", vmin=-emax, vmax=emax, interpolation="nearest")
            B.set_title((f"{short}  |recon| - |ref|\n" if j == 0 else "\n") + tag, fontsize=7)
            for P in (A, B):
                P.add_patch(Rectangle(box, win[1], win[0], fill=False, ec="#e6a700", lw=0.8, ls="--"))
    # beside the first residual row, in an inset, so the last column keeps its width
    fig.colorbar(im, cax=ax[len(runs), -1].inset_axes([1.05, 0.0, 0.07, 1.0]))
    fig.suptitle(f"{vol}  slice {sl}  flagged ghost at row {row}, col {col}.  Input (c y, c sigma), "
                 f"output / c: same noise, same SNR, only intensity changes", fontsize=9)
    fig.savefig(path, dpi=a.dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", required=True, help="scale_sweep.py's --out")
    p.add_argument("--flags", required=True, help="the slices_R<R>.csv the sweep used")
    p.add_argument("--R", type=int, default=8)
    p.add_argument("--runs", nargs="*", help="rows, in order (default: every run in the sweep)")
    p.add_argument("--scales", type=float, nargs="*", help="columns (default: every c dumped)")
    p.add_argument("--top", type=int, default=12, help="how many slices")
    p.add_argument("--by-drop", metavar="RUN", help="pick the slices whose rho_at falls most "
                                                    "for RUN from c = 1 to --drop-to")
    p.add_argument("--drop-to", type=float, default=0.5)
    p.add_argument("--slices", nargs="*", metavar="VOLUME:SLICE", help="explicit slices")
    p.add_argument("--patch", type=int, default=80, help="patch size (px)")
    p.add_argument("--win-rows", type=int, default=25)
    p.add_argument("--win-cols", type=int, default=9)
    p.add_argument("--dpi", type=int, default=130)
    p.add_argument("--out")
    a = p.parse_args(argv)

    dumps = discover(a.sweep)
    if not dumps:
        raise SystemExit(f"no <run>/c<c>/eval_dump under {a.sweep}")
    runs = a.runs or list(dumps)
    missing = [r for r in runs if r not in dumps]
    if missing:
        raise SystemExit(f"not in the sweep: {missing}; have {sorted(dumps)}")
    scales = sorted(a.scales or set.intersection(*[set(dumps[r]) for r in runs]))

    stats = {}
    for r in load_csv(os.path.join(a.sweep, "sweep.csv")):
        stats[r["run"], r["volume"], int(r["slice"]), float(r["c"])] = {
            k: float(r[k]) for k in ("rho_at", "nrmse") if r.get(k) not in (None, "", "nan")}
    flags = {(r["volume"], int(r["slice"])): r for r in load_csv(a.flags)}
    swept = {(v, s) for (_, v, s, _) in stats}

    if a.slices:
        pick = [(v, int(s)) for v, s in (x.rsplit(":", 1) for x in a.slices)]
    else:
        fl = [k for k in swept if int(flags[k]["flagged"])]
        if a.by_drop:
            def drop(k):
                hi = stats.get((a.by_drop, *k, 1.0), {}).get("rho_at", np.nan)
                lo = stats.get((a.by_drop, *k, a.drop_to), {}).get("rho_at", np.nan)
                return hi - lo if np.isfinite(hi - lo) else -np.inf
            pick = sorted(fl, key=drop, reverse=True)[:a.top]
        else:
            pick = sorted(fl, key=lambda k: int(flags[k]["rank"]))[:a.top]

    out = a.out or os.path.join(a.sweep, "patches")
    os.makedirs(out, exist_ok=True)
    n = 0
    for k, (vol, sl) in enumerate(pick):
        f = flags.get((vol, sl))
        if f is None or int(f["row"]) < 0:
            print(f"[skip] {vol} s{sl}: no flagged location in {a.flags}")
            continue
        path = os.path.join(out, f"scale_R{a.R}_{k:03d}_{vol}_s{sl}.png")
        if render(vol, sl, int(f["row"]), int(f["col"]), runs, dumps, scales, stats, a, path):
            n += 1
            print(f"  {path}")
        else:
            print(f"[skip] {vol} s{sl}: not in every dump")
    print(f"wrote {n} figures to {out}")


if __name__ == "__main__":
    main()
