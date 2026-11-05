#!/usr/bin/env python3
"""
Where does the aliasing the nets leave on the flagged slices come from? CG-SENSE answers
what no prior is involved in.

    python scripts/cg_sense_check.py --runs trained_nets/mg_flagged_cgsense/knee \
        --flags results/alias_slices_cf32_all/knee/slices_R8.csv --R 8 \
        --methods lpdsnetcf32 mg81v6vtxAnestUcf32 varnetmapscf32 ramcf32 \
        --out results/cg_sense_check/knee_R8

Reads dumps made by `dump_eval.py --flags ... --cg-sense-lam LAM --cg-sense-stored`
(torch/cg_sense_check.sbatch): every net on the flagged slices with the same noise, and
beside them two Tikhonov CG-SENSE solves of the SAME measurement,

  cg_online   (E^H E + lam I) x = E^H y with the operator's maps, the online ESPIRiT
              estimate the nets are given
  cg_stored   the same with the STORED maps, the ones the reference S^H c is made with

No torch, no GPU.

The ghost score is alias_check.py's rho (local correlation of the PE-high-passed
magnitude residual with the PE-high-passed zero-filled image) AT each slice's flagged
pixel, so every image is scored on the same ghost. Each flagged slice is put in one class
(--thr, --gap):

  maps     cg_online leaves the ghost (rho_at > thr - gap) and cg_stored scores at least
           gap lower: the online maps disagree with
           the reference's there, so y is consistent with a ghost under E. No prior on x
           can remove that, only better maps (or a map-robust loss / target).
  data     both CG-SENSE solves leave it: the fold is not resolved by the coils at this R
           (the null space), so only the prior can remove it.
  prior    neither leaves it: a plain linear solve resolves this ghost and the nets do
           not, so it is the nets' failure.
  unclear  anything in between.

Outputs under --out: slices.csv (one row per slice, every image's rho_at and magnitude
NRMSE inside the organ mask), summary.txt, and patch_R<R>_<rank>_<volume>_s<slice>.png
for the --png highest-ranked slices: reference, zero-filled, both CG-SENSE images and
every net, with |x| - |ref| underneath.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import h5py
import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alias_check import hp_pe, local_corr  # noqa: E402

CG = ("cg_online", "cg_stored")


def load_flags(path):
    with open(path, newline="") as f:
        return {(r["volume"], int(r["slice"])): r for r in csv.DictReader(f) if int(r["flagged"])}


def find_dumps(root, methods, R):
    out = {}
    for m in methods:
        d = os.path.join(root, f"{m}_R{R}", "eval_dump")
        if not os.path.isdir(d):
            raise SystemExit(f"no dump for {m} at {d}")
        out[m] = d
    return out


def read(dumps, methods, vol, sl):
    """{name: |image|} for one slice, plus reference, zero-filled and organ mask, or None."""
    ims = {}
    for m in methods:
        with h5py.File(os.path.join(dumps[m], vol + ".h5"), "r") as h:
            idx = np.flatnonzero(h["slice_index"][:] == sl)
            if not len(idx):
                return None
            j = int(idx[0])
            ims[m] = np.abs(h["recon"][j])
            if "ref" not in ims:
                ims["ref"] = np.abs(h["reference"][j])
                ims["zf"] = np.abs(h["zero_filled"][j])
                ims["organ"] = h["organ_mask"][j] > 0
            for key, name in (("cg_sense", "cg_online"), ("cg_sense_stored", "cg_stored")):
                if name not in ims and key in h:
                    ims[name] = np.abs(h[key][j])
    return ims


def score(ims, names, row, col, win):
    t = hp_pe(ims["zf"])
    inner = ndi.binary_erosion(ims["organ"], iterations=4)
    ref = ims["ref"]
    out = {}
    for n in names:
        if n not in ims:
            out[n] = (np.nan, np.nan)
            continue
        rho = local_corr(hp_pe(ims[n] - ref), t, win)
        nrmse = np.linalg.norm((ims[n] - ref)[inner]) / (np.linalg.norm(ref[inner]) + 1e-12)
        out[n] = (float(rho[row, col]), float(nrmse))
    return out


def classify(s, thr, gap):
    on, st = s["cg_online"][0], s["cg_stored"][0]
    if not (np.isfinite(on) and np.isfinite(st)):
        return "unclear"
    # The maps class is judged on the GAP between the two solves (same y, same lam, only
    # the maps differ), so its floor is looser: a ghost the stored maps clearly remove
    # counts even if the online solve leaves it a little under --thr.
    if on > thr - gap and st < on - gap:
        return "maps"
    if on > thr and st > thr:
        return "data"
    if on < thr - gap and st < thr - gap:
        return "prior"
    return "unclear"


def render(ims, names, sc, vol, sl, row, col, cls, a, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    ref = ims["ref"]
    H, W = ref.shape
    half = a.patch // 2
    y0 = min(max(row - half, 0), max(H - a.patch, 0))
    x0 = min(max(col - half, 0), max(W - a.patch, 0))
    crop = (slice(y0, y0 + a.patch), slice(x0, x0 + a.patch))
    shown = [n for n in names if n in ims]
    vmax = np.percentile(ref[crop], 99.5) or 1.0
    errs = {n: ims[n][crop] - ref[crop] for n in shown}
    emax = np.percentile(np.abs(np.stack(list(errs.values()))), 99.5) or 1.0
    nc = len(shown) + 1
    fig, ax = plt.subplots(2, nc, figsize=(2.0 * nc, 4.5), squeeze=False)
    for A in ax.ravel():
        A.axis("off")
    ax[0, 0].imshow(ref[crop], cmap="gray", vmin=0, vmax=vmax, interpolation="nearest")
    ax[0, 0].set_title("reference", fontsize=8)
    ax[1, 0].imshow(ims["zf"][crop], cmap="gray", vmin=0,
                    vmax=np.percentile(ims["zf"][crop], 99.5) or 1.0, interpolation="nearest")
    ax[1, 0].set_title("zero-filled\n(own scale)", fontsize=8)
    win = (a.win_rows, a.win_cols)
    box = (col - x0 - win[1] / 2, row - y0 - win[0] / 2)
    for j, n in enumerate(shown):
        rho, nr = sc[n]
        label = n.replace(f"_R{a.R}", "").replace("cf32", "")
        ax[0, j + 1].imshow(ims[n][crop], cmap="gray", vmin=0, vmax=vmax, interpolation="nearest")
        ax[0, j + 1].set_title(f"{label}\nrho {rho:.2f}  nrmse {nr:.3f}", fontsize=7)
        im = ax[1, j + 1].imshow(errs[n], cmap="RdBu_r", vmin=-emax, vmax=emax, interpolation="nearest")
        ax[1, j + 1].set_title("|x| - |ref|", fontsize=7)
        for P in (ax[0, j + 1], ax[1, j + 1]):
            P.add_patch(Rectangle(box, win[1], win[0], fill=False, ec="#e6a700", lw=0.8, ls="--"))
    fig.colorbar(im, cax=ax[1, -1].inset_axes([1.05, 0.0, 0.07, 1.0]))
    fig.suptitle(f"{vol}  slice {sl}  ghost at row {row}, col {col}   class: {cls}   "
                 f"(CG-SENSE lam {a.lam_label})", fontsize=9)
    fig.savefig(path, dpi=a.dpi, bbox_inches="tight")
    plt.close(fig)


def med(x):
    x = [v for v in x if np.isfinite(v)]
    return np.median(x) if x else np.nan


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", required=True, help="the dump tree (<net>_R<R>/eval_dump/)")
    p.add_argument("--flags", required=True, help="the alias_check.py slices_R<R>.csv")
    p.add_argument("--R", type=int, default=8)
    p.add_argument("--methods", nargs="+", required=True, help="net cells, without _R<R>")
    p.add_argument("--thr", type=float, default=0.6, help="rho above which a ghost is left")
    p.add_argument("--gap", type=float, default=0.2, help="how much lower counts as removed")
    p.add_argument("--png", type=int, default=20, help="figures for the top N ranked slices")
    p.add_argument("--patch", type=int, default=80)
    p.add_argument("--win-rows", type=int, default=25)
    p.add_argument("--win-cols", type=int, default=9)
    p.add_argument("--dpi", type=int, default=130)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    dumps = find_dumps(a.runs, a.methods, a.R)
    with h5py.File(next(os.path.join(dumps[a.methods[0]], f)
                        for f in os.listdir(dumps[a.methods[0]]) if f.endswith(".h5")), "r") as h:
        lam = float(h.attrs.get("cg_sense_lam", -1))
        if "cg_sense" not in h or "cg_sense_stored" not in h:
            raise SystemExit("the dumps hold no cg_sense / cg_sense_stored: make them with "
                             "--cg-sense-lam LAM --cg-sense-stored (torch/cg_sense_check.sbatch)")
    a.lam_label = f"{lam:g}"
    os.makedirs(a.out, exist_ok=True)
    nets = [f"{m}_R{a.R}" for m in a.methods]
    dumps = {f"{m}_R{a.R}": d for m, d in dumps.items()}
    names = list(CG) + nets
    win = (a.win_rows, a.win_cols)

    flags = load_flags(a.flags)
    rows, cache = [], {}
    for (vol, sl), f in sorted(flags.items(), key=lambda kv: int(kv[1]["rank"])):
        if not os.path.exists(os.path.join(dumps[nets[0]], vol + ".h5")):
            continue
        ims = read(dumps, nets, vol, sl)
        r0, c0 = int(f["row"]), int(f["col"])
        if ims is None or r0 < 0:
            continue
        sc = score(ims, names, r0, c0, win)
        cls = classify(sc, a.thr, a.gap)
        row = dict(volume=vol, slice=sl, rank=int(f["rank"]), cls=cls)
        for n in names:
            short = n.replace(f"_R{a.R}", "")
            row[f"rho_{short}"], row[f"nrmse_{short}"] = (round(v, 4) for v in sc[n])
        rows.append(row)
        if len(cache) < a.png:
            cache[(vol, sl)] = (ims, sc, r0, c0, cls, int(f["rank"]))
    if not rows:
        raise SystemExit("no flagged slice found in the dumps")

    with open(os.path.join(a.out, "slices.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    shorts = [n.replace(f"_R{a.R}", "") for n in names]
    L = [f"{len(rows)} flagged slices, R={a.R}, CG-SENSE lam {a.lam_label} | ghost left: rho_at > "
         f"{a.thr:g}; removed: at least {a.gap:g} lower", ""]
    classes = ("maps", "data", "prior", "unclear")
    L.append(f"{'class':<10}{'n':>5}  " + "".join(f"{s[:16]:>17}" for s in shorts))
    for cls in classes + ("all",):
        rr = rows if cls == "all" else [r for r in rows if r["cls"] == cls]
        if not rr:
            continue
        L.append(f"{cls:<10}{len(rr):>5}  " + "".join(f"{med([r[f'rho_{s}'] for r in rr]):>17.2f}"
                                                     for s in shorts))
    L += ["", "median magnitude NRMSE inside the organ mask",
          f"{'class':<10}{'n':>5}  " + "".join(f"{s[:16]:>17}" for s in shorts)]
    for cls in classes + ("all",):
        rr = rows if cls == "all" else [r for r in rows if r["cls"] == cls]
        if not rr:
            continue
        L.append(f"{cls:<10}{len(rr):>5}  " + "".join(f"{med([r[f'nrmse_{s}'] for r in rr]):>17.3f}"
                                                     for s in shorts))
    L += ["", "classes:  maps = online-map CG-SENSE leaves the ghost, stored-map does not;",
          "          data = both leave it;  prior = neither leaves it, the nets do;",
          "          table entries are median rho_at per image (lower = less ghost)"]
    txt = "\n".join(L)
    print(txt)
    with open(os.path.join(a.out, "summary.txt"), "w") as f:
        f.write(txt + "\n")

    for (vol, sl), (ims, sc, r0, c0, cls, rank) in cache.items():
        render(ims, names, sc, vol, sl, r0, c0, cls, a,
               os.path.join(a.out, f"patch_R{a.R}_{rank:03d}_{vol}_s{sl}.png"))
    print(f"wrote slices.csv, summary.txt and {len(cache)} figures to {a.out}")


if __name__ == "__main__":
    main()
