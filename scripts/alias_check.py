#!/usr/bin/env python3
"""
Find residual ALIASING in each network's reconstruction, using the zero-filled
image as the template of what aliasing looks like on that slice.

    python scripts/alias_check.py --runs trained_nets/mg_recon/brain --out results/alias_check/brain
    python scripts/alias_check.py --runs trained_nets/mg_recon/knee --R 12 --png 10 \
        --out results/alias_check/knee

Reads the HDF5 written by scripts/dump_eval.py (`<run>/eval_dump/<volume>.h5`)
and nothing else: no torch, no GPU.

What it measures
----------------
Per slice and network, on magnitudes, all high-passed along phase-encode (HP):

    T = HP(|zero-filled|)     the aliasing template
    E = HP(|recon| - |ref|)   the network's residual
    H = HP(|ref|)             the reference's own edges

    rho  local correlation of E and T, in a sliding window

T shows the slice's aliasing directly: copies of its edges spaced N/d apart. The
true edge is one more copy, attenuated, but an edge the network under-reconstructed
has E ~ -c H and so correlates NEGATIVELY with it, while aliasing a network failed
to remove keeps its sign. So only rho > 0 counts.

    null  the largest rho of T shifted along PE by (k + 1/2) N/d: between the
          aliasing shifts, so the slice's own chance agreement at no real offset
    z     |E| / sigma, sigma = 1.4826 MAD of E over the eroded organ mask

    edge  the largest |H| within 2 px. An under-reconstructed edge has |E| < |H|,
          so a flagged pixel must have |E| > edge.

A pixel is flagged when rho > --rho, rho - null > --margin, z > --z and |E| > edge,
so the flagged pixels follow the line itself rather than the window. Blobs are
8-connected, with at least MIN_PX pixels, MIN_LENGTH px long and MIN_ASPECT times
as long as wide. The slice score is the largest blob's summed z.

The template is the zero-filled image of THAT dump, so the dump's `kspace_type`
is a CSV column and the summary warns when a scan mixes types.

--top-slices N skips all of those tests and ranks whole SLICES by their highest rho
instead, per method: slices_R<R>.csv has every slice, so the ranking can be read at
any rho, and the top N get an image of the patch around that peak, the reference,
zero-filled and every method side by side, each method with its own rho, as slice_R<R>_<method>_<rank>_<volume>_s<slice>.png. A slice counts once, so one volume
can fill at most its own slices of the list. rho is only read at least half a window
inside the eroded organ mask, where the window cannot reach the mask's edge (E and T
share that edge whatever the network did), and only where the residual is strong
enough to see: strength > --min-strength, the rms of E over the window in units of
sigma, the median over the methods of 1.4826 MAD of E, so every method on a slice
shares one scale and a method with more error is not scaled down. rho alone ignores
size, so without this a faint residual can rank as high as a strong one. A slice
with no such place has rho -1. A slice is FLAGGED when its rho > --flag-rho, or the
value for its R in --flag-rho-by-r (knee's rho runs higher than brain's); only
flagged slices get an image, and flags_R<R>.txt counts them per method and k-space
type. Both cutoffs were chosen by eye on about a dozen slices, so a flag is a slice
to LOOK at, not a verdict.

Outputs (under --out): check_R<R>.csv (one row per slice x method), check_R<R>.txt,
and with --png N the N strongest per method: reference, zero-filled, each method,
the scored method's residual and rho map.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import h5py
import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alias_detect import METHODS, discover, find_blobs, viewer_filter  # noqa: E402
from alias_scan import ghost_period  # noqa: E402


def hp_pe(x, k=7):
    """High-pass along phase-encode (axis 1): x minus its k-tap moving average."""
    return x - ndi.uniform_filter1d(x, k, axis=1, mode="nearest")


def local_corr(e, t, win):
    """Local correlation of e and t in a `win` box."""
    def m(a):
        return ndi.uniform_filter(a, win, mode="nearest")

    me, mt = m(e), m(t)
    ee = np.maximum(m(e * e) - me * me, 0)
    tt = np.maximum(m(t * t) - mt * mt, 0)
    return np.clip((m(e * t) - me * mt) / np.sqrt(ee * tt + 1e-12), -1, 1)


def null_offsets(mask_row, n=2):
    """PE shifts halfway between the aliasing shifts, both signs."""
    _, step = ghost_period(mask_row)
    return [int(round(s * (k + 0.5) * step)) for k in range(n) for s in (1, -1)]


def check_slice(ref, zf, rec, inner, offsets, a):
    """(rho, null, z, flagged blobs) for one network on one slice."""
    t, e, h = hp_pe(zf), hp_pe(rec - ref), hp_pe(ref)
    win = (a.win_rows, a.win_cols)
    rho = local_corr(e, t, win)
    null = np.max([local_corr(e, np.roll(t, s, axis=1), win) for s in offsets], axis=0)
    sigma = 1.4826 * np.median(np.abs(e[inner] - np.median(e[inner])))
    z = np.abs(e) / (sigma + 1e-12)
    edge = ndi.maximum_filter(np.abs(h), size=5)
    keep = inner & (rho > a.rho) & (rho - null > a.margin) & (z > a.z) & (np.abs(e) > edge)
    return rho, null, z, find_blobs(z * keep, _blob_args(a))


def _blob_args(a):
    return argparse.Namespace(k=0.0, min_px=a.min_px, min_length=a.min_length,
                              min_aspect=a.min_aspect)


def top_patches(rho, z, inner, a, n=5):
    """The n largest rho [(rho, z, row, col)] of one slice, one per neighbourhood."""
    win = (a.win_rows, a.win_cols)
    core = ndi.binary_erosion(inner, iterations=max(win) // 2)
    r = np.where(core, rho, -np.inf)
    out = []
    for _ in range(n):
        i, j = np.unravel_index(np.argmax(r), r.shape)
        if not np.isfinite(r[i, j]):
            break
        out.append((float(r[i, j]), float(z[i, j]), int(i), int(j)))
        r[max(i - win[0], 0):i + win[0] + 1, max(j - win[0], 0):j + win[0] + 1] = -np.inf
    return out


def slice_sigma(ref, recs, inner):
    """The scale of a slice's residuals, shared by its methods."""
    es = [hp_pe(x - ref)[inner] for x in recs]
    return float(np.median([1.4826 * np.median(np.abs(e - np.median(e))) for e in es]))


def strength_map(e, sigma, win):
    """rms of e over the window, in units of sigma."""
    m = ndi.uniform_filter(e, win, mode="nearest")
    v = ndi.uniform_filter(e * e, win, mode="nearest") - m * m
    return np.sqrt(np.maximum(v, 0)) / (sigma + 1e-12)


def scan_slices(dumps, vol, methods, a):
    """One row [dict] per slice x method: its highest rho and where it is."""
    data, kspace = {}, {}
    for m in methods:
        with h5py.File(os.path.join(dumps[m], vol + ".h5"), "r") as h:
            data[m] = np.abs(h["recon"][:])
            kspace[m] = str(h.attrs.get("kspace_type", "unrecorded"))
            if m == methods[0]:
                ref, zf = np.abs(h["reference"][:]), np.abs(h["zero_filled"][:])
                organ = h["organ_mask"][:] > 0
                mask = h["sampling_mask"][:]
                sl_idx = h["slice_index"][:]
    rows = []
    for j in range(ref.shape[0]):
        inner = ndi.binary_erosion(organ[j], iterations=4)
        if inner.sum() < 100:
            continue
        offsets = null_offsets(mask[j][0])
        sigma = slice_sigma(ref[j], [data[m][j] for m in methods], inner)
        win = (a.win_rows, a.win_cols)
        for m in methods:
            e = hp_pe(data[m][j] - ref[j])
            st = strength_map(e, sigma, win)
            rho = np.where(st > a.min_strength, local_corr(e, hp_pe(zf[j]), win), -np.inf)
            z = np.abs(e) / (sigma + 1e-12)
            top = top_patches(rho, z, inner, a, n=1)
            r, zz, i, c = top[0] if top else (-1.0, 0.0, -1, -1)
            rows.append(dict(volume=vol, slice=int(sl_idx[j]), pos=j, method=m,
                             kspace_type=kspace[m], rho=round(r, 3),
                             strength=round(float(st[i, c]), 1) if top else 0.0,
                             z=round(zz, 1), row=i, col=c))
    return rows


def render_patch(dumps, methods, method, vol, pos, sl, row, col, a, path):
    """The patch around (row, col) in the reference and each method's recon, with that
    method's rho:
    its highest in the window centred there, so a peak a pixel away still counts."""
    ims, rhos = {}, {}
    for m in methods:
        with h5py.File(os.path.join(dumps[m], vol + ".h5"), "r") as h:
            ims[m] = np.abs(h["recon"][pos])
            if m == methods[0]:
                ref, zf = np.abs(h["reference"][pos]), np.abs(h["zero_filled"][pos])
    win = (a.win_rows, a.win_cols)
    t = hp_pe(zf)
    w = (slice(max(row - win[0] // 2, 0), row + win[0] // 2 + 1),
         slice(max(col - win[1] // 2, 0), col + win[1] // 2 + 1))
    organ = None
    with h5py.File(os.path.join(dumps[methods[0]], vol + ".h5"), "r") as h:
        organ = h["organ_mask"][pos] > 0
    sigma = slice_sigma(ref, list(ims.values()), ndi.binary_erosion(organ, iterations=4))
    stren = {}
    for m in methods:
        e = hp_pe(ims[m] - ref)
        rho = local_corr(e, t, win)[w]
        k = np.unravel_index(np.argmax(rho), rho.shape)
        rhos[m], stren[m] = float(rho[k]), float(strength_map(e, sigma, win)[w][k])
    half = a.patch_size // 2
    y0, x0 = max(row - half, 0), max(col - half, 0)
    crop = (slice(y0, row + half), slice(x0, col + half))
    vmax = max(np.percentile(x[crop], 99.5) for x in [ref, *ims.values()]) or 1.0
    fig, axs = plt_subplots(len(methods) + 2)
    axs[0].imshow(ref[crop], cmap="gray", vmin=0, vmax=vmax, interpolation="nearest")
    axs[0].set_title("reference\n", fontsize=9)
    # its own brightness scale: a ghost in zero-filled is faint against anything else
    axs[1].imshow(zf[crop], cmap="gray", vmin=0, vmax=np.percentile(zf[crop], 99.5) or 1.0,
                  interpolation="nearest")
    axs[1].set_title("zero-filled\n(own scale)", fontsize=9)
    axs[0].axis("off")
    axs[1].axis("off")
    for ax, m in zip(axs[2:], methods):
        ax.imshow(ims[m][crop], cmap="gray", vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_title(f"{m}{' (ranked)' if m == method else ''}\nrho {rhos[m]:.2f}, strength {stren[m]:.1f}", fontsize=8)
        ax.axis("off")
    fig.suptitle(f"{vol}  slice {sl}  (array row {row}, col {col})", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt_close(fig)


def plt_subplots(n):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt.subplots(1, n, figsize=(2.6 * n, 3.2))


def plt_close(fig):
    import matplotlib.pyplot as plt
    plt.close(fig)


def parse_by_r(spec):
    """'8:0.85,12:0.80' -> {8: 0.85, 12: 0.8}."""
    return {int(k): float(v) for k, v in (x.split(":") for x in (spec or "").split(",") if x)}


def run_top_slices(dumps, vols, R, a):
    rows = []
    for i, v in enumerate(vols):
        print(f"\r  R={R}: {i + 1}/{len(vols)} volumes", end="", file=sys.stderr, flush=True)
        rows += scan_slices(dumps, v, a.methods, a)
    print(file=sys.stderr)
    rows.sort(key=lambda r: -r["rho"])
    flag_rho = parse_by_r(a.flag_rho_by_r).get(R, a.flag_rho)
    ranked = [dict(r, rank=k, flagged=int(r["rho"] > flag_rho)) for m in a.methods
              for k, r in enumerate(x for x in rows if x["method"] == m)]
    with open(os.path.join(a.out, f"slices_R{R}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ranked[0]))
        w.writeheader()
        w.writerows(ranked)
    if len({r["kspace_type"] for r in ranked}) > 1:
        print(f"R={R} WARNING: this ranking mixes k-space types", file=sys.stderr)
    kept = [r for r in ranked if r["rank"] < a.top_slices and r["flagged"]]
    lines = [f"R={R}: {len(vols)} volumes, {len(ranked) // len(a.methods)} slices | flagged: "
             f"rho > {flag_rho:g} where strength > {a.min_strength:g}, "
             f"window {a.win_rows}x{a.win_cols}", "",
             f"{'method':<14}{'k-space':<20}{'flagged':>9}{'of':>7}{'share':>8}"]
    for m in a.methods:
        for k in sorted({r["kspace_type"] for r in ranked}):
            x = [r for r in ranked if r["method"] == m and r["kspace_type"] == k]
            if x:
                n = sum(r["flagged"] for r in x)
                lines.append(f"{m:<14}{k:<20}{n:>9}{len(x):>7}{100 * n / len(x):>7.0f}%")
    for m in a.methods:
        top = [r for r in ranked if r["method"] == m and r["flagged"]][:a.top_slices]
        lines += ["", f"TOP FLAGGED: {m}", f"{'volume':<30}{'sl':>4}{'rho':>7}{'strength':>10}{'row':>6}{'col':>6}"]
        lines += [f"{r['volume']:<30}{r['slice']:>4}{r['rho']:>7.2f}{r['strength']:>10.1f}{r['row']:>6}{r['col']:>6}"
                  for r in top]
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(a.out, f"flags_R{R}.txt"), "w") as f:
        f.write(txt + "\n")
    for r in kept:
        render_patch(dumps, a.methods, r["method"], r["volume"], r["pos"], r["slice"], r["row"], r["col"], a,
                     os.path.join(a.out, f"slice_R{R}_{r['method']}_{r['rank']:03d}_{r['volume']}_s{r['slice']}.png"))
    print(f"R={R}: wrote slices_R{R}.csv ({len(ranked)} rows) and {len(kept)} crops to {a.out}")


def scan_volume(dumps, vol, methods, a):
    data, kspace = {}, {}
    for m in methods:
        with h5py.File(os.path.join(dumps[m], vol + ".h5"), "r") as h:
            data[m] = np.abs(h["recon"][:])
            kspace[m] = str(h.attrs.get("kspace_type", "unrecorded"))
            if m == methods[0]:
                ref, zf = np.abs(h["reference"][:]), np.abs(h["zero_filled"][:])
                organ = h["organ_mask"][:] > 0
                mask = h["sampling_mask"][:]
                sl_idx = h["slice_index"][:]
    rows, blobs_of = [], {}
    for j in range(ref.shape[0]):
        inner = ndi.binary_erosion(organ[j], iterations=4)
        if inner.sum() < 100:
            continue
        offsets = null_offsets(mask[j][0])
        for m in methods:
            _, _, _, blobs = check_slice(ref[j], zf[j], data[m][j], inner, offsets, a)
            top = blobs[0] if blobs else (0.0, 0, -1, -1, None)
            rows.append(dict(volume=vol, slice=int(sl_idx[j]), pos=j, method=m,
                             kspace_type=kspace[m], score=round(top[0], 2),
                             n_blobs=len(blobs), area=sum(b[1] for b in blobs),
                             row=top[2], col=top[3]))
            blobs_of[(j, m)] = blobs
    return rows, blobs_of


def render(dumps, methods, method, vol, pos, sl, blob, a, path, label=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    ims = {}
    for m in methods:
        with h5py.File(os.path.join(dumps[m], vol + ".h5"), "r") as h:
            ims[m] = np.abs(h["recon"][pos])
            if m == methods[0]:
                ref, zf = np.abs(h["reference"][pos]), np.abs(h["zero_filled"][pos])
                organ = h["organ_mask"][pos] > 0
                offsets = null_offsets(h["sampling_mask"][pos][0])
    inner = ndi.binary_erosion(organ, iterations=4)
    rho, _, _, _ = check_slice(ref, zf, ims[method], inner, offsets, a)
    _, area, cy, cx, (r0, r1, c0, c1) = blob
    half = max(40, (r1 - r0) // 2 + 15, (c1 - c0) // 2 + 15)
    y0, y1, x0, x1 = max(cy - half, 0), cy + half, max(cx - half, 0), cx + half
    vmax = np.percentile(ref[y0:y1, x0:x1], 99.5) or 1.0
    res = (ims[method] - ref)[y0:y1, x0:x1]
    panels = [("reference", ref[y0:y1, x0:x1], "gray", 0, vmax),
              ("zero-filled", zf[y0:y1, x0:x1], "gray", 0, vmax)]
    panels += [(m + (" (scored)" if m == method else ""), ims[m][y0:y1, x0:x1], "gray", 0, vmax)
               for m in methods]
    vr = np.abs(res).max() or 1.0
    panels += [("residual", res, "RdBu_r", -vr, vr), ("rho", rho[y0:y1, x0:x1], "RdBu_r", -1, 1)]
    fig, axs = plt.subplots(1, len(panels), figsize=(2.2 * len(panels), 2.6))
    for ax, (name, img, cm, lo, hi) in zip(axs, panels):
        ax.imshow(img, cmap=cm, vmin=lo, vmax=hi, interpolation="nearest")
        ax.add_patch(Rectangle((c0 - x0 - 0.5, r0 - y0 - 0.5), c1 - c0, r1 - r0,
                               fill=False, ec="lime", lw=0.8))
        ax.set_title(name, fontsize=7)
        ax.axis("off")
    fig.suptitle(f"{vol}  slice {sl}  {method}: {label or f'{area} px, score {blob[0]:.0f}'}"
                 f"  (array row {cy}, col {cx})", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default="trained_nets/mg_recon/brain",
                   help="anatomy dir holding <method>_R<R>/eval_dump/")
    p.add_argument("--R", type=int, nargs="*", help="accelerations (default: all dumped)")
    p.add_argument("--methods", nargs="*", default=METHODS)
    p.add_argument("--win-rows", type=int, default=25, help="window along readout (px)")
    p.add_argument("--win-cols", type=int, default=9, help="window along phase-encode (px)")
    p.add_argument("--rho", type=float, default=0.8, help="smallest partial correlation")
    p.add_argument("--margin", type=float, default=0.2, help="smallest rho above the null")
    p.add_argument("--z", type=float, default=3.0, help="smallest |residual| / sigma")
    p.add_argument("--min-px", type=int, default=8, help="smallest blob kept")
    p.add_argument("--min-length", type=float, default=10.0, help="shortest blob kept (px)")
    p.add_argument("--min-aspect", type=float, default=3.0, help="smallest length / width kept")
    p.add_argument("--top", type=int, default=20, help="detections listed per method")
    p.add_argument("--png", type=int, default=10, help="crops saved per method (0: none)")
    p.add_argument("--min-strength", type=float, default=2.5,
                   help="--top-slices: smallest window rms of E, in units of sigma")
    p.add_argument("--flag-rho", type=float, default=0.75,
                   help="--top-slices: a slice is flagged when its rho is above this")
    p.add_argument("--flag-rho-by-r", default="",
                   help="--top-slices: per-R --flag-rho, e.g. 8:0.85,12:0.80")
    p.add_argument("--patch-size", type=int, default=48, help="patch shown by --top-slices (px)")
    p.add_argument("--top-slices", type=int, default=0,
                   help="rank slices by their highest rho and save the top N per method, "
                        "instead of flagging")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    runs = discover(a.runs)
    if not runs:
        raise SystemExit(f"no <method>_R<R>/eval_dump/*.h5 under {a.runs}")
    os.makedirs(a.out, exist_ok=True)
    for R in sorted(a.R or runs):
        dumps = {m: runs[R][m] for m in a.methods if m in runs.get(R, {})}
        missing = [m for m in a.methods if m not in dumps]
        if missing:
            print(f"R={R}: no dumps for {missing} -- skipped", file=sys.stderr)
            continue
        vols = sorted(set.intersection(*[
            {os.path.basename(f)[:-3] for f in glob.glob(os.path.join(d, "*.h5"))}
            for d in dumps.values()]))
        if a.top_slices:
            run_top_slices(dumps, vols, R, a)
            continue
        rows, blobs = [], {}
        for i, v in enumerate(vols):
            print(f"\r  R={R}: {i + 1}/{len(vols)} volumes", end="", file=sys.stderr, flush=True)
            r, b = scan_volume(dumps, v, a.methods, a)
            rows += r
            blobs.update({(v,) + k: bb for k, bb in b.items()})
        print(file=sys.stderr)

        csv_path = os.path.join(a.out, f"check_R{R}.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

        lines = [f"R={R}: {len(vols)} volumes, {len(rows) // len(a.methods)} slices | "
                 f"rho > {a.rho:g}, rho - null > {a.margin:g}, z > {a.z:g}, "
                 f"window {a.win_rows}x{a.win_cols}"]
        kinds = {}
        for r in rows:
            kinds.setdefault(r["kspace_type"], set()).add(r["volume"])
        if len(kinds) > 1:
            lines.append("WARNING: this scan mixes k-space types, so scores are not comparable: "
                         + ", ".join(f"{k} ({len(v)} volumes)" for k, v in sorted(kinds.items())))
        lines += ["", f"{'method':<14}{'slices w/ alias':>16}{'median score':>14}{'max score':>11}"]
        for m in a.methods:
            s = np.array([r["score"] for r in rows if r["method"] == m])
            lines.append(f"{m:<14}{100 * np.mean(s > 0):>15.0f}%{np.median(s):>14.0f}{s.max():>11.0f}")
        for m in a.methods:
            top = sorted((r for r in rows if r["method"] == m and r["score"] > 0),
                         key=lambda r: -r["score"])[:a.top]
            lines += ["", f"STRONGEST: {m}",
                      f"{'volume':<30}{'sl':>4}{'score':>8}{'blobs':>7}{'area':>7}"
                      f"{'row':>6}{'col':>6}   viewer filter"]
            for k, r in enumerate(top):
                lines.append(f"{r['volume']:<30}{r['slice']:>4}{r['score']:>8.0f}{r['n_blobs']:>7}"
                             f"{r['area']:>7}{r['row']:>6}{r['col']:>6}"
                             f"   {viewer_filter(r['volume'], r['slice'])}")
                if k < a.png:
                    render(dumps, a.methods, m, r["volume"], r["pos"], r["slice"],
                           blobs[(r["volume"], r["pos"], m)][0], a,
                           os.path.join(a.out, f"R{R}_{m}_{k:02d}_{r['volume']}_s{r['slice']}.png"))
        txt = "\n".join(lines)
        print(txt)
        with open(os.path.join(a.out, f"check_R{R}.txt"), "w") as f:
            f.write(txt + "\n")


if __name__ == "__main__":
    sys.exit(main())
