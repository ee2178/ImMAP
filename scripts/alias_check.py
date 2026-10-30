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

--patches N skips all of those tests and saves the N patches with the largest rho
per method instead: patches_R<R>.csv, and a crop of each as
patch_R<R>_<method>_<rank>_<volume>_s<slice>.png. rho is only read at least half a
window inside the eroded organ mask, where the window cannot reach the mask's edge
(E and T share that edge whatever the network did), and one patch is kept per
window-sized neighbourhood. A list of patches to LOOK at: rho says the residual
follows the zero-filled error there, not how strong it is, which is why each row
also has z.

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


def scan_patches(dumps, vol, methods, a):
    """Patch rows [dict] of one volume, with the same reading as scan_volume."""
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
        for m in methods:
            rho, _, z, _ = check_slice(ref[j], zf[j], data[m][j], inner, offsets, a)
            for r, zz, i, c in top_patches(rho, z, inner, a):
                rows.append(dict(volume=vol, slice=int(sl_idx[j]), pos=j, method=m,
                                 kspace_type=kspace[m], rho=round(r, 3), z=round(zz, 1),
                                 row=i, col=c))
    return rows


def run_patches(dumps, vols, R, a):
    rows = []
    for i, v in enumerate(vols):
        print(f"\r  R={R}: {i + 1}/{len(vols)} volumes", end="", file=sys.stderr, flush=True)
        rows += scan_patches(dumps, v, a.methods, a)
    print(file=sys.stderr)
    rows.sort(key=lambda r: -r["rho"])
    kept = [r for m in a.methods for r in [x for x in rows if x["method"] == m][:a.patches]]
    with open(os.path.join(a.out, f"patches_R{R}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(kept[0]))
        w.writeheader()
        w.writerows(kept)
    if len({r["kspace_type"] for r in kept}) > 1:
        print(f"R={R} WARNING: these patches mix k-space types", file=sys.stderr)
    counts = {m: 0 for m in a.methods}
    for r in kept:
        k = counts[r["method"]]
        counts[r["method"]] += 1
        i, c = r["row"], r["col"]
        box = (i - a.win_rows // 2, i + a.win_rows // 2 + 1, c - a.win_cols // 2, c + a.win_cols // 2 + 1)
        render(dumps, a.methods, r["method"], r["volume"], r["pos"], r["slice"],
               (r["rho"], a.win_rows * a.win_cols, i, c, box), a,
               os.path.join(a.out, f"patch_R{R}_{r['method']}_{k:02d}_{r['volume']}_s{r['slice']}.png"),
               label=f"rho {r['rho']:.2f}, z {r['z']:.1f}")
    print(f"R={R}: wrote patches_R{R}.csv and {len(kept)} crops to {a.out}")


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
    p.add_argument("--patches", type=int, default=0,
                   help="save the N largest-rho patches per method instead of flagging")
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
        if a.patches:
            run_patches(dumps, vols, R, a)
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
