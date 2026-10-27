#!/usr/bin/env python3
"""
Find NEW LINES in each network's reconstruction: thin structures the reference
does not have. A candidate detector for aliasing artifacts, to be checked by eye.

    python scripts/alias_detect.py --runs trained_nets/mg_recon/knee --out results/alias_detect/knee
    python scripts/alias_detect.py --runs trained_nets/mg_recon/brain --R 8 --png 15 \
        --out results/alias_detect/brain

Reads the HDF5 written by scripts/dump_eval.py (`<run>/eval_dump/<volume>.h5`)
and nothing else: no torch, no GPU. Runs on a login or CPU node.

What it measures
----------------
Every network is scored against the REFERENCE only -- never against the other
networks -- so each gets its own number. Per slice, on magnitudes:

    L(u)  line strength: |l1| - |l2| of the Hessian of u at scale S (|l1| >= |l2|),
          times S^2. High on thin lines of any orientation, low on flat regions
          and round blobs.
    z     ( L(|recon|) - maxfilter_TOL( L(|ref|) ) ) / sigma
          line strength the reference does not have. The max filter lets a
          reference line move by TOL//2 px without counting (a shifted edge is
          not a new line). sigma = 1.4826 MAD of L(|ref|) over the eroded organ
          mask: the SAME scale for every network on that slice.
    blobs 8-connected components of z > K with at least MIN_PX pixels.

Blur weakens lines, so it cannot raise z; only an added line can. The slice
score is the largest blob's summed z (`score`); `n_blobs` and `area` count all
of them.

Calibrated on two artifacts judged by eye (2026-10-27): the LPDSNet streak on
brain R8 file_brain_AXT2_200_2000182 slice 0 (peak z ~4.5 for LPDSNet, < 1 for
E2E-VarNet and MGLPDS) and a line beside an edge on knee R12 file1000033 slice 21
(z 7-18, all three networks). Faint streaks in line-rich anatomy (brain) sit
close to K, so the ranking is a list of candidates to LOOK at, not a verdict.

Outputs (under --out): detect_R<R>.csv (one row per slice x method),
detect_R<R>.txt (summary + the top detections per method, with the web viewer's
filter string), and with --png N the N strongest detections per method as
R<R>_<method>_<rank>_<volume>_s<slice>.png: reference, zero-filled, every
method, and the scored method's z map, cropped around the blob (a box marks it).
Knee crops are flipped up-down like the viewer (figures/ FLIP 'ud').
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import os
import re
import sys

import h5py
import numpy as np
from scipy import ndimage as ndi

RUN_RE = re.compile(r"^(?P<method>.+)_R(?P<R>\d+)$")
METHODS = ["varnetmaps", "lpdsnet", "mg81v6"]


def discover(root):
    """{R: {method: eval_dump dir}} for every run with at least one dump."""
    out = collections.defaultdict(dict)
    for d in sorted(glob.glob(os.path.join(root, "*", "eval_dump"))):
        m = RUN_RE.match(os.path.basename(os.path.dirname(d)))
        if m and glob.glob(os.path.join(d, "*.h5")):
            out[int(m["R"])][m["method"]] = d
    return out


def line_strength(u, s):
    hrr = ndi.gaussian_filter(u, s, order=(2, 0))
    hcc = ndi.gaussian_filter(u, s, order=(0, 2))
    hrc = ndi.gaussian_filter(u, s, order=(1, 1))
    half_tr = (hrr + hcc) / 2
    disc = np.sqrt(np.maximum(half_tr ** 2 - (hrr * hcc - hrc ** 2), 0))
    a, b = np.abs(half_tr + disc), np.abs(half_tr - disc)
    return (np.maximum(a, b) - np.minimum(a, b)) * s ** 2


def new_lines(rec, ref_lines, sigma, organ, a):
    """z map and its blobs [(mass, area, row, col, (r0, r1, c0, c1))], strongest first."""
    z = (line_strength(rec, a.scale) - ref_lines) / sigma * organ
    lab, n = ndi.label(z > a.k, structure=np.ones((3, 3)))
    if not n:
        return z, []
    idx = np.arange(1, n + 1)
    area = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    mass = ndi.sum(z, lab, index=idx)
    boxes = ndi.find_objects(lab)
    blobs = []
    for j in np.flatnonzero(area >= a.min_px):
        sl = boxes[j]
        cy, cx = ndi.center_of_mass(lab == j + 1)
        blobs.append((float(mass[j]), int(area[j]), int(cy), int(cx),
                      (sl[0].start, sl[0].stop, sl[1].start, sl[1].stop)))
    return z, sorted(blobs, reverse=True)


def scan_volume(dumps, vol, methods, a):
    data = {}
    for m in methods:
        with h5py.File(os.path.join(dumps[m], vol + ".h5"), "r") as h:
            data[m] = np.abs(h["recon"][:])
            if m == methods[0]:
                ref = np.abs(h["reference"][:])
                organ = h["organ_mask"][:] > 0
                sl_idx = h["slice_index"][:]
    rows, blobs_of = [], {}
    for j in range(ref.shape[0]):
        inner = ndi.binary_erosion(organ[j], iterations=4)
        if inner.sum() < 100:
            continue
        lr = line_strength(ref[j], a.scale)
        sigma = 1.4826 * np.median(np.abs(lr[inner] - np.median(lr[inner])))
        if not sigma > 0:
            continue
        ref_lines = ndi.maximum_filter(lr, size=a.tol)
        for m in methods:
            _, blobs = new_lines(data[m][j], ref_lines, sigma, inner, a)
            top = blobs[0] if blobs else (0.0, 0, -1, -1, None)
            rows.append(dict(volume=vol, slice=int(sl_idx[j]), pos=j, method=m,
                             score=round(top[0], 2), n_blobs=len(blobs),
                             area=sum(b[1] for b in blobs), row=top[2], col=top[3]))
            blobs_of[(j, m)] = blobs
    return rows, blobs_of


def render(dumps, methods, method, vol, pos, sl, blob, anatomy, a, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    ims = {}
    for m in methods:
        with h5py.File(os.path.join(dumps[m], vol + ".h5"), "r") as h:
            ims[m] = np.abs(h["recon"][pos])
            if m == methods[0]:
                ref = np.abs(h["reference"][pos])
                zf = np.abs(h["zero_filled"][pos])
                organ = h["organ_mask"][pos] > 0
    inner = ndi.binary_erosion(organ, iterations=4)
    lr = line_strength(ref, a.scale)
    sigma = 1.4826 * np.median(np.abs(lr[inner] - np.median(lr[inner])))
    z, _ = new_lines(ims[method], ndi.maximum_filter(lr, size=a.tol), sigma, inner, a)

    _, area, cy, cx, (r0, r1, c0, c1) = blob
    half = max(40, (r1 - r0) // 2 + 15, (c1 - c0) // 2 + 15)
    y0, y1 = max(cy - half, 0), min(cy + half, ref.shape[0])
    x0, x1 = max(cx - half, 0), min(cx + half, ref.shape[1])
    flip = anatomy == "knee"

    def crop(img):
        c = img[y0:y1, x0:x1]
        return c[::-1] if flip else c

    vmax = np.percentile(ref[y0:y1, x0:x1], 99.5) or 1.0
    cols = [("reference", ref), ("zero-filled", zf)] + [(m, ims[m]) for m in methods]
    fig, axs = plt.subplots(1, len(cols) + 1, figsize=(2.2 * (len(cols) + 1), 2.5))
    for ax, (name, img) in zip(axs, cols):
        ax.imshow(crop(img), cmap="gray", vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_title(name + (" (scored)" if name == method else ""), fontsize=7)
    axs[-1].imshow(crop(z), cmap="magma", vmin=0, vmax=max(2 * a.k, 8), interpolation="nearest")
    axs[-1].set_title(f"{method}: new-line z", fontsize=7)
    by0 = (y1 - r1) if flip else (r0 - y0)
    for ax in axs:
        ax.add_patch(Rectangle((c0 - x0 - 0.5, by0 - 0.5), c1 - c0, r1 - r0,
                               fill=False, ec="cyan", lw=0.8))
        ax.axis("off")
    fig.suptitle(f"{vol}  slice {sl}  {method}: blob {area} px, score {blob[0]:.0f}"
                 f"  (array row {cy}, col {cx})", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def viewer_filter(vol, sl):
    tail = vol.split("_")[-1] if vol.startswith("file_brain") else vol
    return f"{tail} {sl}"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default="trained_nets/mg_recon/brain",
                   help="anatomy dir holding <method>_R<R>/eval_dump/")
    p.add_argument("--R", type=int, nargs="*", help="accelerations (default: all dumped)")
    p.add_argument("--methods", nargs="*", default=METHODS)
    p.add_argument("--scale", type=float, default=1.5, help="Hessian scale S (px)")
    p.add_argument("--tol", type=int, default=3, help="max-filter size on the reference's lines")
    p.add_argument("-k", type=float, default=3.0, help="z threshold for a blob")
    p.add_argument("--min-px", type=int, default=8, help="smallest blob kept")
    p.add_argument("--top", type=int, default=20, help="detections listed per method")
    p.add_argument("--png", type=int, default=10, help="crops saved per method (0: none)")
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
        with h5py.File(os.path.join(dumps[a.methods[0]], vols[0] + ".h5"), "r") as h:
            anatomy = str(h.attrs.get("anatomy", ""))
        rows, blobs = [], {}
        for i, v in enumerate(vols):
            print(f"\r  R={R}: {i + 1}/{len(vols)} volumes", end="", file=sys.stderr, flush=True)
            r, b = scan_volume(dumps, v, a.methods, a)
            rows += r
            blobs.update({(v,) + k: bb for k, bb in b.items()})
        print(file=sys.stderr)

        csv_path = os.path.join(a.out, f"detect_R{R}.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

        n_sl = len(rows) // len(a.methods)
        lines = [f"R={R}: {len(vols)} volumes, {n_sl} slices | new-line z > {a.k:g}, "
                 f">= {a.min_px} px, Hessian scale {a.scale:g}, tolerance {a.tol} px", "",
                 f"{'method':<14}{'slices w/ blob':>15}{'median score':>14}{'p90 score':>11}"
                 f"{'max score':>11}"]
        for m in a.methods:
            s = np.array([r["score"] for r in rows if r["method"] == m])
            lines.append(f"{m:<14}{100 * np.mean(s > 0):>14.0f}%{np.median(s):>14.0f}"
                         f"{np.percentile(s, 90):>11.0f}{s.max():>11.0f}")
        for m in a.methods:
            top = sorted((r for r in rows if r["method"] == m and r["score"] > 0),
                         key=lambda r: -r["score"])[:a.top]
            lines += ["", f"STRONGEST new lines: {m}",
                      f"{'volume':<30}{'sl':>4}{'score':>8}{'blobs':>7}{'area':>7}"
                      f"{'row':>6}{'col':>6}   viewer filter"]
            for k, r in enumerate(top):
                lines.append(f"{r['volume']:<30}{r['slice']:>4}{r['score']:>8.0f}"
                             f"{r['n_blobs']:>7}{r['area']:>7}{r['row']:>6}{r['col']:>6}"
                             f"   {viewer_filter(r['volume'], r['slice'])}")
                if k < a.png:
                    blob = blobs[(r["volume"], r["pos"], m)][0]
                    render(dumps, a.methods, m, r["volume"], r["pos"], r["slice"], blob, anatomy, a,
                           os.path.join(a.out, f"R{R}_{m}_{k:02d}_{r['volume']}_s{r['slice']}.png"))
        txt = "\n".join(lines)
        print(txt)
        with open(os.path.join(a.out, f"detect_R{R}.txt"), "w") as f:
            f.write(txt + "\n")
        print(f"wrote {csv_path}, detect_R{R}.txt" + (f" and up to {a.png} PNGs per method"
                                                       if a.png else ""))


if __name__ == "__main__":
    sys.exit(main())
