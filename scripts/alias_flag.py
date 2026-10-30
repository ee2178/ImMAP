#!/usr/bin/env python3
"""
Flag slices where a reconstruction network left residual aliasing. Self-contained:
needs only numpy, scipy, h5py and matplotlib; no GPU, and nothing else from the repo.

    python alias_flag.py --runs dumps/brain --out flags/brain
    python alias_flag.py --runs dumps/knee --flag-rho-by-r 8:0.85,12:0.80 --out flags/knee

Input
-----
--runs holds one directory per network and acceleration:

    <runs>/<net>_R<R>/eval_dump/<volume>.h5

The nets in --methods (default varnetmaps lpdsnet mg81v6) are compared on each R, over
the volumes all of them have. Each file holds, for n slices of size H x W, with the
phase-encode direction along W:

    recon         (n, H, W)  the network's output (real or complex; only |.| is used)
    reference     (n, H, W)  the fully sampled image
    zero_filled   (n, H, W)  the zero-filled image, from the same undersampled data
    organ_mask    (n, H, W)  nonzero inside the anatomy
    slice_index   (n,)       the slice numbers

and optionally the attribute kspace_type, which only labels the output. Compare nets
within one k-space type: `flags_R<R>.txt` splits its counts by it.

Method
------
On magnitudes, with HP = x minus its 7-pixel moving average along phase-encode:

    E = HP(|recon| - |reference|)   the network's residual
    T = HP(|zero-filled|)           the aliasing template: zero-filled shows the slice's
                                    aliasing as copies of its edges

    rho       the correlation of E and T in a sliding window (25 along readout, 9 along
              phase-encode). Aliasing a network failed to remove keeps its sign, while
              an edge it under-reconstructed has E ~ -c HP(|reference|), so only large
              positive rho counts.
    sigma     the median over the nets of 1.4826 MAD(E) inside the organ mask eroded by
              4 pixels: one scale per slice, so a net with more error is not scaled down.
    strength  the standard deviation of E over the window, divided by sigma. rho ignores
              size, so without this a faint residual can rank as high as a strong one.

A net's score on a slice is its highest rho among pixels with strength > --min-strength,
at least half a window further inside that mask, where the window cannot reach the mask's
edge (E and T share that edge whatever the network did). With no such pixel it is -1.

A slice's rho is the highest score of any net. It is FLAGGED when that rho > --flag-rho, or
the value for its R in --flag-rho-by-r. Slices are ranked by rho, each appearing once.

The defaults were chosen by eye on a small number of slices, so a flag is a slice to LOOK
at, not a verdict. On our knee data rho runs higher than on brain, hence the per-R cutoffs
above. It compares against a fully sampled reference, so it cannot run without one.

Output
------
    slices_R<R>.csv   every slice, ranked: best_method, rho, strength, flagged, the location,
                      and each net's own score (rho_<net>, which may be elsewhere on the slice)
    flags_R<R>.txt    flagged-slice counts per net and k-space type, and the flagged slices
    slice_R<R>_<rank>_<volume>_s<slice>.png
                      one per flagged slice: the patch around the peak in the reference,
                      zero-filled and every net, each net with its rho and strength in the
                      window centred there
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


def slice_sigma(ref, recs, inner):
    """The scale of a slice's residuals, shared by its methods."""
    es = [hp_pe(x - ref)[inner] for x in recs]
    return float(np.median([1.4826 * np.median(np.abs(e - np.median(e))) for e in es]))


def strength_map(e, sigma, win):
    """Standard deviation of e over the window, in units of sigma."""
    m = ndi.uniform_filter(e, win, mode="nearest")
    v = ndi.uniform_filter(e * e, win, mode="nearest") - m * m
    return np.sqrt(np.maximum(v, 0)) / (sigma + 1e-12)


def net_score(rec, ref, t, sigma, core, a):
    """A net's (rho, strength, z, row, col) on one slice; rho -1 if no pixel qualifies."""
    win = (a.win_rows, a.win_cols)
    e = hp_pe(rec - ref)
    st = strength_map(e, sigma, win)
    rho = np.where(core & (st > a.min_strength), local_corr(e, t, win), -np.inf)
    i, c = np.unravel_index(np.argmax(rho), rho.shape)
    if not np.isfinite(rho[i, c]):
        return -1.0, 0.0, 0.0, -1, -1
    return (round(float(rho[i, c]), 3), round(float(st[i, c]), 1),
            round(float(abs(e[i, c]) / (sigma + 1e-12)), 1), int(i), int(c))


def scan_volume(dumps, vol, a):
    """One row per slice of one volume."""
    data, kspace = {}, {}
    for m in a.methods:
        with h5py.File(os.path.join(dumps[m], vol + ".h5"), "r") as h:
            data[m] = np.abs(h["recon"][:])
            kspace[m] = str(h.attrs.get("kspace_type", "unrecorded"))
            if m == a.methods[0]:
                ref, zf = np.abs(h["reference"][:]), np.abs(h["zero_filled"][:])
                organ = h["organ_mask"][:] > 0
                sl_idx = h["slice_index"][:]
    rows = []
    for j in range(ref.shape[0]):
        inner = ndi.binary_erosion(organ[j], iterations=4)
        if inner.sum() < 100:
            continue
        core = ndi.binary_erosion(inner, iterations=max(a.win_rows, a.win_cols) // 2)
        sigma = slice_sigma(ref[j], [data[m][j] for m in a.methods], inner)
        t = hp_pe(zf[j])
        scores = {m: net_score(data[m][j], ref[j], t, sigma, core, a) for m in a.methods}
        best = max(a.methods, key=lambda m: scores[m][0])
        rho, st, z, i, c = scores[best]
        row = dict(volume=vol, slice=int(sl_idx[j]), pos=j, kspace_type=kspace[best], rho=rho,
                   strength=st, z=z, row=i, col=c, best_method=best)
        row.update({f"rho_{m}": scores[m][0] for m in a.methods})
        rows.append(row)
    return rows


def render(dumps, row, a, path):
    """The patch at the slice's peak: reference, zero-filled, every net with its rho there."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pos, r0, c0 = row["pos"], row["row"], row["col"]
    ims = {}
    for m in a.methods:
        with h5py.File(os.path.join(dumps[m], row["volume"] + ".h5"), "r") as h:
            ims[m] = np.abs(h["recon"][pos])
            if m == a.methods[0]:
                ref, zf = np.abs(h["reference"][pos]), np.abs(h["zero_filled"][pos])
                organ = h["organ_mask"][pos] > 0
    win = (a.win_rows, a.win_cols)
    w = (slice(max(r0 - win[0] // 2, 0), r0 + win[0] // 2 + 1),
         slice(max(c0 - win[1] // 2, 0), c0 + win[1] // 2 + 1))
    t = hp_pe(zf)
    sigma = slice_sigma(ref, list(ims.values()), ndi.binary_erosion(organ, iterations=4))
    titles = {}
    for m in a.methods:
        e = hp_pe(ims[m] - ref)
        rho = local_corr(e, t, win)[w]
        k = np.unravel_index(np.argmax(rho), rho.shape)
        titles[m] = (f"{m}{' (top)' if m == row['best_method'] else ''}\n"
                     f"rho {rho[k]:.2f}, strength {strength_map(e, sigma, win)[w][k]:.1f}")
    half = a.patch_size // 2
    crop = (slice(max(r0 - half, 0), r0 + half), slice(max(c0 - half, 0), c0 + half))
    vmax = max(np.percentile(x[crop], 99.5) for x in [ref, *ims.values()]) or 1.0
    n = len(a.methods) + 2
    fig, axs = plt.subplots(1, n, figsize=(2.6 * n, 3.2))
    axs[0].imshow(ref[crop], cmap="gray", vmin=0, vmax=vmax, interpolation="nearest")
    axs[0].set_title("reference\n", fontsize=9)
    # its own brightness scale: a ghost in zero-filled is faint against anything else
    axs[1].imshow(zf[crop], cmap="gray", vmin=0, vmax=np.percentile(zf[crop], 99.5) or 1.0,
                  interpolation="nearest")
    axs[1].set_title("zero-filled\n(own scale)", fontsize=9)
    for ax, m in zip(axs[2:], a.methods):
        ax.imshow(ims[m][crop], cmap="gray", vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_title(titles[m], fontsize=8)
    for ax in axs:
        ax.axis("off")
    fig.suptitle(f"{row['volume']}  slice {row['slice']}  (array row {r0}, col {c0})", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def parse_by_r(spec):
    """'8:0.85,12:0.80' -> {8: 0.85, 12: 0.8}."""
    return {int(k): float(v) for k, v in (x.split(":") for x in (spec or "").split(",") if x)}


def run(dumps, vols, R, a):
    ranked = []
    for i, v in enumerate(vols):
        print(f"\r  R={R}: {i + 1}/{len(vols)} volumes", end="", file=sys.stderr, flush=True)
        ranked += scan_volume(dumps, v, a)
    print(file=sys.stderr)
    cut = parse_by_r(a.flag_rho_by_r).get(R, a.flag_rho)
    ranked.sort(key=lambda r: -r["rho"])
    for k, r in enumerate(ranked):
        r.update(flagged=int(r["rho"] > cut), rank=k)
    with open(os.path.join(a.out, f"slices_R{R}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ranked[0]))
        w.writeheader()
        w.writerows(ranked)

    flagged = [r for r in ranked if r["flagged"]]
    kept = flagged[:a.top_slices]
    lines = [f"R={R}: {len(vols)} volumes, {len(ranked)} slices, {len(flagged)} flagged | "
             f"rho > {cut:g} on any method, where strength > {a.min_strength:g}, "
             f"window {a.win_rows}x{a.win_cols}", "",
             f"{'method':<14}{'k-space':<20}{'over cutoff':>12}{'flagged slices':>16}{'all slices':>12}"]
    for k in sorted({r["kspace_type"] for r in ranked}):
        x = [r for r in ranked if r["kspace_type"] == k]
        fl = [r for r in x if r["flagged"]]
        for m in a.methods:
            lines.append(f"{m:<14}{k:<20}{sum(r[f'rho_{m}'] > cut for r in fl):>12}{len(fl):>16}{len(x):>12}")
    lines += ["", "TOP FLAGGED SLICES",
              f"{'volume':<30}{'sl':>4}{'rho':>7}{'strength':>10}{'row':>6}{'col':>6}  "
              + "".join(f"{m:>12}" for m in a.methods)]
    lines += [f"{r['volume']:<30}{r['slice']:>4}{r['rho']:>7.2f}{r['strength']:>10.1f}{r['row']:>6}{r['col']:>6}  "
              + "".join(f"{r[f'rho_{m}']:>12.2f}" for m in a.methods) for r in kept]
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(a.out, f"flags_R{R}.txt"), "w") as f:
        f.write(txt + "\n")
    for r in kept:
        render(dumps, r, a, os.path.join(a.out, f"slice_R{R}_{r['rank']:03d}_{r['volume']}_s{r['slice']}.png"))
    print(f"R={R}: wrote slices_R{R}.csv ({len(ranked)} slices) and {len(kept)} images to {a.out}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", required=True, help="directory holding <method>_R<R>/eval_dump/")
    p.add_argument("--out", required=True)
    p.add_argument("--R", type=int, nargs="*", help="accelerations (default: all dumped)")
    p.add_argument("--methods", nargs="*", default=METHODS)
    p.add_argument("--win-rows", type=int, default=25, help="window along readout (px)")
    p.add_argument("--win-cols", type=int, default=9, help="window along phase-encode (px)")
    p.add_argument("--min-strength", type=float, default=2.5,
                   help="smallest window standard deviation of E, in units of sigma")
    p.add_argument("--flag-rho", type=float, default=0.75,
                   help="a slice is flagged when its rho is above this")
    p.add_argument("--flag-rho-by-r", default="", help="per-R --flag-rho, e.g. 8:0.85,12:0.80")
    p.add_argument("--top-slices", type=int, default=100,
                   help="images saved per R, from the top of the flagged slices")
    p.add_argument("--patch-size", type=int, default=48, help="patch shown in the images (px)")
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
        if not vols:
            print(f"R={R}: no volume is dumped for all of {a.methods} -- skipped", file=sys.stderr)
            continue
        run(dumps, vols, R, a)


if __name__ == "__main__":
    sys.exit(main())
