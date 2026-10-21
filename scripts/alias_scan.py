#!/usr/bin/env python3
"""
Rank dumped slices by how much of a method's error is ALIASING.

    python scripts/alias_scan.py --runs trained_nets/mg_recon/brain
    python scripts/alias_scan.py --runs trained_nets/mg_recon/brain --R 12 \
        --target lpdsnet --top 20 --png 8 --out trained_nets/mg_recon/brain/alias_scan

Reads the HDF5 written by scripts/dump_eval.py (`<run>/eval_dump/<volume>.h5`)
and nothing else, so it runs on a laptop or a login node; no torch, no GPU.

What it measures
----------------
Equispaced Cartesian undersampling, one phase-encode line in every d, folds the
image onto itself shifted by N/d along the PE axis (N = PE width). Aliasing
that a network fails to remove is therefore error that LOOKS LIKE THE REFERENCE
SHIFTED BY k*N/d. Per slice and method, with r = |recon| - |ref| and g_k the
reference shifted by k*N/d (fractional shifts by an FFT phase ramp):

    ghost   max_k |<r, g_k>| / (||r|| ||g_k||)   how much of the error is ONE
            ghost -- 0 none, 1 the error is nothing but a ghost
    gerr    max_k |<r, g_k>| / ||g_k|| / ||ref||  how BIG that ghost is, as a
            fraction of the image. The ranking key: a pure but faint ghost is
            not an artifact anyone reports.
    bg      ||r outside the dilated anatomy|| / ||ref||  error spilled into the
            background, where folded anatomy lands
    nrmse   the stored per-slice metric, for scale

MAGNITUDES, not the complex images: the dumped reference and the network
outputs do not share a phase convention (mean phase difference ~2.6 rad inside
the anatomy on the brain dumps of 2026-10-13), so a complex residual is mostly
phase and says nothing about aliasing. The stored metrics are magnitude too.

`--target` is compared against the median of every other method dumped at the
same R, so a slice that is hard for EVERY method (gerr high across the board)
is told apart from one where the target alone fails (high `x others`).

A method is a run directory named `<method>_R<R>`; tagged probes
(`<cell>_500ep`, ...) do not match and are skipped. Only volumes present for
every method are scanned.
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
from scipy.ndimage import binary_dilation

RUN_RE = re.compile(r"^(?P<method>.+)_R(?P<R>\d+)$")


def discover(root):
    """{R: {method: eval_dump dir}} for every run with at least one dump."""
    out = collections.defaultdict(dict)
    for d in sorted(glob.glob(os.path.join(root, "*", "eval_dump"))):
        m = RUN_RE.match(os.path.basename(os.path.dirname(d)))
        if m and glob.glob(os.path.join(d, "*.h5")):
            out[int(m["R"])][m["method"]] = d
    return out


def ghost_period(mask_row):
    """(d, N/d) from one row of the sampling mask: the modal line spacing.

    The ACS block is contiguous (spacing 1), so the mode of the spacings among
    the OUTER lines is the equispaced step. Refuses a mask that is not mostly
    equispaced, where "the ghost" is not one shift and this scan means nothing.
    """
    pe = np.flatnonzero(mask_row > 0)
    gaps = np.diff(pe)
    gaps = gaps[gaps > 1]
    if gaps.size == 0:
        raise SystemExit("sampling mask has no outer lines -- nothing to alias")
    d = int(np.bincount(gaps).argmax())
    if np.mean(np.abs(gaps - d) <= 1) < 0.8:
        raise SystemExit(f"sampling mask is not equispaced (modal spacing {d} covers "
                         f"{np.mean(gaps == d):.0%} of gaps); a single ghost period "
                         f"does not describe its aliasing")
    return d, mask_row.size / d


def shifted(x, s, axis=1):
    """x circularly shifted by s pixels along `axis` (fractional: phase ramp)."""
    n = x.shape[axis]
    ph = np.exp(-2j * np.pi * np.fft.fftfreq(n) * s)
    shape = [1] * x.ndim
    shape[axis] = n
    return np.fft.ifft(np.fft.fft(x, axis=axis) * ph.reshape(shape), axis=axis).real


_MASKED_WARNED = set()


def scan_volume(dumps, vol, methods, target):
    """One row per (slice, method)."""
    rec = {}
    for m in methods:
        with h5py.File(os.path.join(dumps[m], vol + ".h5"), "r") as h:
            rec[m] = (np.abs(h["recon"][:]), h["nrmse_slice"][:])
            if h.attrs.get("images_masked", False) and not _MASKED_WARNED:
                _MASKED_WARNED.add(m)
                print(f"warning: {dumps[m]}/{vol}.h5 stores organ-masked images "
                      f"(dump_eval since 2026-10-21): `bg` is 0 by construction "
                      f"and ghosts folding outside the mask are not seen",
                      file=sys.stderr)
            if m == target:
                ref = np.abs(h["reference"][:])
                organ = h["organ_mask"][:] > 0
                mask = h["sampling_mask"][0]
                sl_idx = h["slice_index"][:]
    # The PE axis is the one the mask varies along: whole columns sampled
    # (every row identical) means PE runs along axis 1.
    axis = 1 if not (mask != mask[:1]).any() else 0
    row = mask[0] if axis == 1 else mask[:, 0]
    d, period = ghost_period(row)

    rows = []
    for j in range(ref.shape[0]):
        x = ref[j]
        xn = np.linalg.norm(x) + 1e-12
        bg = ~binary_dilation(organ[j], iterations=10)
        ghosts = [shifted(x, sgn * k * period, axis)
                  for k in range(1, d // 2 + 1) for sgn in (1, -1)]
        gnorm = [np.linalg.norm(g) + 1e-12 for g in ghosts]
        for m in methods:
            r = rec[m][0][j] - x
            rn = np.linalg.norm(r) + 1e-12
            ip = [abs(np.vdot(g, r)) for g in ghosts]
            rows.append(dict(
                volume=vol, slice=int(sl_idx[j]), pos=j, method=m,
                ghost=max(i / (rn * n) for i, n in zip(ip, gnorm)),
                gerr=max(i / n for i, n in zip(ip, gnorm)) / xn,
                bg=float(np.linalg.norm(r[bg]) / xn),
                nrmse=float(rec[m][1][j])))
    return rows, d, period


def render(dumps, methods, vol, pos, path, gain=8.0):
    """Magnitudes (top) and |error| x gain (bottom), cropped to the anatomy."""
    from PIL import Image, ImageDraw

    ims, ref = {}, None
    for m in methods:
        with h5py.File(os.path.join(dumps[m], vol + ".h5"), "r") as h:
            ims[m] = np.abs(h["recon"][pos])
            if ref is None:
                ref = np.abs(h["reference"][pos])
                zf = np.abs(h["zero_filled"][pos])
                organ = h["organ_mask"][pos] > 0
    cols = [("zero-filled", zf)] + [(m, ims[m]) for m in methods] + [("reference", ref)]
    rr, cc = np.where(organ) if organ.any() else np.where(np.ones_like(organ))
    r0, r1 = max(rr.min() - 10, 0), rr.max() + 10
    c0, c1 = max(cc.min() - 10, 0), cc.max() + 10
    vmax = np.percentile(ref[organ], 99.5) if organ.any() else ref.max()

    def tile(a, v):
        return Image.fromarray((np.clip(a[r0:r1, c0:c1] / v, 0, 1) * 255).astype(np.uint8))

    w, h = tile(ref, vmax).size
    out = Image.new("L", (len(cols) * w, 2 * h + 16), 0)
    dr = ImageDraw.Draw(out)
    for i, (name, a) in enumerate(cols):
        out.paste(tile(a, vmax), (i * w, 16))
        if name != "reference":
            out.paste(tile(np.abs(a - ref), vmax / gain), (i * w, 16 + h))
        dr.text((i * w + 3, 2), name, fill=255)
    dr.text((3, 16 + h + 3), f"|error| x{gain:g}", fill=255)
    out.save(path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default="trained_nets/mg_recon/brain",
                   help="directory holding <method>_R<R>/eval_dump/")
    p.add_argument("--R", type=int, nargs="*", help="accelerations (default: all dumped)")
    p.add_argument("--target", default="lpdsnet", help="the method under suspicion")
    p.add_argument("--methods", nargs="*",
                   help="methods to compare (default: every one dumped at that R)")
    p.add_argument("--top", type=int, default=15, help="rows in each ranking")
    p.add_argument("--png", type=int, default=0,
                   help="render the N worst target slices (by gerr) into --out")
    p.add_argument("--out", help="write scan_R<R>.csv / .txt (and PNGs) here")
    a = p.parse_args(argv)

    found = discover(a.runs)
    if not found:
        raise SystemExit(f"no <method>_R<R>/eval_dump/*.h5 under {a.runs}")
    if a.out:
        os.makedirs(a.out, exist_ok=True)

    for R in (a.R or sorted(found)):
        dumps = found.get(R, {})
        methods = [m for m in (a.methods or sorted(dumps)) if m in dumps]
        if a.target not in methods:
            print(f"R={R}: no {a.target} dump -- skipped (have {sorted(dumps)})")
            continue
        methods = [a.target] + [m for m in methods if m != a.target]
        if len(methods) < 2:
            print(f"R={R}: only {a.target} dumped, nothing to compare -- skipped")
            continue
        vols = set.intersection(*[
            {os.path.basename(f)[:-3] for f in glob.glob(os.path.join(dumps[m], "*.h5"))}
            for m in methods])
        if not vols:
            print(f"R={R}: no volume is dumped for every method -- skipped")
            continue

        rows = []
        for i, vol in enumerate(sorted(vols)):
            r, d, period = scan_volume(dumps, vol, methods, a.target)
            rows += r
            print(f"\r  R={R}: {i + 1}/{len(vols)} volumes", end="", file=sys.stderr)
        print(file=sys.stderr)

        by = collections.defaultdict(dict)
        for r in rows:
            by[(r["volume"], r["slice"], r["pos"])][r["method"]] = r
        others = methods[1:]

        def excess(k, key):
            return by[k][a.target][key] / (np.median([by[k][m][key] for m in others]) + 1e-12)

        lines = [f"R={R}: {len(vols)} volumes, {len(by)} slices | PE spacing d={d}, "
                 f"ghost period {period:.2f} px | target {a.target} vs {', '.join(others)}",
                 "", f"{'mean over slices':<18}{'ghost':>7}{'gerr':>9}{'bg':>9}{'nrmse':>9}"]
        for m in methods:
            v = [by[k][m] for k in by]
            lines.append(f"  {m:<16}{np.mean([x['ghost'] for x in v]):>7.3f}"
                         f"{np.mean([x['gerr'] for x in v]):>9.4f}"
                         f"{np.mean([x['bg'] for x in v]):>9.4f}"
                         f"{np.mean([x['nrmse'] for x in v]):>9.4f}")
        head = (f"{'volume':<32}{'sl':>4}{'gerr':>8}{'x others':>9}{'ghost':>7}{'bg':>8}"
                f"{'nrmse':>8}{'x others':>9}")

        def table(title, keys):
            lines.extend(["", title, head])
            for k in keys:
                t = by[k][a.target]
                lines.append(f"{k[0]:<32}{k[1]:>4}{t['gerr']:>8.4f}{excess(k, 'gerr'):>8.1f}x"
                             f"{t['ghost']:>7.2f}{t['bg']:>8.4f}{t['nrmse']:>8.4f}"
                             f"{excess(k, 'nrmse'):>8.2f}x")

        worst = sorted(by, key=lambda k: -by[k][a.target]["gerr"])
        table(f"WORST {a.target} slices by ghost size (gerr)", worst[:a.top])
        table(f"WORST {a.target} slices RELATIVE to the others (gerr / median of others)",
              sorted(by, key=lambda k: -excess(k, "gerr"))[:a.top])
        table(f"WORST {a.target} slices by nrmse relative to the others",
              sorted(by, key=lambda k: -excess(k, "nrmse"))[:a.top])
        text = "\n".join(lines)
        print(text + "\n")

        if a.out:
            with open(os.path.join(a.out, f"scan_R{R}.txt"), "w") as f:
                f.write(text + "\n")
            with open(os.path.join(a.out, f"scan_R{R}.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            for rank, k in enumerate(worst[:a.png]):
                render(dumps, methods, k[0], k[2],
                       os.path.join(a.out, f"R{R}_{rank:02d}_{k[0]}_s{k[1]}.png"))
            print(f"wrote {a.out}/scan_R{R}.txt, .csv"
                  + (f" and {min(a.png, len(worst))} PNGs" if a.png else ""))


if __name__ == "__main__":
    main()
