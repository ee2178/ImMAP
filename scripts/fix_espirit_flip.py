#!/usr/bin/env python3
"""
Correct ESPIRiT maps written before the kernel-flip fix, without rerunning ESPIRiT.

`physics/smaps.py::espirit` used to skip the flip that turns calibration
kernels (correlations) into image-domain operators, so every map it produced is
point-reflected through the FOV centre. That reflection is exactly undone by
reflecting the stored maps back -- no ESPIRiT needed.

WHAT "FLIP" MEANS HERE. The reflection is about the fftshift centre, index
N // 2, i.e. i -> (2 (N // 2) - i) mod N:

    odd N    flip                          (i -> N - 1 - i)
    even N   flip, then roll by one pixel  (i -> N - i)

A plain `flip` is a pixel off on every even-sized axis (brain's grids). Checked
on a phantom with known maps: this correction reproduces the fixed espirit()
exactly on odd and even grids (coherence 1.000, SENSE residual 0.008), and a
plain flip on an even grid does not (residual 0.043).

`image` IS RECOMPUTED, NOT FLIPPED. It is `sum_c conj(s_c) ifftc(k_c)`, and the
k-space was never reflected -- so the ground truth is rebuilt from the raw
k-space and the corrected maps, exactly as make_espirit_smaps.py builds it. It
is written as (S, H, W), which also retires the old (S, 1, H, W) layout.

SAFETY
  * Applying the reflection twice undoes it, so each file is checked before it
    is touched: the SENSE model residual on the middle slice is computed for the
    stored maps and for the reflected ones, and the file is corrected only if
    the reflection clearly lowers it. Otherwise it is reported and left alone.
  * Files carrying `orientation_fixed` (written by the fixed script, or already
    corrected here) are skipped without being read.
  * Each file is rewritten to `<name>.fixing` and renamed over the original, so
    an interrupted run never leaves a half-written map. Stale `.fixing` files
    are removed on start. `.partial` files from a running ESPIRiT job are never
    touched; rerun this afterwards to pick up whatever that job wrote.

Usage
-----
    python scripts/fix_espirit_flip.py --anatomy brain --split val              # dry run
    python scripts/fix_espirit_flip.py --anatomy brain --split train --apply
    python scripts/fix_espirit_flip.py ... --apply --shard 0 --num-shards 8     # array job

Needs no GPU. The dry run reads every file and reports the residuals, so it is
also the check that this is the right correction for your data.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from operators.fourier import ifftc                       # noqa: E402

ROOT = "/home/ee2178/scratch/ee2178/datasets"
KSPACE = {
    "brain": f"{ROOT}/fastmri/brain/multicoil_{{split}}",
    "knee": f"{ROOT}/fastmri/knee/multicoil_{{split}}",
}
# make_espirit_smaps.py writes `<old smap root>_espirit`
ESPIRIT = {
    "brain": f"{ROOT}/fastmri_preprocessed/brain_T2W_coil_combined/{{split}}_espirit",
    "knee": f"{ROOT}/fastmri_preprocessed/knee_coil_combined/pd/{{split}}_espirit",
}

# Correct only when the reflection lowers the residual by at least this factor.
# Mirrored maps sit near 0.8 and correct ones well under 0.3, so a real case
# clears it by a wide margin; anything closer is not the situation this fixes.
MIN_IMPROVEMENT = 2.0


def reflect(x):
    """Point reflection through the fftshift centre over the last two dims."""
    x = x.flip(-2, -1)
    return torch.roll(x, shifts=(1 - x.shape[-2] % 2, 1 - x.shape[-1] % 2),
                      dims=(-2, -1))


def coil_images(kspace):
    return ifftc(torch.from_numpy(np.asarray(kspace)).to(torch.complex64))


def residual(smaps, coils):
    """||c - s x|| / ||c|| inside the maps' support, x = sum conj(s) c."""
    sup = smaps.abs().sum(1, keepdim=True) > 0
    x = (smaps.conj() * coils).sum(1, keepdim=True)
    den = (coils * sup).norm()
    return float((coils - smaps * x).mul(sup).norm() / den) if den > 0 else float("nan")


def fix_file(path, kroot, apply):
    """(status, residual as stored, residual reflected)."""
    fname = os.path.basename(path)
    with h5py.File(path, "r") as f:
        if bool(f.attrs.get("orientation_fixed", False)):
            return "already-fixed", None, None
        attrs = dict(f.attrs)
        smaps = torch.from_numpy(np.asarray(f["smaps"]))

    kpath = os.path.join(kroot, fname)
    if not os.path.exists(kpath):
        return "no-kspace", None, None
    with h5py.File(kpath, "r") as f:
        kspace = np.asarray(f["kspace"])
    if kspace.shape[0] != smaps.shape[0] or kspace.shape[-2:] != tuple(smaps.shape[-2:]):
        return "shape-mismatch", None, None

    m = smaps.shape[0] // 2
    c_mid = coil_images(kspace[m:m + 1])
    r_old = residual(smaps[m:m + 1], c_mid)
    r_new = residual(reflect(smaps[m:m + 1]), c_mid)
    if not (r_new * MIN_IMPROVEMENT < r_old):
        return "not-mirrored", r_old, r_new
    if not apply:
        return "would-fix", r_old, r_new

    smaps = reflect(smaps)
    image = (smaps.conj() * coil_images(kspace)).sum(dim=1)          # (S, H, W)

    tmp = path + ".fixing"
    with h5py.File(tmp, "w") as f:
        f.create_dataset("smaps", data=smaps.numpy())
        f.create_dataset("image", data=image.numpy())
        attrs.update(orientation_fixed=True, orientation_fix="reflected by fix_espirit_flip.py",
                     residual_before=r_old, residual_after=r_new)
        f.attrs.update(attrs)
    os.replace(tmp, path)
    return "fixed", r_old, r_new


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--anatomy", choices=("brain", "knee"), default="brain")
    p.add_argument("--split", choices=("train", "val"), required=True)
    p.add_argument("--dir", default=None, help="ESPIRiT map directory (default from anatomy/split)")
    p.add_argument("--kspace-root", default=None)
    p.add_argument("--apply", action="store_true", help="rewrite files (default: report only)")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    args = p.parse_args()

    d = args.dir or ESPIRIT[args.anatomy].format(split=args.split)
    kroot = args.kspace_root or KSPACE[args.anatomy].format(split=args.split)
    for x in (d, kroot):
        if not os.path.isdir(x):
            raise SystemExit(f"missing: {x}")

    for stale in glob.glob(os.path.join(d, "*.fixing")):
        if args.apply:
            os.remove(stale)
    files = sorted(glob.glob(os.path.join(d, "*.h5")))[args.shard::args.num_shards]
    print(f"{'APPLY' if args.apply else 'dry run'}: {d}  shard {args.shard}/{args.num_shards}, "
          f"{len(files)} files")

    counts, t0 = {}, time.time()
    for i, path in enumerate(files):
        try:
            status, r_old, r_new = fix_file(path, kroot, args.apply)
        except (OSError, KeyError) as e:
            status, r_old, r_new = f"error: {e}", None, None
        counts[status.split(":")[0]] = counts.get(status.split(":")[0], 0) + 1
        res = "" if r_old is None else f"  residual {r_old:.3f} -> {r_new:.3f}"
        print(f"  [{i + 1}/{len(files)}] {status:14s} {os.path.basename(path)}{res}", flush=True)

    print("\n  " + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
          + f"   ({time.time() - t0:.0f}s)")
    if counts.get("not-mirrored"):
        print("  not-mirrored files were left alone: the reflection did not clearly lower "
              "their residual. Inspect one before assuming anything.")
    if not args.apply and counts.get("would-fix"):
        print("  dry run -- rerun with --apply to rewrite")
    return 1 if any(k in ("error", "shape-mismatch", "no-kspace") for k in counts) else 0


if __name__ == "__main__":
    sys.exit(main())
