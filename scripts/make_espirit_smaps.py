#!/usr/bin/env python3
"""
Regenerate the preprocessed sensitivity maps with ESPIRiT.

The brain maps on disk are Walsh. Two things are wrong with that for this
pipeline:

  * Walsh produces NO EXACT ZEROS outside the anatomy, so the organ mask --
    built in datasets/fastmri/loader.py as `smaps.abs().sum(0) > 0`, a strict
    float test -- is true almost everywhere and `use_organ_mask` is a no-op.
    ESPIRiT's eigenvalue threshold gives a hard support.
  * Walsh maps are not unit-RSS, and `operators/noise.py::mri_awgn` assumes
    `sum_c |s_c|^2 = 1` -- that assumption is what makes sigma the noise std of
    the coil-combined adjoint, i.e. the quantity the thresholds are calibrated
    against. ESPIRiT's power-method eigenvector is unit-norm per pixel.

BOTH `smaps` AND `image` ARE REWRITTEN, and that is the point rather than a
convenience. `image` is the coil-combined ground truth, `sum_c conj(s_c) c_c`.
It is a function OF THE MAPS. Replacing the maps and keeping the old `image`
would leave a file whose ground truth is not what its own operator produces,
and every simulated measurement built from it (`kspace_type: "simulated"`
pushes `image` through Sense -> Fourier -> mask) would be inconsistent in a way
no shape check catches.

WRITES TO A NEW DIRECTORY by default and refuses to touch the input. The old
maps are the input to every run already trained; `--overwrite` exists but you
almost certainly want a new root and a one-line config change instead.

Usage
-----
    # one split, new directory (default)
    python scripts/make_espirit_smaps.py --anatomy brain --split val

    # the hyperparameters from notebooks/espirit_brain.ipynb are the defaults
    python scripts/make_espirit_smaps.py --anatomy brain --split train \\
        --acs 20 --kernel-size 8 --thresh-eig 0.95

    # shard for an array job
    python scripts/make_espirit_smaps.py --anatomy brain --split train \\
        --shard 3 --num-shards 8

    --dry-run lists what would be written and computes nothing.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from operators.fourier import ifftc                       # noqa: E402
from physics.smaps import espirit                         # noqa: E402

ROOT = "/home/ee2178/scratch/ee2178/datasets"
KSPACE = {
    "brain": f"{ROOT}/fastmri/brain/multicoil_{{split}}",
    "knee": f"{ROOT}/fastmri/knee/multicoil_{{split}}",
}
SMAPS = {
    "brain": f"{ROOT}/fastmri_preprocessed/brain_T2W_coil_combined/{{split}}",
    "knee": f"{ROOT}/fastmri_preprocessed/knee_coil_combined/pd/{{split}}",
}


# ---------------------------------------------------------------------------
def _espirit_chunk(kspace, args, device):
    """(smaps, image) for a block of slices, computed on `device`, returned on CPU."""
    k = torch.from_numpy(kspace).to(device=device, dtype=torch.complex64)

    sm = espirit(k, acs_size=(args.acs, args.acs),
                 kernel_size=args.kernel_size,
                 thresh_rowspace=args.thresh_rowspace,
                 thresh_eig=args.thresh_eig)

    # The ground truth MUST come from these maps, not the old ones.
    img = (sm.conj() * ifftc(k)).sum(dim=1, keepdim=True)
    return sm.cpu(), img.cpu()


def maps_for_volume(kspace, args, device):
    """(smaps, image) for one volume, in slice chunks.

    ESPIRiT's Hankel SVD and power method are per-slice and independent, so the
    chunking is purely a memory knob and changes no number.

    Memory per slice is dominated by the kernel images -- coils x retained
    kernels x the full k-space grid, complex -- which is gigabytes per slice on
    brain's readout-oversampled grid, and the retained-kernel count varies by
    volume. So a chunk that fits one volume can fail on the next. On CUDA OOM the
    chunk is halved and the SAME slices retried, down to one slice per call; the
    reduced value stays in `args.chunk` for the rest of the run, so later volumes
    do not pay for the same failure again. OOM at one slice is re-raised.
    """
    S, C, H, W = kspace.shape
    smaps = torch.empty((S, C, H, W), dtype=torch.complex64)
    image = torch.empty((S, 1, H, W), dtype=torch.complex64)

    a = 0
    while a < S:
        b = min(a + args.chunk, S)
        try:
            smaps[a:b], image[a:b] = _espirit_chunk(kspace[a:b], args, device)
            a = b
            continue
        except torch.cuda.OutOfMemoryError:
            if args.chunk == 1:
                raise
        # Outside the handler on purpose: until it exits, the traceback keeps
        # the failed call's tensors alive, and emptying the cache frees nothing.
        torch.cuda.empty_cache()
        args.chunk = max(1, args.chunk // 2)
        print(f"    CUDA OOM on slices {a}:{b} -- retrying with chunk={args.chunk}",
              flush=True)

    return smaps, image


def stats(smaps, image, kspace, device):
    """Support fraction, RSS error inside support, and the model residual.

    The residual is the honest check that the maps explain the data; the RSS
    error is the check that `mri_awgn`'s unit-RSS assumption holds. Computed on
    the middle slice only -- this runs per volume and is diagnostics, not
    output.
    """
    m = smaps.shape[0] // 2
    sm = smaps[m:m + 1].to(device)
    k = torch.from_numpy(kspace[m:m + 1]).to(device=device, dtype=torch.complex64)
    c = ifftc(k)
    x = image[m:m + 1].to(device)

    sup = (sm.abs().sum(1) > 0)
    rss = sm.abs().pow(2).sum(1).sqrt()
    rss_err = float((rss[sup] - 1).abs().max()) if bool(sup.any()) else float("nan")
    res = float((c - sm * x).norm() / c.norm().clamp_min(1e-12))
    return float(sup.float().mean()), rss_err, res


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--anatomy", choices=("brain", "knee"), required=True)
    p.add_argument("--split", choices=("train", "val"), required=True)
    p.add_argument("--kspace-root", default=None)
    p.add_argument("--smap-root", default=None,
                   help="the EXISTING maps; read only, to enumerate the files "
                        "that belong to this split")
    p.add_argument("--out", default=None,
                   help="output directory (default: <smap-root>_espirit)")
    p.add_argument("--overwrite", action="store_true",
                   help="write into --smap-root itself, destroying the Walsh "
                        "maps and the ground truth derived from them. Every "
                        "trained run used those; prefer a new root.")

    # defaults are the values chosen in notebooks/espirit_brain.ipynb
    p.add_argument("--acs", type=int, default=20)
    p.add_argument("--kernel-size", type=int, default=8)
    p.add_argument("--thresh-eig", type=float, default=0.95)
    p.add_argument("--thresh-rowspace", type=float, default=0.05)

    p.add_argument("--chunk", type=int, default=4,
                   help="starting slices per ESPIRiT call; halved on CUDA OOM, "
                        "down to 1, and kept for the rest of the run")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--redo", action="store_true",
                   help="recompute files that already exist in --out")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    kroot = args.kspace_root or KSPACE[args.anatomy].format(split=args.split)
    sroot = args.smap_root or SMAPS[args.anatomy].format(split=args.split)
    out = args.out or (sroot if args.overwrite else sroot.rstrip("/") + "_espirit")

    if os.path.abspath(out) == os.path.abspath(sroot) and not args.overwrite:
        raise SystemExit("--out equals the input maps; pass --overwrite to mean it.")
    for d in (kroot, sroot):
        if not os.path.isdir(d):
            raise SystemExit(f"missing: {d}")

    # The existing map directory defines the split. Enumerating the raw kspace
    # root instead would silently pull in volumes this split excludes.
    files = sorted(f for f in os.listdir(sroot) if f.endswith(".h5"))
    files = files[args.shard::args.num_shards]
    if not files:
        raise SystemExit(f"no .h5 in {sroot} for shard {args.shard}")

    print(f"anatomy={args.anatomy} split={args.split} "
          f"shard {args.shard}/{args.num_shards}")
    print(f"  kspace in  {kroot}")
    print(f"  maps in    {sroot}")
    print(f"  writing to {out}"
          f"{'   *** OVERWRITING THE WALSH MAPS ***' if os.path.abspath(out) == os.path.abspath(sroot) else ''}")
    print(f"  espirit    acs=({args.acs},{args.acs}) kernel={args.kernel_size} "
          f"thresh_eig={args.thresh_eig} thresh_rowspace={args.thresh_rowspace}")
    print(f"  {len(files)} volumes\n")

    if args.dry_run:
        for f in files:
            print(f"    would write {os.path.join(out, f)}")
        return 0

    os.makedirs(out, exist_ok=True)
    device = torch.device(args.device)
    t0 = time.time()
    done = skipped = 0

    for i, fname in enumerate(files):
        dst = os.path.join(out, fname)
        if os.path.exists(dst) and not args.redo:
            skipped += 1
            continue

        with h5py.File(os.path.join(kroot, fname), "r") as f:
            kspace = np.asarray(f["kspace"])
        if kspace.ndim != 4:
            print(f"  [{i}] {fname}: unexpected kspace {kspace.shape} -- skipped")
            continue

        t = time.time()
        smaps, image = maps_for_volume(kspace, args, device)
        sup, rss_err, res = stats(smaps, image, kspace, device)

        # write to a temp name and rename, so an interrupted job never leaves a
        # half-written file that the resume check would then treat as done
        tmp = dst + ".partial"
        with h5py.File(tmp, "w") as f:
            f.create_dataset("smaps", data=smaps.numpy())
            f.create_dataset("image", data=image.numpy())
            f.attrs.update(dict(
                method="espirit", acs=args.acs, kernel_size=args.kernel_size,
                thresh_eig=args.thresh_eig, thresh_rowspace=args.thresh_rowspace,
                source_kspace=os.path.join(kroot, fname)))
        os.replace(tmp, dst)
        done += 1

        print(f"  [{i + 1}/{len(files)}] {fname}  {kspace.shape[0]} sl  "
              f"support {sup:5.1%}  |RSS-1| {rss_err:.1e}  residual {res:.4f}  "
              f"{time.time() - t:.1f}s  chunk {args.chunk}")

    print(f"\n{done} written, {skipped} already present, "
          f"{time.time() - t0:.0f}s total")
    print(f"\nPoint the configs at it:  smap_root -> {out}")
    print("Regenerate them afterwards (the launchers do this with REGENERATE=1).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
