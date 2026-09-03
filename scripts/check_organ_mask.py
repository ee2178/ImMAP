#!/usr/bin/env python3
"""
Does the organ mask actually select anything?

`datasets/fastmri/loader.py` builds it as `smaps.abs().sum(0) > 0` -- a STRICT
float test. That is exact only if the preprocessing wrote hard zeros outside the
support. ESPIRiT with an eigenvalue threshold does; Walsh does not, and neither
does anything that smooths, interpolates or resamples the maps afterwards. When
there are no exact zeros the mask is all-True, masking becomes a no-op, and
every masked metric silently equals its unmasked value -- which looks exactly
like the mask not being wired up.

This measures the coverage instead of assuming it, and if the mask is degenerate
it reports the |smaps| distribution so a threshold can be chosen from data.

Usage
-----
    python scripts/check_organ_mask.py --config config/knee/mg/varnet_R8.json
    python scripts/check_organ_mask.py --config config/knee/mg/varnet_R8.json \\
        --split val --n 12 --threshold 1e-3
    python scripts/check_organ_mask.py --self-test      # no data needed

Reads the smaps/image h5 directly, exactly as the loader does, so it needs
h5py + numpy and nothing else.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


# ---------------------------------------------------------------------------
def coverage(smaps, threshold=0.0):
    """(mask, kept_fraction) for one slice's `smaps` of shape (NC, H, W).

    `threshold=0` reproduces the loader exactly. Anything larger is the
    question this script exists to answer: how much would a real threshold
    remove that `> 0` does not.
    """
    energy = np.abs(smaps).sum(axis=0)
    m = energy > threshold
    return m, float(m.mean())


def rss_floor(sigma, n_coils):
    """E||n||_2 for `2 * n_coils` real Gaussian components of std sigma.

    Chi with 2*NC degrees of freedom. A SCALE REFERENCE, not a prediction of
    what VarNet outputs: its cascades do denoise, and undersampling means the
    reconstruction never sees all of this. What it shows is the direction and
    order of magnitude of an asymmetry that IS structural --
    `rss(complex_abs(...))` is non-negative and positively biased, so its
    background cannot reach zero, while a complex-output unrolled net's can.
    Unmasked, that difference is charged to VarNet as reconstruction error,
    and it grows with sigma and with coil count.
    """
    from math import lgamma, log, sqrt
    k = 2 * n_coils
    # E[chi_k] = sqrt(2) * Gamma((k+1)/2) / Gamma(k/2)
    return sigma * sqrt(2.0) * float(np.exp(lgamma((k + 1) / 2) - lgamma(k / 2)))


# ---------------------------------------------------------------------------
def report(files, smap_root, threshold, n_slices):
    import h5py

    print(f"{'file':38s} {'NC':>3s} {'exact 0s':>9s} {'>0 keeps':>9s} "
          f"{'>thr keeps':>11s}")
    print("  " + "-" * 74)

    kept_strict, kept_thr, any_zeros = [], [], []
    ncoils = []

    for fname in files:
        path = os.path.join(smap_root, fname)
        with h5py.File(path, "r") as f:
            n = f["smaps"].shape[0]
            picks = np.linspace(0, n - 1, min(n_slices, n)).astype(int)
            for sl in picks:
                sm = np.asarray(f["smaps"][sl])
                sm = np.squeeze(sm)
                if sm.ndim != 3:
                    print(f"  {fname[:36]:38s}  unexpected smaps shape "
                          f"{sm.shape} -- skipped")
                    continue
                zfrac = float((np.abs(sm) == 0).all(axis=0).mean())
                _, k0 = coverage(sm, 0.0)
                _, kt = coverage(sm, threshold)
                kept_strict.append(k0)
                kept_thr.append(kt)
                any_zeros.append(zfrac)
                ncoils.append(sm.shape[0])
        print(f"  {fname[:36]:38s} {ncoils[-1]:3d} {any_zeros[-1]:8.1%} "
              f"{kept_strict[-1]:8.1%} {kept_thr[-1]:10.1%}")

    if not kept_strict:
        raise SystemExit("no slices read.")

    k0 = float(np.mean(kept_strict))
    kt = float(np.mean(kept_thr))
    nc = int(np.median(ncoils))

    print("\nSUMMARY over "
          f"{len(kept_strict)} slices, {len(files)} volumes")
    print(f"  the loader's mask (`> 0`) keeps        {k0:6.2%} of pixels")
    print(f"  a threshold of {threshold:g} would keep      {kt:6.2%}")

    print()
    if k0 > 0.995:
        print("  VERDICT: the mask is DEGENERATE. `smaps.abs().sum(0) > 0` is")
        print("  true almost everywhere, so `use_organ_mask` is currently a")
        print("  no-op -- masked and unmasked metrics are the same number, and")
        print("  a background-bound model is not being helped by it at all.")
        print("  The preprocessing did not write hard zeros outside the support.")
        if kt < 0.9:
            print(f"\n  A threshold of {threshold:g} DOES separate the anatomy "
                  f"({kt:.1%} kept).")
            print("  Change datasets/fastmri/loader.py to compare against it")
            print("  rather than 0, and regenerate nothing -- the mask is built")
            print("  in the loader, not stored in the config.")
        else:
            print(f"\n  {threshold:g} does not separate it either ({kt:.1%} "
                  f"kept). Try --threshold larger, or derive the")
            print("  mask from the IMAGE (e.g. a fraction of its max) instead")
            print("  of the sensitivity maps.")
    elif k0 > 0.85:
        print(f"  VERDICT: the mask keeps {k0:.1%} -- it excludes something, but")
        print("  not much. Expect masking to move the metrics only slightly.")
    else:
        print(f"  VERDICT: the mask is doing real work ({k0:.1%} kept, "
              f"{1 - k0:.1%} excluded).")
        print("  If a model still looks background-bound with masking ON, the")
        print("  mask is not the explanation -- check the run's saved config")
        print("  actually has use_organ_mask: true.")

    print(f"\nScale reference: E[RSS of pure noise] over {nc} coils "
          f"(chi, 2x{nc} dof),")
    print("against a unit-scale image. NOT a prediction of VarNet's output --")
    print("its cascades denoise -- but the direction is structural:")
    for s in (0.01, 0.04, 0.05, 0.06):
        print(f"    sigma {s:<5} -> background magnitude ~{rss_floor(s, nc):.3f}")
    print("  RSS is non-negative and positively biased, so VarNet's background")
    print("  cannot reach 0; a complex-output unrolled net's can. Unmasked that")
    print("  gap is charged to VarNet as error, and it grew ~5x when sigma")
    print("  moved from 0.01 to 0.05.")


# ---------------------------------------------------------------------------
def self_test():
    """Both cases, fabricated -- no data, no h5py."""
    print("[self-test] fabricated maps, to show what each case looks like\n")
    rng = np.random.default_rng(0)
    H = W = 128
    yy, xx = np.mgrid[0:H, 0:W]
    support = ((yy - H / 2) ** 2 + (xx - W / 2) ** 2) ** 0.5 < 40

    base = rng.normal(size=(15, H, W)) + 1j * rng.normal(size=(15, H, W))

    hard = base * support                       # ESPIRiT-like: exact zeros
    soft = base * np.maximum(support, 1e-4)     # smoothed: no exact zeros

    heads = ("case", ">0", ">1e-3", ">1e-2", ">0.1")
    print(f"  {heads[0]:30s}" + "".join(f"{h:>9s}" for h in heads[1:]))
    for name, sm in (("hard zeros outside support", hard),
                     ("no exact zeros (smoothed)", soft)):
        ks = [coverage(sm, t)[1] for t in (0.0, 1e-3, 1e-2, 0.1)]
        print(f"  {name:30s}" + "".join(f"{k:8.1%} " for k in ks))

    print(f"\n  true support is {support.mean():.1%} of the field.")
    print("  Row 1 is fine: `> 0` already finds the support exactly.")
    print("  Row 2 is the failure this script detects -- `> 0` keeps")
    print("  everything, so masking silently does nothing. A threshold only")
    print("  helps once it clears the smoothed floor: 1e-3 is still below it,")
    print("  1e-2 recovers the support exactly. The right value comes from")
    print("  YOUR maps -- sweep --threshold until the kept fraction stops")
    print("  moving, which is the plateau at the true support.")
    print("\n  RSS floor, 15 coils:")
    for s in (0.01, 0.05):
        print(f"    sigma {s:<5} -> ~{rss_floor(s, 15):.3f}")


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="a run config; its data block names the "
                                    "smap_root to read")
    p.add_argument("--split", default="val", choices=("train", "val"))
    p.add_argument("--n", type=int, default=6, help="volumes to sample")
    p.add_argument("--slices", type=int, default=3, help="slices per volume")
    p.add_argument("--threshold", type=float, default=1e-3,
                   help="the alternative threshold to compare `> 0` against, "
                        "relative to |smaps| summed over coils")
    p.add_argument("--self-test", action="store_true",
                   help="fabricate both cases; needs no data")
    args = p.parse_args()

    if args.self_test:
        return self_test()
    if not args.config:
        raise SystemExit("--config is required (or --self-test)")

    with open(args.config) as f:
        cfg = json.load(f)
    data = cfg["data"][args.split]
    root = data["smap_root"]
    # configs store the root relative to the repo root, as the loader resolves it
    if not os.path.isabs(root):
        root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), root)
    if not os.path.isdir(root):
        raise SystemExit(f"smap_root does not exist: {root}\n"
                         f"Run this on the cluster, or pass a config whose "
                         f"paths resolve here.")

    files = sorted(x for x in os.listdir(root) if x.endswith(".h5"))[:args.n]
    if not files:
        raise SystemExit(f"no .h5 under {root}")
    print(f"smap_root: {root}")
    print(f"{len(files)} volumes, {args.slices} slices each, "
          f"threshold={args.threshold:g}\n")
    report(files, root, args.threshold, args.slices)


if __name__ == "__main__":
    sys.exit(main())
