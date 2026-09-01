#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Size the I2SB schedule for NYUMets from the data, instead of inheriting BraTS's numbers.

THE CRITERION (see notebooks/inspect_beta_max.ipynb for the derivation). At inference x1 is
known, so from x_t = mu0*x0 + mu1*x1 + std_sb*eps you can form

    u(t) = (x_t - mu1*x1)/mu0 = x0 + sigma_eff(t)*eps,   sigma_eff = std_sb/mu0

i.e. every bridge step is Gaussian denoising of x0 at level sigma_eff. Choosing `beta_max`
is therefore choosing that noise ladder, and it must be matched to

    varsigma = RMS(x0 - x1)   over BRAIN voxels, in the SCALED units the loader feeds the net

by centring the ladder: pick the schedule whose std_fwd[-1] == varsigma. For a brownian
schedule that reduces to tau = varsigma/2.

Reports varsigma whole-brain and, because the two disagree, emits a RANGE of beta_max rather
than one number: whole-brain sets the floor, the high-residual tail sets the ceiling.

    python -m scripts.calibrate_nyumets_bridge --root ~/scratch/datasets/NYUMets_h5_train
"""

import os
import glob
import argparse

import numpy as np
import torch
import h5py

from sb.base import build_schedule


def varsigma(root, x0_idx, x1_idx, image_key, scale, max_subjects, max_slices, seed):
    """RMS(x0 - x1) over brain voxels, pooled across subjects, in scaled units."""
    files = sorted(glob.glob(os.path.join(root, "*", "*_img.h5")))
    if not files:
        raise SystemExit(f"no */*_img.h5 under {root}")
    rng = np.random.default_rng(seed)
    if max_subjects and len(files) > max_subjects:
        files = [files[i] for i in sorted(rng.permutation(len(files))[:max_subjects])]

    sq, n, per_subj = 0.0, 0, []
    for p in files:
        with h5py.File(p, "r") as f:
            if image_key not in f:
                print(f"  !! {os.path.basename(p)}: no '{image_key}', skipped")
                continue
            N = f[image_key].shape[0]
            sel = np.arange(N)
            if max_slices and N > max_slices:
                sel = np.sort(rng.permutation(N)[:max_slices])
            img = f[image_key][sel]                      # (n, H, W, C)
            msk = f["mask"][sel][..., 0].astype(bool)
        d = (img[..., x0_idx] - img[..., x1_idx]) / scale
        v = d[msk]
        if v.size == 0:
            continue
        sq += float(np.sum(v.astype(np.float64) ** 2))
        n += v.size
        per_subj.append(float(np.sqrt(np.mean(v.astype(np.float64) ** 2))))
    if n == 0:
        raise SystemExit("no brain voxels found")
    return float(np.sqrt(sq / n)), np.asarray(per_subj), len(per_subj)


def solve_beta_max(target, n_points, lo=0.01, iters=60, hi_cap=1e4):
    """beta_max whose std_fwd[-1] matches `target`. -> (beta_max, achieved, status).

    std_fwd[-1] is monotone in beta_max, so bisection is exact and avoids depending on the
    closed form of i2sb_betas (a mirrored quadratic ramp with a hardcoded linear_start, not a
    clean square root). The bracket is EXPANDED to reach the target rather than fixed: there
    is no ceiling -- std_fwd[-1] is 0.38 at beta_max 0.3 but 1.48 at 20 and 9.3 at 1000 -- so
    a fixed upper bound would silently return the bound itself for a large target.
    """
    f = lambda b: float(build_schedule("i2sb", n_points=n_points, beta_max=b).std_fwd[-1])
    if target <= f(lo):
        return lo, f(lo), "floor"          # below the variance floor set by linear_start
    hi = max(1.0, lo * 2)
    while f(hi) < target:
        hi *= 2
        if hi > hi_cap:
            return hi_cap, f(hi_cap), "unreachable"
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if f(mid) < target:
            lo = mid
        else:
            hi = mid
    return hi, f(hi), "ok"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="the TRAIN split dir of session folders")
    ap.add_argument("--x0-idx", type=int, default=2, dest="x0_idx")   # T1ce / CT1
    ap.add_argument("--x1-idx", type=int, default=1, dest="x1_idx")   # T1
    ap.add_argument("--image-key", default="img_median_mad", dest="image_key")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="the loader's `scales` divisor for these channels")
    ap.add_argument("--n-points", type=int, default=1000, dest="n_points")
    ap.add_argument("--max-subjects", type=int, default=60, dest="max_subjects")
    ap.add_argument("--max-slices", type=int, default=40, dest="max_slices")
    ap.add_argument("--seed", type=int, default=0)
    cfg = ap.parse_args()

    vs, per_subj, n_subj = varsigma(cfg.root, cfg.x0_idx, cfg.x1_idx, cfg.image_key,
                                    cfg.scale, cfg.max_subjects, cfg.max_slices, cfg.seed)
    hi_vs = float(np.percentile(per_subj, 90))

    print(f"\n{n_subj} sessions, image_key={cfg.image_key}, scales divisor={cfg.scale}")
    print(f"varsigma = RMS(x0 - x1) over brain")
    print(f"  pooled          {vs:.4f}")
    print(f"  per-session     median {np.median(per_subj):.4f}   "
          f"p10 {np.percentile(per_subj,10):.4f}   p90 {hi_vs:.4f}   max {per_subj.max():.4f}")

    print("\nschedule settings that centre the ladder (std_fwd[-1] == varsigma):")
    for name, target in (("pooled", vs), ("p90 session", hi_vs)):
        b, got, status = solve_beta_max(target, cfg.n_points)
        note = {
            "floor": "  <-- AT THE FLOOR: i2sb_betas' hardcoded linear_start bounds the total "
                     "variance below; smaller targets are unreachable by beta_max alone",
            "unreachable": "  <-- UNREACHABLE, rescale the data instead",
            "ok": "  <-- far outside the paper's regime; fix `scales` rather than beta_max"
                  if b > 3.0 else "",
        }[status]
        print(f"  {name:<14} beta_max {b:8.4f}   std_fwd[-1] {got:.4f}   "
              f"(brownian tau {target/2:.4f}){note}")

    print("\nEmit a RANGE, not one number: pooled sets the floor, the p90 session the ceiling.")
    print("beta_max lives in BOTH cfg['i2sb'] and cfg['model']['params'] for SBUnet -- it")
    print("carries its own schedule copy to invert std_fwd back to the step index, and")
    print("train_i2sb calls assert_schedule_matches() at startup. Change both or the job dies.")
    print("\nIf varsigma is far from ~1, prefer fixing `scales` in the data block so the")
    print("bridge endpoints sit in a sane range, THEN re-run this to pick beta_max.")


if __name__ == "__main__":
    main()
