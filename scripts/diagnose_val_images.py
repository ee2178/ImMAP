#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Why is `val/example` black? Reproduce train_i2sb's validation panel on real data and print
every intermediate, instead of guessing from the picture.

    python -m scripts.diagnose_val_images --config config/NYUMets/i2sb_unet_all.json

Reads the config's own data block -- image_key, scales, x0/x1_idx, crop -- so what it measures is
what the training loop feeds the net. No checkpoint and no GPU needed: the panel's brightness is a
property of the DATA and the display window, not of the model.

WHAT IT CHECKS, in the order a black panel is usually caused

  1. mask coverage        an empty (or near-empty) brain mask zeroes the whole grid. NYUMets keeps
                          slices down to min_brain_frac=0.02, so a 2%-brain slice is legitimately
                          almost entirely background.
  2. the display window   the panel is (x - lo)/(hi - lo) with lo = -data_range/2. If the brain
                          does not live in that window it clips to one flat value: everything
                          below lo is black, everything above hi is white.
  3. intensity scale      the in-brain distribution in the SCALED units the loader emits. If its
                          std is far from ~0.3, `scales` is wrong for this dataset and both the
                          display AND the bridge schedule are mis-sized (see
                          scripts/calibrate_nyumets_bridge.py for the schedule half).
  4. background           x1 is logged UNMASKED. If its background is not ~0 it used to drive the
                          old global min-max window; it no longer can, but a non-zero background
                          still says the normalization differs from BraTS's.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datasets                                    # noqa: F401  registers the loaders
from datasets.registry import build_loader


def pct(x):
    return f"{100 * x:6.2f}%"


def stats(t, name, width=22):
    v = t.flatten().float()
    if v.numel() == 0:
        print(f"   {name:<{width}} (empty)")
        return
    q = torch.quantile(v[torch.randperm(v.numel())[:200_000]] if v.numel() > 200_000 else v,
                       torch.tensor([0.005, 0.5, 0.995]))
    print(f"   {name:<{width}} min {float(v.min()):8.3f}  p0.5 {float(q[0]):7.3f}  "
          f"med {float(q[1]):7.3f}  p99.5 {float(q[2]):7.3f}  max {float(v.max()):8.3f}  "
          f"std {float(v.std()):6.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="the run's config.json")
    ap.add_argument("--split", default="val")
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    cfg_p = ap.parse_args()

    with open(cfg_p.config) as f:
        cfg = json.load(f)
    d = dict(cfg["data"][cfg_p.split])
    d["num_workers"] = 0
    data_range = float(cfg.get("training", {}).get("data_range", 1.0))
    hi = 0.5 * data_range
    lo = -hi

    print(f"config      {cfg_p.config}")
    print(f"split       {cfg_p.split}   image_key={d.get('image_key')}   "
          f"scales={d.get('scales')}")
    print(f"data_range  {data_range}  ->  display window [{lo:g}, {hi:g}]")
    print(f"x0_idx={d.get('x0_idx')}  x1_idx={d.get('x1_idx')}  cond_idx={d.get('cond_idx')}  "
          f"crop_size={d.get('crop_size')}  center_crop={d.get('center_crop')}")

    torch.manual_seed(cfg_p.seed); np.random.seed(cfg_p.seed)
    loader = build_loader(d, shuffle=True, drop_last=False)

    n = 0
    mask_fracs, clip_fracs, grid_means, bg_absmax = [], [], [], []
    acc = {"x0 in-brain": [], "x1 in-brain": [], "x0 - x1 in-brain": []}
    it = iter(loader)
    for _ in range(cfg_p.batches):
        try:
            batch = next(it)
        except StopIteration:
            break
        x0, x1, cond, mask = batch[:4]
        m = mask > 0.5
        for i in range(x0.shape[0]):
            mi = m[i]
            mask_fracs.append(float(mi.float().mean()))
            if mi.any():
                a, b = x0[i][mi], x1[i][mi]
                acc["x0 in-brain"].append(a)
                acc["x1 in-brain"].append(b)
                acc["x0 - x1 in-brain"].append(a - b)
                clip_fracs.append(float(((a < lo) | (a > hi)).float().mean()))
            bg = x1[i][~mi]
            bg_absmax.append(float(bg.abs().max()) if bg.numel() else 0.0)
            # the panel exactly as train_i2sb builds it
            cols = torch.cat([x1[i:i+1], x0[i:i+1]], dim=0)
            grid = mask[i:i+1] * ((cols - lo) / (hi - lo)).clamp(0, 1)
            grid_means.append(float(grid.mean()))
            n += 1

    if n == 0:
        raise SystemExit("no batches loaded")
    print(f"\n{n} slices from {cfg_p.batches} batch(es)")

    print("\n1. MASK COVERAGE   (an empty mask zeroes the panel outright)")
    mf = np.array(mask_fracs)
    print(f"   brain fraction   min {pct(mf.min())}  med {pct(np.median(mf))}  "
          f"max {pct(mf.max())}   slices under 5%: {int((mf < 0.05).sum())}/{n}")

    print("\n2. INTENSITY, in the scaled units the loader emits")
    for k, v in acc.items():
        if v:
            stats(torch.cat([t.flatten() for t in v]), k)

    print("\n3. DISPLAY WINDOW")
    cf, gm = np.array(clip_fracs), np.array(grid_means)
    print(f"   in-brain voxels outside [{lo:g}, {hi:g}]   med {pct(np.median(cf))}   "
          f"max {pct(cf.max())}")
    print(f"   panel mean brightness (0 = black, 1 = white)  med {np.median(gm):.4f}   "
          f"min {gm.min():.4f}   max {gm.max():.4f}")

    print("\n4. BACKGROUND of x1 (logged UNMASKED)")
    bg = np.array(bg_absmax)
    print(f"   max |value| outside the brain   med {np.median(bg):.4f}   max {bg.max():.4f}"
          + ("   <- not ~0: this data is not background-zeroed like BraTS"
             if np.median(bg) > 1e-3 else "   (background is zeroed, as BraTS is)"))

    print("\nVERDICT")
    sig = torch.cat([t.flatten() for t in acc["x0 in-brain"]]).std().item() if acc["x0 in-brain"] \
        else float("nan")
    if np.median(mf) < 0.05:
        print("   The brain mask is nearly empty on most slices -- the panel is black because")
        print("   there is almost nothing to draw. Raise --min-brain-frac when building the h5,")
        print("   or centre-crop the val block so the brain fills more of the frame.")
    elif np.median(cf) > 0.5:
        print(f"   {pct(np.median(cf))} of brain voxels fall OUTSIDE the display window, so the")
        print(f"   panel saturates. In-brain std is {sig:.3f}; the window assumes ~0.3.")
        print(f"   Fix `scales` (currently {d.get('scales')}) so the brain lands in "
              f"[{lo:g}, {hi:g}], then re-run scripts/calibrate_nyumets_bridge.py -- the same")
        print("   scale sets the bridge schedule, so beta_max needs re-picking with it.")
    elif np.median(gm) < 0.05:
        print("   The window is fine and the mask is populated, but the panel is still dark:")
        print("   the in-brain values sit near the bottom of the window. Check the sign/centre")
        print("   of the normalization -- median-MAD centres the brain at 0, which should render")
        print("   mid-grey, not black.")
    else:
        print(f"   Nothing wrong here: mask {pct(np.median(mf))}, "
              f"{pct(np.median(cf))} clipped, panel mean {np.median(gm):.3f}.")
        print("   If wandb still shows black, the problem is downstream of the data -- check")
        print("   that this run's config.json is the one the job actually used.")


if __name__ == "__main__":
    main()
