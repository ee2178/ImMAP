#!/usr/bin/env python3
"""
Evaluate the multigrid fastMRI reconstruction grid over a noise sweep.

Counterpart of Sljiva's `scripts/eval_fastmri.jl`. Walks a directory of trained
runs, rebuilds each from its saved `config.json` + `net.ckpt`, and scores it on
the validation set at each sigma in a fixed grid, writing one tidy CSV row per
(run, sigma).

Everything that is not the model is held fixed across runs:

  * the sampling mask comes from the run's own `mri` block (R differs by design),
  * the noise realization is drawn from a seeded generator reset at the start of
    every (run, sigma) pass, so cell 0 and cell 27 see the SAME noise,
  * sigma is pinned per pass rather than sampled, so the sweep reads as a
    robustness curve rather than an average over the training distribution.

`n_params` is reported alongside the metrics because the V-cycle models are only
COMPUTE-matched to the `cdlnet` / `groupcdl` baselines, not parameter-matched --
the comparison needs both numbers to be read honestly.

Usage
-----
    python scripts/eval_mg_recon.py --runs trained_nets/mg_recon --out results/mg_recon.csv
    python scripts/eval_mg_recon.py --runs trained_nets/mg_recon --sigmas 0 0.005 0.01
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datasets                              # noqa: F401  triggers loader registration
from datasets.registry import build_loader
from models import build_model
from operators import FFT2D, Mask, Sense
from physics.mask import get_mask_cached as get_mask
from training.common import prepare_measurement
from training.metrics import compute_metrics


def find_runs(root):
    """Every directory under `root` holding both a config.json and a net.ckpt."""
    runs = []
    for dirpath, _, filenames in os.walk(root):
        if "config.json" in filenames and "net.ckpt" in filenames:
            runs.append(dirpath)
    return sorted(runs)


def load_run(run_dir, device):
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)

    model = build_model(cfg).to(device)

    ckpt = torch.load(os.path.join(run_dir, "net.ckpt"), map_location=device,
                      weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    if getattr(model, "attn_backend", None) == "flex" and device.type == "cuda":
        model.compile_flex()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return cfg, model, n_params, ckpt.get("step")


@torch.no_grad()
def evaluate(model, cfg, sigma, loader, device, seed):
    """Mean PSNR / SSIM / NRMSE over `loader` at a pinned noise level."""
    mri = cfg["mri"]
    gen = torch.Generator(device=device).manual_seed(int(seed))

    totals = {"psnr": 0.0, "ssim": 0.0, "nrmse": 0.0}
    n = 0

    for kspace, smaps, image, _ in loader:
        kspace = kspace.to(device, non_blocking=True)
        smaps = smaps.to(device, non_blocking=True)
        image = image.to(device, non_blocking=True)

        mask = get_mask(image, R=mri["R"], acs_lines=mri["acs_lines"],
                        mode=mri.get("mask_dist", "uniform"),
                        offset=mri.get("mask_offset", 0))

        y, sigma_n, extra = prepare_measurement(
            image=image, kspace=kspace, mask=mask, smaps=smaps,
            kspace_type=mri["kspace_type"],
            noise_std=sigma,                       # a number: pinned, not sampled
            noise_dist=cfg["training"].get("noise_dist", "uniform"),
            whiten_kspace=mri.get("whiten_kspace", False),
            generator=gen,
        )

        E = Mask(mask) @ FFT2D() @ Sense(extra["smaps"])

        recon, _ = model(y, E=E, sigma=sigma_n)

        m = compute_metrics(image.abs(), recon.abs())
        for k in totals:
            totals[k] += float(m[k].detach())
        n += 1

    return {k: v / max(n, 1) for k, v in totals.items()}, n


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default="trained_nets/mg_recon",
                   help="root of the trained-run tree")
    p.add_argument("--out", default="results/mg_recon.csv")
    p.add_argument("--sigmas", type=float, nargs="*",
                   default=[0.0, 0.0025, 0.005, 0.01],
                   help="noise levels to evaluate at (pinned, not sampled)")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    runs = find_runs(args.runs)
    if not runs:
        raise SystemExit(f"no runs with both config.json and net.ckpt under {args.runs}")

    print(f"found {len(runs)} runs under {args.runs}")
    rows = []

    for run_dir in runs:
        try:
            cfg, model, n_params, step = load_run(run_dir, device)
        except Exception as e:                     # a crashed cell shouldn't sink the sweep
            print(f"[skip] {run_dir}: {type(e).__name__}: {e}")
            continue

        # The val loader is rebuilt per run because R (and therefore nothing in
        # the data block) can differ; shuffle=False keeps the slice order fixed.
        loader = build_loader(cfg["data"]["val"], shuffle=False, drop_last=False)

        anatomy = cfg["data"]["val"]["anatomy"]
        model_tag = os.path.basename(run_dir)
        mtype = cfg["model"]["type"]

        for sigma in args.sigmas:
            metrics, n = evaluate(model, cfg, sigma, loader, device, args.seed)
            row = {
                "run": os.path.relpath(run_dir, args.runs),
                "anatomy": anatomy,
                "R": cfg["mri"]["R"],
                "model": model_tag,
                "type": mtype,
                "sigma": sigma,
                "n_slices": n,
                "n_params": n_params,
                "step": step,
                **{k: round(v, 4) for k, v in metrics.items()},
            }
            rows.append(row)
            print(f"  {row['run']:<40} sigma={sigma:<6} "
                  f"psnr={metrics['psnr']:.2f} ssim={metrics['ssim']:.4f} "
                  f"nrmse={metrics['nrmse']:.4f}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nwrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
