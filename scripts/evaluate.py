#!/usr/bin/env python3
"""
Evaluate trained runs over their validation set.

Task- and metric-agnostic: the task comes from each run's own `config.json`
(`cfg["task"]` -> an adapter in `evaluation/tasks.py`), and the metrics come
from an evaluation config (`evaluation/metrics.py`). Adding a task or a metric
needs no change here.

    python scripts/evaluate.py --eval-config config/eval/mri_recon.json
    python scripts/evaluate.py --runs trained_nets/mg_recon/knee --metrics psnr ssim
    python scripts/evaluate.py --runs trained_nets/... --sigmas 0 0.005 0.01

What is held fixed across runs
------------------------------
* the noise realisation -- one seeded generator, reset at the start of every
  (run, sigma) pass, so run 0 and run 7 see the SAME noise;
* sigma -- pinned per pass, not sampled, so a multi-sigma sweep reads as a
  robustness curve instead of an average over the training distribution;
* the validation slices -- `shuffle=False`, and the loader is rebuilt per run
  because R lives in each run's own config.

Not held fixed, by design: the sampling mask, whose R differs per cell.

`n_params` is reported next to the metrics because the multigrid cells are
compute-matched to their baselines, not parameter-matched; the comparison needs
both numbers to be read honestly.

Metrics are per-slice; the CSV carries mean and std so a difference can be
weighed against the spread it sits in.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datasets                              # noqa: F401  registers the loaders
from datasets.registry import build_loader
from evaluation.metrics import DEFAULT_METRICS, REGISTRY as METRIC_REGISTRY
from evaluation.metrics import build_metrics, warm_up
from evaluation.tasks import build_adapter, default_sigma
from models import build_model


def find_runs(root):
    """Every directory under `root` holding both a config.json and a net.ckpt."""
    out = []
    for dirpath, _, filenames in os.walk(root):
        if "config.json" in filenames and "net.ckpt" in filenames:
            out.append(dirpath)
    return sorted(out)


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
def evaluate(model, cfg, loader, metrics, sigma, device, seed):
    """Per-slice metric values over `loader` at one pinned noise level."""
    adapter = build_adapter(cfg["task"])
    gen = torch.Generator(device=device).manual_seed(int(seed))
    acc = {name: [] for name in metrics}

    for batch in loader:
        gt, recon, organ_mask = adapter(model, batch, cfg, device, sigma, gen)
        for name, fn in metrics.items():
            acc[name].extend(
                fn(gt, recon, mask=organ_mask).detach().cpu().tolist())
    return acc


def summarise(acc):
    out = {}
    for name, vals in acc.items():
        out[f"{name}_mean"] = sum(vals) / len(vals) if vals else float("nan")
        out[f"{name}_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return out


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-config", help="JSON with any of the options below; "
                                         "explicit CLI flags win over it")
    p.add_argument("--runs", help="root of the trained-run tree")
    p.add_argument("--out", help="CSV output path")
    p.add_argument("--metrics", nargs="*",
                   help=f"subset of {sorted(METRIC_REGISTRY)}")
    p.add_argument("--sigmas", type=float, nargs="*",
                   help="noise levels; default is each run's val_noise_std")
    p.add_argument("--seed", type=int)
    p.add_argument("--device")
    p.add_argument("--limit", type=int, help="stop after N batches (smoke test)")
    args = p.parse_args()

    ecfg = {}
    if args.eval_config:
        with open(args.eval_config) as f:
            ecfg = json.load(f)

    def opt(name, default=None):
        v = getattr(args, name, None)
        return v if v not in (None, []) else ecfg.get(name, default)

    runs_root = opt("runs", "trained_nets/mg_recon")
    out_path = opt("out", "results/evaluation.csv")
    metric_names = opt("metrics", DEFAULT_METRICS)
    sigmas = opt("sigmas", None)
    seed = opt("seed", 1234)
    device = torch.device(opt("device",
                              "cuda" if torch.cuda.is_available() else "cpu"))
    limit = opt("limit", None)

    metrics = build_metrics(metric_names)
    # Anything that downloads or compiles is built now, not mid-sweep.
    warm_up(metric_names, device)

    runs = find_runs(runs_root)
    if not runs:
        raise SystemExit(
            f"no runs with both config.json and net.ckpt under {runs_root}")
    print(f"{len(runs)} run(s) under {runs_root} | metrics: {', '.join(metric_names)} "
          f"| device: {device}")

    rows = []
    for run_dir in runs:
        # `--runs` may point AT a single run, in which case relpath is "."
        rel = os.path.relpath(run_dir, runs_root)
        if rel == ".":
            rel = os.path.basename(os.path.normpath(run_dir))
        try:
            cfg, model, n_params, step = load_run(run_dir, device)
        except Exception as e:                 # a bad cell must not sink the sweep
            print(f"[skip] {rel}: {type(e).__name__}: {e}")
            continue

        loader = build_loader(cfg["data"]["val"], shuffle=False, drop_last=False)
        if limit:
            loader = [b for _, b in zip(range(limit), loader)]

        run_sigmas = sigmas if sigmas else [default_sigma(cfg)]
        for sigma in run_sigmas:
            acc = evaluate(model, cfg, loader, metrics, sigma, device, seed)
            stats = summarise(acc)
            n = len(next(iter(acc.values()))) if acc else 0
            rows.append({
                "run": rel,
                "task": cfg["task"],
                "type": cfg["model"]["type"],
                "name": cfg.get("experiment", {}).get("name", ""),
                "R": cfg.get("mri", {}).get("R", ""),
                "sigma": sigma,
                "n_slices": n,
                "n_params": n_params,
                "step": step,
                **{k: round(v, 6) for k, v in stats.items()},
            })
            shown = "  ".join(
                f"{m}={METRIC_REGISTRY[m]['fmt'].format(stats[f'{m}_mean'])}"
                for m in metric_names)
            print(f"  {rel:<34} sigma={sigma:<7g} n={n:<4} {shown}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not rows:
        raise SystemExit("every run failed to load; nothing written")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
