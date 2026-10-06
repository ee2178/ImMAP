# -*- coding: utf-8 -*-
"""
How large a batch fits, and what does it cost?  One TRAINING step timed across batch sizes.

    python -m scripts.time_batch_scaling config/NYUMets/i2sb_sbcdlnet_sgate.json [more configs]
           [--batches 4 8 16 32 64 128 256] [--size 128] [--targets 64 256] [--steps 300000]

For each config the model is built exactly as train.py builds it and fed a SYNTHETIC batch with
the shapes the config's loader would produce (no dataset needed: this measures the network, and
the data pipeline is the same at every batch size). A step is what the trainer does:

    forward -> loss (+ s_weight * S loss when the config has one) -> backward        per micro-batch
    clip -> optimizer step -> net.project()                                            per optimizer step

The first table is per micro-batch: time, throughput and peak GPU memory, up to the first size
that runs out of memory. The second answers the question directly: for each EFFECTIVE batch in
--targets, the largest micro-batch that fits, the accumulation that reaches it, the time per
optimizer step and the wall clock for --steps optimizer steps.

Throughput (samples / s) flat across batch sizes means the GPU is already saturated and a bigger
batch buys nothing per sample -- the effective batch then costs the same through accumulation.
"""

import argparse
import csv
import json
import time

import torch
import yaml

from models import build_model
from models.enhancement import enhancement_loss
from sb.base import build_schedule, forward_std, n_steps, predict_x0


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def _is_oom(err):
    return "out of memory" in str(err).lower()


def batch_shapes(cfg):
    """(n_cond, n_guide, guide_as_cond) the config's TRAIN loader would hand the net."""
    d = cfg["data"]["train"]
    n_cond = len(d.get("cond_idx") or [])
    modes = d.get("guide_mode") or "none"
    modes = [modes] if isinstance(modes, str) else list(modes)
    n_guide = 0
    if modes != ["none"]:
        gi = d.get("guide_idx")
        n_c = len(gi) if isinstance(gi, (list, tuple)) else 1
        n_guide = n_c * (2 * int(d.get("guide_window") or 0) + 1) * int(d.get("n_guides") or 1)
    return n_cond, n_guide, bool(d.get("guide_as_cond"))


def make_step(cfg, net, dev, size):
    """-> fn(batch_size) running forward + loss + backward on a fresh synthetic batch."""
    n_cond, n_guide, as_cond = batch_shapes(cfg)
    s_weight = float(cfg["training"].get("s_weight") or 0.0)
    task = cfg.get("task")

    if task == "i2sb":
        i2 = cfg["i2sb"]
        bridge = build_schedule(kind=i2.get("kind", "brownian"), tau=i2.get("tau", 0.1),
                                n_points=i2.get("n_points", 1000),
                                beta_max=i2.get("beta_max", 0.3), device=dev)

        def step(b):
            x0 = torch.rand(b, 1, size, size, device=dev)
            x1 = torch.rand(b, 1, size, size, device=dev)
            cond = torch.rand(b, n_cond + (n_guide if as_cond else 0), size, size, device=dev)
            guide = (torch.rand(b, n_guide, 1, size, size, device=dev)
                     if (n_guide and not as_cond) else None)
            k = torch.randint(0, n_steps(bridge), (b,), device=dev)
            pred = predict_x0(net, x0, forward_std(bridge, k, xdim=x0.shape[1:]),
                              cond=cond if cond.shape[1] else None, guide=guide)
            loss = ((pred - x0) ** 2).mean()
            if s_weight:
                loss = loss + s_weight * enhancement_loss(net, x0, x1)
            return loss
    elif task == "synthesis":
        src = getattr(net, "prior_idx", None)

        def step(b):
            X = torch.rand(b, n_cond + n_guide, size, size, device=dev)
            y = torch.rand(b, 1, size, size, device=dev)
            out = net(X)
            pred = out[0] if isinstance(out, (tuple, list)) else out
            loss = ((pred - y) ** 2).mean()
            if s_weight:
                loss = loss + s_weight * enhancement_loss(net, y, X[:, src:src + 1])
            return loss
    else:
        raise ValueError(f"task {task!r}: only i2sb and synthesis are timed here")
    return step


def time_config(path, args, dev):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for k, v in (args.model_override or {}).items():
        cfg["model"]["params"][k] = v
    name = path.replace("\\", "/").split("/")[-1].replace(".json", "")
    net = build_model(cfg).to(dev).train()
    opt = torch.optim.Adam(net.parameters(), lr=1e-6)      # real update, negligible movement
    clip = cfg["training"].get("clip_grad")
    step = make_step(cfg, net, dev, args.size)
    n_par = sum(p.numel() for p in net.parameters()) / 1e6
    print(f"\n=== {name}   {type(net).__name__}, {n_par:.2f}M params, task {cfg.get('task')}, "
          f"frame {args.size}, config batch {cfg['data']['train'].get('batch_size')} "
          f"x accum {cfg['training'].get('accum_steps', 1)} ===")
    print(f"{'batch':>6} {'ms/micro':>9} {'ms/opt':>8} {'samples/s':>10} {'peak GB':>8}")

    rows = []
    for b in args.batches:
        try:
            if dev.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            t_micro, t_opt = [], []
            for i in range(args.warmup + args.reps):
                opt.zero_grad(set_to_none=True)
                _sync(dev); t0 = time.perf_counter()
                step(b).backward()
                _sync(dev); t1 = time.perf_counter()
                if clip is not None:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
                opt.step()
                if hasattr(net, "project"):
                    net.project()
                _sync(dev); t2 = time.perf_counter()
                if i >= args.warmup:
                    t_micro.append((t1 - t0) * 1e3)
                    t_opt.append((t2 - t1) * 1e3)
            ms = sorted(t_micro)[len(t_micro) // 2]
            ms_opt = sorted(t_opt)[len(t_opt) // 2]
            gb = torch.cuda.max_memory_allocated() / 2 ** 30 if dev.type == "cuda" else float("nan")
            rows.append(dict(config=name, batch=b, ms_micro=ms, ms_opt=ms_opt,
                             samples_per_s=b / ms * 1e3, peak_gb=gb))
            print(f"{b:>6d} {ms:>9.1f} {ms_opt:>8.1f} {b / ms * 1e3:>10.1f} {gb:>8.2f}")
        except RuntimeError as err:
            if not _is_oom(err):
                raise
            print(f"{b:>6d}       OOM")
            opt.zero_grad(set_to_none=True)
            break
    del net, opt
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return name, rows


def plan(name, rows, targets, steps):
    """For each effective batch: the cheapest (micro, accum) among the sizes that fit."""
    out = []
    for tgt in targets:
        best = None
        for r in rows:
            if r["batch"] > tgt or tgt % r["batch"]:
                continue
            accum = tgt // r["batch"]
            ms = accum * r["ms_micro"] + r["ms_opt"]
            if best is None or ms < best["ms_step"]:
                best = dict(config=name, effective=tgt, micro=r["batch"], accum=accum, ms_step=ms,
                            days=ms * steps / 1e3 / 86400, peak_gb=r["peak_gb"])
        if best:
            out.append(best)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--batches", type=int, nargs="+", default=[4, 8, 16, 32, 64, 128, 256])
    ap.add_argument("--size", type=int, default=128, help="frame side (the training crop)")
    ap.add_argument("--targets", type=int, nargs="+", default=[8, 64, 256],
                    help="effective batch sizes to plan for")
    ap.add_argument("--steps", type=int, default=300000, help="optimizer steps of a full run")
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--model-override", type=json.loads, default=None,
                    help='JSON merged into model.params, e.g. \'{"attn_backend": "flex"}\'')
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}, "
              f"{torch.cuda.get_device_properties(0).total_memory / 2 ** 30:.0f} GB")
    else:
        print("NO GPU: these timings say nothing about the cluster; plumbing check only")

    all_rows, plans = [], []
    for path in args.configs:
        try:
            name, rows = time_config(path, args, dev)
        except Exception as err:                       # one config must not cost the others
            print(f"!! {path} failed: {type(err).__name__}: {err}")
            continue
        all_rows += rows
        plans += plan(name, rows, args.targets, args.steps)

    print(f"\n{'config':<46} {'effective':>9} {'= micro':>8} {'x accum':>8} {'s/step':>8} "
          f"{'days/' + str(args.steps // 1000) + 'k':>10} {'peak GB':>8}")
    for p in plans:
        print(f"{p['config']:<46} {p['effective']:>9d} {p['micro']:>8d} {p['accum']:>8d} "
              f"{p['ms_step'] / 1e3:>8.2f} {p['days']:>10.1f} {p['peak_gb']:>8.2f}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["config", "batch", "ms_micro", "ms_opt",
                                              "samples_per_s", "peak_gb"])
            w.writeheader()
            w.writerows(all_rows)
        with open(args.csv.replace(".csv", "_plan.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["config", "effective", "micro", "accum", "ms_step",
                                              "days", "peak_gb"])
            w.writeheader()
            w.writerows(plans)
        print(f"\nwrote {args.csv} and {args.csv.replace('.csv', '_plan.csv')}")


if __name__ == "__main__":
    main()
