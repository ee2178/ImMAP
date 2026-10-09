# -*- coding: utf-8 -*-
"""
Guided nets timed across BACKENDS x NUMBER OF GUIDES x BATCH SIZE, on synthetic data.

    python -m scripts.time_guided_sweep config/NYUMets/i2sb_lggs_session.json
           [--backends gather_loop gather flex triton] [--guides 1 3 5] [--batches 1 2 4 8 16]
           [--size 128] [--reps 5] [--warmup 2] [--csv out.csv]

The model is built from the config exactly as train.py builds it, with only `attn_backend` (and,
where a backend cannot carry the config's similarity, `sim_fun`) replaced. Inputs are random
tensors of the shapes the trainer hands the net: the bridge state, the config's conditioning
channels, and G guide planes -- G is the sweep variable, whatever the config's loader would
return. No dataset is read.

Backends:
    gather_loop   the gather backend with its ORIGINAL one-offset-at-a-time similarity
                  (IMMAP_SIM_LOOP): the baseline before 2026-10-09
    gather        the gather backend, vectorised similarity
    flex          FlexAttention, fused; compiled as train.py compiles it
    triton        the hand-written fused kernel (phase-invariant similarities only)

Every cell reports a TRAINING step (forward, loss, backward, clip, optimizer step, project()),
an INFERENCE forward (no grad), and the peak GPU memory of the training step. A cell that runs
out of memory is recorded as OOM and the larger batches of that (backend, guides) pair are
skipped; any other failure is recorded with its message and the sweep goes on.

The last table is the one to read: for each backend and guide count, the largest batch that
fit and the best throughput.

flex and triton need CUDA; on a CPU-only machine those cells are reported as unavailable.
"""

import argparse
import csv
import time

import torch
import yaml

import models.circulant_similarity as circ_sim
from models import build_model
from models.enhancement import enhancement_loss
from models.prox import FLEX_SIMS, TRITON_SIMS
from sb.base import build_schedule, forward_std, n_steps, predict_x0

BACKENDS = ("gather_loop", "gather", "flex", "triton")


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def _is_oom(err):
    return "out of memory" in str(err).lower()


def model_for(cfg, backend):
    """-> (cfg for this backend, note). gather_loop is gather with the loop switched on."""
    cfg = yaml.safe_load(yaml.safe_dump(cfg))                # deep copy
    p = cfg["model"]["params"]
    real = "gather" if backend == "gather_loop" else backend
    p["attn_backend"] = real
    sim, note = p.get("sim_fun", "distance"), ""
    if real == "triton" and sim not in TRITON_SIMS:
        p["sim_fun"], note = "pidistance", f"sim_fun {sim} -> pidistance (triton)"
    if real == "flex" and sim not in FLEX_SIMS:
        p["sim_fun"], note = "distance", f"sim_fun {sim} -> distance (flex)"
    return cfg, note


def make_step(cfg, net, dev, size, n_guides):
    """fn(batch) -> loss, on a fresh synthetic batch with `n_guides` guide planes."""
    if cfg.get("task") != "i2sb":
        raise ValueError(f"task {cfg.get('task')!r}: this sweep times the bridge regressors")
    i2 = cfg["i2sb"]
    bridge = build_schedule(kind=i2.get("kind", "brownian"), tau=i2.get("tau", 0.1),
                            n_points=i2.get("n_points", 1000), beta_max=i2.get("beta_max", 0.3),
                            device=dev)
    n_cond = len(cfg["data"]["train"].get("cond_idx") or [])
    s_weight = float(cfg["training"].get("s_weight") or 0.0)

    def step(b):
        x0 = torch.rand(b, 1, size, size, device=dev)
        x1 = torch.rand(b, 1, size, size, device=dev)
        cond = torch.rand(b, n_cond, size, size, device=dev) if n_cond else None
        guide = torch.rand(b, n_guides, 1, size, size, device=dev) if n_guides else None
        k = torch.randint(0, n_steps(bridge), (b,), device=dev)
        pred = predict_x0(net, x0, forward_std(bridge, k, xdim=x0.shape[1:]), cond=cond, guide=guide)
        loss = (pred - x0).abs().pow(2).mean()
        if s_weight:
            loss = loss + s_weight * enhancement_loss(net, x0, x1)
        return loss

    return step


def time_cell(net, opt, step, clip, b, dev, warmup, reps):
    """-> (train ms, inference ms, peak GB) for one batch size."""
    if dev.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    t_train = []
    net.train()
    for i in range(warmup + reps):
        opt.zero_grad(set_to_none=True)
        _sync(dev); t0 = time.perf_counter()
        step(b).backward()
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
        opt.step()
        if hasattr(net, "project"):
            net.project()
        _sync(dev)
        if i >= warmup:
            t_train.append((time.perf_counter() - t0) * 1e3)
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if dev.type == "cuda" else float("nan")
    opt.zero_grad(set_to_none=True)

    t_inf = []
    net.eval()
    with torch.no_grad():
        for i in range(warmup + reps):
            _sync(dev); t0 = time.perf_counter()
            step(b)
            _sync(dev)
            if i >= warmup:
                t_inf.append((time.perf_counter() - t0) * 1e3)
    med = lambda v: sorted(v)[len(v) // 2]                    # noqa: E731
    return med(t_train), med(t_inf), peak


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--backends", nargs="+", default=list(BACKENDS), choices=BACKENDS)
    ap.add_argument("--guides", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--size", type=int, default=128, help="frame side (the training crop)")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--model-override", type=yaml.safe_load, default=None,
                    help="YAML/JSON merged into model.params, e.g. '{K: 10, guide_window: 7}'")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.config) as f:
        base = yaml.safe_load(f)
    base["model"]["params"].update(args.model_override or {})
    p0 = base["model"]["params"]
    if dev.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}, "
              f"{torch.cuda.get_device_properties(0).total_memory / 2 ** 30:.0f} GB")
    else:
        print("NO GPU: these timings say nothing about the cluster, and flex / triton cannot run")
    print(f"{base['model']['type']}  K={p0.get('K')} M={p0.get('M')} Mh={p0.get('Mh')} "
          f"window={p0.get('window')} guide_window={p0.get('guide_window')} dK={p0.get('dK')} "
          f"joint_softmax={p0.get('joint_softmax', True)} sim={p0.get('sim_fun')}   frame {args.size}")

    rows = []
    for backend in args.backends:
        cfg, note = model_for(base, backend)
        circ_sim.VECTORIZED = backend != "gather_loop"
        try:
            net = build_model(cfg).to(dev)
            if backend == "flex":
                if dev.type != "cuda":
                    raise RuntimeError("flex needs CUDA")
                net.compile_flex()
            if backend == "triton" and dev.type != "cuda":
                raise RuntimeError("triton needs CUDA")
        except Exception as err:                              # noqa: BLE001
            print(f"\n=== {backend}: unavailable -- {type(err).__name__}: {str(err)[:150]}")
            rows += [dict(backend=backend, guides=g, batch=b, status="unavailable")
                     for g in args.guides for b in args.batches]
            continue
        opt = torch.optim.Adam(net.parameters(), lr=1e-6)     # real update, negligible movement
        clip = cfg["training"].get("clip_grad")
        print(f"\n=== {backend}" + (f"   [{note}]" if note else "") + " ===")
        print(f"{'guides':>6} {'batch':>6} {'train ms':>9} {'infer ms':>9} {'train smp/s':>12} {'peak GB':>8}")
        for g in args.guides:
            step = make_step(cfg, net, dev, args.size, g)
            for b in args.batches:
                row = dict(backend=backend, guides=g, batch=b)
                try:
                    tr, inf, gb = time_cell(net, opt, step, clip, b, dev, args.warmup, args.reps)
                    row.update(status="ok", train_ms=tr, infer_ms=inf, peak_gb=gb,
                               samples_per_s=b / tr * 1e3)
                    print(f"{g:>6d} {b:>6d} {tr:>9.1f} {inf:>9.1f} {b / tr * 1e3:>12.2f} {gb:>8.2f}")
                    rows.append(row)
                except Exception as err:                      # noqa: BLE001
                    oom = isinstance(err, RuntimeError) and _is_oom(err)
                    row["status"] = "OOM" if oom else f"{type(err).__name__}: {str(err)[:120]}"
                    print(f"{g:>6d} {b:>6d}   {row['status']}")
                    rows.append(row)
                    opt.zero_grad(set_to_none=True)
                    if dev.type == "cuda":
                        torch.cuda.empty_cache()
                    if oom:
                        break                                 # larger batches will not fit either
                    if b == args.batches[0]:
                        break                                 # not a size problem: skip this guide count
        del net, opt
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    circ_sim.VECTORIZED = True

    # ---- the summary: per backend and guide count, the largest batch that fit ----------------
    print(f"\n{'backend':<12} {'guides':>6} {'max batch':>10} {'best smp/s':>11} {'at batch':>9} "
          f"{'ms/step':>9} {'peak GB':>8}")
    for backend in args.backends:
        for g in args.guides:
            ok = [r for r in rows if r["backend"] == backend and r["guides"] == g and r["status"] == "ok"]
            if not ok:
                why = next((r["status"] for r in rows
                            if r["backend"] == backend and r["guides"] == g), "not run")
                print(f"{backend:<12} {g:>6d}   -- {why[:70]}")
                continue
            best = max(ok, key=lambda r: r["samples_per_s"])
            big = max(ok, key=lambda r: r["batch"])
            print(f"{backend:<12} {g:>6d} {big['batch']:>10d} {best['samples_per_s']:>11.2f} "
                  f"{best['batch']:>9d} {best['train_ms']:>9.1f} {big['peak_gb']:>8.2f}")

    if args.csv:
        keys = ["backend", "guides", "batch", "status", "train_ms", "infer_ms", "samples_per_s", "peak_gb"]
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
