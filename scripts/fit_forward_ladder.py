"""
Fit a ladder of learned forward operators E: CT1 -> T1 on NYUMets and report, per rung,
how well T1 is explained and what a data-consistency step through E would cost.

    python -m scripts.fit_forward_ladder config/NYUMets/forward_ladder.json

The question it answers: is there an E small enough that J_E^T r is cheap next to the bridge
regressor, yet accurate enough that || T1 - E(CT1) || is clearly below the trivial || T1 - CT1 ||?
If no rung beats identity by much, a T1 data term carries little information and the bridge
code is not worth changing.

All rungs train SIMULTANEOUSLY on the same batches (one data stream, one optimizer each), so the
comparison is not confounded by data order and the job is IO-bound once, not once per rung.
Everything goes to ONE wandb run, keyed `<rung>/...`, plus a summary table.

Validation, on a fixed random subset of val slices, inside the brain mask:
    rmse          sqrt(mean (T1 - E(CT1))^2)                -- also the sigma to weight a DC term by
    rmse_enh      same, on the pixels where CT1 - T1 is in the top `enh_quantile` of that slice
                  (a proxy for enhancement; there is no segmentation)
    rmse_rest     same, on the remaining pixels
    rel_err       || T1 - E(CT1) || / || T1 ||
    gain          rmse(identity) / rmse(E)                  -- > 1 means E beats T1 ~ CT1
    enh_removed   <CT1 - E(CT1), CT1 - T1> / ||CT1 - T1||^2 on the enhancement pixels
                  1 = E removes exactly the CT1/T1 difference there, 0 = leaves it untouched
    x_swap_rmse   rmse of T1 - E(CT1', c) with CT1' taken from ANOTHER val slice (the batch rolled
                  by one; the val subset is in random order, so partners are unrelated slices),
                  while c (T2, FLAIR) and the target stay those of the true slice
    x_swap_ratio  x_swap_rmse / rmse. ~1 means E IGNORES x -- it is synthesising T1 from c, its
                  x-Jacobian is ~0, and it is useless as a data-consistency operator however low
                  its rmse. Large means the output really depends on CT1.

Operators may be conditioned on side information c (the loader's cond_idx, e.g. T2 + FLAIR) by
giving a rung `cond_channels`; `use_x: false` makes the c-only reference E(c). Rungs without
`cond_channels` ignore c, so x-only and conditioned rungs share one run and one data order.

Timing (ms, median, at the val frame size): E forward, VJP, VJP with create_graph + backward
(the cost of training through a DC step), and a forward of each reference bridge net.
"""

import json
import os
import sys
import time

import numpy as np
import torch
import yaml
import wandb
import matplotlib.pyplot as plt       # backend is chosen in main(), never at import
from torch.utils.data import DataLoader, Subset

import datasets                                   # noqa: F401  (registers loaders)
from datasets.registry import build_loader
from models.forward_ops import ForwardOp
from training.forward_op import _inputs, masked_mse, batch_xyc, validate, panel_figure
from visualization.image import set_display_orient


# ---------------------------------------------------------------------------------------------
def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_ms(fn, device, warmup, reps):
    for _ in range(warmup):
        fn()
    _sync(device)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        _sync(device)
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))


def time_ops(ops, shape, device, warmup, reps):
    out = {}
    x = torch.randn(shape, device=device)
    y = torch.randn(shape, device=device)
    for name, E in ops.items():
        E.eval()
        xg = x.clone().requires_grad_(True)
        c = (torch.randn(shape[0], E.cond_channels, *shape[2:], device=device)
             if E.cond_channels else None)

        def fwd():
            with torch.no_grad():
                E(x, c)

        def vjp():
            E.data_grad(x, y, c)

        def vjp2():
            g = E.data_grad(xg, y, c, create_graph=True)
            if g.requires_grad:                  # a c-only E has an identically zero x-gradient
                g.sum().backward()
            xg.grad = None
            for p in E.parameters():
                p.grad = None

        out[name] = {"fwd_ms": time_ms(fwd, device, warmup, reps),
                     "vjp_ms": time_ms(vjp, device, warmup, reps),
                     "vjp2_ms": time_ms(vjp2, device, warmup, reps)}
        E.train()
    return out


def time_references(paths, shape, device, warmup, reps):
    """Forward time of each reference bridge regressor at the same frame size. Best effort:
    a reference that fails to build or run is reported and skipped, never fatal."""
    from models import build_model
    from sb.base import predict_x0

    out = {}
    B, _, H, W = shape
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            with open(path) as f:
                rcfg = yaml.safe_load(f)
            net = build_model(rcfg).to(device).eval()
            C = int(rcfg["model"]["params"].get("C", 1))
            xt = torch.randn(B, 1, H, W, device=device)
            cond = torch.randn(B, C - 1, H, W, device=device)
            sigma = torch.full((B, 1, 1, 1), 0.1, device=device)

            def fwd():
                with torch.no_grad():
                    predict_x0(net, xt, sigma, cond=cond)

            out[name] = {"fwd_ms": time_ms(fwd, device, max(2, warmup // 2), max(5, reps // 5))}
            del net
        except Exception as err:                   # noqa: BLE001
            print(f"[timing] reference {path} skipped: {type(err).__name__}: {err}")
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------------------------------------
def main(config_path):
    plt.switch_backend("Agg")                 # headless figures for wandb; only when RUN, not imported
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, tm = cfg["training"], cfg.get("timing", {})
    torch.manual_seed(int(tr.get("seed", 0)))
    # DISPLAY ONLY, and this ladder is NYUMets-only: its h5s store canonical-RAS axes, so the
    # val panels come out with the eyes on the image's right unless they are rotated.
    set_display_orient(tr.get("display_orient", "radiological"))

    save_dir = cfg["paths"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    train_loader = build_loader(cfg["data"]["train"], shuffle=True, drop_last=True)
    full_val = build_loader(cfg["data"]["val"], shuffle=False, drop_last=False)
    n_val = len(full_val.dataset)
    k = min(int(tr.get("val_slices", 4000)), n_val)
    # RANDOM order, not sorted: x_swap pairs each slice with its batch neighbour, and in sorted
    # order that neighbour is usually the adjacent slice of the same volume -- nearly the same CT1,
    # which would make every operator look like it ignores x.
    idx = np.random.default_rng(int(tr.get("val_seed", 0))).choice(n_val, size=k, replace=False)
    val_loader = DataLoader(Subset(full_val.dataset, idx.tolist()),
                            batch_size=full_val.batch_size, shuffle=False,
                            num_workers=full_val.num_workers, pin_memory=True)
    print(f"train slices {len(train_loader.dataset)}  |  val subset {k}/{n_val}")
    n_cond = len(cfg["data"]["train"].get("cond_idx", []) or [])

    ops, opts, scheds = {}, {}, {}
    for spec in cfg["rungs"]:
        spec = dict(spec)
        name = spec.pop("name")
        if name in ops or name == "identity":
            raise ValueError(f"duplicate/reserved rung name {name!r}")
        ops[name] = ForwardOp(**spec).to(device)
        if ops[name].cond_channels not in (0, n_cond):
            raise ValueError(f"rung {name}: cond_channels={ops[name].cond_channels} but the loader "
                             f"supplies {n_cond} (data.train.cond_idx)")
        opts[name] = torch.optim.Adam(ops[name].parameters(), lr=float(tr["lr"]))
        scheds[name] = torch.optim.lr_scheduler.CosineAnnealingLR(
            opts[name], T_max=int(tr["steps"]), eta_min=float(tr.get("eta_min", 0.0)))
        print(f"  {name:>22s}  inputs {_inputs(ops[name]):>3s}  "
              f"params {sum(p.numel() for p in ops[name].parameters()):>8d}  "
              f"receptive field {ops[name].receptive_field}")

    wandb.init(project=cfg["wandb"]["project"], name=cfg["experiment"]["name"],
               id=cfg["wandb"].get("id"), resume="allow", config=cfg)

    steps, log_every = int(tr["steps"]), int(tr.get("log_every", 100))
    val_every, clip = int(tr.get("val_every", 5000)), float(tr.get("clip_grad", 0.0))
    enh_q = float(tr.get("enh_quantile", 0.98))

    it = iter(train_loader)
    run = {name: 0.0 for name in ["identity", *ops]}
    t_start = time.time()
    res = None
    for step in range(1, steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)
        x, y, c, m = batch_xyc(batch, device)
        with torch.no_grad():
            run["identity"] += float(masked_mse(x, y, m))
        for name, E in ops.items():
            opts[name].zero_grad(set_to_none=True)
            loss = masked_mse(E(x, c), y, m)
            loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(E.parameters(), clip)
            opts[name].step()
            scheds[name].step()
            run[name] += float(loss)

        if step % log_every == 0:
            log = {f"{n}/train_mse": v / log_every for n, v in run.items()}
            log["lr"] = scheds[next(iter(ops))].get_last_lr()[0]
            log["steps_per_sec"] = step / (time.time() - t_start)
            wandb.log(log, step=step)
            run = {n: 0.0 for n in run}

        if step % val_every == 0 or step == steps:
            res, panel = validate(ops, val_loader, device, enh_q)
            wandb.log({f"{n}/val_{k}": v for n, r in res.items() for k, v in r.items()}, step=step)
            if panel is not None:
                fig = panel_figure(panel, step, res)
                wandb.log({"val/panel": wandb.Image(fig)}, step=step)
                plt.close(fig)
            for name, E in ops.items():
                spec = next(dict(s) for s in cfg["rungs"] if s["name"] == name)
                torch.save({"spec": spec, "state_dict": E.state_dict(), "step": step,
                            "val": res[name]}, os.path.join(save_dir, f"E_{name}.pt"))
            print(f"[step {step}] " + "  ".join(
                f"{n}: rmse {r['rmse']:.4f} gain {r['gain']:.3f}" for n, r in res.items()))

    # ---- timing, at the val frame size ------------------------------------------------------
    H, W = next(iter(val_loader))[0].shape[-2:]
    warmup, reps = int(tm.get("warmup", 10)), int(tm.get("reps", 50))
    timing = {}
    for B in tm.get("batch_sizes", [1, 8]):
        shape = (int(B), 1, int(H), int(W))
        t_ops = time_ops(ops, shape, device, warmup, reps)
        t_ref = time_references(tm.get("reference_configs", []), shape, device, warmup, reps)
        timing[int(B)] = (t_ops, t_ref)
        wandb.log({f"timing/b{B}/{n}/{k}": v for d in (t_ops, t_ref)
                   for n, r in d.items() for k, v in r.items()}, step=steps)

    # ---- summary table ----------------------------------------------------------------------
    B0 = int(tm.get("batch_sizes", [1, 8])[-1])
    t_ops, t_ref = timing[B0]
    ref_ms = {n: r["fwd_ms"] for n, r in t_ref.items()}
    cols = ["rung", "kind", "inputs", "params", "receptive_field", "rmse", "rmse_enh", "rmse_rest",
            "rel_err", "gain", "enh_removed", "x_swap_rmse", "x_swap_ratio",
            f"fwd_ms_b{B0}", f"vjp_ms_b{B0}", f"vjp2_ms_b{B0}"]
    cols += [f"vjp_over_{n}" for n in ref_ms]
    table = wandb.Table(columns=cols)
    rows = []
    for name in ["identity", *ops]:
        r = res[name]
        if name == "identity":
            meta = ["identity", "x", 0, 1]
            t = {"fwd_ms": 0.0, "vjp_ms": 0.0, "vjp2_ms": 0.0}
        else:
            E = ops[name]
            meta = [E.kind, _inputs(E), sum(p.numel() for p in E.parameters()), E.receptive_field]
            t = t_ops[name]
        row = [name, *meta, r["rmse"], r["rmse_enh"], r["rmse_rest"], r["rel_err"], r["gain"],
               r["enh_removed"], r["x_swap_rmse"], r["x_swap_ratio"],
               t["fwd_ms"], t["vjp_ms"], t["vjp2_ms"]]
        row += [t["vjp_ms"] / ms if ms else float("nan") for ms in ref_ms.values()]
        table.add_data(*row)
        rows.append(dict(zip(cols, row)))
    wandb.log({"summary": table}, step=steps)
    for n, ms in ref_ms.items():
        wandb.summary[f"reference/{n}/fwd_ms_b{B0}"] = ms

    # plain results file for the log directory (NOT a config: never fed back to train.py)
    with open(os.path.join(save_dir, "ladder_results.json"), "w") as f:
        json.dump({"rows": rows, "reference_fwd_ms": ref_ms, "batch": B0,
                   "frame": [int(H), int(W)]}, f, indent=2)

    print(f"\n{'rung':>22s} {'in':>3s} {'params':>8s} {'RF':>3s} {'rmse':>7s} {'enh':>7s} "
          f"{'rest':>7s} {'gain':>6s} {'removed':>7s} {'swap':>7s} {'swap/rm':>7s} "
          f"{'fwd':>7s} {'vjp':>7s} {'vjp2':>7s}   (ms, B={B0})")
    for rw in rows:
        print(f"{rw['rung']:>22s} {rw['inputs']:>3s} {rw['params']:>8d} {rw['receptive_field']:>3d} "
              f"{rw['rmse']:7.4f} {rw['rmse_enh']:7.4f} {rw['rmse_rest']:7.4f} {rw['gain']:6.3f} "
              f"{rw['enh_removed']:7.3f} {rw['x_swap_rmse']:7.4f} {rw['x_swap_ratio']:7.2f} "
              f"{rw[f'fwd_ms_b{B0}']:7.2f} {rw[f'vjp_ms_b{B0}']:7.2f} {rw[f'vjp2_ms_b{B0}']:7.2f}")
    for n, ms in ref_ms.items():
        print(f"  reference {n}: forward {ms:.2f} ms")
    wandb.finish()


if __name__ == "__main__":
    main(sys.argv[1])
