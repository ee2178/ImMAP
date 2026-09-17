"""
Time a CDLNet forward pass with the learned data-consistency operator.

    python -m scripts.time_cdlnet_dc --config config/NYUMets/i2sb_cdlnet_learned_dc_mini.json \
           --size 240 --batch 8 --backward

What it answers: how much the learned E actually costs inside an unrolled net. Each of the K
layers adds one E forward + one VJP (operators/learned.py), so the overhead should be
K * vjp_ms -- this measures whether it is, and what it is next to the dictionary convolutions.

Reported, all medians over `--reps` after `--warmup` (ms):

    cdlnet+dc        CDLNet forward through BridgeDCOperator  (the real regressor call)
    cdlnet+identity  the SAME net and K on the linear path    (the dictionary-only cost)
    overhead         cdlnet+dc - cdlnet+identity, and that divided by K (per-layer E cost)
    E fwd / E vjp    the operator alone, one call, same frame and batch
    train step       forward + loss + backward, with the second-order term through the VJP
                     (--backward; this is the number that sets epoch time)

`--reference` also times one forward of other regressors' configs (SBCDLNet, SBUnet, ...) so the
comparison is against what you would otherwise run. Peak CUDA memory is reported per measurement.

Everything runs on random data: this is a cost measurement, not a quality one. It does need a
real E checkpoint, since the operator's cost depends on E's size (cfg["i2sb"]["learned_dc"]).
"""

import argparse
import os
import time

import numpy as np
import torch
import yaml

from models import build_model
from sb.base import build_schedule, forward_std, n_steps, predict_x0
from sb.learned_dc import LearnedBridgeDC
from operators import Identity


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_ms(fn, device, warmup, reps):
    for _ in range(warmup):
        fn()
    _sync(device)
    ts, peak = [], 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        _sync(device)
        ts.append((time.perf_counter() - t0) * 1e3)
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 2 ** 20
    return float(np.median(ts)), float(peak)


def _row(name, ms, mem, extra=""):
    print(f"{name:>26s} {ms:9.2f} ms {mem:9.1f} MB  {extra}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/NYUMets/i2sb_cdlnet_learned_dc_mini.json")
    ap.add_argument("--size", type=int, default=240, help="frame size (square)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--t", type=float, default=0.5, help="bridge position in [0, 1]")
    ap.add_argument("--K", type=int, default=None, help="override the config's unrolled depth")
    ap.add_argument("--backward", action="store_true", help="also time a full training step")
    ap.add_argument("--reference", nargs="*", default=[],
                    help="other configs to time one forward of, e.g. config/NYUMets/i2sb_sbcdlnet_all.json")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.K is not None:
        cfg["model"]["params"]["K"] = args.K
    # spectral_init runs a 200-iteration power method; irrelevant to timing, slow to build
    cfg["model"]["params"]["init"] = False

    net = build_model(cfg).to(device).eval()
    K = int(cfg["model"]["params"].get("K", 0))
    ldc = dict(cfg["i2sb"].get("learned_dc") or {})
    if not ldc:
        raise ValueError(f"{args.config} has no i2sb.learned_dc block")

    i2 = cfg["i2sb"]
    sched = build_schedule(kind=i2.get("kind", "brownian"), tau=i2.get("tau", 0.19),
                           n_points=i2.get("n_points", 1000), beta_max=i2.get("beta_max", 0.3),
                           device=device)
    dc = LearnedBridgeDC(sched, device, **ldc)

    B, S = args.batch, args.size
    step = torch.full((B,), int(round(args.t * (n_steps(sched) - 1))), device=device,
                      dtype=torch.long)
    sigma = forward_std(sched, step, xdim=(1, S, S))
    xt = torch.randn(B, 1, S, S, device=device)
    x1 = torch.randn(B, 1, S, S, device=device)
    x0 = torch.randn(B, 1, S, S, device=device)
    cond = torch.randn(B, dc.cond_channels, S, S, device=device)

    n_par = sum(p.numel() for p in net.parameters())
    n_E = sum(p.numel() for p in dc.E.parameters())
    print(f"\ndevice {device.type}  |  {type(net).__name__} K={K} params {n_par/1e6:.2f}M  |  "
          f"E params {n_E/1e6:.3f}M  |  batch {B} at {S}x{S}, t={args.t}\n")
    print(f"{'measurement':>26s} {'median':>12s} {'peak':>12s}")

    def f_dc():
        with torch.no_grad():
            predict_x0(net, xt, sigma, cond=cond, dc=dc, x1=x1)

    ms_dc, mem_dc = time_ms(f_dc, device, args.warmup, args.reps)
    _row("cdlnet+dc", ms_dc, mem_dc)

    # same net, linear path: C=1 means the identity operator is a valid stand-in for the
    # dictionary-only cost (it skips the data term's E entirely)
    def f_id():
        with torch.no_grad():
            net(xt, E=Identity(), sigma=sigma)

    ms_id, mem_id = time_ms(f_id, device, args.warmup, args.reps)
    _row("cdlnet+identity", ms_id, mem_id)
    over = ms_dc - ms_id
    print(f"{'overhead':>26s} {over:9.2f} ms {'':>12s}  "
          f"{over / max(K, 1):.2f} ms/layer over {K} layers "
          f"({100 * over / max(ms_dc, 1e-9):.0f}% of the forward)")

    op = dc.bind(sigma, x1, cond)
    x = torch.randn(B, 1, S, S, device=device)

    def f_E():
        with torch.no_grad():
            op.E(x)

    def f_vjp():
        with torch.no_grad():
            op.data_grad(x, xt)

    _row("E forward (1 call)", *time_ms(f_E, device, args.warmup, args.reps))
    ms_vjp, mem_vjp = time_ms(f_vjp, device, args.warmup, args.reps)
    _row("E data_grad (1 call)", ms_vjp, mem_vjp, f"x{K} = {ms_vjp * K:.1f} ms predicted overhead")

    if args.backward:
        net.train()
        opt = torch.optim.Adam(net.parameters(), lr=0.0)

        def f_train():
            opt.zero_grad(set_to_none=True)
            pred = predict_x0(net, xt, sigma, cond=cond, dc=dc, x1=x1)
            ((pred - x0) ** 2).mean().backward()
            opt.step()

        _row("train step (dc)", *time_ms(f_train, device, args.warmup, args.reps),
             "fwd+bwd, 2nd order through the VJP")

        def f_train_id():
            opt.zero_grad(set_to_none=True)
            pred, _ = net(xt, E=Identity(), sigma=sigma)
            ((pred - x0) ** 2).mean().backward()
            opt.step()

        _row("train step (identity)", *time_ms(f_train_id, device, args.warmup, args.reps))
        net.eval()

    for path in args.reference:
        try:
            with open(path) as f:
                rcfg = yaml.safe_load(f)
            rnet = build_model(rcfg).to(device).eval()
            C = int(rcfg["model"]["params"].get("C", 1))
            rcond = torch.randn(B, C - 1, S, S, device=device) if C > 1 else None

            def f_ref():
                with torch.no_grad():
                    predict_x0(rnet, xt, sigma, cond=rcond)

            ms, mem = time_ms(f_ref, device, max(2, args.warmup // 2), max(5, args.reps // 4))
            _row(os.path.splitext(os.path.basename(path))[0], ms, mem,
                 f"{sum(p.numel() for p in rnet.parameters())/1e6:.2f}M params")
            del rnet
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as err:                       # noqa: BLE001
            print(f"  reference {path} skipped: {type(err).__name__}: {err}")

    print()


if __name__ == "__main__":
    main()
