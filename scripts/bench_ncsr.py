#!/usr/bin/env python3
"""NCSR-LPDS on the GPU: is the fused (triton) path right, and how fast is a training step?

    python scripts/bench_ncsr.py                    # knee R8 cell, check + time
    python scripts/bench_ncsr.py --K 5 --iters 3    # quicker

1. CHECK (small grid, few layers): the fused path (models/circulant_window.py kernels) against
   the roll path, outputs and every gradient. Stops without timing if they disagree -- that is a
   kernel or layout bug; send the log.
2. TIME: one training step (forward, backward, optimizer step) of the knee cell's net at the
   knee latent size, fused, and the roll path for reference (skip with --no-roll; it is slow).
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import models.ncsr_lpds as m                     # noqa: E402
from scripts.make_guided_recon_configs import NCSR_LPDS_PARAMS  # noqa: E402


def build(params, device, seed=0):
    torch.manual_seed(seed)
    return m.NCSRLPDSNet(**params).to(device)


def check(device):
    p = dict(NCSR_LPDS_PARAMS, K=3, M=24, Mh=8, spectral_init=False)
    g = torch.Generator(device="cpu").manual_seed(1)
    y = torch.randn(2, 1, 64, 72, dtype=torch.complex64, generator=g).to(device)
    w = torch.randn(2, 1, 64, 72, dtype=torch.complex64, generator=g).to(device)
    out = {}
    for fused in (False, True):
        m.FUSED = fused
        net = build(p, device)
        x, _ = net(y, guide=w)
        x.abs().pow(2).sum().backward()
        out[fused] = (x.detach(), {n: q.grad for n, q in net.named_parameters()
                                   if q.grad is not None})
    m.FUSED = "auto"
    (x0, g0), (x1, g1) = out[False], out[True]
    err = float((x0 - x1).abs().max() / x0.abs().max())
    gerr = max(float((g0[n] - g1[n]).abs().max() / (g0[n].abs().max() + 1e-12)) for n in g0)
    ok = err < 1e-4 and gerr < 1e-3
    print(f"[check] fused vs roll: output rel err {err:.2e}, worst grad rel err {gerr:.2e} "
          f"-> {'OK' if ok else 'MISMATCH'}")
    return ok


def time_step(fused, args, device):
    m.FUSED = fused
    p = dict(NCSR_LPDS_PARAMS, spectral_init=False)
    if args.K:
        p["K"] = args.K
    net = build(p, device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-5)
    y = torch.randn(args.batch, 1, args.H, args.W, dtype=torch.complex64, device=device)
    w = torch.randn_like(y)
    ts = []
    for i in range(args.warmup + args.iters):
        torch.cuda.synchronize()
        t = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        x, _ = net(y, guide=w)
        x.abs().mean().backward()
        opt.step()
        torch.cuda.synchronize()
        if i >= args.warmup:
            ts.append(time.perf_counter() - t)
    m.FUSED = "auto"
    mem = torch.cuda.max_memory_allocated() / 2 ** 30
    torch.cuda.reset_peak_memory_stats()
    s = sorted(ts)[len(ts) // 2]
    print(f"[time] {'fused' if fused else 'roll '}: {s:.3f} s / step (median of {len(ts)}), "
          f"peak {mem:.1f} GB, K={p['K']} batch={args.batch} image {args.H}x{args.W}")
    return s, mem


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--K", type=int, default=None, help="layers (default: the cell's)")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--H", type=int, default=640, help="image grid (knee: 640 x 368)")
    ap.add_argument("--W", type=int, default=368)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--no-roll", action="store_true", help="time the fused path only")
    ap.add_argument("--json")
    a = ap.parse_args()
    device = torch.device("cuda")
    print(f"[bench] {torch.cuda.get_device_name()} | triton: {m.cw.HAVE_TRITON}")
    if not check(device):
        sys.exit(1)
    res = {"fused": time_step(True, a, device)}
    if not a.no_roll:
        res["roll"] = time_step(False, a, device)
        print(f"[time] speedup {res['roll'][0] / res['fused'][0]:.1f}x")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({k: {"s_per_step": v[0], "peak_gb": v[1]} for k, v in res.items()}, f)


if __name__ == "__main__":
    main()
