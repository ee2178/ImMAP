#!/usr/bin/env python3
"""
One SENSE Gram application per multigrid level, timed: the fine Gram, the
Galerkin coarse Gram, the rediscretized coarse Gram, and a Galerkin Gram with a
hand-written backward.

    python scripts/time_coarse_gram.py --size 640 320 --coils 20
    python scripts/time_coarse_gram.py --size 768 396 --coils 16 --levels 2

What it answers. scripts/time_net.py --ab-coarse gives the whole-network price of
the coarse operator; this splits it per application, so the two can be read
together (applications per forward x cost per application):

    galerkin       E_c = E . P          the coarse Gram P^H (E^H E) P runs the
                                        FINE multicoil FFTs, plus a prolong and
                                        a restrict, at every coarse level
    rediscretize   E_c = M_c F_c S_c    the same Gram posed on the coarse grid
                                        (operators/coarse.py)
    galerkin+bwd   the Galerkin Gram through `_HermitianGram`, an autograd
                   Function whose backward is ONE more Gram application on the
                   gradient (the Gram is Hermitian: its VJP is itself) and which
                   saves nothing from the forward

The last arm is the "backprop through Galerkin more efficiently" option. Autograd
already keeps little for this chain -- the maps and mask carry no gradient, the
FFTs are linear -- so what it can save is mostly bookkeeping and the coil-sized
temporaries some ops retain; the FFT work of the backward is the same either
way. The table says how much that is in practice. Its gradient is checked
against autograd's.

Columns (ms, median over --reps; peak = extra memory over the inputs, MB):
    fwd        no autograd -- inference
    fwd+bwd    x requires grad; <Gram x, w> backward -- one training application
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)


class _HermitianGram(torch.autograd.Function):
    """y = G x for a Hermitian G (a Gram E^H E). Backward: G g. Saves nothing."""

    @staticmethod
    def forward(ctx, x, op):
        ctx.op = op
        return op.gram(x)

    @staticmethod
    def backward(ctx, g):
        with torch.no_grad():
            return ctx.op.gram(g), None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, nargs=2, default=[640, 320], metavar=("H", "W"))
    ap.add_argument("--coils", type=int, default=20)
    ap.add_argument("--R", type=int, default=8)
    ap.add_argument("--levels", type=int, default=2, help="coarse levels below the fine")
    ap.add_argument("--filter", default=None,
                    help="transfer filter (operators/resample.py); default = the module's")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    import time_net as tn
    tn._import_repo()
    from operators.coarse import coarsen

    dev = torch.device(a.device)
    cuda = dev.type == "cuda"
    mri = dict(center_frac=0.04, adjust_accel=True)
    y, E, _ = tn.build_problem(tuple(a.size), a.coils, mri, a.R, dev)

    def sync():
        if cuda:
            torch.cuda.synchronize()

    def timed(fn):
        for _ in range(a.warmup):
            fn()
        ts = []
        for _ in range(a.reps):
            sync()
            t0 = tn.time.perf_counter()
            fn()
            sync()
            ts.append(1e3 * (tn.time.perf_counter() - t0))
        return statistics.median(ts)

    def peak(fn):
        if not cuda:
            return float("nan")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        fn()
        torch.cuda.synchronize()
        return (torch.cuda.max_memory_allocated() - base) / 2 ** 20

    def arm(apply, x0, w):
        def fwd():
            with torch.no_grad():
                apply(x0)

        def fwdbwd():
            x = x0.detach().requires_grad_(True)
            (apply(x) * w.conj()).real.sum().backward()
            return x.grad
        return dict(fwd=timed(fwd), fwdbwd=timed(fwdbwd), peak=peak(fwdbwd)), fwdbwd

    g = torch.Generator(device="cpu").manual_seed(0)
    ops = {"galerkin": E, "rediscretize": E}
    rows = []
    for lev in range(a.levels + 1):
        H, W = a.size[0] >> lev, a.size[1] >> lev
        x0 = torch.randn(1, 1, H, W, dtype=torch.complex64, generator=g).to(dev)
        w = torch.randn(1, 1, H, W, dtype=torch.complex64, generator=g).to(dev)
        if lev == 0:
            r, _ = arm(E.gram, x0, w)
            rows.append((lev, f"{H}x{W}", "fine", r, None))
            continue
        ops = {k: coarsen(v, k, filter=a.filter) for k, v in ops.items()}
        Eg, Er = ops["galerkin"], ops["rediscretize"]
        rg, grad_auto = arm(Eg.gram, x0, w)
        rr, _ = arm(Er.gram, x0, w)
        rc, grad_custom = arm(lambda x: _HermitianGram.apply(x, Eg), x0, w)
        ga, gc = grad_auto(), grad_custom()
        err = float((ga - gc).abs().norm() / ga.abs().norm().clamp_min(1e-30))
        rows += [(lev, f"{H}x{W}", "galerkin", rg, None),
                 (lev, f"{H}x{W}", "rediscretize", rr, None),
                 (lev, f"{H}x{W}", "galerkin+bwd", rc, err)]

    name = torch.cuda.get_device_name(0) if cuda else "CPU"
    print(f"{name} | torch {torch.__version__} | {a.size[0]}x{a.size[1]}, {a.coils} coils, "
          f"R={a.R} | {a.reps} reps after {a.warmup} warm-up\n")
    print(f"{'level':<7}{'grid':<11}{'operator':<15}{'fwd ms':>9}{'fwd+bwd ms':>12}"
          f"{'peak MB':>10}   note")
    for lev, grid, op, r, err in rows:
        note = "" if err is None else f"grad vs autograd: rel. err {err:.1e}"
        print(f"{lev:<7}{grid:<11}{op:<15}{r['fwd']:>9.3f}{r['fwdbwd']:>12.3f}"
              f"{r['peak']:>10.1f}   {note}")
    print("\nper coarse level: x = galerkin / rediscretize, and the custom backward's saving")
    for lev in range(1, a.levels + 1):
        d = {op: r for l_, _, op, r, _ in rows if l_ == lev}
        g_, r_, c_ = d["galerkin"], d["rediscretize"], d["galerkin+bwd"]
        print(f"  level {lev}: rediscretize fwd {g_['fwd'] / r_['fwd']:.1f}x, fwd+bwd "
              f"{g_['fwdbwd'] / r_['fwdbwd']:.1f}x faster than galerkin;  custom backward: "
              f"fwd+bwd {g_['fwdbwd']:.2f} -> {c_['fwdbwd']:.2f} ms, peak "
              f"{g_['peak']:.0f} -> {c_['peak']:.0f} MB")


if __name__ == "__main__":
    main()
