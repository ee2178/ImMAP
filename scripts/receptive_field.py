#!/usr/bin/env python3
"""
Receptive field of the LEARNED (prior) path: flat LPDS vs the MGLPDS V-cycle.

Run with `python scripts/receptive_field.py` (CPU is fine; see --help).

What is measured
----------------
The output pixel at the canvas centre is differentiated w.r.t. every input
pixel, with E = Identity, so the data-consistency Gram contributes nothing
spatial. With the real SENSE operator the Gram is GLOBAL (FFTs, coil maps):
every iteration couples every pixel to its aliases, so the receptive field of
the full unroll is the whole image after one layer, for both architectures.
What differs between them -- and what kernel size P controls -- is how far the
learned prior reaches, which is this.

The operating point is a tiny input (1e-6), where every dual clip
(prox_{g*} = projection onto the lam-ball) passes its input through. That is the
LINEAR regime: the receptive field the architecture CAN express, with no path
switched off. A trained net with saturated duals uses a subset of it.

Two numbers:

  theoretical   radius of the nonzero support of the gradient: float64 and an
                EXACT nonzero test (far paths multiply dozens of small taps and
                vanish below any float32 threshold), on a tall canvas so the
                support fits -- the support along one axis does not depend on
                the canvas width. Mostly astronomically small at its edge.
  effective     radii holding 50 / 90 / 99 % of the gradient ENERGY on a square
                canvas (Luo et al. 2016): random-walk-like, it grows with the
                SQUARE ROOT of depth, so it is the one that says what the prior
                actually leans on

M is reduced (default 8): the support depends only on P, stride, the K
structure and the transfer filter, and the energy profile depends on M only
weakly. Weights are the init (spectral-normalised), so the effective radii
describe the architecture at initialisation; a trained net's are measured the
same way from its checkpoint.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mg_lpds import MGLPDSNet  # noqa: E402


def generator():
    spec = importlib.util.spec_from_file_location(
        "_gen", os.path.join(os.path.dirname(__file__), "make_mg_recon_configs.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build(K, P, M, seed=0):
    g = generator()
    p = dict(g.LPDS_COMMON, K=K, P=P, M=M, preproc="identity", resize_noise=False)
    torch.manual_seed(seed)
    return MGLPDSNet(**p).eval()


def center_gradient(net, H, W, seed=0, dtype=torch.float32):
    """|d x_hat[centre] / d y| over the canvas, in the linear regime."""
    g = torch.Generator().manual_seed(seed)
    y = 1e-6 * torch.complex(torch.randn(1, 1, H, W, generator=g, dtype=dtype),
                             torch.randn(1, 1, H, W, generator=g, dtype=dtype))
    y = y.requires_grad_(True)
    sigma = torch.full((1, 1, 1, 1), 0.01, dtype=dtype)
    x_hat, _ = net(y, sigma=sigma)
    cy, cx = H // 2, W // 2
    grad = torch.autograd.grad(x_hat[0, 0, cy, cx].real, y)[0]
    return grad[0, 0].abs().double(), (cy, cx)


def theoretical_radius(g, centre, axis):
    """Largest |offset| along `axis` (0 = rows) with an EXACTLY nonzero gradient."""
    prof = g.amax(dim=1 - axis)
    nz = torch.nonzero(prof > 0).flatten()
    c = centre[axis]
    return int(max(c - nz.min().item(), nz.max().item() - c))


def effective_radii(g, centre, qs=(0.5, 0.9, 0.99)):
    H, W = g.shape
    yy = torch.arange(H, dtype=torch.float64)[:, None] - centre[0]
    xx = torch.arange(W, dtype=torch.float64)[None, :] - centre[1]
    r = torch.sqrt(yy ** 2 + xx ** 2).flatten()
    e = (g ** 2).flatten()
    order = torch.argsort(r)
    cum = torch.cumsum(e[order], 0) / e.sum()
    out = [float(r[order][torch.searchsorted(cum, torch.tensor(q, dtype=cum.dtype))])
           for q in qs]
    # energy in the outer 5% band of the canvas: if this is not ~0, the canvas
    # truncated the field and the radii are lower bounds
    edge = min(centre[0], centre[1], H - 1 - centre[0], W - 1 - centre[1])
    border = float(e[r > 0.95 * edge].sum() / e.sum())
    return out, border


def analytic_radius(K, P, s, filt_len):
    """Support radius in fine pixels, counted from the architecture.

    One LPDS sweep at grid spacing h moves information x -> z -> x through an
    analysis conv and a synthesis conv, each reaching (P-1)/2 pixels of the
    IMAGE grid at that level: (P-1) * h per sweep (the x radius after K flat
    sweeps is exactly K (P-1)). A grid transfer (restrict or prolong) adds its
    filter's half-length on the finer grid. An upper bound for the V-cycle: it
    assumes the longest chain through every level and the correction path; the
    float64 measurement says how much of it is realised.
    """
    if isinstance(K, int):
        return K * (P - 1)
    k_out, iters = K
    per_cycle, h = 0, 1
    for lvl, n in enumerate(iters):
        per_cycle += n * (P - 1) * h
        if lvl < len(iters) - 1:
            per_cycle += 2 * (filt_len // 2) * h   # restrict down + prolong up
        h *= 2
    return k_out * per_cycle


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--P", type=int, nargs="+", default=[3, 4, 5, 7])
    ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--lpds-K", type=int, default=None,
                    help="flat LPDS depth (default: the generator's LPDS_BASELINE_K)")
    ap.add_argument("--mg-K", type=str, default=None,
                    help='V-cycle K as JSON, e.g. "[6,[4,4,6]]" (default: LPDS_VCYCLE_K)')
    ap.add_argument("--erf-size", type=int, default=512,
                    help="square canvas for the effective radii (multiple of 8)")
    ap.add_argument("--support-size", type=int, default=3072,
                    help="tall canvas height for the theoretical support")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    g = generator()
    s = g.LPDS_COMMON["s"]
    K_lpds = args.lpds_K or g.LPDS_BASELINE_K
    K_mg = json.loads(args.mg_K) if args.mg_K else g.LPDS_VCYCLE_K
    from operators.resample import filter_length
    flen = filter_length()

    print(f"flat LPDS K={K_lpds}  vs  MGLPDS K={K_mg}   (s={s}, M={args.M} for the "
          f"measurement, transfer filter {flen} taps)")
    print("linear regime, E = Identity: the reach of the learned prior only\n")
    head = (f"  {'arch':<8}{'P':>3} | {'support radius':>15} {'(analytic)':>11} |"
            f" {'ERF r50':>8}{'r90':>7}{'r99':>7} {'edge E':>8} | {'conv MACs':>10}")
    print(head)
    print("  " + "-" * (len(head) - 2))
    rows = []
    for P in args.P:
        for name, K in (("LPDS", K_lpds), ("MGLPDS", K_mg)):
            net = build(K, P, args.M)
            # theoretical support: tall canvas, narrow width
            gs, cs = center_gradient(net.double(), args.support_size, 64,
                                     dtype=torch.float64)
            net = net.float()
            sup = theoretical_radius(gs, cs, axis=0)
            truncated = sup >= cs[0] - 1
            # effective field: square canvas
            ge, ce = center_gradient(net, args.erf_size, args.erf_size)
            (r50, r90, r99), border = effective_radii(ge, ce)
            # conv work per fine pixel, relative: sum_l n_l * P^2 / 4^l (2 convs/sweep)
            if isinstance(K, int):
                work = K * P * P
            else:
                work = K[0] * sum(n * P * P / 4 ** l for l, n in enumerate(K[1]))
            ana = analytic_radius(K, P, s, flen)
            rows.append(dict(arch=name, P=P, K=K, support=sup, truncated=truncated,
                             analytic=ana, r50=r50, r90=r90, r99=r99,
                             edge_energy=border, conv_work=work))
            print(f"  {name:<8}{P:>3} | {('>=' if truncated else '') + str(sup):>15}"
                  f" {ana:>11} | {r50:>8.1f}{r90:>7.1f}{r99:>7.1f} {border:>8.1e} |"
                  f" {work:>10.0f}")
    print("\n  support radius: fine pixels from the centre to the farthest input "
          "that reaches it")
    print("  ERF rq: radius holding q of the gradient energy; edge E: energy in the "
          "canvas's outer 5% (should be ~0)")
    print("  conv MACs: relative conv work per fine pixel, sum over sweeps of P^2/4^level")
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
