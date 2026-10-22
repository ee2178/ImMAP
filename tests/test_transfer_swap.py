"""
Swapping the transfer FILTER of a built net (operators/resample.py::
set_transfer_filter / swap_transfer_filter) -- e.g. turning the sinc off.

Run with `python -m tests.test_transfer_swap`. CPU, a few seconds.

1. the swap is EXACTLY a net built with that filter: a sinc net swapped to
   "box" / "none" / "bilinear" gives bit-for-bit the output of a net
   constructed with that filter and loaded with the same weights -- so every
   transfer (restrict, prolong, coarse maps, coarse noise) follows the swap;
2. it changes the output, and on exit the output is bit-identical to before;
3. rediscretized and Galerkin coarse operators alike;
4. learned-transfer nets and nets without transfers are refused.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from operators import resample as rs                                 # noqa: E402
from tests.test_level_trace import problem                           # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(f"[{'ok ' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def net(op, filt=None, learn=False):
    from models import build_model
    p = dict(K=[2, [2, 2, 2]], M=8, C=1, P=3, s=2, degrees=1, lam0=1e-3, tau0=0.5,
             theta0=0.0, alpha0=0.3, is_complex=True, preproc="kspace",
             resize_noise=True, coarse_op=op)
    if filt:
        p["transfer_filter"] = filt
    if learn:
        p["learn_transfer"] = True
    torch.manual_seed(0)
    return build_model({"model": {"type": "MGLPDSNet", "params": p}}).eval()


def main():
    E, y = problem()
    for op in ("rediscretize", "galerkin"):
        a = net(op)
        check(f"{op}: built nets default to the sinc", rs.transfer_filter_of(a) == "sinc")
        with torch.no_grad():
            before = a(y, E=E, sigma=0.01)[0]
        for f in ("box", "none", "bilinear"):
            b = net(op, filt=f)
            b.load_state_dict(a.state_dict())
            with torch.no_grad():
                built = b(y, E=E, sigma=0.01)[0]
                with rs.swap_transfer_filter(a, f):
                    check(f"{op}/{f}: every transfer follows the swap",
                          rs.transfer_filter_of(a) == f
                          and all(m.transfer_filter == f for m in a.modules()
                                  if hasattr(m, "transfer_filter")))
                    swapped = a(y, E=E, sigma=0.01)[0]
                after = a(y, E=E, sigma=0.01)[0]
            check(f"{op}/{f}: swapped == built with {f!r} (same weights)",
                  torch.equal(swapped, built),
                  f"max |diff| {float((swapped - built).abs().max()):.1e}")
            rel = float((swapped - before).norm() / before.norm())
            check(f"{op}/{f}: and it changes the output", rel > 1e-4, f"relative change {rel:.2e}")
            check(f"{op}/{f}: restored exactly on exit",
                  torch.equal(before, after) and rs.transfer_filter_of(a) == "sinc")

    try:
        rs.set_transfer_filter(net("rediscretize", learn=True), "box")
        check("learned-transfer net refused", False)
    except ValueError:
        check("learned-transfer net refused", True)
    try:
        rs.set_transfer_filter(torch.nn.Conv2d(1, 1, 3), "box")
        check("net without transfers refused", False)
    except ValueError:
        check("net without transfers refused", True)

    print(f"\n{'FAILED: ' + ', '.join(FAIL) if FAIL else 'all checks passed'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
