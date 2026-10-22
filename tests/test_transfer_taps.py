"""
Implanting a longer sinc at inference (operators/resample.py::use_filter_length).

Run with `python -m tests.test_transfer_taps`. CPU, a few seconds.

1. inside the block the sinc has the requested taps, sums to one, and
   restriction / prolongation stay exact (scaled) adjoints;
2. on exit the table and the kernels are back to exactly what they were;
3. a TRAINED-style MGLPDSNet (rediscretized and Galerkin coarse operators) runs
   inside the block, gives a different output there, and the same output as
   before once the block ends -- nothing of the other length survives (cache);
4. invalid lengths are refused with a reason.
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


def adjoint_gap(H=48, W=40):
    """|<P u, v> - c <u, R v>| / scale, for random u (coarse) and v (fine)."""
    g = torch.Generator().manual_seed(0)
    u = torch.randn(1, 1, H // 2, W // 2, generator=g, dtype=torch.complex64)
    v = torch.randn(1, 1, H, W, generator=g, dtype=torch.complex64)
    c = rs.adjoint_scale("sinc")
    lhs = torch.vdot(rs.prolong(u, filter="sinc").flatten(), v.flatten())
    rhs = c * torch.vdot(u.flatten(), rs.restrict(v, filter="sinc").flatten())
    return float((lhs - rhs).abs() / lhs.abs())


def main():
    trained = rs.filter_length("sinc")
    k8 = rs.transfer_kernel("sinc").clone()
    check("as shipped: the sinc is 8 taps at factor 2", trained == 8 and k8.shape[-1] == 8)

    # 1 -----------------------------------------------------------------
    with rs.use_filter_length("sinc", 12) as taps:
        k12 = rs.transfer_kernel("sinc")
        check("inside: 12 taps, summing to one",
              taps == 12 and rs.filter_length("sinc") == 12 and k12.shape[-1] == 12
              and abs(float(k12.sum()) - 1.0) < 1e-6, f"shape {tuple(k12.shape)}")
        gap = adjoint_gap()
        check("inside: prolong == c . restrict^T still holds", gap < 1e-5, f"rel. gap {gap:.1e}")

    # 2 -----------------------------------------------------------------
    check("after: back to 8 taps, kernel bit-identical",
          rs.filter_length("sinc") == trained and torch.equal(rs.transfer_kernel("sinc"), k8))

    # 3 -----------------------------------------------------------------
    from models import build_model
    for op in ("rediscretize", "galerkin"):
        torch.manual_seed(0)
        net = build_model({"model": {"type": "MGLPDSNet", "params": dict(
            K=[2, [2, 2, 2]], M=8, C=1, P=3, s=2, degrees=1, lam0=1e-3, tau0=0.5,
            theta0=0.0, alpha0=0.3, is_complex=True, preproc="kspace",
            resize_noise=True, coarse_op=op)}}).eval()
        E, y = problem()
        with torch.no_grad():
            before = net(y, E=E, sigma=0.01)[0]
            with rs.use_filter_length("sinc", 16):
                implanted = net(y, E=E, sigma=0.01)[0]
            after = net(y, E=E, sigma=0.01)[0]
        rel = float((implanted - before).norm() / before.norm())
        check(f"{op}: a 16-tap sinc changes the output (same weights)",
              bool(torch.isfinite(implanted).all()) and rel > 1e-4, f"relative change {rel:.2e}")
        check(f"{op}: and after the block the output is bit-identical to before",
              torch.equal(before, after))

    # 4 -----------------------------------------------------------------
    for bad in (7, 9, 2 + 1):
        try:
            with rs.use_filter_length("sinc", bad):
                pass
            check(f"taps={bad} refused", False)
        except ValueError:
            check(f"taps={bad} refused", True)
    check("the table survives a refused call", rs.filter_length("sinc") == trained)

    print(f"\n{'FAILED: ' + ', '.join(FAIL) if FAIL else 'all checks passed'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
