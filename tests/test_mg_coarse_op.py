"""
`MGLPDSNet(coarse_op=...)`: Galerkin vs rediscretized coarse levels.

Run with `python -m tests.test_mg_coarse_op`.

The operator itself is checked against Galerkin in tests/test_coarse_operator.py.
This pins the WIRING:

1. the option changes no parameter -- same seed, same state_dict, so either
   mode loads the other's checkpoint;
2. "rediscretize" really runs the coarse levels' physics on the coarse grids:
   the FFTs recorded during a forward pass land on H x W, H/2 x W/2 and
   H/4 x W/4, where "galerkin" runs every one at H x W;
3. an operator the rediscretization cannot express (an already-Galerkin
   `E @ Resample`, a measured grid that has become odd) falls back to Galerkin
   rather than failing -- and the image-domain embedding `E @ Truncate` is NOT
   one of those: its coarse levels run on the measured grid's half and quarter;
4. it trains: a backward pass reaches every parameter the Galerkin one does;
5. the two modes' outputs, from identical weights, are close -- reported, and
   bounded loosely (they are different coarse problems, not the same one).

Small CPU problem, no LAPACK needed.
"""

import math
from collections import Counter

import torch
import torch.fft as tfft

from models.mg_lpds import MGLPDSNet
from operators import FFT2D, Mask, Sense
from operators.coarse import coarsen, rediscretize
from operators.resample import Resample
from operators.truncate import embed_operator
from physics.mask import make_acc_mask

FAIL = []
H, W, C = 64, 48, 4
NET = dict(K=[1, [2, 2, 2]], M=8, P=3, s=2, lam0=1e-3, tau0=0.5, theta0=0.0,
           alpha0=1.0, is_complex=True, preproc="kspace", resize_noise=True)


def check(name, ok, detail=""):
    print(f"[{'ok ' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def problem(h=H, w=W, seed=0):
    g = torch.Generator().manual_seed(seed)
    yy = torch.arange(h)[:, None] - h / 2
    xx = torch.arange(w)[None, :] - w / 2
    sm = []
    for c in range(C):
        a = 2 * math.pi * c / C
        mag = torch.exp(-((yy - 0.4 * h * math.sin(a)) ** 2 + (xx - 0.4 * w * math.cos(a)) ** 2)
                        / (2 * (0.4 * h) ** 2))
        sm.append(mag * torch.exp(1j * (2 * math.pi * c / C + 0.02 * (yy + xx))))
    sm = torch.stack(sm)
    sm = (sm / sm.abs().pow(2).sum(0, keepdim=True).sqrt()).to(torch.complex64)[None]
    x = torch.complex(torch.randn(1, 1, h, w, generator=g), torch.randn(1, 1, h, w, generator=g))
    m = make_acc_mask((h, w), accel=4, acs_lines=8, dim=1, mode="uniform").float()
    E = Mask(m) @ FFT2D() @ Sense(sm)
    y = E.forward(x)
    return y, E


def build(mode, seed=0):
    torch.manual_seed(seed)
    return MGLPDSNet(coarse_op=mode, **NET)


class FFTSizes:
    """Record the grid of every 2-D FFT/IFFT during a block."""

    def __enter__(self):
        self.sizes = Counter()
        self._f, self._i = tfft.fftn, tfft.ifftn

        def wrap(fn):
            def inner(x, *a, **k):
                self.sizes[tuple(x.shape[-2:])] += 1
                return fn(x, *a, **k)
            return inner

        tfft.fftn, tfft.ifftn = wrap(self._f), wrap(self._i)
        return self

    def __exit__(self, *exc):
        tfft.fftn, tfft.ifftn = self._f, self._i


def test_parameters_identical():
    a, b = build("galerkin"), build("rediscretize")
    sa, sb = a.state_dict(), b.state_dict()
    same = sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
    check("coarse_op changes no parameter (same seed -> same state_dict)", same,
          f"{len(sa)} tensors")
    try:
        MGLPDSNet(coarse_op="nope", **NET)
        check("an unknown coarse_op is rejected", False, "no error")
    except ValueError:
        check("an unknown coarse_op is rejected", True)


def test_coarse_grids():
    y, E = problem()
    sig = torch.full((1, 1, 1, 1), 0.01)
    grids = {}
    for mode in ("galerkin", "rediscretize"):
        net = build(mode).eval()
        with torch.no_grad(), FFTSizes() as rec:
            net(y, E=E, sigma=sig)
        grids[mode] = dict(rec.sizes)
        print(f"       {mode:>12}: FFT grids {dict(sorted(rec.sizes.items(), reverse=True))}")
    g, r = grids["galerkin"], grids["rediscretize"]
    check("galerkin: every FFT runs on the fine grid", set(g) == {(H, W)}, f"{g}")
    check("rediscretize: FFTs run on all three grids",
          {(H, W), (H // 2, W // 2), (H // 4, W // 4)} <= set(r), f"{r}")
    check("rediscretize: fewer fine-grid FFTs than galerkin",
          r.get((H, W), 0) < g.get((H, W), 0),
          f"{r.get((H, W), 0)} vs {g.get((H, W), 0)}")


def test_fallback():
    y, E = problem()
    E_c = coarsen(E, "rediscretize")
    ok = (type(E_c.ops[-1]) is Sense and tuple(E_c.ops[-1].smaps.shape[-2:]) == (H // 2, W // 2))
    check("plain SENSE is rediscretized to the half grid", ok)
    E_g = coarsen(E, "galerkin")
    check("galerkin mode is unchanged: E @ Resample", isinstance(E_g.ops[-1], Resample))
    check("an already-Galerkin operator (E @ Resample) has no rediscretized form",
          rediscretize(E_g) is None)
    E_gc = coarsen(E_g, "rediscretize")
    check("...and coarsen falls back to Galerkin for it",
          isinstance(E_gc.ops[-1], Resample), f"{[type(o).__name__ for o in E_gc.ops]}")
    _, E_odd = problem(H, W - 2)                           # 46 columns: 23 is odd
    check("a measured grid that is odd one level down stops there, not before",
          rediscretize(E_odd) is not None and rediscretize(rediscretize(E_odd)) is None)


def test_embedded_grids():
    """E @ Truncate (a measured size that is not a multiple of the stride) is
    rediscretized too: the coarse levels' FFTs run at half and quarter of the
    MEASURED grid, 44 -> 22 -> 11 columns inside the 48 -> 24 -> 12 image."""
    h, w = H, W - 4                                        # 64 x 44 -> 64 x 48
    y, E = problem(h, w)
    E_t, T = embed_operator(E, (h, w), 8)
    check("the embedding adds a Truncate", not T.is_identity and len(E_t.ops) == 4, repr(T))
    E_c = coarsen(E_t, "rediscretize")
    names = [type(o).__name__ for o in E_c.ops]
    check("E @ Truncate coarsens to Mask @ FFT2D @ Sense @ Truncate",
          names == ["Mask", "FFT2D", "Sense", "Truncate"]
          and tuple(E_c.ops[2].smaps.shape[-2:]) == (h // 2, w // 2), f"{names} {E_c.ops[-1]!r}")
    sig = torch.full((1, 1, 1, 1), 0.01)

    # the switch that restores the pre-2026-10-07 behaviour for evaluation
    import operators.coarse as coarse_mod
    coarse_mod.REDISCRETIZE_EMBEDDED = False
    try:
        check("REDISCRETIZE_EMBEDDED=False: an embedded operator falls back to Galerkin",
              rediscretize(E_t) is None
              and isinstance(coarsen(E_t, "rediscretize").ops[-1], Resample)
              and rediscretize(E) is not None)
    finally:
        coarse_mod.REDISCRETIZE_EMBEDDED = True

    grids, out = {}, {}
    for mode in ("galerkin", "rediscretize"):
        net = build(mode).eval()
        with torch.no_grad(), FFTSizes() as rec:
            out[mode] = net(y, E=E_t, sigma=sig)[0]
        grids[mode] = dict(rec.sizes)
        print(f"       {mode:>12}: FFT grids {dict(sorted(rec.sizes.items(), reverse=True))}")
    g, r = grids["galerkin"], grids["rediscretize"]
    check("embedded, galerkin: every FFT runs on the measured grid", set(g) == {(h, w)}, f"{g}")
    check("embedded, rediscretize: FFTs run on the measured grid, its half and its quarter",
          set(r) == {(h, w), (h // 2, w // 2), (h // 4, w // 4)}, f"{r}")
    d = float((out["rediscretize"] - out["galerkin"]).norm() / out["galerkin"].norm())
    check("embedded: outputs of the two modes are close (untrained net, identical weights)",
          bool(torch.isfinite(out["rediscretize"]).all()) and d < 0.25, f"rel diff {d:.3e}")


def test_backward_and_agreement():
    y, E = problem()
    sig = torch.full((1, 1, 1, 1), 0.01)
    out, grads = {}, {}
    for mode in ("galerkin", "rediscretize"):
        net = build(mode)
        xh, _ = net(y, E=E, sigma=sig)
        xh.abs().pow(2).mean().backward()
        out[mode] = xh.detach()
        grads[mode] = {n for n, p in net.named_parameters()
                       if p.grad is not None and bool(torch.isfinite(p.grad).all())}
    check("rediscretize: backward reaches the same parameters as galerkin",
          grads["rediscretize"] == grads["galerkin"],
          f"{len(grads['rediscretize'])} vs {len(grads['galerkin'])}")
    check("rediscretize: output is finite", bool(torch.isfinite(out["rediscretize"]).all()))
    d = float((out["rediscretize"] - out["galerkin"]).norm() / out["galerkin"].norm())
    check("identical weights: outputs of the two modes are close (untrained net)",
          d < 0.25, f"rel diff {d:.3e}")


def main():
    for fn in (test_parameters_identical, test_coarse_grids, test_fallback,
               test_embedded_grids,
               test_backward_and_agreement):
        print(f"\n--- {fn.__name__}")
        fn()
    print(f"\n{'FAILED: ' + ', '.join(FAIL) if FAIL else 'all checks passed'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
