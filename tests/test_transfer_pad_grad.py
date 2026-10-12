"""
`GridTransfer.ADJOINT_PAD`: hand-written backwards for the circular pad of the
restriction and the halo fold of the prolongation (operators/resample.py).

Run with `python -m tests.test_transfer_pad_grad`.

The two are transposes of each other, so each one's backward is the other's
forward. Autograd reached the same answer through the slice assignments the
pad and the fold are built from, at several full passes over the padded tensor
per assignment -- 53 ms of mg169v6's 197 ms backward on an L40S. Nothing about
the MODEL changes; what has to hold is:

1. VALUES are bit-identical with the switch on and off (the pad is the same
   `F.pad`; the fold is the same function, run without a graph);
2. GRADIENTS agree with autograd's, for `restrict` and `prolong`, the module
   functions and the `GridTransfer` methods, real and complex, and for a
   learned kernel -- to fp roundoff, since the fold's backward adds a corner
   straight onto its pixel rather than via an edge strip;
3. the two custom Functions pass `gradcheck` in float64;
4. a whole multigrid net's forward is unchanged and its gradients agree;
5. the switch removes the slice-assignment nodes from the graph, which is the
   cost it exists to remove.

Each gradient comparison has a NEGATIVE CONTROL: a backward that crops without
folding the halo back -- the natural wrong answer -- must be caught.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

import operators.resample as R
from models.mg_lpds import MGLPDSNet
from operators import FFT2D, Mask, Sense
from operators.resample import GridTransfer, prolong, restrict
from physics.mask import make_acc_mask

FAIL = []


def check(name, ok, detail=""):
    print(f"[{'ok ' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def rel(a, b):
    a, b = a.detach(), b.detach()
    return float((a - b).abs().max() / (b.abs().max() + 1e-30))


def switch(on):
    class _S:
        def __enter__(self):
            self.old = GridTransfer.ADJOINT_PAD
            GridTransfer.ADJOINT_PAD = on

        def __exit__(self, *a):
            GridTransfer.ADJOINT_PAD = self.old
    return _S()


def value_and_grad(fn, x, on):
    """`fn(x)` and d<fn(x), w>/dx for a fixed random w."""
    x = x.clone().requires_grad_(True)
    with switch(on):
        out = fn(x)
        g = torch.Generator().manual_seed(7)
        w = torch.randn(out.shape, generator=g, dtype=out.real.dtype if out.is_complex() else out.dtype)
        s = (torch.view_as_real(out) if out.is_complex() else out)
        s = (s * (w.unsqueeze(-1) if out.is_complex() else w)).sum()
        (grad,) = torch.autograd.grad(s, x)
    return out.detach(), grad


# ===========================================================================
def test_values_and_grads():
    torch.manual_seed(0)
    cases = []
    for filt in ("sinc", "box", "bilinear"):
        for cplx in (False, True):
            dt = torch.complex64 if cplx else torch.float32
            x = torch.randn(2, 3, 16, 24, dtype=dt)
            z = torch.randn(2, 3, 8, 12, dtype=dt)
            tag = f"{filt}, {'complex' if cplx else 'real'}"
            cases += [(f"restrict ({tag})", lambda t, f=filt: restrict(t, filter=f), x),
                      (f"prolong  ({tag})", lambda t, f=filt: prolong(t, filter=f), z)]
    for name, fn, x in cases:
        v1, g1 = value_and_grad(fn, x, True)
        v0, g0 = value_and_grad(fn, x, False)
        check(f"{name}: value bit-identical, gradient == autograd",
              rel(v1, v0) == 0.0 and rel(g1, g0) < 2e-5,
              f"value {rel(v1, v0):.0e}, grad rel {rel(g1, g0):.1e}")


def test_gridtransfer_and_learned_kernel():
    torch.manual_seed(1)
    for learn in (False, True):
        gt = GridTransfer(channels=4, learn=learn)
        x = torch.randn(1, 4, 16, 16)
        z = torch.randn(1, 4, 8, 8)
        for name, fn, t in (("restrict", gt.restrict, x), ("prolong", gt.prolong, z)):
            v1, g1 = value_and_grad(fn, t, True)
            v0, g0 = value_and_grad(fn, t, False)
            check(f"GridTransfer.{name} (learn={learn}): value and input gradient",
                  rel(v1, v0) == 0.0 and rel(g1, g0) < 2e-5,
                  f"grad rel {rel(g1, g0):.1e}")
        if learn:
            def wgrad(on):
                gt.zero_grad()
                with switch(on):
                    xx = x.clone().requires_grad_(True)     # so the pad is on the graph
                    (gt.prolong(gt.restrict(xx)) ** 2).sum().backward()
                return [p.grad.clone() for p in gt.parameters() if p.grad is not None]
            a, b = wgrad(True), wgrad(False)
            check("learned kernel: its gradient is unchanged",
                  len(a) == len(b) > 0 and all(rel(u, v) < 2e-5 for u, v in zip(a, b)),
                  f"{len(a)} tensors")


def test_gradcheck():
    torch.manual_seed(2)
    for a, b in ((3, 4), (4, 3), (2, 0)):
        x = torch.randn(1, 2, 7, 6, dtype=torch.float64, requires_grad=True)
        g = torch.randn(1, 2, 7 + a + b, 6 + a + b, dtype=torch.float64, requires_grad=True)
        ok1 = torch.autograd.gradcheck(lambda t: R._CircPad.apply(t, a, b), (x,))
        ok2 = torch.autograd.gradcheck(lambda t: R._CircFold.apply(t, a, b), (g,))
        check(f"gradcheck, pad ({a},{b}): _CircPad and _CircFold", ok1 and ok2)
        # and they are each other's transpose
        lhs = float((R._CircPad.apply(x, a, b) * g).sum())
        rhs = float((x * R._CircFold.apply(g, a, b)).sum())
        check(f"<pad x, g> == <x, fold g>  ({a},{b})", abs(lhs - rhs) < 1e-10,
              f"diff {abs(lhs - rhs):.1e}")


def test_negative_control():
    """A backward that crops WITHOUT folding the halo back must be caught."""
    torch.manual_seed(3)
    x = torch.randn(1, 2, 16, 16)
    fn = lambda t: restrict(t, filter="sinc")                       # noqa: E731
    _, good = value_and_grad(fn, x, False)
    orig = R._fold_once

    def crop_only(g, a, b):
        H, W = g.shape[-2] - a - b, g.shape[-1] - a - b
        return g[..., a:a + H, a:a + W].clone()

    R._fold_once = crop_only
    try:
        _, bad = value_and_grad(fn, x, True)
    finally:
        R._fold_once = orig
    check("negative control: dropping the halo from the backward is detected",
          rel(bad, good) > 1e-2, f"rel {rel(bad, good):.2e} (must be large)")


def _slice_nodes(t):
    """Count the slice-assignment nodes (CopySlices) behind a tensor."""
    seen, stack, n = set(), [t.grad_fn], 0
    while stack:
        fn = stack.pop()
        if fn is None or fn in seen:
            continue
        seen.add(fn)
        n += type(fn).__name__ == "CopySlices"
        stack.extend(f for f, _ in fn.next_functions)
    return n, len(seen)


def test_graph_is_smaller():
    x = torch.randn(1, 4, 16, 16, requires_grad=True)
    with switch(False):
        n0, t0 = _slice_nodes(prolong(restrict(x)))
    with switch(True):
        n1, t1 = _slice_nodes(prolong(restrict(x)))
    check("the switch removes the slice-assignment nodes from the graph",
          n0 > 0 and n1 == 0, f"CopySlices {n0} -> {n1}; graph nodes {t0} -> {t1}")


def test_network():
    H, W, C = 64, 48, 4
    torch.manual_seed(0)
    sm = torch.randn(1, C, H, W, dtype=torch.complex64)
    sm = sm / sm.abs().pow(2).sum(1, keepdim=True).sqrt()
    m = make_acc_mask((H, W), accel=4, acs_lines=8, dim=1, mode="uniform").float()
    E = Mask(m.reshape(1, 1, H, W)) @ FFT2D() @ Sense(sm)
    y = E.forward(torch.randn(1, 1, H, W, dtype=torch.complex64))

    def run(on):
        torch.manual_seed(0)
        net = MGLPDSNet(K=[2, [2, 2, 2]], M=8, C=1, P=3, s=2, lam0=1e-3, tau0=0.5,
                        theta0=0.0, alpha0=1.0, is_complex=True, degrees=1,
                        preproc="kspace", resize_noise=True, coarse_op="rediscretize")
        with switch(on):
            out = net(y, E=E, sigma=0.01)[0]
            out.abs().pow(2).mean().backward()
        return out.detach(), {n: p.grad.clone() for n, p in net.named_parameters()
                              if p.grad is not None}

    o1, g1 = run(True)
    o0, g0 = run(False)
    worst = max(rel(g1[n], g0[n]) for n in g0)
    check("MGLPDSNet: forward bit-identical", rel(o1, o0) == 0.0)
    check("MGLPDSNet: every parameter gradient agrees with autograd's",
          set(g1) == set(g0) and worst < 1e-4,
          f"{len(g0)} tensors, worst rel {worst:.1e}")


def main():
    for fn in (test_values_and_grads, test_gridtransfer_and_learned_kernel,
               test_gradcheck, test_negative_control, test_graph_is_smaller,
               test_network):
        print(f"\n--- {fn.__name__}")
        fn()
    print(f"\n{'FAILED: ' + ', '.join(FAIL) if FAIL else 'all checks passed'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
