"""
Sanity tests for the Cascade LPDS net (models/cascade_lpds.py).

Run with:  python -m tests.test_cascade_lpds
"""

import sys

import torch

from models import build_model
from models.base import set_weight
from models.cascade_lpds import CascadeLPDSLayer, CascadeLPDSNet
from models.ml_cdlnet import _init_size

torch.manual_seed(0)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"[{'ok ' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")


def _level_norm(lev, size=32):
    w = lev.analysis.weight
    x = torch.rand(1, lev.C, size, size, dtype=w.dtype)
    for _ in range(100):
        x = lev.synthesis(lev.analysis(x))
        x = x / x.norm()
    return (lev.analysis(x).norm() / x.norm()).item()


def test_shape_and_init():
    lay = CascadeLPDSLayer(lam0=1e-3, tau0=0.5, degrees=1)
    shapes = [tuple(lev.analysis.weight.shape) for lev in lay.levels]
    check("channels 1 -> 16 -> 64 -> 256, dense 7x7",
          shapes == [(16, 1, 7, 7), (64, 16, 7, 7), (256, 64, 7, 7)], str(shapes))
    check("stride 2 at every level", lay.strides == (2, 2, 2))
    check("dense convs (groups = 1)",
          all(lev.analysis.groups == 1 for lev in lay.levels))
    check("B_l = A_l^H at init",
          all(torch.allclose(lev.synthesis.weight, lev.analysis.weight.conj())
              for lev in lay.levels))

    # on the grid each level was normalised on: zero padding makes the norm
    # grid-dependent, and a smaller grid reads low
    norms = [_level_norm(lev, _init_size(l)) for l, lev in enumerate(lay.levels, 1)]
    check("each level spectrally normalised on its own: ||A_l|| = 1",
          all(abs(n - 1) < 1e-2 for n in norms), ", ".join(f"{n:.4f}" for n in norms))
    n2 = lay.op_norm2()
    check("||K|| <= prod ||A_l|| = 1 (composed operator NOT renormalised)",
          n2 <= 1.0 + 1e-3, f"||K||^2 = {n2:.4f}")
    check("standard tau0 = 0.5 inside the Condat-Vu bound",
          0.5 * (0.5 + n2) <= 1.0 + 1e-6, f"bound {lay.step_bound():.3f}")

    t = lay.prox.prox.tau.weight
    check("thresholds at lam0 on all 256 deepest channels",
          t.shape[-1] == 256 and bool((t[0] == 1e-3).all()))

    before = [w.clone() for lev in lay.levels
              for w in (lev.analysis.weight, lev.synthesis.weight)]
    for lev in lay.levels:
        lev.project_()
    after = [w for lev in lay.levels for w in (lev.analysis.weight, lev.synthesis.weight)]
    check("project() is a no-op at init",
          all(torch.allclose(a, b, atol=1e-6) for a, b in zip(before, after)))


def test_copied_across_K():
    net = CascadeLPDSNet(K=4, degrees=1, preproc="identity")
    ref = net.layer(0).levels
    same = all(torch.equal(a.analysis.weight, b.analysis.weight)
               and torch.equal(a.synthesis.weight, b.synthesis.weight)
               for lay in net.net.layers[1:] for a, b in zip(ref, lay.levels))
    check("one prototype copied into all K layers", same)


def _complexify(lay):
    for lev in lay.levels:
        w = lev.analysis.weight + 0.1 * torch.randn_like(lev.analysis.weight)
        set_weight(lev.analysis, w)
        set_weight(lev.synthesis, w.conj())


def test_operator():
    lay = CascadeLPDSLayer(spectral_init=False)
    _complexify(lay)
    x = torch.randn(2, 1, 32, 32, dtype=torch.complex64)
    z = torch.randn(2, 256, 4, 4, dtype=torch.complex64)

    ref_a = x
    for lev in lay.levels:
        ref_a = lev.analysis(ref_a)
    ref_h = z
    for lev in reversed(lay.levels):
        ref_h = lev.synthesis(ref_h)
    ea = ((lay.analyse(x) - ref_a).abs().max() / ref_a.abs().max()).item()
    eh = ((lay.adjoint(z) - ref_h).abs().max() / ref_h.abs().max()).item()
    check("interleaved K == Gauss-trick modules (complex weights)", ea < 1e-5, f"{ea:.1e}")
    check("interleaved K^H == Gauss-trick modules (complex weights)", eh < 1e-5, f"{eh:.1e}")

    lay.double()
    x, z = x.to(torch.complex128), z.to(torch.complex128)
    lhs = (lay.analyse(x).conj() * z).sum()
    rhs = (x.conj() * lay.adjoint(z)).sum()
    err = (abs(lhs - rhs) / abs(lhs)).item()
    check("<Kx, z> = <x, K^H z> with complex weights", err < 1e-10, f"rel err {err:.1e}")


def test_net_forward_backward():
    net = build_model({"model": {"type": "CascadeLPDSNet", "params": dict(
        K=4, M=16, L=3, P=7, s=2, widen=4, degrees=1, preproc="identity")}})
    check("build_model registers CascadeLPDSNet", isinstance(net, CascadeLPDSNet))
    check("pad_stride = 8", net.pad_stride == 8)
    y = torch.randn(2, 1, 32, 32, dtype=torch.complex64)
    sigma = torch.full((2, 1, 1, 1), 0.05)
    x_hat, (x, z) = net(y, sigma=sigma)
    check("output shape and finite",
          x_hat.shape == y.shape and bool(torch.isfinite(x_hat.abs()).all()))
    check("dual lives on the depth-3 grid", tuple(z.shape) == (2, 256, 4, 4))
    x_hat.abs().pow(2).sum().backward()
    g = [lev.analysis.conv_imag.weight.grad for lev in net.layer(1).levels]
    check("every level's filters get gradient",
          all(t is not None and t.abs().sum() > 0 for t in g))


if __name__ == "__main__":
    test_shape_and_init()
    test_copied_across_K()
    test_operator()
    test_net_forward_backward()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    sys.exit(1 if FAIL else 0)
