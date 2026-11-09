# -*- coding: utf-8 -*-
"""The materialised-weight window adjacency (models/circulant_window.py).

CPU: the torch reference against the gather backend -- weights, log-sum-exp convention,
apply and transposed apply -- and the guided prox with `triton_materialize` against the gather
prox (values and every gradient). CUDA + triton: each kernel against the reference.
"""
import sys
import types

import pytest
import torch

sys.modules.setdefault("nibabel", types.ModuleType("nibabel"))

import models.circulant_window as cw                               # noqa: E402
import models.guided_prox as gp                                    # noqa: E402
from models.circulant_attention import _abs2                       # noqa: E402
from models.circulant_similarity import circulant_similarity_window  # noqa: E402
from models.circulant_triton import HAVE_TRITON, _img_to_seq       # noqa: E402
from models.guided_prox import GuidedGroupThreshold                # noqa: E402

D = torch.float64


def cplx(*shape, dtype=D, gen=None):
    return torch.complex(torch.randn(*shape, dtype=dtype, generator=gen),
                         torch.randn(*shape, dtype=dtype, generator=gen))


@pytest.mark.parametrize("sim", ["pidistance", "pidot"])
@pytest.mark.parametrize("complex_qk", [True, False])
@pytest.mark.parametrize("heads", [1, 2])
@pytest.mark.parametrize("hw_win", [((9, 7), 5), ((6, 6), 3)])
def test_reference_is_the_gather_adjacency(sim, complex_qk, heads, hw_win):
    (H, W), win = hw_win
    torch.manual_seed(0)
    B, C = 2, 4 * heads
    mk = (lambda: cplx(B, C, H, W)) if complex_qk else (lambda: torch.randn(B, C, H, W, dtype=D))
    q, k = mk(), mk()
    adj = cw.WindowAdjacency(q, k, win, sim=sim, heads=heads)

    qh = q.reshape(B * heads, C // heads, H, W)
    kh = k.reshape(B * heads, C // heads, H, W)
    s, col, crow = circulant_similarity_window(sim, qh, kh, win)
    from models.circulant_attention import Circulant
    ref = Circulant(torch.softmax(s, -1), col, crow, (H, W), win)

    x = torch.randn(B, 3 * heads, H, W, dtype=D)
    xh = x.reshape(B * heads, 3, H, W)
    for t in (False, True):
        want = ref.apply(xh, transpose=t).reshape(B, 3 * heads, H, W)
        assert torch.allclose(adj.apply(x, transpose=t), want, rtol=1e-10, atol=1e-12), t

    # the log-sum-exp in the triton convention: gather's minus its per-query -1/2||q||^2
    qsq = 0.5 * _abs2(qh).sum(1).flatten(1) if sim == "pidistance" else 0.0
    assert torch.allclose(adj._lse, torch.logsumexp(s, -1) + qsq, rtol=1e-10, atol=1e-10)


def test_reference_gradients_through_scores_and_apply():
    """gradcheck the reference (it is what the kernels are tested against)."""
    torch.manual_seed(0)
    H, W, win = 5, 6, 3
    q, k = cplx(1, 2, H, W).requires_grad_(), cplx(1, 2, H, W).requires_grad_()
    x = torch.randn(1, 2, H, W, dtype=D, requires_grad=True)

    def f(q, k, x):
        a = cw.WindowAdjacency(q, k, win)
        return a.apply(x), a.apply(x, transpose=True), a._lse

    assert torch.autograd.gradcheck(f, (q, k, x))


@pytest.mark.parametrize("n_guides", [1, 2])
@pytest.mark.parametrize("heads", [1, 2])
def test_materialized_guided_prox_equals_the_gather_prox(n_guides, heads):
    torch.manual_seed(0)
    M, n = 8, 11
    kw = dict(Mh=4, nheads=heads, window=1, guide_window=5, joint_softmax=True,
              sim_fun="pidistance", attn_backend="gather", dK=1)
    gather = GuidedGroupThreshold(M, **kw).double()
    fused = GuidedGroupThreshold(M, triton_materialize=True, **kw).double()
    fused.load_state_dict(gather.state_dict())
    fused.attn_backend, fused.joint_fused = "triton", True     # constructors only: CPU box
    gp._FUSED_JOINT_CHECKED.add(("triton", "pidistance", heads, 1, 5, True, True, "pidistance"))

    z = cplx(2, M, n, n).requires_grad_()
    gs = [cplx(2, M, n, n) for _ in range(n_guides)]
    outs, grads = [], []
    for p in (gather, fused):
        p.zero_grad()
        zz = z.detach().clone().requires_grad_()
        out, cache = p(zz, gs, None, {}, dual=True)
        assert isinstance(cache["Omega"][0], cw.WindowAdjacency) == (p is fused)
        (out.abs() ** 2).sum().backward()
        outs.append(out.detach())
        grads.append([zz.grad] + [t.grad for t in p.parameters() if t.grad is not None])
    assert torch.allclose(outs[1], outs[0], rtol=1e-10, atol=1e-12)
    assert len(grads[0]) == len(grads[1])
    for a, b in zip(*grads):
        assert torch.allclose(b, a, rtol=1e-8, atol=1e-12)


def test_the_flag_only_changes_the_triton_branch_constructor():
    p = GuidedGroupThreshold(8, Mh=4, window=1, guide_window=5, attn_backend="gather",
                             sim_fun="pidistance", triton_materialize=True)
    q = cplx(1, 4, 6, 6, dtype=torch.float32)
    p.attn_backend = "triton"
    assert isinstance(p._fused_branch(q, q, 5), cw.WindowAdjacency)
    p.triton_materialize = False
    if not HAVE_TRITON:
        from models.circulant_triton import TritonAdjacency
        assert isinstance(p._fused_branch(q, q, 5), TritonAdjacency)    # lazy, builds fine


# =============================================================================================
# the kernels (GPU)
# =============================================================================================
cuda = pytest.mark.skipif(not (HAVE_TRITON and torch.cuda.is_available()),
                          reason="needs CUDA + triton")


@cuda
@pytest.mark.parametrize("complex_qk", [True, False])
@pytest.mark.parametrize("use_knorm", [True, False])
@pytest.mark.parametrize("hw_win", [((40, 33), 15), ((16, 16), 5), ((20, 18), 1)])
def test_score_kernel_matches_reference(complex_qk, use_knorm, hw_win):
    (H, W), win = hw_win
    g = torch.Generator(device="cuda").manual_seed(0)
    B, C = 2, 64
    mk = lambda: torch.randn(B, H * W, C, device="cuda", generator=g) * 0.3   # noqa: E731
    qr, kr = mk().requires_grad_(), mk().requires_grad_()
    qi, ki = (mk().requires_grad_(), mk().requires_grad_()) if complex_qk else (None, None)
    ins = [t for t in (qr, qi, kr, ki) if t is not None]
    dS = torch.randn(B, win * win, H * W, device="cuda", generator=g)

    res = []
    for force in (False, True):
        cw.FORCE_REFERENCE = force
        try:
            S = cw.window_scores(qr, qi, kr, ki, H, W, win, use_knorm=use_knorm)
            res.append((S.detach(),) + torch.autograd.grad(S, ins, dS))
        finally:
            cw.FORCE_REFERENCE = False
    for a, b in zip(*res):
        assert torch.allclose(a, b, rtol=1e-4, atol=1e-4 * b.abs().max().item())


@cuda
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("hw_win", [((40, 33), 15), ((16, 16), 5)])
@pytest.mark.parametrize("DV", [64, 33])
def test_apply_kernel_matches_reference(transpose, hw_win, DV):
    (H, W), win = hw_win
    g = torch.Generator(device="cuda").manual_seed(0)
    B = 2
    P = torch.softmax(torch.randn(B, win * win, H * W, device="cuda", generator=g), 1)
    P.requires_grad_()
    v = torch.randn(B, H * W, DV, device="cuda", generator=g, requires_grad=True)
    dout = torch.randn(B, H * W, DV, device="cuda", generator=g)
    res = []
    for force in (False, True):
        cw.FORCE_REFERENCE = force
        try:
            out = cw.window_apply(P, v, H, W, win, transpose)
            res.append((out.detach(),) + torch.autograd.grad(out, (P, v), dout))
        finally:
            cw.FORCE_REFERENCE = False
    for a, b in zip(*res):
        assert torch.allclose(a, b, rtol=1e-4, atol=1e-5 * b.abs().max().item())
