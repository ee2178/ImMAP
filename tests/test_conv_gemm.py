# -*- coding: utf-8 -*-
"""Unfold + GEMM convolutions (models/conv_gemm.py) against cuDNN's / ATen's, in float64.

The GEMM path is the same linear map as `F.conv2d` / `F.conv_transpose2d`, so outputs and BOTH
gradients (input and weight) must agree to roundoff: the raw functions over strides, kernel
sizes, odd grids and batches; the dictionary conv modules in every input form (planar state,
complex, real); and LGGS end to end with the switch on against off.
"""
import sys
import types

import pytest
import torch
import torch.nn.functional as F

sys.modules.setdefault("nibabel", types.ModuleType("nibabel"))

from models import conv_gemm                                       # noqa: E402
from models.components import Conv2d, ConvTranspose2d, to_planar   # noqa: E402
from models.guided_lpds import LGGSNet                             # noqa: E402
from models.mg_lpds import MGLPDSNet                               # noqa: E402

D = torch.float64


@pytest.fixture
def gemm(monkeypatch):
    def set_(mode):
        monkeypatch.setattr(conv_gemm, "MODE", mode)
    return set_


def close(a, b, tol=1e-12):
    a, b = a.detach(), b.detach()
    return torch.allclose(a, b, rtol=tol, atol=tol * float(b.abs().max()))


def grads(out, ins, seed=0):
    g = torch.randn(out.shape, dtype=out.dtype, generator=torch.Generator().manual_seed(seed))
    return torch.autograd.grad(out, ins, g)


@pytest.mark.parametrize("P,s", [(7, 2), (7, 1), (3, 2), (5, 2), (3, 1)])
@pytest.mark.parametrize("hw", [(20, 14), (17, 13)])
@pytest.mark.parametrize("B", [1, 3])
def test_conv_and_transpose_match_aten(P, s, hw, B, gemm):
    torch.manual_seed(0)
    p = (P - 1) // 2
    Cin, N = 2, 12
    x = torch.randn(B, Cin, *hw, dtype=D, requires_grad=True)
    W = torch.randn(N, Cin, P, P, dtype=D, requires_grad=True)
    gemm("ieee")
    out = conv_gemm.conv2d(x, W, stride=s, padding=p)
    ref = F.conv2d(x, W, stride=s, padding=p)
    assert out.shape == ref.shape and close(out, ref)
    for a, b in zip(grads(out, (x, W)), grads(ref, (x, W))):
        assert close(a, b)

    # the transpose: N -> Cin back onto the grid conv2d came from (output_padding = s - 1)
    z = torch.randn(B, N, *ref.shape[-2:], dtype=D, requires_grad=True)
    Wt = torch.randn(N, Cin, P, P, dtype=D, requires_grad=True)
    op = s - 1
    y = conv_gemm.conv_transpose2d(z, Wt, stride=s, padding=p, output_padding=op)
    yr = F.conv_transpose2d(z, Wt, stride=s, padding=p, output_padding=op)
    assert y.shape == yr.shape and close(y, yr)
    for a, b in zip(grads(y, (z, Wt)), grads(yr, (z, Wt))):
        assert close(a, b)


def test_the_gemm_path_is_actually_taken_and_falls_back_when_inexpressible(gemm, monkeypatch):
    calls = []
    real = conv_gemm._Conv.apply
    monkeypatch.setattr(conv_gemm._Conv, "apply", lambda *a: calls.append(1) or real(*a))
    x, W = torch.randn(1, 2, 9, 9, dtype=D), torch.randn(4, 2, 3, 3, dtype=D)
    gemm(None)
    conv_gemm.conv2d(x, W, padding=1)
    assert not calls                                        # off: the ATen conv
    gemm("ieee")
    conv_gemm.conv2d(x, W, padding=1)
    assert calls == [1]
    conv_gemm.conv2d(x, W, bias=torch.zeros(4, dtype=D), padding=1)            # a bias
    conv_gemm.conv2d(torch.randn(1, 4, 9, 9, dtype=D), torch.randn(4, 2, 3, 3, dtype=D),
                     padding=1, groups=2)                                       # groups
    big = torch.randn(4, 64, 7, 7, dtype=D)                                     # K = 3136
    conv_gemm.conv2d(torch.randn(1, 64, 9, 9, dtype=D), big, padding=3)
    assert calls == [1]                                     # none of those took it


@pytest.mark.parametrize("form", ["planar", "complex", "real"])
def test_dictionary_conv_modules(form, gemm):
    """The LPDS analysis / synthesis pair, C = 1 -> M, in each input form it is fed."""
    torch.manual_seed(0)
    cplx = form != "real"
    A = Conv2d(1, 9, 7, stride=2, complex=cplx).double()
    Bt = ConvTranspose2d(9, 1, 7, stride=2, complex=cplx).double()
    if form == "real":
        x = torch.randn(2, 1, 20, 16, dtype=D)
    else:
        x = torch.complex(torch.randn(2, 1, 20, 16, dtype=D), torch.randn(2, 1, 20, 16, dtype=D))
        if form == "planar":
            x = to_planar(x)
    x.requires_grad_(True)
    params = list(A.parameters()) + list(Bt.parameters())
    res = []
    for mode in (None, "ieee"):
        gemm(mode)
        z = A(x)
        y = Bt(z)
        loss = (y.abs() ** 2).sum() + (z.abs() ** 2).sum()
        res.append([y.detach(), z.detach()] + list(torch.autograd.grad(loss, [x] + params)))
    for a, b in zip(*res):
        assert close(a, b, 1e-11)


@pytest.mark.parametrize("planar_state", [False, True])
def test_lggs_with_gemm_convs_matches_cudnn_convs(planar_state, gemm, monkeypatch):
    monkeypatch.setattr(MGLPDSNet, "PLANAR_STATE", planar_state)
    torch.manual_seed(0)
    net = LGGSNet(K=6, M=12, C=1, P=7, s=2, Mh=4, window=1, guide_window=5, dK=3, lam0=0.05,
                  theta0=0.3, preproc="image", attn_backend="gather", spectral_init=False).double()
    g = torch.Generator().manual_seed(1)
    c = lambda: torch.complex(torch.randn(2, 1, 18, 14, generator=g, dtype=D),   # noqa: E731
                              torch.randn(2, 1, 18, 14, generator=g, dtype=D))
    y, guides = c(), [c(), c()]
    sigma = torch.full((2, 1, 1, 1), 0.05, dtype=D)
    res = []
    for mode in (None, "ieee"):
        gemm(mode)
        net.zero_grad()
        xh, (_, z) = net(y, guide=guides, sigma=sigma)
        (xh.abs().pow(2).mean() + 0.1 * z.abs().pow(2).mean()).backward()
        res.append((xh.detach(), z.detach(),
                    {k: p.grad.clone() for k, p in net.named_parameters() if p.grad is not None}))
    (x0, z0, g0), (x1, z1, g1) = res
    assert close(x1, x0, 1e-11) and close(z1, z0, 1e-11)
    assert g0.keys() == g1.keys()
    for k in g0:
        assert close(g1[k], g0[k], 1e-9), k


def test_precision_is_set_inside_and_restored(gemm):
    """`ieee` forces IEEE GEMMs for its own calls only; the process setting comes back."""
    keep = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        seen = []
        orig = torch.matmul

        def spy(*a, **k):
            seen.append(torch.get_float32_matmul_precision())
            return orig(*a, **k)

        gemm("ieee")
        x = torch.randn(1, 2, 9, 9, requires_grad=True)
        W = torch.randn(4, 2, 3, 3, requires_grad=True)
        torch.matmul = spy
        try:
            conv_gemm.conv2d(x, W, padding=1).sum().backward()
        finally:
            torch.matmul = orig
        assert seen and set(seen) == {"highest"}
        assert torch.get_float32_matmul_precision() == "high"
    finally:
        torch.set_float32_matmul_precision(keep)
