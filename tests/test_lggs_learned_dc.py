# -*- coding: utf-8 -*-
"""LGGS with a learned (nonlinear) data operator, and the magnitude operator it uses.

What is pinned, and why each is a silent-failure risk:

  * MagnitudeOperator.data_grad on a COMPLEX iterate is the gradient w.r.t. (Re x, Im x) packed as
    Re + i Im. Complex autograd has a conjugation convention; getting it backwards turns the
    primal step into ascent on the imaginary part and nothing errors.
  * LGGS's nonlinear branch reduces EXACTLY to the existing linear path when E is the identity
    in disguise -- the strongest available check that the new branch computes the same sweep.
  * the nonlinear branch trains: gradients reach the dictionaries through the data-grad VJP.
"""
import sys
import types

import pytest
import torch

sys.modules.setdefault("nibabel", types.ModuleType("nibabel"))

from models.forward_ops import ForwardOp                       # noqa: E402
from models.guided_lpds import LGGSNet                         # noqa: E402
from operators.learned import MagnitudeOperator                # noqa: E402


def small_E(cond=0, seed=0, perturb=True):
    """A tiny UNet ForwardOp. `perturb` moves its zero-initialised readout so E != identity."""
    torch.manual_seed(seed)
    E = ForwardOp(kind="unet", width=4, levels=1, convs=1, cond_channels=cond)
    if perturb:
        with torch.no_grad():
            E.net.out.weight.normal_(0, 0.1)
            E.net.out.bias.normal_(0, 0.1)
    return E.requires_grad_(False).eval()


def lggs(is_complex, preproc="identity", K=3, seed=0):
    torch.manual_seed(seed)
    return LGGSNet(K=K, M=8, C=1, P=7, s=2, Mh=8, guide_window=5, is_complex=is_complex,
                   preproc=preproc)


# ---------------------------------------------------------------------------------------------
def test_magnitude_data_grad_matches_the_real_pair_gradient():
    """g = dL/dRe + i dL/dIm, computed independently by differentiating w.r.t. two REAL leaves."""
    E = small_E()
    torch.manual_seed(1)
    a = torch.randn(2, 1, 16, 16)
    b = torch.randn(2, 1, 16, 16)
    y = torch.rand(2, 1, 16, 16)

    op = MagnitudeOperator(E)
    with torch.no_grad():
        g = op.data_grad(torch.complex(a, b), y)

    ar, br = a.clone().requires_grad_(True), b.clone().requires_grad_(True)
    mag = (ar ** 2 + br ** 2 + op.eps).sqrt()
    loss = 0.5 * ((E(mag) - y) ** 2).sum()
    ga, gb = torch.autograd.grad(loss, (ar, br))

    assert torch.allclose(g.real, ga, atol=1e-5), "Re of the complex data_grad is wrong"
    assert torch.allclose(g.imag, gb, atol=1e-5), (
        "Im of the complex data_grad is wrong -- check the conjugation convention")


def test_a_primal_step_along_the_data_grad_descends():
    E = small_E()
    torch.manual_seed(2)
    x = torch.complex(torch.rand(1, 1, 16, 16), 0.1 * torch.randn(1, 1, 16, 16))
    y = torch.rand(1, 1, 16, 16)
    op = MagnitudeOperator(E)
    f = lambda v: float(0.5 * ((op(v) - y) ** 2).sum())   # noqa: E731
    with torch.no_grad():
        g = op.data_grad(x, y)
    assert f(x - 1e-3 * g) < f(x), "a small step along -data_grad did not reduce the data term"


def test_magnitude_of_a_real_tensor_is_the_tensor():
    op = MagnitudeOperator(small_E())
    x = torch.randn(1, 1, 8, 8)
    assert torch.equal(op.magnitude(x), x)


# ---------------------------------------------------------------------------------------------
def test_nonlinear_branch_equals_the_linear_path_for_an_identity_operator():
    """Real LGGS, preproc='identity'. A zero-readout ForwardOp IS the identity (residual init),
    and the magnitude of a real tensor is the tensor, so E(x) = x and data_grad = x - y: the very
    residual the linear Identity path computes as gram(I, x) - y. The two must agree bit-for-bit
    up to float noise -- any difference is the new branch doing a different sweep."""
    net = lggs(is_complex=False, preproc="identity")
    torch.manual_seed(3)
    y = torch.rand(2, 1, 32, 32)
    g = torch.rand(2, 1, 1, 32, 32)
    with torch.no_grad():
        lin, _ = net(y, guide=g)
        nl, _ = net(y, guide=g, E=MagnitudeOperator(small_E(perturb=False)))
    assert torch.allclose(lin, nl, atol=1e-5), (
        "nonlinear branch with E = identity differs from the linear path by %.3g"
        % float((lin - nl).abs().max()))


def test_nonlinear_branch_runs_with_image_preprocessing_and_complex_weights():
    """The configuration the timing script measures: complex LGGS, mean removal + padding."""
    net = lggs(is_complex=True, preproc="image")
    torch.manual_seed(4)
    y = torch.rand(2, 1, 30, 30)                  # not a multiple of s: forces padding
    g = torch.rand(2, 1, 1, 30, 30)
    c = torch.rand(2, 2, 30, 30)
    with torch.no_grad():
        out, _ = net(y, guide=g, E=MagnitudeOperator(small_E(cond=2), cond=c))
    assert out.shape == y.shape
    assert torch.is_complex(out)
    assert torch.isfinite(out.abs()).all()


def test_gradients_reach_the_dictionaries_through_the_learned_dc():
    """Training THROUGH the data-grad VJP: second order in E, first order in LGGS's weights."""
    net = lggs(is_complex=True, preproc="image", K=2)
    torch.manual_seed(5)
    y = torch.rand(2, 1, 32, 32)
    tgt = torch.rand(2, 1, 32, 32)
    g = torch.rand(2, 1, 1, 32, 32)
    out, _ = net(y, guide=g, E=MagnitudeOperator(small_E()))
    loss = ((out.abs() - tgt) ** 2).mean()        # the magnitude loss
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.requires_grad]
    assert any(gr is not None and float(gr.abs().sum()) > 0 for gr in grads)
    assert all(gr is None or torch.isfinite(gr).all() for gr in grads)


def test_kspace_preprocessing_is_refused():
    net = LGGSNet(K=1, M=8, C=1, P=7, s=2, Mh=8, guide_window=5, preproc="kspace")
    with pytest.raises(ValueError, match="image domain"):
        net(torch.rand(1, 1, 16, 16), E=MagnitudeOperator(small_E()))
