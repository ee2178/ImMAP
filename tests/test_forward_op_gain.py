# -*- coding: utf-8 -*-
"""ForwardOp(kind="gain"): E(x; c) = a(c) * x + k * x + b(c), linear in x with an exact adjoint.

The reason this kind exists is to replace an autograd VJP with a closed-form adjoint inside an
unrolled net, so the things worth pinning are the ones that claim rests on:

  * `adjoint` really is the adjoint of `apply_linear` (dot-product test, with and without the conv)
  * the closed-form data_grad equals the autograd gradient of 1/2 ||y - E(x)||^2
  * it trains through with NO second-order graph (data_grad's graph has no autograd.grad nodes)
  * it starts as the identity, like every other ForwardOp
"""
import sys
import types

import pytest
import torch

sys.modules.setdefault("nibabel", types.ModuleType("nibabel"))

from models.forward_ops import ForwardOp        # noqa: E402


def make(gain_kernel=0, seed=0, perturb=True):
    torch.manual_seed(seed)
    E = ForwardOp(kind="gain", width=4, levels=1, convs=1, cond_channels=2,
                  gain_kernel=gain_kernel)
    if perturb:                                  # off the identity init, or every check is trivial
        with torch.no_grad():
            E.net.out.weight.normal_(0, 0.3)
            E.net.out.bias.normal_(0, 0.3)
            if E.lin is not None:
                E.lin.weight.normal_(0, 0.2)
    return E


def data(b=2, h=17, w=19, seed=1):
    torch.manual_seed(seed)
    return (torch.randn(b, 1, h, w), torch.randn(b, 1, h, w), torch.randn(b, 2, h, w))


@pytest.mark.parametrize("k", [0, 3, 5])
def test_starts_as_the_identity(k):
    E = make(k, perturb=False)
    x, _, c = data()
    assert torch.allclose(E(x, c), x, atol=1e-6)


@pytest.mark.parametrize("k", [0, 3, 5])
def test_adjoint_passes_the_dot_product_test(k):
    """<A x, u> == <x, A^T u> for random x, u -- on an odd-sized frame, where a padding mismatch
    between the conv and its transpose would show."""
    E = make(k).double()
    x, u, c = (t.double() for t in data())
    a, _ = E.coeffs(c)
    lhs = (E.apply_linear(x, a) * u).sum()
    rhs = (x * E.adjoint(u, a)).sum()
    assert float(lhs) == pytest.approx(float(rhs), rel=1e-10)


@pytest.mark.parametrize("k", [0, 3])
def test_forward_is_affine_in_x(k):
    E = make(k)
    x1, x2, c = data()
    a, b = E.coeffs(c)
    assert torch.allclose(E(x1, c), E.apply_linear(x1, a) + b, atol=1e-6)
    # superposition on the linear part
    assert torch.allclose(E.apply_linear(2 * x1 - 3 * x2, a),
                          2 * E.apply_linear(x1, a) - 3 * E.apply_linear(x2, a), atol=1e-5)


@pytest.mark.parametrize("k", [0, 3])
def test_closed_form_data_grad_equals_autograd(k):
    E = make(k)
    x, y, c = data()
    g = E.data_grad(x, y, c)

    xr = x.clone().requires_grad_(True)
    loss = 0.5 * ((E(xr, c) - y) ** 2).sum()
    (want,) = torch.autograd.grad(loss, xr)
    assert torch.allclose(g, want, atol=1e-5)
    assert not g.requires_grad, "create_graph=False must return a plain tensor"


def test_training_through_it_builds_no_second_order_graph():
    """The cost this kind removes. A UNet ForwardOp's data_grad calls autograd.grad with
    create_graph=True, so backprop through it is a double backward. Here the gradient is ordinary
    forward ops: differentiable, and its graph contains no backward-of-backward nodes."""
    E = make(3)
    x, y, c = data()
    xr = x.clone().requires_grad_(True)
    g = E.data_grad(xr, y, c, create_graph=True)
    assert g.requires_grad

    seen, stack = set(), [g.grad_fn]
    while stack:
        fn = stack.pop()
        if fn is None or fn in seen:
            continue
        seen.add(fn)
        stack.extend(f for f, _ in fn.next_functions)
    names = {type(f).__name__ for f in seen}
    double = {n for n in names if "BackwardBackward" in n}
    assert not double, "the gain data_grad graph contains double-backward nodes: %s" % sorted(double)

    g.sum().backward()                           # and it backpropagates to x
    assert xr.grad is not None and torch.isfinite(xr.grad).all()


def test_gain_needs_side_information():
    with pytest.raises(ValueError, match="cond_channels"):
        ForwardOp(kind="gain", cond_channels=0)
    with pytest.raises(ValueError, match="odd"):
        ForwardOp(kind="gain", cond_channels=2, gain_kernel=4)


def test_properties():
    E = make(5)
    assert E.linear_in_x
    assert E.receptive_field == 5
    assert make(0).receptive_field == 1
    assert not ForwardOp(kind="unet", width=4, levels=1, convs=1).linear_in_x
