# -*- coding: utf-8 -*-
"""ForwardOp(kind="kfilter"): E(x; c) = sum_j a_j(c) * (B_j x) + b(c), a transfer function sampled
on radial k-space bands, linear in x with an exact adjoint.

Pinned here, because the kind is only worth having if they hold:

  * the bands are a partition of unity at any frame size (so E starts as the identity, and
    bands=1 is exactly kind="gain")
  * `adjoint` is the adjoint of `apply_linear` -- odd and even frames, UNet and global gains
  * the closed-form data_grad equals autograd, and training through it is first order
  * the global variant really is a shift-invariant filter; the UNet variant is not
"""
import sys
import types

import pytest
import torch

sys.modules.setdefault("nibabel", types.ModuleType("nibabel"))

from models.forward_ops import ForwardOp, radial_bands        # noqa: E402


def make(bands=4, cond=2, offset=True, seed=0, perturb=True):
    torch.manual_seed(seed)
    E = ForwardOp(kind="kfilter", width=4, levels=1, convs=1, cond_channels=cond, bands=bands,
                  offset=offset)
    if perturb:                                  # off the identity init, or every check is trivial
        with torch.no_grad():
            if E.net is not None:
                E.net.out.weight.normal_(0, 0.3)
                E.net.out.bias.normal_(0, 0.3)
            else:
                E.band_gain.normal_(0, 0.3)
                if E.bias is not None:
                    E.bias.normal_(0, 0.3)
    return E


def data(b=2, h=17, w=19, seed=1):
    torch.manual_seed(seed)
    return (torch.randn(b, 1, h, w), torch.randn(b, 1, h, w), torch.randn(b, 2, h, w))


@pytest.mark.parametrize("hw", [(16, 16), (17, 19), (32, 24), (128, 128)])
@pytest.mark.parametrize("n", [1, 2, 4, 8])
def test_bands_are_a_partition_of_unity(hw, n):
    beta = radial_bands(*hw, n, dtype=torch.float64)
    assert beta.shape == (n, hw[0], hw[1] // 2 + 1)
    assert (beta >= 0).all()
    assert torch.allclose(beta.sum(0), torch.ones_like(beta[0]), atol=1e-12)
    assert float(beta[0, 0, 0]) == 1.0, "DC belongs entirely to band 0"
    if n > 1:
        assert float(beta[-1].max()) == 1.0 and float(beta[:-1, 0, 0].sum()) == 1.0


@pytest.mark.parametrize("cond", [0, 2])
@pytest.mark.parametrize("bands", [1, 4, 8])
def test_starts_as_the_identity(bands, cond):
    E = make(bands, cond, perturb=False)
    x, _, c = data()
    assert torch.allclose(E(x, c if cond else None), x, atol=1e-5)


@pytest.mark.parametrize("hw", [(17, 19), (16, 20)])
@pytest.mark.parametrize("cond", [0, 2])
@pytest.mark.parametrize("bands", [1, 3, 6])
def test_adjoint_passes_the_dot_product_test(bands, cond, hw):
    E = make(bands, cond).double()
    x, u, c = (t.double() for t in data(h=hw[0], w=hw[1]))
    a, _ = E.coeffs(c if cond else None)
    lhs = (E.apply_linear(x, a) * u).sum()
    rhs = (x * E.adjoint(u, a)).sum()
    assert float(lhs) == pytest.approx(float(rhs), rel=1e-10)


def test_split_sums_to_x_and_merge_is_its_adjoint():
    E = make(5).double()
    x, _, _ = (t.double() for t in data())
    v = torch.randn(2, 5, 17, 19, dtype=torch.float64)
    assert torch.allclose(E.split(x).sum(1, keepdim=True), x, atol=1e-12)
    assert float((E.split(x) * v).sum()) == pytest.approx(float((x * E.merge(v)).sum()), rel=1e-10)


def test_one_band_is_exactly_the_gain_kind():
    """bands=1 has B_0 = I, so the two kinds are the same operator given the same UNet."""
    K = make(1)
    G = ForwardOp(kind="gain", width=4, levels=1, convs=1, cond_channels=2)
    G.net.load_state_dict(K.net.state_dict())
    x, y, c = data()
    assert torch.allclose(K(x, c), G(x, c), atol=1e-5)
    assert torch.allclose(K.data_grad(x, y, c), G.data_grad(x, y, c), atol=1e-5)


@pytest.mark.parametrize("cond", [0, 2])
@pytest.mark.parametrize("offset", [True, False])
def test_closed_form_data_grad_equals_autograd(cond, offset):
    E = make(4, cond, offset)
    x, y, c = data()
    c = c if cond else None
    g = E.data_grad(x, y, c)

    xr = x.clone().requires_grad_(True)
    loss = 0.5 * ((E(xr, c) - y) ** 2).sum()
    (want,) = torch.autograd.grad(loss, xr)
    assert torch.allclose(g, want, atol=1e-4)
    assert not g.requires_grad, "create_graph=False must return a plain tensor"


def test_training_through_it_builds_no_second_order_graph():
    E = make(4)
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
    double = {type(f).__name__ for f in seen if "BackwardBackward" in type(f).__name__}
    assert not double, "the kfilter data_grad graph contains double-backward nodes: %s" % sorted(double)

    g.sum().backward()
    assert xr.grad is not None and torch.isfinite(xr.grad).all()


def test_global_filter_is_shift_invariant_and_the_unet_one_is_not():
    """The global variant is one circular convolution: it commutes with circular shifts of x.
    With per-pixel gain maps the operator is tied to the image grid and must NOT commute -- if it
    did, the UNet's maps would be doing nothing."""
    x, _, c = data(h=16, w=16)
    roll = lambda t: torch.roll(t, shifts=(3, -5), dims=(-2, -1))      # noqa: E731

    G = make(6, cond=0, offset=False)
    assert torch.allclose(G(roll(x)), roll(G(x)), atol=1e-5)

    U = make(6, cond=2, offset=False)
    assert not torch.allclose(U(roll(x), c), roll(U(x, c)), atol=1e-3)


def test_no_offset_means_no_additive_path():
    """offset=False: E is LINEAR, not just affine -- E(0; c) = 0, so it cannot output a T1 that
    it synthesised from the side information alone."""
    for cond in (0, 2):
        E = make(4, cond, offset=False)
        x, _, c = data()
        c = c if cond else None
        assert float(E(torch.zeros_like(x), c).abs().max()) == 0.0
        assert float(make(4, cond, offset=True)(torch.zeros_like(x), c).abs().max()) > 0.0


def test_global_filter_ignores_side_information_and_has_no_network():
    E = make(8, cond=0)
    assert E.net is None
    assert sum(p.numel() for p in E.parameters()) == 9          # 8 band gains + 1 offset
    x, _, c = data()
    assert torch.equal(E(x), E(x, c))                           # the ladder passes c to every rung


def test_properties_and_guards():
    assert make(4).linear_in_x
    assert make(4).receptive_field == 0 and make(1).receptive_field == 1
    with pytest.raises(ValueError, match="use_x"):
        ForwardOp(kind="kfilter", cond_channels=2, use_x=False)
    with pytest.raises(ValueError, match="bands"):
        ForwardOp(kind="kfilter", bands=0)
    with pytest.raises(ValueError, match="cond channel"):
        make(4)(data()[0])                                      # conditioned, but no c given
