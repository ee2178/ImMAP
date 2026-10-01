# -*- coding: utf-8 -*-
"""CycleSynth + the cycle term in train_synthesis.

  * forward(X) is G alone, so every consumer that treats this as a plain synthesis net is right.
  * the cycle term is loss(F(G(X)), T1) and gradients reach BOTH nets through it.
  * F is never handed T1 -- otherwise the cycle term is satisfied by copying it.
  * in residual mode F sees the CT1-domain estimate, not the residual.
  * cycle_weight=0 leaves the plain synthesis objective exactly as it was.
"""
import sys
import types

import pytest
import torch

sys.modules.setdefault("nibabel", types.ModuleType("nibabel"))

from models.cycle import CycleSynth                     # noqa: E402
from training.synthesis import cycle_term              # noqa: E402

UNET = {"type": "Unet2D", "params": {"in_chans": 5, "out_chans": 1, "chans": 4,
                                     "num_pool_layers": 2, "max_chans": 16,
                                     "convs_per_block": 1, "norm": "group", "act": "relu",
                                     "group_size": 2, "up_mode": "nearest", "up_conv": True,
                                     "head": "conv"}}
SHALLOW = {"type": "ForwardOp", "params": {"kind": "unet", "width": 4, "levels": 1, "convs": 1,
                                           "cond_channels": 2}}


def make(backward="shallow"):
    torch.manual_seed(0)
    F = SHALLOW if backward == "shallow" else \
        {"type": "Unet2D", "params": dict(UNET["params"], in_chans=3)}
    return CycleSynth(forward=UNET, backward=F, src_idx=1, side_idx=[0, 2])


def X(b=2, c=5, h=16):
    torch.manual_seed(1)
    return torch.rand(b, c, h, h)


@pytest.mark.parametrize("backward", ["shallow", "same"])
def test_forward_is_G_alone(backward):
    net = make(backward)
    x = X()
    assert torch.equal(net(x), net.G(x))


def test_t1_is_the_src_channel():
    net = make()
    x = X()
    assert torch.equal(net.t1(x), x[:, 1:2])


def test_F_never_sees_t1():
    with pytest.raises(ValueError, match="copying"):
        CycleSynth(forward=UNET, backward=SHALLOW, src_idx=1, side_idx=[0, 1])


def test_F_sees_exactly_the_estimate_and_the_side_channels():
    """Replace T1 in X with garbage: F's output must not change, because it never reads T1."""
    net = make("same")
    x = X()
    ct1 = torch.rand(2, 1, 16, 16)
    a = net.back(ct1, x)
    x2 = x.clone()
    x2[:, 1] = 1e3
    b = net.back(ct1, x2)
    assert torch.equal(a, b), "F's output depends on the T1 channel"


@pytest.mark.parametrize("backward", ["shallow", "same"])
def test_cycle_term_trains_both_nets(backward):
    net = make(backward)
    x = X()
    pred = net(x)
    cyc = cycle_term(net, x, pred, None, torch.ones(2, 1, 16, 16), False, "complex-mse")
    want = torch.mean((net.back(pred, x) - x[:, 1:2]) ** 2)
    assert float(cyc) == pytest.approx(float(want), rel=1e-5)
    cyc.backward()
    for name, sub in (("G", net.G), ("F", net.F)):
        g = [p.grad for p in sub.parameters() if p.grad is not None]
        assert g and any(float(t.abs().sum()) > 0 for t in g), f"no gradient reached {name}"


def test_residual_mode_feeds_F_the_ct1_domain_estimate():
    net = make()
    x = X()
    resid = torch.rand(2, 1, 16, 16)
    src = x[:, 1:2]
    a = cycle_term(net, x, resid, src, torch.ones(2, 1, 16, 16), False, "complex-mse")
    b = torch.mean((net.back(resid + src, x) - x[:, 1:2]) ** 2)
    assert float(a) == pytest.approx(float(b), rel=1e-5)


def test_shallow_F_starts_as_the_identity():
    """Zero-initialised residual ForwardOp: F(CT1) = CT1 before training."""
    net = make("shallow")
    x = X()
    ct1 = torch.rand(2, 1, 16, 16)
    assert torch.allclose(net.back(ct1, x), ct1)


# ---------------------------------------------------------------------------------------------
from training.synthesis import backward_term           # noqa: E402


def test_backward_term_is_F_on_the_true_ct1():
    net = make("same")
    x = X()
    ct1 = torch.rand(2, 1, 16, 16)
    got = backward_term(net, x, ct1, torch.ones(2, 1, 16, 16), False, "complex-mse")
    want = torch.mean((net.back(ct1, x) - x[:, 1:2]) ** 2)
    assert float(got) == pytest.approx(float(want), rel=1e-5)


def test_backward_term_trains_F_and_never_G():
    """It is F's supervision on real pairs: G must get no gradient from it."""
    net = make("same")
    x = X()
    backward_term(net, x, torch.rand(2, 1, 16, 16), torch.ones(2, 1, 16, 16), False,
                  "complex-mse").backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in net.F.parameters())
    assert all(p.grad is None or float(p.grad.abs().sum()) == 0 for p in net.G.parameters()), (
        "the direct backward term leaked gradient into G")
