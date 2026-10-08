# -*- coding: utf-8 -*-
"""s_cross: cross-attention between ENHANCEMENT MAPS in SBGuidedGroupCDL.

One extra branch of the guided prox compares the current S estimate with the prior study's
S_prior = (CT1_prior - T1_prior)_+, both through the layer's dictionary, and adds the pooled
energy to xi^2 per atom. Pinned here:

  * it adds exactly one parameter tensor (the per-layer gain) and at gain 0 the net is the
    ordinary guided net run on the prior CT1 alone
  * the branch sees the prior study ONLY through S_prior: a shift common to prior T1 and CT1
    changes nothing, a change in their difference does
  * "gate" weighting acts on the enhancement atoms only (m = 0 -> no effect)
  * the query is built only when the adjacency refreshes
"""
import sys
import types

import pytest
import torch

sys.modules.setdefault("nibabel", types.ModuleType("nibabel"))

from models.enhancement import collapse_param_logs            # noqa: E402
from models.sb_guided_groupcdl import SBGuidedGroupCDL        # noqa: E402

SCHED = dict(kind="brownian", tau=0.1, n_points=50, beta_max=0.3)
H = 32


def net_(mode="gate", s_cross=True, K=2, dK=1, seed=0, **kw):
    torch.manual_seed(seed)
    return SBGuidedGroupCDL(K=K, M=8, C=4, P=3, s=2, Mh=4, window=1, guide_window=3,
                            joint_softmax=True, dK=dK, prior_idx=1, spectral_init=False,
                            attn_backend="gather", s_mode=mode, s_cross=s_cross, **SCHED, **kw)


def batch(seed=1):
    torch.manual_seed(seed)
    xt = torch.randn(2, 1, H, H)
    cond = torch.randn(2, 3, H, H)
    sigma = torch.full((2, 1, 1, 1), 0.1)
    t1p = torch.rand(2, 1, H, H)
    ct1p = t1p + (torch.rand(2, 1, H, H) > 0.9).float() * torch.rand(2, 1, H, H)   # sparse enhancement
    return xt, cond, sigma, t1p, ct1p


def run(net, xt, cond, sigma, t1p, ct1p):
    with torch.no_grad():
        out, _ = net.eval()(torch.cat([xt, cond], 1), sigma=sigma, guide=torch.stack([t1p, ct1p], 1))
    return out


def open_gate(net, value=0.6):
    with torch.no_grad():
        net.coupling.m.fill_(value)


def test_guards():
    with pytest.raises(ValueError, match="needs an S model"):
        net_(None)
    with pytest.raises(ValueError, match="s_cross_atoms"):
        net_("free", s_cross_atoms="gate")
    with pytest.raises(ValueError, match="s_cross_planes"):
        net_("gate", s_cross_planes=(1, 1))
    assert net_("gate").s_cross_atoms == "gate" and net_("free").s_cross_atoms == "all"
    xt, cond, sigma, t1p, ct1p = batch()
    with pytest.raises(ValueError, match="guide plane"):
        net_("gate")(torch.cat([xt, cond], 1), sigma=sigma, guide=ct1p[:, None])   # CT1 only


@pytest.mark.parametrize("mode", ["free", "gate"])
def test_adds_one_tensor_and_gain_zero_is_the_plain_guided_net(mode):
    cross, plain = net_(mode), net_(mode, s_cross=False)
    extra = set(dict(cross.named_parameters())) - set(dict(plain.named_parameters()))
    assert extra == {"cross_gain"} and cross.cross_gain.shape == (2,)
    missing, unexpected = cross.load_state_dict(plain.state_dict(), strict=False)
    assert missing == ["cross_gain"] and not unexpected
    if mode == "gate":
        open_gate(cross); open_gate(plain)
    xt, cond, sigma, t1p, ct1p = batch()
    with torch.no_grad():
        want, _ = plain.eval()(torch.cat([xt, cond], 1), sigma=sigma, guide=ct1p[:, None])
        cross.cross_gain.zero_()
    assert torch.allclose(run(cross, xt, cond, sigma, t1p, ct1p), want, atol=1e-6)
    with torch.no_grad():
        cross.cross_gain.fill_(1.0)
    assert not torch.allclose(run(cross, xt, cond, sigma, t1p, ct1p), want, atol=1e-5), \
        "the cross branch has no effect at gain 1"


@pytest.mark.parametrize("mode", ["free", "gate"])
def test_sees_the_prior_study_only_through_its_enhancement(mode):
    net = net_(mode)
    if mode == "gate":
        open_gate(net)
    xt, cond, sigma, t1p, ct1p = batch()
    base = run(net, xt, cond, sigma, t1p, ct1p)
    # a constant added to BOTH prior scans: S_prior is unchanged, and the CT1 guide is centred
    assert torch.allclose(run(net, xt, cond, sigma, t1p + 0.3, ct1p + 0.3), base, atol=1e-5)
    # prior T1 moved alone: only S_prior changes (prior T1 is not an ordinary guide)
    assert not torch.allclose(run(net, xt, cond, sigma, t1p - 0.2, ct1p), base, atol=1e-5)
    # no prior enhancement at all: the branch pools (almost) nothing
    with torch.no_grad():
        net.cross_gain.zero_()
    off = run(net, xt, cond, sigma, t1p, ct1p)
    with torch.no_grad():
        net.cross_gain.fill_(1.0)
    none = run(net, xt, cond, sigma, ct1p, ct1p)              # T1_prior = CT1_prior -> S_prior = 0
    assert torch.allclose(none, off, atol=1e-3)


def test_gate_weighting_acts_on_enhancement_atoms_only():
    net = net_("gate")
    xt, cond, sigma, t1p, ct1p = batch()
    with torch.no_grad():
        net.coupling.m.zero_()                                # no atom is an enhancement atom
        on = run(net, xt, cond, sigma, t1p, ct1p)
        net.cross_gain.zero_()
        off = run(net, xt, cond, sigma, t1p, ct1p)
    assert torch.equal(on, off)
    allatoms = net_("gate", s_cross_atoms="all")              # ... but "all" ignores the gate
    with torch.no_grad():
        allatoms.coupling.m.zero_()
        a = run(allatoms, xt, cond, sigma, t1p, ct1p)
        allatoms.cross_gain.zero_()
        b = run(allatoms, xt, cond, sigma, t1p, ct1p)
    assert not torch.allclose(a, b, atol=1e-5)


def test_query_is_built_exactly_when_the_adjacency_refreshes():
    """The cross attention follows Phi's own dK schedule: its query (a synthesis + an analysis)
    is built on the layers where Phi is rebuilt and on no others."""
    net = net_("gate", K=6, dK=3)
    open_gate(net)
    q_calls, phi_calls = [], []
    orig = net.coupling.decode_layer
    net.coupling.decode_layer = lambda k, B, z: (q_calls.append(k), orig(k, B, z))[1]
    for k, layer in enumerate(net.layers):
        def spy(*a, _k=k, _f=layer.prox._build_adjacencies, **kw):
            phi_calls.append(_k)
            return _f(*a, **kw)
        layer.prox._build_adjacencies = spy
    xt, cond, sigma, t1p, ct1p = batch()
    run(net, xt, cond, sigma, t1p, ct1p)
    assert q_calls == phi_calls, (q_calls, phi_calls)
    assert 0 in q_calls and len(q_calls) < 6, q_calls


@pytest.mark.parametrize("mode", ["free", "gate"])
def test_trains_projects_and_logs(mode):
    net = net_(mode)
    if mode == "gate":
        open_gate(net)
    xt, cond, sigma, t1p, ct1p = batch()
    out, _ = net(torch.cat([xt, cond], 1), sigma=sigma, guide=torch.stack([t1p, ct1p], 1))
    out.pow(2).mean().backward()
    assert float(net.cross_gain.grad.abs().sum()) > 0
    if mode == "gate":
        assert float(net.coupling.m.grad.abs().sum()) > 0
    with torch.no_grad():
        net.cross_gain.copy_(torch.tensor([-1.0, 0.5]))
    net.project()
    assert net.cross_gain.tolist() == [0.0, 0.5]
    logs = collapse_param_logs(net)
    assert logs["cross_gain_min"] == 0.0 and logs["cross_gain_mean"] == pytest.approx(0.25)
    assert "cross_gain_min" not in collapse_param_logs(net_(mode, s_cross=False))


def test_synthesis_layout():
    """task: synthesis hands the net [cond..., guides...]; the prior pair rides as trailing planes."""
    net = net_("gate", bridge_fidelity=False).eval()
    open_gate(net)
    xt, cond, sigma, t1p, ct1p = batch()
    with torch.no_grad():
        a, _ = net(torch.cat([cond, t1p, ct1p], 1))
        b, _ = net(cond, guide=torch.stack([t1p, ct1p], 1))
    assert torch.allclose(a, b, atol=1e-6) and a.shape == xt.shape
