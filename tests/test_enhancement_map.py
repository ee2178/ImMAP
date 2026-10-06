# -*- coding: utf-8 -*-
"""The shared-code enhancement map S (models/enhancement.py) in SBCDLNet and SBGuidedGroupCDL.

    free   S = D_S z         y = (D - D_S) z
    gate   S = D (m . z)     y = D ((1 - m) . z)

What is pinned:
  * s_mode=None is the old net: same parameters, no S
  * the measurement gradient is the exact gradient of 1/2 ||y - D_y z||^2 (no autograd inside)
  * both start at "no enhancement" (S = 0, D_y = D) and x - S = D_y z holds at readout
  * the gate's two ends: m = 0 is the plain fidelity, m = 1 makes T1 constrain nothing
  * bridge_fidelity=False never reads x_t or sigma, and equals the `task: synthesis` call
  * the S loss reaches the coupling parameters; a plain net raises instead of ignoring it
"""
import sys
import types

import pytest
import torch

sys.modules.setdefault("nibabel", types.ModuleType("nibabel"))

from models.enhancement import (CollapseMeter, collapse_param_logs, enhancement_loss,    # noqa: E402
                                enhancement_panel, enhancement_target, step_logit)
from models.sb_cdlnet import SBCDLNet                                 # noqa: E402
from models.sb_guided_groupcdl import SBGuidedGroupCDL                # noqa: E402

SCHED = dict(kind="brownian", tau=0.1, n_points=50, beta_max=0.3)
H = 32


def cdl(s_mode=None, C=4, prior_idx=1, seed=0, **kw):
    torch.manual_seed(seed)
    return SBCDLNet(K=3, M=8, P=3, s=2, C=C, prior_idx=prior_idx, init=False, complex=False,
                    s_mode=s_mode, **SCHED, **kw)


def guided(s_mode=None, C=4, prior_idx=1, seed=0, **kw):
    torch.manual_seed(seed)
    return SBGuidedGroupCDL(K=2, M=8, C=C, P=3, s=2, Mh=4, window=1, guide_window=3,
                            joint_softmax=True, dK=1, prior_idx=prior_idx, spectral_init=False,
                            attn_backend="gather", s_mode=s_mode, **SCHED, **kw)


def batch(C=4, b=2, seed=1):
    torch.manual_seed(seed)
    xt = torch.randn(b, 1, H, H)
    cond = torch.randn(b, C - 1, H, H)
    sigma = torch.full((b, 1, 1, 1), 0.1)
    return xt, cond, sigma


def perturb(net):
    """Move the coupling off its 'no enhancement' start, keeping A_S the adjoint of B_S."""
    cp = net.coupling
    with torch.no_grad():
        if cp.mode == "gate":
            cp.m.uniform_(0.1, 0.9)
        else:
            for a, b in zip(cp.A_S, cp.B_S):
                w = 0.3 * torch.randn_like(b.conv_real.weight)
                b.conv_real.weight.copy_(w)
                a.conv_real.weight.copy_(w)


# ---------------------------------------------------------------------------------------------
def test_default_is_the_old_net():
    net = cdl(None)
    names = [n for n, _ in net.named_parameters()]
    assert not any("coupling" in n or "a_xi" in n for n in names)
    assert net.A_P[0].conv_real.weight.shape[1] == 3          # every conditioning channel
    xt, cond, sigma = batch()
    out, z = net(torch.cat([xt, cond], 1), sigma=sigma)
    assert out.shape == xt.shape and net.last_S is None
    with pytest.raises(ValueError, match="s_mode"):
        cdl(None, bridge_fidelity=False)
    with pytest.raises(ValueError, match="s_mode must be"):
        cdl("additive")


@pytest.mark.parametrize("mode", ["free", "gate"])
def test_prior_channel_has_no_dictionary_of_its_own(mode):
    net = cdl(mode)
    assert net.side_idx == [0, 2] and net.A_P[0].conv_real.weight.shape[1] == 2
    solo = cdl(mode, C=2, prior_idx=0)                        # T1 is the ONLY conditioning
    assert solo.n_p == 0 and len(solo.A_P) == 0
    xt, cond, sigma = batch(C=2)
    out, _ = solo(torch.cat([xt, cond], 1), sigma=sigma)
    assert out.shape == xt.shape and torch.isfinite(out).all()
    solo.project()                                            # must not index an empty P pair


@pytest.mark.parametrize("mode", ["free", "gate"])
@pytest.mark.parametrize("bridge,n_side,want", [(True, 2, 1 / 3), (False, 2, 0.5), (True, 0, 0.5),
                                                (False, 0, 0.9)])
def test_steps_start_at_one_over_n(mode, bridge, n_side, want):
    net = cdl(mode, C=2 + n_side, prior_idx=0, bridge_fidelity=bridge)
    assert float(torch.sigmoid(net.a_xi[0, 0])) == pytest.approx(want, abs=1e-6)
    assert step_logit(2) == pytest.approx(0.0, abs=1e-7)       # the existing two-fidelity init


@pytest.mark.parametrize("mode", ["free", "gate"])
def test_starts_with_no_enhancement(mode):
    net = cdl(mode)
    xt, cond, sigma = batch()
    net(torch.cat([xt, cond], 1), sigma=sigma)
    assert net.last_S.shape == xt.shape
    assert float(net.last_S.abs().max()) == 0.0
    # and the measurement gradient is then the plain target-dictionary fidelity
    z = torch.randn(2, 8, H // 2, H // 2)
    y = torch.randn(2, 1, H, H)
    A, B = net.A_D[1], net.B_D[1]
    assert torch.allclose(net.coupling.meas_grad(1, A, B, z, y), A(B(z) - y), atol=1e-6)


@pytest.mark.parametrize("mode", ["free", "gate"])
def test_measurement_gradient_is_exact(mode):
    """With a tied pair (A = B^T, as at init) meas_grad is d/dz of 1/2 ||y - D_y z||^2."""
    net = cdl(mode).double()
    perturb(net)
    cp, A, B = net.coupling, net.A_D[1], net.B_D[1]
    z = torch.randn(2, 8, H // 2, H // 2, dtype=torch.float64, requires_grad=True)
    y = torch.randn(2, 1, H, H, dtype=torch.float64)
    Dy = (B(z) - cp.B_S[1](z)) if mode == "free" else B((1 - cp.gate()) * z)
    (want,) = torch.autograd.grad(0.5 * ((Dy - y) ** 2).sum(), z)
    got = cp.meas_grad(1, A, B, z.detach(), y)
    assert torch.allclose(got, want, atol=1e-9)


@pytest.mark.parametrize("mode", ["free", "gate"])
def test_readout_satisfies_x_minus_S_equals_Dy_z(mode):
    net = cdl(mode)
    perturb(net)
    xt, cond, sigma = batch()
    x0_hat, z = net(torch.cat([xt, cond], 1), sigma=sigma)
    dc = cond[:, 1:2].mean(dim=(1, 2, 3), keepdim=True)
    cp, B0 = net.coupling, net.B_D[0]
    Dy = (B0(z) - cp.B_S[0](z)) if mode == "free" else B0((1 - cp.gate()) * z)
    assert float(net.last_S.abs().max()) > 0
    # float32, and an un-normalised dictionary (init=False): compare relative to the scale
    assert float((x0_hat - dc - net.last_S - Dy).abs().max()) < 1e-5 * float(Dy.abs().max())


def test_gate_ends():
    net = cdl("gate")
    cp, A, B = net.coupling, net.A_D[0], net.B_D[0]
    z, y = torch.randn(2, 8, H // 2, H // 2), torch.randn(2, 1, H, H)
    with torch.no_grad():
        cp.m.fill_(1.0)
    assert float(cp.meas_grad(0, A, B, z, y).abs().max()) == 0.0, "m = 1: T1 constrains nothing"
    assert torch.allclose(cp.decode(B, z), B(z)), "m = 1: all of x is 'enhancement'"
    with torch.no_grad():                                      # one open atom: S is that atom only
        cp.m.zero_(); cp.m[3] = 1.0
    only = torch.zeros_like(z); only[:, 3] = z[:, 3]
    assert torch.allclose(cp.decode(B, z), B(only), atol=1e-6)
    with torch.no_grad():
        cp.m.copy_(torch.linspace(-0.5, 1.5, 8))
    net.project()
    assert float(cp.m.min()) == 0.0 and float(cp.m.max()) == 1.0
    with pytest.raises(ValueError, match="gate_init"):
        cdl("gate", gate_init=1.5)


@pytest.mark.parametrize("make", [cdl, guided])
@pytest.mark.parametrize("mode", ["free", "gate"])
def test_dc_only_never_reads_the_bridge_and_matches_the_synthesis_call(make, mode):
    net = make(mode, bridge_fidelity=False).eval()
    perturb(net)
    xt, cond, sigma = batch()
    with torch.no_grad():
        a, _ = net(torch.cat([xt, cond], 1), sigma=sigma)
        b, _ = net(torch.cat([5 * torch.randn_like(xt), cond], 1), sigma=3 * sigma)
        c, _ = net(cond)                                       # task: synthesis -- cond alone
        s_syn = net.last_S
    assert torch.equal(a, b), "bridge_fidelity=False must not depend on x_t or sigma"
    assert torch.allclose(a, c, atol=1e-6)
    assert s_syn.shape == xt.shape
    with torch.no_grad():                                      # and the bridge version DOES
        full = make(mode).eval()
        p, _ = full(torch.cat([xt, cond], 1), sigma=sigma)
        q, _ = full(torch.cat([xt + 1.0, cond], 1), sigma=sigma)
    assert not torch.allclose(p, q, atol=1e-4)


@pytest.mark.parametrize("make", [cdl, guided])
@pytest.mark.parametrize("mode", ["free", "gate"])
@pytest.mark.parametrize("bridge", [True, False])
def test_losses_reach_the_coupling(make, mode, bridge):
    net = make(mode, bridge_fidelity=bridge)
    xt, cond, sigma = batch()
    x0 = cond[:, 1:2] + torch.rand_like(xt)                    # CT1 = T1 + something positive
    pred, _ = net(torch.cat([xt, cond], 1), sigma=sigma)
    loss = ((pred - x0) ** 2).mean() + enhancement_loss(net, x0, cond[:, 1:2])
    loss.backward()
    grads = {n: p.grad for n, p in net.named_parameters() if "coupling" in n}
    assert grads, "no coupling parameters"
    if mode == "gate":
        assert float(grads["coupling.m"].abs().sum()) > 0
    else:
        assert float(grads["coupling.B_S.0.conv_real.weight"].abs().sum()) > 0   # the S readout
        assert float(grads["coupling.A_S.0.conv_real.weight"].abs().sum()) > 0
    assert float(net.a_xi.grad.abs().sum()) > 0
    assert (net.a_eta.grad is not None and float(net.a_eta.grad.abs().sum()) > 0) == bridge


def test_enhancement_loss():
    net = cdl("gate")
    xt, cond, sigma = batch()
    with pytest.raises(ValueError, match="enhancement map"):
        enhancement_loss(cdl(None), xt, xt)                    # plain net: never silently ignored
    net(torch.cat([xt, cond], 1), sigma=sigma)                 # S = 0 at init
    ct1, t1 = cond[:, 1:2] + 0.5, cond[:, 1:2]
    assert float(enhancement_loss(net, ct1, t1)) == pytest.approx(0.25, rel=1e-5)
    assert float(enhancement_loss(net, t1 - 0.5, t1)) == 0.0   # one-sided: CT1 < T1 targets 0
    assert float(enhancement_target(t1 - 1, t1).max()) == 0.0
    m = torch.zeros_like(xt); m[..., :4, :4] = 1
    big = t1.clone(); big[..., :4, :4] += 2.0
    assert float(enhancement_loss(net, big, t1, m, use_mask=True)) == pytest.approx(4.0, rel=1e-5)


@pytest.mark.parametrize("mode", ["free", "gate"])
def test_guided_net_takes_guides_either_way(mode):
    """`task: synthesis` hands the net [cond..., guides...]; that must equal guide= on the prox."""
    net = guided(mode, bridge_fidelity=False).eval()
    perturb(net)
    xt, cond, sigma = batch()
    g = torch.randn(2, 2, H, H)
    with torch.no_grad():
        a, _ = net(torch.cat([cond, g], 1))
        b, _ = net(cond, guide=g)
        none, _ = net(cond)
    assert torch.allclose(a, b, atol=1e-6)
    assert not torch.allclose(a, none, atol=1e-5), "the guides are not reaching the prox"
    with pytest.raises(ValueError, match="one way"):
        net(torch.cat([cond, g], 1), guide=g)


@pytest.mark.parametrize("mode", ["free", "gate"])
def test_guided_bridge_layer_runs_and_logs(mode):
    net = guided(mode)
    xt, cond, sigma = batch()
    out, _ = net(torch.cat([xt, cond], 1), sigma=sigma, guide=torch.randn(2, 1, 1, H, H))
    assert out.shape == xt.shape and net.last_S.shape == xt.shape
    net.project()
    assert any(k.startswith("xi.") for k in net.param_logs())
    logs = collapse_param_logs(net)
    assert ("gate_open" if mode == "gate" else "Dy_over_D_min") in logs
    assert {"step_xi_min", "step_eta_min", "step_nu_min"} <= set(logs)


def test_parameter_side_indicators_show_both_collapses():
    net = cdl("gate")
    assert collapse_param_logs(net)["gate_closed"] == 1.0          # m = 0 everywhere: S = 0
    with torch.no_grad():
        net.coupling.m.fill_(1.0)
    assert collapse_param_logs(net)["gate_open"] == 1.0            # T1 constrains nothing

    free = cdl("free")
    logs = collapse_param_logs(free)
    assert logs["DS_over_D_readout"] == 0.0 and logs["DS_over_D_mean"] == 0.0     # S = 0
    assert logs["Dy_over_D_min"] == pytest.approx(1.0, rel=1e-6)
    with torch.no_grad():                                          # ONE layer's D_y vanishes
        free.coupling.B_S[2].conv_real.weight.copy_(free.B_D[2].conv_real.weight)
    logs = collapse_param_logs(free)
    assert logs["Dy_over_D_min"] == pytest.approx(0.0, abs=1e-6), "a single dead layer must show"
    assert logs["Dy_over_D_mean"] == pytest.approx(2 / 3, rel=1e-5)

    assert collapse_param_logs(cdl(None)) == {}                    # plain net: nothing to report
    dc_only = cdl("gate", C=2, prior_idx=0, bridge_fidelity=False)
    assert "step_eta_min" not in collapse_param_logs(dc_only)      # no bridge term, no side term
    assert "step_nu_min" not in collapse_param_logs(dc_only)
    with torch.no_grad():
        dc_only.a_xi[1, 0] = -30.0                                 # one layer's DC switched off
    assert collapse_param_logs(dc_only)["step_xi_min"] < 1e-9


def test_data_side_indicators():
    """Each indicator moves the way its name says, on S maps planted by hand."""
    net = cdl("gate")
    xt, cond, sigma = batch()
    t1 = cond[:, 1:2]
    tgt = torch.rand_like(t1)                                      # true enhancement, >= 0
    ct1 = t1 + tgt
    net(torch.cat([xt, cond], 1), sigma=sigma)                     # sets last_density

    def run(S, x_hat):
        net.last_S = S
        m = CollapseMeter()
        m.add(net, x_hat, ct1, t1)
        m.add(net, x_hat, ct1, t1)                                 # pooled: two batches = one
        return m.result()

    perfect = run(tgt, ct1)
    assert perfect["S_rms_ratio"] == pytest.approx(1.0, rel=1e-5)
    assert perfect["S_neg_frac"] == 0.0 and perfect["T1_resid"] == pytest.approx(0.0, abs=1e-6)
    assert run(torch.zeros_like(tgt), ct1)["S_rms_ratio"] == 0.0   # S collapsed to zero
    assert run(-tgt, ct1)["S_neg_frac"] == pytest.approx(1.0)      # wrong sign entirely
    # the model's T1 is a constant (D_y z = 0): x_hat - S = mean(T1)
    flat = run(ct1 - t1.mean(dim=(1, 2, 3), keepdim=True), ct1)
    assert flat["T1_resid"] == pytest.approx(1.0, rel=1e-5)
    assert 0.0 <= perfect["code_density"] <= 1.0

    assert CollapseMeter().result() == {}                          # nothing added
    plain = CollapseMeter()
    plain.add(cdl(None), ct1, ct1, t1)                             # a net with no S: a no-op
    assert plain.result() == {}
    both = CollapseMeter()
    both.add(net, ct1, ct1, t1)
    assert "gate_open" in both.result(net) and "step_xi_min" in both.result(net)


def test_code_density_tracks_the_threshold():
    net = cdl("gate")
    xt, cond, sigma = batch()
    net(torch.cat([xt, cond], 1), sigma=sigma)
    dense = float(net.last_density)
    with torch.no_grad():
        net.t.fill_(1e6)                                           # threshold kills every atom
    out, _ = net(torch.cat([xt, cond], 1), sigma=sigma)
    assert dense > 0.5 and float(net.last_density) == 0.0
    dc = cond[:, 1:2].mean(dim=(1, 2, 3), keepdim=True)
    assert torch.allclose(out, dc.expand_as(out)), "a dead code outputs the DC and nothing else"


def test_panel():
    xt, cond, sigma = batch()
    t1 = cond[:, 1:2]
    rgb, cap = enhancement_panel(0.5 * torch.rand_like(t1), t1 + torch.rand_like(t1), t1)
    assert rgb.shape == (3, 3, H, H) and float(rgb.min()) >= 0 and float(rgb.max()) <= 1
    assert "S_hat" in cap and "rms" in cap
