# -*- coding: utf-8 -*-
"""Two more same-numbers speedups for the +S unrolled nets.

1. THE FIDELITY STEP AS TWO CONVS (models/fused_fidelity.py). A layer's measurement, bridge and
   side-contrast gradients go through one stacked synthesis and one stacked analysis instead of
   eight convolutions. Output, enhancement map and EVERY parameter gradient must match the
   module-by-module path, for free / gate, with and without the bridge term and side channels,
   in both nets. `shrink` must be components.ST, gradients included.

2. A ONE-NEIGHBOUR WINDOW WITHOUT THE ATTENTION KERNEL (circulant_flex.PointwiseAdjacency).
   The guided nets run `window: 1` for the self branch. Its adjacency is the identity and its
   log-sum-exp is the score at the pixel itself. The flex kernel cannot run on the CPU box, so
   the joint energy built from it is checked against the concatenated gather softmax, with a
   stand-in guide branch in the fused convention (scores without the per-query -1/2||q||^2).
"""
import sys
import types

import pytest
import torch
import torch.nn.functional as F

sys.modules.setdefault("nibabel", types.ModuleType("nibabel"))

import models.circulant_flex as cf                                 # noqa: E402
import models.fused_fidelity as ff                                 # noqa: E402
from models.circulant_attention import Circulant, _abs2            # noqa: E402
from models.circulant_similarity import circulant_similarity_window  # noqa: E402
from models.components import ST                                   # noqa: E402
from models.guided_prox import GuidedGroupThreshold, merge_joint   # noqa: E402
from models.sb_cdlnet import SBCDLNet                              # noqa: E402
from models.sb_guided_groupcdl import SBGuidedGroupCDL             # noqa: E402

SCHED = dict(kind="brownian", tau=0.1, n_points=50, beta_max=0.3)
H = 16


def perturb(net, seed=1):
    """Move every parameter off its init (D_S starts at exactly 0, the gate at 0)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in net.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g))
        net.project()
    return net


def run(net, y, fused, **kw):
    old = ff.ENABLED
    ff.ENABLED = fused
    try:
        net.zero_grad()
        x, z = net(y, **kw)
        (x.pow(2).mean() + net.last_S.pow(2).mean() + 1e-3 * z.abs().mean()).backward()
        grads = {n: p.grad.clone() for n, p in net.named_parameters() if p.grad is not None}
        return x.detach(), net.last_S.detach(), grads
    finally:
        ff.ENABLED = old


def same(a, b, tol=2e-5):
    scale = max(float(b.abs().max()), 1e-12)
    return float((a - b).abs().max()) / scale < tol


def compare(net, y, **kw):
    x1, s1, g1 = run(net, y, True, **kw)
    x0, s0, g0 = run(net, y, False, **kw)
    assert same(x1, x0) and same(s1, s0), "fused forward differs"
    assert set(g1) == set(g0), set(g1) ^ set(g0)
    # relative to each parameter's own gradient, with a floor set by the largest gradient in the
    # net: a gradient that is ~1e-6 of the others is rounding, and differs in its 4th digit
    top = max(float(g.abs().max()) for g in g0.values())
    assert top > 0
    for n in g0:
        err = float((g1[n] - g0[n]).abs().max())
        assert err <= 2e-4 * float(g0[n].abs().max()) + 1e-6 * top, f"gradient of {n} differs"


@pytest.mark.parametrize("s_mode", ["free", "gate"])
@pytest.mark.parametrize("bridge", [True, False])
@pytest.mark.parametrize("C", [4, 2])                    # with and without side channels
def test_sbcdlnet_fused_fidelity_is_the_same_map(s_mode, bridge, C):
    torch.manual_seed(0)
    net = perturb(SBCDLNet(K=3, M=8, P=3, s=2, C=C, prior_idx=1 if C == 4 else 0, init=False,
                           deg_eta=1, deg_tau=1, t0=0.01, s_mode=s_mode, bridge_fidelity=bridge,
                           gate_init=0.3, **SCHED))
    y = torch.rand(3, C, H, H)
    compare(net, y, step=torch.tensor([3, 20, 45]))
    if not bridge:                                       # the `task: synthesis` call
        compare(net, y[:, 1:])


@pytest.mark.parametrize("s_mode,cross", [("free", False), ("gate", False), ("free", True)])
@pytest.mark.parametrize("bridge", [True, False])
def test_guided_net_fused_fidelity_is_the_same_map(s_mode, cross, bridge):
    torch.manual_seed(0)
    net = perturb(SBGuidedGroupCDL(K=3, M=8, P=3, s=2, Mh=4, C=4, prior_idx=1, window=1,
                                   guide_window=3, dK=2, attn_backend="gather",
                                   spectral_init=False, t0=0.01, s_mode=s_mode, s_cross=cross,
                                   bridge_fidelity=bridge, gate_init=0.3, **SCHED))
    y = torch.rand(2, 4, H, H)
    guide = torch.rand(2, 2, H, H)
    compare(net, y, step=torch.tensor([5, 40]), guide=guide)


def test_complex_net_keeps_the_module_path():
    torch.manual_seed(0)
    net = SBCDLNet(K=2, M=4, P=3, s=2, C=4, prior_idx=1, init=False, complex=True,
                   s_mode="free", **SCHED)
    assert not ff.fusable(net.A_D[0], net.B_D[0], *net.coupling.convs())
    x, _ = net(torch.rand(1, 4, H, H), step=torch.tensor([10]))
    assert torch.isfinite(torch.view_as_real(x) if x.is_complex() else x).all()


def test_shrink_is_soft_thresholding_with_the_same_gradients():
    torch.manual_seed(0)
    u = torch.randn(2, 5, 6, 6, requires_grad=True)
    tau = torch.rand(2, 5, 1, 1) * 0.8
    tau[:, 0] = 0.0                  # where these nets START (t0 = 0): clamp's own derivative
    tau.requires_grad_(True)         # is not ST's there, which is why shrink writes it out
    with torch.no_grad():
        u[0, 1, 0, 0] = 0.0                               # sgn(0) = 0
        u[0, 2, 0, :2] = torch.stack([tau[0, 2, 0, 0], -tau[0, 2, 0, 0]])     # |u| = tau
    w = torch.randn(2, 5, 6, 6)
    outs = []
    for fn in (ff.shrink, ST):
        out = fn(u, tau)
        gu, gt = torch.autograd.grad((out * w).sum(), (u, tau))
        outs.append((out.detach(), gu, gt))
    for a, b in zip(*outs):
        assert torch.allclose(a, b, atol=1e-6)
    assert float((outs[0][0] == 0).float().mean()) > 0.2          # the threshold is doing work


# =============================================================================================
# 2. a window of one neighbour
# =============================================================================================
@pytest.mark.parametrize("sim", ["distance", "pidistance"])
@pytest.mark.parametrize("nheads", [1, 2])
def test_pointwise_self_branch_gives_the_gather_joint_energy(sim, nheads):
    torch.manual_seed(0)
    B, M, Mh, n, gw = 2, 8, 4, 9, 5
    prox = GuidedGroupThreshold(M, Mh=Mh, nheads=nheads, window=1, guide_window=gw,
                                joint_softmax=True, sim_fun=sim, attn_backend="gather", dK=1)
    z, g = torch.randn(B, M, n, n), torch.randn(B, M, n, n)
    q, k_self, k_guides = prox._project_qk(z, [g], None)
    v_self, v_guide = (_abs2(prox.Walpha(t)) for t in (z, g))

    # what gather computes: one simplex over the self pixel and the guide window
    Phi, Omegas = prox._gather_adjacencies(q, k_self, k_guides, (n, n))
    want = prox.apply_gamma(Phi, v_self) + prox.apply_gamma(Omegas[0], v_guide)

    # the fused construction: the pointwise self branch + an independently-softmaxed guide
    # branch with its log-sum-exp in the fused convention (per-query -1/2||q||^2 left out)
    self_branch = cf.PointwiseAdjacency(q, k_self, sim=sim, heads=nheads)
    assert prox.apply_gamma(self_branch, v_self) is v_self
    qh = prox._to_heads(q)
    s, col, crow = circulant_similarity_window(sim, qh, prox._to_heads(k_guides[0]), gw)
    qsq = 0.5 * (qh * qh).sum(dim=1).flatten(1)                    # (B*h, S)
    guide_branch = Circulant(F.softmax(s, dim=-1), col, crow, (n, n), gw)
    lse_guide = torch.logsumexp(s, dim=-1) + qsq
    got = merge_joint([v_self, prox.apply_gamma(guide_branch, v_guide)],
                      [self_branch._lse, lse_guide], nheads)
    assert same(got, want, 1e-5)

    # and the one-neighbour score itself, against gather's 1 x 1 window
    s1, _, _ = circulant_similarity_window(sim, qh, prox._to_heads(k_self), 1)
    assert torch.allclose(self_branch._lse.reshape(B * nheads, -1), s1[..., 0] + qsq, atol=1e-5)


def test_only_a_one_neighbour_flex_window_takes_the_shortcut():
    prox = GuidedGroupThreshold(8, Mh=4, window=1, guide_window=5, joint_softmax=True,
                                attn_backend="gather", dK=1)
    prox.attn_backend = "flex"                           # the branch constructor only
    q = torch.randn(1, 4, 6, 6)
    assert isinstance(prox._fused_branch(q, q, 1), cf.PointwiseAdjacency)
    old = cf.POINTWISE_SELF
    cf.POINTWISE_SELF = False
    try:
        with pytest.raises(Exception):                   # no flex kernel on the CPU box
            prox._fused_branch(q, q, 1)
    finally:
        cf.POINTWISE_SELF = old
