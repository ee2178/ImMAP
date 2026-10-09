# -*- coding: utf-8 -*-
"""Training-speed changes to the guided group threshold nets: same numbers, fewer kernels.

Three changes, each pinned against what it replaced:

1. PROJECTION. `project()` of LGGCDL, LGGS, SBGuidedGroupCDL and SBCDLNet now projects every
   filter of one shape in a single stacked norm (models/base.py::batched_projection), visits
   each constrained module once, and clamps the attention transforms that `tie_attention`
   shares once per step instead of once per layer. The result must be EXACTLY what the old
   one-at-a-time projection gave, on parameters pushed well outside every constraint.

2. THE WINDOWED SIMILARITY of the gather backend (models/circulant_similarity.py) is computed
   for all window offsets in a few kernels instead of one python iteration per offset, and its
   constant index grids are built once. Values and gradients must match the loop.

3. THE JOINT SOFTMAX ON A FUSED BACKEND. A softmax over the union of the self window and the
   guide windows factorises through each branch's log-sum-exp (guided_prox.merge_joint), so
   flex / triton can run the joint simplex. The CPU box cannot run those kernels, so the
   identity is checked here with a stand-in branch that does what a fused branch does --
   an independent softmax plus its log-sum-exp -- against the concatenated gather softmax.
"""
import copy
import sys
import types

import pytest
import torch
import torch.nn.functional as F

sys.modules.setdefault("nibabel", types.ModuleType("nibabel"))

import models.base as base_mod                                    # noqa: E402
import models.circulant_similarity as cs                          # noqa: E402
import models.guided_prox as gp                                   # noqa: E402
import models.prox as prox_mod                                    # noqa: E402
from models.circulant_attention import Circulant                  # noqa: E402
from models.guided_cdl import GuidedGroupCDL                      # noqa: E402
from models.guided_lpds import LGGSNet                            # noqa: E402
from models.guided_prox import GuidedGroupThreshold, merge_joint  # noqa: E402
from models.prox import Polynomial                                # noqa: E402
from models.sb_cdlnet import SBCDLNet                             # noqa: E402
from models.sb_guided_groupcdl import SBGuidedGroupCDL            # noqa: E402

SCHED = dict(kind="brownian", tau=0.1, n_points=50, beta_max=0.3)
GUIDED = dict(M=8, P=3, s=2, Mh=4, window=1, guide_window=3, joint_softmax=True, dK=1,
              attn_backend="gather")


# =============================================================================================
# 1. projection
# =============================================================================================
def nets():
    torch.manual_seed(0)
    return {
        "LGGCDL, shared attention": GuidedGroupCDL(K=3, C=1, preproc="image", **GUIDED),
        "LGGCDL, untied attention": GuidedGroupCDL(K=3, C=1, preproc="image",
                                                   share_attention=False, **GUIDED),
        "LGGS, real": LGGSNet(K=3, C=1, is_complex=False, preproc="image", **GUIDED),
        "LGGS, complex": LGGSNet(K=3, C=1, is_complex=True, preproc="image", **GUIDED),
        "SB guided, free S + cross": SBGuidedGroupCDL(K=3, C=4, prior_idx=1, spectral_init=False,
                                                      s_mode="free", s_cross=True, **GUIDED, **SCHED),
        "SB guided, plain": SBGuidedGroupCDL(K=3, C=2, prior_idx=0, spectral_init=False,
                                             **GUIDED, **SCHED),
        "SBCDLNet, free S": SBCDLNet(K=3, M=8, P=3, s=2, C=4, prior_idx=1, init=False,
                                     complex=False, s_mode="free", **SCHED),
        "SBCDLNet, plain": SBCDLNet(K=3, M=8, P=3, s=2, C=3, prior_idx=0, init=False,
                                    complex=False, **SCHED),
    }


def violate(net, seed=1):
    """Push every parameter outside its constraint: big, and of both signs."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in net.parameters():
            p.mul_(3.0).add_(torch.randn(p.shape, generator=g) * 2.0)


def unbatched_project(net):
    """What project() did before: every constraint applied on the spot, every owner visited.

    Outside `batched_projection` a `project_conv` is immediate and `project_once` is always
    True, so calling the same `project_` methods here IS the old one-at-a-time behaviour;
    LGGS additionally used to walk every module."""
    assert base_mod._BATCH is None
    with torch.no_grad():
        if isinstance(net, LGGSNet):
            for m in net.modules():
                if m is not net and hasattr(m, "project_"):
                    m.project_()
        elif isinstance(net, SBCDLNet):
            net.t.clamp_(0.0)
            for A, B in ((net.A_D, net.B_D), (net.A_P, net.B_P)):
                for a, b in zip(A, B):
                    base_mod.project_conv(a)
                    base_mod.project_conv(b)
            if net.coupling is not None:
                net.coupling.project()
        elif isinstance(net, SBGuidedGroupCDL):
            if net.coupling is not None:
                net.coupling.project()
            if net.s_cross:
                net.cross_gain.clamp_(0.0)
            for layer in net.layers:
                layer.project_()
                layer.prox.tau.weight.data[0].clamp_(min=net.t_floor)
            for a, b in zip(net.A_P, net.B_P):
                base_mod.project_conv(a)
                base_mod.project_conv(b)
        else:
            for layer in net.layers:
                layer.project_()
                if net.tau_floor is not None:
                    layer.prox.tau.weight.data[0].clamp_(min=net.tau_floor)
            base_mod.project_conv(net.D)


@pytest.mark.parametrize("name", list(nets()))
@pytest.mark.parametrize("foreach", [True, False])
def test_projection_is_exactly_the_one_at_a_time_projection(name, foreach, monkeypatch):
    net = nets()[name]
    violate(net)
    ref = copy.deepcopy(net)
    unbatched_project(ref)
    monkeypatch.setattr(base_mod, "_FOREACH_LIST", foreach and base_mod._FOREACH_LIST)
    net.project()
    moved = 0
    before = dict(nets()[name].named_parameters())
    for (n, p), (_, q) in zip(net.named_parameters(), ref.named_parameters()):
        assert torch.equal(p, q), f"{name}: {n} differs from the unbatched projection"
        moved += int(p.shape == before[n].shape)
    assert moved and base_mod._BATCH is None and base_mod._ONCE is None
    # and it really projected: the layers' filters are inside the unit ball, having started
    # far outside it. (Not checked: bit-idempotence. A unit-ball projection rescales by
    # clamp(1 / norm, max=1), and a norm of 1 - 1ulp moves the weight again -- before and after
    # this change alike.)
    layers = list(getattr(net, "layers", [])) or list(net.A_D) + list(net.B_D)
    convs = [m for l in layers for m in l.modules() if hasattr(m, "conv_real")]
    assert convs
    for m in convs:
        w = m.weight
        assert float(w.abs().pow(2).sum(dim=(2, 3)).sqrt().max()) <= 1.0 + 1e-5


def test_shared_attention_is_clamped_once_and_filters_are_batched(monkeypatch):
    K = 4
    net = GuidedGroupCDL(K=K, C=1, preproc="image", **GUIDED)
    assert net.layers[1].prox.gamma is net.layers[0].prox.gamma        # tie_attention
    violate(net)

    calls = {"poly": [], "flush": []}
    orig_poly = Polynomial.project_
    monkeypatch.setattr(Polynomial, "project_",
                        lambda self, *a, **k: (calls["poly"].append(id(self)), orig_poly(self, *a, **k))[1])
    orig_flush = base_mod._flush_projection
    monkeypatch.setattr(base_mod, "_flush_projection",
                        lambda pending: (calls["flush"].append(len(pending)), orig_flush(pending))[1])
    net.project()

    gamma = id(net.layers[0].prox.gamma)
    assert calls["poly"].count(gamma) == 1, "the shared gamma must be clamped once, not K times"
    per_layer = [id(l.prox.tau) for l in net.layers] + [id(l.prox.rho) for l in net.layers]
    assert all(calls["poly"].count(i) == 1 for i in per_layer)
    # one flush for the whole net, holding every filter: K analysis + K synthesis + the readout
    assert calls["flush"] == [2 * K + 1], calls["flush"]


def test_project_once_is_a_no_op_guard_outside_the_context():
    obj = object()
    assert base_mod.project_once(obj) and base_mod.project_once(obj)   # always True outside
    with base_mod.batched_projection():
        assert base_mod.project_once(obj) and not base_mod.project_once(obj)
        with base_mod.batched_projection():                            # nested: same bookkeeping
            assert not base_mod.project_once(obj)
    assert base_mod.project_once(obj)


# =============================================================================================
# 2. the windowed similarity
# =============================================================================================
def loop_reference(sim, x, y, win, monkeypatch):
    monkeypatch.setattr(cs, "VECTORIZED", False)
    out = cs.circulant_similarity_window(sim, x, y, win)
    monkeypatch.setattr(cs, "VECTORIZED", True)
    return out


@pytest.mark.parametrize("sim", ["distance", "realdot", "pidot", "pidistance", "dot"])
@pytest.mark.parametrize("cplx", [False, True])
@pytest.mark.parametrize("hw_win", [(9, 11, 3), (16, 12, 5), (20, 20, 15)])
@pytest.mark.parametrize("chunk", [2 ** 40, 1])
def test_vectorised_similarity_matches_the_loop(sim, cplx, hw_win, chunk, monkeypatch):
    H, W, win = hw_win
    torch.manual_seed(0)
    dt = torch.complex128 if cplx else torch.float64
    x = torch.randn(2, 5, H, W, dtype=dt, requires_grad=True)
    y = torch.randn(2, 5, H, W, dtype=dt, requires_grad=True)
    monkeypatch.setattr(cs, "CHUNK_BYTES", chunk)          # one block / one offset row at a time
    v, col, crow = cs.circulant_similarity_window(sim, x, y, win)
    r, col_r, crow_r = loop_reference(sim, x, y, win, monkeypatch)
    assert v.shape == r.shape == (2, H * W, win * win) and v.dtype == r.dtype
    assert torch.equal(col, col_r) and torch.equal(crow, crow_r)
    assert torch.allclose(v, r, rtol=1e-12, atol=1e-12)
    if not v.is_complex():
        gv = torch.autograd.grad((v ** 2).sum(), (x, y))
        gr = torch.autograd.grad((r ** 2).sum(), (x, y))
        for a, b in zip(gv, gr):
            assert torch.allclose(a, b, rtol=1e-10, atol=1e-10)


def test_similarity_in_float32_and_offset_order(monkeypatch):
    torch.manual_seed(0)
    x, y = torch.randn(3, 8, 16, 16), torch.randn(3, 8, 16, 16)
    v, col, _ = cs.circulant_similarity_window("distance", x, y, 7)
    r, _, _ = loop_reference("distance", x, y, 7, monkeypatch)
    assert float((v - r).abs().max() / r.abs().max()) < 1e-5
    # slot k really is offset k: the query's own position (zero offset) is the centre slot
    centre = (7 * 7) // 2
    assert torch.equal(col[:, centre], torch.arange(16 * 16))
    same, _, _ = cs.circulant_similarity_window("distance", x, x, 7)
    assert float(same[..., centre].abs().max()) == 0.0     # distance to itself


def test_index_grids_are_built_once_and_shared():
    cs._INDEX_CACHE.clear()
    x = torch.randn(1, 2, 8, 8)
    _, col_a, crow_a = cs.circulant_similarity_window("distance", x, x, 3)
    _, col_b, crow_b = cs.circulant_similarity_window("realdot", x, x, 3)
    assert col_a is col_b and crow_a is crow_b and len(cs._INDEX_CACHE) == 1
    cs.circulant_similarity_window("distance", x, x, 5)
    cs.circulant_similarity_window("distance", torch.randn(1, 2, 8, 6), torch.randn(1, 2, 8, 6), 3)
    assert len(cs._INDEX_CACHE) == 3


def test_fallbacks_keep_the_loop(monkeypatch):
    """A window wider than the grid (roll wraps more than once), a 1-D grid, a custom
    similarity: none of these take the vectorised path, and all still work."""
    used = []
    orig = cs._window_values_2d
    monkeypatch.setattr(cs, "_window_values_2d", lambda *a, **k: (used.append(1), orig(*a, **k))[1])
    x = torch.randn(1, 3, 4, 4)
    v, _, _ = cs.circulant_similarity_window("distance", x, x, 9)       # p = 4 >= grid
    assert not used and v.shape == (1, 16, 81)
    x1 = torch.randn(1, 3, 12)
    v1, _, _ = cs.circulant_similarity_window("distance", x1, x1, 3)
    assert not used and v1.shape == (1, 12, 3)
    custom = lambda a, b: (a * b).sum(dim=1)                             # noqa: E731
    v2, _, _ = cs.circulant_similarity_window(custom, x, x, 3)
    assert not used and v2.shape == (1, 16, 9)
    cs.circulant_similarity_window("distance", x, x, 3)
    assert used == [1]


def test_guided_prox_output_is_unchanged_by_the_vectorised_build(monkeypatch):
    torch.manual_seed(0)
    p = GuidedGroupThreshold(8, Mh=4, window=1, guide_window=5, joint_softmax=True, dK=1,
                             attn_backend="gather", tau0=0.05).double()
    z, g = torch.randn(2, 8, 12, 10, dtype=torch.float64), torch.randn(2, 8, 12, 10, dtype=torch.float64)
    out, _ = p(z, [g], None, None)
    monkeypatch.setattr(cs, "VECTORIZED", False)
    ref, _ = p(z, [g], None, None)
    assert torch.allclose(out, ref, rtol=1e-12, atol=1e-13)


# =============================================================================================
# 3. the joint softmax through log-sum-exps
# =============================================================================================
@pytest.mark.parametrize("heads", [1, 2])
@pytest.mark.parametrize("n_branch", [2, 3])
def test_merge_joint_is_the_concatenated_softmax(heads, n_branch):
    torch.manual_seed(0)
    B, C, H, W = 2, 4, 5, 6
    S = H * W
    Ks = [1, 9, 25][:n_branch]
    sims = [3 * torch.randn(B * heads, S, k, dtype=torch.float64) for k in Ks]     # raw scores
    vals = [torch.rand(B * heads, C // heads, S, k, dtype=torch.float64) for k in Ks]

    joint = torch.softmax(torch.cat(sims, dim=-1), dim=-1).split(Ks, dim=-1)
    want = sum((w.unsqueeze(1) * v).sum(-1) for w, v in zip(joint, vals))           # (B*h, C/h, S)
    want = want.reshape(B, C, H, W)

    outs = [(torch.softmax(s, -1).unsqueeze(1) * v).sum(-1).reshape(B, C, H, W)
            for s, v in zip(sims, vals)]
    lses = [torch.logsumexp(s, -1) for s in sims]                                   # (B*h, S)
    assert torch.allclose(merge_joint(outs, lses, heads), want, rtol=1e-12, atol=1e-14)
    # the other layout a backend may hand back, and a per-query constant missing from the scores
    lses3 = [(l + 7.0).reshape(B, heads, S) for l in lses]
    assert torch.allclose(merge_joint(outs, lses3, heads), want, rtol=1e-12, atol=1e-14)


class FakeFused:
    """What a fused branch provides: an independently softmaxed attention and its lse.
    Built on the gather machinery so it runs on the CPU. NOT a Circulant, so the prox treats
    it as it would flex / triton (no blend, re-cached on refresh)."""

    def __init__(self, prox, q, k, win, lse_shift=0.0):
        s, col, crow = cs.circulant_similarity_window(prox.sim_fun, prox._to_heads(q),
                                                      prox._to_heads(k), win)
        self.c = Circulant(F.softmax(s, dim=-1), col, crow, tuple(q.shape[-2:]), win)
        self._lse = torch.logsumexp(s, dim=-1) + lse_shift

    def apply(self, x, transpose=False):
        return self.c.apply(x, transpose=transpose)


def fused_prox(heads=1, lse_shift=None, **kw):
    """A joint-softmax prox on a 'fused' backend whose branches are FakeFused."""
    torch.manual_seed(0)
    args = dict(Mh=4, nheads=heads, window=1, guide_window=5, joint_softmax=True, dK=1,
                sim_fun="distance", tau0=0.05)
    args.update(kw)
    fused = GuidedGroupThreshold(8, attn_backend="flex", **args).double()
    gather = GuidedGroupThreshold(8, attn_backend="gather", **args).double()
    gather.load_state_dict(fused.state_dict())
    shift = iter(lse_shift or [])
    fused._fused_branch = lambda q, k, win: FakeFused(fused, q, k, win, next(shift, 0.0))
    return fused, gather


@pytest.mark.parametrize("heads", [1, 2])
@pytest.mark.parametrize("n_guides", [1, 3])
def test_fused_joint_prox_equals_the_gather_prox(heads, n_guides):
    gp._FUSED_JOINT_CHECKED.clear()
    fused, gather = fused_prox(heads)
    assert fused.joint_fused and not gather.joint_fused
    z = torch.randn(2, 8, 12, 10, dtype=torch.float64, requires_grad=True)
    guides = [torch.randn(2, 8, 12, 10, dtype=torch.float64) for _ in range(n_guides)]
    a, _ = fused(z, guides, None, None)
    b, _ = gather(z, guides, None, None)
    assert torch.allclose(a, b, rtol=1e-10, atol=1e-12)
    ga, = torch.autograd.grad(a.pow(2).sum(), z, retain_graph=True)
    gb, = torch.autograd.grad(b.pow(2).sum(), z)
    assert torch.allclose(ga, gb, rtol=1e-8, atol=1e-10), "the gradient must flow through the lse"
    # the attention weights get gradient through BOTH the outputs and the merge
    a.pow(2).sum().backward()
    assert float(fused.Wtheta.weight.grad.abs().sum()) > 0
    assert len(gp._FUSED_JOINT_CHECKED) == 1                   # the self-check ran, once


def test_self_check_catches_a_wrong_log_sum_exp():
    gp._FUSED_JOINT_CHECKED.clear()
    fused, _ = fused_prox()
    err = fused._check_fused_joint(torch.zeros(1, dtype=torch.float64))
    assert err is not None and err < 1e-10
    assert fused._check_fused_joint(torch.zeros(1, dtype=torch.float64)) is None     # once per process

    gp._FUSED_JOINT_CHECKED.clear()
    bad, _ = fused_prox(lse_shift=[0.0, 2.0])                  # the guide branch's lse is off by 2
    with pytest.raises(RuntimeError, match="disagrees with the gather backend"):
        bad._check_fused_joint(torch.zeros(1, dtype=torch.float64))


def test_fused_joint_without_guides_and_with_a_cross_branch():
    gp._FUSED_JOINT_CHECKED.clear()
    fused, gather = fused_prox()
    z = torch.randn(1, 8, 12, 10, dtype=torch.float64)
    g = torch.randn(1, 8, 12, 10, dtype=torch.float64)
    cross = lambda: {"query": lambda: 0.5 * z, "guide": g, "weight": torch.tensor(0.7)}   # noqa: E731
    a, _ = fused(z, [g], None, None, cross=cross())
    b, _ = gather(z, [g], None, None, cross=cross())
    assert torch.allclose(a, b, rtol=1e-10, atol=1e-12)
    a0, _ = fused(z, None, None, None, cross=cross())          # no ordinary guide: nothing to merge
    b0, _ = gather(z, None, None, None, cross=cross())
    assert torch.allclose(a0, b0, rtol=1e-10, atol=1e-12)


def test_flex_branch_asks_for_its_own_window(monkeypatch):
    """The flex guide branch used to be given the SELF window's block mask."""
    asked = []
    monkeypatch.setattr(prox_mod, "get_block_mask",
                        lambda H, W, win, device, **k: asked.append(win) or ("mask", win))
    p = GuidedGroupThreshold(8, Mh=4, window=1, guide_window=15, joint_softmax=False,
                             attn_backend="flex", sim_fun="distance")
    ref = torch.zeros(1, 4, 16, 16)
    assert p._flex_block_mask(ref) == ("mask", 1)
    assert p._flex_block_mask(ref, 15) == ("mask", 15)
    seen = {}
    monkeypatch.setattr(gp, "FlexAdjacency",
                        lambda q, k, win, **kw: seen.update(win=win, mask=kw["block_mask"]) or "adj")
    p._fused_branch(ref, ref, p.guide_window)
    assert seen == {"win": 15, "mask": ("mask", 15)}, seen
