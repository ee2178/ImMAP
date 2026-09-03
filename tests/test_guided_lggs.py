"""
LGGS: the guided group threshold and the guided primal-dual net it sits in.

Run with `python -m tests.test_guided_lggs`.

What this pins down
-------------------
1. the joint softmax really is ONE simplex over the self window AND every guide
   window -- the property that makes `omega` unnecessary in that mode;
2. the gather adjacency matches a brute-force circular-window reference, for
   both windows independently (self = 1x1, guide = 3x3 is the LGGS shape);
3. the guide changes the answer, and can only ever SHRINK it: the prox is
   `z * relu(1 - tau/xi)`, a gain in [0, 1], so the guide steers the grouping
   and never contributes intensities of its own;
4. `guide=None` falls back to the unguided `GroupThreshold` exactly;
5. `LGGSNet` runs forward and backward, and `build_model` builds it.
"""

import itertools

import torch
import torch.nn.functional as F

from models import build_model
from models.circulant_attention import _abs2
from models.guided_lpds import GuidedLPDSLayer, LGGSNet
from models.guided_prox import GuidedGroupThreshold
from models.prox import GroupThreshold

PASS, FAIL = [], []

PROX = dict(Mh=6, nheads=1, window=1, guide_window=3, sim_fun="distance",
            dK=1, init_strategy="semi_orthogonal", attn_backend="gather")


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def prox(**over):
    torch.manual_seed(0)
    return GuidedGroupThreshold(12, **dict(PROX, **over))


def rand(B=2, C=12, H=6, W=6, complex_=False):
    if complex_:
        return torch.randn(B, C, H, W, dtype=torch.complex64)
    return torch.randn(B, C, H, W)


# ---------------------------------------------------------------------------
def test_joint_softmax_is_one_simplex():
    print("\n[joint softmax normalises self + guides together]")
    p = prox()
    z, v = rand(), rand()
    Phi, Om, _ = p.adjacencies_of(z, [v], None, {})
    total = Phi.values.sum(-1) + Om[0].values.sum(-1)
    check("rows of [Phi | Omega] sum to 1", torch.allclose(total, torch.ones_like(total), atol=1e-5),
          f"max |sum - 1| = {(total - 1).abs().max():.2e}")
    check("neither branch is row-stochastic on its own",
          not torch.allclose(Phi.values.sum(-1), torch.ones_like(total), atol=1e-3),
          f"Phi row sums in [{Phi.values.sum(-1).min():.3f}, {Phi.values.sum(-1).max():.3f}]")

    q = prox(joint_softmax=False)
    Phi, Om, _ = q.adjacencies_of(z, [v], None, {})
    ok = (torch.allclose(Phi.values.sum(-1), torch.ones_like(total), atol=1e-5)
          and torch.allclose(Om[0].values.sum(-1), torch.ones_like(total), atol=1e-5))
    check("joint_softmax=False: each branch row-stochastic on its own", ok)
    check("joint_softmax=False installs the learned blend omega", q.omega is not None)
    check("joint_softmax=True does not", prox().omega is None)


# ---------------------------------------------------------------------------
def _reference_energy(p, z, v):
    """Brute-force xi_a: explicit circular windows, explicit joint softmax."""
    B, _, H, W = z.shape
    rho = p.rho(None, ref=z)
    sq = torch.sqrt(rho + p.eps)
    q = p.Wtheta(z) / sq
    kz = p.Wphi(z) / sq
    kv = p.Wphi(v) / sq
    az, av = _abs2(p.Walpha(z)), _abs2(p.Walpha(v))

    def offs(win):
        r = range(-(win // 2), win // 2 + 1)
        return list(itertools.product(r, r))

    o_self, o_guide = offs(p.window), offs(p.guide_window)
    out = torch.zeros(B, p.Mh, H, W)
    for b, i, j in itertools.product(range(B), range(H), range(W)):
        qi = q[b, :, i, j]
        scores, vals = [], []
        for (dy, dx) in o_self:
            kk = kz[b, :, (i + dy) % H, (j + dx) % W]
            scores.append(-0.5 * (qi - kk).pow(2).sum())
            vals.append(az[b, :, (i + dy) % H, (j + dx) % W])
        for (dy, dx) in o_guide:
            kk = kv[b, :, (i + dy) % H, (j + dx) % W]
            scores.append(-0.5 * (qi - kk).pow(2).sum())
            vals.append(av[b, :, (i + dy) % H, (j + dx) % W])
        w = F.softmax(torch.stack(scores), dim=0)
        out[b, :, i, j] = (w[:, None] * torch.stack(vals)).sum(0)
    return torch.sqrt(out + p.eps)


def test_gather_matches_bruteforce():
    print("\n[the gather adjacency is the windowed similarity it claims to be]")
    p = prox()
    z, v = rand(B=1, H=5, W=5), rand(B=1, H=5, W=5)
    with torch.no_grad():
        Phi, Om, _ = p.adjacencies_of(z, [v], None, {})
        e = p.apply_gamma(Phi, _abs2(p.Walpha(z))) \
            + p.apply_gamma(Om[0], _abs2(p.Walpha(v)))
        got = torch.sqrt(e + p.eps)
        want = _reference_energy(p, z, v)
    err = (got - want).abs().max()
    check("xi_a matches the explicit circular-window reference", err < 1e-5,
          f"max abs err = {err:.2e}")


# ---------------------------------------------------------------------------
def test_guide_enters_only_through_the_adjacency():
    print("\n[the guide steers the grouping, never the intensities]")
    p = prox(window=3)
    z, v = rand(), rand()
    with torch.no_grad():
        out_a, _ = p(z, v, None, {})
        out_b, _ = p(z, v.flip(-1), None, {})
        out_none, _ = p(z, None, None, {})
    check("a permuted guide changes the estimate",
          not torch.allclose(out_a, out_b, atol=1e-6),
          f"max delta = {(out_a - out_b).abs().max():.3e}")
    check("a guide changes the estimate at all",
          not torch.allclose(out_a, out_none, atol=1e-6))

    # The output is z * relu(1 - tau/xi): a real, non-negative, elementwise
    # gain. Whatever the guide does, it can only shrink z towards 0 -- it can
    # never add signal of its own, which is the property the docstring claims.
    gain = torch.where(z.abs() > 0, out_a / z, torch.ones_like(z))
    check("the guide can only shrink: 0 <= gain <= 1",
          bool((gain >= -1e-6).all() and (gain <= 1 + 1e-6).all()),
          f"gain in [{gain.min():.3f}, {gain.max():.3f}]")


def test_unguided_fallback_is_exactly_group_threshold():
    print("\n[guide=None is the self-only view, sharing every weight]")
    p = prox(window=3)
    z = rand()
    with torch.no_grad():
        got, _ = p(z, None, None, {})
        want, _ = GroupThreshold.forward(p, z, None, {})
    check("guide=None == GroupThreshold.forward", torch.allclose(got, want, atol=0),
          "bit-identical")
    check("it is a GroupThreshold", isinstance(p, GroupThreshold))


def test_complex_and_projection():
    print("\n[complex features, and the constraint projection]")
    p = prox(sim_fun="pidistance")
    z, v = rand(complex_=True), rand(complex_=True)
    out, _ = p(z, v, None, {})
    check("complex latent + pidistance runs", out.is_complex() and out.shape == z.shape)

    q = prox(joint_softmax=False)
    with torch.no_grad():
        q.omega.weight.fill_(3.0)
        q.tau.weight.fill_(-1.0)
        q.Wbeta.weight.fill_(-1.0)
        q.project_()
    check("omega clamped to [0.05, 0.95]", float(q.omega.weight.max()) <= 0.95 + 1e-9)
    check("tau clamped >= 0", float(q.tau.weight.min()) >= 0.0)
    check("W_beta clamped >= 0", float(q.Wbeta.weight.min()) >= 0.0)


def test_joint_softmax_rejects_fused_backends():
    print("\n[a joint simplex cannot be a fused kernel]")
    try:
        prox(attn_backend="flex")
        check("joint_softmax + flex raises", False)
    except ValueError as e:
        check("joint_softmax + flex raises", "gather" in str(e))


# ---------------------------------------------------------------------------
def test_layer_uses_its_own_dictionary_for_the_guide():
    print("\n[the guide latent is v = A w, per layer]")
    torch.manual_seed(0)
    layer = GuidedLPDSLayer(1, 12, P=3, stride=1, is_complex=False, Mh=6,
                            window=1, guide_window=3,
                            prox_kws=dict(sim_fun="distance", dK=1,
                                          init_strategy="semi_orthogonal"))
    w = torch.randn(2, 1, 8, 8)
    v = layer.analyse_guides([w])[0]
    check("A w has the latent shape", v.shape == (2, 12, 8, 8), str(tuple(v.shape)))
    check("A w equals the layer's analysis of w",
          torch.allclose(v, layer.analysis(w), atol=0))


def test_net_forward_backward():
    print("\n[LGGSNet forward / backward]")
    torch.manual_seed(0)
    net = LGGSNet(K=3, M=16, C=1, P=3, s=2, Mh=8, window=1, guide_window=3,
                  sim_fun="distance", dK=2, is_complex=False,
                  preproc="identity", spectral_init=False)
    y = torch.randn(1, 1, 16, 16)
    w = torch.randn(1, 2, 1, 16, 16)          # two guides, stacked (B, G, C, H, W)
    x, (xp, z) = net(y, guide=w)
    check("x_hat keeps the input grid", x.shape == y.shape, str(tuple(x.shape)))
    check("latent is the primal-dual pair", z.shape == (1, 16, 8, 8), str(tuple(z.shape)))

    (x.square().mean() + z.square().mean()).backward()

    # Which parameters legitimately receive NO gradient is a statement about
    # the unrolling, so spell it out rather than asserting "all of them":
    #   * layer 0 is the COLD START (`state is None`), which sets x = y~ and
    #     runs only analysis + prox -- no synthesis, no tau, no theta.  Julia's
    #     cold-start method even evaluates tau and discards it, so this matches.
    #   * a prox's `gamma` is the Alg.-4 blend weight, used only on a rebuild;
    #     layer 0's prox is called once, so its gamma never blends.
    #   * dK=2 with K=3 means the adjacency is BUILT at layer 0, REBUILT (and
    #     blended) at layer 1, and REUSED at layer 2 -- so layer 2's similarity
    #     transforms never run.  That is Alg. 4 doing its job; if this set ever
    #     shrinks, the adjacency is being rebuilt more often than dK asks for.
    expected_dry = {"layers.0.synthesis", "layers.0.tau", "layers.0.theta",
                    "layers.0.prox.prox.gamma",
                    "layers.2.prox.prox.Wtheta", "layers.2.prox.prox.Wphi",
                    "layers.2.prox.prox.rho", "layers.2.prox.prox.gamma"}
    dry, bad = set(), []
    for name, par in net.named_parameters():
        stem = name.rsplit(".weight", 1)[0].replace(".conv_real", "")                    .replace(".conv_imag", "").replace(".conv", "")
        if par.grad is None:
            dry.add(stem)
        elif not torch.isfinite(par.grad).all():
            bad.append(name)
    check("no NaN / inf gradients anywhere", not bad, ", ".join(bad))
    check("exactly the cold-start / blend parameters are dry", dry == expected_dry,
          f"unexpected: {sorted(dry - expected_dry)}  missing: {sorted(expected_dry - dry)}")
    nonzero = sum(float(g.abs().sum()) for g in
                  (p.grad for p in net.parameters()) if g is not None)
    check("gradients are not identically zero", nonzero > 0, f"sum |g| = {nonzero:.3e}")

    net.project()
    check("project() runs over the whole net", True)

    with torch.no_grad():
        x_unguided, _ = net(y, guide=None)
    check("the net also runs unguided", not torch.allclose(x, x_unguided, atol=1e-6))


def test_registry():
    print("\n[build_model]")
    cfg = dict(model=dict(type="LGGS", params=dict(
        K=2, M=16, C=1, P=3, s=2, Mh=8, window=1, guide_window=3,
        sim_fun="distance", is_complex=False, preproc="identity",
        spectral_init=False)))
    net = build_model(cfg)
    check("build_model('LGGS') -> LGGSNet", isinstance(net, LGGSNet))

    for bad, why in ((dict(guide_window=1), "guide_window"), (dict(Mh=None), "Mh")):
        params = dict(cfg["model"]["params"], **bad)
        try:
            build_model(dict(model=dict(type="LGGS", params=params)))
            check(f"refuses {why}", False)
        except ValueError as e:
            check(f"refuses {why}", why in str(e))


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for fn in (test_joint_softmax_is_one_simplex,
               test_gather_matches_bruteforce,
               test_guide_enters_only_through_the_adjacency,
               test_unguided_fallback_is_exactly_group_threshold,
               test_complex_and_projection,
               test_joint_softmax_rejects_fused_backends,
               test_layer_uses_its_own_dictionary_for_the_guide,
               test_net_forward_backward,
               test_registry):
        fn()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
        raise SystemExit(1)
