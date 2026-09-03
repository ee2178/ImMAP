"""
Guided GroupCDL: the same guided prox, applied as shrinkage instead of clipping.

Run with `python -m tests.test_guided_cdl`.

The claim this file exists to check
-----------------------------------
On REAL data the Fenchel (clipping) form of the prox -- what an LPDS dual step
applies -- has zero gradient wherever it is active.  `test_clipping_saturates`
measures exactly that, on the SAME weights, by differentiating the radial
direction: scale the input by `s` and ask how the output energy moves.

    shrinkage:  out ~ s z          ->  d/ds ||out||^2 ~ 2 s ||z||^2
    clipping:   out ~ tau z / xi   ->  d/ds ||out||^2 ~ 0     (xi ~ s)

Complex features keep a live gradient through the phase in that region, which is
why LGGS is fine there and this class is what real data needs.
"""

import torch

from models import build_model
from models.guided_cdl import GuidedGroupCDL, GuidedLISTALayer
from models.guided_prox import GuidedFenchelProx, GuidedGroupThreshold

PASS, FAIL = [], []

PROX = dict(Mh=8, nheads=1, window=1, guide_window=3, sim_fun="distance",
            dK=1, init_strategy="semi_orthogonal", attn_backend="gather")


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


# ---------------------------------------------------------------------------
def test_clipping_saturates_and_shrinkage_does_not():
    print("\n[why real data needs the shrinkage form]")
    torch.manual_seed(0)
    shrink = GuidedGroupThreshold(16, **PROX)
    clip = GuidedFenchelProx(shrink)              # the SAME weights
    z = torch.randn(1, 16, 6, 6)
    v = torch.randn(1, 16, 6, 6)

    def radial(prox, amp):
        s = torch.tensor(float(amp), requires_grad=True)
        out, _ = prox(s * z, v, None, {})
        return float(torch.autograd.grad(out.pow(2).sum(), s)[0])

    d_sh, d_cl = radial(shrink, 8.0), radial(clip, 8.0)
    check("shrinkage keeps a radial gradient", abs(d_sh) > 1.0, f"d/ds = {d_sh:.4g}")
    check("clipping's has collapsed", abs(d_cl) < 1e-3 * abs(d_sh),
          f"d/ds = {d_cl:.4g}  (ratio {abs(d_cl) / abs(d_sh):.2e})")

    # The mechanism, stated directly: the clipped output's magnitude stops
    # tracking the input's once the prox is active.
    with torch.no_grad():
        n4 = clip(4.0 * z, v, None, {})[0].norm()
        n8 = clip(8.0 * z, v, None, {})[0].norm()
        m4 = shrink(4.0 * z, v, None, {})[0].norm()
        m8 = shrink(8.0 * z, v, None, {})[0].norm()
    check("clipped norm is ~flat in the input scale", abs(n8 / n4 - 1.0) < 0.05,
          f"||clip(8z)|| / ||clip(4z)|| = {n8 / n4:.4f}")
    check("shrunk norm doubles with the input scale", abs(m8 / m4 - 2.0) < 0.05,
          f"||GT(8z)|| / ||GT(4z)|| = {m8 / m4:.4f}")


def test_layer_applies_the_prox_directly():
    print("\n[the layer holds the prox, not its conjugate]")
    layer = GuidedLISTALayer(1, 16, P=3, stride=1, is_complex=False, Mh=8,
                             window=1, guide_window=3,
                             prox_kws=dict(sim_fun="distance", dK=1,
                                           init_strategy="semi_orthogonal"))
    check("prox is a GuidedGroupThreshold",
          isinstance(layer.prox, GuidedGroupThreshold))
    check("prox is NOT wrapped in the Fenchel conjugate",
          not isinstance(layer.prox, GuidedFenchelProx))
    check("the layer is real-valued", not layer.is_complex)

    w = torch.randn(2, 1, 8, 8)
    v = layer.analyse_guides([w])[0]
    check("A w has the latent shape", v.shape == (2, 16, 8, 8), str(tuple(v.shape)))


# ---------------------------------------------------------------------------
def net(**over):
    torch.manual_seed(0)
    kws = dict(K=3, M=16, C=1, P=3, s=2, Mh=8, window=1, guide_window=3,
               sim_fun="distance", dK=2, is_complex=False, preproc="identity",
               spectral_init=False)
    kws.update(over)
    return GuidedGroupCDL(**kws)


def test_attention_sharing():
    print("\n[share_attention ties the transforms across layers]")
    tied, untied = net(share_attention=True), net(share_attention=False)

    p0, p1 = tied.layers[0].prox, tied.layers[-1].prox
    same = (p0.Wtheta.weight is p1.Wtheta.weight
            and p0.Walpha.weight is p1.Walpha.weight
            and p0.gamma.weight is p1.gamma.weight)
    check("W_theta / W_alpha / gamma are the same tensor", same)
    check("rho stays per-layer", p0.rho.weight is not p1.rho.weight)
    check("tau stays per-layer", p0.tau.weight is not p1.tau.weight)

    n_tied = sum(p.numel() for p in tied.parameters())
    n_untied = sum(p.numel() for p in untied.parameters())
    check("sharing reduces the parameter count", n_tied < n_untied,
          f"{n_tied} vs {n_untied}")

    q0, q1 = untied.layers[0].prox, untied.layers[-1].prox
    check("share_attention=False really unties",
          q0.Wtheta.weight is not q1.Wtheta.weight)


def test_readout_dictionary():
    print("\n[the read-out dictionary]")
    m = net()
    check("D is seeded from layer 0's synthesis",
          torch.allclose(m.D.weight, m.layers[0].synthesis.weight, atol=0))
    check("D is a separate parameter, not an alias",
          m.D.weight is not m.layers[0].synthesis.weight)


def test_forward_backward():
    print("\n[GuidedGroupCDL forward / backward]")
    m = net()
    y = torch.randn(1, 1, 16, 16)
    w = torch.randn(1, 2, 1, 16, 16)          # two guides, (B, G, C, H, W)
    x, z = m(y, guide=w)
    check("x_hat keeps the input grid", x.shape == y.shape, str(tuple(x.shape)))
    check("z lives on the strided latent grid", z.shape == (1, 16, 8, 8),
          str(tuple(z.shape)))
    check("everything stays real", not x.is_complex() and not z.is_complex())

    (x.square().mean() + z.square().mean()).backward()
    bad = [n for n, p in m.named_parameters()
           if p.grad is not None and not torch.isfinite(p.grad).all()]
    dry = [n.rsplit(".weight", 1)[0].replace(".conv_real", "").replace(".conv_imag", "")
           for n, p in m.named_parameters() if p.grad is None]
    check("no NaN / inf gradients", not bad, ", ".join(bad))
    # Layer 0 takes the cold-start shortcut (z^(0) = 0), so its synthesis never
    # runs; with dK=2 and K=3 the adjacency is built at layer 0, rebuilt at
    # layer 1 and REUSED at layer 2, so layer 2's rho is never evaluated. Every
    # attention transform is shared, hence reached by layers 0-1 regardless.
    check("only the structurally-unreachable weights are dry",
          set(dry) == {"layers.0.synthesis", "layers.2.prox.rho"},
          f"dry = {sorted(dry)}")

    m.project()
    check("project() runs", True)

    with torch.no_grad():
        x_un, _ = m(y, guide=None)
    check("the same net runs unguided", not torch.allclose(x, x_un, atol=1e-6))


def test_registry():
    print("\n[build_model]")
    params = dict(K=2, M=16, C=1, P=3, s=2, Mh=8, window=1, guide_window=3,
                  sim_fun="distance", is_complex=False, preproc="identity",
                  spectral_init=False)
    m = build_model(dict(model=dict(type="LGGCDL", params=params)))
    check("build_model('LGGCDL') -> GuidedGroupCDL", isinstance(m, GuidedGroupCDL))

    for bad, why in ((dict(guide_window=1), "guide_window"),
                     (dict(Mh=None), "Mh"),
                     (dict(is_complex=True), "LGGS")):
        try:
            build_model(dict(model=dict(type="LGGCDL",
                                        params=dict(params, **bad))))
            check(f"refuses {why}", False)
        except ValueError as e:
            check(f"refuses {why}", why in str(e))


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for fn in (test_clipping_saturates_and_shrinkage_does_not,
               test_layer_applies_the_prox_directly,
               test_attention_sharing,
               test_readout_dictionary,
               test_forward_backward,
               test_registry):
        fn()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
        raise SystemExit(1)
