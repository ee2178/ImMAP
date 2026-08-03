"""
Initialization correctness, the LPDS duality, and the flex attention backend.

Run with:  python -m tests.test_init
"""

import sys

import numpy as np
import torch

from models import build_model
from models.base import set_weight
from models.components import ComplexConvTranspose2d, Conv2d, ConvTranspose2d
from models.multigrid import MGCDLNet, first_layer
from models.prox import GroupThreshold, SoftThreshold, soft_threshold
from operators import Identity
from solvers.eigen import power_method

torch.manual_seed(0)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"[{'ok ' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")


def rel(a, b):
    return (a - b).abs().max().item() / (b.abs().max().item() + 1e-12)


def op_norm(f, C, dtype, n=128, iters=300):
    return abs(power_method(f, torch.rand(1, C, n, n, dtype=dtype),
                            num_iter=iters, verbose=False)[0])


# ---------------------------------------------------------------------------
def test_set_weight():
    """The property-vs-Parameter trap that made complex spectral_init a no-op."""
    c = Conv2d(1, 4, 5, stride=2, bias=False, complex=True)
    W = torch.randn(4, 1, 5, 5, dtype=torch.complex64)
    set_weight(c, W)
    check("set_weight writes a complex Gauss conv", rel(c.weight, W) < 1e-6)

    # the failing idiom, demonstrated
    before = c.weight.clone()
    c.weight.data /= 7.0
    check("in-place .data write on a complex conv IS a no-op (the old bug)",
          rel(c.weight, before) < 1e-9)
    set_weight(c, c.weight / 7.0)
    check("set_weight actually rescales", rel(c.weight, before / 7.0) < 1e-6)

    r = Conv2d(1, 4, 5, bias=False, complex=False)
    set_weight(r, torch.randn(4, 1, 5, 5))
    check("set_weight writes a real conv", r.weight.shape == (4, 1, 5, 5))

    raw = torch.nn.Conv2d(1, 4, 5, bias=False, dtype=torch.cfloat)
    set_weight(raw, W)
    check("set_weight writes a raw nn.Conv2d Parameter",
          rel(raw.weight.data, W) < 1e-6 and isinstance(raw.weight, torch.nn.Parameter))

    ct = ComplexConvTranspose2d(4, 1, 5, stride=2, bias=False)
    set_weight(ct, W)
    check("ComplexConvTranspose2d exposes a writable weight", rel(ct.weight, W) < 1e-6)


def test_spectral_init():
    """||B A|| must be 1 after init -- for REAL and COMPLEX alike."""
    from models import CDLNet, GroupCDL, IPALMNet

    for cx in (False, True):
        n = CDLNet(K=3, M=8, P=5, s=2, C=1, complex=cx, init=True)
        L = op_norm(lambda x: n.B[0](n.A[0](x)), 1, n.A[0].weight.dtype)
        check(f"CDLNet(complex={cx}) has ||BA|| = 1", abs(L - 1.0) < 5e-2,
              f"||BA|| = {L:.4f}")

    g = GroupCDL(M=8, Mh=4, K=2, W=5, sc=2, P=5, attn_backend="gather",
                 is_complex=True, init=True)
    L = op_norm(lambda x: g.B[0](g.A[0](x)), 1, g.A[0].weight.dtype)
    check("GroupCDL(complex) has ||BA|| = 1", abs(L - 1.0) < 5e-2, f"||BA|| = {L:.4f}")

    n = IPALMNet(K=2, M=8, P=5, s=2, C=1, init=True)
    L = op_norm(lambda x: n.B[0](n.A[0](x)), 1, torch.cfloat)
    check("IPALMNet constructs and has ||BA|| = 1", abs(L - 1.0) < 5e-2,
          f"||BA|| = {L:.4f}")

    for cx in (False, True):
        net = MGCDLNet(K=[1, [2, 2]], M=6, C=1, P=5, s=1, is_complex=cx)
        lay = first_layer(net.lista.first)
        L = op_norm(lambda x: lay.synthesis(lay.analysis(x)), 1, lay.analysis.weight.dtype)
        check(f"MGCDLNet(complex={cx}) has ||BA|| = 1", abs(L - 1.0) < 5e-2,
              f"||BA|| = {L:.4f}")


def test_project_filters():
    from models import CDLNet
    n = CDLNet(K=2, M=8, P=5, s=2, C=1, complex=True, init=False)
    set_weight(n.A[0], 50.0 * n.A[0].weight)
    n.project()
    nrm = torch.linalg.norm(n.A[0].weight, dim=(2, 3))
    check("project_filters actually shrinks a complex conv",
          nrm.max().item() <= 1.0 + 1e-4, f"max filter norm = {nrm.max().item():.4f}")


# ---------------------------------------------------------------------------
def test_lpds_is_a_residual_connection():
    """MG-LPDS == MG-CDLNet with a residual connection around the threshold."""
    z = torch.randn(2, 8, 6, 6)
    st = SoftThreshold(8, tau0=0.3)
    from models.prox import FenchelProx
    fp = FenchelProx(SoftThreshold(8, tau0=0.3))
    a, _ = st(z, 0.0)
    b, _ = fp(z, 0.0)
    check("LPDS prox = z - ST(z) (clipping)", rel(b, z - a) < 1e-6)
    check("Moreau identity: prox_g + prox_g* = id", rel(a + b, z) < 1e-6)

    gt = GroupThreshold(8, Mh=4, window=5, tau0=0.05)
    fg = FenchelProx(GroupThreshold(8, Mh=4, window=5, tau0=0.05))
    fg.prox.load_state_dict(gt.state_dict())
    a, _ = gt(z, 0.0, {})
    b, _ = fg(z, 0.0, {})
    check("group LPDS = group clipping", rel(b, z - a) < 1e-6)

    # network level: the dual read-out is y~ - D z (identity preproc -> exact)
    y = torch.randn(1, 1, 16, 16)
    net = MGCDLNet(K=[1, [2, 2]], M=4, C=1, P=5, s=1, is_complex=False,
                   dual=True, preproc="identity")
    out, z = net(y, E=Identity(), sigma=0.05)
    check("MG-LPDS runs", out.shape == y.shape)
    with torch.no_grad():
        check("MG-LPDS read-out is y~ - D z", rel(out, y - net.D(z)) < 1e-5,
              f"rel={rel(out, y - net.D(z)):.2e}")

    # the ONLY difference from MG-CDLNet is the prox / read-out pair: same
    # weights, primal read-out must be D z
    prim = MGCDLNet(K=[1, [2, 2]], M=4, C=1, P=5, s=1, is_complex=False,
                    dual=False, preproc="identity")
    with torch.no_grad():
        o2, z2 = prim(y, E=Identity(), sigma=0.05)
        check("MG-CDLNet read-out is D z", rel(o2, prim.D(z2)) < 1e-5)

    for name in ("MGLPDS", "MGGroupLPDS"):
        p = dict(K=[1, [2, 2]], M=4, C=1, P=5, s=1, is_complex=False)
        if name == "MGGroupLPDS":
            p.update(W=5, Mh=2)
        m = build_model({"model": {"type": name, "params": p}})
        check(f"build_model({name}) pins dual=True", m.dual)

    try:
        build_model({"model": {"type": "MGGroupCDL",
                               "params": dict(K=4, M=4, C=1, P=5)}})
        check("MGGroupCDL rejects a missing group prox", False)
    except ValueError:
        check("MGGroupCDL rejects a missing group prox", True)


# ---------------------------------------------------------------------------
def test_flex_backend():
    """flex must match gather numerically (forward only -- torch has no CPU
    backward for FlexAttention, so training with it requires a GPU)."""
    torch.manual_seed(4)
    kw = dict(M=8, Mh=4, window=5, tau0=0.05, nheads=1)
    gat = GroupThreshold(attn_backend="gather", **kw)
    flx = GroupThreshold(attn_backend="flex", flex_compile_mask=False, **kw)
    flx.load_state_dict(gat.state_dict())

    z = torch.randn(2, 8, 8, 8)
    with torch.no_grad():
        a, _ = gat(z, 0.02, {})
        b, _ = flx(z, 0.02, {})
    check("flex prox matches gather", rel(b, a) < 1e-4, f"rel={rel(b, a):.2e}")

    for mode in ("moreau", "simple", "rigorous"):
        with torch.no_grad():
            ga, _ = gat.subgradient(z, 0.02, {}, mode=mode)
            gb, _ = flx.subgradient(z, 0.02, {}, mode=mode)
        check(f"flex dg[{mode}] matches gather", rel(gb, ga) < 1e-4,
              f"rel={rel(gb, ga):.2e}")

    # complex latents: q/k get [Re; Im] stacked, Gamma still acts on real values
    zc = torch.randn(1, 8, 8, 8, dtype=torch.complex64)
    with torch.no_grad():
        oc, _ = flx(zc, 0.02, {})
        gc, _ = flx.subgradient(zc, 0.02, {}, mode="rigorous")
    check("flex handles complex latents",
          oc.is_complex() and torch.isfinite(gc.abs()).all().item())

    # whole network on the flex backend
    net = MGCDLNet(K=[1, [2, 2]], M=8, Mh=4, C=1, P=5, s=2, W=5,
                   is_complex=False, attn_backend="flex")
    with torch.no_grad():
        out, _ = net(torch.randn(1, 1, 32, 32), E=Identity(), sigma=0.05)
    check("MG-GroupCDL runs end to end on flex", out.shape == (1, 1, 32, 32))

    from models.circulant_flex import _BLOCK_MASK_CACHE
    n_prox = sum(1 for m in net.modules() if isinstance(m, GroupThreshold))
    grids = {(k[0], k[1]) for k in _BLOCK_MASK_CACHE}
    check("block masks are shared across every prox instance",
          len(_BLOCK_MASK_CACHE) <= len(grids) and n_prox > len(grids),
          f"{len(_BLOCK_MASK_CACHE)} masks / {len(grids)} grids / {n_prox} prox modules")


if __name__ == "__main__":
    for fn in (test_set_weight, test_spectral_init, test_project_filters,
               test_lpds_is_a_residual_connection, test_flex_backend):
        print(f"\n--- {fn.__name__} ---")
        fn()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("failed:", FAIL)
    sys.exit(1 if FAIL else 0)
