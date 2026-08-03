"""
Sanity tests for the multigrid / LADMM port (models/prox.py, models/lista.py,
models/multigrid.py, models/ladmm.py, operators/resample.py, solvers/cg.py).

Run with:  python -m tests.test_multigrid
"""

import math
import sys

import torch
import torch.nn.functional as F

from models.lista import LISTALayer, make_lista
from models.multigrid import (MGCDLNet, VCycle, identity_widen_weight,
                              widen_filter, widen_pixel)
from models.prox import GroupThreshold, SoftThreshold, Polynomial
from operators import FFT2D, Identity, Mask, Sense
from operators.resample import galerkin, prolong, restrict, restrict_noise

torch.manual_seed(0)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"[{'ok ' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")


def rel(a, b):
    return (a - b).abs().max().item() / (b.abs().max().item() + 1e-12)


# ---------------------------------------------------------------------------
def test_grid_transfer():
    x = torch.randn(2, 3, 16, 16)
    r = restrict(x)
    check("restrict halves the grid", r.shape == (2, 3, 8, 8), str(tuple(r.shape)))
    check("prolong doubles the grid", prolong(r).shape == x.shape)

    xc = torch.randn(2, 3, 16, 16, dtype=torch.complex64)
    rc = restrict(xc)
    check("restrict is complex-safe",
          rc.is_complex() and rel(rc.real, restrict(xc.real)) < 1e-6)

    const = torch.full((1, 1, 8, 8), 3.0)
    check("restriction of a constant is exact", rel(restrict(const), const[..., :4, :4]) < 1e-6)

    # noise level: an averaged 2x2 block of iid noise has half the std
    n = torch.randn(1, 1, 256, 256)
    check("sigma_c = sigma / 2 matches the empirical std",
          abs(restrict(n).std().item() - 0.5) < 0.02,
          f"std={restrict(n).std().item():.3f}")
    check("restrict_noise passes scalars", abs(restrict_noise(0.1) - 0.05) < 1e-9)
    check("restrict_noise passes (B,1,1,1)",
          restrict_noise(torch.full((4, 1, 1, 1), 0.2)).shape == (4, 1, 1, 1))


def test_galerkin():
    check("galerkin(Identity) short-circuits", isinstance(galerkin(Identity()), Identity))
    smaps = torch.randn(1, 4, 16, 16, dtype=torch.complex64)
    mask = (torch.rand(1, 1, 16, 16) > 0.5).to(torch.complex64)
    E = Mask(mask) @ FFT2D() @ Sense(smaps)
    Ec = galerkin(E)
    zc = torch.randn(1, 1, 8, 8, dtype=torch.complex64)
    check("coarse Gram maps coarse -> coarse", Ec.gram(zc).shape == zc.shape)


def test_widening():
    """The widened level must reproduce the fine level's synthesis exactly."""
    w = 3
    M, C, P = 5, 1, 7
    D = torch.randn(M, C, P, P)
    Dw = widen_filter(D, w)
    z = torch.randn(2, M, 8, 8)
    zw = F.conv2d(z, identity_widen_weight(M * w, M, w))            # widen_z
    lhs = F.conv_transpose2d(zw, Dw, padding=P // 2)
    rhs = F.conv_transpose2d(z, D, padding=P // 2)
    check("D_c (widen_z z) == D z", rel(lhs, rhs) < 1e-5, f"rel={rel(lhs, rhs):.2e}")

    # spectral norm preservation
    A = torch.randn(8, 4)
    Aw = A.repeat(3, 1) / math.sqrt(3)
    s0 = torch.linalg.matrix_norm(A, ord=2)
    s1 = torch.linalg.matrix_norm(Aw, ord=2)
    check("widening preserves the spectral norm", abs(s0 - s1) / s0 < 1e-5)

    Wp = torch.randn(6, 4, 1, 1)
    check("widen_pixel grows both axes",
          widen_pixel(Wp, 2).shape == (12, 8, 1, 1))


def test_prox_subgradients():
    z = torch.randn(2, 8, 6, 6)
    st = SoftThreshold(8, tau0=0.3)
    g, _ = st.subgradient(z, 0.0)
    tau = st.tau(None).squeeze()
    check("dg of l1 is the clip", rel(g, torch.sgn(z) * torch.minimum(z.abs(), tau.view(1, -1, 1, 1))) < 1e-6)
    check("|dg| <= tau", (g.abs() <= tau.max() + 1e-5).all().item())

    gt = GroupThreshold(8, Mh=4, window=5, tau0=0.05, nheads=2)
    out, cache = gt(z, 0.0, {})
    check("group prox keeps its shape", out.shape == z.shape)
    check("adjacency is cached", "Gamma" in cache)
    for mode in ("moreau", "simple", "rigorous"):
        g, _ = gt.subgradient(z, 0.0, {}, mode=mode)
        check(f"group dg[{mode}] shape + finite",
              g.shape == z.shape and torch.isfinite(g).all().item())

    # the rigorous subgradient must be the true gradient of
    #   g(z) = tau . 1^T Wbeta sqrt(Gamma (Walpha z)^2)
    gt0 = GroupThreshold(6, Mh=4, window=3, tau0=1.0, nheads=1, eps=1e-10)
    z0 = torch.randn(1, 6, 5, 5, dtype=torch.float64).requires_grad_(True)
    gt0 = gt0.double()
    cache = {}
    Gamma, cache = gt0.gamma_of(z0.detach(), 0.0, cache)

    def energy(zz):
        xi = gt0.beta_apply(torch.sqrt(gt0.apply_gamma(Gamma, gt0.Walpha(zz) ** 2) + gt0.eps))
        return (gt0.tau(None) * xi).sum()

    auto = torch.autograd.grad(energy(z0), z0)[0]
    manual, _ = gt0.subgradient(z0.detach(), 0.0, dict(cache), mode="rigorous")
    check("rigorous dg == autograd of the group energy",
          rel(manual, auto) < 1e-6, f"rel={rel(manual, auto):.2e}")

    # complex path
    zc = torch.randn(1, 8, 6, 6, dtype=torch.complex64)
    gtc = GroupThreshold(8, Mh=4, window=5)
    oc, _ = gtc(zc, 0.05, {})
    gc, _ = gtc.subgradient(zc, 0.05, {})
    check("group prox / dg handle complex latents",
          oc.is_complex() and gc.is_complex() and torch.isfinite(gc.abs()).all().item())


def test_vcycle_shapes_and_grads():
    net = MGCDLNet(K=[2, [4, 4, 2]], M=6, C=1, P=5, s=1, W=1, is_complex=False)
    v = net.lista.first
    check("V-cycle depth", v.depth == 3, str(v.depth))
    check("V-cycle iters per level", v.iters_per_level == [4, 4, 2],
          str(v.iters_per_level))
    check("pad stride = s * 2^(L-1)", net.pad_stride == 4, str(net.pad_stride))

    y = torch.randn(2, 1, 32, 32)
    x, z = net(y, E=Identity(), sigma=0.05)
    check("MG denoising output shape", x.shape == y.shape, str(tuple(x.shape)))
    x.abs().sum().backward()
    named = dict(net.named_parameters())
    # The only structurally inert weight is the very first synthesis: layer 0 of
    # the first V-cycle runs the z=0 cold start, so B is applied to nothing.
    # (Same quirk as CDLNet's B[0].)  Everything else must train.
    inert = {"lista.layers.0.listaA.layers.0.synthesis.conv_real.weight",
             "lista.layers.0.listaA.layers.0.synthesis.conv_imag.weight"}
    missing = [n for n, p in named.items() if p.grad is None and n not in inert]
    check("every live parameter receives a gradient", not missing,
          f"{len(missing)} missing: {missing[:4]}")
    check("alpha is trained", named["lista.layers.0.alpha.conv.weight"].grad.abs().sum() > 0)
    check("eta (pi scale) is trained",
          named["lista.layers.0.mglayer.listaA.layers.0.eta.weight"].grad.abs().sum() > 0)
    check("outermost level allocates no dead eta",
          not any(n.startswith("lista.layers.0.listaA") and "eta" in n for n in named))

    net.project()
    check("project() runs and keeps thresholds >= 0",
          all((p >= 0).all() for n, p in net.named_parameters() if n.endswith("tau.weight")))

    # odd input sizes must survive the reflect padding
    x2, _ = net(torch.randn(1, 1, 30, 27), E=Identity(), sigma=0.05)
    check("odd input sizes round-trip", x2.shape == (1, 1, 30, 27), str(tuple(x2.shape)))


def test_vcycle_mri_and_group():
    net = MGCDLNet(K=[1, [2, 2]], M=4, Mh=2, C=1, P=5, s=2, W=5, is_complex=True,
                   widen=2, dK=2)
    check("widened level channel counts",
          net.lista.first.mglayer.first.M == 8, str(net.lista.first.mglayer.first.M))

    smaps = torch.randn(1, 4, 32, 32, dtype=torch.complex64)
    smaps = smaps / smaps.abs().pow(2).sum(1, keepdim=True).sqrt()
    mask = (torch.rand(1, 1, 32, 32) > 0.4).to(torch.complex64)
    E = Mask(mask) @ FFT2D() @ Sense(smaps)
    y = E(torch.randn(1, 1, 32, 32, dtype=torch.complex64))
    x, z = net(y, E=E, sigma=torch.tensor([0.01]))
    check("MG CS-MRI output shape", x.shape == (1, 1, 32, 32), str(tuple(x.shape)))
    check("MG CS-MRI output is complex + finite",
          x.is_complex() and torch.isfinite(x.abs()).all().item())
    x.abs().sum().backward()
    check("group + widening + complex backward works",
          net.lista.first.dF.widen_z.weight.grad is not None)


def test_plain_lista():
    net = MGCDLNet(K=4, M=6, C=1, P=5, s=1, is_complex=False)
    check("plain CDLNet has no V-cycle", not isinstance(net.lista.first, VCycle))
    y = torch.randn(1, 1, 24, 24)
    x, z = net(y, sigma=0.1)
    check("plain CDLNet output shape", x.shape == y.shape)
    lay = make_lista(3, 1, 5, P=5, is_complex=False)
    check("make_lista replicates identical layers",
          torch.equal(lay.layers[0].analysis.weight, lay.layers[2].analysis.weight))
    check("layers are untied",
          lay.layers[0].analysis.conv_real.weight is not lay.layers[2].analysis.conv_real.weight)


def test_fixed_point_consistency():
    """FAS: if z is a fixed point of the fine iteration, pi must cancel the
    coarse gradient at z_c = R z, i.e. the coarse iteration keeps R z fixed."""
    torch.manual_seed(3)
    v = VCycle([2, 2], C=1, M=4, is_complex=False, P=5, stride=1, window=1,
               prox_kws=dict(tau0=0.0))                     # tau=0 -> pure gradient
    y = torch.randn(1, 1, 16, 16)
    z = torch.randn(1, 4, 16, 16)
    E = Identity()

    # one fine gradient step at z, using dF's own (tied) copies
    Bz = v.dF.synthesis_fine(z)
    step_f = v.dF.analysis_fine(Bz - y)

    z_c, pi, y_c, E_c, s_c = v.dF(z, y, E, 0.0, {})
    Bz_c = v.dF.synthesis_coarse(z_c)
    step_c = v.dF.analysis_coarse(E_c.gram(Bz_c) - y_c)

    # coarse step corrected by pi should equal the restricted fine step
    corrected = step_c - pi
    check("FAS: dF_c(Rz) - pi == R dF_f(z)",
          rel(corrected, restrict(step_f)) < 1e-5,
          f"rel={rel(corrected, restrict(step_f)):.2e}")


if __name__ == "__main__":
    for fn in (test_grid_transfer, test_galerkin, test_widening,
               test_prox_subgradients, test_vcycle_shapes_and_grads,
               test_vcycle_mri_and_group, test_plain_lista,
               test_fixed_point_consistency):
        print(f"\n--- {fn.__name__} ---")
        fn()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("failed:", FAIL)
    sys.exit(1 if FAIL else 0)
