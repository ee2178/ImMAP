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
from operators.resample import (TRANSFER_FILTERS, GridTransfer, default_filter,
                                filter_length, galerkin, is_adjoint_pair,
                                noise_scale, prolong, restrict, restrict_noise,
                                use_filter)

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

    # circular padding, so constants survive at the border too -- not just in
    # the interior the way a zero-padded long kernel would leave them
    const = torch.full((1, 1, 8, 8), 3.0)
    check("restriction of a constant is exact", rel(restrict(const), const[..., :4, :4]) < 1e-6)
    check("prolongation of a constant is exact",
          rel(prolong(const), torch.full((1, 1, 16, 16), 3.0)) < 1e-6)

    # THE invariant: P == 4 R^T exactly, not approximately.  This is what makes
    # the coarse Gram Hermitian, and hence what makes the coarse level a real
    # optimisation problem for the FAS correction to correct.
    z = torch.randn(2, 3, 8, 8)
    lhs = (restrict(x) * z).sum()
    rhs = (x * prolong(z)).sum() / 4
    check("prolong is exactly 4 . restrict^T",
          abs(lhs - rhs).item() / abs(lhs).item() < 1e-5,
          f"rel={abs(lhs - rhs).item() / abs(lhs).item():.2e}")

    # noise level: filtering by h scales the std by ||h||_2 (decimation does
    # not), so this tracks the kernel rather than a hardcoded 1/2
    n = torch.randn(1, 1, 256, 256)
    check("sigma_c / sigma == ||h||_2, matching the empirical std",
          abs(restrict(n).std().item() - noise_scale()) < 0.02,
          f"std={restrict(n).std().item():.4f} vs ||h||_2={noise_scale():.4f}")
    check("restrict_noise passes scalars",
          abs(restrict_noise(0.1) - 0.1 * noise_scale()) < 1e-9)
    check("restrict_noise passes (B,1,1,1)",
          restrict_noise(torch.full((4, 1, 1, 1), 0.2)).shape == (4, 1, 1, 1))
    check("restrict passes singleton spatial dims",
          restrict(torch.randn(2, 3, 1, 1)).shape == (2, 3, 1, 1))


def test_grid_transfer_module():
    """`GridTransfer` must stay a tied pair whether or not it is learned."""
    x = torch.randn(2, 6, 16, 16)
    z = torch.randn(2, 6, 8, 8)

    fixed = GridTransfer(channels=6, learn=False)
    check("GridTransfer(learn=False) == the module-level functions",
          rel(fixed.restrict(x), restrict(x)) < 1e-6
          and rel(fixed.prolong(z), prolong(z)) < 1e-6)
    check("a fixed GridTransfer exposes no parameters",
          len(list(fixed.parameters())) == 0)

    learned = GridTransfer(channels=6, learn=True)
    with torch.no_grad():                       # stand in for a few SGD steps
        learned.weight.add_(0.05 * torch.randn_like(learned.weight))
    lhs = (learned.restrict(x) * z).sum()
    rhs = (x * learned.prolong(z)).sum() / 4
    check("a MOVED learned kernel still satisfies P == 4 R^T",
          abs(lhs - rhs).item() / abs(lhs).item() < 1e-5,
          f"rel={abs(lhs - rhs).item() / abs(lhs).item():.2e}")
    check("sum-normalisation keeps restrict(const) == const after learning",
          rel(learned.restrict(torch.full((1, 6, 16, 16), 3.0)),
              torch.full((1, 6, 8, 8), 3.0)) < 1e-4)

    n = torch.randn(1, 6, 256, 256)
    check("learned noise_scale tracks the moved kernel",
          abs(learned.restrict(n).std().item() - learned.noise_scale.item()) < 0.02,
          f"empirical={learned.restrict(n).std():.4f} vs "
          f"noise_scale={learned.noise_scale:.4f}")

    learned.zero_grad()
    learned.prolong(learned.restrict(
        torch.randn(1, 6, 16, 16, dtype=torch.complex64))).abs().sum().backward()
    check("gradient reaches the learned kernel through a complex round-trip",
          learned.weight.grad is not None
          and learned.weight.grad.abs().sum().item() > 0)


def test_transfer_filters():
    """Every selectable filter, swept the way an ablation would sweep them.

    The point of the sweep is that switching filters changes the anti-aliasing
    and NOTHING else: each one is still an exact adjoint pair, still preserves
    constants, and still yields a Hermitian PSD coarse Gram. `legacy` is the
    deliberate exception -- it is the pre-existing mean-pool / bilinear pairing,
    kept so an ablation table has a "what we had before" row.
    """
    x = torch.randn(2, 3, 16, 16)
    zc = torch.randn(2, 3, 8, 8)
    n = torch.randn(1, 1, 256, 256)

    smaps = torch.randn(1, 4, 16, 16, dtype=torch.complex64)
    mask = (torch.rand(1, 1, 16, 16) > 0.5).to(torch.complex64)
    E = Mask(mask) @ FFT2D() @ Sense(smaps)

    for f in TRANSFER_FILTERS:
        check(f"[{f}] restrict / prolong round-trip the grid",
              restrict(x, filter=f).shape == (2, 3, 8, 8)
              and prolong(zc, filter=f).shape == (2, 3, 16, 16),
              f"L={filter_length(f)}")

        lhs = (restrict(x, filter=f) * zc).sum()
        rhs = (x * prolong(zc, filter=f)).sum() / 4
        adj = abs(lhs - rhs).item() / abs(lhs).item()
        if is_adjoint_pair(f):
            check(f"[{f}] prolong == 4 . restrict^T", adj < 1e-5, f"rel={adj:.2e}")
        else:
            check(f"[{f}] is NOT an adjoint pair, as documented", adj > 1e-2,
                  f"rel={adj:.2e}")

        check(f"[{f}] restrict preserves constants",
              rel(restrict(torch.full((1, 1, 16, 16), 3.0), filter=f),
                  torch.full((1, 1, 8, 8), 3.0)) < 1e-5)

        emp = restrict(n, filter=f).std().item()
        check(f"[{f}] noise_scale matches the empirical std",
              abs(emp - noise_scale(filter=f)) < 0.02,
              f"{emp:.4f} vs {noise_scale(filter=f):.4f}")

        # the coarse Gram this filter induces
        Ec = galerkin(E, filter=f)
        G = torch.zeros(64, 64, dtype=torch.complex64)
        for i in range(64):
            e = torch.zeros(1, 1, 8, 8, dtype=torch.complex64)
            e.view(-1)[i] = 1
            G[:, i] = Ec.gram(e).reshape(-1)
        herm = ((G - G.conj().T).norm() / G.norm()).item()
        if is_adjoint_pair(f):
            lo = torch.linalg.eigvalsh(
                ((G + G.conj().T) / 2).to(torch.complex128)).real.min().item()
            check(f"[{f}] coarse Gram Hermitian PSD",
                  herm < 1e-5 and lo > -1e-6, f"herm={herm:.2e} lo={lo:.3f}")
        else:
            check(f"[{f}] coarse Gram is non-Hermitian, as documented",
                  herm > 1e-2, f"herm={herm:.2e}")

    # legacy must be BIT-identical to what the module used to do
    check("legacy restrict == the old avg_pool2d",
          rel(restrict(x, filter="legacy"), F.avg_pool2d(x, 2)) < 1e-12)
    check("legacy prolong == the old bilinear interpolate",
          rel(prolong(zc, filter="legacy"),
              F.interpolate(zc, scale_factor=2.0, mode="bilinear",
                            align_corners=False)) < 1e-12)
    check("legacy noise scale is the old hardcoded 1/2",
          abs(restrict_noise(0.1, filter="legacy") - 0.05) < 1e-9)

    # a raw kernel works anywhere a name does
    check("a custom tensor kernel == the equivalent name",
          rel(restrict(x, filter=torch.tensor([1., 3, 3, 1]) / 8),
              restrict(x, filter="bilinear")) < 1e-6)

    # scoping
    check("default filter is sinc", default_filter() == "sinc")
    with use_filter("box"):
        check("use_filter redirects an unqualified call",
              default_filter() == "box"
              and rel(restrict(x), restrict(x, filter="box")) < 1e-6)
    check("use_filter restores on exit", default_filter() == "sinc")

    for bad in ("nope", "legacy"):
        try:
            GridTransfer(filter=bad)
            check(f"GridTransfer rejects filter={bad!r}", False)
        except ValueError:
            check(f"GridTransfer rejects filter={bad!r}", True)


def test_galerkin():
    check("galerkin(Identity) short-circuits", isinstance(galerkin(Identity()), Identity))
    smaps = torch.randn(1, 4, 16, 16, dtype=torch.complex64)
    mask = (torch.rand(1, 1, 16, 16) > 0.5).to(torch.complex64)
    E = Mask(mask) @ FFT2D() @ Sense(smaps)
    Ec = galerkin(E)
    zc = torch.randn(1, 1, 8, 8, dtype=torch.complex64)
    check("coarse Gram maps coarse -> coarse", Ec.gram(zc).shape == zc.shape)

    # E_c = E . P with restrict == P^T / 4, so the coarse Gram is
    # (1/4) P^H E^H E P -- Hermitian PSD, i.e. the gradient of an actual coarse
    # least-squares objective.  The bilinear-up / bilinear-down pair this
    # replaced was 14.5% non-Hermitian, so `dF_coarse` was the subgradient of
    # no objective at all and the FAS correction had nothing well-posed to
    # correct.  Built densely: 8x8 coarse grid, 64 columns.
    n = 8 * 8
    G = torch.zeros(n, n, dtype=torch.complex64)
    for i in range(n):
        e = torch.zeros(1, 1, 8, 8, dtype=torch.complex64)
        e.view(-1)[i] = 1
        G[:, i] = Ec.gram(e).reshape(-1)
    herm = ((G - G.conj().T).norm() / G.norm()).item()
    check("coarse Gram is Hermitian", herm < 1e-5, f"rel={herm:.2e}")
    lo = torch.linalg.eigvalsh(((G + G.conj().T) / 2).to(torch.complex128)).real.min()
    check("coarse Gram is PSD", lo.item() > -1e-6, f"lambda_min={lo:.2e}")


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
    for fn in (test_grid_transfer, test_grid_transfer_module,
               test_transfer_filters, test_galerkin, test_widening,
               test_prox_subgradients, test_vcycle_shapes_and_grads,
               test_vcycle_mri_and_group, test_plain_lista,
               test_fixed_point_consistency):
        print(f"\n--- {fn.__name__} ---")
        fn()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("failed:", FAIL)
    sys.exit(1 if FAIL else 0)
