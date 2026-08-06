"""
Checks for the Multigrid Learned Primal-Dual Splitting port.

`models/lpds.py` (LPDSLayer / LPDSStack, from `lpds.jl`) and
`models/mg_lpds.py` (PDObjectiveDownsample / PDVCycle / MGLPDSNet, from
`mg_lpds.jl`).

The load-bearing test is `test_fas_consistency`. Everything else confirms the
plumbing; only that one confirms the CORRECTION, and a wrong `pi_x` / `pi_z`
passes every shape and gradient check while quietly making the coarse level
solve a different problem.

Run with:  python -m tests.test_mg_lpds
"""

import sys

import torch

from models.lpds import LPDSLayer, LPDSStack, make_lpds_layer, make_lpds_stack
from models.mg_lpds import MGLPDSNet, PDVCycle, first_pd_layer
from operators import FFT2D, Identity, Mask, Sense

torch.manual_seed(0)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"[{'ok ' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")


def rel(a, b, eps=1e-12):
    return ((a - b).abs().max() / (b.abs().max() + eps)).item()


def mri_operator(B=1, C=4, N=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    smaps = torch.randn(B, C, N, N, dtype=torch.complex64, generator=g) + 1.0
    smaps = smaps / smaps.abs().pow(2).sum(1, keepdim=True).sqrt()
    mask = torch.zeros(1, 1, N, N)
    mask[..., ::2] = 1.0
    mask[..., N // 2 - 4:N // 2 + 4] = 1.0
    return Mask(mask) @ FFT2D() @ Sense(smaps)


# ---------------------------------------------------------------------------
def test_layer_shapes():
    """Cold start, warm sweep, and the FAS-corrected sweep."""
    layer = LPDSLayer(C=1, M=6, P=5, stride=2, is_complex=True)
    y = torch.randn(2, 1, 16, 16, dtype=torch.complex64)

    (x, z), cache = layer(None, y, E=Identity(), sigma=0.01)
    check("cold start: x on the image grid", x.shape == (2, 1, 16, 16),
          str(tuple(x.shape)))
    check("cold start: z on the latent grid", z.shape == (2, 6, 8, 8),
          str(tuple(z.shape)))

    (x2, z2), _ = layer((x, z), y, E=Identity(), sigma=0.01, cache=cache)
    check("warm sweep preserves both shapes",
          x2.shape == x.shape and z2.shape == z.shape)

    pi = (torch.randn_like(x), torch.randn_like(z))
    (x3, z3), _ = layer((x, z), y, E=Identity(), sigma=0.01, pi=pi)
    check("pi changes the result", rel(x3, x2) > 1e-6 and rel(z3, z2) > 1e-6)


def test_extrapolation_is_live():
    """theta must actually move the iterate -- it is what makes this LPDS.

    `MGCDLNet(dual=True)` has no extrapolation at all, so a port that quietly
    dropped theta would look like a working network and be the wrong algorithm.
    """
    y = torch.randn(1, 1, 16, 16, dtype=torch.complex64)
    state = (torch.randn(1, 1, 16, 16, dtype=torch.complex64),
             torch.randn(1, 6, 8, 8, dtype=torch.complex64))

    torch.manual_seed(3)
    a = LPDSLayer(C=1, M=6, P=5, stride=2, is_complex=True, theta0=0.0)
    torch.manual_seed(3)
    b = LPDSLayer(C=1, M=6, P=5, stride=2, is_complex=True, theta0=0.7)
    b.load_state_dict({**a.state_dict(), "theta.weight": b.theta.weight})

    (_, za), _ = a(state, y, E=Identity(), sigma=0.01)
    (_, zb), _ = b(state, y, E=Identity(), sigma=0.01)
    check("theta changes the dual iterate", rel(zb, za) > 1e-4,
          f"rel={rel(zb, za):.2e}")

    lay = LPDSLayer(C=1, M=4, P=5, stride=2, is_complex=False)
    with torch.no_grad():
        lay.theta.weight.fill_(5.0)
        lay.project_()
    check("project_ clamps theta into [0, 1]",
          float(lay.theta.weight.max()) <= 1.0 + 1e-6,
          f"max={float(lay.theta.weight.max()):.3f}")


def test_widening_preserves_synthesis():
    """The coarse level must start as an exact replica of the fine one.

    With `widen = w` the coarse dictionary is the fine one tiled w times and
    scaled by 1/sqrt(w), and `widen_z` lifts z the same way, so
    `D_c (widen_z z) == D z` at construction.
    """
    for widen in (1, 2, 3):
        V = PDVCycle([2, 2], C=1, M=4, P=5, stride=2, widen=widen,
                     is_complex=False)
        fine = V.lpdsA.first
        coarse = first_pd_layer(V.mglayer)
        check(f"coarse level widened M (widen={widen})",
              coarse.M == fine.M * widen, f"{fine.M} -> {coarse.M}")

        z = torch.randn(1, fine.M, 8, 8)
        with torch.no_grad():
            lhs = coarse.synthesis(V.dF.widen_z(z))
            rhs = fine.synthesis(z)
        check(f"D_c (widen_z z) == D z (widen={widen})", rel(lhs, rhs) < 1e-5,
              f"rel={rel(lhs, rhs):.2e}")


def test_fas_consistency():
    """The property the corrections exist to provide.

    If `(x, z)` is a fixed point of the FINE iteration, then `(Rx, Rz)` must be
    a fixed point of the COARSE iteration once `(pi_x, pi_z)` are added. That is
    what makes the coarse-grid correction consistent, and it is the only check
    here that would catch a sign error or a swapped residual in `pi`.

    Construct the fixed point rather than iterate to it: pick `(x, z)`, then
    choose `y~` so the fine residual vanishes.
    """
    torch.manual_seed(7)
    V = PDVCycle([2, 2], C=1, M=4, P=5, stride=2, widen=1, is_complex=False)
    fine = V.lpdsA.first
    dF = V.dF

    x = torch.randn(1, 1, 16, 16)
    z = torch.randn(1, 4, 8, 8)

    with torch.no_grad():
        # y~ chosen so grad f(x) + A^T z = 0, i.e. the fine primal residual is 0
        y = x + fine.synthesis(z)                      # E = Identity => gram = I
        cache = {}
        x_c, z_c, pi_x, pi_z, y_c, E_c, sigma_c = dF(x, z, y, Identity(),
                                                     torch.tensor(0.01), cache)

        # the fine primal residual really is zero at this (x, z, y)
        r_fine = x - y + fine.synthesis(z)
        check("constructed fine primal residual is zero",
              r_fine.abs().max().item() < 1e-5,
              f"max={r_fine.abs().max().item():.2e}")

        # coarse primal residual, WITH the correction, must also vanish
        coarse = first_pd_layer(V.mglayer)
        r_coarse = (x_c - y_c + dF.synthesis_coarse(z_c)) - pi_x
        # pi_x = r_coarse_uncorrected - R r_fine, and R r_fine = 0 here, so
        # subtracting pi_x must leave exactly zero.
        check("pi_x makes the coarse primal residual vanish",
              r_coarse.abs().max().item() < 1e-5,
              f"max={r_coarse.abs().max().item():.2e}")

        # the same statement for the dual fixed-point residual
        zhat_c, _ = dF.prox_coarse(z_c + dF.analysis_coarse(x_c), sigma_c, {})
        rz_coarse = (z_c - zhat_c) - pi_z
        zhat_f, _ = dF.prox_fine(z + dF.analysis_fine(x), torch.tensor(0.01), {})
        from operators.resample import restrict
        Rrz_fine = restrict(z - zhat_f)
        check("pi_z reproduces the restricted fine dual residual",
              rel(rz_coarse, Rrz_fine) < 1e-5, f"rel={rel(rz_coarse, Rrz_fine):.2e}")


def test_vcycle_and_stack_are_interchangeable():
    """A V-cycle must be usable anywhere a stack is, and vice versa."""
    kws = dict(C=1, M=4, P=5, stride=2, is_complex=False)
    stack = make_lpds_stack(4, **kws)
    V = PDVCycle([2, 2], widen=1, **kws)

    y = torch.randn(1, 1, 16, 16)
    for name, net in (("LPDSStack", stack), ("PDVCycle", V)):
        (x, z), _ = net(None, y, E=Identity(), sigma=0.02)
        check(f"{name} cold-starts from None",
              x.shape == (1, 1, 16, 16) and z.shape == (1, 4, 8, 8))
        (x2, _), _ = net((x, z), y, E=Identity(), sigma=0.02)
        check(f"{name} continues from a pair", x2.shape == x.shape)

    # `iters_per_level` sums lpdsA + lpdsB per level, so it must reproduce the
    # `iters` the V-cycle was built from -- a stronger and less error-prone
    # assertion than a hardcoded pair (the first version of this test hardcoded
    # the wrong one). Matches Julia's `iters_per_level` exactly.
    for iters in ([2, 2], [4, 4, 2]):
        Vi = PDVCycle(iters, widen=1, **kws)
        check(f"iters_per_level round-trips {iters}",
              Vi.iters_per_level == iters, str(Vi.iters_per_level))
        check(f"depth equals the level count for {iters}",
              Vi.depth == len(iters), str(Vi.depth))


def test_K_parsing():
    """`K` follows the MGCDLNet convention, so the ablation is one key."""
    plain = MGLPDSNet(K=6, M=4, C=1, P=5, s=2, is_complex=False,
                      preproc="identity")
    check("K as an int gives a plain stack", plain.iters is None,
          f"levels={plain.levels}")
    check("plain net pads only by the stride", plain.pad_stride == 2,
          str(plain.pad_stride))

    mg = MGLPDSNet(K=[2, [4, 4, 2]], M=4, C=1, P=5, s=2, is_complex=False,
                   preproc="identity")
    check("K as a pair gives a V-cycle", mg.iters == [4, 4, 2], str(mg.iters))
    check("pad_stride is s * 2^(levels-1)", mg.pad_stride == 8,
          str(mg.pad_stride))

    try:
        MGLPDSNet(K=[1, [3, 2]], M=4, C=1, P=5, is_complex=False)
        check("odd non-coarsest iters are rejected", False)
    except AssertionError:
        check("odd non-coarsest iters are rejected", True)

    degen = MGLPDSNet(K=[3, [5]], M=4, C=1, P=5, s=2, is_complex=False,
                      preproc="identity")
    check("a single level collapses to K_outer * i0", degen.K == 15,
          str(degen.K))


def test_end_to_end_mri():
    """Forward + backward through a real encoding operator, both preprocs."""
    E = mri_operator(B=1, C=4, N=32)
    image = torch.randn(1, 1, 32, 32, dtype=torch.complex64)
    y = E(image)
    sigma = torch.full((1, 1, 1, 1), 0.01)

    for K in (4, [1, [2, 2]], [2, [2, 2, 2]]):
        for preproc in ("kspace", "identity"):
            tag = f"K={K} preproc={preproc}"
            try:
                net = MGLPDSNet(K=K, M=6, C=1, P=5, s=2, widen=1,
                                is_complex=True, preproc=preproc)
                x_hat, z = net(y, E=E, sigma=sigma)
                loss = (x_hat - image).abs().pow(2).mean()
                loss.backward()
                grads = [p.grad for p in net.parameters() if p.grad is not None]
                ok = (x_hat.shape == image.shape
                      and torch.isfinite(x_hat.abs()).all().item()
                      and bool(grads)
                      and all(torch.isfinite(g).all() for g in grads))
                net.project()
                check(f"end to end {tag}", ok,
                      f"loss={float(loss):.3e} {len(grads)} grads")
            except Exception as exc:                              # noqa: BLE001
                check(f"end to end {tag}", False,
                      f"{type(exc).__name__}: {exc}")


def test_widen_end_to_end():
    """widen > 1 widens the DUAL only; the primal stays at C."""
    E = mri_operator(B=1, C=4, N=32)
    image = torch.randn(1, 1, 32, 32, dtype=torch.complex64)
    net = MGLPDSNet(K=[1, [2, 2]], M=4, C=1, P=5, s=2, widen=2,
                    is_complex=True, preproc="kspace")
    x_hat, z = net(E(image), E=E, sigma=torch.full((1, 1, 1, 1), 0.01))
    check("widen=2 runs end to end", x_hat.shape == image.shape,
          str(tuple(x_hat.shape)))

    V = net.net.first
    check("primal channels are NOT widened",
          V.dF.analysis_coarse.weight.shape[1] == V.dF.analysis_fine.weight.shape[1],
          "C stays constant across levels")
    check("dual channels ARE widened",
          V.dF.analysis_coarse.weight.shape[0]
          == 2 * V.dF.analysis_fine.weight.shape[0], "M -> M*widen")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for fn in (test_layer_shapes, test_extrapolation_is_live,
               test_widening_preserves_synthesis, test_fas_consistency,
               test_vcycle_and_stack_are_interchangeable, test_K_parsing,
               test_end_to_end_mri, test_widen_end_to_end):
        print(f"\n--- {fn.__name__} ---")
        try:
            fn()
        except Exception as exc:                                  # noqa: BLE001
            check(fn.__name__, False, f"{type(exc).__name__}: {exc}")
            import traceback
            traceback.print_exc()

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("failed: " + ", ".join(FAIL))
    sys.exit(1 if FAIL else 0)
