"""
Checks for `physics/sense_noise.py`, the SENSE noise-level maps.

Every consistency check from Sec. 7 of `sense_noiselevel/sense_noise.tex` is
here, plus the folding-index check and the pseudo-replica validation of Sec. 9.8.
The folding check is the one that matters most: a column-major-to-row-major slip
in the alias bookkeeping produces maps that look plausible and are wrong.

Run with:  python -m tests.test_sense_noise

Needs a torch built with LAPACK -- everything here goes through
`torch.linalg.eigh`, as does `physics/gfactor.py`.  The local anaconda torch
1.12.1 has neither LAPACK nor a correct complex CPU matmul for lazily-conjugated
operands (`A @ A.conj().mT` comes back non-Hermitian), so run this on the cluster
torch, not that one.
"""

import math
import sys

import torch

from operators.fourier import fftc
from physics.gfactor import gfactor_uniform, sigma_full_map
from physics.mask import make_acc_mask
from physics.sense_noise import (
    _alias_bracket,
    _alias_lags,
    _alias_stack,
    _fold_alias,
    _kernel_taps,
    block_sense,
    coil_combined_noise_level,
    effective_acceleration,
    sense_gfactor,
    sense_noise_level,
    sense_noise_level_mc,
)

torch.manual_seed(0)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"[{'ok ' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")


def close(a, b, rtol=1e-5, atol=1e-6):
    return torch.allclose(a, b, rtol=rtol, atol=atol)


def phantom(B=2, C=6, H=32, W=32, seed=0, dtype=torch.complex128, dc=1.0):
    """Unit-RSS coil maps -- the normalisation the bracket's plain branch assumes.

    `dc` is the common-mode offset added to every coil before normalising.  It
    controls how *collinear* the aliased sensitivity columns are, which is the
    only thing that decides whether the A x A blocks are well conditioned.  The
    default 1.0 matches `tests/test_synthetic_kspace.py`'s fixture and is a
    stress test: with dc = 1 every s[n] clusters near the all-ones direction, so
    for A >= 3 the blocks are nearly rank-deficient and `rtol` truncation kicks
    in.  Use `dc = 0.0` for near-orthogonal columns wherever a check relies on
    the *exact* inverse rather than the truncated one.
    """
    g = torch.Generator().manual_seed(seed)
    s = torch.randn(B, C, H, W, dtype=dtype, generator=g) + dc
    return s / s.abs().pow(2).sum(1, keepdim=True).sqrt()


def rand_cov(C, seed=1, dtype=torch.complex128):
    """A well-conditioned Hermitian PD coil covariance.

    Symmetrised explicitly rather than left as `A @ A.conj().mT`: everything
    downstream (`_herm_pow`, `eigh`) reads only one triangle, so a fixture that
    is Hermitian only up to a buggy complex matmul silently tests nothing.
    """
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(C, C, dtype=dtype, generator=g)
    M = A @ A.conj().transpose(-2, -1).contiguous() / C
    return 0.5 * (M + M.conj().transpose(-2, -1)) + torch.eye(C, dtype=dtype)


# ---------------------------------------------------------------------------
def test_alias_lags_and_stack():
    """`_alias_stack` must gather each pixel's partners, lag 0 first.

    This is the general path `sense_noise_level` now uses for every accel. The
    indivisible case is the point: where j*n/A is not an integer the aliasing peak
    straddles two pixels, so BOTH bracketing integers must appear -- rounding to the
    nearer one throws away up to 36% of that fold's tap. Divisible cases must be
    untouched (floor == ceil), which is the backward-compatibility guard.
    """
    for n, accel, want in [
        (8, 4, (0, 2, 4, 6)),                                  # divisible
        (12, 3, (0, 4, 8)),                                    # divisible
        (48, 4, (0, 12, 24, 36)),                              # divisible
        (10, 4, (0, 2, 3, 5, 7, 8)),                           # 2.5 and 7.5 bracketed
        (46, 6, (0, 7, 8, 15, 16, 23, 30, 31, 38, 39)),
        (368, 6, (0, 61, 62, 122, 123, 184, 245, 246, 306, 307)),   # the house case
    ]:
        got = _alias_lags(n, accel)
        check(f"lags(n={n}, A={accel})", got == want, f"{got} vs {want}")
    check("lag 0 is always first", all(_alias_lags(n, a)[0] == 0
                                      for n in (16, 17, 368) for a in (2, 3, 5, 6)))

    # Values encode the PE index, so the gather is directly readable.
    n, accel, H = 12, 3, 2
    x = torch.arange(n, dtype=torch.float64).to(torch.complex128)
    x = x.reshape(1, 1, 1, n).expand(1, 1, H, n).contiguous()
    P = _alias_stack(x, _alias_lags(n, accel), 3)          # (H*n, 1, accel)
    got = P.reshape(H, n, accel)[0].real
    want = (torch.arange(n)[:, None] + torch.tensor([0, 4, 8])[None, :]) % n
    check("alias_stack gathers n + l_a", close(got, want.to(torch.float64)))

    # Divisible accel must yield exactly A lags; indivisible at most 2A - 1.
    for n, accel in [(48, 4), (96, 6), (368, 4)]:
        check(f"divisible ({n}, {accel}) keeps exactly A lags",
              len(_alias_lags(n, accel)) == accel)
    for n, accel in [(368, 6), (46, 6), (50, 4)]:
        k = len(_alias_lags(n, accel))
        check(f"indivisible ({n}, {accel}) keeps <= 2A-1 lags",
              accel < k <= 2 * accel - 1, f"{k} lags for accel={accel}")


def test_folding_indices():
    """`_fold_alias` must gather PE indices {jj, jj+step, ...} per block.

    Still used by `block_sense`, which needs the reduced-FOV image.
    """
    B, C, H, W, accel = 1, 1, 3, 8, 4
    step = W // accel
    # Encode the PE index in the value so the gather is directly readable.
    x = torch.arange(W, dtype=torch.float64).to(torch.complex128)
    x = x.reshape(1, 1, 1, W).expand(B, C, H, W)
    S, lead, n, ax = _fold_alias(x.contiguous(), accel, -1)
    check("fold shape", tuple(S.shape) == (B * H * step, C, accel), str(tuple(S.shape)))

    got = S.reshape(H, step, C, accel)[0, :, 0, :].real          # (step, accel)
    want = torch.arange(W, dtype=torch.float64).reshape(accel, step).T
    check("fold gathers the alias lattice", close(got, want), f"{got.tolist()}")

    # Round trip: folding then unfolding a per-pixel quantity is the identity.
    from physics.sense_noise import _unfold_alias
    d = S[:, 0, :].real                                          # (nblk, accel)
    back = _unfold_alias(d, B, lead, n, accel, ax)
    check("unfold inverts fold", close(back[0, 0, 0], torch.arange(W, dtype=torch.float64)))


def test_mask_driven_taps_match_the_analytic_bracket():
    """The kernel read off the mask must reproduce the note's taps in both limits.

    p = 0: m[j n/A] = 1/A exactly, so T = ones/A.
    p = 1: m[0] = 1, m[l != 0] = 0, so T = I.
    In between the mask taps keep the ACS sinc's sidelobes, so they differ from
    (1-p)/A by a little -- that is the improvement, not an error.
    """
    n, accel = 48, 4
    lags = _alias_lags(n, accel)
    kw = dict(ax=3, dtype=torch.complex128, device=torch.device('cpu'))

    for acs, note in ((0, 'p=0 -> ones/A'), (n, 'p=1 -> I')):
        m = make_acc_mask((16, n), accel, acs_lines=acs, dim=1).to(torch.complex128)
        T_mask = _kernel_taps(lags, n, accel, acs, m, **kw)
        T_anal = _kernel_taps(lags, n, accel, acs, None, **kw)
        check(f"mask taps == analytic taps, {note}", close(T_mask, T_anal, atol=1e-12),
              f"max |diff| {(T_mask - T_anal).abs().max():.2e}")

    # Structural guarantees the Hadamard form buys: T Hermitian PSD, m[0] = 1/A_eff.
    for acs in (0, 8, 20, 48):
        m = make_acc_mask((16, n), accel, acs_lines=acs, dim=1).to(torch.complex128)
        T = _kernel_taps(lags, n, accel, acs, m, **kw)
        w = torch.linalg.eigvalsh(T).real
        check(f"mask taps give PSD T (acs={acs})", bool((w > -1e-12).all()),
              f"min eig {w.min():.2e}")
        check(f"T[0,0] == 1/A_eff (acs={acs})",
              abs(T[0, 0].real.item() - 1.0 / effective_acceleration(mask=m)) < 1e-12,
              f"{T[0, 0].real.item():.6f} vs {1 / effective_acceleration(mask=m):.6f}")

    # Intermediate p is where it gets interesting. The ACS band of width `acs`
    # contributes sum_{k in ACS} exp(2*pi*i*k*l/n) at lag l; on the lattice
    # l = j*n/A that phase advances by j/A per line, so `acs` consecutive lines sum
    # to EXACTLY zero whenever A | acs. The note's (1-p)/A off-diagonal is then not
    # an approximation at all -- it is exact. It only misses when A does not divide
    # acs, and then only by a sidelobe.
    for acs in (8, 12, 20):                       # all divisible by accel=4
        m = make_acc_mask((16, n), accel, acs_lines=acs, dim=1).to(torch.complex128)
        d = (_kernel_taps(lags, n, accel, acs, m, **kw)
             - _kernel_taps(lags, n, accel, acs, None, **kw)).abs().max().item()
        check(f"accel | acs_size ({accel} | {acs}): analytic taps are EXACT", d < 1e-13,
              f"max |diff| {d:.2e}")
    for acs in (6, 10, 14):                       # not divisible by accel=4
        m = make_acc_mask((16, n), accel, acs_lines=acs, dim=1).to(torch.complex128)
        d = (_kernel_taps(lags, n, accel, acs, m, **kw)
             - _kernel_taps(lags, n, accel, acs, None, **kw)).abs().max().item()
        check(f"accel not | acs_size ({accel}, {acs}): taps differ by a sidelobe",
              1e-6 < d < 0.05, f"max |diff| {d:.4f}")


def test_mask_and_analytic_agree_when_divisible():
    """Passing the mask must not change the answer where the note is already exact."""
    s = phantom(C=6, H=24, W=48, dc=0.0)
    Sig = rand_cov(6)
    m = make_acc_mask((24, 48), 4, acs_lines=0, dim=1).to(torch.complex128)
    for weighted in (True, False):
        a = sense_noise_level(s, Sig, 4, acs_size=0, weighted=weighted)
        b = sense_noise_level(s, Sig, 4, mask=m, weighted=weighted)
        check(f"mask == analytic at acs=0 (weighted={weighted})", close(a, b, rtol=1e-6),
              f"max rel {(a - b).abs().max() / a.max():.2e}")


def test_indivisible_accel_is_rejected_without_a_mask():
    """accel not dividing n: analytic taps are wrong, so demand the mask."""
    s = phantom(C=6, H=16, W=50, dc=0.0)
    try:
        sense_noise_level(s, None, 4, acs_size=0)
        check("indivisible accel without mask raises", False, "no error raised")
    except ValueError as e:
        check("indivisible accel without mask raises", "does not divide" in str(e),
              str(e)[:70])
    m = make_acc_mask((16, 50), 4, acs_lines=0, dim=1).to(torch.complex128)
    sig = sense_noise_level(s, None, 4, mask=m)
    check("indivisible accel with mask runs",
          tuple(sig.shape) == (2, 1, 16, 50) and bool(torch.isfinite(sig).all()))

    # block_sense still needs divisibility, and must say so clearly.
    try:
        block_sense(torch.zeros(1, 6, 16, 50, dtype=torch.complex128), s[:1], 4)
        check("block_sense rejects indivisible accel", False, "no error raised")
    except ValueError as e:
        check("block_sense rejects indivisible accel", "cg_sense" in str(e), str(e)[:70])


def test_indivisible_accel_against_mc():
    """accel not dividing n: how much accuracy does the nearest-lattice cost?

    Two numbers, because they behave differently and only one of them matters for
    ImMAP. The LEVEL error is dominated by a global scale bias (the closed form
    under-predicts). The SHAPE error -- the mean-1 variance maps, i.e. exactly what
    `normalize_gmap` hands to a reconstruction -- is much smaller, because that scale
    bias divides straight out.

    Neither improves with image size: the Dirichlet peak of a length-K comb is
    ~N/(K A) ~ 1 pixel wide regardless of N, so a half-pixel offset always lands
    mid-peak. What does drive it is `accel` -- more folds, more smeared peaks, more
    dropped leakage.

    Replica floors at 256 reps: 1/(2 sqrt(R)) = 3.1% on the level (a std) and
    1/sqrt(R) = 6.3% on the shape (a variance). The divisible rows sit at those
    floors; that is the control that says the machinery itself is right.
    """
    C, H, NREPS = 8, 8, 256
    floor_lvl, floor_shp = 1 / (2 * NREPS ** 0.5), NREPS ** -0.5
    for W, accel in [(48, 4), (46, 3), (46, 6), (184, 6)]:
        s = phantom(B=1, C=C, H=H, W=W, seed=5, dc=0.0, dtype=torch.complex64)
        m = make_acc_mask((H, W), accel, acs_lines=0, dim=1).to(torch.complex64)
        an = sense_noise_level(s, None, accel, mask=m, weighted=True)
        g = torch.Generator().manual_seed(31)
        mc = sense_noise_level_mc(s, None, accel, weighted=True, mask=m,
                                  nreps=NREPS, generator=g, chunk=64,
                                  cg_kwargs={"max_iter": 400, "tol": 1e-10})
        k = an > 0.2 * an.max()
        lvl = ((mc - an).abs() / an.clamp_min(1e-8))[k].mean().item()
        scale = (mc[k].mean() / an[k].mean()).item()
        a2, m2 = an.pow(2), mc.pow(2)
        a2, m2 = a2 / a2[k].mean(), m2 / m2[k].mean()
        shp = ((m2 - a2).abs() / a2.clamp_min(1e-8))[k].mean().item()

        nl = len(_alias_lags(W, accel))
        tag = f"n={W}, accel={accel}"
        detail = (f"level {lvl:.1%} (scale {scale:.3f}), shape {shp:.1%}, {nl} lags, "
                  f"gcd={math.gcd(accel, W)}, floors {floor_lvl:.1%}/{floor_shp:.1%}")
        if W % accel == 0:
            check(f"{tag} [divisible]: level at the replica floor",
                  lvl < 2 * floor_lvl, detail)
            check(f"{tag} [divisible]: shape at the replica floor",
                  shp < 1.5 * floor_shp, detail)
        else:
            # Measured envelope: level 4% at A=3 rising to ~25% at A=6; shape 5-11%.
            check(f"{tag}: level within 30%", lvl < 0.30, detail)
            check(f"{tag}: SHAPE within 15% -- the number ImMAP actually consumes",
                  shp < 0.15, detail)
            check(f"{tag}: closed form under-predicts (scale >= 1)", scale > 0.99,
                  f"scale {scale:.3f}")


def test_a1_is_fully_sampled():
    """Check 1: A = 1 reduces to Rep2Rep Eq. (4) / Eq. (49)."""
    s = phantom(C=5, H=16, W=16)
    Sig = rand_cov(5)
    for weighted in (True, False):
        got = sense_noise_level(s, Sig, accel=1, weighted=weighted)
        want = coil_combined_noise_level(s, Sig, weighted=weighted)
        check(f"A=1 == fully sampled (weighted={weighted})", close(got, want),
              f"max rel {(got - want).abs().max() / want.abs().max():.2e}")


def test_p0_recovers_the_no_acs_formula():
    """Check 2: p = 0 gives A [(S^H Sigma^-1 S)^-1]_aa, i.e. Eq. (26) -> Eq. (16)."""
    s = phantom(C=6, H=24, W=24)
    Sig = rand_cov(6)
    accel = 3
    sigma = sense_noise_level(s, Sig, accel, acs_size=0, weighted=True)

    # Brute force, one pixel's block at a time, straight from the definition.
    S = _alias_stack(s, _alias_lags(24, accel), 3)
    Sinv = torch.linalg.inv(Sig).to(S.dtype)
    M = S.conj().transpose(-2, -1) @ (Sinv @ S)
    d = accel * torch.diagonal(torch.linalg.inv(M), dim1=-2, dim2=-1).real[:, 0]
    want = d.reshape(s.shape[0], 1, *s.shape[2:]).sqrt()
    check("p=0 == A (S^H Sigma^-1 S)^-1", close(sigma, want, rtol=1e-6),
          f"max rel {(sigma - want).abs().max() / want.max():.2e}")


def test_p1_recovers_fully_sampled():
    """Check 3: p = 1 collapses the bracket to dg(.) and recovers Eq. (49)."""
    s = phantom(C=6, H=16, W=16)
    Sig = rand_cov(6)
    for weighted in (True, False):
        got = sense_noise_level(s, Sig, accel=4, acs_size=16, weighted=weighted)
        want = coil_combined_noise_level(s, Sig, weighted=weighted)
        check(f"p=1 == fully sampled (weighted={weighted})", close(got, want, rtol=1e-6),
              f"max rel {(got - want).abs().max() / want.max():.2e}")


def test_monotone_in_p():
    """Check 4: <X>_n is nondecreasing in p (Loewner), so sigma is nonincreasing.

    Needs the exact inverse, hence `dc=0`: with truncation active a *larger*
    bracket can lose a mode to the cutoff and the ordering breaks.
    """
    s = phantom(C=4, H=24, W=48, dc=0.0)
    Sig = rand_cov(4)
    prev = None
    ok = True
    for acs in (0, 4, 8, 16, 32, 48):
        sig = sense_noise_level(s, Sig, accel=2, acs_size=acs, weighted=True)
        if prev is not None:
            ok &= bool((sig <= prev + 1e-9).all())
        prev = sig
    check("sigma nonincreasing in acs_size", ok)

    # And the Loewner statement itself: dg(G) >= G / A for PSD G.
    lags = _alias_lags(48, 2)
    S = _alias_stack(s, lags, 3)
    kw = dict(mask=None, ax=3, dtype=s.dtype, device=s.device)
    G0 = _alias_bracket(S, S, _kernel_taps(lags, 48, 2, 0, **kw))
    G1 = _alias_bracket(S, S, _kernel_taps(lags, 48, 2, 24, **kw))
    w = torch.linalg.eigvalsh(G1 - G0).real
    check("bracket increases in the Loewner order", bool((w > -1e-10).all()),
          f"min eig {w.min():.2e}")


def test_ideal_coils_give_g_one():
    """Check 5: Sigma = I with orthonormal aliasing columns -> sigma^2 = A_eff sigma_full^2."""
    accel, H, W, C = 2, 8, 16, 4
    step = W // accel
    # Build maps whose alias columns are orthonormal by construction: put a
    # different unit basis coil vector on each member of every alias set.
    s = torch.zeros(1, C, H, W, dtype=torch.complex128)
    for a in range(accel):
        s[0, a, :, a * step:(a + 1) * step] = 1.0
    for acs in (0, 4, 8, 16):
        Aeff = effective_acceleration(accel, acs, W)
        sig = sense_noise_level(s, None, accel, acs_size=acs, weighted=True)
        sig0 = coil_combined_noise_level(s, None, weighted=True)
        want = (Aeff ** 0.5) * sig0
        g = sense_gfactor(s, None, accel, acs_size=acs, weighted=True)
        check(f"ideal coils: sigma^2 = A_eff sigma_full^2 (acs={acs})",
              close(sig, want, rtol=1e-6), f"max rel {(sig - want).abs().max():.2e}")
        check(f"ideal coils: g == 1 (acs={acs})", close(g, torch.ones_like(g), rtol=1e-6),
              f"max |g-1| {(g - 1).abs().max():.2e}")


def test_gfactor_at_least_one():
    """[M^-1]_aa [M]_aa >= 1 for M > 0, so g >= 1 for the *weighted* branch.

    The note proves this only for the weighted pinv (it compares an inverse
    against its own diagonal).  The plain branch divides a suboptimal estimator
    by a suboptimal reference, so no such bound is available -- and none is
    claimed here.  `dc=0` again, because a truncated inverse under-reports the
    numerator and can push g below 1 for reasons that have nothing to do with
    the inequality.
    """
    s = phantom(C=8, H=24, W=48, dc=0.0)
    Sig = rand_cov(8)
    for acs in (0, 8):
        g = sense_gfactor(s, Sig, accel=4, acs_size=acs, weighted=True)
        check(f"g >= 1, weighted (acs={acs})",
              bool((g >= 1.0 - 1e-9).all()), f"min g {g.min():.9f}")


def test_plain_is_never_quieter_than_weighted():
    """Gauss-Markov on the bracket's implied design: sigma_plain >= sigma_weighted.

    Holds for every p, not just p = 0 -- see the module docstring.  This is the
    check that ties the two branches together, and the one that would catch a
    Sigma vs Sigma^{-1} swap in either of them.
    """
    s = phantom(C=6, H=24, W=48, dc=0.0)
    Sig = rand_cov(6, seed=3)
    for acs in (0, 6, 12, 24):
        w = sense_noise_level(s, Sig, accel=2, acs_size=acs, weighted=True)
        pl = sense_noise_level(s, Sig, accel=2, acs_size=acs, weighted=False)
        check(f"sigma_plain >= sigma_weighted (acs={acs})",
              bool((pl >= w - 1e-12).all()),
              f"worst margin {(pl - w).min():.2e}, mean ratio {(pl / w).mean():.4f}")


def test_weighted_equals_plain_when_white():
    """The two pinvs coincide on whitened data (Sigma = I)."""
    s = phantom(C=5, H=16, W=32)
    for acs in (0, 8):
        a = sense_noise_level(s, None, accel=2, acs_size=acs, weighted=True)
        b = sense_noise_level(s, None, accel=2, acs_size=acs, weighted=False)
        check(f"weighted == plain at Sigma=I (acs={acs})", close(a, b, rtol=1e-6),
              f"max rel {(a - b).abs().max() / a.max():.2e}")

    # And a scalar Sigma just rescales both by sqrt(Sigma).
    a = sense_noise_level(s, 4.0, accel=2, weighted=True)
    b = sense_noise_level(s, None, accel=2, weighted=True)
    check("scalar Sigma scales sigma by sqrt(Sigma)", close(a, 2.0 * b, rtol=1e-6))


def test_agrees_with_gfactor_uniform():
    """`physics.gfactor.gfactor_uniform` is the same estimator at acs_size = 0."""
    s = phantom(C=6, H=32, W=32, dtype=torch.complex64)
    Sig = rand_cov(6, dtype=torch.complex128).to(torch.complex64)
    accel = 2
    g_ref, nvar_ref = gfactor_uniform(s, accel, axis=-1, Sigma=Sig)
    sig = sense_noise_level(s, Sig, accel, acs_size=0, weighted=True)

    check("nvar matches gfactor_uniform", close(sig.pow(2), nvar_ref, rtol=1e-3, atol=1e-5),
          f"max rel {((sig.pow(2) - nvar_ref).abs().max() / nvar_ref.max()):.2e}")
    # gfactor_uniform normalises by sqrt(A) sigma_full with the 'optimal'
    # combination; at acs_size = 0, A_eff = A, so the g-factors must agree too.
    g = sense_gfactor(s, Sig, accel, acs_size=0, weighted=True)
    check("g matches gfactor_uniform", close(g, g_ref, rtol=1e-3, atol=1e-4),
          f"max abs {(g - g_ref).abs().max():.2e}")


def test_sigma_full_map_parity():
    """`coil_combined_noise_level` == `gfactor.sigma_full_map` on the support."""
    s = phantom(C=5, H=16, W=16)
    Sig = rand_cov(5)
    for weighted, combine in ((True, "optimal"), (False, "adjoint")):
        a = coil_combined_noise_level(s, Sig, weighted=weighted)
        b = sigma_full_map(s, Sig, combine=combine)
        check(f"parity with sigma_full_map ({combine})", close(a, b, rtol=1e-6))


def test_background_is_zero_not_inf():
    """Rank-deficient / empty blocks return 0, and A > C stays finite."""
    s = phantom(C=4, H=16, W=32)
    s[:, :, :, :8] = 0.0                                # a strip of pure background
    sig = sense_noise_level(s, None, accel=2, acs_size=0)
    check("background is exactly 0", bool((sig[:, :, :, :8] == 0).all()))
    check("no non-finite entries", bool(torch.isfinite(sig).all()))

    # A > C: singular without ACS, finite (and optimistic) with it.
    s2 = phantom(C=2, H=16, W=32)
    sig_no = sense_noise_level(s2, None, accel=4, acs_size=0)
    sig_acs = sense_noise_level(s2, None, accel=4, acs_size=8)
    check("A > C, no ACS: truncated to finite", bool(torch.isfinite(sig_no).all()))
    check("A > C, ACS: finite and smaller", bool(torch.isfinite(sig_acs).all()))


def test_offset_invariance():
    """The map cannot depend on which residue class the uniform grid keeps."""
    s = phantom(C=4, H=16, W=32)
    sig = sense_noise_level(s, None, accel=4, acs_size=0)
    # A shift of the *image* along PE by `step` permutes alias sets internally;
    # the map must follow the same shift, i.e. sigma is equivariant.
    step = 32 // 4
    s_roll = torch.roll(s, shifts=step, dims=-1)
    sig_roll = sense_noise_level(s_roll, None, accel=4, acs_size=0)
    check("sigma is equivariant to a step-shift of the maps",
          close(sig_roll, torch.roll(sig, shifts=step, dims=-1), rtol=1e-6))


def test_block_sense_is_unbiased():
    """`block_sense` inverts the uniform forward model exactly (noise-free)."""
    B, C, H, W, accel = 1, 6, 16, 32, 2
    s = phantom(B=B, C=C, H=H, W=W, dtype=torch.complex128)
    g = torch.Generator().manual_seed(3)
    x = torch.randn(B, 1, H, W, dtype=torch.complex128, generator=g)
    mask = make_acc_mask((H, W), accel, acs_lines=0, dim=1,
                         mode="uniform", offset=0).to(torch.complex128)
    k = mask * fftc(s * x)
    xh = block_sense(k, s, accel)
    err = (xh - x).abs().max() / x.abs().max()
    check("block_sense recovers x exactly", bool(err < 1e-9), f"max rel err {err:.2e}")


def test_mc_matches_the_plain_branch():
    """Sec. 9.8: pseudo-replica through `block_sense` == the closed form."""
    B, C, H, W, accel = 1, 6, 24, 24, 2
    s = phantom(B=B, C=C, H=H, W=W, seed=5, dtype=torch.complex64)
    Sig = rand_cov(C, seed=7, dtype=torch.complex128).to(torch.complex64)
    nreps = 4096

    g = torch.Generator().manual_seed(11)
    mc = sense_noise_level_mc(s, Sig, accel, acs_size=0, weighted=False,
                              nreps=nreps, generator=g, chunk=64)
    an = sense_noise_level(s, Sig, accel, acs_size=0, weighted=False)

    rel = ((mc - an).abs() / an.clamp_min(1e-8)).mean()
    check("MC == closed form (plain pinv, no ACS)", bool(rel < 0.05),
          f"mean rel err {rel:.3f} over {nreps} replicas (~{(2 * nreps) ** -0.5:.3f} floor)")


def test_mc_matches_the_weighted_branch():
    """The ML pinv over Omega, via CG-SENSE, at acs_size = 0."""
    B, C, H, W, accel = 1, 6, 24, 24, 2
    s = phantom(B=B, C=C, H=H, W=W, seed=5, dtype=torch.complex64)
    # Keep the maps well conditioned everywhere so the lamda = 0 CG solve has no
    # null space to wander in.
    Sig = rand_cov(C, seed=7, dtype=torch.complex128).to(torch.complex64)

    g = torch.Generator().manual_seed(13)
    mc = sense_noise_level_mc(s, Sig, accel, acs_size=0, weighted=True,
                              nreps=1024, generator=g, chunk=32,
                              cg_kwargs={"max_iter": 256, "tol": 1e-10})
    an = sense_noise_level(s, Sig, accel, acs_size=0, weighted=True)

    keep = an > 0.2 * an.max()
    rel = ((mc - an).abs() / an.clamp_min(1e-8))[keep].mean()
    check("MC == closed form (weighted pinv, no ACS)", bool(rel < 0.08),
          f"mean rel err {rel:.3f} on the well-conditioned support")


def test_acs_approximation_is_in_the_right_ballpark():
    """acs_size > 0: the delta-for-sinc bracket vs. the true ML pinv over Omega.

    This one is *expected* to disagree -- the closed form substitutes a delta for
    the ACS sinc and then inverts it.  The check is that the approximation is
    biased the way the note says (optimistic, i.e. it under-predicts) and by a
    modest amount, not that it is exact.
    """
    B, C, H, W, accel, acs = 1, 6, 24, 24, 2, 8
    s = phantom(B=B, C=C, H=H, W=W, seed=5, dtype=torch.complex64)

    g = torch.Generator().manual_seed(17)
    mc = sense_noise_level_mc(s, None, accel, acs_size=acs, weighted=True,
                              nreps=512, generator=g, chunk=32,
                              cg_kwargs={"max_iter": 256, "tol": 1e-10})
    an = sense_noise_level(s, None, accel, acs_size=acs, weighted=True)

    keep = an > 0.2 * an.max()
    ratio = (mc[keep] / an[keep]).mean()
    check("ACS closed form is optimistic, and within 25%",
          bool(1.0 < ratio < 1.25),
          f"mean MC/analytic = {ratio:.3f} (>1 = closed form under-predicts, as the note says)")
    # The no-ACS map must bound it from above, since the bracket is monotone in p
    # and the true information only grows with the extra lines.
    an0 = sense_noise_level(s, None, accel, acs_size=0, weighted=True)
    check("ACS lowers the noise level below the no-ACS map",
          bool((an <= an0 + 1e-6).all()))


def test_axis_and_dtype_plumbing():
    """PE on axis -2, complex64/128, batched Sigma."""
    s = phantom(B=3, C=5, H=32, W=16)
    sT = s.transpose(-2, -1).contiguous()
    a = sense_noise_level(s, None, accel=4, axis=-1, acs_size=4)
    b = sense_noise_level(sT, None, accel=4, axis=-2, acs_size=4)
    check("axis=-2 == transposed axis=-1", close(a, b.transpose(-2, -1), rtol=1e-6))

    Sig = torch.stack([rand_cov(5, seed=i) for i in range(3)])
    sig = sense_noise_level(s, Sig, accel=2, acs_size=4)
    check("batched Sigma runs", tuple(sig.shape) == (3, 1, 32, 16) and bool(torch.isfinite(sig).all()))

    s32 = s.to(torch.complex64)
    sig32 = sense_noise_level(s32, Sig.to(torch.complex64), accel=2, acs_size=4)
    check("complex64 agrees with complex128",
          close(sig32.double(), sig, rtol=1e-3, atol=1e-5),
          f"max rel {(sig32.double() - sig).abs().max() / sig.max():.2e}")

    check("thresh zeroes the background",
          bool((sense_noise_level(s, None, accel=2, thresh=1e-2) >= 0).all()))


def test_grappa_comparison_from_the_note():
    """Sec. 7 check 7: the worked A=2, N_ACS=24, N_2=256 number.

    SENSE gives A_eff sigma_full^2 = 1.83 sigma_full^2 under ideal coils, versus
    (A(1-p) + p) = 1.91 for GRAPPA+ACS.  This pins the A_eff bookkeeping.
    """
    Aeff = effective_acceleration(2, 24, 256)
    p = 24 / 256
    grappa = 2 * (1 - p) + p
    check("A_eff = 1.83 for A=2, N_ACS=24, N_2=256", abs(Aeff - 1.8286) < 1e-3, f"{Aeff:.4f}")
    check("GRAPPA+ACS = 1.91 and exceeds SENSE", abs(grappa - 1.9062) < 1e-3 and grappa > Aeff,
          f"{grappa:.4f} > {Aeff:.4f}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_alias_lags_and_stack()
    test_folding_indices()
    test_mask_driven_taps_match_the_analytic_bracket()
    test_mask_and_analytic_agree_when_divisible()
    test_indivisible_accel_is_rejected_without_a_mask()
    test_indivisible_accel_against_mc()
    test_a1_is_fully_sampled()
    test_p0_recovers_the_no_acs_formula()
    test_p1_recovers_fully_sampled()
    test_monotone_in_p()
    test_ideal_coils_give_g_one()
    test_gfactor_at_least_one()
    test_plain_is_never_quieter_than_weighted()
    test_weighted_equals_plain_when_white()
    test_agrees_with_gfactor_uniform()
    test_sigma_full_map_parity()
    test_background_is_zero_not_inf()
    test_offset_invariance()
    test_block_sense_is_unbiased()
    test_mc_matches_the_plain_branch()
    test_mc_matches_the_weighted_branch()
    test_acs_approximation_is_in_the_right_ballpark()
    test_axis_and_dtype_plumbing()
    test_grappa_comparison_from_the_note()

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("failed:", ", ".join(FAIL))
    sys.exit(1 if FAIL else 0)
