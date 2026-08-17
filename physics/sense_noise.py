"""
SENSE noise-level maps under uniform undersampling, with and without ACS.

PyTorch port of `sense_noiselevel/sense_noise.jl` and its companion note
(`sense_noise.tex`), which derives sqrt(diag Cov(x_hat)) for a SENSE
reconstruction -- x_hat = E^dagger k, E = I_Omega F S -- from uniform A-fold
undersampling along the phase-encode axis, optionally with a fully sampled
central ACS band of `acs_size` lines.  This is the SENSE counterpart of
`physics.gfactor`: that module characterises GRAPPA-style fixed operators and
Monte-Carlo reconstructions, this one the pseudoinverse.

Theory in one screen
--------------------
Uniform undersampling folds the FOV, so E^H Psi^{-1} E is block-diagonal over
*alias sets*: the A pixels n^(a) = n_pe + a * (N_pe / A) that land on each other.
With

    S_n = [s[n^(1)] ... s[n^(A)]]  in  C^{C x A},        p = N_ACS / N_pe,
    <X>_n = (1-p)/A * S_n^H X S_n  +  p * dg(S_n^H X S_n),

where dg(.) keeps only the diagonal, the covariance restricted to one alias set is

    weighted / pre-whitened pinv (ML, attains the CRB):
        Cov(x_hat)|_n = <Sigma^{-1}>_n^{-1}
    plain pinv (what a block-SENSE solve on un-whitened data computes):
        Cov(x_hat)|_n = <I>_n^{-1} <Sigma>_n <I>_n^{-1}

and sigma[n] is the square root of the a-th diagonal entry.  The ACS band enters
*additively in the information* (the bracket), not additively in the covariance
the way it does for GRAPPA -- that is the structural difference between the two,
and it is why SENSE+ACS is always the quieter of the two.  The bracket itself is
an approximation: it replaces the ACS sinc in M_Omega^conv by a delta at its peak
amplitude p, exactly as in the GRAPPA+ACS derivation.  Since the substituted
quantity is then *inverted*, treat the closed form as reliable for A <= C with a
well-conditioned uniform part, and as optimistic otherwise (e.g. it stays
invertible at A > C, where the true normal operator is not).

How optimistic, measured: `notebooks/sense_noise_testbench.ipynb` compares against
a CG-SENSE solve over the true M_Omega.  On an 8-coil 96x96 phantom with 16 ACS
lines, sigma comes out low by 3% at A = 2, 11% at A = 3 and 51% at A = 4 -- and at
A = 4 the per-pixel spread of the error is 0.23, so the *shape* is wrong there too,
not just the scale.  Note A = 4 <= C = 8, so this is not the A > C failure mode; the
error tracks the conditioning of the uniform part, because that is what decides how
much the fictitious full-resolution ACS information is doing.  At acs_size = 0 the
same bench reproduces sigma to within replica noise (ratio 0.999 +/- 0.031 at 256
replicas, against a predicted 1/(2 sqrt(R)) = 0.031).  So: with ACS at high
acceleration, read these maps as a lower bound on the noise.

Sanity checks the derivation must satisfy, all covered in
`tests/test_sense_noise.py`: p = 0 recovers A * [(S_n^H Sigma^{-1} S_n)^{-1}]_aa;
p = 1 recovers the fully sampled (s^H Sigma^{-1} s)^{-1}; A = 1 recovers Rep2Rep
Eq. (4); the bracket is nondecreasing in p in the Loewner order, so including the
ACS never raises the noise level.

One more, not in the note but worth knowing because it is what pins the two
branches to each other: the three brackets are the exact moments of a genuine
linear model, namely the design D stacking sqrt((1-p)/A) S_n against A separate
rows sqrt(p) s[n^(a)] e_a^T (one per ACS-constrained pixel), each observed
through Sigma.  Then D^H D = <I>_n, D^H Sigma D = <Sigma>_n and
D^H Sigma^{-1} D = <Sigma^{-1}>_n, so Gauss-Markov applies term by term and

    <I>_n^{-1} <Sigma>_n <I>_n^{-1}  >=  <Sigma^{-1}>_n^{-1}   (Loewner)

for every p, not just p = 0: the plain branch never reports less noise than the
weighted one.  So the delta-for-sinc substitution, crude as it is, at least
produces an internally consistent observation model rather than an ad-hoc
formula.

Caveat: the diagonal is not the whole story.  Without ACS the covariance is
supported *exactly* on the aliasing lattice (A - 1 nonzero off-diagonals per row,
at PE offsets +/- j N_pe / A) and is zero between alias sets; adding ACS makes it
dense, with local correlations on a ~1/p pixel scale.  As with GRAPPA+ACS that
matters for SURE-type losses, not for Rep2Rep.

Conventions
-----------
Tensors are the repo standard (B, C, H, W) with maps out as (B, 1, H, W) real,
so the Julia (Nx, Ny, C, B) / PE-on-axis-2 layout becomes PE on `axis=-1` by
default -- which is what `physics.mask.make_acc_mask(..., dim=1)` produces and
what `physics.gfactor.gfactor_uniform` already assumes.  `Sigma` follows
`physics.gfactor`: None (identity), a scalar *variance*, (C, C), or (B, C, C).
Everything returns a noise *level* (a standard deviation) in the units of the
k-space noise, not a relative g-factor; `sense_gfactor` gives the latter.

One deliberate departure from the Julia: the reference `sense_noise_level_mc`
pushes replicas through Sljiva's `blocksense`, whose `Fourier` operator is
`fft`/`ifft` rather than an orthonormal pair.  F is then not unitary, so k-space
noise of covariance Sigma becomes image noise of covariance Sigma/N and the MC
map comes out a factor sqrt(N) below the analytic one.  `block_sense` here uses
the repo's ortho `fftc`/`ifftc`, under which MC and closed form agree exactly (up
to replica error) -- which is the point of the check.

Beyond the note
---------------
Two generalisations the Julia does not have, both needed for real masks.

*The bracket is a Hadamard product with the mask's own kernel.*  Writing
`m = ifft(mask)` for the image-domain kernel of `M^conv = F^H M F`,

    <X>_n = G (o) T,   G[a,a'] = s[n+l_a]^H X s[n+l_a'],  T[a,a'] = m[l_a - l_a'],

for alias offsets `l_0 = 0, l_1, ...`.  Pass `mask=` and the taps are read off the
actual pattern by one inverse FFT instead of assumed; the note's
`(1-p)/A * G + p * dg(G)` is the special case `m[0] = 1/A_eff`,
`m[j N/A] = (1-p)/A`.  It stops caring whether the ACS is one centred contiguous
block, and it makes the Hermitian-PSD property structural (`m` is the transform of
a nonnegative function, so `T` is PSD by Bochner; `G` is PSD; Schur product
theorem) rather than something to hope for -- which is what the `eigh` path needs.

Doing it this way also shows the note undersells its own bracket.  The ACS band
contributes `sum_{k in ACS} exp(2 pi i k l / N)` at lag `l`; on the lattice
`l = j N/A` that phase advances by `j/A` per line, so `acs_size` consecutive lines
sum to *exactly* zero whenever `A | acs_size`.  The `(1-p)/A` off-diagonal is then
not an approximation at all, and neither is `m[0] = 1/A_eff`: the whole tap matrix
is exact.  When `A` does not divide `acs_size` the taps differ by a sidelobe, under
0.05 in the cases tested.  So the "delta-for-sinc" substitution is *not* where the
ACS branch's error comes from.  What it actually drops is the coupling at lags off
the alias lattice, where the sinc also has support -- the covariance is dense and
only its lattice part is being modelled.  That is the approximation to distrust.

*`accel` need not divide the PE length.*  `arange(off, N, A)` is a coset of a
subgroup of Z_N only when `A | N`; otherwise the exact deltas sit at multiples of
`N/gcd(A,N)` and the other folds smear into Dirichlet lobes near `j N/A`.  We take
the integers *bracketing* each `j N/A` (both, not the nearer one) with the exact
taps there, so the block grows from `A` to at most `2A - 1` lags.  Rounding to one
neighbour is not good enough and, importantly, does not get better on bigger
images: the Dirichlet peak of a length-K comb is `~N/(K A) ~ 1` pixel wide
*regardless of N*, so a half-pixel offset always lands mid-peak and the single tap
it keeps is down by `sinc(delta)` -- up to 36%.  Bracketing recovered 27% -> 18%
(N=46) and 35% -> 24% (N=184) at A=6.

Even bracketed, the alias sets no longer partition the image, so this is a *local*
inverse -- the submatrix of the normal operator on one pixel's partner list -- and
leakage onto unmodelled lags is still dropped.  Measured against CG-SENSE replicas,
8 coils, `acs_size=0` (`test_indivisible_accel_against_mc`):

    N=48, A=4  (A | N)   level  2.8%   shape  5.5%   <- both at the replica floor
    N=46, A=3            level  4.1%   shape  5.1%   <- also at the floor
    N=46, A=6            level 18.2%   shape  9.5%
    N=184, A=6           level 24.5%   shape 11.1%

The level error is mostly a global scale bias (the closed form under-predicts, by
1.18x and 1.25x on those two rows), and it is driven by `A` -- more folds, more
smeared peaks -- not by N.  The *shape* error is much smaller because that bias
divides out, which is the number that matters if the map is going through
`physics.gfactor.normalize_gmap` into a reconstruction: there only the mean-1 shape
survives.  If you need the absolute level right at an indivisible `accel`, use
`physics.gfactor.gfactor_replica`.

`block_sense` still requires `A | N`, since a folded image of `N/A` pixels only
exists then; use `physics.gfactor.cg_sense` otherwise.

Cost: one batched A x A Hermitian eigendecomposition per *pixel* (not per folded
pixel), i.e. H * W * B of them -- A times the folded cost, in exchange for handling
any acceleration.  Still a small multiple of a single coil combination; `chunk`
caps peak memory.  (The Julia uses a truncated SVD; for the Hermitian PSD brackets
built here `eigh` is the same decomposition, cheaper, and is what
`physics.gfactor` already uses.)
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence, Tuple

import torch
from torch import Tensor

from operators.fourier import fftc, ifftc
from physics.gfactor import (
    _as_cov,
    _herm_pow,
    sample_kspace_noise,
    support_mask,
    whiten_smaps,
)


__all__ = [
    "effective_acceleration",
    "coil_combined_noise_level",
    "sense_noise_level",
    "sense_gfactor",
    "block_sense",
    "sense_noise_level_mc",
]


# ---------------------------------------------------------------------------
# Alias sets and the sampling kernel
# ---------------------------------------------------------------------------
#
# The note writes the bracket as (1-p)/A * S_n^H X S_n + p * dg(S_n^H X S_n).  That
# is a special case of something more useful.  For ANY mask,
#
#     [S^H (M^conv (x) X) S]_{n,n'} = s[n]^H X s[n'] * m[n - n'],
#     m = the image-domain kernel of M^conv = F^H M F,
#
# so restricting to an alias set with PE offsets l_0 = 0, l_1, ..., l_{A-1} gives
#
#     <X>_n = G (o) T,     G[a,a'] = s[n + l_a]^H X s[n + l_a'],
#                          T[a,a'] = m[l_a - l_a']          (Hadamard product)
#
# i.e. the Gram elementwise-multiplied by the Toeplitz matrix of the kernel taps.
# Two things follow.
#
# (1) The taps can be read off the ACTUAL mask by one inverse FFT instead of being
#     assumed.  Checking it reproduces the note: for A | N with no ACS the taps are
#     m[j N/A] = 1/A, so T = ones/A and <X> = G/A.  For a centred ACS band,
#     m[0] = |Omega|/N = 1/A + p - p/A and m[j N/A] = (1-p)/A + O(sinc sidelobe), so
#     T = (1-p)/A * ones + p * I and the Hadamard form IS the note's bracket -- but
#     with the sidelobes kept rather than dropped, and with no assumption that the
#     ACS is one centred contiguous block.
#
# (2) It is automatically Hermitian PSD, which is what `eigh` downstream needs.  m is
#     the inverse transform of a nonnegative function, so it is a positive-definite
#     sequence and T is PSD (Bochner); G is PSD; the Schur product theorem does the
#     rest.  The previous hand-built form had no such guarantee.
#
# When accel does not divide N the lattice {j N/A} is not integral and, worse, not a
# subgroup of Z_N -- `arange(off, N, A)` is a coset of a subgroup only when A | N.
# The exact deltas then sit only at multiples of N/gcd(A, N) and the remaining folds
# smear into Dirichlet lobes near j N/A.  We take the A nearest-integer offsets
# round(j N/A) and the exact taps at those offsets.  Two consequences worth being
# explicit about: the alias sets no longer partition the image, so this is a *local*
# inverse (the submatrix of the normal operator on one pixel's partner list) rather
# than an exact block factorisation; and leakage onto unmodelled lags is dropped.
# `tests/test_sense_noise.py::test_indivisible_accel_against_mc` measures what that
# costs -- a few percent, not a factor.

def _alias_lags(n: int, accel: int) -> Tuple[int, ...]:
    """PE offsets of one pixel's alias partners, `l_0 = 0` first.

    Exactly `{j * n // accel}` when `accel | n`; nearest integers otherwise, with
    collisions dropped (so a very large `accel` yields fewer than `accel` lags).
    """
    seen, lags = set(), []
    for j in range(accel):
        x = j * n / accel
        # When x is not an integer the aliasing peak straddles two pixels, so take
        # BOTH. Rounding to the nearer one is not good enough: the Dirichlet peak of
        # a length-K comb is ~N/(K A) ~ 1 pixel wide *regardless of N*, so a half-
        # pixel offset always lands mid-peak and the single tap it keeps is down by
        # sinc(delta) -- up to 36% -- with the rest of the energy on the neighbour.
        # That error does not shrink as the image grows (measured: 27% at n=46 and
        # 35% at n=184, both accel=6). Bracketing recovers it and costs a bigger
        # block, at most 2A - 1 lags instead of A.
        for l in (int(math.floor(x)) % n, int(math.ceil(x)) % n):
            if l not in seen:
                seen.add(l)
                lags.append(l)
    return tuple(lags)


def _pe_profile(mask: Tensor, ax: int, n: int) -> Tensor:
    """Reduce a sampling mask to its 1-D profile along `ax`, as a float in {0, 1}.

    Requires separability -- a mask that varies along the readout is not a 1-D
    Cartesian pattern and none of this applies to it.
    """
    m = (mask != 0)
    while m.dim() < 4:
        m = m.unsqueeze(0)
    m = m.movedim(ax, -1)
    flat = m.reshape(-1, n)
    prof = flat.any(dim=0)
    if not torch.equal(prof, flat.all(dim=0)):
        raise ValueError(
            "mask is not separable along the phase-encode axis (it varies along the "
            "readout), so it is not a 1-D Cartesian pattern; use "
            "physics.gfactor.gfactor_replica for such masks")
    # float64 on purpose: the taps are one length-n FFT, so precision here is free,
    # and in float32 the ACS band's exact cancellation on the alias lattice only
    # closes to ~1e-8 -- enough to muddy an exactness check.
    return prof.to(torch.float64)


def _kernel_taps(lags: Sequence[int], n: int, accel: int, acs_size: int,
                 mask: Optional[Tensor], ax: int, dtype, device) -> Tensor:
    """The `(A, A)` Hermitian tap matrix `T[a,a'] = m[l_a - l_a']`.

    With `mask=None` the analytic delta-for-sinc taps of the note are used, which
    presume the uniform-plus-centred-ACS pattern *and* `accel | n`.  With a mask they
    are exact for that mask: `m = ifft(ifftshift(profile))`, matching the repo's
    ortho `fftc` (which fftshifts), and `m[0] = mean(mask) = 1/A_eff`.
    """
    A = len(lags)
    L = torch.as_tensor(lags, device=device)
    D = (L[:, None] - L[None, :]) % n                      # (A, A) lag differences

    if mask is None:
        if n % accel:
            raise ValueError(
                f"accel={accel} does not divide the phase-encode length {n}, so the "
                "analytic taps do not apply. Pass the actual sampling `mask=` and the "
                "kernel is read from it exactly.")
        p = float(acs_size) / float(n)
        off = (1.0 - p) / accel
        T = torch.full((A, A), off, dtype=dtype, device=device)
        T += p * torch.eye(A, dtype=dtype, device=device)
        return T

    prof = _pe_profile(mask, ax, n).to(device)
    # F = fftc includes an fftshift, so undo it before the transform; then m[l] is
    # indexed by lag with m[0] = mean(mask).
    m = torch.fft.ifft(torch.fft.ifftshift(prof))
    return m[D].to(dtype)


def _alias_stack(x: Tensor, lags: Sequence[int], ax: int) -> Tensor:
    """`(B, C, H, W) -> (B * npix, C, A)` holding `s[n + l_a]` for every pixel `n`.

    A roll per lag rather than the free reshape the divisible case allows: every
    pixel gets its own block, so this is `A` times the work of the folded version
    (still a small multiple of one coil combination) and it handles any `accel`.
    """
    P = torch.stack([torch.roll(x, shifts=-int(l), dims=ax) for l in lags], dim=-1)
    C = x.shape[1]
    P = P.movedim(1, -2)                                   # (B, H, W, C, A)
    return P.reshape(-1, C, len(lags))


def _alias_bracket(P: Tensor, Q: Tensor, T: Tensor) -> Tensor:
    """`<X>_n = (P^H Q) (o) T`, (npix, A, A) Hermitian PSD."""
    return (P.conj().transpose(-2, -1) @ Q) * T


# The fold/unfold pair below is used only by `block_sense`, which genuinely operates
# on the folded measurement -- a reduced-FOV image of n/accel pixels, which exists
# only when accel divides n.  `sense_noise_level` no longer needs it: one block per
# pixel is both simpler and general.

def _fold_alias(x: Tensor, accel: int, axis: int) -> Tuple[Tensor, Tuple[int, ...], int, int]:
    """`(B, C, H, W) -> (B * L * step, C, accel)`, alias axis for free.

    A row-major reshape of the PE axis into `(accel, step)` puts index
    `a * step + jj` at `[a, jj]`, so the alias set is exactly the `accel` axis --
    the same trick the Julia gets from a column-major
    `reshape(smaps, Nx, space, accel, C, B)`.  Returns `(S, lead, n, ax)`.
    """
    if x.dim() != 4:
        raise ValueError(f"expected (B, C, H, W); got {tuple(x.shape)}")
    ax = axis % x.dim()
    if ax < 2:
        raise ValueError("axis must be a spatial axis (2 or 3, or -1/-2)")

    s = x.movedim(ax, -1)                          # (B, C, *lead, n)
    B, C = s.shape[0], s.shape[1]
    lead = tuple(s.shape[2:-1])
    n = s.shape[-1]
    if n % accel:
        raise ValueError(
            f"accel={accel} does not divide axis length {n}; block_sense needs a "
            "folded image of n/accel pixels. Use physics.gfactor.cg_sense instead "
            "(sense_noise_level itself has no such restriction).")
    step = n // accel

    S = s.reshape(B, C, -1, accel, step)           # (B, C, L, accel, step)
    S = S.permute(0, 2, 4, 1, 3)                   # (B, L, step, C, accel)
    return S.reshape(-1, C, accel), lead, n, ax


def _unfold_alias(d: Tensor, B: int, lead: Sequence[int], n: int,
                  accel: int, ax: int) -> Tensor:
    """`(B * L * step, accel) -> (B, 1, H, W)`, undoing `_fold_alias`."""
    step = n // accel
    t = d.reshape(B, -1, step, accel).permute(0, 1, 3, 2)   # (B, L, accel, step)
    return t.reshape(B, 1, *lead, n).movedim(-1, ax)


# ---------------------------------------------------------------------------
# Batched A x A linear algebra, rank-truncated
# ---------------------------------------------------------------------------
#
# The per-block inverse is truncated rather than exact for robustness, not
# accuracy: background pixels have <X>_n = 0 and near-degenerate coil geometry
# (or A > C without ACS) is rank-deficient.  Truncation returns 0 there instead
# of inf/nan, which keeps the maps usable as side information for a
# noise-adaptive denoiser.

def _inv_spectrum(w: Tensor, rtol: float) -> Tensor:
    """1/w with modes below `rtol * w_max` (per block) zeroed."""
    wmax = w.amax(dim=-1, keepdim=True)
    tiny = torch.finfo(w.dtype).tiny
    inv = w.clamp_min(tiny).reciprocal()
    return torch.where(w > rtol * wmax, inv, torch.zeros_like(w))


def _batched_inv_diag(M: Tensor, rtol: float = 1e-6) -> Tensor:
    """`diag(M^{-1})` for batched Hermitian PSD `M`, (nblk, n) real.

    For M = U diag(w) U^H, [M^{-1}]_aa = sum_k |U_ak|^2 / w_k -- real and
    nonnegative by construction, no `.real` fudge needed.
    """
    w, U = torch.linalg.eigh(M)
    wi = _inv_spectrum(w.real, rtol)
    return (U.abs().pow(2) * wi.unsqueeze(-2)).sum(dim=-1)


def _batched_pinv(M: Tensor, rtol: float = 1e-6) -> Tensor:
    """Truncated pseudo-inverse of batched Hermitian PSD `M`."""
    w, U = torch.linalg.eigh(M)
    wi = _inv_spectrum(w.real, rtol).to(U.dtype)
    return U @ (wi.unsqueeze(-1) * U.conj().transpose(-2, -1))


def _batched_diag(M: Tensor) -> Tensor:
    """Real part of the diagonal of a batched square matrix, (nblk, n)."""
    return torch.diagonal(M, dim1=-2, dim2=-1).real


def _apply_cov(X: Optional[Tensor], smaps: Tensor) -> Tensor:
    """`X` applied pixel-wise across the coil axis; `None` is the identity."""
    if X is None:
        return smaps
    C = smaps.shape[1]
    X = X.expand(smaps.shape[0], C, C)
    return torch.einsum("bde,bexy->bdxy", X, smaps)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def effective_acceleration(accel: int = 0, acs_size: int = 0, n_pe: int = 1,
                           mask: Optional[Tensor] = None) -> float:
    """`A_eff`, the reciprocal of the acquired fraction of k-space.

    With a `mask` this is just `1 / mean(mask)` -- exact for any pattern, and the
    only form that is right when `accel` does not divide `n_pe` (there
    `arange(off, n, accel)` keeps `ceil(n / accel)` lines, not `n / accel`).

    Without one it falls back to the note's `A / (1 + p (A - 1))`,
    `p = acs_size / n_pe`, which is the central amplitude of M_Omega^conv,
    `1/A + p - p/A`, and presumes `accel | n_pe`.

    This is the SNR penalty attributable to *scan time*; what is left in the
    g-factor is then purely the conditioning penalty, which is why
    `sense_gfactor` divides by sqrt(A_eff) rather than sqrt(A).
    """
    if mask is not None:
        frac = (mask != 0).to(torch.float64).mean().item()
        if frac <= 0:
            raise ValueError("mask is empty")
        return 1.0 / frac
    p = float(acs_size) / float(n_pe)
    return float(accel) / (1.0 + p * (accel - 1))


def coil_combined_noise_level(smaps: Tensor, Sigma=None, weighted: bool = True,
                             eps: float = 1e-12) -> Tensor:
    """Fully sampled coil-combined noise level, (B, 1, H, W) real.

    `weighted=True`   (s[n]^H Sigma^{-1} s[n])^{-1/2}  -- pre-whitened / optimal
                      combination, Rep2Rep Eq. (49); the reference the
                      Pruessmann g-factor is defined against.
    `weighted=False`  (s[n]^H Sigma s[n])^{1/2}        -- matched filter (the
                      naive S^H combination), Rep2Rep Eq. (4).

    The two coincide for Sigma proportional to the identity with unit-RSS maps.

    Same quantity as `physics.gfactor.sigma_full_map`, but with the Julia's
    background convention: the weighted branch returns exactly 0 where the coil
    energy vanishes, rather than the large value an eps-floored `rsqrt` gives.
    That is what makes it safe as the denominator in `sense_gfactor`.
    """
    rdtype = smaps.real.dtype
    if weighted:
        sw = whiten_smaps(smaps, Sigma, eps=eps)
        q = sw.abs().pow(2).sum(dim=1, keepdim=True)
        floor = torch.finfo(rdtype).eps
        return torch.where(q > floor, q.clamp_min(floor).rsqrt(), torch.zeros_like(q))
    if Sigma is None:
        return smaps.abs().pow(2).sum(dim=1, keepdim=True).sqrt()
    Sig = _as_cov(Sigma, smaps.shape[1], smaps.device, smaps.dtype)
    q = (smaps.conj() * _apply_cov(Sig, smaps)).sum(dim=1, keepdim=True).real
    return q.clamp_min(0.0).sqrt()


def sense_noise_level(
    smaps: Tensor,
    Sigma=None,
    accel: int = 2,
    axis: int = -1,
    acs_size: int = 0,
    mask: Optional[Tensor] = None,
    weighted: bool = True,
    rtol: float = 1e-6,
    thresh: float = 0.0,
    eps: float = 1e-12,
    chunk: int = 1 << 20,
) -> Tensor:
    """Spatially varying image-domain noise level of a SENSE reconstruction.

    Parameters
    ----------
    smaps : (B, C, H, W) complex
        Coil sensitivities, assumed unit-RSS across coils (`|s[n]|_2 = 1`).  The
        normalisation is what makes `dg(S_n^H S_n) = I_A`, hence
        `<I>_n = (1-p)/A S_n^H S_n + p I_A` in the plain branch.
    Sigma : None | float | (C, C) | (B, C, C)
        k-space coil noise covariance; None means the identity, i.e. the map
        comes back relative to the per-coil noise std.
    accel : int
        Uniform acceleration A along `axis`.  Need not divide the axis length --
        see `mask`.
    axis : int
        Image axis the aliasing folds along -- the undersampled (phase-encode)
        k-space axis.  Default -1 matches `physics.mask.make_acc_mask(dim=1)`.
    acs_size : int
        Number of fully sampled central PE lines included in the reconstruction.
        Ignored when `mask` is given (the mask says what was acquired).
    mask : (B, 1, H, W) | (1, 1, H, W) | (H, W), optional
        The actual sampling pattern.  **Strongly preferred**, and *required* when
        `accel` does not divide the PE length: the kernel taps are then read off
        the mask by one inverse FFT rather than assumed, which

        * removes the delta-for-sinc guess for the ACS band (sidelobes kept),
        * drops the assumption that the ACS is one centred contiguous block,
        * gets the acquired fraction right when `ceil(n/A) != n/A`.

        Without it the analytic taps of the note are used and `accel | n` is
        enforced.  Either way the mask must be separable along the PE axis; random
        or variable-density patterns break the alias-set structure entirely and
        need `physics.gfactor.gfactor_replica`.
    weighted : bool
        True for the noise-weighted (pre-whitened, ML) pinv, which attains the
        Cramer-Rao bound; False for the plain pinv on un-whitened data, which is
        what `block_sense` computes.  Identical when Sigma is None.
    rtol : float
        Relative eigenvalue cutoff for the per-pixel A x A inverse.  Background
        and ill-conditioned blocks return 0 rather than inf.
    thresh : float
        If > 0, zero the output outside `support_mask(smaps, thresh)`.
    chunk : int
        Pixels per batch through the A x A solve.  Only a memory knob.

    Returns
    -------
    (B, 1, H, W) real.  Square it for the noise *variance* -- the `nvar` that
    `physics.gfactor` returns and that `normalize_gmap` consumes.
    """
    if smaps.dim() != 4:
        raise ValueError(f"smaps must be (B, C, H, W); got {tuple(smaps.shape)}")
    accel = int(accel)
    ax = axis % smaps.dim()
    if ax < 2:
        raise ValueError("axis must be a spatial axis (2 or 3, or -1/-2)")
    n_pe = smaps.shape[ax]
    if not 0 <= acs_size <= n_pe:
        raise ValueError(f"acs_size must be in [0, {n_pe}]; got {acs_size}")

    lags = _alias_lags(n_pe, accel)
    T = _kernel_taps(lags, n_pe, accel, acs_size, mask, ax,
                     smaps.dtype, smaps.device)

    # <Sigma^{-1}>_n needs Sigma^{-1/2}-whitened maps and <Sigma>_n needs
    # Sigma^{+1/2}-coloured ones, since S^H X S = (X^{1/2} S)^H (X^{1/2} S).
    if weighted:
        u = whiten_smaps(smaps, Sigma, eps=eps)
        v = None
    else:
        u = smaps
        if Sigma is None:
            v = None
        else:
            Sig = _as_cov(Sigma, smaps.shape[1], smaps.device, smaps.dtype)
            v = _apply_cov(_herm_pow(Sig, 0.5, eps=eps), smaps)

    P = _alias_stack(u, lags, ax)                             # (npix, C, A)
    Pv = _alias_stack(v, lags, ax) if v is not None else None

    out = []
    for i in range(0, P.shape[0], max(int(chunk), 1)):
        Pi = P[i:i + chunk]
        if weighted:
            M = _alias_bracket(Pi, Pi, T)
            d0 = _batched_inv_diag(M, rtol=rtol)[:, 0]            # eq. (26)
        else:
            Bm = _alias_bracket(Pi, Pi, T)                        # <I>
            Cm = Bm if Pv is None else _alias_bracket(Pv[i:i + chunk],
                                                      Pv[i:i + chunk], T)
            Bi = _batched_pinv(Bm, rtol=rtol)
            d0 = _batched_diag(Bi @ Cm @ Bi)[:, 0]                # eq. (27)
        out.append(d0)

    # `_alias_stack` keeps the pixel index in the original (B, H, W) order and
    # `lags[0] == 0`, so entry 0 of each block is the pixel itself: the result drops
    # straight back into place, with no fold/unfold bookkeeping to get wrong.
    sigma = torch.cat(out).reshape(smaps.shape[0], 1, *smaps.shape[2:])
    sigma = sigma.clamp_min(0.0).sqrt()
    if thresh > 0:
        sigma = sigma * support_mask(smaps, thresh).to(sigma.dtype)
    return sigma


def sense_gfactor(
    smaps: Tensor,
    Sigma=None,
    accel: int = 2,
    axis: int = -1,
    acs_size: int = 0,
    mask: Optional[Tensor] = None,
    weighted: bool = True,
    rtol: float = 1e-6,
    thresh: float = 0.0,
    eps: float = 1e-12,
) -> Tensor:
    """Relative g-factor `sigma_sense / (sqrt(A_eff) sigma_full)`, (B, 1, H, W).

    Equals 1 where the aliased sensitivities are Sigma^{-1}-orthogonal and >= 1
    elsewhere (from `[M^{-1}]_aa [M]_aa >= 1` for M positive definite); 0 in the
    background.  With `acs_size = 0` and `weighted=True` this is exactly the
    classical Pruessmann g-factor,
    `sqrt([(S_n^H Sigma^{-1} S_n)^{-1}]_aa [S_n^H Sigma^{-1} S_n]_aa)`.

    Dividing by sqrt(A_eff) rather than sqrt(A) is deliberate -- see
    `effective_acceleration`.  Pass `mask` and A_eff comes from `1 / mean(mask)`,
    which is the only correct normaliser when `accel` does not divide the PE length.
    """
    n_pe = smaps.shape[axis % smaps.dim()]
    Aeff = effective_acceleration(accel, acs_size, n_pe, mask=mask)

    sigma = sense_noise_level(smaps, Sigma, accel, axis=axis, acs_size=acs_size,
                              mask=mask, weighted=weighted, rtol=rtol,
                              thresh=thresh, eps=eps)
    sigma0 = coil_combined_noise_level(smaps, Sigma, weighted=weighted, eps=eps)

    den = Aeff ** 0.5 * sigma0
    floor = torch.finfo(sigma.real.dtype).eps
    return torch.where(den > floor, sigma / den.clamp_min(floor), torch.zeros_like(sigma))


# ---------------------------------------------------------------------------
# The reconstruction the no-ACS plain-pinv branch describes
# ---------------------------------------------------------------------------

def block_sense(
    kspace: Tensor,
    smaps: Tensor,
    accel: int,
    axis: int = -1,
    lamda: float = 0.0,
    rtol: float = 1e-6,
) -> Tensor:
    """Block-SENSE: the plain pinv of `Mask(uniform) @ fftc @ Sense(smaps)`.

    Port of Sljiva's `blocksense`, vectorised over the folded pixels instead of
    looping.  `kspace` is (B, C, H, W) and already undersampled (zeros off the
    sampling grid); the return is (B, 1, H, W).

    Per alias set this solves `(S_n^H S_n + lamda I) x_n = S_n^H y_n` with `y_n`
    the aliased coil vector, then rescales by `accel`.  The rescale is what makes
    the estimator unbiased: with an ortho FFT the diagonal of `F^H M_Xi F` is
    `1/A`, so the raw solve returns `x_n / A`.  Noise-wise the same factor turns
    the per-pixel `Sigma / A` of the folded coil images into

        Cov = A (S_n^H S_n)^{-1} S_n^H Sigma S_n (S_n^H S_n)^{-1},

    i.e. exactly `sense_noise_level(..., acs_size=0, weighted=False)` squared --
    which is the identity `sense_noise_level_mc` checks.

    Only one representative per alias set is read (PE index `jj < W/A`), so the
    A-fold correlation that `F^H M_Xi F` induces between alias partners never
    enters, and the result is independent of the sampling offset.  At `lamda = 0`
    rank-deficient blocks come back as 0 (truncated pinv), matching the
    `isnan` guard in the Julia.
    """
    accel = int(accel)
    yimg = ifftc(kspace)

    # One map set may serve a stack of replicas on the batch axis (`chunk` in
    # `sense_noise_level_mc`), so broadcast the maps up to the k-space batch.
    Bk, C = kspace.shape[0], smaps.shape[1]
    if smaps.shape[0] != Bk:
        if smaps.shape[0] != 1:
            raise ValueError(
                f"smaps batch {smaps.shape[0]} matches neither the kspace batch {Bk} nor 1")
        smaps = smaps.expand(Bk, -1, -1, -1)

    S, lead, n, ax = _fold_alias(smaps, accel, axis)               # (nblk, C, A)
    step = n // accel

    y = yimg.movedim(ax, -1)[..., :step]                           # (B, C, *lead, step)
    y = y.movedim(1, -1).reshape(-1, C, 1)                         # (nblk, C, 1)

    G = S.conj().transpose(-2, -1) @ S                             # (nblk, A, A)
    rhs = S.conj().transpose(-2, -1) @ y                           # (nblk, A, 1)
    if lamda:
        eye = torch.eye(accel, device=G.device, dtype=G.dtype)
        Gi = _batched_pinv(G + float(lamda) * eye, rtol=rtol)
    else:
        Gi = _batched_pinv(G, rtol=rtol)

    xa = (Gi @ rhs).squeeze(-1)                                    # (nblk, A)
    return accel * _unfold_alias(xa, Bk, lead, n, accel, ax)


# ---------------------------------------------------------------------------
# Empirical verification (Rep2Rep Eq. 51): pseudo-replica
# ---------------------------------------------------------------------------

def sense_noise_level_mc(
    smaps: Tensor,
    Sigma=None,
    accel: int = 2,
    axis: int = -1,
    acs_size: int = 0,
    weighted: bool = False,
    nreps: int = 256,
    offset: int = 0,
    recon: Optional[Callable[[Tensor], Tensor]] = None,
    mask: Optional[Tensor] = None,
    generator: Optional[torch.Generator] = None,
    chunk: int = 1,
    cg_kwargs: Optional[dict] = None,
    progress: bool = False,
) -> Tensor:
    """Monte-Carlo noise level: `nreps` noise-only replicas through a SENSE solve.

    Rep2Rep Eq. (51) with the mean known to be zero (the input is noise only, and
    every reconstruction here is linear), so this is `sqrt(mean |x_hat|^2)` over
    replicas rather than the `1/(R-1)` sample variance -- one fewer degree of
    freedom spent, same estimand.

    Which `recon` is used by default, and what it validates:

    * `acs_size = 0, weighted = False` -> `block_sense`.  Exact: the closed form
      is the covariance of precisely this estimator, so agreement is limited only
      by replica error (~1/sqrt(2 nreps) relative; 256 replicas gives ~4%).
    * `weighted = True` (any `acs_size`) -> `physics.gfactor.cg_sense` on the
      whitened problem over the full mask Omega, i.e. the ML pinv.  At
      `acs_size = 0` this is again exact; at `acs_size > 0` the CG solve uses the
      true M_Omega^conv while the closed form uses the delta-for-sinc bracket, so
      a systematic gap of a few percent is expected and *is* the approximation
      being measured.
    * `acs_size > 0, weighted = False` has no default -- pass your own `recon`.

    Slow.  Use on a small phantom (96 x 96 was enough to catch folding-index bugs
    in the reference implementation) before trusting full-size maps.

    Parameters
    ----------
    offset : int
        Which residue class of PE lines the uniform grid keeps.  The analytic map
        does not depend on it (the offset only phases the aliasing deltas), so a
        sweep over `offset` is a free extra check.
    mask : optional (1, 1, H, W) or (B, 1, H, W)
        Sampling pattern to use instead of the uniform+ACS one built from
        `accel`/`acs_size`/`offset`.
    chunk : int
        Reconstruct this many replicas at once on the batch axis.  Requires
        `B = 1` and a `recon` that broadcasts over the batch (both defaults do).
    """
    from physics.gfactor import cg_sense
    from physics.mask import make_acc_mask

    B, C, H, W = smaps.shape
    accel = int(accel)
    chunk = max(int(chunk), 1)
    if chunk > 1 and B != 1:
        raise ValueError("chunk > 1 requires a single image (smaps batch of 1)")

    ax = axis % smaps.dim()
    if mask is None:
        mask = make_acc_mask((H, W), accel, acs_lines=int(acs_size),
                             dim=1 if ax == 3 else 0, mode="uniform",
                             offset=int(offset), device=smaps.device)

    if recon is None:
        if weighted:
            recon = cg_sense(smaps, mask, lamda=0.0, Sigma=Sigma, **(cg_kwargs or {}))
        elif acs_size == 0:
            def recon(k: Tensor) -> Tensor:
                return block_sense(k, smaps, accel, axis=axis)
        else:
            raise ValueError(
                "no default reconstruction for the plain pinv with ACS "
                "(block_sense is uniform-only); pass `recon=` or set weighted=True")

    pbar = None
    if progress:
        try:
            from tqdm import tqdm
            pbar = tqdm(total=int(nreps), desc="replicas")
        except ImportError:
            pass

    acc = torch.zeros(B, 1, H, W, device=smaps.device, dtype=smaps.real.dtype)
    done, total = 0, int(nreps)
    while done < total:
        c = min(chunk, total - done)
        xi = sample_kspace_noise((B * c, C, H, W), mask=mask, Sigma=Sigma,
                                 generator=generator, device=smaps.device,
                                 dtype=smaps.dtype)
        xr = recon(xi).abs().pow(2)
        acc = acc + (xr.sum(dim=0, keepdim=True) if c > 1 else xr)
        done += c
        if pbar is not None:
            pbar.update(c)
    if pbar is not None:
        pbar.close()

    return (acc / float(total)).sqrt()
