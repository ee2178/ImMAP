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

Cost: one batched A x A Hermitian eigendecomposition per folded pixel, i.e.
H * (W / A) * B of them.  For A = 2 that is a small multiple of a single
coil-combination.  (The Julia uses a truncated SVD; for the Hermitian PSD
brackets built here `eigh` is the same decomposition, cheaper, and is what
`physics.gfactor` already uses.)
"""

from __future__ import annotations

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
# Alias-set folding
# ---------------------------------------------------------------------------

def _fold_alias(x: Tensor, accel: int, axis: int) -> Tuple[Tensor, Tuple[int, ...], int, int]:
    """Gather the aliasing sensitivity matrices S_n, no gather needed.

    (B, C, H, W) -> (B * L * step, C, accel), where `step = n // accel` and `n`
    is the length of the folded axis.  A row-major reshape of that axis into
    (accel, step) puts PE index `a * step + jj` at `[a, jj]`, so the alias set
    {jj, jj + step, ..., jj + (accel-1) step} *is* the `accel` axis -- the same
    free-folding trick the Julia gets from a column-major
    `reshape(smaps, Nx, space, accel, C, B)`.

    Returns `(S, lead, n, ax)`; `lead` is the untouched spatial axis, needed to
    undo the fold.
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
        raise ValueError(f"accel={accel} does not divide axis length {n}")
    step = n // accel

    S = s.reshape(B, C, -1, accel, step)           # (B, C, L, accel, step)
    S = S.permute(0, 2, 4, 1, 3)                   # (B, L, step, C, accel)
    return S.reshape(-1, C, accel), lead, n, ax


def _unfold_alias(d: Tensor, B: int, lead: Sequence[int], n: int,
                  accel: int, ax: int) -> Tensor:
    """(B * L * step, accel) -> (B, 1, H, W), undoing `_fold_alias`."""
    step = n // accel
    t = d.reshape(B, -1, step, accel).permute(0, 1, 3, 2)   # (B, L, accel, step)
    return t.reshape(B, 1, *lead, n).movedim(-1, ax)


def _alias_gram(S: Tensor, XS: Tensor, accel: int, p: float) -> Tensor:
    """The sampled bracket `<X>_n`, (nblk, accel, accel).

    `S` holds S_n and `XS` holds X S_n (X applied pixel-wise across coils), both
    from `_fold_alias`.  The Gram is one batched product and the ACS term is a
    Hadamard product with the A x A identity:

        <X>_n = (1-p)/A * S_n^H X S_n  +  p * dg(S_n^H X S_n).

    `dg(S_n^H X S_n)` has entries s[n^(a)]^H X s[n^(a)]: the ACS contributes the
    *same diagonal* as the uniform term with the off-diagonals stripped, which is
    the mechanism by which it breaks the aliasing degeneracy.
    """
    G = S.conj().transpose(-2, -1) @ XS            # G[a, a'] = s[n^(a)]^H X s[n^(a')]
    eye = torch.eye(accel, device=G.device, dtype=G.dtype)
    return ((1.0 - p) / accel) * G + p * (G * eye)


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

def effective_acceleration(accel: int, acs_size: int, n_pe: int) -> float:
    """`A_eff = A / (1 + p (A - 1))`, `p = acs_size / n_pe`.

    The reciprocal of the acquired fraction of k-space once the ACS band is
    counted as acquired data: the central amplitude of M_Omega^conv is
    1/A + p - p/A = 1/A_eff.  This is the SNR penalty attributable to *scan
    time*; what is left in the g-factor is then purely the conditioning penalty,
    which is why `sense_gfactor` divides by sqrt(A_eff) rather than sqrt(A).
    """
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
    weighted: bool = True,
    rtol: float = 1e-6,
    thresh: float = 0.0,
    eps: float = 1e-12,
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
        Uniform acceleration A along `axis`; must divide that axis's length.
    axis : int
        Image axis the aliasing folds along -- the undersampled (phase-encode)
        k-space axis.  Default -1 matches `physics.mask.make_acc_mask(dim=1)`.
    acs_size : int
        Number of fully sampled central PE lines *included in the
        reconstruction*.  0 is the pure uniform case and is exact; > 0 uses the
        delta-for-sinc bracket (see the module docstring).
    weighted : bool
        True for the noise-weighted (pre-whitened, ML) pinv, which attains the
        Cramer-Rao bound; False for the plain pinv on un-whitened data, which is
        what `block_sense` computes.  Identical when Sigma is None.
    rtol : float
        Relative eigenvalue cutoff for the per-pixel A x A inverse.  Background
        and ill-conditioned blocks return 0 rather than inf.
    thresh : float
        If > 0, zero the output outside `support_mask(smaps, thresh)`.

    Returns
    -------
    (B, 1, H, W) real.  Square it for the noise *variance* -- the `nvar` that
    `physics.gfactor` returns and that `normalize_gmap` consumes.
    """
    accel = int(accel)
    B = smaps.shape[0]
    n_pe = smaps.shape[axis % smaps.dim()]
    p = float(acs_size) / float(n_pe)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"acs_size must be in [0, {n_pe}]; got {acs_size}")

    if weighted:
        # <Sigma^{-1}>_n = <I>_n on Sigma^{-1/2}-whitened maps, since
        # S_n^H Sigma^{-1} S_n = (Sigma^{-1/2} S_n)^H (Sigma^{-1/2} S_n).
        Sw, lead, n, ax = _fold_alias(whiten_smaps(smaps, Sigma, eps=eps), accel, axis)
        M = _alias_gram(Sw, Sw, accel, p)
        d = _batched_inv_diag(M, rtol=rtol)                       # eq. (26)
    else:
        S, lead, n, ax = _fold_alias(smaps, accel, axis)
        Bm = _alias_gram(S, S, accel, p)                          # <I>
        if Sigma is None:
            Cm = Bm
        else:
            Sig = _as_cov(Sigma, smaps.shape[1], smaps.device, smaps.dtype)
            # S_n^H Sigma S_n = (Sigma^{1/2} S_n)^H (Sigma^{1/2} S_n)
            Sc, _, _, _ = _fold_alias(_apply_cov(_herm_pow(Sig, 0.5, eps=eps), smaps),
                                      accel, axis)
            Cm = _alias_gram(Sc, Sc, accel, p)                    # <Sigma>
        Bi = _batched_pinv(Bm, rtol=rtol)
        d = _batched_diag(Bi @ Cm @ Bi)                            # eq. (27)

    sigma = _unfold_alias(d.clamp_min(0.0), B, lead, n, accel, ax).sqrt()
    if thresh > 0:
        sigma = sigma * support_mask(smaps, thresh).to(sigma.dtype)
    return sigma


def sense_gfactor(
    smaps: Tensor,
    Sigma=None,
    accel: int = 2,
    axis: int = -1,
    acs_size: int = 0,
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
    `effective_acceleration`.
    """
    n_pe = smaps.shape[axis % smaps.dim()]
    Aeff = effective_acceleration(accel, acs_size, n_pe)

    sigma = sense_noise_level(smaps, Sigma, accel, axis=axis, acs_size=acs_size,
                              weighted=weighted, rtol=rtol, thresh=thresh, eps=eps)
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
