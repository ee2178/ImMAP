"""
g-factor (noise-amplification) maps for parallel-imaging reconstructions.

Rep2Rep (Janjusevic et al., arXiv:2504.17698, Supporting Information Sec. 11)
makes the point this module implements: the noise in a coil-combined MRI
reconstruction is Gaussian, but once the acquisition is accelerated its
covariance is neither white nor flat.  Writing a linear reconstruction as a map
R applied to the measurement,

    y = A x + xi,   xi ~ CN(0, Sigma),   x_hat = R y,

the reconstruction noise covariance is  R Sigma_y R^H.  Off-diagonal entries are
spatial noise correlations (Rep2Rep Eq. 60/66); the *diagonal* is the per-pixel
noise power, which is what "noise-level map" means throughout that paper and
what ImMAP needs.  We report the diagonal two ways:

    nvar[n] = [R Sigma_y R^H]_nn                      physical variance units
    g[n]    = sigma_accel[n] / (sqrt(R_net) sigma_full[n])   Pruessmann g-factor

`nvar` is the quantity ImMAP wants.  It is the `diag(g)` in

    A^dagger y  ~  CN(x, sigma^2 diag(g)),

i.e. wherever the algorithm currently treats the reconstruction noise as white
with a single level sigma, it should instead work with residuals divided by
sqrt(g).  `normalize_gmap` puts `nvar` in exactly that form (mean 1 over the
support, so sigma^2 carries the overall scale).

`g` is the dimensionless, convention-standard quantity (Pruessmann 1999): it is
1 wherever the acceleration costs nothing beyond the unavoidable sqrt(R_net)
loss from acquiring fewer samples, and > 1 where the unfolding is
ill-conditioned.  It is an *amplitude* ratio, so `nvar = R_net * sigma_full^2 *
g^2`; with whitened, unit-RSS maps sigma_full is flat and `g^2` and the
normalized `nvar` agree up to a constant.

Two ways to get the diagonal are provided.

`gfactor_uniform` -- exact, closed form, for uniform Cartesian undersampling
    (no ACS).  There A^H A is block-diagonal: it couples only the R pixels that
    alias onto each other, so the whole map is a batch of R x R Hermitian
    inversions.  With the repo's operator  E = Mask(m) @ FFT2D() @ Sense(s)
    (ortho FFT), the aliasing block of E^H E is (1/R) E_a^H E_a where E_a is the
    C x R matrix of sensitivities of the aliased partners, giving

        Cov(x_hat) |_block = R * B^{-1} M B^{-1},   M = E_a^H Sigma^{-1} E_a,
                                                    B = M + R lamda I

    and, at lamda = 0,  g_j = sqrt([M^{-1}]_jj [M]_jj).

`gfactor_replica` -- Monte-Carlo (the "pseudo multiple replica" method):
    push synthetic k-space noise through *any* reconstruction and take the
    pixel-wise sample variance.  Slower, but it handles the masks this repo
    actually generates (ACS lines included -> not uniform), random masks, and
    regularized/iterative reconstructions.

Tensor layout is the repo standard: (B, C, H, W), masks (B, 1, H, W) or
(1, 1, H, W), maps out as (B, 1, H, W) real.

Note on which reconstruction has a g-factor at all: the *adjoint* A^H does not.
diag(S^H F^H M F S) = mean(m) * sum_c |s_c|^2 is flat for unit-RSS maps,
because F^H M F is a convolution and the diagonal of a convolution is its
zero-lag tap.  The spatially varying noise appears only once you invert the
unfolding -- so `A^dagger` above must be a pseudo-inverse / CG-SENSE solve, not
`E.H`.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import torch
from torch import Tensor

from operators.fourier import fftc, ifftc


__all__ = [
    "whiten_smaps",
    "sigma_full_map",
    "support_mask",
    "mask_is_uniform",
    "gfactor_uniform",
    "gfactor_replica",
    "gfactor",
    "cg_sense",
    "sample_kspace_noise",
    "empirical_noise_var",
    "local_noise_std",
    "normalize_gmap",
]


# ---------------------------------------------------------------------------
# Coil-noise covariance plumbing
# ---------------------------------------------------------------------------

def _as_cov(Sigma, C: int, device, dtype) -> Optional[Tensor]:
    """Normalise `Sigma` to a (B, C, C) complex tensor, or None for identity.

    Accepts None (identity), a scalar variance, a (C, C) matrix, or a batch of
    them.  A scalar `s` means Sigma = s * I, i.e. per-coil noise *variance* s,
    matching `operators.noise`'s convention that a complex `sigma * randn` has
    E|xi|^2 = sigma^2.
    """
    if Sigma is None:
        return None
    if not torch.is_tensor(Sigma):
        Sigma = torch.as_tensor(Sigma, device=device)
    if Sigma.dim() == 0:
        eye = torch.eye(C, device=device, dtype=dtype)
        return (Sigma.to(dtype) * eye).unsqueeze(0)
    Sigma = Sigma.to(device=device, dtype=dtype)
    if Sigma.dim() == 2:
        Sigma = Sigma.unsqueeze(0)
    if Sigma.dim() != 3 or Sigma.shape[-1] != C or Sigma.shape[-2] != C:
        raise ValueError(f"Sigma must be scalar, (C, C) or (B, C, C); got {tuple(Sigma.shape)}")
    return Sigma


def _herm_pow(Sigma: Tensor, p: float, eps: float = 1e-12) -> Tensor:
    """Sigma^p for a Hermitian PSD (B, C, C), eigenvalues clamped from below."""
    w, U = torch.linalg.eigh(Sigma)
    w = w.real
    w = w.clamp_min(eps * w.amax(dim=-1, keepdim=True).clamp_min(eps))
    wp = w.pow(p).to(U.dtype)
    return U @ (wp.unsqueeze(-1) * U.conj().transpose(-2, -1))


def whiten_smaps(smaps: Tensor, Sigma=None, eps: float = 1e-12) -> Tensor:
    """Sigma^{-1/2} applied to the coil axis of `smaps`.

    Whitening the measurement (y_w = Sigma^{-1/2} y) turns the coil-noise
    covariance into the identity and folds Sigma into the maps, so every
    formula below can be written with plain Gram matrices:
    E^H Sigma^{-1} E = (Sigma^{-1/2} E)^H (Sigma^{-1/2} E).
    """
    if Sigma is None:
        return smaps
    C = smaps.shape[1]
    Sig = _as_cov(Sigma, C, smaps.device, smaps.dtype)
    inv_sqrt = _herm_pow(Sig, -0.5, eps=eps)                     # (B or 1, C, C)
    return torch.einsum("bde,bexy->bdxy", inv_sqrt.expand(smaps.shape[0], C, C), smaps)


# ---------------------------------------------------------------------------
# Reference (fully-sampled) noise level
# ---------------------------------------------------------------------------

def sigma_full_map(smaps: Tensor, Sigma=None, eps: float = 1e-12,
                   combine: str = "optimal") -> Tensor:
    """Noise std of the *fully sampled* coil-combined image, (B, 1, H, W).

    `combine="optimal"`  sigma^2 = 1 / (s^H Sigma^{-1} s)
        The noise of the maximum-likelihood / matched-filter combination.  This
        is the reference the Pruessmann g-factor is defined against.

    `combine="adjoint"`  sigma^2 = s^H Sigma s
        The noise of the naive S^H combination, i.e. Rep2Rep Eq. (4).

    The two agree when Sigma is proportional to the identity *and* the maps are
    unit-RSS, which is the case this repo's `whiten`/preprocessing produces.
    """
    if combine == "optimal":
        sw = whiten_smaps(smaps, Sigma, eps=eps)
        energy = sw.abs().pow(2).sum(dim=1, keepdim=True)         # s^H Sigma^-1 s
        return energy.clamp_min(eps).rsqrt()
    if combine == "adjoint":
        if Sigma is None:
            return smaps.abs().pow(2).sum(dim=1, keepdim=True).sqrt()
        C = smaps.shape[1]
        Sig = _as_cov(Sigma, C, smaps.device, smaps.dtype).expand(smaps.shape[0], C, C)
        Ss = torch.einsum("bde,bexy->bdxy", Sig, smaps)
        return (smaps.conj() * Ss).sum(dim=1, keepdim=True).real.clamp_min(0).sqrt()
    raise ValueError("combine must be 'optimal' or 'adjoint'")


def support_mask(smaps: Tensor, thresh: float = 1e-2) -> Tensor:
    """Where the coil maps carry energy, (B, 1, H, W) bool.

    ESPIRiT zeroes the maps outside the object, and the g-factor is undefined
    there (the unfolding matrix is the zero matrix).  `thresh` is relative to
    the per-image max coil energy.
    """
    energy = smaps.abs().pow(2).sum(dim=1, keepdim=True)
    peak = energy.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-30)
    return energy > thresh * peak


# ---------------------------------------------------------------------------
# Sampling-pattern inspection
# ---------------------------------------------------------------------------

def mask_is_uniform(mask: Tensor) -> Optional[Tuple[int, int]]:
    """Detect uniform 1-D Cartesian undersampling.

    Returns `(axis, accel)` with `axis` indexed into a (B, 1, H, W) tensor
    (2 or 3), or None if the pattern is not a plain arithmetic progression
    whose period divides the axis length -- which is the case for every mask
    with ACS lines folded in, and for random masks.
    """
    m = mask
    while m.dim() > 2:
        m = m[0]
    if m.dim() != 2:
        raise ValueError("mask must be at least 2-D")
    H, W = m.shape
    b = (m != 0)

    for axis, n in ((3, W), (2, H)):
        # collapse the fully sampled direction; require the pattern be separable
        line = b.all(dim=0) if axis == 3 else b.all(dim=1)
        any_ = b.any(dim=0) if axis == 3 else b.any(dim=1)
        if not torch.equal(line, any_):
            continue                                   # not constant along the other axis
        idx = torch.nonzero(line, as_tuple=False).flatten()
        k = idx.numel()
        if k == 0 or k == n:
            continue
        if n % k:
            continue
        accel = n // k
        expected = torch.arange(idx[0].item(), n, accel, device=idx.device)
        if expected.numel() == k and torch.equal(idx, expected):
            return axis, int(accel)
    return None


def net_acceleration(mask: Tensor) -> Tensor:
    """N_total / N_sampled per batch element, (B, 1, 1, 1)."""
    m = (mask != 0).to(torch.float64)
    if m.dim() == 2:
        m = m[None, None]
    elif m.dim() == 3:
        m = m[:, None]
    n = m.shape[-1] * m.shape[-2]
    return (n / m.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)).to(torch.float32)


# ---------------------------------------------------------------------------
# Exact g-factor: uniform Cartesian undersampling
# ---------------------------------------------------------------------------

def gfactor_uniform(
    smaps: Tensor,
    accel: int,
    axis: int = -1,
    Sigma=None,
    lamda: float = 0.0,
    eps: float = 1e-8,
    thresh: float = 0.0,
) -> Tuple[Tensor, Tensor]:
    """Closed-form g-factor for R-fold uniform Cartesian SENSE.

    Parameters
    ----------
    smaps : (B, C, H, W) complex
        Coil sensitivities.  Not required to be unit-RSS; `g` is invariant to a
        per-pixel rescaling of the maps, `nvar` is not.
    accel : int
        Undersampling factor R.  Must divide the length of `axis`.
    axis : int
        Image axis along which aliasing folds -- the same axis index as the
        undersampled k-space axis.  `physics.mask.make_acc_mask(..., dim=1)`
        drops columns, so the default -1 is the one to use with it.
    Sigma : None | float | (C, C) | (B, C, C)
        Coil noise covariance.  None means the identity (i.e. `nvar` comes back
        relative to the per-coil noise variance).
    lamda : float
        Tikhonov weight of the reconstruction being characterised,
        x_hat = (A^H A + lamda I)^{-1} A^H y.  `g` is always reported for the
        unregularised (lamda = 0) inverse, per the standard definition;
        `lamda` only affects `nvar`.
    eps : float
        Eigenvalue floor, relative to the largest eigenvalue over the image.
        Keeps background pixels (all-zero maps) finite rather than infinite.
    thresh : float
        If > 0, zero both outputs outside `support_mask(smaps, thresh)`.

    Returns
    -------
    g    : (B, 1, H, W) real   Pruessmann g-factor
    nvar : (B, 1, H, W) real   reconstruction noise variance, [R Sigma_y R^H]_nn

    Notes
    -----
    Off-diagonal noise correlation is real and is *not* removed by dividing by
    sqrt(nvar): the R aliased partners of a pixel stay correlated with each
    other (they come from one shared R x R inverse).  What the division does
    remove is the spatially varying noise *power*, which is what ImMAP's
    sigma_t estimate and PiGDM covariance care about.
    """
    if smaps.dim() != 4:
        raise ValueError(f"smaps must be (B, C, H, W); got {tuple(smaps.shape)}")
    accel = int(accel)
    B, C = smaps.shape[0], smaps.shape[1]

    sw = whiten_smaps(smaps, Sigma)
    ax = axis % sw.dim()
    if ax < 2:
        raise ValueError("axis must be a spatial axis (2 or 3, or -1/-2)")

    s = sw.movedim(ax, -1)                                # (B, C, *lead, n)
    lead = s.shape[2:-1]
    n = s.shape[-1]
    if n % accel:
        raise ValueError(f"accel={accel} does not divide axis length {n}")
    step = n // accel

    # Partners of index r are r + j*step, j = 0..R-1.  Row-major reshape of the
    # last axis into (accel, step) puts index j*step + r at [j, r], so the
    # aliasing group is exactly the `accel` axis.
    E = s.reshape(B, C, -1, accel, step)                  # (B, C, P, R, step)
    E = E.permute(0, 2, 4, 1, 3)                          # (B, P, step, C, R)

    M = E.conj().transpose(-2, -1) @ E                    # (B, P, step, R, R)
    w, U = torch.linalg.eigh(M)                           # w real, ascending
    w = w.real

    scale = w.amax().clamp_min(1e-30)
    wc = w.clamp_min(eps * scale)
    # M = U diag(w) U^H, so [f(M)]_jj = sum_k |U_jk|^2 f(w_k).  The eigenvalue
    # axis is the *last* one of U, hence the unsqueeze(-2) on every spectrum.
    U2 = U.abs().pow(2)                                   # |U_jk|^2

    diag_Minv = (U2 / wc.unsqueeze(-2)).sum(dim=-1)       # [M^-1]_jj
    diag_M = (U2 * w.unsqueeze(-2)).sum(dim=-1)           # [M]_jj
    g = (diag_Minv * diag_M).clamp_min(0.0).sqrt()        # (B, P, step, R)

    # Cov = R * B^{-1} M B^{-1},  B = M + R lamda I
    den = (wc + accel * float(lamda)).clamp_min(eps * scale)
    nvar = accel * (U2 * (wc / den.pow(2)).unsqueeze(-2)).sum(dim=-1)

    def _unfold(t: Tensor) -> Tensor:
        t = t.permute(0, 1, 3, 2).reshape(B, 1, *lead, n)  # (B,P,step,R)->(B,1,*lead,n)
        return t.movedim(-1, ax)

    g, nvar = _unfold(g), _unfold(nvar)

    if thresh > 0:
        keep = support_mask(smaps, thresh).to(g.dtype)
        g, nvar = g * keep, nvar * keep
    return g, nvar


# ---------------------------------------------------------------------------
# Monte-Carlo g-factor: any mask, any (linear) reconstruction
# ---------------------------------------------------------------------------

def sample_kspace_noise(
    shape,
    mask: Optional[Tensor] = None,
    Sigma=None,
    generator: Optional[torch.Generator] = None,
    device=None,
    dtype=torch.complex64,
) -> Tensor:
    """Draw xi ~ CN(0, Sigma) per k-space sample, restricted to `mask`.

    Follows `operators.noise`: a complex `randn` has E|z|^2 = 1 split evenly
    between real and imaginary parts, so Sigma = s * I gives E|xi_c|^2 = s.
    """
    B, C = shape[0], shape[1]
    if generator is None:
        z = torch.randn(*shape, device=device, dtype=dtype)
    else:
        z = torch.randn(*shape, device=device, dtype=dtype, generator=generator)
    if Sigma is not None:
        Sig = _as_cov(Sigma, C, z.device, z.dtype)
        L = _herm_pow(Sig, 0.5).expand(B, C, C)
        z = torch.einsum("bde,bexy->bdxy", L, z)
    if mask is not None:
        z = mask * z
    return z


def gfactor_replica(
    recon: Callable[[Tensor], Tensor],
    smaps: Tensor,
    mask: Tensor,
    Sigma=None,
    n_replicas: int = 64,
    y: Optional[Tensor] = None,
    generator: Optional[torch.Generator] = None,
    eps: float = 1e-12,
    thresh: float = 0.0,
    chunk: int = 1,
    progress: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Pseudo-replica g-factor: sample variance of `recon` under k-space noise.

    Parameters
    ----------
    recon : callable
        k-space (B, C, H, W) -> image (B, 1, H, W).  `cg_sense` builds the
        usual one.  Anything linear is exact in the limit of many replicas.
    y : optional
        Baseline measurement.  If given, each replica reconstructs `y + xi` and
        subtracts `recon(y)`, which is what you want for a *nonlinear*
        reconstruction (regularized, or a network); the local linearisation
        around `y` is then what gets characterised.  Leave as None for a linear
        `recon` -- reconstructing the noise alone is exact and half the cost.
    n_replicas : int
        Sample-variance error is ~1/sqrt(2 n_replicas) relative; 64 replicas
        gives ~9% per-pixel, enough to see structure, too coarse for a
        pixel-wise correctness check.  Use a few hundred for that.
    chunk : int
        Reconstruct this many replicas at once by stacking them on the batch
        axis.  Only available for a single image (B = 1), and only correct if
        `recon` broadcasts over the batch -- `cg_sense` does, since it reduces
        its inner products per batch element.  Worth a large factor on a GPU.

    Returns
    -------
    (g, nvar), same convention as `gfactor_uniform`.
    """
    B, C, H, W = smaps.shape
    dev = smaps.device
    chunk = max(int(chunk), 1)
    if chunk > 1 and B != 1:
        raise ValueError("chunk > 1 requires a single image (smaps batch of 1)")

    base = recon(y) if y is not None else None
    acc_sum = None
    acc_sq = None

    n_done, total = 0, int(n_replicas)
    pbar = None
    if progress:
        try:
            from tqdm import tqdm
            pbar = tqdm(total=total, desc="replicas")
        except ImportError:
            pass

    while n_done < total:
        c = min(chunk, total - n_done)
        xi = sample_kspace_noise((B * c, C, H, W), mask=mask, Sigma=Sigma,
                                 generator=generator, device=dev, dtype=smaps.dtype)
        xr = recon(y + xi) - base if y is not None else recon(xi)
        s = xr.sum(dim=0, keepdim=True) if c > 1 else xr
        s2 = xr.abs().pow(2).sum(dim=0, keepdim=True) if c > 1 else xr.abs().pow(2)
        if acc_sum is None:
            acc_sum = torch.zeros_like(s)
            acc_sq = torch.zeros(s2.shape, device=dev, dtype=s2.dtype)
        acc_sum = acc_sum + s
        acc_sq = acc_sq + s2
        n_done += c
        if pbar is not None:
            pbar.update(c)
    if pbar is not None:
        pbar.close()

    n = float(n_replicas)
    nvar = (acc_sq - acc_sum.abs().pow(2) / n) / max(n - 1.0, 1.0)
    nvar = nvar.clamp_min(0.0)

    R_net = net_acceleration(mask).to(dev)
    var_full = sigma_full_map(smaps, Sigma, eps=eps).pow(2)
    g = (nvar / (R_net * var_full).clamp_min(eps)).clamp_min(0.0).sqrt()

    if thresh > 0:
        keep = support_mask(smaps, thresh).to(g.dtype)
        g, nvar = g * keep, nvar * keep
    return g, nvar


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def gfactor(
    smaps: Tensor,
    mask: Tensor,
    Sigma=None,
    lamda: float = 0.0,
    method: str = "auto",
    recon: Optional[Callable[[Tensor], Tensor]] = None,
    n_replicas: int = 64,
    generator: Optional[torch.Generator] = None,
    thresh: float = 0.0,
    chunk: int = 1,
    cg_kwargs: Optional[dict] = None,
    progress: bool = False,
) -> Tuple[Tensor, Tensor]:
    """g-factor and noise-variance maps for `Mask(mask) @ FFT2D() @ Sense(smaps)`.

    method
        "uniform"  force the closed form (errors if the mask is not uniform)
        "replica"  force Monte Carlo
        "auto"     closed form when the mask is uniform Cartesian, else replica

    `recon` (replica path only) defaults to CG-SENSE with the given `lamda`;
    pass your own callable to characterise a different reconstruction.

    Returns (g, nvar) as (B, 1, H, W) real tensors.
    """
    uni = mask_is_uniform(mask)
    if method == "auto":
        method = "uniform" if uni is not None else "replica"

    if method == "uniform":
        if uni is None:
            raise ValueError(
                "mask is not uniform Cartesian (ACS lines or random sampling); "
                "use method='replica'")
        axis, accel = uni
        return gfactor_uniform(smaps, accel, axis=axis, Sigma=Sigma,
                               lamda=lamda, thresh=thresh)

    if method == "replica":
        if recon is None:
            recon = cg_sense(smaps, mask, lamda=lamda, Sigma=Sigma,
                             **(cg_kwargs or {}))
        return gfactor_replica(recon, smaps, mask, Sigma=Sigma,
                               n_replicas=n_replicas, generator=generator,
                               thresh=thresh, chunk=chunk, progress=progress)

    raise ValueError("method must be 'auto', 'uniform' or 'replica'")


# ---------------------------------------------------------------------------
# The reconstruction the maps describe
# ---------------------------------------------------------------------------

def cg_sense(
    smaps: Tensor,
    mask: Tensor,
    lamda: float = 0.0,
    Sigma=None,
    max_iter: int = 128,
    tol: float = 1e-8,
) -> Callable[[Tensor], Tensor]:
    """Build  y -> (A^H A + lamda I)^{-1} A^H y  with A = Mask(m) F Sense(s).

    When `Sigma` is given, both the data and the maps are whitened first, so
    the solve is the maximum-likelihood (noise-weighted) SENSE estimate -- the
    one `gfactor_uniform`/`gfactor_replica` characterise.

    At lamda = 0 the normal operator is singular wherever the maps vanish, and
    CG will wander in that null space; either pass a small lamda or restrict
    attention to `support_mask`.
    """
    from solvers.cg import batched_cg

    sw = whiten_smaps(smaps, Sigma)
    C = smaps.shape[1]
    inv_sqrt = None
    if Sigma is not None:
        Sig = _as_cov(Sigma, C, smaps.device, smaps.dtype)
        inv_sqrt = _herm_pow(Sig, -0.5)

    def A(x: Tensor) -> Tensor:
        return mask * fftc(sw * x)

    def AH(k: Tensor) -> Tensor:
        return (sw.conj() * ifftc(mask * k)).sum(dim=1, keepdim=True)

    def normal(x: Tensor) -> Tensor:
        return AH(A(x)) + lamda * x

    def recon(y: Tensor) -> Tensor:
        if inv_sqrt is not None:
            y = torch.einsum("bde,bexy->bdxy", inv_sqrt.expand(y.shape[0], C, C), y)
        return batched_cg(normal, AH(y), tol=tol, max_iter=max_iter)

    return recon


# ---------------------------------------------------------------------------
# Empirical checks / consumption helpers
# ---------------------------------------------------------------------------

def empirical_noise_var(residuals: Tensor, dim: int = 0, unbiased: bool = True) -> Tensor:
    """Pixel-wise noise variance from a stack of residual realizations.

    `residuals` : (Nrep, ...) complex, e.g. x_hat^(r) - x_true stacked on dim 0.
    This is Rep2Rep Eq. (51) squared; it is the ground truth `nvar` is predicting.
    """
    n = residuals.shape[dim]
    mu = residuals.mean(dim=dim, keepdim=True)
    v = (residuals - mu).abs().pow(2).sum(dim=dim)
    return v / (n - 1 if unbiased and n > 1 else n)


def local_noise_std(x: Tensor, ks: int = 9) -> Tensor:
    """Boxcar local std of a (complex) image, (B, 1, H, W) real.

    Purely diagnostic: a flat local-std map is the visual signature of
    successfully whitened residuals.  For complex x this is the std of the
    complex value, i.e. sqrt(E|x - mean|^2).
    """
    import torch.nn.functional as F

    if x.dim() == 3:
        x = x.unsqueeze(1)
    pad = ks // 2
    k = torch.ones(1, 1, ks, ks, device=x.device, dtype=torch.float32) / (ks * ks)

    def box(t):
        return F.conv2d(F.pad(t, (pad,) * 4, mode="reflect"), k)

    if x.is_complex():
        m_re, m_im = box(x.real), box(x.imag)
        m2 = box(x.abs().pow(2))
        var = m2 - m_re.pow(2) - m_im.pow(2)
    else:
        m = box(x)
        var = box(x.pow(2)) - m.pow(2)
    return var.clamp_min(0).sqrt()


def normalize_gmap(nvar: Tensor, mask: Optional[Tensor] = None,
                   eps: float = 1e-12) -> Tensor:
    """`nvar` rescaled to mean 1 -- the `diag(g)` ImMAP's update equations want.

    ImMAP estimates a single sigma_t per iteration and multiplies it by diag(g);
    with mean(g) = 1 over the support that estimate keeps its usual meaning
    ("the average noise level") and g carries only the shape:

        sigma_t^2 <- || (z_hat_t - z_t) / sqrt(g) ||^2 / N

    `mask` restricts the mean to the object (recommended -- background pixels
    have no meaningful g and would otherwise dominate the average).
    """
    if mask is None:
        mu = nvar.mean(dim=(-2, -1), keepdim=True)
        return nvar / mu.clamp_min(eps)
    m = mask.to(nvar.dtype)
    mu = (nvar * m).sum(dim=(-2, -1), keepdim=True) / m.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
    out = nvar / mu.clamp_min(eps)
    return torch.where(mask.expand_as(out), out, torch.ones_like(out))
