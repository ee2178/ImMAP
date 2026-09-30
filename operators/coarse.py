"""
A REDISCRETIZED coarse SENSE operator: the MRI forward model posed directly on
a grid `factor` times coarser, as opposed to the Galerkin `E_c = E . P` of
`operators/resample.py::galerkin`.

    Galerkin          E_c = M F S P          every coarse Gram runs the fine,
                                             full-resolution multicoil FFTs
    rediscretized     E_c = M_c F_c S_c      everything at the coarse size:
                                             S_c = restrict(S), F_c the coarse
                                             centred FFT, M_c the centre crop
                                             of the mask

Halving the image resolution keeps the CENTRAL half of k-space, so the coarse
mask and data are centre crops of the fine ones. Scale follows from the
transfer normalisation: under the default "dc" split a restricted image keeps
the fine image's POINTWISE amplitude, and with an orthonormal FFT on each grid
the fine k-space centre is `factor` (= sqrt(factor^2) pixels per coarse pixel)
times the coarse k-space of the restricted image. Hence `coarse_data(y) =
crop(y) / factor`.

What the rediscretized operator CANNOT see: sampled lines outside the central
band. The Galerkin coarse Gram folds them in through R and P; this one drops
them. At R=16 on a 320-line axis the central 160 columns hold the ACS and only
a few outer lines, so the two coarse problems differ -- by how much is what
tests/test_coarse_operator.py measures. Used by
`MGLPDSNet(coarse_op="rediscretize")` through `coarsen` below.
"""

from __future__ import annotations

import torch

from operators.fourier import FFT2D
from operators.mask import Mask
from operators.resample import DEFAULT_FACTOR, restrict
from operators.sense import Sense


def crop_center(t, factor=DEFAULT_FACTOR):
    """The central `1/factor` of the last two dims (k-space, centred).

    A size-1 dim is a broadcast dim (a readout-constant mask stored as
    `(..., 1, W)`) and is left alone. The crop starts at `n/2 - n_c/2`, which
    keeps the centred-FFT DC sample (index `n // 2`) at the coarse DC index
    (`n_c // 2`) for even sizes.
    """
    out = t
    for d in (-2, -1):
        n = out.shape[d]
        if n == 1:
            continue
        if n % factor:
            raise ValueError(f"size {n} along dim {d} is not a multiple of {factor}")
        nc = n // factor
        lo = n // 2 - nc // 2
        out = out.narrow(d, lo, nc)
    return out


def coarse_smaps(smaps, factor=DEFAULT_FACTOR, filter=None, renorm=True,
                 eps=1e-12):
    """`restrict(S)`, optionally renormalised to unit RSS where nonzero.

    Restriction is an average, so near the edge of the support (or where coil
    phases rotate within a coarse pixel) the RSS drops below 1; `renorm`
    restores the unit-RSS convention the fine maps satisfy.
    """
    sc = restrict(smaps, factor, filter=filter)
    if renorm:
        rss = sc.abs().pow(2).sum(dim=1, keepdim=True).sqrt()
        sc = torch.where(rss > eps, sc / rss.clamp_min(eps), sc)
    return sc


def coarse_sense(mask, smaps, factor=DEFAULT_FACTOR, filter=None, renorm=True):
    """`(E_c, mask_c, smaps_c)` with `E_c = Mask(mask_c) @ FFT2D() @ Sense(smaps_c)`."""
    mask_c = crop_center(mask, factor)
    smaps_c = coarse_smaps(smaps, factor, filter=filter, renorm=renorm)
    return Mask(mask_c) @ FFT2D() @ Sense(smaps_c), mask_c, smaps_c


def coarse_data(y, factor=DEFAULT_FACTOR):
    """The measurement the coarse operator is consistent with: `crop(y) / factor`."""
    return crop_center(y, factor) / factor


COARSE_OPS = ("galerkin", "rediscretize")


def rediscretize(E, factor=DEFAULT_FACTOR, filter=None):
    """The rediscretized `E_c` for a PLAIN `Mask @ FFT2D @ Sense`, else None.

    Anything else -- a `Truncate` from the image-domain embedding (a measured
    size that is not a multiple of the model's stride), a whitening gain, an
    already-Galerkin `E @ Resample`, a soft (multi-map) Sense -- has no
    rediscretized form here, and the caller falls back to Galerkin. So does an
    odd grid, which the centre crop cannot halve.
    """
    from operators.accessors import _ops

    ops = _ops(E)
    if len(ops) != 3 or not (isinstance(ops[0], Mask) and isinstance(ops[1], FFT2D)
                             and type(ops[2]) is Sense):
        return None
    mask, smaps = ops[0].mask, ops[2].smaps
    if not torch.is_tensor(mask):
        return None
    H, W = smaps.shape[-2:]
    if H % factor or W % factor:
        return None
    E_c, _, _ = coarse_sense(mask, smaps, factor, filter=filter)
    return E_c


def coarsen(E, coarse_op="galerkin", filter=None):
    """One level's coarse operator: Galerkin `E . P`, or rediscretized.

    "rediscretize" falls back to Galerkin wherever `rediscretize` returns None
    (see there), so a batch the rediscretization cannot express still trains,
    at the old cost.
    """
    from operators.resample import galerkin

    if coarse_op not in COARSE_OPS:
        raise ValueError(f"coarse_op must be one of {COARSE_OPS}, got {coarse_op!r}")
    if coarse_op == "rediscretize":
        E_c = rediscretize(E, filter=filter)
        if E_c is not None:
            return E_c
    return galerkin(E, filter=filter)
