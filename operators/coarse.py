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
tests/test_coarse_operator.py measures. Not wired into any model.
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
