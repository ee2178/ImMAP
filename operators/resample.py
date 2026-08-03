"""
Resampling operators and the multigrid restriction / prolongation pair.

Port of the `Resample` / `galerkin` / `meanpool` / `upsample` section of
Sljiva's `src/operators.jl`, in PyTorch (B, C, H, W) layout.

Two things live here:

  * `restrict` / `prolong` -- the grid-transfer pair used by the V-cycle.
    `restrict` is a 2x2 mean (a full-weighting-style restriction on a
    cell-centred grid); `prolong` is bilinear interpolation by 2.

  * `Resample` / `galerkin` -- the *operator-level* coarsening.  The coarse
    forward model is  E_c = E . R  (R = prolongation), so the coarse Gram is
    R^H E^H E R: a coarse latent is lifted to the fine grid, sees the true
    physics, and comes back.  No coarse measurement model is ever needed.

    `galerkin(Identity()) -> Identity()` is the one optimisation that matters
    most in practice: for denoising the coarse Gram is the identity, so the
    whole V-cycle runs without a single resampling round-trip.

Notes on fidelity to the Julia source
-------------------------------------
`align_corners`: NNlib's `upsample_bilinear` defaults to align_corners=True
(vertex-centred).  The restriction here is a 2x2 cell average, which is
cell-centred, so align_corners=False is the variationally consistent partner
(it makes P proportional to R^T).  We default to False and expose the flag.

`restrict` padding: the Julia code always pads (1,0,1,0) before mean-pooling,
which for an EVEN input introduces a half-cell shift and attenuates the first
row/column (its window is (0+0+0+x11)/4).  Since the network pads inputs up to
a multiple of s*2^L, every level is even and that padding is never needed.  We
pad only when a dimension is odd; pass `julia_compat=True` to always pad.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from operators.base import Operator
from operators.identity import Identity


# ---------------------------------------------------------------------------
# complex-safe wrappers (torch's pooling / interpolation are real-only)
# ---------------------------------------------------------------------------
def _apply_complex(fn, x):
    if torch.is_complex(x):
        return torch.complex(fn(x.real), fn(x.imag))
    return fn(x)


def prolong(x, scale=2, align_corners=False):
    """Bilinear interpolation by `scale` (the multigrid prolongation P)."""
    def fn(t):
        return F.interpolate(t, scale_factor=float(scale), mode="bilinear",
                             align_corners=align_corners)
    return _apply_complex(fn, x)


def restrict(x, julia_compat=False):
    """2x2 mean-pooling restriction R.  Output is ceil(H/2) x ceil(W/2).

    Singleton spatial dims are passed through untouched, so this is safe to
    call on a broadcast noise map of shape (B, 1, 1, 1).
    """
    H, W = x.shape[-2], x.shape[-1]
    kh = 2 if H > 1 else 1
    kw = 2 if W > 1 else 1
    if kh == 1 and kw == 1:
        return x
    ph = 1 if (kh == 2 and (julia_compat or H % 2)) else 0
    pw = 1 if (kw == 2 and (julia_compat or W % 2)) else 0

    def fn(t):
        if ph or pw:
            t = F.pad(t, (pw, 0, ph, 0))
        return F.avg_pool2d(t, (kh, kw), stride=(kh, kw))

    return _apply_complex(fn, x)


def restrict_noise(sigma, julia_compat=False):
    """Coarse-grid noise level.

    Averaging a 2x2 block of i.i.d. noise divides the standard deviation by 2,
    so sigma_c = R(sigma) / 2.  Works for a scalar, a per-batch scalar, or a
    full spatial noise map.
    """
    if sigma is None:
        return None
    if torch.is_tensor(sigma) and sigma.dim() == 4:
        sigma = restrict(sigma, julia_compat=julia_compat)
    return sigma * 0.5


# ---------------------------------------------------------------------------
# operator form
# ---------------------------------------------------------------------------
class Resample(Operator):
    """Bilinear resampling as a linear operator.

    forward  : interpolate by `scale`      (prolongation, coarse -> fine)
    adjoint  : interpolate by `1 / scale`  (restriction,  fine -> coarse)

    The adjoint is the *approximate* one used by the Julia source -- bilinear
    down-sampling rather than the exact transpose of bilinear up-sampling.
    That is fine here because this operator is only ever used inside a gradient
    step (never inside a CG solve, where a non-symmetric Gram would break the
    Krylov recursion).
    """

    def __init__(self, scale=2, align_corners=False):
        self.scale = float(scale)
        self.align_corners = align_corners

    def forward(self, x):
        return prolong(x, self.scale, self.align_corners)

    def adjoint(self, x):
        return prolong(x, 1.0 / self.scale, self.align_corners)

    def __repr__(self):
        return f"Resample(scale={self.scale})"


def make_resample(scale, align_corners=False):
    """`Resample(1)` is the identity -- return the cheap operator instead."""
    return Identity() if float(scale) == 1.0 else Resample(scale, align_corners)


def galerkin(E, scale=2, align_corners=False):
    """Coarse-grid forward operator  E_c = E . R.

    Dispatches on the identity so that denoising V-cycles never resample:
    `galerkin(Identity()) is Identity()` and the coarse Gram stays free.
    """
    if E is None:
        return Identity()
    if isinstance(E, Identity):
        return E
    return E @ Resample(scale, align_corners)
