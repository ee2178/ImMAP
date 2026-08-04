"""
Measurement-domain preprocessing for unrolled reconstruction.

Port of `KSpacePreprocess` in Sljiva's `src/networks/preproc.jl`, plus the DC
correction that version is missing.

Why plain mean subtraction is wrong under a real operator
---------------------------------------------------------
`preprocessing/image.py::pre_process` subtracts `mean(E^H y)` and
`post_process` adds it back.  For denoising (`E = I`) that is exactly right.
For reconstruction it is not, and the algebra says why.  Split the image into a
DC part and the rest, `x = v + mu 1`:

    1/2 ||y - E(v + mu 1)||^2  =  1/2 || (y - mu E 1) - E v ||^2

    grad_v  =  E^H E v  -  ( E^H y  -  mu E^H E 1 )

A LISTA layer computes `z <- prox(z - A(E^H E B z - y~))`, so the surrogate the
network must be handed is

    y~  =  E^H y  -  mu . E^H E 1                                        (*)

and the read-out must add `mu` back.  The old code used `E^H y - mean(E^H y) 1`,
which is wrong twice over: the DC image is pushed through `E^H E` in (*), not
left alone, and `mean(E^H y)` is not `mean(x)` once coil maps are involved --
`E^H E` is only a Fourier multiplier in the single-coil case, and the maps
modulate the DC away from it.  Both errors vanish at `E = I`
(`E^H E 1 = 1`, `mean(E^H y) = mean(y)`), which is why denoising never showed it.

Choosing mu
-----------
Any scalar `mu` makes (*) an exact reformulation, provided the same `mu` is
added back -- the data-fidelity bookkeeping does not care.  What `mu` buys is
centering, so the dictionary does not have to spend atoms representing DC (and
it does shift the regularizer, which acts on the code of `x - mu 1`).

So take the `mu` that best explains the measured DC content, i.e. the exact
minimizer of `1/2 ||y - E(mu 1)||^2`:

    mu = <E 1, y> / ||E 1||^2 = <1, E^H y> / <1, E^H E 1>
       = sum(E^H y) / sum(E^H E 1)                                       (**)

which is a one-line change from `mean(E^H y)` -- it is that mean divided by
`mean(E^H E 1)` -- reuses the `E^H E 1` that (*) needs anyway, needs no ACS
window or low-resolution reconstruction, and collapses to `mean(y)` at `E = I`.
`||E 1||^2 > 0` whenever the mask keeps any DC content, which a centre-sampled
Cartesian mask always does.

An ACS-based estimate (inverse-FFT the fully sampled centre, take the mean of
the low-resolution image) also works and is what one reaches for first, but it
is only approximate -- it needs the window's DC gain to be one and the maps to
be smooth across the window -- and it introduces a window hyperparameter that
(**) does not need.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from operators.accessors import (get_mask, get_sensitivity_map, set_mask,
                                 set_sensitivity_map)
from operators.identity import Identity
from operators.padding import calc_pad_2d, unpad


def _pad_complex(x, pad, mode="reflect"):
    """`F.pad` that survives a complex tensor (torch's reflect pad is real-only)."""
    if not torch.is_complex(x):
        return F.pad(x, pad, mode=mode)
    return torch.complex(F.pad(x.real, pad, mode=mode),
                         F.pad(x.imag, pad, mode=mode))


def pad_operator(E, pad, size):
    """Grow `E` onto the padded grid: mask stretched, coil maps reflect-padded.

    Follows `KSpacePreprocess` in `preproc.jl`: the sampling mask is resampled
    with nearest-neighbour interpolation (a Cartesian line pattern stays a line
    pattern) and the sensitivity maps are reflect-padded like the image.

    This is inherently approximate -- a Fourier transform on a larger grid is a
    different transform, so the padded operator is not the original one -- and
    it is why the configs size their data to a multiple of `pad_stride`, where
    the pad is empty and this is the identity.
    """
    H, W = size

    mask = get_mask(E)
    if mask.shape[-2:] != (H, W):
        mask = F.interpolate(mask.to(torch.float32).expand(-1, 1, -1, -1),
                             size=(H, W), mode="nearest")
        E = set_mask(E, mask)

    smaps = get_sensitivity_map(E)
    if smaps.shape[-2:] != (H, W):
        E = set_sensitivity_map(E, _pad_complex(smaps, pad))

    return E


def kspace_pre_process(y, E, stride):
    """Adjoint, pad, and DC-correct a measurement for an unrolled network.

    Returns `(y_tilde, E, params)` where `y_tilde` is the surrogate (*) above
    and `params` is what `kspace_post_process` needs.

    `E` comes back possibly replaced -- padding grows its mask and maps -- so
    the caller MUST use the returned operator for the unroll.  The noise map is
    left alone; the caller resizes it against `y_tilde`'s latent grid with the
    `resize_noise` it already uses for every other preprocessing mode.
    """
    x_adj = E.adjoint(y) if not isinstance(E, Identity) else y

    # ---- pad the image AND the operator, so the two still agree -------------
    pad = calc_pad_2d(*x_adj.shape[-2:], stride)
    if any(pad):
        x_adj = _pad_complex(x_adj, pad)
        if not isinstance(E, Identity):
            E = pad_operator(E, pad, x_adj.shape[-2:])

    # ---- DC removal --------------------------------------------------------
    if isinstance(E, Identity):
        # E^H E 1 = 1: (*) and (**) both collapse to plain mean subtraction.
        mu = x_adj.mean(dim=(-2, -1), keepdim=True)
        return x_adj - mu, E, (mu, pad)

    EtE1 = E.gram(torch.ones_like(x_adj))

    num = x_adj.sum(dim=(-2, -1), keepdim=True)
    den = EtE1.sum(dim=(-2, -1), keepdim=True)
    # den = ||E 1||^2 is real and positive whenever the mask keeps any DC (a
    # centre-sampled Cartesian mask always does); the guard covers an all-zero
    # mask only.
    mu = num / torch.where(den.abs() > 1e-12, den, torch.ones_like(den))

    return x_adj - mu * EtE1, E, (mu, pad)


def kspace_post_process(x, params):
    """Add the DC back and undo the padding."""
    mu, pad = params
    return unpad(x + mu, pad)
