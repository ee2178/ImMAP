import torch
import torch.fft as fft
import torch.nn.functional as F
import math
from typing import Tuple
from operators.base import Operator

### General MRI Utilities

def fftc(x, dim = (-2, -1), mode = 'ortho', real = False):
    # If our input is real, use rfft and irfft
    if real is True:
        return fft.fftshift(fft.rfftn(fft.ifftshift(x, dim = dim), dim = dim, norm = mode), dim = dim)
    else:
        return fft.fftshift(fft.fftn(fft.ifftshift(x, dim = dim), dim = dim, norm = mode), dim = dim)

def ifftc(x, dim = (-2, -1), mode = 'ortho', real = False):
    if real is True:
        return fft.fftshift(fft.irfftn(fft.ifftshift(x, dim = dim), dim = dim, norm = mode), dim = dim)
    else:
        return fft.fftshift(fft.ifftn(fft.ifftshift(x, dim = dim), dim = dim, norm = mode), dim = dim)

def mri_encoding(x, mask, smaps):
    # x         B x 1 x H x W
    # smaps     B x C x H x W
    # mask      B x 1 x H x W
    x_coils = smaps * x         # B x C x H x W
    y_coils = fftc(x_coils)     # B x C x H x W
    y_mask = y_coils * mask     # B x C x H x W
    return y_mask

def mri_decoding(y, mask, smaps):
    # y         B x C x H x W
    # smaps     B x C x H x W
    # mask      B x 1 x H x W
    y_mask = mask * y           # B x C x H x W
    x_coils = ifftc(y_mask)     # B x C x H x W
    x = torch.sum(smaps.conj()*x_coils, dim = 1, keepdim = True) # B x 1 x H x W
    return x

# NOTE: `mri_awgn` lives in operators/noise.py. A stale 4-argument copy used to
# sit here; it never applied the sensitivity maps and referenced an undefined
# name, and having two functions of the same name in sibling modules made the
# wrong one easy to import.

### Creating Operator Classes for Fourier and MRI Ops

class FFT2D(Operator):
    # Use our operator class and existing fftc functions to define a fourier operator class with a proper adjoint
    OP_KIND = "centered_fft"

    def forward(self, x):
        return fftc(x)

    def adjoint(self, x):
        return ifftc(x)

    # -- fused SENSE Gram ---------------------------------------------------
    def sense_gram(self, x, smaps, mask):
        """`E^H E x` for `E = Mask(m) @ FFT2D() @ Sense(s)`, without the shifts.

        Writing `A = fftshift` and `B = ifftshift`, so `fftc = A F B` and
        `ifftc = A F^-1 B`, the Gram expands to

            sum_c conj(s) . A F^-1 B [ conj(m) . m . A F B (s.x) ]

        `B` and `A` are mutually inverse circular rolls, so `B(q . A F B u)`
        is `B(q) . F(B u)` and the inner pair collapses.  What is left is
        `A G(B u)` with `G = F^-1 diag(B(|m|^2)) F` -- a circular convolution,
        which commutes with the roll `B`, so `A G B = G`.  Every shift cancels:

            E^H E x = sum_c conj(s) . F^-1 [ ifftshift(|m|^2) . F (s . x) ]

        That drops four `fftshift`s over a (B, coils, H, W) complex tensor and
        one of the two mask multiplies.  Exact, not an approximation: for a 0/1
        mask it agrees with the generic path bitwise, and the roll-commutation
        argument needs no parity assumption on H or W.
        """
        w = self._gram_weight(mask)
        U = fft.fftn(smaps * x, dim=(-2, -1), norm="ortho")
        v = fft.ifftn(U * w, dim=(-2, -1), norm="ortho")
        return torch.sum(torch.conj(smaps) * v, dim=1, keepdim=True)

    def _gram_weight(self, mask):
        """`ifftshift(|mask|^2)`, memoised against the mask's identity.

        One `E` serves every level of a V-cycle (`galerkin` reuses the same
        sub-operators), so this is built once per forward pass rather than once
        per Gram.  A mask that requires grad is never cached -- the accessors
        in `operators/accessors.py` build a fresh `Mask` whenever the maps or
        the mask change, so a stale hit is impossible, but keeping an autograd
        node alive across calls is not worth the saving.
        """
        cached = getattr(self, "_gram_weight_cache", None)
        if cached is not None and cached[0] is mask:
            return cached[1]
        w = fft.ifftshift(mask.abs() ** 2, dim=(-2, -1))
        if not mask.requires_grad:
            self._gram_weight_cache = (mask, w)
        return w


