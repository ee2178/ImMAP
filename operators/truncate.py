"""
Image-domain embedding: solve on a larger grid, measure on the original one.

An unrolled multigrid network needs its image grid divisible by
`pad_stride = s * 2**(levels - 1)` -- every level halves, and every level's
strided convolution has to tile its own grid exactly.  fastMRI matrix sizes do
not oblige.

`preprocessing/kspace.py` handles that by growing the OPERATOR: the sampling
mask is nearest-neighbour resampled and the coil maps are reflect-padded.  That
is approximate by construction (its own docstring says so -- a Fourier transform
on a larger grid is a different transform), and the resampled mask claims
phase-encode lines the data was never measured at.

This module takes the other route.  The measurement is fine; only the network
wants a rounder grid.  So leave `E` alone and enlarge the IMAGE:

    y  =  M F S . Truncate(x'),        x' in C^(H' x W'),  H' = ceil(H/m) * m

`Truncate` crops the padded grid down to the measured one; its adjoint is
zero-padding.  Crop and zero-pad are an exact adjoint pair, so

    E' = E @ Truncate      is a genuine linear operator, and
    E'^H E' = T^H E^H E T  holds exactly

with no resampling anywhere.  `y`, the mask and the maps are untouched; the
network solves for a slightly larger image and the read-out crops it back.

Reasonable for MRI specifically because the anatomy already sits inside the FOV,
so the added pixels carry no signal rather than invented content.

Two properties worth knowing
----------------------------
`Mask @ FFT2D @ Sense @ Truncate` still matches the fused SENSE Gram:
`operators.base._match_sense_gram` keys on the first three operators and passes
the rest through as an untyped tail -- the same slot `galerkin` uses for its
grid transfers -- so the fast path costs nothing extra and computes literally
`T^H (E^H E) T`.

The border is UNCONSTRAINED.  `T` annihilates it, so the data term's gradient
there is exactly zero: those pixels move only through the dictionary term, and
the analysis operator reads them back into the code on the next sweep.  They are
cropped at read-out, so they cannot corrupt the reported image directly, but
they are capacity spent on nothing.  `border_fraction` below reports how much
output energy lands there; if it ever grows, zeroing `x` outside the window
between sweeps is the next lever.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from operators.base import Operator


def next_multiple(n, m):
    """Smallest multiple of `m` that is >= `n`."""
    m = int(m)
    if m <= 1:
        return int(n)
    return ((int(n) + m - 1) // m) * m


def embedded_size(hw, multiple):
    """`(H', W')`, the smallest grid >= `hw` divisible by `multiple`."""
    return (next_multiple(hw[0], multiple), next_multiple(hw[1], multiple))


class Truncate(Operator):
    """Centred crop `big -> small`, with zero-padding as its exact adjoint.

    forward : x' on the padded grid  ->  x on the measured grid
    adjoint : x                      ->  x' with zeros outside the window

    `big == small` makes both directions the identity, so composing this
    unconditionally costs nothing when the size already divides.
    """

    def __init__(self, big, small):
        self.big = (int(big[0]), int(big[1]))
        self.small = (int(small[0]), int(small[1]))
        if self.big[0] < self.small[0] or self.big[1] < self.small[1]:
            raise ValueError(
                f"Truncate needs big >= small, got {self.big} < {self.small}. "
                f"The embedding only ever GROWS the image grid.")
        self.top = (self.big[0] - self.small[0]) // 2
        self.left = (self.big[1] - self.small[1]) // 2
        self.is_identity = self.big == self.small

    def forward(self, x):
        if self.is_identity:
            return x
        t, l = self.top, self.left
        return x[..., t:t + self.small[0], l:l + self.small[1]]

    def adjoint(self, x):
        if self.is_identity:
            return x
        dh = self.big[0] - self.small[0]
        dw = self.big[1] - self.small[1]
        pad = (self.left, dw - self.left, self.top, dh - self.top)
        # F.pad's kernels are real-only for complex input on some builds;
        # mirror preprocessing.kspace._pad_complex and pad the halves.
        if torch.is_complex(x):
            return torch.complex(F.pad(x.real, pad), F.pad(x.imag, pad))
        return F.pad(x, pad)

    def border_fraction(self, x):
        """Share of `x`'s energy outside the measured window, in [0, 1]."""
        if self.is_identity:
            return 0.0
        total = float(x.abs().pow(2).sum())
        if total <= 0.0:
            return 0.0
        inside = float(self.forward(x).abs().pow(2).sum())
        return max(0.0, (total - inside) / total)

    def __repr__(self):
        return f"Truncate({self.big[0]}x{self.big[1]}->{self.small[0]}x{self.small[1]})"


def embed_operator(E, hw, multiple):
    """`(E @ Truncate, T)` for a measured grid `hw` embedded to `multiple`.

    Returns the operator the network should unroll with and the `Truncate`
    needed to crop its output back. `multiple <= 1`, or a size that already
    divides, gives an identity `Truncate` and an operator equal to `E`.
    """
    big = embedded_size(hw, multiple)
    T = Truncate(big, hw)
    return (E if T.is_identity else E @ T), T
