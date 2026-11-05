"""
RAM's input realignment as a start for any net that reconstructs through E.

RAM (models/ram/_upstream/ram.py::RAM.realign_input) does not hand its U-Net the
adjoint E^H y. It hands it deepinv's `LinearPhysics.prox_l2` at z = E^H y,

    x0 = argmin_x  1/2 ||E x - y||^2 + mu/2 ||x - E^H y||^2
       = (E^H E + mu I)^{-1} (1 + mu) E^H y,            mu = 1 / (gamma * fact)

i.e. CG-SENSE up to the scalar 1 + mu. This module reproduces it with solvers/cg.py,
so no net but RAM needs deepinv, at deepinv 0.4.2's defaults:

  solve     CG on (E^H E + mu I) x = E^H y + mu E^H y, started AT z = E^H y
            (deepinv passes init=z), 50 iterations at most, stopping once
            |r| < 1e-4 |b| for every image in the batch (`batched_cg(stop="rhs")`)
  gradient  implicit: the forward solve runs without a graph and the backward is
            one more CG from zero (`least_squares_implicit_backward`, deepinv's
            default `implicit_backward_solver=True`; here `tcg(implicit=True)`).
            Neither y nor E is learned, so the gradient that matters is mu's.
  mu        RAM's rule, with sigma the per-REAL-component std (ImMAP's complex
            sigma / sqrt 2, as models/ram/adapter.py converts it) and f = 1, since
            no MRI physics carries a `factor`:
                snr   = mean(|Re y|, |Im y|) / (sigma + 1e-4)
                gamma = 1 / (1e-4 + 1 / snr)
            The mean runs over every k-space entry, unsampled zeros included,
            exactly as RAM's does on its (B, 2, C, H, W) tensor.
  fact      one learned scalar, unconstrained (RAM's `fact_realign`). It starts at 1
            as RAM's does unless `fact_init` says otherwise. Trained RAM reaches ~23,
            where mu ~ 0.15 and the start is a partial CG-SENSE, while at 1 our mu is
            3-5 and the start is ~E^H y (results/cg_init/brain_s5*.txt).

What is NOT reproduced: RAM's top-left zero padding to a multiple of 8. On the
padded rows E sees nothing, the system is mu I x = 0 there, and every inner product
is unchanged, so the solve on the original grid is the same solve.
tests/test_cg_init.py checks this module against RAM's own `realign_input`.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from solvers.cg import tcg

DEEPINV_MAX_ITER = 50
DEEPINV_TOL = 1e-4


def per_image_sigma(sigma, batch, device):
    """ImMAP's complex noise std as a (B,) tensor; one level per image, as RAM needs."""
    s = torch.as_tensor(sigma, device=device, dtype=torch.float32)
    if s.numel() == 1:
        return s.reshape(1).expand(batch)
    if s.numel() != batch:
        raise ValueError(f"cg_init needs one noise level per image; got sigma of shape "
                         f"{tuple(s.shape)} for a batch of {batch}")
    return s.reshape(batch)


def ram_signal_level(y):
    """RAM's signal statistic `num` for each image of `y`, as a (B,) tensor.

    mean(|Re y|, |Im y|) over every entry, unsampled zeros included: the mean of
    RAM's real (B, 2, C, H, W) tensor. RAM's snr is this over sigma. A real `y`
    has no imaginary part, so the same statistic is mean|y| / 2 there."""
    B = y.shape[0]
    a = y.real.abs() + y.imag.abs() if torch.is_complex(y) else y.abs()
    return a.reshape(B, -1).sum(1) / (2 * y[0].numel())


def ram_gamma(y, sigma):
    """RAM's prox weight gamma for each image of `y` (B, C, H, W), complex k-space.

    `sigma` is ImMAP's complex std, (B,) or a scalar."""
    B = y.shape[0]
    s = per_image_sigma(sigma, B, y.device) / math.sqrt(2.0)
    num = ram_signal_level(y)
    snr = num / (s + 1e-4)
    return 1.0 / (1e-4 + 1.0 / snr)


class CGInit(nn.Module):
    """x0 = RAM's realigned input, for a net that reconstructs through E.

    `forward(y, E, sigma)` returns the complex image (B, 1, H, W) on E's grid."""

    def __init__(self, max_iter=DEEPINV_MAX_ITER, tol=DEEPINV_TOL, fact_init=1.0):
        super().__init__()
        if not fact_init > 0:
            raise ValueError(f"fact_init must be positive, got {fact_init}")
        self.max_iter, self.tol = int(max_iter), float(tol)
        self.fact = nn.Parameter(torch.tensor([float(fact_init)]))

    def forward(self, y, E, sigma):
        z = E.adjoint(y)
        gamma = ram_gamma(y, sigma).to(z.real.dtype).view(-1, *([1] * (z.dim() - 1)))
        mu = 1.0 / (gamma * self.fact)
        return tcg(E.gram, mu, z + mu * z, implicit=True, tol=self.tol,
                   max_iter=self.max_iter, x0=z, stop="rhs")

    def extra_repr(self):
        return f"max_iter={self.max_iter}, tol={self.tol:g}"
