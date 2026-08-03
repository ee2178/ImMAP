"""
LISTA building blocks: one unrolled proximal-gradient step, and a stack of them.

PyTorch port of Sljiva's `src/networks/lista.jl`.  `CDLNet`, `GroupCDL`, the
multigrid V-cycle and the LADMM denoiser prox are all assembled from these two
classes.

One layer computes

    z <- prox( z + eta(sigma) . pi  -  A ( E^H E (B z) - y~ ) ,  sigma )

with `pi` the multigrid FAS correction (absent, i.e. skipped, outside a
V-cycle) and `eta` a learned noise-adaptive per-channel scale.  With `z = None`
the layer takes the cold-start shortcut `z = prox(A y~, sigma)` -- which is
what `z^(0) = 0` reduces to, at the cost of one synthesis and one Gram apply
saved on the first iteration.

A `LISTA` block holds K structurally identical but *untied* layers.  They are
built once and deep-copied, so every layer starts from the same
spectrally-normalised dictionary -- i.e. layer k is initially an exact ISTA
step, and training pulls them apart from there.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

from models.base import set_weight
from models.components import Conv2d, ConvTranspose2d
from models.prox import Polynomial, build_prox
from operators.identity import Identity
from operators.projections import uball_project
from solvers.eigen import power_method


# ---------------------------------------------------------------------------
def gram(E, x):
    """`E^H E x`, with the identity short-circuited (denoising costs nothing)."""
    if E is None or isinstance(E, Identity):
        return x
    return E.gram(x)


# ---------------------------------------------------------------------------
class LISTALayer(nn.Module):
    """A single unrolled proximal-gradient iteration."""

    def __init__(self, C, M, P=7, stride=1, tau0=1e-2, degrees=0,
                 is_complex=True, multigrid=False, eta0=1e-1, eta_degrees=0,
                 window=1, Mh=None, dual=False, prox_kws=None):
        super().__init__()
        self.C, self.M, self.P, self.stride = int(C), int(M), int(P), int(stride)
        self.is_complex = bool(is_complex)

        self.analysis = Conv2d(C, M, P, stride=stride, bias=False, complex=is_complex)
        self.synthesis = ConvTranspose2d(M, C, P, stride=stride, bias=False,
                                         complex=is_complex)

        prox_kws = dict(prox_kws or {})
        prox_kws.setdefault("tau0", tau0)
        prox_kws.setdefault("degrees", degrees)
        self.prox = build_prox(M, window=window, Mh=Mh, dual=dual, **prox_kws)

        # eta scales the multigrid FAS correction pi.  Only allocated inside a
        # V-cycle -- a plain LISTA never sees a pi and would carry dead weights.
        self.eta = Polynomial(M, degrees=eta_degrees, tau0=eta0) if multigrid else None

    # -- forward ------------------------------------------------------------
    def forward(self, z, y_tilde, E=None, sigma=None, pi=None, cache=None):
        if z is None:                                  # cold start, z^(0) = 0
            return self.prox(self.analysis(y_tilde), sigma, cache)

        residual = gram(E, self.synthesis(z)) - y_tilde
        u = z - self.analysis(residual)
        if pi is not None and self.eta is not None:
            u = u + self.eta(sigma, ref=z) * pi
        return self.prox(u, sigma, cache)

    # -- constraints --------------------------------------------------------
    @torch.no_grad()
    def project_(self):
        set_weight(self.analysis, uball_project(self.analysis.weight))
        set_weight(self.synthesis, uball_project(self.synthesis.weight))
        if self.eta is not None:
            self.eta.project_(lo=0.0)

    # -- initialisation -----------------------------------------------------
    @torch.no_grad()
    def init_filters(self, weight=None):
        """Tie the synthesis to the analysis: `B = A^H` (conjugate filters)."""
        dtype = torch.complex64 if self.is_complex else torch.float32
        if weight is None:
            weight = torch.randn(self.M, self.C, self.P, self.P, dtype=dtype)
        set_weight(self.analysis, weight)
        set_weight(self.synthesis, weight.conj())
        return weight

    @torch.no_grad()
    def spectral_normalize(self, size=128, num_iter=200, verbose=False):
        """Scale (A, B) so that `||B A||_2 = 1`, making the ISTA step size 1.

        Writes go through `set_weight` -- see its docstring for why
        `layer.analysis.weight.data /= s` is a silent no-op on complex convs.
        """
        w = self.analysis.weight
        x0 = torch.rand(1, self.C, size, size, dtype=w.dtype)
        L = power_method(lambda x: self.synthesis(self.analysis(x)), x0,
                         num_iter=num_iter, verbose=verbose)[0]
        scale = float(np.sqrt(np.abs(L)))
        set_weight(self.analysis, self.analysis.weight / scale)
        set_weight(self.synthesis, self.synthesis.weight / scale)
        return L


# ---------------------------------------------------------------------------
class LISTA(nn.Module):
    """K untied iterations of `layer_factory()`.

    `layer_factory` returns either a `LISTALayer` (plain LISTA) or a
    `VCycle` (multigrid LISTA) -- the two are interchangeable because they
    share the `(z, y, E, sigma, pi, cache)` signature.
    """

    def __init__(self, K, layer_factory):
        super().__init__()
        self.K = int(K)
        proto = layer_factory()
        self.layers = nn.ModuleList([proto] + [copy.deepcopy(proto)
                                               for _ in range(self.K - 1)])

    @property
    def first(self):
        return self.layers[0]

    def forward(self, z, y_tilde, E=None, sigma=None, pi=None, cache=None):
        if cache is None:
            cache = {}
        for layer in self.layers:
            z, cache = layer(z, y_tilde, E=E, sigma=sigma, pi=pi, cache=cache)
        return z, cache

    def extra_repr(self):
        return f"K={self.K}"


# ---------------------------------------------------------------------------
def make_lista(K, C, M, **kws):
    """Build a `LISTA` of plain `LISTALayer`s, spectrally initialised once.

    The prototype layer is initialised (random filters -> power-method scaling)
    *before* being replicated, so every one of the K layers starts identical --
    matching `Lux.initialparameters(LISTA{K})`, which deep-copies one parameter
    set K times.
    """
    def factory():
        layer = LISTALayer(C, M, **kws)
        layer.init_filters()
        layer.spectral_normalize()
        return layer

    return LISTA(K, factory)
