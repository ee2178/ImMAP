"""
Learned Primal-Dual Splitting: the smoother, and a stack of them.

PyTorch port of Sljiva's `src/networks/lpds.jl` (`LPDSLayer`) plus the
`LPDSStack` from `mg_lpds.jl` §1.  This is the LPDS counterpart of
`models/lista.py`, and it exists because LPDS is a genuinely different
classical algorithm from ISTA -- not a different prox on the same iteration.

A LISTA layer propagates ONE variable, the sparse code, and `x = D z` is
derived.  LPDS solves the saddle-point problem

    min_x max_z  f(x) + <A x, z> - g*(z),        f(x) = 1/2 ||E x - y||^2

so it propagates a PAIR `u = (x, z)`:

    x  -- primal, IMAGE grid,  C channels (never widened across MG levels)
    z  -- dual,   LATENT grid, M channels (widened U-Net style)

One sweep:

    x+ = x - tau (grad f(x) + A^T z - pi_x),   grad f(x) = E^H E x - y~
    xt = x+ + theta (x+ - x)                   primal over-relaxation
    z+ = prox_{g*}(z + A xt - pi_z)            dual step (Moreau / FenchelProx)

`(pi_x, pi_z)` are the two FAS corrections, absent (and skipped) outside a
V-cycle.  There are TWO of them because the saddle-point problem has two
variables -- which is why `models/multigrid.py`'s single-`pi`, single-`alpha`
`VCycle` cannot host this layer, and `models/mg_lpds.py` exists.

`tau` and `theta` are noise-adaptive per-PRIMAL-channel polynomials
(`Polynomial(C)`), matching `lpds.jl`'s `Polynomial(2, first(ch))`.  `theta` is
clamped to [0, 1] by `project_`, `tau` to [0, inf).

The prox is the FENCHEL conjugate's -- clipping rather than shrinkage -- which
`build_prox(..., dual=True)` already provides via `FenchelProx`.  A group prox
drops in through the same slot (`window > 1`, `Mh`), though nothing in the
current grid uses one.
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


def gram(E, x):
    """`E^H E x`, with the identity short-circuited."""
    if E is None or isinstance(E, Identity):
        return x
    return E.gram(x)


# ---------------------------------------------------------------------------
class LPDSLayer(nn.Module):
    """One primal-dual sweep.  Port of `lpds.jl::LPDSLayer`."""

    def __init__(self, C, M, P=7, stride=1, lam0=1e-2, tau0=1e-1, theta0=1e-1,
                 degrees=0, is_complex=True, window=1, Mh=None, prox_kws=None):
        super().__init__()
        self.C, self.M, self.P, self.stride = int(C), int(M), int(P), int(stride)
        self.is_complex = bool(is_complex)

        self.analysis = Conv2d(C, M, P, stride=stride, bias=False,
                               complex=is_complex)
        self.synthesis = ConvTranspose2d(M, C, P, stride=stride, bias=False,
                                         complex=is_complex)

        # dual=True: prox_{g*} = I - prox_g, i.e. clipping. That IS the LPDS
        # dual step; `lpds.jl` spells it `fenchel(l.prox, ...)` at the call site.
        prox_kws = dict(prox_kws or {})
        prox_kws.setdefault("tau0", lam0)
        prox_kws.setdefault("degrees", degrees)
        self.prox = build_prox(M, window=window, Mh=Mh, dual=True, **prox_kws)

        # Per-PRIMAL-channel, noise-adaptive. tau is the primal step, theta the
        # over-relaxation weight -- both live on the image grid, hence C not M.
        self.tau = Polynomial(C, degrees=degrees, tau0=tau0)
        self.theta = Polynomial(C, degrees=degrees, tau0=theta0)

    # -- forward ------------------------------------------------------------
    def forward(self, state, y_tilde, E=None, sigma=None, pi=None, cache=None):
        """One sweep.  `state` is `(x, z)`, or None for the cold start.

        `pi` is `(pi_x, pi_z)` or None.  Returns `((x, z), cache)`.
        """
        if cache is None:
            cache = {}

        if state is None:
            # Cold start (`lpds.jl` third method): x = y~, z = prox_{g*}(A y~).
            x = y_tilde
            z, cache = self.prox(self.analysis(x), sigma, cache)
            return (x, z), cache

        x, z = state
        pi_x, pi_z = (None, None) if pi is None else pi

        tau = self.tau(sigma, ref=x)
        theta = self.theta(sigma, ref=x)

        residual = gram(E, x) - y_tilde + self.synthesis(z)
        if pi_x is not None:
            residual = residual - pi_x
        x_new = x - tau * residual

        x_bar = x_new + theta * (x_new - x)          # over-relaxation

        u = z + self.analysis(x_bar)
        if pi_z is not None:
            u = u - pi_z
        z_new, cache = self.prox(u, sigma, cache)

        return (x_new, z_new), cache

    # -- constraints --------------------------------------------------------
    @torch.no_grad()
    def project_(self):
        set_weight(self.analysis, uball_project(self.analysis.weight))
        set_weight(self.synthesis, uball_project(self.synthesis.weight))
        self.tau.project_(lo=0.0)
        self.theta.project_(lo=0.0, hi=1.0)      # over-relaxation stays in [0,1]
        if hasattr(self.prox, "project_"):
            self.prox.project_()

    # -- initialisation -----------------------------------------------------
    @torch.no_grad()
    def init_filters(self, weight=None):
        """Tie the synthesis to the analysis: `B = A^H`.

        Julia writes `synthesis = deepcopy(analysis)` because Lux's
        ConvTranspose stores the subband axis in the same slot; in torch the
        conjugate is what makes `B = A^H` (see docs/multigrid_port.md).
        """
        dtype = torch.complex64 if self.is_complex else torch.float32
        if weight is None:
            weight = torch.randn(self.M, self.C, self.P, self.P, dtype=dtype)
        set_weight(self.analysis, weight)
        set_weight(self.synthesis, weight.conj())
        return weight

    @torch.no_grad()
    def spectral_normalize(self, size=128, num_iter=200, verbose=False):
        """Scale (A, B) so `||A B||_2 = 1`, as `lpds.jl` does at init."""
        w = self.analysis.weight
        x0 = torch.rand(1, self.C, size, size, dtype=w.dtype)
        L = power_method(lambda x: self.synthesis(self.analysis(x)), x0,
                         num_iter=num_iter, verbose=verbose)[0]
        scale = float(np.sqrt(np.abs(L)))
        set_weight(self.analysis, self.analysis.weight / scale)
        set_weight(self.synthesis, self.synthesis.weight / scale)
        return L


def make_lpds_layer(C, M, spectral_init=True, **kws):
    """One `LPDSLayer`, filters tied and optionally spectrally normalised.

    Coarse levels skip the power method: `preload_with_widening` overwrites
    their filters immediately afterwards and the widening preserves the norm.
    """
    layer = LPDSLayer(C, M, **kws)
    layer.init_filters()
    if spectral_init:
        layer.spectral_normalize()
    return layer


# ---------------------------------------------------------------------------
class LPDSStack(nn.Module):
    """J untied `LPDSLayer`s.  The LPDS counterpart of `models/lista.py::LISTA`.

    Interchangeable with `PDVCycle` -- both take
    `(state, y_tilde, E, sigma, pi, cache)` and return `((x, z), cache)`, which
    is what lets a V-cycle hold either one as its coarser level.
    """

    def __init__(self, J, layer_factory):
        super().__init__()
        self.J = int(J)
        proto = layer_factory()
        self.layers = nn.ModuleList([proto] + [copy.deepcopy(proto)
                                               for _ in range(self.J - 1)])

    @property
    def first(self):
        return self.layers[0]

    def forward(self, state, y_tilde, E=None, sigma=None, pi=None, cache=None):
        if cache is None:
            cache = {}
        for layer in self.layers:
            state, cache = layer(state, y_tilde, E=E, sigma=sigma, pi=pi,
                                 cache=cache)
        return state, cache

    def extra_repr(self):
        return f"J={self.J}"


def make_lpds_stack(J, C, M, **kws):
    """A `LPDSStack` of plain layers, spectrally initialised once.

    The prototype is initialised BEFORE replication so every layer starts
    identical -- layer j is initially the same primal-dual step, and training
    pulls them apart.
    """
    return LPDSStack(J, lambda: make_lpds_layer(C, M, **kws))
