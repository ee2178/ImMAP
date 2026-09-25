"""
Cascade LPDS: LPDSNet whose analysis operator is a composition of convolutions.

    min_x  1/2 ||E x - y||^2  +  || lambda . K x ||_1,     K = A_L ... A_2 A_1

`A_l` is a dense stride-2 complex conv from level l-1 to level l, with no
nonlinearity in between, and only the deepest code is penalised -- ONE dual.
It is `models/wavelet_lpds.py` with the wavelet structure removed: no tree
groups, no fixed Q mix, no carry channels, no LL^3 thresholds, and a random
init instead of the DT-CWT filters.  The iteration is
`models/lpds.py::LPDSLayer` with `A -> K` and `B -> K^H`.

How it differs from `models/ml_lpds.py`: there EVERY level is penalised (L
duals, L clips); here only level L is, so the shallower levels are purely a
factorisation of the deep analysis filters.

Channels 1 -> 16 -> 64 -> 256 (M = 16, widen = 4, L = 3, s = 2), the same shape
as WaveletLPDSNet, so the two differ only in init and structure.

Init
----
* Each `A_l` is drawn at random (`LISTALayer.init_filters`, complex Gaussian),
  `B_l = A_l^H`, and the pair is spectrally normalised ON ITS OWN
  (`LISTALayer.spectral_normalize`: ||B_l A_l|| = 1, so ||A_l|| = 1) -- per
  conv layer, not on the composed operator.  One prototype is built and
  deep-copied into all K layers by `LPDSStack`, so every layer starts as the
  same classical Condat-Vu step.
* Step sizes are the standard ones (tau0, theta0 from the config, as for
  LPDSNet and MLLPDSNet).  ||K|| <= prod_l ||A_l|| = 1, so tau0 = 0.5 is inside
  the Condat-Vu bound tau (1/2 + ||K||^2) <= 1; `step_bound` measures it.
* Per-channel thresholds at lam0 on all M_L deepest channels.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.components import Conv2d, ConvTranspose2d
from models.lista import LISTALayer, gram
from models.lpds import LPDSStack
from models.ml_cdlnet import _MLIO, _init_size, level_channels, level_strides
from models.prox import Polynomial, build_prox
from models.wavelet_lpds import _cblock, _from_ri, _to_ri
from operators.identity import Identity
from operators.projections import proj_dims
from solvers.eigen import power_method


# -- K and K^H on the interleaved real layout (see models/wavelet_lpds.py) -----
# Pure functions of the weights, so one torch.compile serves all K layers.

def _analyse(x, ws, strides):
    """`K x`: complex (B, C, H, W) -> complex (B, M_L, H/S, W/S)."""
    x = _to_ri(x)
    for (wr, wi), s in zip(ws, strides):
        x = F.conv2d(x, _cblock(wr, wi), stride=s, padding=wr.shape[-1] // 2)
    return _from_ri(x)


def _adjoint(z, ws, strides):
    """`K^H z` with the synthesis weights: complex (B, M_L, h, w) -> (B, C, Sh, Sw)."""
    z = _to_ri(z)
    for (wr, wi), s in zip(reversed(ws), reversed(strides)):
        z = F.conv_transpose2d(z, _cblock(wr, wi, transpose=True), stride=s,
                               padding=wr.shape[-1] // 2, output_padding=s - 1)
    return _from_ri(z)


class CascadeLevel(nn.Module):
    """`A_l` and `B_l` of one level -- a `LISTALayer` without the prox.

    Only the deepest level is penalised, so a per-level prox would be dead
    weight.  The init, normalisation and projection are LISTALayer's own,
    borrowed rather than copied so the three nets cannot drift apart.
    """

    init_filters = LISTALayer.init_filters
    spectral_normalize = LISTALayer.spectral_normalize
    project_ = LISTALayer.project_

    def __init__(self, C, M, P=7, stride=2, is_complex=True, proj_mode="slice"):
        super().__init__()
        self.C, self.M, self.P, self.stride = int(C), int(M), int(P), int(stride)
        self.is_complex = bool(is_complex)
        self.proj_dims = proj_dims(proj_mode)
        self.eta = None                                  # read by project_
        self.analysis = Conv2d(C, M, P, stride=stride, bias=False, complex=is_complex)
        self.synthesis = ConvTranspose2d(M, C, P, stride=stride, bias=False,
                                         complex=is_complex)


class CascadeLPDSLayer(nn.Module):
    """One unrolled Condat-Vu step with `K = A_L ... A_1`.

    Signature-compatible with `LPDSLayer`: `(state, y_tilde, E, sigma, pi,
    cache) -> ((x, z), cache)`.
    """

    def __init__(self, C=1, M=16, L=3, widen=4, P=7, s=2, lam0=1e-3, tau0=5e-1,
                 theta0=0.0, degrees=0, is_complex=True, proj_mode="slice",
                 spectral_init=True):
        super().__init__()
        if not is_complex:
            raise ValueError("CascadeLPDSLayer runs its convs on complex maps; "
                             "is_complex=False is not implemented.")
        self.C, self.L = int(C), int(L)
        self.Mch = level_channels(C, M, self.L, widen)
        self.strides = tuple(level_strides(s, self.L)[1:])

        self.levels = nn.ModuleList()
        for l in range(1, self.L + 1):
            lev = CascadeLevel(self.Mch[l - 1], self.Mch[l], P=P,
                               stride=self.strides[l - 1], proj_mode=proj_mode)
            lev.init_filters()
            if spectral_init:
                lev.spectral_normalize(size=_init_size(l))
            self.levels.append(lev)

        self.prox = build_prox(self.Mch[-1], dual=True, tau0=lam0, degrees=degrees)
        self.tau = Polynomial(C, degrees=degrees, tau0=tau0)
        self.theta = Polynomial(C, degrees=degrees, tau0=theta0)

        # swapped for one shared torch.compile'd pair by the net
        self._analyse_fn, self._adjoint_fn = _analyse, _adjoint

    # -- K and K^H -----------------------------------------------------------
    def _weights(self, which):
        convs = [getattr(lev, which) for lev in self.levels]
        return [(c.conv_real.weight, c.conv_imag.weight) for c in convs]

    def analyse(self, x):
        return self._analyse_fn(x, self._weights("analysis"), self.strides)

    def adjoint(self, z):
        return self._adjoint_fn(z, self._weights("synthesis"), self.strides)

    # -- forward -------------------------------------------------------------
    def forward(self, state, y_tilde, E=None, sigma=None, pi=None, cache=None):
        if pi is not None:
            raise ValueError("CascadeLPDSLayer takes no FAS correction.")
        if cache is None:
            cache = {}
        if state is None:                                   # cold start
            z, cache = self.prox(self.analyse(y_tilde), sigma, cache)
            return (y_tilde, z), cache

        x, z = state
        tau = self.tau(sigma, ref=x)
        theta = self.theta(sigma, ref=x)
        x_new = x - tau * (gram(E, x) - y_tilde + self.adjoint(z))
        x_bar = x_new + theta * (x_new - x)
        z, cache = self.prox(z + self.analyse(x_bar), sigma, cache)
        return (x_new, z), cache

    # -- constraints ---------------------------------------------------------
    @torch.no_grad()
    def project_(self):
        """tau >= 0, theta in [0, 1].

        Filters and the prox are projected by their own `project_`, which the
        net's `project()` module walk reaches (as in MLLPDSLayer).
        """
        self.tau.project_(lo=0.0)
        self.theta.project_(lo=0.0, hi=1.0)

    # -- diagnostics ---------------------------------------------------------
    @torch.no_grad()
    def op_norm2(self, size=64, num_iter=100):
        """`||K||^2` by power iteration on `K^H K` (exact while B = A^H)."""
        S = 1
        for s in self.strides:
            S *= s
        size = max(size - size % S, S)
        w = self.levels[0].analysis.weight
        x0 = torch.rand(1, self.C, size, size, dtype=w.dtype, device=w.device)
        return abs(power_method(lambda x: self.adjoint(self.analyse(x)), x0,
                                num_iter=num_iter, verbose=False)[0])

    def step_bound(self, **kws):
        """Condat-Vu's largest primal step `1 / (1/2 + ||K||^2)`, ||E|| <= 1."""
        return 1.0 / (0.5 + self.op_norm2(**kws))


class CascadeLPDSNet(_MLIO):
    """`preprocess -> K Cascade-LPDS layers -> postprocess`.

    Same interface as `WaveletLPDSNet` / `MLLPDSNet`: `forward(y, E, sigma,
    state)` returns `(x_hat, (x, z))`.
    """

    def __init__(self, K=30, M=16, L=3, C=1, P=7, s=2, widen=4, lam0=1e-3,
                 tau0=5e-1, theta0=0.0, degrees=0, is_complex=True,
                 preproc="kspace", proj_mode="slice", spectral_init=True,
                 compile_operator=False):
        super().__init__()
        if preproc not in ("image", "kspace", "identity"):
            raise ValueError("preproc must be 'image', 'kspace' or 'identity'; "
                             "got %r" % (preproc,))
        self.K, self.M, self.L, self.C = int(K), int(M), int(L), int(C)
        self.P, self.s, self.widen = int(P), int(s), widen
        self.preproc = preproc
        self.pad_stride = self.s * 2 ** (self.L - 1)

        self.net = LPDSStack(self.K, lambda: CascadeLPDSLayer(
            C=C, M=M, L=L, widen=widen, P=P, s=s, lam0=lam0, tau0=tau0,
            theta0=theta0, degrees=degrees, is_complex=is_complex,
            proj_mode=proj_mode, spectral_init=spectral_init))
        if compile_operator:
            self.compile_operator()

    def compile_operator(self, **kws):
        """torch.compile K and K^H once, shared by every layer (see
        `WaveletLPDSNet.compile_operator`)."""
        fa, fh = torch.compile(_analyse, **kws), torch.compile(_adjoint, **kws)
        for lay in self.net.layers:
            lay._analyse_fn, lay._adjoint_fn = fa, fh
        return self

    def forward(self, y, E=None, sigma=None, state=None):
        if E is None:
            E = Identity()
        self._check_sigma(sigma)
        y_tilde, E, params, post = self._pre(y, E)
        (x, z), _ = self.net(state, y_tilde, E=E, sigma=sigma, cache={})
        x_hat = x if params is None else post(x, params)
        return x_hat, (x, z)

    def layer(self, k=-1):
        return self.net.layers[k]

    def step_bound(self, k=-1, **kws):
        return self.layer(k).step_bound(**kws)

    def extra_repr(self):
        return "K=%d, channels=%s, P=%d, preproc=%r" % (
            self.K, "/".join(map(str, self.layer(0).Mch[1:])), self.P,
            self.preproc)
