"""
Wavelet LPDS: LPDSNet whose analysis operator is a DT-CWT-initialised cascade.

    min_x  1/2 ||E x - y||^2  +  || lambda . K x ||_1,     K = Q T_3 T_2 T_1

`T_l` is a stride-2 complex conv, grouped by tree (hard: the four DT-CWT
lineages never mix), and `Q` is the FIXED unitary that forms the oriented
complex channels from the trees just before the clip.  Every band is carried
down to depth 3 (one-hot stride-2 kernels = pixel-unshuffle), so only the
deepest code is penalised and there is ONE dual.  The iteration is therefore
`models/lpds.py::LPDSLayer` with `A -> K` and `B -> K^H`; see
`models/wavelets.py` for the initialisation and its channel layout.

Channels 16 -> 64 -> 256 (M = 16): one 2D DT-CWT, LeGall 5/3 at level 1 and
`qshift_06` (end taps truncated to fit P = 7) at levels 2-3.

Init
----
* `B_l = A_l^H`, so layer 0..K-1 all start as the same classical Condat-Vu
  step.  Training unties them (the biorthogonal pair is a later option).
* ||K|| = 1 by power iteration, one scalar on level 1, so tau0 = 0.5 sits
  inside the Condat-Vu bound tau (1/2 + ||K||^2) <= 1 as in `lpdsnet`.  Every
  filter slice then has norm <= 1, so `project()` is a no-op at init.
* Clip thresholds are per channel at lam0 (`lpdsnet`'s init), except the four
  coarse LL^3 channels, which start at 0 -- the classical convention of not
  penalising the scaling coefficients -- and stay learnable (>= 0).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.base import set_weight
from models.components import Conv2d, ConvTranspose2d
from models.lista import gram
from models.lpds import LPDSStack
from models.ml_cdlnet import _MLIO
from models.prox import Polynomial, build_prox
from models.wavelets import NTREES, dtcwt_Q, dtcwt_weights
from operators.identity import Identity
from operators.projections import proj_dims, uball_project
from solvers.eigen import power_method


class WaveletLPDSLayer(nn.Module):
    """One unrolled Condat-Vu step with `K = Q T_3 T_2 T_1`.

    Signature-compatible with `LPDSLayer`: `(state, y_tilde, E, sigma, pi,
    cache) -> ((x, z), cache)`.
    """

    def __init__(self, P=7, lam0=1e-3, tau0=5e-1, theta0=0.0, degrees=0,
                 proj_mode="slice", spectral_init=True):
        super().__init__()
        weights, tags = dtcwt_weights(P, levels=3)
        self.proj_dims = proj_dims(proj_mode)

        self.analysis, self.synthesis = nn.ModuleList(), nn.ModuleList()
        for l, w in enumerate(weights):
            cout, cin_g = w.shape[:2]
            g = 1 if l == 0 else NTREES
            a = Conv2d(cin_g * g, cout, P, stride=2, groups=g)
            b = ConvTranspose2d(cout, cin_g * g, P, stride=2, groups=g)
            set_weight(a, w)
            set_weight(b, w)                 # real at init, so A^H = A^T
            self.analysis.append(a)
            self.synthesis.append(b)

        Q = dtcwt_Q(tags)                                   # (64, 4, 4)
        self.register_buffer("Q", Q)
        self.register_buffer("QH", Q.conj().transpose(1, 2).resolve_conj().contiguous())
        self.Cg = len(tags)
        self.M = NTREES * self.Cg

        self.prox = build_prox(self.M, dual=True, tau0=lam0, degrees=degrees)
        with torch.no_grad():                               # LL^3: no penalty
            self.prox.prox.tau.weight[:, self.ll_channels] = 0.0

        self.tau = Polynomial(1, degrees=degrees, tau0=tau0)
        self.theta = Polynomial(1, degrees=degrees, tau0=theta0)

        if spectral_init:
            self.normalize()

    @property
    def ll_channels(self):
        return [t * self.Cg for t in range(NTREES)]

    # -- K and K^H -----------------------------------------------------------
    def _mix(self, z, Q):
        B, _, H, W = z.shape
        z = z.reshape(B, NTREES, self.Cg, H, W)
        return torch.einsum("cij,bjchw->bichw", Q, z).reshape(B, self.M, H, W)

    def analyse(self, x):
        for a in self.analysis:
            x = a(x)
        return self._mix(x, self.Q)

    def adjoint(self, z):
        z = self._mix(z, self.QH)
        for b in reversed(self.synthesis):
            z = b(z)
        return z

    # -- forward -------------------------------------------------------------
    def forward(self, state, y_tilde, E=None, sigma=None, pi=None, cache=None):
        if pi is not None:
            raise ValueError("WaveletLPDSLayer takes no FAS correction.")
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
        """Filters onto the unit ball (per slice), tau >= 0, theta in [0, 1].
        The thresholds are clamped >= 0 by the prox's own `project_`."""
        for conv in list(self.analysis) + list(self.synthesis):
            set_weight(conv, uball_project(conv.weight, dim=self.proj_dims))
        self.tau.project_(lo=0.0)
        self.theta.project_(lo=0.0, hi=1.0)

    # -- normalisation -------------------------------------------------------
    @torch.no_grad()
    def op_norm2(self, size=64, num_iter=100):
        """`||K||^2` by power iteration on `K^H K` (exact while B = A^H)."""
        w = self.analysis[0].weight
        x0 = torch.rand(1, 1, size, size, dtype=w.dtype, device=w.device)
        return abs(power_method(lambda x: self.adjoint(self.analyse(x)), x0,
                                num_iter=num_iter, verbose=False)[0])

    def step_bound(self, **kws):
        """Condat-Vu's largest primal step `1 / (1/2 + ||K||^2)`, ||E|| <= 1."""
        return 1.0 / (0.5 + self.op_norm2(**kws))

    @torch.no_grad()
    def normalize(self, **kws):
        """Scale level 1 (A and B) so that `||K|| = 1`."""
        g = float(self.op_norm2(**kws)) ** 0.5
        for conv in (self.analysis[0], self.synthesis[0]):
            set_weight(conv, conv.weight / g)


class WaveletLPDSNet(_MLIO):
    """`preprocess -> K Wavelet-LPDS layers -> postprocess`.

    Same interface as `MLLPDSNet`: `forward(y, E, sigma, state)` returns
    `(x_hat, (x, z))`.  Only the single DT-CWT baseline (M = 16, L = 3, s = 2,
    one complex image channel) is implemented; the arguments are kept so the
    config states the shape it trains.
    """

    def __init__(self, K=30, M=16, L=3, C=1, P=7, s=2, lam0=1e-3, tau0=5e-1,
                 theta0=0.0, degrees=0, is_complex=True, preproc="kspace",
                 proj_mode="slice", spectral_init=True):
        super().__init__()
        if (M, L, C, s, is_complex) != (16, 3, 1, 2, True):
            raise ValueError(
                "WaveletLPDSNet implements the single DT-CWT only: M=16, L=3, "
                "C=1, s=2, is_complex=True; got M=%r L=%r C=%r s=%r "
                "is_complex=%r" % (M, L, C, s, is_complex))
        if preproc not in ("image", "kspace", "identity"):
            raise ValueError("preproc must be 'image', 'kspace' or 'identity'; "
                             "got %r" % (preproc,))
        self.K, self.M, self.L, self.C, self.P, self.s = int(K), M, L, C, P, s
        self.preproc = preproc
        self.pad_stride = s * 2 ** (L - 1)

        self.net = LPDSStack(self.K, lambda: WaveletLPDSLayer(
            P=P, lam0=lam0, tau0=tau0, theta0=theta0, degrees=degrees,
            proj_mode=proj_mode, spectral_init=spectral_init))

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
        return "K=%d, channels=16/64/256, P=%d, preproc=%r" % (
            self.K, self.P, self.preproc)
