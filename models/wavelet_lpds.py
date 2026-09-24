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

Carries (`carry=`)
------------------
* "unshuffle" (default): at levels 2-3 only the LL split is a learnable conv
  (1 -> 4 per tree); every other channel is a FIXED pixel-unshuffle (its
  adjoint a pixel-shuffle).  The operator at init is identical to "conv", at
  ~1/15 of the FLOPs, but the carries cannot learn and the LL split reads LL
  only.
* "conv": the dense grouped conv of `dtcwt_weights` -- carries are learnable
  7x7 filters that may mix every channel of a tree (4 -> 16, 16 -> 64).  Needed
  to load checkpoints trained before the unshuffle path existed.

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
import torch.nn.functional as F

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

CARRY_MODES = ("unshuffle", "conv")


# -- K and K^H on an interleaved real layout -----------------------------------
# Inside the cascade a complex map (B, C, H, W) is carried as a real
# (B, 2C, H, W) with channel 2c = Re, 2c + 1 = Im.  Each level is then ONE real
# conv with the block weight [[wr, -wi], [wi, wr]] instead of the Gauss trick's
# three convs plus the Re/Im split and recombine; conv groups stay contiguous in
# this layout.  Complex <-> real happens once per operator: at the image end
# (one channel) and fused into the Q mix at depth 3.  Both are pure functions of
# the weights so ONE `torch.compile` serves all K layers (see
# `WaveletLPDSNet(compile_operator=True)`).

# (Explicit .real/.imag + torch.complex rather than view_as_real/_complex: the
# latter's gradients come back with a non-unit last stride under
# torch.compile's backward and fail.  Same one copy either way.)

def _to_ri(x):
    """complex (B, C, H, W) -> real (B, 2C, H, W), channels interleaved."""
    B, C, H, W = x.shape
    if not x.is_complex():
        return torch.stack([x, torch.zeros_like(x)], 2).view(B, 2 * C, H, W)
    return torch.stack([x.real, x.imag], 2).view(B, 2 * C, H, W)


def _from_ri(x):
    """Inverse of `_to_ri`."""
    B, C2, H, W = x.shape
    x = x.view(B, C2 // 2, 2, H, W)
    return torch.complex(x[:, :, 0], x[:, :, 1])


def _cblock(wr, wi, transpose=False):
    """Real (2O, 2I, P, P) weight of the complex conv `wr + i wi` (O, I, P, P).

    Blocks are [out part][in part] = [[wr, -wi], [wi, wr]]; a conv_transpose
    weight is indexed (in, out), so there they are read transposed.
    """
    if transpose:
        w = torch.stack([torch.stack([wr, wi], 2), torch.stack([-wi, wr], 2)], 1)
    else:
        w = torch.stack([torch.stack([wr, -wi], 2), torch.stack([wi, wr], 2)], 1)
    return w.reshape(2 * wr.shape[0], 2 * wr.shape[1], *wr.shape[2:])


def _real_Q(Q):
    """(n, 4, 4) complex -> (n, 4, 2, 4, 2) real, [out part][in part] blocks."""
    r, m = Q.real, Q.imag
    return torch.stack([torch.stack([r, -m], -1), torch.stack([m, r], -1)], 2)


def _unshuffle(x):
    """(B, T, n, 2, H, W) -> (B, T, 4n, 2, H/2, W/2), channel 4c + 2dy + dx.

    The stride-2 one-hot kernels of `dtcwt_weights` in their order (and
    `F.pixel_unshuffle`'s), with the Re/Im axis kept innermost.
    """
    B, T, n, _, H, W = x.shape
    x = x.reshape(B, T, n, 2, H // 2, 2, W // 2, 2)
    return x.permute(0, 1, 2, 5, 7, 3, 4, 6).reshape(B, T, 4 * n, 2, H // 2, W // 2)


def _shuffle(x):
    """Inverse (and adjoint) of `_unshuffle`."""
    B, T, c, _, h, w = x.shape
    x = x.reshape(B, T, c // 4, 2, 2, 2, h, w)
    return x.permute(0, 1, 2, 5, 6, 3, 7, 4).reshape(B, T, c // 4, 2, 2 * h, 2 * w)


def _analyse(x, ws, Qr, carry):
    """`K x`: complex (B, 1, H, W) -> complex (B, M, H/8, W/8).

    `ws[l] = (wr, wi)` are level l's analysis weights, `Qr = _real_Q(Q)`.
    """
    x = _to_ri(x)
    for l, (wr, wi) in enumerate(ws):
        w, p = _cblock(wr, wi), wr.shape[-1] // 2
        if l == 0 or carry == "conv":
            x = F.conv2d(x, w, stride=2, padding=p, groups=1 if l == 0 else NTREES)
            continue
        B, _, H, W = x.shape
        x = x.view(B, NTREES, -1, 2, H, W)
        ll = F.conv2d(x[:, :, 0].reshape(B, 2 * NTREES, H, W), w,
                      stride=2, padding=p, groups=NTREES)
        x = torch.cat([ll.view(B, NTREES, 4, 2, H // 2, W // 2),
                       _unshuffle(x[:, :, 1:])], 2)
        x = x.view(B, -1, H // 2, W // 2)
    B, _, h, w = x.shape
    x = x.view(B, NTREES, Qr.shape[0], 2, h, w)
    z = torch.einsum("cipja,njcahw->pnichw", Qr, x)
    return torch.complex(z[0], z[1]).reshape(B, -1, h, w)


def _adjoint(z, ws, QHr, carry):
    """`K^H z`: complex (B, M, h, w) -> complex (B, 1, 8h, 8w).

    `ws[l] = (wr, wi)` are level l's synthesis weights, `QHr = _real_Q(Q^H)`.
    """
    B, _, h, w = z.shape
    z = torch.stack([z.real, z.imag]).view(2, B, NTREES, QHr.shape[0], h, w)
    z = torch.einsum("cjpia,anichw->njcphw", QHr, z).reshape(B, -1, h, w)
    for l in reversed(range(len(ws))):
        wr, wi = ws[l]
        wt, p = _cblock(wr, wi, transpose=True), wr.shape[-1] // 2
        g = 1 if l == 0 else NTREES
        if l == 0 or carry == "conv":
            z = F.conv_transpose2d(z, wt, stride=2, padding=p, output_padding=1,
                                   groups=g)
            continue
        B, _, h, w = z.shape
        z = z.view(B, NTREES, -1, 2, h, w)
        ll = F.conv_transpose2d(z[:, :, :4].reshape(B, 8 * NTREES, h, w), wt,
                                stride=2, padding=p, output_padding=1, groups=g)
        z = torch.cat([ll.view(B, NTREES, 1, 2, 2 * h, 2 * w),
                       _shuffle(z[:, :, 4:])], 2)
        z = z.view(B, -1, 2 * h, 2 * w)
    return _from_ri(z)


class WaveletLPDSLayer(nn.Module):
    """One unrolled Condat-Vu step with `K = Q T_3 T_2 T_1`.

    Signature-compatible with `LPDSLayer`: `(state, y_tilde, E, sigma, pi,
    cache) -> ((x, z), cache)`.
    """

    def __init__(self, P=7, lam0=1e-3, tau0=5e-1, theta0=0.0, degrees=0,
                 proj_mode="slice", spectral_init=True, carry="unshuffle"):
        super().__init__()
        if carry not in CARRY_MODES:
            raise ValueError("carry must be one of %r; got %r" % (CARRY_MODES, carry))
        self.carry = carry
        weights, tags = dtcwt_weights(P, levels=3)
        self.proj_dims = proj_dims(proj_mode)

        self.analysis, self.synthesis = nn.ModuleList(), nn.ModuleList()
        for l, w in enumerate(weights):
            if l > 0 and carry == "unshuffle":
                # keep rows 0-3 (the LL split) of each tree, reading LL only
                w = w.view(NTREES, -1, *w.shape[1:])[:, :4, :1].reshape(
                    4 * NTREES, 1, P, P)
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
        # real forms for `_analyse` / `_adjoint`; derived, so kept out of the
        # state dict (old checkpoints load unchanged)
        self.register_buffer("Qr", _real_Q(self.Q), persistent=False)
        self.register_buffer("QHr", _real_Q(self.QH), persistent=False)
        # swapped for one shared torch.compile'd pair by the net
        self._analyse_fn, self._adjoint_fn = _analyse, _adjoint
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
    # The Conv2d / ConvTranspose2d modules only hold the (real, imag) weights;
    # the convs themselves run in `_analyse` / `_adjoint`.
    @staticmethod
    def _weights(convs):
        return [(c.conv_real.weight, c.conv_imag.weight) for c in convs]

    def analyse(self, x):
        return self._analyse_fn(x, self._weights(self.analysis), self.Qr, self.carry)

    def adjoint(self, z):
        return self._adjoint_fn(z, self._weights(self.synthesis), self.QHr, self.carry)

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
                 proj_mode="slice", spectral_init=True, carry="unshuffle",
                 compile_operator=False):
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
            proj_mode=proj_mode, spectral_init=spectral_init, carry=carry))
        self.carry = carry
        if compile_operator:
            self.compile_operator()

    def compile_operator(self, **kws):
        """torch.compile K and K^H once, shared by every layer.

        The weights are arguments, so the K layers hit one graph per shape
        instead of K recompiles (compiling each layer's bound method would
        blow dynamo's recompile limit and silently fall back to eager).
        Fuses the layout shuffles, block-weight builds and Q mixes around the
        cuDNN convs.  Compiles lazily on the first call, on whatever device.
        """
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
        return "K=%d, channels=16/64/256, P=%d, preproc=%r, carry=%r" % (
            self.K, self.P, self.preproc, self.carry)
