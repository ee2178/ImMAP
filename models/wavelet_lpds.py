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

Where the redundancy goes (`dual_step`)
---------------------------------------
The frame is redundant -- ||K||^2 = 4 for four orthonormal trees -- and
Condat-Vu needs that factor somewhere.  `"absorbed"` (the original) rescales
level 1 so ||K|| = 1, which leaves level 1 at 0.2-0.5 and the deep LL-split
rows at 1: under the per-slice unit-ball `project_`, gain can then only enter
through level 1, and Haar's deep rows sit on the ball from the first step.
`"learned"` keeps every filter at its wavelet normalisation and carries the
factor in an explicit, learnable dual step (a Polynomial, like tau):

    z <- clip_{c lam}(z + sigma_d K xb),   sigma_d = c^2,   c = 1/||K||

which is the absorbed iteration exactly, reparametrised (z scaled by c).  The
bound is tau (1/2 + sigma_d ||K||^2) <= 1.  With learned thresholds the two
are equally expressive; they differ in which parameters absorb gain under the
projection.

Several families
----------------
`family=["dtcwt", "haar"]` stacks the two transforms as a union of frames,
K = [K_dtcwt; K_haar] with 4 trees each (32/128/512).  Q is block-diagonal, so
each family forms its own bands; one dual and one clip cover all 512
channels.  At init every family is normalised to ||K_f|| = 1
(`normalize_families`) before the global ||K|| = 1, so neither dominates by
its raw filter scale.

Init
----
* `band_norm="equal"` rescales each band where it is born so that every deep
  channel has the same noise gain `||K^H e_m||` (`equalize_bands`), before
  the ||K|| = 1 normalisation.  The DT-CWT's level-1 LeGall pair has unit DC
  gain and a small highpass while levels 2-3 are q-shift (near-orthonormal),
  which otherwise leaves the noise std differing ~2.3x between bands under one
  shared lam0.  Haar is orthonormal and already equal.  Default "none" keeps
  wlpds16 as trained.
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
from models.wavelets import FAMILIES, NTREES
from operators.identity import Identity
from operators.projections import proj_dims, uball_project
from solvers.eigen import power_method

CARRY_MODES = ("unshuffle", "conv")


def _as_families(family):
    """`family` as a tuple of FAMILIES names: one name, or a list of them."""
    fams = (family,) if isinstance(family, str) else tuple(family)
    bad = [f for f in fams if f not in FAMILIES]
    if not fams or bad or len(set(fams)) != len(fams):
        raise ValueError("family must be one of %r, or a list of distinct ones; "
                         "got %r" % (tuple(FAMILIES), family))
    return fams
BAND_NORMS = ("none", "equal")
DUAL_STEPS = ("absorbed", "learned")


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
    """(n, T, T) complex -> (n, T, 2, T, 2) real, [out part][in part] blocks."""
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

    `ws[l] = (wr, wi)` are level l's analysis weights, `Qr = _real_Q(Q)`.  The
    number of trees T (4 per family) is read off `Qr`, a static shape.
    """
    T = Qr.shape[1]
    x = _to_ri(x)
    for l, (wr, wi) in enumerate(ws):
        w, p = _cblock(wr, wi), wr.shape[-1] // 2
        if l == 0 or carry == "conv":
            x = F.conv2d(x, w, stride=2, padding=p, groups=1 if l == 0 else T)
            continue
        B, _, H, W = x.shape
        x = x.view(B, T, -1, 2, H, W)
        ll = F.conv2d(x[:, :, 0].reshape(B, 2 * T, H, W), w,
                      stride=2, padding=p, groups=T)
        x = torch.cat([ll.view(B, T, 4, 2, H // 2, W // 2),
                       _unshuffle(x[:, :, 1:])], 2)
        x = x.view(B, -1, H // 2, W // 2)
    B, _, h, w = x.shape
    x = x.view(B, T, Qr.shape[0], 2, h, w)
    z = torch.einsum("cipja,njcahw->pnichw", Qr, x)
    return torch.complex(z[0], z[1]).reshape(B, -1, h, w)


def _adjoint(z, ws, QHr, carry):
    """`K^H z`: complex (B, M, h, w) -> complex (B, 1, 8h, 8w).

    `ws[l] = (wr, wi)` are level l's synthesis weights, `QHr = _real_Q(Q^H)`.
    """
    T = QHr.shape[1]
    B, _, h, w = z.shape
    z = torch.stack([z.real, z.imag]).view(2, B, T, QHr.shape[0], h, w)
    z = torch.einsum("cjpia,anichw->njcphw", QHr, z).reshape(B, -1, h, w)
    for l in reversed(range(len(ws))):
        wr, wi = ws[l]
        wt, p = _cblock(wr, wi, transpose=True), wr.shape[-1] // 2
        g = 1 if l == 0 else T
        if l == 0 or carry == "conv":
            z = F.conv_transpose2d(z, wt, stride=2, padding=p, output_padding=1,
                                   groups=g)
            continue
        B, _, h, w = z.shape
        z = z.view(B, T, -1, 2, h, w)
        ll = F.conv_transpose2d(z[:, :, :4].reshape(B, 8 * T, h, w), wt,
                                stride=2, padding=p, output_padding=1, groups=g)
        z = torch.cat([ll.view(B, T, 1, 2, 2 * h, 2 * w),
                       _shuffle(z[:, :, 4:])], 2)
        z = z.view(B, -1, 2 * h, 2 * w)
    return _from_ri(z)


class WaveletLPDSLayer(nn.Module):
    """One unrolled Condat-Vu step with `K = Q T_3 T_2 T_1`.

    Signature-compatible with `LPDSLayer`: `(state, y_tilde, E, sigma, pi,
    cache) -> ((x, z), cache)`.
    """

    def __init__(self, P=7, lam0=1e-3, tau0=5e-1, theta0=0.0, degrees=0,
                 proj_mode="slice", spectral_init=True, carry="unshuffle",
                 family="dtcwt", band_norm="none", dual_step="absorbed"):
        super().__init__()
        if dual_step not in DUAL_STEPS:
            raise ValueError("dual_step must be one of %r; got %r"
                             % (DUAL_STEPS, dual_step))
        if band_norm not in BAND_NORMS:
            raise ValueError("band_norm must be one of %r; got %r"
                             % (BAND_NORMS, band_norm))
        if carry not in CARRY_MODES:
            raise ValueError("carry must be one of %r; got %r" % (CARRY_MODES, carry))
        self.families = _as_families(family)
        self.carry, self.band_norm, self.dual_step = carry, band_norm, dual_step
        self.family = "+".join(self.families)
        self.T = NTREES * len(self.families)           # trees, 4 per family

        # Every family builds the same carried layout (same tags), so several
        # stack family-major: trees 0-3 are the first family, 4-7 the second.
        # Level 1 concatenates rows; deeper levels concatenate conv groups.
        built = [FAMILIES[f][0](P, levels=3) for f in self.families]
        tags = built[0][1]
        if any(t != tags for _, t in built):
            raise ValueError("families disagree on the band layout")
        self.proj_dims = proj_dims(proj_mode)

        self.analysis, self.synthesis = nn.ModuleList(), nn.ModuleList()
        for l in range(len(built[0][0])):
            parts = []
            for weights_family, (weights, _) in zip(self.families, built):
                w = weights[l]
                if l > 0 and carry == "unshuffle":
                    # keep rows 0-3 (the LL split) of each tree, reading LL only;
                    # the rest must be the fixed one-hot carries it replaces
                    v = w.view(NTREES, -1, *w.shape[1:])
                    ref = FAMILIES["haar"][0](P, levels=3)[0][l]
                    ref = ref.view(NTREES, -1, *ref.shape[1:])
                    if v[:, :4, 1:].abs().sum() > 0 or not torch.equal(v[:, 4:], ref[:, 4:]):
                        raise ValueError(
                            "family %r learns its carries, which carry='unshuffle' "
                            "would drop; use carry='conv'" % (weights_family,))
                    w = v[:, :4, :1].reshape(4 * NTREES, 1, P, P)
                parts.append(w)
            w = torch.cat(parts, 0)
            cout, cin_g = w.shape[:2]
            g = 1 if l == 0 else self.T
            a = Conv2d(cin_g * g, cout, P, stride=2, groups=g)
            b = ConvTranspose2d(cout, cin_g * g, P, stride=2, groups=g)
            set_weight(a, w)
            set_weight(b, w)                 # real at init, so A^H = A^T
            self.analysis.append(a)
            self.synthesis.append(b)

        # block-diagonal: each family mixes only its own four trees
        Q = torch.stack([torch.block_diag(*qs) for qs in zip(
            *[FAMILIES[f][1](tags) for f in self.families])])   # (64, T, T)
        self.register_buffer("Q", Q)
        self.register_buffer("QH", Q.conj().transpose(1, 2).resolve_conj().contiguous())
        # real forms for `_analyse` / `_adjoint`; derived, so kept out of the
        # state dict (old checkpoints load unchanged)
        self.register_buffer("Qr", _real_Q(self.Q), persistent=False)
        self.register_buffer("QHr", _real_Q(self.QH), persistent=False)
        # swapped for one shared torch.compile'd pair by the net
        self._analyse_fn, self._adjoint_fn = _analyse, _adjoint
        self.Cg = len(tags)
        self.M = self.T * self.Cg
        self.tags = tags

        self.prox = build_prox(self.M, dual=True, tau0=lam0, degrees=degrees)
        with torch.no_grad():                               # LL^3: no penalty
            self.prox.prox.tau.weight[:, self.ll_channels] = 0.0

        self.tau = Polynomial(1, degrees=degrees, tau0=tau0)
        self.theta = Polynomial(1, degrees=degrees, tau0=theta0)

        if spectral_init and len(self.families) > 1:
            self.normalize_families()
        if band_norm == "equal":
            self.equalize_bands()

        # Where the frame's redundancy goes (see the module docstring).
        #   absorbed  level 1 is rescaled so ||K|| = 1 (the original init)
        #   learned   the filters stay at their wavelet normalisation and the
        #             Condat-Vu dual step sigma_d = 1/||K||^2 carries it, with
        #             the thresholds scaled by 1/||K|| -- the same algorithm at
        #             init, exactly (see `test_dual_step`)
        self.sigma_d = None
        if dual_step == "learned":
            n2 = float(self.op_norm2()) if spectral_init else 1.0
            self.sigma_d = Polynomial(1, degrees=degrees, tau0=1.0 / n2)
            with torch.no_grad():
                self.prox.prox.tau.weight.mul_(n2 ** -0.5)
        elif spectral_init:
            self.normalize()

    @property
    def ll_channels(self):
        return [t * self.Cg for t in range(self.T)]

    def _family_of(self, m):
        """Index into `self.families` of deep channel m."""
        return (m // self.Cg) // NTREES

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
            z, cache = self.prox(self._dual_scale(self.analyse(y_tilde), sigma, y_tilde),
                                 sigma, cache)
            return (y_tilde, z), cache

        x, z = state
        tau = self.tau(sigma, ref=x)
        theta = self.theta(sigma, ref=x)
        x_new = x - tau * (gram(E, x) - y_tilde + self.adjoint(z))
        x_bar = x_new + theta * (x_new - x)
        z, cache = self.prox(z + self._dual_scale(self.analyse(x_bar), sigma, x),
                             sigma, cache)
        return (x_new, z), cache

    def _dual_scale(self, Kx, sigma, ref):
        """`sigma_d K x` under dual_step="learned", `K x` otherwise."""
        if self.sigma_d is None:
            return Kx
        return self.sigma_d(sigma, ref=ref) * Kx

    # -- constraints ---------------------------------------------------------
    @torch.no_grad()
    def project_(self):
        """Filters onto the unit ball (per slice), tau >= 0, theta in [0, 1].
        The thresholds are clamped >= 0 by the prox's own `project_`."""
        for conv in list(self.analysis) + list(self.synthesis):
            set_weight(conv, uball_project(conv.weight, dim=self.proj_dims))
        self.tau.project_(lo=0.0)
        self.theta.project_(lo=0.0, hi=1.0)
        if self.sigma_d is not None:
            self.sigma_d.project_(lo=0.0)

    # -- normalisation -------------------------------------------------------
    @torch.no_grad()
    def op_norm2(self, size=64, num_iter=100, family=None):
        """`||K||^2` by power iteration on `K^H K` (exact while B = A^H).

        `family` (an index into `self.families`) restricts to that family's
        channels: `||K_f||^2 = ||K^H P_f K||`, P_f the channel selector.
        """
        w = self.analysis[0].weight
        x0 = torch.rand(1, 1, size, size, dtype=w.dtype, device=w.device)
        if family is None:
            op = lambda x: self.adjoint(self.analyse(x))      # noqa: E731
        else:
            keep = torch.zeros(1, self.M, 1, 1, dtype=w.dtype, device=w.device)
            keep[:, [m for m in range(self.M) if self._family_of(m) == family]] = 1
            op = lambda x: self.adjoint(keep * self.analyse(x))   # noqa: E731
        return abs(power_method(op, x0, num_iter=num_iter, verbose=False)[0])

    def step_bound(self, **kws):
        """Condat-Vu's largest primal step `1 / (1/2 + sigma_d ||K||^2)`,
        ||E|| <= 1 (sigma_d = 1 under dual_step="absorbed"; its sigma-free
        coefficient otherwise)."""
        sd = 1.0 if self.sigma_d is None else float(self.sigma_d.weight[0, 0])
        return 1.0 / (0.5 + sd * self.op_norm2(**kws))

    @torch.no_grad()
    def band_norms(self, size=16):
        """`||K^H e_m||` per deep channel: the std, in channel m, of unit white
        noise on the image (exact while B = A^H).  One delta per channel,
        adjointed; `size` is the dual grid, and 16 (128 px) holds the widest
        effective filter (~43 px at P = 7) with room to spare."""
        w = self.analysis[0].weight
        out = []
        for lo in range(0, self.M, 32):
            m = torch.arange(lo, min(lo + 32, self.M), device=w.device)
            z = torch.zeros(len(m), self.M, size, size, dtype=w.dtype, device=w.device)
            z[torch.arange(len(m)), m, size // 2, size // 2] = 1
            out.append(self.adjoint(z).flatten(1).norm(dim=1))
        return torch.cat(out)

    @torch.no_grad()
    def equalize_bands(self):
        """Give every deep channel the same noise gain.

        A band is scaled on the level it is BORN on -- its LL-split row in
        T_l, in every tree -- and the fixed carries and the unitary Q pass the
        scale through unchanged.  The LL rows of levels 1..L-1 are not
        touched: they feed every deeper band, and those are scaled on their
        own rows.  The four trees of a band share one norm (Q mixes within a
        band), so scaling them together keeps Q's action.  `normalize` then
        restores ||K|| = 1.  A no-op on an orthonormal family (Haar).

        Equalised DOWN, to the weakest band.  The common target is otherwise
        free -- `normalize` only rescales level 1, so the target just moves
        scale between the level-1 LL row and the deep birth rows -- and going
        down only ever shrinks the deep rows, which keeps every filter slice
        inside the unit ball `project_` enforces.  Equalising up to the mean
        put DT-CWT's level-3 slices at ~1.9, and the first `project_` would
        have clipped them and undone half of it.
        """
        nrm = self.band_norms()
        per_tag = {}
        for m in range(self.M):
            key = (self._family_of(m),) + tuple(self.tags[m % self.Cg])
            per_tag.setdefault(key, []).append(float(nrm[m]))
        target = min(sum(v) / len(v) for v in per_tag.values())
        for (fam, level, band), v in per_tag.items():
            f = target * len(v) / sum(v)
            trees = range(fam * NTREES, (fam + 1) * NTREES)
            for conv in (self.analysis[level - 1], self.synthesis[level - 1]):
                w = conv.weight.clone()
                per_tree = w.shape[0] // self.T
                w[[t * per_tree + band for t in trees]] *= f
                set_weight(conv, w)

    @torch.no_grad()
    def normalize_families(self, **kws):
        """Give every family the same `||K_f||` (on its level-1 rows), so
        that no family outweighs another just because of its filters' raw
        scale -- DT-CWT's unit-DC-gain LeGall and orthonormal Haar differ by
        ~2x there.

        Equalised DOWN, to the weakest family, like `equalize_bands`: only
        shrinking keeps every slice inside the unit ball.  Under
        dual_step="absorbed" the global `normalize` rescales level 1 after
        this anyway, so the target makes no difference there."""
        rows = self.analysis[0].weight.shape[0] // len(self.families)
        g = [float(self.op_norm2(family=f, **kws)) ** 0.5
             for f in range(len(self.families))]
        for f, gf in enumerate(g):
            for conv in (self.analysis[0], self.synthesis[0]):
                w = conv.weight.clone()
                w[f * rows:(f + 1) * rows] *= min(g) / gf
                set_weight(conv, w)

    @torch.no_grad()
    def normalize(self, **kws):
        """Scale level 1 (A and B) so that `||K|| = 1`."""
        g = float(self.op_norm2(**kws)) ** 0.5
        for conv in (self.analysis[0], self.synthesis[0]):
            set_weight(conv, conv.weight / g)


class WaveletLPDSNet(_MLIO):
    """`preprocess -> K Wavelet-LPDS layers -> postprocess`.

    Same interface as `MLLPDSNet`: `forward(y, E, sigma, state)` returns
    `(x_hat, (x, z))`.  `family` picks the filters ("dtcwt", "haar"; see
    `models/wavelets.py::FAMILIES`) in one shared 4-tree layout, or a LIST of
    them for a union of frames: ["dtcwt", "haar"] is 8 trees, 32/128/512
    channels, one dual over both (M = 16 per family).  Only that layout
    (L = 3, s = 2,
    one complex image channel) is implemented; the arguments are kept so the
    config states the shape it trains.
    """

    def __init__(self, K=30, M=16, L=3, C=1, P=7, s=2, lam0=1e-3, tau0=5e-1,
                 theta0=0.0, degrees=0, is_complex=True, preproc="kspace",
                 proj_mode="slice", spectral_init=True, carry="unshuffle",
                 family="dtcwt", band_norm="none", dual_step="absorbed",
                 compile_operator=False):
        super().__init__()
        n_fam = len(_as_families(family))
        if (M, L, C, s, is_complex) != (16 * n_fam, 3, 1, 2, True):
            raise ValueError(
                "WaveletLPDSNet implements the 4-tree layout only: M=16 per "
                "family (%d here), L=3, C=1, s=2, is_complex=True; got M=%r "
                "L=%r C=%r s=%r is_complex=%r"
                % (16 * n_fam, M, L, C, s, is_complex))
        if preproc not in ("image", "kspace", "identity"):
            raise ValueError("preproc must be 'image', 'kspace' or 'identity'; "
                             "got %r" % (preproc,))
        self.K, self.M, self.L, self.C, self.P, self.s = int(K), M, L, C, P, s
        self.preproc = preproc
        self.pad_stride = s * 2 ** (L - 1)

        self.net = LPDSStack(self.K, lambda: WaveletLPDSLayer(
            P=P, lam0=lam0, tau0=tau0, theta0=theta0, degrees=degrees,
            proj_mode=proj_mode, spectral_init=spectral_init, carry=carry,
            family=family, band_norm=band_norm, dual_step=dual_step))
        self.carry, self.band_norm, self.dual_step = carry, band_norm, dual_step
        self.family = "+".join(_as_families(family))
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
        return ("K=%d, family=%r, band_norm=%r, dual_step=%r, channels=%d/%d/%d, "
                "P=%d, preproc=%r, carry=%r" % (
                    self.K, self.family, self.band_norm, self.dual_step, self.M,
                    4 * self.M, 16 * self.M, self.P, self.preproc, self.carry))
