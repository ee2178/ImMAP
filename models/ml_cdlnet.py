# -*- coding: utf-8 -*-
"""
Multi*level* convolutional dictionary learning networks (ML-CSC), and their
variable-split cousin.

    MLCDLNet       unrolled ML-ISTA          (Sulam, Aberdam, Beck, Elad 2018)
    MLSplitCDLNet  unrolled linearized ADMM  over the same model

How this differs from `models/multigrid.py`
-------------------------------------------
`MGCDLNet` solves ONE problem on one grid and uses coarse grids as a *solver
device*: the unknown is `z` on the fine latent grid, `z_c = R z` is a
restriction of it, the transfer operators are fixed (or lightly learned)
interpolation kernels, and the coarse level only contributes a correction
`alpha P (w_c - z_c)`.  Coarse capacity buys convergence rate; nothing coarse
ever touches the read-out.

Here the hierarchy is the MODEL:

    x = D_1 g_1,   g_1 = D_2 g_2,   ...,   g_{L-1} = D_L g_L

There is one unknown, `g_L`, and the intermediate codes are priors rather than
auxiliary variables.  The transfer operators ARE the learned strided
dictionaries, so there is no GridTransfer, no galerkin, no FAS pi.  And because
`x = D_1 D_2 ... D_L g_L`, every level sits on the reconstruction path -- which
is what justifies widening the channel count with depth (deeper codes describe
more complex "molecules", not coarser image content).

Cost: level l runs at N/4^(l-1) pixels with M_{l-1} M_l channel pairs, so
successive levels are in the ratio widen^2 / 4.  widen=2 gives exactly constant
FLOPs per level -- the UNet balance -- and `E` is applied at level 1 ONLY,
unlike a V-cycle whose Galerkin coarse Grams run at full resolution.

The two algorithms
------------------
ML-ISTA (`MLCDLNet`) carries a single tensor, `g_L`, and rebuilds every
intermediate code from it each outer iteration:

    ghat_L = g_L ;  ghat_l = B_{l+1} ghat_{l+1}         (synthesis sweep, down)
    g_l = prox_l( ghat_l - A_l ( G_l(B_l ghat_l) - g_{l-1} ) )  (analysis, up)

with `g_0 := y~` and `G_1 = E^H E`, `G_l = Id` for l >= 2.  The analysis-sweep
line is EXACTLY `LISTALayer.forward`, which is why this file adds no new
low-level module for it: level 1 is called with the encoding operator, the
deeper levels with `E = None`.

The catch is the line `ghat_l = B_{l+1} ghat_{l+1}`: level l's state is
OVERWRITTEN by the coarse synthesis every outer iteration, so anything level l
learned that the coarse levels cannot express is discarded.  That single line is
both the deepest-code bottleneck and the reason there is no skip connection.

`MLSplitCDLNet` un-eliminates the intermediate codes.  Keeping them as genuine
variables coupled by `g_l = D_{l+1} g_{l+1}` with duals `u_l` gives, per level,

    e_l = A_l ( rho_{l-1} . ( G_l(B_l g_l) - g_{l-1} - u_{l-1} ) )  encoder arm
    d_l = rho_l . ( g_l - B_{l+1} g_{l+1} + u_l )                   decoder / SKIP
    g_l = prox_l( g_l - mu_l (e_l + d_l) )
    u_l = u_l + ( g_l - B_{l+1} g_{l+1} )                           after the sweeps

One outer iteration runs these blocks ascending (l = 1..L) then descending
(l = L-1..1) -- symmetric block Gauss-Seidel, i.e. a literal U: fine-scale
information reaches level L within the ascending half, coarse information
reaches level 1 within the descending half.

What that buys, concretely:
  * `g_l` persists across outer iterations instead of being overwritten, so
    texture the coarse levels cannot represent survives in `g_1`;
  * `rho_l` is LEARNED, turning the hard multilevel constraint into a penalised
    one whose strength the data sets.  rho -> 0 degenerates level 1 to plain
    CDLNet, which is a real escape hatch ML-ISTA does not have;
  * `u_l` is a running record of what the coarse levels failed to explain, fed
    back additively -- a skip WITH state.

Run to convergence with large rho, ADMM solves the same problem as ML-ISTA and
the bottleneck returns; the relief is a property of the truncated unroll plus
the learned rho.  That is the honest framing, and it is the same framing as
every other unrolled net in this repo.  Note also that multi-block ADMM (L >= 3
blocks) has no general convergence guarantee -- the coupling here is a chain,
which is far better behaved, but do not quote Algorithm 1's guarantee from the
ML-ISTA paper: that is two blocks with exact inner solves.

Step sizes
----------
`MLCDLNet` follows the repo convention: mu is absorbed into the filter norms by
`LISTALayer.spectral_normalize` (||B A|| = 1, so the ISTA step is 1) and training
moves it from there.  `MLSplitCDLNet` CANNOT do that -- its two arms carry
different weights rho_{l-1} and rho_l, which no single filter scale absorbs --
so mu_l is an explicit per-channel parameter, clamped in `project_` to the
majorisation bound

    mu_l <= 1 / ( max rho_{l-1} + max rho_l )

which holds because ||A_l B_l|| = 1 by construction and ||E|| <= 1.

Noise
-----
sigma is passed UNCHANGED to every level.  For a per-batch scalar sigma -- the
(B,1,1,1) form this repo trains on -- that costs nothing: each level's threshold
is `Polynomial(sigma)` with learned coefficients, so any constant per-level
rescaling of sigma (the multigrid 1/2 rule, ||a_m||_2 propagation, anything) is
absorbed exactly by those coefficients and gives the SAME trained model.  A
*spatial* noise map is different -- its profile cannot be absorbed by
per-channel coefficients, and it would not even broadcast against a coarse level
-- so `_check_sigma` rejects one rather than letting it fail cryptically two
levels down.

Read-out
--------
`readout='level1'` returns `D g_1`; `readout='cascade'` returns
`D (B_2 ... B_L g_L)`.  These are equal under the model but not after finitely
many iterations, and the difference is exactly where the two formulations
differ:

  * cascade is model-faithful, and makes K=1 with g_L = 0 reduce EXACTLY to a
    feed-forward strided CNN encoder/decoder (the paper's Eq. 14).  It is also
    the full deepest-code bottleneck.
  * level1 keeps the slack `g_1 != B_2 g_2`, which is where fine texture lives,
    but at K=1 the deeper levels are dead (ghat_1 = 0) and it degenerates to one
    CDLNet iteration.

They are the same cycle cut at different phases; see `residuals()` for the
diagnostic that says how much slack the trained model is actually using.

One consequence to know before launching a distributed run.  ML-ISTA updates
level 1 FIRST in its ascending sweep, so under `readout='level1'` everything the
FINAL sweep does above level 1 is downstream of the output: the analysis and
prox of levels 2..L in sweep K-1 are structurally unreachable, and no way of
writing the forward makes them live.  `MLSweep`'s `stop` argument avoids paying
for them, but they still hold parameters that never receive a gradient -- so
`MLCDLNet(readout='level1')` needs `find_unused_parameters=True` under DDP.
Neither `readout='cascade'` (where g_L is the output path) nor `MLSplitCDLNet`
(whose descending half updates level 1 LAST, so every level feeds g_1 inside the
same sweep) has this tail.  That is a small structural point in the split
formulation's favour, on top of the skips.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from models.base import set_weight
from models.components import ConvTranspose2d
from models.lista import LISTALayer, gram
from models.prox import GroupThreshold, Polynomial
from operators.identity import Identity
from operators.projections import uball_project
from preprocessing.image import post_process, pre_process
from preprocessing.kspace import kspace_post_process, kspace_pre_process


# ===========================================================================
#  level schedule
# ===========================================================================
def level_channels(C, M, L, widen):
    """`[M_0, M_1, ..., M_L]` with `M_0 = C` and `M_l = M widen^(l-1)`."""
    return [int(C)] + [max(1, int(round(M * widen ** (l - 1))))
                       for l in range(1, int(L) + 1)]


def level_strides(s, L):
    """`[None, s, 2, 2, ...]`, 1-indexed by level.

    Level 1 carries the model's own latent stride; every deeper level halves the
    grid, which is what makes level l a *resolution* level and not merely
    another dictionary factorisation.
    """
    return [None, int(s)] + [2] * (int(L) - 1)


def _init_size(level):
    """Power-method grid for level `l`: 128, 64, 32, 32, ...

    The spectral norm of a convolution does not depend on the grid, and deep
    levels carry many channels, so there is no reason to estimate it on a
    128x128 field of M_{l-1} = 256 channels.
    """
    return max(32, 128 >> (int(level) - 1))


# ===========================================================================
#  ML-ISTA
# ===========================================================================
class _SweepBase(nn.Module):
    """Shared level bookkeeping for the two sweeps.

    `levels[i]` is level `i+1`, so level l's analysis/synthesis pair is
    `levels[l-1]` and the operator that synthesises level l FROM level l+1 is
    `levels[l].synthesis`.  Getting that off-by-one wrong is the single easiest
    mistake in this file, which is why both sweeps are tested against a
    from-the-paper reference implementation.
    """

    def synthesize(self, g_L, stop=1):
        """`B_{stop+1} ... B_L g_L`: the level-`stop` code implied by `g_L`.

        `stop=1` gives ghat_1; `stop=0` runs one more synthesis and lands in the
        image domain.  `None` propagates, since a zero code synthesises to a
        zero code and `LISTALayer` takes its cold-start shortcut on `None`.
        """
        g = g_L
        for l in range(self.L - 1, int(stop) - 1, -1):
            g = None if g is None else self.levels[l].synthesis(g)
        return g

    def extra_repr(self):
        return "L=%d, channels=%s, strides=%s" % (
            self.L, self.Mch, self.strides[1:])


class MLSweep(_SweepBase):
    """One outer ML-ISTA iteration over `L` levels."""

    def __init__(self, C, M, L, widen=2, P=7, s=1, Mh=None, spectral_init=True,
                 **layer_kws):
        super().__init__()
        self.L = int(L)
        self.Mch = level_channels(C, M, self.L, widen)
        self.strides = level_strides(s, self.L)

        levels = []
        for l in range(1, self.L + 1):
            Mh_l = None if Mh is None else max(1, int(round(Mh * widen ** (l - 1))))
            layer = LISTALayer(self.Mch[l - 1], self.Mch[l], P=P,
                               stride=self.strides[l], Mh=Mh_l, **layer_kws)
            layer.init_filters()
            if spectral_init:
                layer.spectral_normalize(size=_init_size(l))
            levels.append(layer)
        self.levels = nn.ModuleList(levels)

    def forward(self, g_L, y_tilde, E=None, sigma=None, cache=None, codes=None,
                stop=None):
        """`stop=l` truncates the analysis sweep after level l.

        Only the container uses it, and only for the FINAL sweep under
        `readout='level1'`: the read-out is `D g_1`, and g_1 is produced before
        levels 2..L in the ascending sweep, so those levels' analysis and prox
        would be computed and then discarded -- dead compute, and worse, dead
        parameters that never receive a gradient (which is also what trips DDP's
        unused-parameter check).  This is the "where you cut the cycle" choice
        made concrete: `level1` cuts right after level 1's update, so the rest
        of that ascending half simply does not belong to the network.

        `codes` seeds the returned list, so the levels a truncated sweep does
        not touch keep their value from the previous sweep and `residuals()`
        still has something to compare.
        """
        if cache is None:
            cache = {}
        stop = self.L if stop is None else int(stop)

        # ---- synthesis sweep (coarse -> fine): the decoder ------------------
        ghat = [None] * (self.L + 1)
        ghat[self.L] = g_L
        for l in range(self.L - 1, 0, -1):
            g = ghat[l + 1]
            ghat[l] = None if g is None else self.levels[l].synthesis(g)

        # ---- analysis sweep (fine -> coarse): the encoder -------------------
        # Each level gets its OWN cache.  The group prox's adjacency is specific
        # to a (grid, channel count), so unlike `LISTA` -- where every layer
        # deliberately shares one -- the levels must not share, or level 2 would
        # reuse level 1's Gamma.
        out = [None] * (self.L + 1) if codes is None else list(codes)
        prev = y_tilde
        for l in range(1, stop + 1):
            key = "level%d" % l
            # E only at level 1: deeper levels have no measurement, their
            # "measurement" is the finer code, and `gram(None, .)` is identity.
            prev, cache[key] = self.levels[l - 1](
                ghat[l], prev, E=(E if l == 1 else None), sigma=sigma,
                cache=cache.setdefault(key, {}))
            out[l] = prev
        return out, cache


# ===========================================================================
#  linearized ADMM over the same model
# ===========================================================================
class MLSplitLevel(LISTALayer):
    """One level of the split formulation.

    Subclasses `LISTALayer` purely for its analysis / synthesis / prox
    construction and its `init_filters`, `spectral_normalize` and `project_`.
    The update differs enough to override `forward` outright, so the two are NOT
    interchangeable the way `LISTALayer` and `VCycle` are.

    `rho` penalises the constraint `g_l = D_{l+1} g_{l+1}`, which lives on level
    l's channels -- so it is owned here, per-channel, and level l+1 reaches back
    for it, applying it INSIDE its own analysis where the residual still has
    M_l channels.  The coarsest level has no constraint below it, hence no rho.
    """

    def __init__(self, C, M, has_rho=True, mu0=0.5, coupling0=1.0, **kws):
        kws.pop("multigrid", None)              # no FAS correction here
        super().__init__(C, M, multigrid=False, **kws)
        # `coupling0`, not `rho0`: `rho0` is already taken by GroupThreshold's
        # adjacency-blend rate and travels inside `prox_kws`. Two unrelated
        # rhos, and conflating them would silently retune the prox.
        self.mu = Polynomial(M, degrees=0, tau0=mu0)
        self.rho = Polynomial(M, degrees=0, tau0=coupling0) if has_rho else None

    def forward(self, g, target, u_prev=None, rho_prev=None, Bg_next=None,
                u=None, E=None, sigma=None, cache=None):
        """One linearized-ADMM block.

        g         g_l, the current level-l code (never None -- see `_zeros`)
        target    g_{l-1}, or y~ at level 1
        u_prev    u_{l-1}, or None at level 1 (u_0 = 0)
        rho_prev  rho_{l-1} already evaluated to (1, M_{l-1}, 1, 1), or None
                  at level 1 (rho_0 = 1)
        Bg_next   B_{l+1} g_{l+1}, or None at the coarsest level (rho_L = 0)
        u         u_l, or None at the coarsest level
        """
        # -- encoder arm (at level 1 this IS the data-fidelity gradient) ------
        resid = gram(E, self.synthesis(g)) - target
        if u_prev is not None:
            resid = resid - u_prev
        if rho_prev is not None:
            resid = resid * rho_prev
        step = self.analysis(resid)

        # -- decoder arm: the skip -------------------------------------------
        if self.rho is not None and Bg_next is not None:
            d = g - Bg_next
            if u is not None:
                d = d + u
            step = step + self.rho(sigma, ref=g) * d

        return self.prox(g - self.mu(sigma, ref=g) * step, sigma, cache)


class MLSplitSweep(_SweepBase):
    """One outer iteration: ascending blocks, descending blocks, dual ascent.

    Note that level 1 is updated LAST (it is the turning point of the descending
    half), so unlike `MLSweep` there is nothing to truncate under
    `readout='level1'` -- every level already feeds g_1 within the same sweep.
    """

    def __init__(self, C, M, L, widen=2, P=7, s=1, Mh=None, mu0=0.5,
                 coupling0=1.0, spectral_init=True, **layer_kws):
        super().__init__()
        self.L = int(L)
        self.Mch = level_channels(C, M, self.L, widen)
        self.strides = level_strides(s, self.L)

        levels = []
        for l in range(1, self.L + 1):
            Mh_l = None if Mh is None else max(1, int(round(Mh * widen ** (l - 1))))
            layer = MLSplitLevel(self.Mch[l - 1], self.Mch[l], P=P,
                                 stride=self.strides[l], Mh=Mh_l,
                                 has_rho=(l < self.L), mu0=mu0,
                                 coupling0=coupling0, **layer_kws)
            layer.init_filters()
            if spectral_init:
                layer.spectral_normalize(size=_init_size(l))
            levels.append(layer)
        self.levels = nn.ModuleList(levels)

    # -- helpers -------------------------------------------------------------
    def _B_next(self, g, l):
        """`B_{l+1} g_{l+1}`, or None at the coarsest level."""
        return None if l >= self.L else self.levels[l].synthesis(g[l + 1])

    def _block(self, g, u, l, y_tilde, E, sigma, cache):
        lev = self.levels[l - 1]
        rho_prev = None if l == 1 else self.levels[l - 2].rho(sigma, ref=None)
        out, cache["level%d" % l] = lev(
            g[l],
            y_tilde if l == 1 else g[l - 1],
            u_prev=(None if l == 1 else u[l - 1]),
            rho_prev=rho_prev,
            Bg_next=self._B_next(g, l),
            u=(None if l >= self.L else u[l]),
            E=(E if l == 1 else None),
            sigma=sigma,
            cache=cache.setdefault("level%d" % l, {}),
        )
        return out

    def forward(self, g, u, y_tilde, E=None, sigma=None, cache=None):
        if cache is None:
            cache = {}
        g, u = list(g), list(u)

        # ---- encoder half-sweep (fine -> coarse) ---------------------------
        for l in range(1, self.L + 1):
            g[l] = self._block(g, u, l, y_tilde, E, sigma, cache)

        # ---- decoder half-sweep (coarse -> fine) ---------------------------
        # Level L is deliberately skipped: it was just updated at the end of the
        # encoder half, and symmetric Gauss-Seidel does not touch the endpoint
        # twice.
        for l in range(self.L - 1, 0, -1):
            g[l] = self._block(g, u, l, y_tilde, E, sigma, cache)

        # ---- dual ascent ----------------------------------------------------
        for l in range(1, self.L):
            u[l] = u[l] + (g[l] - self._B_next(g, l))

        return g, u, cache

    # -- constraints ---------------------------------------------------------
    @torch.no_grad()
    def project_(self):
        """rho >= 0 FIRST, then mu into the majorisation bound it implies.

        Deliberately not split across `MLSplitLevel.project_`: `nn.Module`
        walks parents before children, so a per-level clamp of mu would read a
        stale rho.  Filters and prox are left to the inherited `LISTALayer`
        / prox `project_`, which the container's module walk also reaches.
        """
        for lev in self.levels:
            if lev.rho is not None:
                lev.rho.project_(lo=0.0)
        for i, lev in enumerate(self.levels):
            rho_prev = 1.0 if i == 0 else float(self.levels[i - 1].rho.weight.max())
            rho_l = 0.0 if lev.rho is None else float(lev.rho.weight.max())
            lev.mu.weight.clamp_(0.0, 1.0 / max(rho_prev + rho_l, 1e-8))


# ===========================================================================
#  shared container
# ===========================================================================
class _MLIO(nn.Module):
    """Preprocessing, preconditions and hooks shared by every multilevel net.

    Split out of `_MLBase` so `models/ml_lpds.py::MLLPDSNet` -- which has no
    sweeps and no read-out dictionary -- can reuse them.  Needs `self.preproc`,
    `self.pad_stride`, `self.s` and `self.L`.
    """

    # -- preconditions -------------------------------------------------------
    def _check_sigma(self, sigma):
        """Reject a spatial noise map.

        This port passes sigma UNCHANGED to every level, which is exact (up to a
        reparameterisation the learned thresholds absorb) for a scalar or a
        (B,1,1,1) per-image level, and wrong for a map: the profile cannot be
        absorbed by per-channel coefficients, and a (B,1,H,W) tau would not even
        broadcast against level 2's (B,M_2,H/2,W/2) code.  Failing here beats
        failing there.
        """
        if torch.is_tensor(sigma) and sigma.dim() == 4 and \
                (sigma.shape[-1] > 1 or sigma.shape[-2] > 1):
            raise ValueError(
                "%s got a spatial noise map of shape %s. Per-level noise "
                "propagation is not implemented -- sigma is passed unchanged to "
                "every level, which is exact for a scalar or a (B,1,1,1) "
                "per-image level but not for a map. Reduce it to a per-image "
                "level, or implement propagation (see this module's docstring)."
                % (type(self).__name__, tuple(sigma.shape)))

    def _check_grid(self, hw):
        H, W = int(hw[0]), int(hw[1])
        if H % self.pad_stride or W % self.pad_stride:
            raise ValueError(
                "%s got a %dx%d input, which is not a multiple of pad_stride=%d "
                "(= s=%d x 2^(L-1=%d)). With preproc='identity' nothing pads it "
                "for you, so the deeper levels would not land back on the input "
                "grid. Crop or pad the data, or reduce L."
                % (type(self).__name__, H, W, self.pad_stride, self.s, self.L - 1))

    def _check_padding(self, hw, E):
        """`preproc='image'` pads y~ but NOT the operator -- so E must not care.

        Same trap as `MGCDLNet._check_padding`: the mask and sensitivity maps
        inside E stay at the original size, so a non-zero pad makes
        `gram(E, B z)` multiply tensors of different extent.  Denoising
        (E = Identity) is unaffected.
        """
        if isinstance(E, Identity):
            return
        H, W = int(hw[0]), int(hw[1])
        if H % self.pad_stride or W % self.pad_stride:
            raise ValueError(
                "%s(preproc=%r) would pad a %dx%d input up to a multiple of "
                "pad_stride=%d, but the encoding operator %r still holds %dx%d "
                "masks/maps. Use preproc='kspace' for reconstruction -- it pads "
                "the operator alongside y~ and applies the E^H E DC correction."
                % (type(self).__name__, self.preproc, H, W, self.pad_stride,
                   E, H, W))

    # -- preprocessing -------------------------------------------------------
    def _pre(self, y, E):
        if self.preproc == "kspace":
            y_tilde, E, params = kspace_pre_process(y, E, self.pad_stride)
            return y_tilde, E, params, kspace_post_process
        x_adj = E.adjoint(y) if not isinstance(E, Identity) else y
        if self.preproc == "identity":
            self._check_grid(x_adj.shape[-2:])
            return x_adj, E, None, None
        self._check_padding(x_adj.shape[-2:], E)
        y_tilde, params = pre_process(x_adj, self.pad_stride)
        return y_tilde, E, params, (lambda x, p: post_process(x, list(p)))

    # -- hooks ---------------------------------------------------------------
    @torch.no_grad()
    def project(self):
        for m in self.modules():
            if hasattr(m, "project_"):
                m.project_()

    def compile_flex(self):
        """torch.compile every group prox's fused kernel (GPU; call once)."""
        for m in self.modules():
            if isinstance(m, GroupThreshold) and m.attn_backend == "flex":
                m.compile_flex()
        return self


class _MLBase(_MLIO):
    """`preprocess -> K outer sweeps -> read-out -> postprocess`.

    Mirrors `MGCDLNet`'s container contract: same `forward(y, E, sigma, z0)`
    signature, same `(x_hat, z)` return, same `project()` module walk, same
    `attn_backend` / `compile_flex` hooks that `train.py` looks for.
    """

    sweep_cls = None                      # set by the subclasses

    def __init__(self, K=5, L=2, M=32, C=1, P=7, s=1, widen=2, Mh=None, W=1,
                 tau0=1e-2, degrees=1, is_complex=True, preproc="image",
                 readout="level1", tie_outer=False, dK=1, sim_fun="distance",
                 nheads=1, rho0=1.0, gamma0=0.8, init_strategy="spectral_norm",
                 subgrad_mode="rigorous", attn_backend="gather",
                 flex_block_size=128, **sweep_kws):
        super().__init__()
        self.K, self.L = int(K), int(L)
        self.M, self.C, self.P, self.s = int(M), int(C), int(P), int(s)
        self.widen = widen
        self.is_complex = bool(is_complex)
        if self.L < 1:
            raise ValueError("L must be >= 1; got %r" % (L,))
        if self.K < 1:
            raise ValueError("K must be >= 1; got %r" % (K,))
        if preproc not in ("image", "kspace", "identity"):
            raise ValueError(
                "preproc must be 'image' (denoising), 'kspace' (reconstruction) "
                "or 'identity'; got %r" % (preproc,))
        if readout not in ("level1", "cascade"):
            raise ValueError(
                "readout must be 'level1' (D g_1, keeps the model slack) or "
                "'cascade' (D B_2..B_L g_L, model-faithful); got %r" % (readout,))
        self.preproc, self.readout = preproc, readout
        self.tie_outer = bool(tie_outer)
        self.attn_backend = attn_backend

        self.Mch = level_channels(C, M, self.L, widen)
        self.strides = level_strides(s, self.L)
        # Every level must halve exactly, on the image AND the strided latent
        # grid -- same contract as MGCDLNet.pad_stride.
        self.pad_stride = self.s * (2 ** (self.L - 1))

        prox_kws = dict(tau0=tau0, degrees=degrees, nheads=nheads, dK=dK,
                        sim_fun=sim_fun, rho0=rho0, gamma0=gamma0,
                        init_strategy=init_strategy, subgrad_mode=subgrad_mode,
                        attn_backend=attn_backend, flex_block_size=flex_block_size)
        layer_kws = dict(P=P, s=s, Mh=Mh, is_complex=is_complex, window=W,
                         prox_kws=prox_kws, **sweep_kws)

        # One prototype, deep-copied K times, so every outer iteration starts
        # from the SAME spectrally-normalised dictionaries -- the convention
        # `LISTA` uses, and what makes iteration k an exact repeat at init.
        proto = self.sweep_cls(C, M, self.L, widen=widen, **layer_kws)
        n = 1 if self.tie_outer else self.K
        self.sweeps = nn.ModuleList(
            [proto] + [copy.deepcopy(proto) for _ in range(n - 1)])

        # Read-out dictionary, initialised from level 1's synthesis.
        self.D = ConvTranspose2d(self.Mch[1], C, P, stride=s, bias=False,
                                 complex=is_complex)
        with torch.no_grad():
            set_weight(self.D, self.sweeps[0].levels[0].synthesis.weight)

    # -- per-iteration sweep -------------------------------------------------
    def _sweep(self, k):
        return self.sweeps[0] if self.tie_outer else self.sweeps[k]

    # -- forward -------------------------------------------------------------
    def _readout_code(self, codes, k_last):
        if self.readout == "level1" or self.L == 1:
            return codes[1]
        return self._sweep(k_last).synthesize(codes[self.L], stop=1)

    def forward(self, y, E=None, sigma=None, z0=None):
        """`(x_hat, z)` -- the repo-wide model interface.

        `z` is the code actually handed to the read-out, i.e. `g_1` under
        `readout='level1'` and `B_2..B_L g_L` under `'cascade'`.
        """
        x_hat, z, _ = self._run(y, E, sigma, z0)
        return x_hat, z

    def forward_codes(self, y, E=None, sigma=None, z0=None):
        """`(x_hat, state)`: the diagnostic entry point.

        `state` is the per-level code list for `MLCDLNet` and the `(g, u)` pair
        for `MLSplitCDLNet`.  Feed the codes to `residuals()`.
        """
        x_hat, _, state = self._run(y, E, sigma, z0)
        return x_hat, state

    def _run(self, y, E, sigma, z0):
        """`(x_hat, z, state)`; implemented per algorithm."""
        raise NotImplementedError

    # -- constraints ---------------------------------------------------------
    @torch.no_grad()
    def project_(self):
        set_weight(self.D, uball_project(self.D.weight))

    # -- diagnostics ---------------------------------------------------------
    def residuals(self, codes, k=-1):
        """`[g_l - B_{l+1} g_{l+1}]` for l = 1..L-1: the model-consistency error.

        This is the number to log alongside PSNR.  Under the ML-CSC model it is
        zero; after finitely many iterations it is the slack the network is
        actually using -- how much fine detail lives outside what the coarse
        levels can represent.  Near zero means the multilevel prior is satisfied
        exactly and the deepest code really is the bottleneck; large means the
        model is being violated to fit the data.

        Takes whatever `forward_codes` returned -- the code list, or the
        `(g, u)` pair from the split net.  For `MLCDLNet(readout='level1')` the
        final sweep stops at level 1, so this compares g_1 from sweep K-1 with
        the codes levels 2..L reached at sweep K-2 -- one sweep of skew, which
        is the price of not computing levels the read-out never sees.
        """
        if isinstance(codes, tuple):
            codes = codes[0]
        sweep = self._sweep(self.K - 1 if k < 0 else k)
        return [codes[l] - sweep.levels[l].synthesis(codes[l + 1])
                for l in range(1, self.L)]

    def extra_repr(self):
        return ("K=%d, L=%d, channels=%s, strides=%s, readout=%r, "
                "tie_outer=%s, preproc=%r"
                % (self.K, self.L, self.Mch, self.strides[1:], self.readout,
                   self.tie_outer, self.preproc))


# ===========================================================================
#  the two models
# ===========================================================================
class MLCDLNet(_MLBase):
    """Unrolled ML-ISTA.  State is the single deepest code `g_L`.

    `z0` seeds `g_L`; `None` means the zero code, which propagates through the
    synthesis sweep as `None` and makes `LISTALayer` take its cold-start
    shortcut -- so k=0 costs exactly a feed-forward pass, with no wasted
    synthesis of a zero tensor.

    Degeneracies worth asserting:
      * `L=1` reduces to plain CDLNet exactly (the synthesis sweep is empty and
        the analysis sweep is one `LISTALayer` step per outer iteration).
      * `K=1`, `z0=None`, `readout='cascade'` reduces to a feed-forward strided
        CNN encoder followed by the transposed decoder.
    """

    sweep_cls = MLSweep

    def _run(self, y, E, sigma, z0):
        if E is None:
            E = Identity()
        self._check_sigma(sigma)
        y_tilde, E, params, post = self._pre(y, E)

        # Under `readout='level1'` the final ascending sweep is cut after level
        # 1 -- see `MLSweep.forward`.  Not at K=1, where truncating would leave
        # levels 2..L with no value at all: that config is the documented
        # degeneracy (one CDLNet iteration, deeper levels dead either way), and
        # a complete `codes` list is worth more there than the saved work.
        stop = 1 if (self.readout == "level1" and self.L > 1 and self.K > 1) \
            else None

        g_L, codes, cache = z0, [None] * (self.L + 1), {}
        for k in range(self.K):
            codes, cache = self._sweep(k)(
                g_L, y_tilde, E=E, sigma=sigma, cache=cache, codes=codes,
                stop=(stop if k == self.K - 1 else None))
            g_L = codes[self.L]

        z = self._readout_code(codes, self.K - 1)
        x = self.D(z)
        return (x if params is None else post(x, params)), z, codes


class MLSplitCDLNet(_MLBase):
    """Unrolled linearized ADMM over the multilevel model.

    State is `(g, u)`: one code per level and one dual per constraint.  `z0`
    seeds `g_L` only -- the other levels and all duals start at zero -- so a
    warm start from an `MLCDLNet` checkpoint's deepest code is meaningful.

    Degeneracies worth asserting:
      * `L=1` reduces to plain CDLNet with an explicit step (no rho, no dual).
      * all `rho = 0` decouples the levels entirely; with `readout='level1'` the
        output then depends on level 1 alone, which is exactly CDLNet.
    """

    sweep_cls = MLSplitSweep

    def __init__(self, *args, mu0=0.5, coupling0=1.0, **kws):
        super().__init__(*args, mu0=mu0, coupling0=coupling0, **kws)
        self.project()          # start inside the mu <= 1/(rho+rho) bound

    def _zeros(self, y_tilde):
        """Materialise `g` and `u` at zero.

        Unlike ML-ISTA, `None` cannot stand in for the zero code here: the
        blocks read `g_l` directly (not just through a synthesis), and the duals
        need real tensors from the first update onwards.  Grid sizes are exact
        rather than probed -- `pad_stride` guarantees every level divides, and
        `Conv2d`'s padding is `(P-1)//2` with odd P, so H_l = H_{l-1} / s_l.
        """
        B = y_tilde.shape[0]
        H, W = int(y_tilde.shape[-2]), int(y_tilde.shape[-1])
        dtype = torch.complex64 if self.is_complex else (
            y_tilde.real.dtype if torch.is_complex(y_tilde) else y_tilde.dtype)
        g, u, hw = [None] * (self.L + 1), [None] * (self.L + 1), (H, W)
        for l in range(1, self.L + 1):
            hw = (hw[0] // self.strides[l], hw[1] // self.strides[l])
            g[l] = torch.zeros(B, self.Mch[l], hw[0], hw[1],
                               dtype=dtype, device=y_tilde.device)
            if l < self.L:
                u[l] = torch.zeros_like(g[l])
        return g, u

    def _run(self, y, E, sigma, z0):
        if E is None:
            E = Identity()
        self._check_sigma(sigma)
        y_tilde, E, params, post = self._pre(y, E)

        g, u = self._zeros(y_tilde)
        if z0 is not None:
            g[self.L] = z0
        cache = {}
        for k in range(self.K):
            g, u, cache = self._sweep(k)(g, u, y_tilde, E=E, sigma=sigma,
                                         cache=cache)

        z = self._readout_code(g, self.K - 1)
        x = self.D(z)
        return (x if params is None else post(x, params)), z, (g, u)
