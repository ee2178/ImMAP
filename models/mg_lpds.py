"""
Multigrid Learned Primal-Dual Splitting.

PyTorch port of Sljiva's `src/networks/mg_lpds.jl`.  Where `models/multigrid.py`
applies FAS to the single-variable synthesis problem of CDLNet/LISTA, this
applies it to the SADDLE-POINT problem solved by LPDS:

    min_x max_z  f(x) + <A x, z> - g*(z),        f(x) = 1/2 ||E x - y||^2

Why this is not `VCycle` with a different smoother
---------------------------------------------------
The propagated state is a pair, and the two halves live on different grids:

    x -- primal, IMAGE grid,  C channels, never widened
    z -- dual,   LATENT grid, M channels, widened per level

so a V-cycle has to restrict BOTH, form TWO corrections, and apply TWO
coarse-correction steps:

    pi_x = ( grad f_c(Rx) + A_c^T Rz )          - R( grad f_f(x) + A_f^T z )
    pi_z = ( Rz - prox_{g*_c}(Rz + A_c Rx) )    - R( z - prox_{g*_f}(z + A_f x) )

    x <- x + alpha_x * P(wx_c - x_c)     alpha_x: per-channel scale on C
    z <- z + alpha_z * P(wz_c - z_c)     alpha_z: (M*widen -> M) 1x1 conv

`models/multigrid.py::VCycle` carries one `pi` and one `alpha`, so it cannot
host this; `PDVCycle` below is the primal-dual sibling, not a subclass.

Note the asymmetry in the corrections: `pi_x` is a residual of the primal
GRADIENT, `pi_z` a FIXED-POINT residual of the dual map (`z - prox_{g*}(...)`).
A saddle-point problem has no single objective whose subgradient could serve as
both -- which is exactly why `ObjectiveDownsample` does not generalise.

Not ported (deliberately, see docs/multigrid_port.md)
-----------------------------------------------------
* the GUIDED prox variant.  The GROUP prox is supported: `window > 1` plus
  `Mh` puts a `GroupThreshold` in `LPDSLayer`'s prox slot, which is the real
  GroupLPDS -- a primal-dual sweep with nonlocal thresholding, extrapolation
  and all.  Reach it through `MGGroupLPDS` in `models/__init__.py`.

  `MGGroupLPDS` PREVIOUSLY meant `MGCDLNet(dual=True, W>1, Mh)`: a LISTA
  layer -- ONE variable, no `(x, z)` pair, no `tau`/`theta` extrapolation --
  whose only LPDS-like feature was the Fenchel prox. That reading is gone; the
  registry rejects its keys (`W`, `eta0`, `dual`) with a migration message.
  The ungrouped `MGLPDS` still means the LISTA-layer version.
* `SensitivityNet`.  `mg_lpds.jl` makes it a no-op whenever `use_smaps=true`,
  which is what every reference config and this repo's pipeline use (the maps
  come from the dataset).  `MGLPDSNet` takes the maps through `E`, so adding a
  sensitivity network later means estimating them and calling
  `operators.accessors.set_sensitivity_map` before the unroll -- one seam, in
  `forward`, marked below.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from models.base import set_weight
from models.lpds import LPDSLayer, LPDSStack, make_lpds_layer
from models.multigrid import (_ChannelScale, identity_widen_weight,
                              widen_filter, widen_prox_)
from models.prox import GroupThreshold, PixelConv
from operators.identity import Identity
from operators.projections import uball_project
from operators.resample import (GridTransfer, _check_filter, galerkin,
                                prolong, restrict, restrict_noise)
from preprocessing.image import post_process, pre_process
from preprocessing.kspace import kspace_post_process, kspace_pre_process


# ===========================================================================
#  weight transfer, fine -> coarse
# ===========================================================================
def first_pd_layer(module):
    """The `LPDSLayer` a level presents to its parent."""
    if isinstance(module, LPDSLayer):
        return module
    if isinstance(module, LPDSStack):
        return first_pd_layer(module.first)
    if isinstance(module, PDVCycle):
        return module.lpdsA.first
    raise TypeError(f"no LPDSLayer inside {type(module).__name__}")


@torch.no_grad()
def _copy_widened_layer(coarse, fine, widen):
    set_weight(coarse.analysis, widen_filter(fine.analysis.weight, widen))
    set_weight(coarse.synthesis, widen_filter(fine.synthesis.weight, widen))
    widen_prox_(coarse.prox, fine.prox, widen)
    # tau / theta live on the PRIMAL channels, which are never widened, so they
    # transfer verbatim rather than being tiled.
    coarse.tau.load_state_dict(fine.tau.state_dict())
    coarse.theta.load_state_dict(fine.theta.state_dict())


@torch.no_grad()
def preload_pd_with_widening(coarse, fine_layer, widen):
    """Initialise a coarse level from the fine level's filters, widened.

    Recurses, so the whole hierarchy descends from one spectrally-normalised
    dictionary rather than independent random draws -- the same invariant
    `models/multigrid.py::preload_with_widening` maintains.
    """
    if isinstance(coarse, LPDSStack):
        for layer in coarse.layers:
            _copy_widened_layer(layer, fine_layer, widen)
    elif isinstance(coarse, PDVCycle):
        for layer in coarse.lpdsA.layers:
            _copy_widened_layer(layer, fine_layer, widen)
        coarse.lpdsB.load_state_dict(coarse.lpdsA.state_dict())
        preload_pd_with_widening(coarse.mglayer, coarse.lpdsA.first, coarse.widen)
        coarse.retie_dF()
    else:
        raise TypeError(f"cannot preload into {type(coarse).__name__}")


# ===========================================================================
#  Delta dF -- restriction + the two FAS corrections
# ===========================================================================
class PDObjectiveDownsample(nn.Module):
    """Restrict `(x, z, y, E, sigma)` and form `(pi_x, pi_z)`.

    Owns its own copies of the fine and coarse analysis / synthesis / prox --
    tied to the enclosing smoothers at construction, untied by training, since
    this is evaluated once per V-cycle at a fixed point and can afford to
    specialise.
    """

    def __init__(self, fine_layer, coarse_layer, widen=1, julia_compat=False,
                 transfer_filter=None, transfer=None):
        super().__init__()
        self.widen = int(widen)
        self.julia_compat = bool(julia_compat)
        self.transfer_filter = _check_filter(transfer_filter)
        # LATENT-grid transfer, shared with the enclosing V-cycle. Only the
        # DUAL variable z lives there; the primal x is on the image grid and
        # keeps the fixed filter, alongside y / E / sigma.
        self.transfer = transfer

        self.analysis_fine = copy.deepcopy(fine_layer.analysis)
        self.synthesis_fine = copy.deepcopy(fine_layer.synthesis)
        self.prox_fine = copy.deepcopy(fine_layer.prox)
        self.analysis_coarse = copy.deepcopy(coarse_layer.analysis)
        self.synthesis_coarse = copy.deepcopy(coarse_layer.synthesis)
        self.prox_coarse = copy.deepcopy(coarse_layer.prox)

        M_fine, M_coarse = fine_layer.M, coarse_layer.M
        if M_coarse == M_fine:
            self.widen_z = nn.Identity()
        else:
            self.widen_z = PixelConv(M_fine, M_coarse)
            with torch.no_grad():
                self.widen_z.weight.copy_(
                    identity_widen_weight(M_coarse, M_fine, self.widen))

    def forward(self, x, z, y, E, sigma, cache, static=None):
        R = dict(julia_compat=self.julia_compat, filter=self.transfer_filter)

        # ---- fine-level primal-dual residual --------------------------------
        # primal:  grad f(x) + A^T z
        rx_fine = gram_or_x(E, x) - y + self.synthesis_fine(z)
        # dual: the FIXED-POINT residual z - prox_{g*}(z + A x), not a gradient
        zhat, cache["_dF_fine"] = self.prox_fine(
            z + self.analysis_fine(x), sigma, cache.setdefault("_dF_fine", {}))
        rz_fine = z - zhat

        # ---- restriction: x and y on the image grid, z on the latent grid ----
        # IMAGE path -- the primal variable and its residual sit on the same
        # grid as y and E, so they coarsen with the fixed filter. Learning here
        # would be learning the coarse forward model.
        x_c = restrict(x, **R)
        Rrx = restrict(rx_fine, **R)

        # LATENT path -- the dual variable and its residual.
        if self.transfer is not None:
            z_c = self.widen_z(self.transfer.restrict(z))
            Rrz = self.widen_z(self.transfer.restrict(rz_fine))
        else:
            z_c = self.widen_z(restrict(z, **R))
            Rrz = self.widen_z(restrict(rz_fine, **R))

        y_c, E_c, sigma_c = _restrict_measurement(y, E, sigma, R, static)

        # ---- coarse-level primal-dual residual ------------------------------
        # Handed on to the coarse level, whose first sweep needs the same Gram.
        gram_x_c = gram_or_x(E_c, x_c)
        rx_coarse = gram_x_c - y_c + self.synthesis_coarse(z_c)
        zhat_c, cache["_dF_coarse"] = self.prox_coarse(
            z_c + self.analysis_coarse(x_c), sigma_c,
            cache.setdefault("_dF_coarse", {}))
        rz_coarse = z_c - zhat_c

        return (x_c, z_c, rx_coarse - Rrx, rz_coarse - Rrz, y_c, E_c, sigma_c,
                gram_x_c)

    @torch.no_grad()
    def project_(self):
        for m in (self.analysis_fine, self.synthesis_fine,
                  self.analysis_coarse, self.synthesis_coarse):
            set_weight(m, uball_project(m.weight))


def gram_or_x(E, x):
    return x if (E is None or isinstance(E, Identity)) else E.gram(x)


def _restrict_measurement(y, E, sigma, R, static):
    """Coarsen `(y, E, sigma)` -- the parts of the level that never move.

    These depend on the measurement alone, not on the iterate, so the `K_outer`
    V-cycles at a level all want the same three objects and were each building
    their own.  `static` is a per-forward-pass slot threaded down the cache
    tree; the memo is guarded by object identity, so a miss costs exactly the
    recomputation it was trying to avoid and can never return the wrong grid.
    """
    key = (id(y), id(E), id(sigma))
    if static is not None:
        hit = static.get("_measurement")
        if hit is not None and hit[0] == key:
            return hit[1]

    val = (restrict(y, **R), galerkin(E, filter=R["filter"]),
           restrict_noise(sigma, **R))
    if static is not None:
        # The memo holds `val` alive, which is what keeps the ids in the next
        # level's key valid for the rest of the pass.
        static["_measurement"] = (key, val)
    return val


# ===========================================================================
#  the primal-dual V-cycle
# ===========================================================================
class PDVCycle(nn.Module):
    """One primal-dual V-cycle, recursive at construction time.

    Interchangeable with `LPDSStack`: same
    `(state, y_tilde, E, sigma, pi, cache)` signature, so either can be a
    level's `mglayer`.
    """

    def __init__(self, iters, C, M, widen=1, alpha0=1e-1, julia_compat=False,
                 spectral_init=True, transfer_filter=None,
                 learn_transfer=False, **layer_kws):
        super().__init__()
        assert len(iters) >= 2, "a V-cycle needs at least two levels"
        assert iters[0] % 2 == 0, \
            f"non-coarsest level iters must be even, got {iters[0]}"
        self.widen = int(widen)
        self.C, self.M = int(C), int(M)
        self.julia_compat = bool(julia_compat)
        # Resolved once, at construction: a V-cycle built inside a
        # `use_filter` block keeps that filter for the rest of its life, so a
        # checkpoint records which ablation arm produced it.
        self.transfer_filter = _check_filter(transfer_filter)
        self.learn_transfer = bool(learn_transfer)

        # One latent-grid kernel per level, shared between this level's dual
        # restriction (inside dF) and its dual prolongation (in forward), so
        # the two stay exact transposes while training moves them. channels=1
        # because `widen_z` / `alpha_z` change the channel count in between --
        # see the note in models/multigrid.py::VCycle.
        self.transfer = None
        if learn_transfer:
            if not torch.is_tensor(self.transfer_filter) and \
                    self.transfer_filter == "legacy":
                raise ValueError(
                    "learn_transfer=True is incompatible with "
                    "transfer_filter='legacy': its restriction and "
                    "prolongation are not transposes, so there is no single "
                    "kernel to learn. Pick any other filter to start from.")
            self.transfer = GridTransfer(channels=1, filter=self.transfer_filter,
                                         learn=True)

        half = iters[0] // 2
        self.lpdsA = LPDSStack(half, lambda: make_lpds_layer(
            C, M, spectral_init=spectral_init, **layer_kws))
        self.lpdsB = copy.deepcopy(self.lpdsA)

        # Only the DUAL channels widen; the primal stays at C.
        M_c = M * self.widen
        # The group prox's attention width has to widen WITH the dual channels.
        # `widen_prox_` tiles Walpha/Wbeta/Wtheta/Wphi by `widen` along both
        # channel axes, so a coarse prox left at the fine Mh is the wrong shape
        # and `preload_pd_with_widening` dies on the copy. `VCycle` already
        # does this; PDVCycle did not, which stayed invisible while widen == 1.
        coarse_kws = dict(layer_kws)
        if coarse_kws.get("Mh") is not None:
            coarse_kws["Mh"] = int(coarse_kws["Mh"]) * self.widen
            pk = dict(coarse_kws.get("prox_kws") or {})
            if pk.get("Mh") is not None:
                pk["Mh"] = int(pk["Mh"]) * self.widen
                coarse_kws["prox_kws"] = pk

        if len(iters) > 2:
            self.mglayer = PDVCycle(iters[1:], C, M_c, widen=self.widen,
                                    alpha0=alpha0, julia_compat=julia_compat,
                                    spectral_init=False,
                                    transfer_filter=self.transfer_filter,
                                    learn_transfer=learn_transfer,
                                    **coarse_kws)
        else:
            self.mglayer = LPDSStack(iters[1], lambda: make_lpds_layer(
                C, M_c, spectral_init=False, **coarse_kws))

        preload_pd_with_widening(self.mglayer, self.lpdsA.first, self.widen)

        self.dF = PDObjectiveDownsample(self.lpdsA.first,
                                        first_pd_layer(self.mglayer),
                                        widen=self.widen,
                                        julia_compat=julia_compat,
                                        transfer_filter=self.transfer_filter,
                                        transfer=self.transfer)

        # Two coarse-correction steps. Left unclamped by `project_`: they are
        # signed step sizes, unlike tau / theta.
        self.alpha_x = _ChannelScale(C, alpha0)
        if self.widen == 1:
            self.alpha_z = _ChannelScale(M, alpha0)
        else:
            self.alpha_z = PixelConv(M_c, M)
            with torch.no_grad():
                self.alpha_z.weight.copy_(
                    identity_widen_weight(M, M_c, self.widen, gain=alpha0))

    # -- construction-time weight tying -------------------------------------
    @torch.no_grad()
    def retie_dF(self):
        fine, coarse = self.lpdsA.first, first_pd_layer(self.mglayer)
        self.dF.analysis_fine.load_state_dict(fine.analysis.state_dict())
        self.dF.synthesis_fine.load_state_dict(fine.synthesis.state_dict())
        self.dF.prox_fine.load_state_dict(fine.prox.state_dict())
        self.dF.analysis_coarse.load_state_dict(coarse.analysis.state_dict())
        self.dF.synthesis_coarse.load_state_dict(coarse.synthesis.state_dict())
        self.dF.prox_coarse.load_state_dict(coarse.prox.state_dict())

    # -- forward ------------------------------------------------------------
    def forward(self, state, y_tilde, E=None, sigma=None, pi=None, cache=None):
        if cache is None:
            cache = {}
        static = cache.setdefault("_static", {})

        # pre-smooth (pi from the parent level, if any)
        state, cache = self.lpdsA(state, y_tilde, E=E, sigma=sigma, pi=pi,
                                  cache=cache)
        x, z = state

        # restrict + the two FAS corrections
        (x_c, z_c, pi_x, pi_z, y_c, E_c, sigma_c,
         gram_x_c) = self.dF(x, z, y_tilde, E, sigma, cache, static)

        # recurse -- the coarse level keeps its own adjacency cache, and
        # inherits both the shared static slot and the Gram dF just formed at
        # (E_c, x_c), which its first sweep would otherwise recompute verbatim.
        coarse = cache.setdefault("_coarse", {})
        coarse.setdefault("_static", static.setdefault("_coarse", {}))
        coarse["_gram_x"] = (x_c, gram_x_c)
        (wx_c, wz_c), cache["_coarse"] = self.mglayer(
            (x_c, z_c), y_c, E=E_c, sigma=sigma_c, pi=(pi_x, pi_z),
            cache=coarse)

        # prolongate and apply the two coarse-correction steps separately
        # image path stays fixed; latent path follows dF's restriction
        x = x + self.alpha_x(prolong(wx_c - x_c, filter=self.transfer_filter))
        corr_z = wz_c - z_c
        if self.transfer is not None:
            corr_z = self.transfer.prolong(corr_z)
        else:
            corr_z = prolong(corr_z, filter=self.transfer_filter)
        z = z + self.alpha_z(corr_z)

        # post-smooth, inheriting the pre-smoother's cache
        state, cache = self.lpdsB((x, z), y_tilde, E=E, sigma=sigma, pi=pi,
                                  cache=cache)
        return state, cache

    # -- introspection ------------------------------------------------------
    @property
    def depth(self):
        return 1 + (self.mglayer.depth if isinstance(self.mglayer, PDVCycle) else 1)

    @property
    def iters_per_level(self):
        here = [self.lpdsA.J + self.lpdsB.J]
        if isinstance(self.mglayer, PDVCycle):
            return here + self.mglayer.iters_per_level
        return here + [self.mglayer.J]

    def extra_repr(self):
        return (f"depth={self.depth}, iters_per_level={self.iters_per_level}, "
                f"M={self.M} x widen^l (widen={self.widen})")


# ===========================================================================
#  MGLPDSNet
# ===========================================================================
class MGLPDSNet(nn.Module):
    """LPDS whose iterations are primal-dual V-cycles.

    `K` follows the same convention as `MGCDLNet`, so the multigrid ablation is
    a one-key change rather than a different class:

        K = [K_outer, [i0, i1, ...]]   -> K_outer primal-dual V-cycles
        K = int                        -> a plain LPDS stack, no V-cycle

    Returns `(x_hat, (x, z))`. The second element is the repo's usual "latent",
    but for a primal-dual method the state is the PAIR, not the code alone --
    and `models/ladmm.py` threads exactly this back in as `state` when
    `reuse_latent=True`, so handing back `z` only would make the warm start half
    a warm start (`x` would cold-restart from `y~` every ADMM layer).
    """

    def __init__(self, K=(1, (8, 8, 8)), M=169, C=1, P=7, s=2, widen=1,
                 lam0=1e-2, tau0=1e-1, theta0=1e-1, degrees=0, alpha0=1e-1,
                 is_complex=True, window=1, Mh=None, dK=1, sim_fun="distance",
                 nheads=1, rho0=1.0, gamma0=0.8,
                 init_strategy="spectral_norm", subgrad_mode="rigorous",
                 attn_backend="gather", flex_block_size=128,
                 preproc="kspace", resize_noise=False, julia_compat=False,
                 transfer_filter=None, learn_transfer=False):
        super().__init__()
        K_outer, iters = _parse_K(K)
        self.K, self.iters = K_outer, iters
        self.M, self.C, self.P, self.s = int(M), int(C), int(P), int(s)
        self.preproc, self.resize_noise = preproc, bool(resize_noise)
        self.transfer_filter = _check_filter(transfer_filter)
        self.learn_transfer = bool(learn_transfer)
        self.levels = 1 if iters is None else len(iters)
        self.attn_backend = attn_backend
        if preproc not in ("image", "kspace", "identity"):
            raise ValueError(
                f"preproc must be 'image' (denoising), 'kspace' "
                f"(reconstruction) or 'identity'; got {preproc!r}")

        # Every level halves, on the image grid and the strided latent grid.
        self.pad_stride = self.s * (2 ** (self.levels - 1))

        prox_kws = dict(tau0=lam0, degrees=degrees, nheads=nheads, dK=dK,
                        sim_fun=sim_fun, rho0=rho0, gamma0=gamma0,
                        init_strategy=init_strategy, subgrad_mode=subgrad_mode,
                        attn_backend=attn_backend, flex_block_size=flex_block_size)
        layer_kws = dict(P=P, stride=s, is_complex=is_complex, window=window,
                         Mh=Mh, lam0=lam0, tau0=tau0, theta0=theta0,
                         degrees=degrees, prox_kws=prox_kws)

        if iters is None:                                   # plain LPDS
            self.net = LPDSStack(K_outer, lambda: make_lpds_layer(
                C, M, **layer_kws))
        else:
            self.net = _OuterStack(K_outer, lambda: PDVCycle(
                iters, C, M, widen=widen, alpha0=alpha0,
                julia_compat=julia_compat,
                transfer_filter=self.transfer_filter,
                learn_transfer=self.learn_transfer, **layer_kws))

    # ----------------------------------------------------------------------
    def forward(self, y, E=None, sigma=None, state=None):
        if E is None:
            E = Identity()

        # SEAM for a future SensitivityNet: estimate the maps from the ACS of
        # `y` here and `set_sensitivity_map(E, smaps)` before preprocessing.
        # `mg_lpds.jl` does exactly that when `use_smaps=false`.

        if self.preproc == "kspace":
            y_tilde, E, params = kspace_pre_process(y, E, self.pad_stride)
            post = kspace_post_process
        else:
            x_adj = E.adjoint(y) if not isinstance(E, Identity) else y
            if self.preproc == "identity":
                y_tilde, params, post = x_adj, None, None
                self._check_grid(x_adj.shape[-2:])
            else:
                y_tilde, params = pre_process(x_adj, self.pad_stride)
                post = lambda x, p: post_process(x, list(p))    # noqa: E731

        (x, z), _ = self.net(state, y_tilde, E=E, sigma=sigma, cache={})
        x_hat = x if params is None else post(x, params)
        return x_hat, (x, z)

    def _check_grid(self, hw):
        H, W = int(hw[0]), int(hw[1])
        if H % self.pad_stride or W % self.pad_stride:
            raise ValueError(
                f"{type(self).__name__} got a {H}x{W} input, not a multiple of "
                f"pad_stride={self.pad_stride} (= s={self.s} x "
                f"2^(levels-1={self.levels - 1})). With preproc='identity' "
                f"nothing pads it, so the V-cycle's restrict/prolong round trip "
                f"would not land back on the input grid.")

    # ----------------------------------------------------------------------
    def compile_flex(self):
        """torch.compile every group prox's fused kernel (GPU; call once).

        `train.py` calls this whenever `attn_backend == "flex"`. Without it a
        group-prox MGLPDSNet raises AttributeError there -- the flag was
        already threaded through the constructor, but the hook was missing.
        """
        for m in self.modules():
            if isinstance(m, GroupThreshold) and m.attn_backend == "flex":
                m.compile_flex()
        return self

    @torch.no_grad()
    def project(self):
        for m in self.modules():
            if m is not self and hasattr(m, "project_"):
                m.project_()

    def extra_repr(self):
        return f"K={self.K}, iters={self.iters}, M={self.M}, s={self.s}"


class _OuterStack(nn.Module):
    """K untied V-cycles, threading `(x, z)` between them."""

    def __init__(self, K, factory):
        super().__init__()
        self.K = int(K)
        proto = factory()
        self.layers = nn.ModuleList([proto] + [copy.deepcopy(proto)
                                               for _ in range(self.K - 1)])

    @property
    def first(self):
        return self.layers[0]

    def forward(self, state, y_tilde, E=None, sigma=None, pi=None, cache=None):
        if cache is None:
            cache = {}
        # One static slot for the whole stack: every outer V-cycle coarsens the
        # same (y~, E, sigma), so the restrictions are built once per pass.
        static = cache.setdefault("_static", {})
        for k, layer in enumerate(self.layers):
            # `cache` was previously rebound to the layer's own sub-dict, so
            # `_outer1` was created inside `_outer0` rather than beside it. The
            # sub-dicts start empty either way, so nothing read the difference,
            # but the shared static slot below needs them to be siblings.
            sub = cache.setdefault(f"_outer{k}", {})
            sub.setdefault("_static", static)
            state, cache[f"_outer{k}"] = layer(
                state, y_tilde, E=E, sigma=sigma, pi=pi, cache=sub)
        return state, cache

    def extra_repr(self):
        return f"K={self.K}"


def _parse_K(K):
    """`[K_outer, [i0, i1, ...]]` -> multigrid; a bare int -> plain LPDS."""
    if isinstance(K, int) or (torch.is_tensor(K) and K.dim() == 0):
        return int(K), None
    K = list(K)
    assert len(K) == 2, "K must be [K_outer, iters_per_level] or an int"
    K_outer, iters = int(K[0]), list(K[1])
    assert len(iters) >= 1
    if len(iters) == 1:                       # degenerate: no coarse level
        return K_outer * iters[0], None
    return K_outer, iters
