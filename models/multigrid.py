"""
Multigrid (V-cycle) convolutional dictionary learning networks.

PyTorch port of Sljiva's `src/networks/mg_lista.jl` plus the `CDLNet` container
from `src/networks/cdlnet.jl`.

    VCycle
      listaA   : pre-smoothing   (iters[0] // 2 LISTA iterations, this level)
      dF       : restriction with the FAS pi-correction + channel widening
      mglayer  : recursion -- another VCycle, or a plain LISTA at the coarsest
                 level
      alpha    : learned coarse-correction step / channel narrowing
      listaB   : post-smoothing  (iters[0] // 2 iterations, this level)

Level l carries `M * widen^l` subbands and `Mh * widen^l` attention channels,
U-Net style.

Full Approximation Scheme
-------------------------
The coarse level does not solve for an error (the problem is nonsmooth and
nonlinear), it solves for a full coarse approximation, corrected by

    pi = dF_c(R z) - R dF_f(z),

where `dF` is a subgradient of the CDLNet objective
`1/2 ||E B z - y||^2 + g(z)`.  Adding `eta . pi` inside the coarse prox makes
`z_c = R z` a fixed point of the coarse iteration whenever `z` is a fixed point
of the fine one -- the property that makes the correction consistent.  The
smooth half of dF is formed here, where A, B, E and y are all in scope; the
nonsmooth half comes from `prox.subgradient` (see models/prox.py).

Why the coarse level starts as an exact replica of the fine one
---------------------------------------------------------------
With `widen = w`, the coarse dictionary is the fine one tiled w times along the
subband axis and scaled by 1/sqrt(w) (spectral-norm preserving), and `widen_z`
lifts z to `[z; ...; z] / sqrt(w)`.  Then

    D_c (widen_z z) = (1/sqrt(w)) sum_j D (z / sqrt(w)) = D z

exactly.  The coarse level therefore begins life reproducing the fine level's
synthesis, and `alpha`'s identity-tiled initialisation means the coarse
correction enters as a plain scaled sum over the w copies.  Training breaks the
symmetry from there.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn

from models.base import set_weight
from models.components import ConvTranspose2d
from models.lista import LISTA, LISTALayer, gram
from models.prox import GroupThreshold, PixelConv, FenchelProx, resize_noise
from operators.identity import Identity
from operators.projections import uball_project
from operators.resample import galerkin, prolong, restrict, restrict_noise
from preprocessing.image import post_process, pre_process
from preprocessing.kspace import kspace_post_process, kspace_pre_process


def make_layer(C, M, Mh=None, spectral_init=True, **kws):
    """One `LISTALayer` with tied conjugate filters, optionally normalised.

    Coarse levels skip the power method: their filters are overwritten by
    `preload_with_widening` immediately afterwards, and the widening is norm
    preserving, so re-estimating a spectral norm there is wasted work.
    """
    layer = LISTALayer(C, M, Mh=Mh, **kws)
    layer.init_filters()
    if spectral_init:
        layer.spectral_normalize()
    return layer


# ===========================================================================
#  Widening helpers -- preserve spectral norm across levels
# ===========================================================================
def widen_filter(w, widen):
    """Tile a conv / conv-transpose filter along the subband axis.

    Both `Conv2d(C, M, P)` and `ConvTranspose2d(M, C, P)` store weights as
    `(M, C, P, P)` in torch, so widening M is a tile of dim 0 in both cases.
    The 1/sqrt(widen) factor keeps `||D||_2` unchanged.
    """
    if widen == 1:
        return w.clone()
    return w.repeat(widen, 1, 1, 1) / math.sqrt(widen)


def widen_pixel(w, widen):
    """Tile a 1x1 (grouped) transform along BOTH channel axes.

    Weight is `(cout, cin // groups, 1, 1)`.  Widening cout and cin by the same
    factor with the head count fixed tiles both axes; 1/widen preserves the
    block's spectral norm.
    """
    if widen == 1:
        return w.clone()
    return w.repeat(widen, widen, 1, 1) / widen


def identity_widen_weight(out_ch, in_ch, widen, gain=1.0):
    """Identity 1x1 weight, tiled along whichever axis the widening grew.

    For `widen_z` (M -> M*widen) this is `[I; ...; I] / sqrt(widen)`, i.e. copy
    z into every widened group.  For `alpha` (M*widen -> M) it is
    `[I ... I] / sqrt(widen)`, i.e. sum the groups back down.  `widen == 1`
    degenerates to `gain * I`.
    """
    if out_ch >= in_ch:
        core = torch.eye(out_ch // widen, in_ch) * gain
        w = core.repeat(widen, 1)
    else:
        core = torch.eye(out_ch, in_ch // widen) * gain
        w = core.repeat(1, widen)
    return (w / math.sqrt(widen)).reshape(out_ch, in_ch, 1, 1)


def _base_prox(prox):
    """Unwrap a `FenchelProx` to get at the underlying thresholding weights."""
    return prox.prox if isinstance(prox, FenchelProx) else prox


@torch.no_grad()
def widen_prox_(coarse_prox, fine_prox, widen):
    """Copy the fine prox's learned transforms into the coarse one, widened.

    Only the channel-mixing weights (W_alpha, W_beta, W_theta, W_phi) transfer:
    the thresholds tau / rho / gamma are per-channel polynomials of a different
    width and keep their fresh initialisation, exactly as in `mg_lista.jl`.
    No-op for soft-thresholding or an ungrouped group prox.
    """
    c, f = _base_prox(coarse_prox), _base_prox(fine_prox)
    if not (isinstance(c, GroupThreshold) and isinstance(f, GroupThreshold)):
        return
    if not (c.grouped and f.grouped):
        return
    for name in ("Walpha", "Wbeta", "Wtheta", "Wphi"):
        getattr(c, name).weight.copy_(widen_pixel(getattr(f, name).weight, widen))


@torch.no_grad()
def _copy_widened_layer(coarse_layer, fine_layer, widen):
    set_weight(coarse_layer.analysis, widen_filter(fine_layer.analysis.weight, widen))
    set_weight(coarse_layer.synthesis, widen_filter(fine_layer.synthesis.weight, widen))
    widen_prox_(coarse_layer.prox, fine_layer.prox, widen)


def first_layer(module):
    """The `LISTALayer` a level presents to its parent (for weight transfer)."""
    if isinstance(module, LISTALayer):
        return module
    if isinstance(module, LISTA):
        return first_layer(module.first)
    if isinstance(module, VCycle):
        return module.listaA.first
    raise TypeError(f"no LISTALayer inside {type(module).__name__}")


@torch.no_grad()
def preload_with_widening(coarse, fine_layer, widen):
    """Initialise a coarse level from the fine level's filters, widened.

    Recurses so the entire hierarchy descends from the single finest
    spectrally-normalised dictionary rather than from independent random draws.
    """
    if isinstance(coarse, LISTA):
        for layer in coarse.layers:
            _copy_widened_layer(layer, fine_layer, widen)
    elif isinstance(coarse, VCycle):
        for layer in coarse.listaA.layers:
            _copy_widened_layer(layer, fine_layer, widen)
        coarse.listaB.load_state_dict(coarse.listaA.state_dict())
        preload_with_widening(coarse.mglayer, coarse.listaA.first, coarse.widen)
        coarse.retie_dF()
    else:
        raise TypeError(f"cannot preload into {type(coarse).__name__}")


# ===========================================================================
#  Objective downsampling  (Delta dF)
# ===========================================================================
class ObjectiveDownsample(nn.Module):
    """Restrict `(z, y, E, sigma)` and form the FAS correction pi.

    Owns its own copies of the fine and coarse analysis / synthesis / prox --
    tied to the enclosing smoothers at construction, untied by training.  That
    is deliberate: dF is evaluated once per V-cycle at a *fixed* point, so it
    can afford to specialise away from the smoothers it started as.
    """

    def __init__(self, fine_layer, coarse_layer, widen=1, julia_compat=False):
        super().__init__()
        self.widen = int(widen)
        self.julia_compat = bool(julia_compat)

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

    def forward(self, z, y, E, sigma, cache):
        R = dict(julia_compat=self.julia_compat)

        # ---- fine-level subgradient  dF_f(z) = A^H (E^H E B z - y) + dg(z) ----
        Bz = self.synthesis_fine(z)
        Ar = self.analysis_fine(gram(E, Bz) - y)
        gz, cache["_dF_fine"] = self.prox_fine.subgradient(
            z, sigma, cache.setdefault("_dF_fine", {}))
        dF_fine = Ar + gz

        # ---- restriction: space first, then channels -------------------------
        z_c = self.widen_z(restrict(z, **R))
        RdF = self.widen_z(restrict(dF_fine, **R))
        y_c = restrict(y, **R)
        sigma_c = restrict_noise(sigma, **R)
        E_c = galerkin(E)

        # ---- coarse-level subgradient ---------------------------------------
        Bz_c = self.synthesis_coarse(z_c)
        Ar_c = self.analysis_coarse(gram(E_c, Bz_c) - y_c)
        gz_c, cache["_dF_coarse"] = self.prox_coarse.subgradient(
            z_c, sigma_c, cache.setdefault("_dF_coarse", {}))
        dF_coarse = Ar_c + gz_c

        return z_c, dF_coarse - RdF, y_c, E_c, sigma_c

    @torch.no_grad()
    def project_(self):
        for m in (self.analysis_fine, self.synthesis_fine,
                  self.analysis_coarse, self.synthesis_coarse):
            set_weight(m, uball_project(m.weight))


# ===========================================================================
#  V-cycle
# ===========================================================================
class VCycle(nn.Module):
    """One multigrid V-cycle, recursive at construction time.

    `iters` gives the iteration count per level, finest first.  Every
    non-coarsest entry must be even: half the iterations smooth before the
    coarse-grid correction and half after.
    """

    def __init__(self, iters, C, M, widen=1, alpha0=1e-1, alpha_conv=True,
                 Mh=None, julia_compat=False, spectral_init=True,
                 receives_pi=False, **layer_kws):
        super().__init__()
        assert len(iters) >= 2, "a V-cycle needs at least two levels"
        assert iters[0] % 2 == 0, \
            f"non-coarsest level iters must be even, got {iters[0]}"
        self.widen = int(widen)
        self.C, self.M, self.Mh = int(C), int(M), Mh
        self.julia_compat = bool(julia_compat)

        half = iters[0] // 2
        layer_kws = dict(layer_kws)
        layer_kws.pop("multigrid", None)

        # ---- this level's smoothers -----------------------------------------
        # `eta` (the pi scale) is only allocated where a pi can actually arrive.
        # The OUTERMOST V-cycle never receives one -- the outer LISTA calls it
        # with pi=None -- so its smoothers would otherwise carry a parameter
        # that never gets a gradient (and trips DDP's unused-parameter check).
        self.listaA = LISTA(half, lambda: make_layer(
            C, M, Mh=Mh, spectral_init=spectral_init,
            multigrid=bool(receives_pi), **layer_kws))
        self.listaB = copy.deepcopy(self.listaA)

        # ---- the coarser level ----------------------------------------------
        M_c = M * self.widen
        Mh_c = None if Mh is None else Mh * self.widen
        if len(iters) > 2:
            self.mglayer = VCycle(iters[1:], C, M_c, widen=self.widen,
                                  alpha0=alpha0, alpha_conv=alpha_conv, Mh=Mh_c,
                                  julia_compat=julia_compat, spectral_init=False,
                                  receives_pi=True, **layer_kws)
        else:
            self.mglayer = LISTA(iters[1], lambda: make_layer(
                C, M_c, Mh=Mh_c, spectral_init=False, multigrid=True, **layer_kws))

        # Derive the whole coarse subtree from this level's dictionary.
        preload_with_widening(self.mglayer, self.listaA.first, self.widen)

        # ---- restriction + coarse-correction step ---------------------------
        self.dF = ObjectiveDownsample(self.listaA.first, first_layer(self.mglayer),
                                      widen=self.widen, julia_compat=julia_compat)

        if alpha_conv:
            self.alpha = PixelConv(M_c, M)
            with torch.no_grad():
                self.alpha.weight.copy_(
                    identity_widen_weight(M, M_c, self.widen, gain=alpha0))
        else:
            assert self.widen == 1, "alpha_conv=False requires widen=1"
            self.alpha = _ChannelScale(M, alpha0)

    # -- construction-time weight tying ------------------------------------
    @torch.no_grad()
    def retie_dF(self):
        """Re-sync dF's copies with the (possibly re-preloaded) smoothers."""
        fine, coarse = self.listaA.first, first_layer(self.mglayer)
        self.dF.analysis_fine.load_state_dict(fine.analysis.state_dict())
        self.dF.synthesis_fine.load_state_dict(fine.synthesis.state_dict())
        self.dF.prox_fine.load_state_dict(fine.prox.state_dict())
        self.dF.analysis_coarse.load_state_dict(coarse.analysis.state_dict())
        self.dF.synthesis_coarse.load_state_dict(coarse.synthesis.state_dict())
        self.dF.prox_coarse.load_state_dict(coarse.prox.state_dict())

    # -- forward ------------------------------------------------------------
    def forward(self, z, y_tilde, E=None, sigma=None, pi=None, cache=None):
        if cache is None:
            cache = {}

        # pre-smooth (pi from the parent level, if any)
        z, cache = self.listaA(z, y_tilde, E=E, sigma=sigma, pi=pi, cache=cache)

        # restrict + FAS correction
        z_c, pi_c, y_c, E_c, sigma_c = self.dF(z, y_tilde, E, sigma, cache)

        # recurse -- the coarse level keeps its own adjacency cache
        w_c, cache["_coarse"] = self.mglayer(
            z_c, y_c, E=E_c, sigma=sigma_c, pi=pi_c,
            cache=cache.setdefault("_coarse", {}))

        # prolongate the coarse correction, narrow the channels, step
        z = z + self.alpha(prolong(w_c - z_c))

        # post-smooth, inheriting the pre-smoother's adjacency for free
        z, cache = self.listaB(z, y_tilde, E=E, sigma=sigma, pi=pi, cache=cache)
        return z, cache

    # -- introspection ------------------------------------------------------
    @property
    def depth(self):
        return 1 + (self.mglayer.depth if isinstance(self.mglayer, VCycle) else 1)

    @property
    def iters_per_level(self):
        here = [self.listaA.K + self.listaB.K]
        if isinstance(self.mglayer, VCycle):
            return here + self.mglayer.iters_per_level
        return here + [self.mglayer.K]

    def extra_repr(self):
        return (f"depth={self.depth}, iters_per_level={self.iters_per_level}, "
                f"M={self.M} x widen^l (widen={self.widen})")


class _ChannelScale(nn.Module):
    """Per-channel real scale; the `alpha_conv=False` branch of `mgVCycleLayer`."""

    def __init__(self, channels, gain=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.full((1, channels, 1, 1), float(gain)))

    def forward(self, x):
        return x * self.weight


# ===========================================================================
#  CDLNet container (cdlnet.jl)
# ===========================================================================
class MGCDLNet(nn.Module):
    """CDLNet whose iterations are multigrid V-cycles.

    `K` may be given either as the Julia-style `[K_outer, [i0, i1, ...]]` or as
    a plain int (plain CDLNet, no multigrid) -- in both cases this class is the
    `preprocess -> LISTA -> synthesis -> postprocess` container of `cdlnet.jl`.

    Returns `(x_hat, z)` to match the rest of the repo's model interface.
    """

    def __init__(self, K=(3, (8, 8, 4)), M=32, C=1, P=7, s=1, Mh=None, W=1,
                 widen=1, tau0=1e-2, degrees=0, eta0=1e-1, eta_degrees=0,
                 alpha0=1e-1, alpha_conv=True, is_complex=True, dual=False,
                 dK=1, sim_fun="distance", nheads=1, rho0=1.0, gamma0=0.8,
                 init_strategy="spectral_norm", subgrad_mode="rigorous",
                 attn_backend="gather", flex_block_size=128,
                 preproc="image", resize_noise=False, julia_compat=False):
        super().__init__()
        K_outer, iters = _parse_K(K)
        self.K, self.iters = K_outer, iters
        self.M, self.C, self.P, self.s = int(M), int(C), int(P), int(s)
        self.is_complex, self.dual = bool(is_complex), bool(dual)
        if preproc not in ("image", "kspace", "identity"):
            raise ValueError(
                f"preproc must be 'image' (denoising), 'kspace' "
                f"(reconstruction) or 'identity'; got {preproc!r}")
        self.preproc, self.resize_noise = preproc, bool(resize_noise)
        self.levels = 1 if iters is None else len(iters)
        # train.py compiles the fused FlexAttention kernel via
        # `getattr(model, "attn_backend", None) == "flex"`, the same hook
        # GroupCDL exposes. Without this attribute the compile silently never
        # happens and MGGroupCDL runs the uncompiled kernel.
        self.attn_backend = attn_backend

        # Inputs are padded up to a multiple of this so every level halves
        # exactly, in both the image grid and the strided latent grid.
        self.pad_stride = self.s * (2 ** (self.levels - 1))

        prox_kws = dict(tau0=tau0, degrees=degrees, nheads=nheads, dK=dK,
                        sim_fun=sim_fun, rho0=rho0, gamma0=gamma0,
                        init_strategy=init_strategy, subgrad_mode=subgrad_mode,
                        attn_backend=attn_backend, flex_block_size=flex_block_size)
        layer_kws = dict(P=P, stride=s, is_complex=is_complex, window=W,
                         dual=dual, eta0=eta0, eta_degrees=eta_degrees,
                         prox_kws=prox_kws)

        if iters is None:                                   # plain CDLNet
            self.lista = LISTA(K_outer,
                               lambda: make_layer(C, M, Mh=Mh, **layer_kws))
        else:
            self.lista = LISTA(K_outer, lambda: VCycle(
                iters, C, M, widen=widen, alpha0=alpha0, alpha_conv=alpha_conv,
                Mh=Mh, julia_compat=julia_compat, **layer_kws))

        # Read-out dictionary D, initialised from the first synthesis.
        self.D = ConvTranspose2d(M, C, P, stride=s, bias=False, complex=is_complex)
        with torch.no_grad():
            set_weight(self.D, first_layer(self.lista.first).synthesis.weight)

    # ----------------------------------------------------------------------
    def forward(self, y, E=None, sigma=None, z0=None):
        if E is None:
            E = Identity()

        if self.preproc == "kspace":
            # Pads the OPERATOR alongside y~, and removes DC through E^H E --
            # see preprocessing/kspace.py for why plain mean subtraction is
            # wrong once E is not the identity.
            y_tilde, E, params = kspace_pre_process(y, E, self.pad_stride)
            post = kspace_post_process
        else:
            x_adj = E.adjoint(y) if not isinstance(E, Identity) else y
            if self.preproc == "identity":
                y_tilde, params, post = x_adj, None, None
                self._check_grid(x_adj.shape[-2:])
            else:
                self._check_padding(x_adj.shape[-2:], E)
                y_tilde, params = pre_process(x_adj, self.pad_stride)
                post = lambda x, p: post_process(x, list(p))    # noqa: E731

        sig = sigma
        if self.resize_noise and torch.is_tensor(sigma) and sigma.dim() == 4:
            latent = (y_tilde.shape[-2] // self.s, y_tilde.shape[-1] // self.s)
            sig = resize_noise(sigma, latent)

        z, _ = self.lista(z0, y_tilde, E=E, sigma=sig, cache={})
        Dz = self.D(z)
        x = (y_tilde - Dz) if self.dual else Dz
        x_hat = x if params is None else post(x, params)
        return x_hat, z

    # -- input-size preconditions -------------------------------------------
    def _check_grid(self, hw):
        """Every level must halve exactly, on the image AND the latent grid."""
        H, W = int(hw[0]), int(hw[1])
        if H % self.pad_stride or W % self.pad_stride:
            raise ValueError(
                f"{type(self).__name__} got a {H}x{W} input, which is not a "
                f"multiple of pad_stride={self.pad_stride} "
                f"(= s={self.s} x 2^(levels-1={self.levels - 1})). With "
                f"preproc='identity' nothing pads it for you, so the V-cycle's "
                f"restrict/prolong round-trip would not land back on the input "
                f"grid. Crop or pad the data, or reduce the level count.")

    def _check_padding(self, hw, E):
        """`preproc='image'` pads y~ but NOT the operator -- so E must not care.

        pre_process pads the adjoint image up to `pad_stride`; the mask and the
        sensitivity maps inside E stay at the original size, so as soon as the
        pad is non-zero `gram(E, B z)` multiplies tensors of different spatial
        extent. Denoising (E = Identity) is unaffected, which is why the BSD432
        configs never hit this. `preproc='kspace'` is the reconstruction mode:
        it pads the operator too, and removes DC through E^H E rather than
        subtracting a plain mean (see preprocessing/kspace.py).
        """
        if isinstance(E, Identity):
            return
        H, W = int(hw[0]), int(hw[1])
        if H % self.pad_stride or W % self.pad_stride:
            raise ValueError(
                f"{type(self).__name__}(preproc={self.preproc!r}) would pad a "
                f"{H}x{W} input up to a multiple of pad_stride="
                f"{self.pad_stride}, but the encoding operator {E!r} still "
                f"holds {H}x{W} masks/maps. Use preproc='kspace' for "
                f"reconstruction -- it pads the operator alongside y~ and "
                f"applies the E^H E DC correction.")

    # ----------------------------------------------------------------------
    def compile_flex(self):
        """torch.compile every group prox's fused kernel (GPU; call once)."""
        for m in self.modules():
            if isinstance(m, GroupThreshold) and m.attn_backend == "flex":
                m.compile_flex()
        return self

    @torch.no_grad()
    def project_(self):
        """This module's own constraint: the read-out dictionary."""
        set_weight(self.D, uball_project(self.D.weight))

    @torch.no_grad()
    def project(self):
        """Constraint projection; recurses over every submodule that wants it.

        Simpler than the hand-written recursive `project!` chain of the Julia
        source: a new constrained submodule is covered the moment it defines
        `project_`, with no container to remember to update.
        """
        for m in self.modules():
            if hasattr(m, "project_"):
                m.project_()

    def extra_repr(self):
        return f"K={self.K}, iters={self.iters}, M={self.M}, s={self.s}"


def _parse_K(K):
    """`[K_outer, [i0, i1, ...]]` -> multigrid; a bare int -> plain CDLNet."""
    if isinstance(K, (int,)) or (torch.is_tensor(K) and K.dim() == 0):
        return int(K), None
    K = list(K)
    assert len(K) == 2, "K must be [K_outer, iters_per_level] or an int"
    K_outer, iters = int(K[0]), list(K[1])
    assert len(iters) >= 1
    if len(iters) == 1:                       # degenerate: no coarse level
        return K_outer * iters[0], None
    return K_outer, iters
