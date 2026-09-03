"""
LGGS -- Longitudinally-Guided Group-Sparse reconstruction.

PyTorch port of Sljiva's `src/networks/guided_lpds.jl` (`GuidedGroupLPDSLayer`,
`GuidedLPDSNet`) on top of `models/lpds.py::LPDSLayer` and
`models/guided_prox.py::GuidedGroupThreshold`.

"LGGS" is not a class in the Julia source -- it is the NAME the experiments give
to one configuration of `guided_lpdsnet`, and `scripts/makeconfigs_longi.jl`
pins it down:

    network        = config/guided_lpdsnet.yaml     (network_type: lpdsnet)
    network.guided = true , multiguide = true , joint_softmax = true
    network.windowsize = 1 , guide_windowsize = 15
    similarity     = pidistance
    K = 30 , M = 169 , p = 7 , stride = 2 , init_strategy = semi_orthogonal
    data.guided    = true , num_guides in {1, 3, 5}

so: a learned primal-dual splitting network whose DUAL step is a guided group
threshold, with a trivial (1x1) self window and a 15x15 guide window normalised
in one joint softmax.  `lggs_defaults()` below returns exactly those keys.

The algorithm
-------------
One sweep, `w` the fully-sampled guide image(s):

    Bz  = B z
    x+  = x - tau (E^H E x - y~ + Bz)                primal gradient step
    xb  = x+ + theta (x+ - x)                        over-relaxation
    v   = A w                                        guide, THIS layer's dictionary
    z+  = prox_{g*}( z + A xb , v )                  guided group clipping

and the cold start is `x = y~`, `z = prox_{g*}(A y~, A w)`.

Two details that are easy to get wrong and are deliberate here:

* `v = A w` is recomputed every layer.  The layers are UNTIED (`LPDSNet`'s
  parameters are a `deepcopy` per layer in `lpds.jl`), so layer k's guide latent
  is layer k's dictionary applied to the guide -- not a value hoisted out of the
  loop.  Hoisting it would silently tie the guide to layer 0's dictionary.
* the guide is preprocessed with its OWN mean subtraction, on the same padded
  grid as `y~` (`preproc.jl`: `w_tilde, _ = preprocess(ImagePreprocess(stride,
  false), w)`).  It is never DC-matched to the measurement -- the whole premise
  is that the guide's intensities are a different contrast / timepoint and only
  its geometry transfers.

Why this is the network for longitudinally-guided contrast synthesis
--------------------------------------------------------------------
The guide enters ONLY through the adjacency: it decides which pixels are pooled
together in the group-sparsity penalty, and its own intensities are never added
to the estimate (see `models/guided_prox.py`).  A prior fully-sampled CT1 study
therefore contributes anatomy and lesion geometry while leaving enhancement to
be recovered from the measured contrasts -- which is exactly the failure mode a
naive "concatenate the prior as an extra input channel" model has, where a
model can copy the prior's enhancement forward.

`E` is optional and defaults to `Identity`, so the same class covers the
reconstruction setting (`E` = SENSE/Fourier) and the synthesis setting
(`E = Identity`, `y` the observed contrast stack).
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from models.guided_prox import (GuidedGroupThreshold, as_guide_list,
                                build_guided_prox)
from models.lpds import LPDSLayer, gram
from operators.identity import Identity
from operators.padding import calc_pad_2d
from preprocessing.image import post_process, pre_process
from preprocessing.kspace import (_pad_complex, kspace_post_process,
                                  kspace_pre_process)


# ---------------------------------------------------------------------------
class GuidedLPDSLayer(LPDSLayer):
    """One primal-dual sweep whose dual prox is guided.

    Identical to `LPDSLayer` except for the prox slot and the extra `guide`
    argument threaded to it.  `guide` is in the IMAGE domain; this layer applies
    its own `analysis` to it, matching `guided_lpds.jl`.
    """

    def __init__(self, C, M, P=7, stride=1, lam0=1e-2, tau0=1e-1, theta0=1e-1,
                 degrees=0, is_complex=True, window=1, guide_window=None,
                 joint_softmax=True, Mh=None, prox_kws=None):
        # Build the plain layer first (convs, tau, theta, and a prox we replace)
        # so the analysis/synthesis construction and init stay in one place.
        super().__init__(C, M, P=P, stride=stride, lam0=lam0, tau0=tau0,
                         theta0=theta0, degrees=degrees, is_complex=is_complex,
                         window=1, Mh=None, prox_kws=None)

        prox_kws = dict(prox_kws or {})
        prox_kws.setdefault("tau0", lam0)
        prox_kws.setdefault("degrees", degrees)
        self.prox = build_guided_prox(M, Mh=Mh, window=window,
                                      guide_window=guide_window,
                                      joint_softmax=joint_softmax, dual=True,
                                      **prox_kws)

    # -- guide latents ------------------------------------------------------
    def analyse_guides(self, guides):
        """`[A w_g]` -- this layer's dictionary applied to each guide image."""
        return [self.analysis(w) for w in guides]

    # -- forward ------------------------------------------------------------
    def forward(self, state, y_tilde, guide=None, E=None, sigma=None, pi=None,
                cache=None):
        """One sweep.  `state` is `(x, z)`, or None for the cold start.

        `guide` is a list / stacked tensor of image-domain guides (or None, in
        which case the prox falls back to its unguided self-only form and this
        layer is an ordinary `LPDSLayer`).
        """
        if cache is None:
            cache = {}
        guides = as_guide_list(guide)

        hint = cache.pop("_gram_x", None)
        v = self.analyse_guides(guides) if guides else None

        if state is None:
            x = y_tilde
            z, cache = self.prox(self.analysis(x), v, sigma, cache)
            return (x, z), cache

        x, z = state
        pi_x, pi_z = (None, None) if pi is None else pi

        tau = self.tau(sigma, ref=x)
        theta = self.theta(sigma, ref=x)

        Ex = hint[1] if (hint is not None and hint[0] is x) else gram(E, x)
        residual = Ex - y_tilde + self.synthesis(z)
        if pi_x is not None:
            residual = residual - pi_x
        x_new = x - tau * residual

        x_bar = x_new + theta * (x_new - x)

        u = z + self.analysis(x_bar)
        if pi_z is not None:
            u = u - pi_z
        z_new, cache = self.prox(u, v, sigma, cache)

        return (x_new, z_new), cache


def make_guided_lpds_layer(C, M, spectral_init=True, **kws):
    """One `GuidedLPDSLayer`, filters tied (`B = A^H`) and spectrally scaled."""
    layer = GuidedLPDSLayer(C, M, **kws)
    layer.init_filters()
    if spectral_init:
        layer.spectral_normalize()
    return layer


# ---------------------------------------------------------------------------
class LGGSNet(nn.Module):
    """K untied guided primal-dual sweeps -- the network the LGGS runs use.

    Returns `(x_hat, (x, z))`, the same pair `MGLPDSNet` returns: for a
    primal-dual method the state IS the pair, and handing back `z` alone would
    make a warm start half a warm start.

    Parameters follow `config/guided_lpdsnet.yaml`, renamed to this repo's
    spelling (`p -> P`, `stride -> s`, `lambda0 -> lam0`, `windowsize -> window`,
    `guide_windowsize -> guide_window`, `Delta K -> dK`).
    """

    def __init__(self, K=30, M=169, C=1, P=7, s=2, Mh=64, window=1,
                 guide_window=15, joint_softmax=True, sim_fun="pidistance",
                 lam0=1e-3, tau0=0.5, theta0=0.0, degrees=0, nheads=1, dK=5,
                 rho0=1.0, gamma0=0.8, rho_inv=True,
                 init_strategy="semi_orthogonal", attn_backend="gather",
                 flex_block_size=128, is_complex=True, preproc="kspace",
                 spectral_init=True):
        super().__init__()
        self.K, self.M, self.C, self.P, self.s = int(K), int(M), int(C), int(P), int(s)
        self.is_complex = bool(is_complex)
        self.attn_backend = attn_backend
        if preproc not in ("image", "kspace", "identity"):
            raise ValueError(
                f"preproc must be 'image' (denoising / synthesis), 'kspace' "
                f"(reconstruction) or 'identity'; got {preproc!r}")
        self.preproc = preproc
        self.pad_stride = self.s

        prox_kws = dict(nheads=nheads, dK=dK, sim_fun=sim_fun, rho0=rho0,
                        gamma0=gamma0, rho_inv=rho_inv,
                        init_strategy=init_strategy, attn_backend=attn_backend,
                        flex_block_size=flex_block_size)
        layer_kws = dict(P=P, stride=s, lam0=lam0, tau0=tau0, theta0=theta0,
                         degrees=degrees, is_complex=is_complex, window=window,
                         guide_window=guide_window, joint_softmax=joint_softmax,
                         Mh=Mh, prox_kws=prox_kws)

        proto = make_guided_lpds_layer(C, M, spectral_init=spectral_init,
                                       **layer_kws)
        self.layers = nn.ModuleList(
            [proto] + [copy.deepcopy(proto) for _ in range(self.K - 1)])

    # -- guide preprocessing -------------------------------------------------
    def _prep_guides(self, guide, ref):
        """Mean-subtract each guide and pad it onto `y~`'s grid.

        `preproc.jl` runs the guide through `ImagePreprocess` regardless of the
        network's own preprocessing mode, with `resize_noise=false` and its OWN
        mean -- the guide is a different contrast, so borrowing the
        measurement's DC would be wrong.  The pad is computed from the guide's
        shape, which equals the target's, so both land on the same grid; we
        assert that rather than trusting it.
        """
        guides = as_guide_list(guide)
        if not guides:
            return []
        out = []
        pad = calc_pad_2d(*guides[0].shape[-2:], self.pad_stride)
        for w in guides:
            w = w.to(ref.dtype)
            w = w - w.mean(dim=(-3, -2, -1), keepdim=True)
            if any(pad):
                w = _pad_complex(w, pad)
            if w.shape[-2:] != ref.shape[-2:]:
                raise ValueError(
                    f"guide grid {tuple(w.shape[-2:])} does not match the "
                    f"preprocessed measurement grid {tuple(ref.shape[-2:])}; "
                    f"guides must be registered to the target and share its "
                    f"field of view.")
            out.append(w)
        return out

    # -- forward -------------------------------------------------------------
    def forward(self, y, guide=None, E=None, sigma=None, state=None):
        if E is None:
            E = Identity()

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

        guides = self._prep_guides(guide, y_tilde)

        cache = {}
        for layer in self.layers:
            state, cache = layer(state, y_tilde, guide=guides, E=E,
                                 sigma=sigma, cache=cache)
        x, z = state
        x_hat = x if params is None else post(x, params)
        return x_hat, (x, z)

    def _check_grid(self, hw):
        H, W = int(hw[0]), int(hw[1])
        if H % self.pad_stride or W % self.pad_stride:
            raise ValueError(
                f"{type(self).__name__} got a {H}x{W} input, not a multiple of "
                f"the dictionary stride s={self.s}. With preproc='identity' "
                f"nothing pads it, so the strided analysis / synthesis pair "
                f"would not land back on the input grid.")

    # -- hooks ---------------------------------------------------------------
    def compile_flex(self):
        """torch.compile every guided prox's fused kernel (GPU; call once)."""
        for m in self.modules():
            if isinstance(m, GuidedGroupThreshold) and m.attn_backend == "flex":
                m.compile_flex()
        return self

    @torch.no_grad()
    def project(self):
        for m in self.modules():
            if m is not self and hasattr(m, "project_"):
                m.project_()

    def extra_repr(self):
        return (f"K={self.K}, M={self.M}, C={self.C}, s={self.s}, "
                f"preproc={self.preproc}")


# Alias: the class is the port of `guided_lpds.jl`'s `GuidedLPDSNet`; `LGGSNet`
# is the name the experiments use for it.
GuidedLPDSNet = LGGSNet


def lggs_defaults(num_guides=1, **over):
    """The exact LGGS grid point from `scripts/makeconfigs_longi.jl`.

    `num_guides` is a DATA-side setting (how many prior studies the loader
    stacks), not a network hyper-parameter -- the same weights handle any
    number of guides, since every guide contributes one adjacency built from
    the shared W_phi.  It is returned here only so a config can carry it.
    """
    cfg = dict(K=30, M=169, C=1, P=7, s=2, Mh=64, window=1, guide_window=15,
               joint_softmax=True, sim_fun="pidistance", lam0=1e-3, tau0=0.5,
               theta0=0.0, dK=5, nheads=1, init_strategy="semi_orthogonal",
               is_complex=True, preproc="kspace")
    cfg.update(over)
    return cfg, int(num_guides)
