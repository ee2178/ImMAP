"""
Guided GroupCDL -- the ISTA-unrolled sibling of LGGS, for REAL-valued data.

`models/guided_lpds.py` puts the guided group threshold in an LPDS dual step,
which applies its FENCHEL conjugate: `prox_{g*}(z) = z - prox_g(z)`, i.e.
clipping.  For soft-thresholding that map is `z min(1, tau/|z|)`, whose modulus
is pinned at `tau` for every `|z| > tau` -- so d|out|/d|z| = 0 there, and the
group version saturates the same way.  On complex data the phase still carries
gradient through that region and the unrolling trains; on real data it does not,
and the dual iterate stops learning wherever it is active.

This module keeps the guided prox and swaps the algorithm underneath it:
proximal gradient descent (ISTA) on the convolutional BPDN problem, where the
prox is applied DIRECTLY as shrinkage,

    z^(k+1) = GT_tau( z^(k) - A^(k)( E^H E (B^(k) z^(k)) - y~ ) ;  Phi, Omega_g )
    x_hat   = D z^(K) + mu

`GT(z) = z relu(1 - tau/xi)` has gain -> 1 as `xi >> tau`, so large coefficients
keep unit gradient.  That is the whole reason this class exists, and it is why
`CDLNet` / `GroupCDL` are the real-valued members of this family.

Everything guide-related is unchanged and shared with LGGS: the same
`GuidedGroupThreshold` builds `Phi` from the latent to itself and one `Omega_g`
per guide, joint-softmaxes them, and pools both branches into one group energy.
The guide still enters ONLY through the adjacency, so a prior fully-sampled CT1
shapes the grouping without contributing its intensities.

Differences from the LGGS defaults, and why
-------------------------------------------
* `is_complex=False`.  This class exists for real data; a complex one should use
  `LGGSNet`, whose primal-dual iteration has the better-conditioned data term.
* `sim_fun="distance"` rather than `"pidistance"`.  Phase invariance is what
  `pidistance` buys, and real features have no phase.  It is still accepted and
  still meaningful on real data -- it reduces to a SIGN-invariant distance, so
  it is worth trying when the guide is a different contrast in which a feature
  can invert (a lesion that is bright on the guide and dark on the target).
* `share_attention=True`.  `config/guided_groupcdl.yaml` shares
  `W_alpha/W_beta`, `W_theta/W_phi` and `gamma` across layers (`share: ab, tp,
  g = true`) while leaving `rho` and the thresholds per-layer.  Untying the
  attention transforms across K=30 layers multiplies their parameter count by
  30 for no reported gain.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from models.base import set_weight
from models.components import ConvTranspose2d
from models.guided_prox import (GuidedGroupThreshold, as_guide_list,
                                build_guided_prox)
from models.lista import LISTALayer, gram
from models.prox import resize_noise as _resize_noise
from operators.identity import Identity
from operators.padding import calc_pad_2d
from operators.projections import uball_project
from preprocessing.image import post_process, pre_process
from preprocessing.kspace import (_pad_complex, kspace_post_process,
                                  kspace_pre_process)


# ---------------------------------------------------------------------------
class GuidedLISTALayer(LISTALayer):
    """One unrolled proximal-gradient step whose prox is guided.

    Identical to `LISTALayer` except for the prox slot and the extra `guide`
    argument threaded to it.  `guide` is in the IMAGE domain; this layer applies
    its own `analysis` to it, the same convention `GuidedLPDSLayer` uses.

    Note `dual=False`: the prox is the shrinkage map itself, not its conjugate.
    That is the one line that separates this from the LPDS layer, and it is the
    line that matters for real data.
    """

    def __init__(self, C, M, P=7, stride=1, tau0=1e-2, degrees=0,
                 is_complex=False, multigrid=False, eta0=1e-1, eta_degrees=0,
                 window=1, guide_window=None, joint_softmax=True, Mh=None,
                 prox_kws=None):
        # Build the plain layer first (convs, eta, and a prox we replace) so the
        # analysis / synthesis construction and init stay in one place.
        super().__init__(C, M, P=P, stride=stride, tau0=tau0, degrees=degrees,
                         is_complex=is_complex, multigrid=multigrid, eta0=eta0,
                         eta_degrees=eta_degrees, window=1, Mh=None, dual=False,
                         prox_kws=None)

        prox_kws = dict(prox_kws or {})
        prox_kws.setdefault("tau0", tau0)
        prox_kws.setdefault("degrees", degrees)
        self.prox = build_guided_prox(M, Mh=Mh, window=window,
                                      guide_window=guide_window,
                                      joint_softmax=joint_softmax, dual=False,
                                      **prox_kws)

    # -- guide latents ------------------------------------------------------
    def analyse_guides(self, guides):
        """`[A w_g]` -- this layer's dictionary applied to each guide image."""
        return [self.analysis(w) for w in guides]

    # -- forward ------------------------------------------------------------
    def forward(self, z, y_tilde, guide=None, E=None, sigma=None, pi=None,
                cache=None):
        guides = as_guide_list(guide)
        v = self.analyse_guides(guides) if guides else None

        if z is None:                                  # cold start, z^(0) = 0
            return self.prox(self.analysis(y_tilde), v, sigma, cache)

        residual = gram(E, self.synthesis(z)) - y_tilde
        u = z - self.analysis(residual)
        if pi is not None and self.eta is not None:
            u = u + self.eta(sigma, ref=z) * pi
        return self.prox(u, v, sigma, cache)

    @torch.no_grad()
    def project_(self):
        super().project_()
        self.prox.project_()


def make_guided_lista_layer(C, M, spectral_init=True, **kws):
    """One `GuidedLISTALayer`, filters tied (`B = A^H`) and spectrally scaled."""
    layer = GuidedLISTALayer(C, M, **kws)
    layer.init_filters()
    if spectral_init:
        layer.spectral_normalize()
    return layer


# ---------------------------------------------------------------------------
def tie_attention(layers):
    """Point every layer's attention transforms at layer 0's.

    `config/guided_groupcdl.yaml`'s `share:` block, spelled in torch: assigning
    a submodule makes the parameter literally the same object, so the optimiser
    sees one copy and `named_parameters()` deduplicates it.

    `rho` and the thresholds stay per-layer -- the yaml shares neither, and both
    are the noise-adaptive knobs an unrolling is supposed to vary with depth.
    """
    src = layers[0].prox
    if not src.grouped:
        return layers
    for layer in layers[1:]:
        dst = layer.prox
        dst.Wtheta, dst.Wphi = src.Wtheta, src.Wphi
        dst.Walpha, dst.Wbeta = src.Walpha, src.Wbeta
        dst.gamma = src.gamma
    return layers


class GuidedGroupCDL(nn.Module):
    """K untied guided ISTA iterations, with a shared attention front-end.

    Returns `(x_hat, z)` -- the same pair `CDLNet` / `GroupCDL` / `MGCDLNet`
    return, so it drops into the existing training and evaluation loops.

    `forward(y, guide=None, ...)`: with `guide=None` the prox falls back to its
    unguided self-only form and this is an ordinary GroupCDL, which makes the
    guided / unguided ablation a single keyword.
    """

    def __init__(self, K=30, M=169, C=1, P=7, s=2, Mh=64, window=1,
                 guide_window=15, joint_softmax=True, sim_fun="distance",
                 tau0=1e-3, degrees=0, nheads=1, dK=5, rho0=1.0, gamma0=0.8,
                 rho_inv=True, init_strategy="semi_orthogonal",
                 attn_backend="gather", flex_block_size=128, is_complex=False,
                 preproc="image", resize_noise=False, share_attention=True,
                 spectral_init=True):
        super().__init__()
        self.K, self.M, self.C = int(K), int(M), int(C)
        self.P, self.s = int(P), int(s)
        self.is_complex = bool(is_complex)
        self.attn_backend = attn_backend
        self.resize_noise = bool(resize_noise)
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
        layer_kws = dict(P=P, stride=s, tau0=tau0, degrees=degrees,
                         is_complex=is_complex, window=window,
                         guide_window=guide_window, joint_softmax=joint_softmax,
                         Mh=Mh, prox_kws=prox_kws)

        proto = make_guided_lista_layer(C, M, spectral_init=spectral_init,
                                        **layer_kws)
        layers = [proto] + [copy.deepcopy(proto) for _ in range(self.K - 1)]
        if share_attention:
            layers = tie_attention(layers)
        self.layers = nn.ModuleList(layers)
        self.share_attention = bool(share_attention)

        # Read-out dictionary, seeded from the first layer's synthesis (the
        # convention `MGCDLNet` uses).  It is a separate operator because layer
        # 0 takes the cold-start shortcut and never applies its own synthesis.
        self.D = ConvTranspose2d(M, C, P, stride=s, bias=False,
                                 complex=is_complex)
        set_weight(self.D, proto.synthesis.weight)

    # -- guide preprocessing -------------------------------------------------
    def _prep_guides(self, guide, ref):
        """Mean-subtract each guide and pad it onto `y~`'s grid.

        `preproc.jl` runs the guide through `ImagePreprocess` whatever the
        network's own preprocessing mode, with its OWN mean -- the guide is a
        different contrast, so borrowing the measurement's DC would be wrong.
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
    def forward(self, y, guide=None, E=None, sigma=None, z0=None):
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

        sig = sigma
        if self.resize_noise and torch.is_tensor(sigma) and sigma.dim() == 4:
            latent = (y_tilde.shape[-2] // self.s, y_tilde.shape[-1] // self.s)
            sig = _resize_noise(sigma, latent)

        guides = self._prep_guides(guide, y_tilde)

        z, cache = z0, {}
        for layer in self.layers:
            z, cache = layer(z, y_tilde, guide=guides, E=E, sigma=sig,
                             cache=cache)

        x = self.D(z)
        x_hat = x if params is None else post(x, params)
        return x_hat, z

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
        for layer in self.layers:
            layer.project_()
        set_weight(self.D, uball_project(self.D.weight))

    def extra_repr(self):
        return (f"K={self.K}, M={self.M}, C={self.C}, s={self.s}, "
                f"preproc={self.preproc}, share_attention={self.share_attention}")


# The experiments' name for LGGS's ISTA sibling.
LGGCDLNet = GuidedGroupCDL
