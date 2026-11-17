"""
Joint LGGS -- reconstruct the target AND an equally accelerated guide together (2026-11-17).

Published LGGS needs a fully-sampled guide (or a second network that reconstructs it first).
Here both images are unknowns, each with its own data term, and ONE group threshold acts on the
codes of both at once:

    min_{x, g}  1/2 ||E_t x - y_t||^2 + 1/2 ||E_g g - y_g||^2 + R(A_t x, A_g g; Gamma)

Stack them, X = [x; g], so the problem is a single-image LPDS problem on a 2-channel image:

    E = blockdiag(E_t, E_g)        A = blockdiag(A_t, A_g)  (a grouped conv, groups = 2)

so the per-contrast dictionaries are UNTIED (each contrast its own analysis / synthesis convs,
which absorb the contrast mismatch) while the prox sees all 2M code channels together:

    GroupThreshold on [z; c]   (models/prox.py; GroupCDL Eq. 11)
      q = W_theta [z; c],  k = W_phi [z; c]       one JOINT grouping: pixels group when they
                                                  look alike in both contrasts
      xi = W_beta sqrt( Gamma (W_alpha [z; c])^2 )  energies mixed across contrasts by the
                                                  learned W_alpha, mapped back per contrast
      [z; c] <- [z; c] * min(1, tau / xi)          a threshold PER CHANNEL, so per contrast

THE WARM-UP. The first `K0` layers use a plain soft threshold instead (window 1): with the
grouped convs that is two INDEPENDENT LPDS reconstructions -- no grouping, no cross-talk -- while
both images are still aliased. The remaining K - K0 layers group jointly. LGGS is the asymmetric
special case (guide given, queries from the target only); K0 = K is two independent LPDSNets.

The guide's operator is the target's with the mask replaced by the guide's own sampled lines
(read off the side k-space), so it shares the target's coil maps: same session, same coils. The
guide arrives as the side k-space (`guide_domain = "kspace"`), like TGVN's. The output is the
target; the guide reconstruction comes back in the second return value.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from models.base import batched_projection, set_weight
from models.components import Conv2d, ConvTranspose2d
from models.guided_lpds import tie_transforms_
from models.lpds import LPDSLayer
from operators.accessors import get_mask, set_mask
from preprocessing.kspace import kspace_post_process, kspace_pre_process


class _StackedGram:
    """The normal map of blockdiag(E_t, E_g) on a stacked (B, 2, H, W) image."""

    nonlinear = False

    def __init__(self, ops):
        self.ops = list(ops)

    def gram(self, X):
        return torch.cat([E.gram(X[:, i:i + 1]) for i, E in enumerate(self.ops)], dim=1)


class JointLPDSLayer(LPDSLayer):
    """One primal-dual sweep on the stacked (target, guide) image with PER-CONTRAST convs.

    `LPDSLayer(C=2, M=2M)` with its analysis / synthesis replaced by grouped convs (groups = 2):
    codes 0..M-1 come from the target alone, M..2M-1 from the guide alone. The prox (soft
    threshold, or the group threshold for window > 1) and the per-channel steps tau, theta are
    LPDSLayer's own, on 2M code / 2 image channels."""

    def __init__(self, M, P=7, stride=2, lam0=1e-3, tau0=0.5, theta0=0.0, degrees=0,
                 window=1, Mh=None, prox_kws=None):
        super().__init__(2, 2 * M, P=P, stride=stride, lam0=lam0, tau0=tau0, theta0=theta0,
                         degrees=degrees, is_complex=True, window=window, Mh=Mh,
                         prox_kws=prox_kws)
        self.M_c = int(M)
        self.analysis = Conv2d(2, 2 * M, P, stride=stride, bias=False, complex=True, groups=2)
        self.synthesis = ConvTranspose2d(2 * M, 2, P, stride=stride, bias=False, complex=True,
                                         groups=2)

    @torch.no_grad()
    def init_filters(self, weight=None):
        """B = A^H, and both contrasts start from the SAME dictionary (training unties them)."""
        if weight is None:
            weight = torch.randn(self.M_c, 1, self.P, self.P, dtype=torch.complex64)
        w = torch.cat([weight, weight], dim=0)                     # (2M, 1, P, P)
        set_weight(self.analysis, w)
        set_weight(self.synthesis, w.conj())
        return weight


class JointLGGSNet(nn.Module):
    """K sweeps on the stacked (target, guide) image: K0 independent-LPDS warm-up layers, then
    K - K0 joint group-threshold layers. Returns `(x_hat, {"guide_recon": g_hat})`."""

    guided = True
    guide_domain = "kspace"          # the loops pass the (accelerated) side k-space

    def __init__(self, K=30, K0=10, M=169, P=7, s=2, Mh=64, window=15, lam0=1e-3, tau0=0.5,
                 theta0=0.0, degrees=0, prox_degrees=1, dK=5, nheads=1, sim_fun="pidistance",
                 rho0=1.0, gamma0=0.8, rho_inv=True, init_strategy="semi_orthogonal",
                 attn_backend="gather", flex_block_size=128, spectral_init=True):
        super().__init__()
        K, K0 = int(K), int(K0)
        if not 0 <= K0 <= K:
            raise ValueError(f"need 0 <= K0 <= K, got K0={K0}, K={K}")
        if K0 < K and (int(window) <= 1 or int(window) % 2 == 0 or Mh is None):
            raise ValueError("the joint layers need an odd window > 1 and Mh (the attention "
                             f"channel count); got window={window}, Mh={Mh}")
        self.K, self.K0, self.M, self.s = K, K0, int(M), int(s)
        self.pad_stride = self.s
        common = dict(P=P, stride=s, lam0=lam0, tau0=tau0, theta0=theta0, degrees=degrees)
        warm = JointLPDSLayer(M, window=1, prox_kws={"degrees": prox_degrees}, **common)
        weight = warm.init_filters()
        if spectral_init:
            warm.spectral_normalize()
        layers = [copy.deepcopy(warm) for _ in range(K0)]
        if K0 < K:
            group = dict(degrees=prox_degrees, nheads=nheads, dK=dK, sim_fun=sim_fun, rho0=rho0,
                         gamma0=gamma0, rho_inv=rho_inv, init_strategy=init_strategy,
                         attn_backend=attn_backend, flex_block_size=flex_block_size)
            joint = JointLPDSLayer(M, window=window, Mh=Mh, prox_kws=group, **common)
            joint.init_filters(weight)                         # every layer starts identical
            set_weight(joint.analysis, warm.analysis.weight)
            set_weight(joint.synthesis, warm.synthesis.weight)
            jl = [joint] + [copy.deepcopy(joint) for _ in range(K - K0 - 1)]
            tie_transforms_(jl)            # W_theta, W_phi, W_alpha, W_beta, gamma shared, as LGGS
            layers += jl
        self.layers = nn.ModuleList(layers)

    # -- the guide's operator ---------------------------------------------------------------
    @staticmethod
    def guide_operator(E, side):
        """E with its mask replaced by the guide's sampled phase-encode lines (any nonzero
        sample in a column, over coils and readout)."""
        m = get_mask(E)
        cols = side.abs().sum(dim=1, keepdim=True).sum(dim=-2, keepdim=True) > 0   # (B,1,1,W)
        return set_mask(E, cols.to(m.dtype).expand(-1, 1, side.shape[-2], -1))

    def forward(self, y, E=None, sigma=None, guide=None, state=None):
        if E is None or guide is None:
            raise ValueError("JointLGGSNet needs the operator E and the side k-space `guide`")
        if guide.dim() == 5:
            if guide.shape[1] != 1:
                raise ValueError(f"JointLGGSNet takes one side scan, got {guide.shape[1]}")
            guide = guide[:, 0]
        if guide.shape != y.shape:
            raise ValueError(f"side k-space {tuple(guide.shape)} must match the main k-space "
                             f"{tuple(y.shape)}: the guide shares the target's coil maps")
        E_g = self.guide_operator(E, guide)
        yt, Et, pt = kspace_pre_process(y, E, self.pad_stride)
        yg, Eg, pg = kspace_pre_process(guide, E_g, self.pad_stride)
        Y = torch.cat([yt, yg], dim=1)
        Es = _StackedGram([Et, Eg])
        cache = {}
        for layer in self.layers:
            state, cache = layer(state, Y, E=Es, sigma=sigma, cache=cache)
        X, _ = state
        x_hat = kspace_post_process(X[:, :1], list(pt))
        g_hat = kspace_post_process(X[:, 1:2], list(pg))
        return x_hat, {"guide_recon": g_hat}

    @torch.no_grad()
    def project(self):
        with batched_projection():
            for layer in self.layers:
                layer.project_()
        return self

    def extra_repr(self):
        return f"K={self.K}, K0={self.K0} (LPDS warm-up), M={self.M} per contrast, s={self.s}"
