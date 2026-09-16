# -*- coding: utf-8 -*-
"""
SBGuidedGroupCDL -- the Schrodinger-bridge regressor with a GUIDED group-sparse prox.

Two existing nets, joined at exactly one seam:

    SBGroupCDL      the two-fidelity bridge unrolling (debiased residual r, mu_0-weighted target
                    fidelity, prior fidelity, learned eta/nu steps, bridge-adaptive threshold, the
                    analytic DC) -- models/sb_groupcdl.py
    LGGCDL          the guided group threshold, whose adjacency pools the target code's own
                    self-similarity with one extra branch per GUIDE image -- models/guided_cdl.py

Per layer k, with r the debiased bridge residual, c the DC-removed conditioning stack and w_g the
guide images:

    g_D = mu_0 * A_k( mu_0 * B_k z - r )                     target fidelity
    g_P =        A_P[k]( B_P[k] z - c )                      prior fidelity
    u   = z - eta_k g_D - nu_k g_P
    v_g = A_k w_g                                            THIS layer's dictionary on each guide
    z   = GT( u ; guides v_g, tau_k(s_hat), Gamma )          guided group threshold

and the readout is x0_hat = B_0 z + dc.

WHY THE GUIDE GOES IN THE ADJACENCY AND NOT THE INPUT. GT(u) = u * relu(1 - tau/xi) is a gain in
[0, 1] on the code the fidelities produced. A guide can only change WHICH pixels group together
(through xi); it can never write its own intensities into the estimate. That is the property a
longitudinal CT1 guide needs -- a prior scan's enhancement must not be pasted into today's
prediction -- and it is what concatenating the guide as an input channel would give up.

WHY THIS IS NOT A SUBCLASS OF EITHER. SBGroupCDL's shrinkage is GroupCDL._threshold with
GroupCDL._update_attention; the guided prox is models/prox.py::GroupThreshold, a separate
implementation with its own attention module. The two do not interoperate, so this net is built
from GuidedLISTALayers (for the guided prox and the target dictionary pair) plus the bridge pieces
of SBGroupCDL, reproduced here.

THE THRESHOLD FLOOR. GroupThreshold.project_ clamps tau's coefficients at 0, INCLUDING the constant
term. At tau = 0 the group threshold is the identity: xi -- the only place the adjacency enters --
drops out of the graph and Wtheta / Wphi / Walpha / Wbeta / gamma get exactly zero gradient. tau
itself still has gradient there and can climb back out, but whenever the loss pushes it DOWN the
projection holds it at 0 and the attention stays cut off. SBGroupCDL guards against this with a
floor on its threshold; `project()` here re-floors each layer's constant term at `t0` after the
prox's own projection. (Plain LGGCDL has the same exposure; it now has an opt-in `tau_floor`.)

Calling convention, so sb.base.predict_x0 and train_i2sb work:

    x0_hat, z = net(cat([x_t, cond]), E=Identity(), sigma=std_fwd, guide=guides)

`guide` is (B, G, 1, H, W) -- the stacked layout as_guide_list documents, and what the NYUMets
loader emits -- or (B, G, H, W), or a list of (B, 1, H, W). `guide=None` runs the
prox's self-only form, i.e. an ordinary (unguided) SB group-CDL: the guided/unguided ablation is a
single keyword. THE SCHEDULE PARAMETERS MUST MATCH cfg["i2sb"]; `assert_schedule_matches` checks.

Guides must share the target's grid. Same-session contrasts are co-registered by construction; a
guide from ANOTHER study is not, and relies on `guide_window` (a nonlocal search window, 15 by
default = +-7 latent = +-14 image pixels at stride 2) to absorb the misalignment.

PARAMETERS THAT NEVER RECEIVE GRADIENT -- both inherited, both harmless, both flat in param logs:
  * `layers[k].prox.rho` on every layer that does NOT refresh the adjacency. The adjacency is
    recomputed every `dK` layers and reused in between, so those layers' rho never enters the
    graph. At K=30, dK=5 that is most of the 30 rho tensors (each only Mh scalars). Plain LGGCDL
    shows the identical pattern. Tying rho across each refresh window would recover them.
  * `B_P[0]`: at k=0 the code is zero, so B_P[0](0) = 0 whatever its weights. SBGroupCDL has the
    same dead tensor. (The target pair's B_0 survives only because it is also the readout.)

MEMORY. attn_backend="gather" materializes every window position: one intermediate is
B x Mh x (H/s)(W/s) x guide_window^2 floats -- 450 MB at B=4, Mh=32, 128px, window 15 -- and one
exists per guide branch per refreshing layer, all held for backward. Batch size and Mh are the
knobs if this does not fit; joint_softmax=True rules out the fused flex/triton backends.
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import set_weight
from models.components import Conv2d, ConvTranspose2d
from models.guided_cdl import make_guided_lista_layer, tie_attention
from models.guided_prox import as_guide_list
from models.sb_schedule import BridgeScheduleMixin, horner as _horner
from operators.padding import unpad
from operators.projections import uball_project
from solvers.eigen import power_method


class SBGuidedGroupCDL(BridgeScheduleMixin, nn.Module):
    """Two-fidelity unrolled bridge regressor with a guided group-sparse prox.

    Parameters
    ----------
    K, M, P, s, Mh
        Unroll depth, atoms, filter size, stride, attention width -- as in LGGCDL.
    window, guide_window, joint_softmax, sim_fun, nheads, dK, rho0, gamma0, rho_inv,
    init_strategy, attn_backend, flex_block_size, share_attention
        The guided prox, as in GuidedGroupCDL. `joint_softmax=True` requires attn_backend="gather".
    C          input width = 1 + len(cond_idx); >= 2 because the bridge prior x_1 lives in cond.
    prior_idx  which CONDITIONING channel is x_1 (0-based within cond).
    t0         constant term of the threshold; MUST be > 0 (see the module docstring).
    deg_eta    degree of the step polynomials in s_log.
    deg_tau    degree of the threshold polynomial in s_hat = sigma_eff / max(sigma_eff).
    kind, tau, n_points, beta_max
               the bridge schedule -- MUST match cfg["i2sb"].
    """

    def __init__(self, K=30, M=169, C=2, P=7, s=2, Mh=64,
                 window=1, guide_window=15, joint_softmax=True, sim_fun="distance",
                 nheads=1, dK=5, rho0=1.0, gamma0=0.8, rho_inv=True,
                 init_strategy="semi_orthogonal", attn_backend="gather",
                 flex_block_size=128, share_attention=True,
                 prior_idx=0, t0=1e-3, deg_eta=0, deg_tau=1,
                 kind="brownian", tau=0.19, n_points=1000, beta_max=0.3,
                 spectral_init=True, init=None):
        super().__init__()
        if C < 2:
            raise ValueError(
                f"SBGuidedGroupCDL needs the bridge prior x_1 as a conditioning channel, so "
                f"C = 1 + len(cond_idx) >= 2; got C={C}.")
        n_cond = C - 1
        if not 0 <= prior_idx < n_cond:
            raise ValueError(
                f"prior_idx={prior_idx} out of range for {n_cond} conditioning channel(s); it "
                f"indexes cond (0-based), not the stored contrasts.")
        if float(t0) <= 0.0:
            raise ValueError(
                f"t0={t0} would make the group threshold the identity at init, cutting every "
                f"attention parameter off from the gradient. Use a small positive value (1e-3).")
        if init is not None:                       # accept SBGroupCDL's spelling
            spectral_init = bool(init)

        self.K, self.M, self.C, self.P, self.s = int(K), int(M), int(C), int(P), int(s)
        self.n_cond = n_cond
        self.prior_idx = int(prior_idx)
        self.t_floor = float(t0)
        self.cdtype = torch.float32

        # ---- target-domain layers: dictionary pair (A_k, B_k) + the guided prox ----
        # C=1: these ARE the target pair, so the readout is 1-channel and a 3-contrast conditioning
        # stack never widens it. tau0 = t0 and degrees = deg_tau put the bridge-adaptive threshold
        # inside each prox's own Polynomial, evaluated at s_hat.
        prox_kws = dict(nheads=nheads, dK=dK, sim_fun=sim_fun, rho0=rho0, gamma0=gamma0,
                        rho_inv=rho_inv, init_strategy=init_strategy,
                        attn_backend=attn_backend, flex_block_size=flex_block_size)
        proto = make_guided_lista_layer(
            1, M, spectral_init=spectral_init, P=P, stride=s, tau0=t0, degrees=deg_tau,
            is_complex=False, window=window, guide_window=guide_window,
            joint_softmax=joint_softmax, Mh=Mh, prox_kws=prox_kws)
        layers = [proto] + [copy.deepcopy(proto) for _ in range(self.K - 1)]
        if share_attention:
            layers = tie_attention(layers)
        self.layers = nn.ModuleList(layers)
        self.share_attention = bool(share_attention)

        # ---- prior-domain pair (the conditioning stack) ----
        self.A_P = nn.ModuleList([Conv2d(n_cond, M, P, stride=s, bias=False, complex=False)
                                  for _ in range(self.K)])
        self.B_P = nn.ModuleList([ConvTranspose2d(M, n_cond, P, stride=s, bias=False,
                                                  complex=False)
                                  for _ in range(self.K)])
        W = torch.randn(M, n_cond, P, P)
        for k in range(self.K):
            set_weight(self.A_P[k], W)
            set_weight(self.B_P[k], W.conj())
        if spectral_init:
            self._spectral_init_prior()

        # ---- bridge steps, identical parameterization to SBGroupCDL ----
        # sigmoid(0) = 0.5 on each of the two fidelities: the combined-step analogue of a single
        # unit ISTA step (each pair is spectrally normalized to a unit step).
        self.a_eta = nn.Parameter(torch.zeros(self.K, deg_eta + 1))
        self.a_nu = nn.Parameter(torch.zeros(self.K, deg_eta + 1))

        self._init_bridge_tables(kind=kind, tau=tau, n_points=n_points, beta_max=beta_max)

    # -----------------------------------------------------------------
    @torch.no_grad()
    def _spectral_init_prior(self):
        L = power_method(lambda x: self.B_P[0](self.A_P[0](x)),
                         torch.rand(1, self.n_cond, 128, 128),
                         num_iter=200, verbose=False)[0]
        scale = float(np.sqrt(np.abs(L)))
        for k in range(self.K):
            set_weight(self.A_P[k], self.A_P[k].weight / scale)
            set_weight(self.B_P[k], self.B_P[k].weight / scale)

    # -----------------------------------------------------------------
    def _prep_guides(self, guide, pad, ref):
        """Guides -> list of (B, 1, H', W'), each mean-subtracted and padded like `r`.

        (B, G, H, W) is split into G single-channel guides. It must NOT be handed to
        `as_guide_list` as-is: a 4-D tensor would be read as ONE guide with G channels, which the
        C=1 analysis conv would then reject -- or, worse, silently mis-read if G happened to be 1.
        """
        if guide is None:
            return []
        if torch.is_tensor(guide) and guide.dim() == 4:
            planes = [guide[:, g:g + 1] for g in range(guide.shape[1])]
        else:
            planes = as_guide_list(guide)
        out = []
        for w in planes:
            if w.dim() != 4 or w.shape[1] != 1:
                raise ValueError(f"each guide must be (B, 1, H, W); got {tuple(w.shape)}")
            w = w.to(ref.dtype)
            w = w - w.mean(dim=(1, 2, 3), keepdim=True)          # its OWN DC, never the target's
            w = F.pad(w, pad, mode="reflect")                     # same geometry as r and c
            if w.shape[-2:] != ref.shape[-2:]:
                raise ValueError(
                    f"guide grid {tuple(w.shape[-2:])} != bridge grid {tuple(ref.shape[-2:])}: "
                    f"guides must share the target's field of view (same crop, same size).")
            out.append(w)
        return out

    # -----------------------------------------------------------------
    def forward(self, y, E=None, sigma=None, step=None, guide=None):
        """`y = cat([x_t, cond])` as sb.base.predict_x0 builds it. `E` is ignored. -> (x0_hat, z)"""
        r, c, dc, pad, mu0, s_log, s_hat = self.bridge_inputs(y, sigma=sigma, step=step)
        guides = self._prep_guides(guide, pad, r)

        z = torch.zeros_like(self.layers[0].analysis(r))
        cache = {}
        for k, layer in enumerate(self.layers):
            eta = torch.sigmoid(_horner(self.a_eta[k], s_log))
            nu = torch.sigmoid(_horner(self.a_nu[k], s_log))

            g_D = mu0 * layer.analysis(mu0 * layer.synthesis(z) - r)
            g_P = self.A_P[k](self.B_P[k](z) - c)
            u = z - eta * g_D - nu * g_P

            v = layer.analyse_guides(guides) if guides else None
            z, cache = layer.prox(u, v, s_hat, cache)

        x0_hat = unpad(self.layers[0].synthesis(z), pad) + dc
        return x0_hat, z

    # -----------------------------------------------------------------
    @torch.no_grad()
    def project(self):
        for layer in self.layers:
            layer.project_()               # unit-ball filters + the prox's tau/gamma/rho/Wbeta
            # re-floor the threshold's CONSTANT term: the prox clamps it at 0, and 0 kills the
            # attention gradient permanently (module docstring).
            layer.prox.tau.weight.data[0].clamp_(min=self.t_floor)
        for k in range(self.K):
            set_weight(self.A_P[k], uball_project(self.A_P[k].weight))
            set_weight(self.B_P[k], uball_project(self.B_P[k].weight))

    @torch.no_grad()
    def param_logs(self, probes=(0.0, 0.5, 1.0)):
        """Derived step sizes and threshold at a few bridge positions (see BridgeScheduleMixin)."""
        return self._sb_param_logs({
            "eta": lambda sl, sh: [torch.sigmoid(_horner(self.a_eta[j], sl)) for j in range(self.K)],
            "nu": lambda sl, sh: [torch.sigmoid(_horner(self.a_nu[j], sl)) for j in range(self.K)],
            "tau": lambda sl, sh: [self.layers[j].prox.tau(sh, ref=sh) for j in range(self.K)],
        }, probes=probes)

    def extra_repr(self):
        return (f"K={self.K}, M={self.M}, C={self.C}, s={self.s}, n_cond={self.n_cond}, "
                f"prior_idx={self.prior_idx}, share_attention={self.share_attention}")
