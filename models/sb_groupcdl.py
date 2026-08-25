# -*- coding: utf-8 -*-
"""
SBGroupCDL -- SBCDLNet with a GROUP-SPARSITY prior.

SBCDLNet unrolls the two-fidelity Schrodinger-bridge problem

    J_k(z) = 1/2 ||r - mu_0 D_D z||^2 + gamma/2 ||c - D_P z||^2 + sum_m lambda_m ||z_m||_1

under a plain weighted l1, whose prox is the soft threshold. This net changes ONE thing: the
prior. Replacing the separable l1 with GroupCDL's nonlocal group-sparsity penalty

    sum_q lambda_q || (I (x) Gamma)^{1/2} W_alpha^T z ||_2      (Janjusevic et al., Eq. 11)

makes the prox the GROUP threshold -- a joint shrinkage in which pixels deemed similar by a
learned circulant adjacency Gamma share one magnitude, so an atom that is active at one pixel is
encouraged to be active at its nonlocal neighbours too. Everything else is untouched:

    DATA terms   identical. They come from the bridge, not the prior, so the debiased residual
                 r = x_t - mu_1 x_1, the mu_0-weighted target gradient, the separate learned
                 steps (eta_k, nu_k) and the analytic DC are inherited verbatim from
                 BridgeScheduleMixin.
    PROX         ST(u; tau)  ->  GT(u; tau, Gamma).
    THRESHOLD    still the bridge-adaptive polynomial in sigma_eff (SBCDLNet's `t`), NOT
                 GroupCDL's tau0 + sigma*tau1. The bridge's effective noise sigma_eff = sigma_sb
                 / mu_0 spans ~3 decades, and it -- not the raw std_fwd -- is what the shrinkage
                 has to track; deg_tau=1 recovers GroupCDL's affine form as a special case.

Per layer k, with r the debiased bridge residual and c the (DC-removed) conditioning stack:

    Gamma^(k) = AdjUpdate(Gamma^(k-1), z)                    # every dK layers, GroupCDL Alg. 4
    g_D = mu_0 * A_D[k]( mu_0 * B_D[k] z - r )               # target fidelity
    g_P =         A_P[k]( B_P[k] z - c )                     # prior-contrast fidelity
    z  <- GT( z - eta_k * g_D - nu_k * g_P ; tau_k, Gamma^(k) )

and the readout is x0_hat = B_D[0] z + dc.

WHY GROUP SPARSITY SHOULD HELP HERE SPECIFICALLY. The thing being synthesized is enhancement --
a spatially coherent, repetitive structure that occupies ~0.3% of a slice. A separable l1
thresholds each pixel's code independently, so at the prior end (mu_0 -> 0, sigma_eff -> 14) the
target fidelity has vanished and l1 has nothing left to tie one tumour pixel's code to its
neighbours'. The nonlocal adjacency does exactly that tying, and it is built from z, which is
still informative there because the PRIOR fidelity never weakens.

INHERITANCE. This subclasses GroupCDL to reuse the attention machinery (Wtheta/Wphi/Walpha/Wbeta,
the gather and flex backends, the block-mask cache, `_threshold`, `compile_flex`) unchanged, and
BridgeScheduleMixin for the schedule tables and the sigma -> bridge-position lookup. GroupCDL is
constructed with C=1 so its A/B ARE the target-domain pair (which also makes
`visualization/filters.py` render the readout dictionary with no extra work); the prior-domain
pair A_P/B_P is added on top, and `A_D`/`B_D` are read-only aliases of A/B so the forward reads
like SBCDLNet's. GroupCDL's own tau0/tau1 are removed -- the polynomial `t` replaces them, and
leaving dead parameters behind would show up in every param/grad log.

Calling convention matches the rest of the repo's denoisers so sb.base.predict_x0 works unchanged:

    x0_hat, z = net(y, E=Identity(), sigma=std_fwd)

with `y = cat([x_t, cond])`. `E` is accepted and ignored (pure translation has no forward
operator). THE SCHEDULE PARAMETERS HERE MUST MATCH cfg["i2sb"] (kind / tau / n_points /
beta_max) -- `assert_schedule_matches` checks that, and train_i2sb calls it at startup.

REQUIRES a conditioning channel: the bridge prior x_1 must be in `cond` (set the loader's
`cond_idx` to include it and point `prior_idx` at its position), so C = 1 + len(cond_idx) >= 2.
"""

import numpy as np
import torch
import torch.nn as nn

from models.base import set_weight
from models.components import Conv2d, ConvTranspose2d
from models.groupcdl import GroupCDL
from models.sb_schedule import BridgeScheduleMixin, horner as _horner
from operators.padding import unpad
from operators.projections import uball_project
from solvers.eigen import power_method


class SBGroupCDL(BridgeScheduleMixin, GroupCDL):
    """Two-fidelity unrolled proximal gradient for the Schrodinger bridge, group-sparse prior.

    Parameters
    ----------
    M, Mh, P, sc, K, W, dK, sim_fun, eps, attn_backend, blend, flex_block_size, is_complex,
    fenchel
        As in GroupCDL -- the dictionary/attention geometry and the prox variant. `sc` is the
        stride (SBCDLNet calls the same thing `s`; both names are accepted here).
    C          input width, = 1 + len(cond_idx). Must be >= 2 (the prior lives in cond).
    prior_idx  which CONDITIONING channel is the bridge prior x_1 (0-based within cond).
               With cond_idx=[0,1,3] = [FLAIR,T1,T2] and x1_idx=1 (T1), this is 1.
    t0         constant term of the threshold polynomial. Defaults to GroupCDL's 1e-3 rather
               than CDLNet's 0, and MUST stay > 0 -- see the note in __init__.
    deg_eta    degree of the step polynomials in s = log(sigma_eff), remapped to [-1, 1]. 0 = a
               plain learned scalar per layer (the recommended starting point).
    deg_tau    degree of the threshold polynomial in the normalized sigma_eff. 1 reproduces
               GroupCDL's affine-in-sigma threshold; 2 additionally spans the sigma^2 law.
    kind, tau, n_points, beta_max
               the bridge schedule -- MUST match cfg["i2sb"].
    """

    def __init__(self, M=169, Mh=64, C=2, P=7, sc=2, K=30, W=35, dK=5,
                 prior_idx=0, t0=1e-3, deg_eta=0, deg_tau=1,
                 kind="brownian", tau=0.19, n_points=1000, beta_max=0.3,
                 sim_fun="distance", eps=1e-6, attn_backend="gather", blend=True,
                 flex_block_size=128, is_complex=False, fenchel=False, init=True,
                 s=None):
        if C < 2:
            raise ValueError(
                f"SBGroupCDL needs the bridge prior x_1 as a conditioning channel, so "
                f"C = 1 + len(cond_idx) >= 2; got C={C}. Set the loader's cond_idx to include "
                f"the contrast used as x1_idx.")
        if s is not None:                      # accept SBCDLNet's name for the stride
            sc = s
        n_cond = C - 1
        if not 0 <= prior_idx < n_cond:
            raise ValueError(
                f"prior_idx={prior_idx} out of range for {n_cond} conditioning channel(s). "
                f"It indexes cond (0-based), not the stored contrasts.")

        # Build GroupCDL with C=1 so its A/B are the TARGET-domain pair. init=False: our
        # spectral_init covers both pairs and cannot run before A_P/B_P exist.
        super().__init__(M=M, Mh=Mh, C=1, P=P, sc=sc, K=K, W=W, dK=dK,
                         sim_fun=sim_fun, eps=eps, attn_backend=attn_backend, blend=blend,
                         flex_block_size=flex_block_size, is_complex=is_complex,
                         fenchel=fenchel, init=False)

        # GroupCDL set self.C = 1 (the target width). Restore the real INPUT width, and record
        # the pieces BridgeScheduleMixin.bridge_inputs needs.
        self.C = C
        self.n_cond = n_cond
        self.prior_idx = int(prior_idx)
        self.s = sc                            # the mixin pads to a multiple of the stride

        # ---- prior-domain dictionary pair (the target pair is GroupCDL's A / B) ----
        self.A_P = nn.ModuleList([
            Conv2d(n_cond, M, P, stride=sc, bias=False, complex=self.complex)
            for _ in range(K)])
        self.B_P = nn.ModuleList([
            ConvTranspose2d(M, n_cond, P, stride=sc, bias=False, complex=self.complex)
            for _ in range(K)])

        # ---- learned coefficients (identical parameterization to SBCDLNet) ----
        # Steps are sigmoid-squashed, so they live in (0,1) and cannot destabilize the iteration.
        # Init at pre-activation 0 -> eta = nu = 0.5: spectral_init makes each pair's proximal
        # step 1, so a HALF step on each of the two fidelities is the combined-step analogue of
        # GroupCDL's single unit step. Starting both at ~1 would double it at mu_0 = 1.
        self.a_eta = nn.Parameter(torch.zeros(K, deg_eta + 1))
        self.a_nu = nn.Parameter(torch.zeros(K, deg_eta + 1))
        # Per-atom threshold, shaped like CDLNet's t = (K, deg+1, M, 1, 1).
        #
        # t0 MUST BE > 0 HERE, unlike SBCDLNet. The group threshold is
        #     factor = clamp(1 - tau / xi, min=0),  xi = W_beta sqrt(sqrt((I (x) Gamma)|W_alpha^T u|^2))
        # so at tau = 0 the factor is identically 1: the prox is the IDENTITY and xi -- the only
        # place the adjacency enters -- drops out of the graph entirely. d(factor)/d(xi) = tau/xi^2
        # is then 0, so Wtheta / Wphi / Walpha / Wbeta / gamma all receive EXACTLY zero gradient
        # and the whole attention block never starts learning. (The soft threshold has no such
        # problem: d(ST)/d(tau) = -sgn(u) is nonzero at tau = 0, which is why SBCDLNet can start
        # there.) GroupCDL initializes tau0 = 1e-3 for the same reason; we match it.
        if float(t0) <= 0.0:
            raise ValueError(
                f"t0={t0} would make the group threshold the identity at init, cutting every "
                f"attention parameter (Wtheta/Wphi/Walpha/Wbeta/gamma) off from the gradient. "
                f"Use a small positive value (GroupCDL's default is 1e-3).")
        t = torch.zeros(K, deg_tau + 1, M, 1, 1)
        t[:, 0] = float(t0)
        self.t = nn.Parameter(t)

        # GroupCDL's affine-in-sigma threshold is superseded by `t`; drop it rather than leave
        # dead parameters that every param/grad log would still walk.
        del self.tau0
        del self.tau1

        self.t_floor = float(t0)      # project() keeps the constant term at or above this

        self._init_bridge_tables(kind=kind, tau=tau, n_points=n_points, beta_max=beta_max)

        self.init_filters(dtype=self.cdtype)
        if init:
            self.spectral_init()

    # Readable aliases so the forward matches SBCDLNet's. Properties, not assignments: assigning
    # a ModuleList to a second attribute would register it twice in the state_dict.
    @property
    def A_D(self):
        return self.A

    @property
    def B_D(self):
        return self.B

    # -----------------------------------------------------------------
    # initialization / projection, run once per dictionary pair
    # -----------------------------------------------------------------
    def _init_pair(self, A, B, cin, dtype):
        W = torch.randn(self.M, cin, self.P, self.P, dtype=dtype)
        for k in range(self.K):
            set_weight(A[k], W)
            set_weight(B[k], W.conj())

    def init_filters(self, dtype=torch.cfloat):
        self._init_pair(self.A, self.B, 1, dtype)
        # GroupCDL.__init__ calls this before A_P exists; skip the prior pair on that first pass.
        if hasattr(self, "A_P"):
            self._init_pair(self.A_P, self.B_P, self.n_cond, dtype)

    @torch.no_grad()
    def _spectral_init_pair(self, A, B, cin):
        """Scale (A, B) so ||B A||_2 = 1, i.e. that pair's proximal step is 1 -- run per pair so
        eta and nu are both interpretable as a FRACTION of an exact unit step."""
        L = power_method(
            lambda x: B[0](A[0](x)),
            torch.rand(1, cin, 128, 128, dtype=A[0].weight.dtype),
            num_iter=200, verbose=False,
        )[0]
        scale = np.sqrt(np.abs(L))
        for k in range(self.K):
            set_weight(A[k], A[k].weight / scale)
            set_weight(B[k], B[k].weight / scale)

    @torch.no_grad()
    def spectral_init(self):
        self._spectral_init_pair(self.A, self.B, 1)
        self._spectral_init_pair(self.A_P, self.B_P, self.n_cond)

    @torch.no_grad()
    def project_filters(self):
        for A, B in ((self.A, self.B), (self.A_P, self.B_P)):
            for k in range(self.K):
                set_weight(A[k], uball_project(A[k].weight))
                set_weight(B[k], uball_project(B[k].weight))

    @torch.no_grad()
    def project(self):
        # Nonnegative threshold coefficients keep tau >= 0 AND nondecreasing in sigma_eff (more
        # noise -> more shrinkage), the constraint CDLNet imposes with t.clamp_(0.). The gamma /
        # Wbeta constraints are GroupCDL's (Eq. 16) and are unchanged.
        # clamp_min at t0's floor, not 0: dropping the constant term to exactly 0 would put the
        # attention block back in the zero-gradient regime described in __init__.
        self.t.data[:, 0].clamp_(min=self.t_floor)
        if self.t.shape[1] > 1:
            self.t.data[:, 1:].clamp_(min=0.0)
        self.gamma.clamp_(0.0, 1.0)
        self.Wbeta.weight.clamp_min_(1e-4)
        self.project_filters()

    # -----------------------------------------------------------------
    # logging hook (visualization/params.py picks this up automatically)
    # -----------------------------------------------------------------
    @torch.no_grad()
    def param_logs(self, probes=(0.0, 0.5, 1.0)):
        """The step sizes and threshold this net will ACTUALLY use, at a few bridge positions.
        See BridgeScheduleMixin._sb_param_logs for why the raw coefficients are not enough."""
        out = self._sb_param_logs({
            "eta": lambda sl, sh: [torch.sigmoid(_horner(self.a_eta[j], sl)) for j in range(self.K)],
            "nu": lambda sl, sh: [torch.sigmoid(_horner(self.a_nu[j], sl)) for j in range(self.K)],
            "tau": lambda sl, sh: [_horner(self.t[j], sh) for j in range(self.K)],
        }, probes=probes)
        out["gamma"] = float(self.gamma)          # adjacency blend, GroupCDL's only extra scalar
        return out

    # -----------------------------------------------------------------
    # forward
    # -----------------------------------------------------------------
    def forward(self, y, E=None, sigma=None, step=None):
        """`y = cat([x_t, cond], dim=1)`, the tensor sb.base.predict_x0 builds. `E` is accepted
        for signature parity with the repo's denoisers and ignored. Returns (x0_hat, z)."""
        # split, debias, DC-correct and pad -- all shared with SBCDLNet
        r, c, dc, pad, mu0, s_log, s_hat = self.bridge_inputs(y, sigma=sigma, step=step)

        z = torch.zeros_like(self.A[0](r))
        state = None                                           # Gamma^(0) = I
        for k in range(self.K):
            # Adjacency from the CURRENT code, refreshed every dK layers (GroupCDL Alg. 4). At
            # k = 0 z is still zero, so `_is_update_layer` leaves Gamma = I and the group
            # threshold degenerates to the soft threshold -- the same warm-up GroupCDL has.
            apply_adj, state = self._update_attention(state, z, k)

            eta = torch.sigmoid(_horner(self.a_eta[k], s_log))  # (B,1,1,1) in (0,1)
            nu = torch.sigmoid(_horner(self.a_nu[k], s_log))
            # NO relu on tau here. Nonnegativity is enforced by project() clamping the
            # COEFFICIENTS after every optimizer step (train_i2sb calls it each iteration), as
            # CDLNet does with t.clamp_(0.). Clamping the OUTPUT instead would put the common
            # t0 = 0 start right on relu's kink, where the subgradient is 0 -- the threshold
            # would then receive no gradient at any degree and stay pinned at zero all run.
            tau = _horner(self.t[k], s_hat)                     # (B,M,1,1)

            g_D = mu0 * self.A[k](mu0 * self.B[k](z) - r)       # target fidelity
            g_P = self.A_P[k](self.B_P[k](z) - c)               # prior fidelity
            u = z - eta * g_D - nu * g_P
            if self.fenchel:
                z = z - self._threshold(u, apply_adj, tau)      # group clipping (Moreau)
            else:
                z = self._threshold(u, apply_adj, tau)          # group threshold, Eq. 11

        x0_hat = unpad(self.B[0](z), pad) + dc
        return x0_hat, z
