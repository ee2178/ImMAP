# -*- coding: utf-8 -*-
"""
SBCDLNet -- a Schrodinger-bridge CDLNet: unrolled ISTA on a TWO-domain sparse coding problem.

Ordinary CDLNet unrolls a denoising MAP problem,

    argmin_z  1/2 ||y - D z||^2 + lambda ||z||_1,

but an I2SB regressor is not denoising: at bridge step k it sees the state x_t and must return
the target endpoint x_0, with the prior endpoint x_1 known EXACTLY. This net unrolls the
corresponding two-fidelity problem instead. One shared sparse code z explains BOTH contrasts
through domain-specific dictionaries,

    x_0 = D_D z        (target, e.g. T1ce)
    x_1 = D_P z        (prior,  e.g. T1)

so the bridge interpolant x_t = mu_0 x_0 + mu_1 x_1 + sigma_sb * eps gives a residual that is
linear in z once the KNOWN prior contribution is removed:

    r = x_t - mu_1 x_1 = mu_0 x_0 + sigma_sb * eps  ~  mu_0 D_D z + sigma_sb * eps

The scaffold minimized (loosely -- see NOT AN OPTIMIZER below) is

    J_k(z) = 1/2 ||r - mu_0 D_D z||^2 + gamma/2 ||c - D_P z||^2 + sum_m lambda_m ||z_m||_1

where `c` is the conditioning stack (the prior contrast alone in the base case). The third term
is a WEIGHTED l1: lambda_m is one number per dictionary atom, shared across that atom's spatial
map, exactly as CDLNet's per-channel threshold `t[k, :, m]`.

WHY THE DEBIASED RESIDUAL. Anchoring the target fidelity on x_t directly (i.e. ||x_t - D_D z||^2)
is INCONSISTENT: at the true code its residual is mu_1 (x_1 - x_0), a systematic bias that grows
to the full inter-contrast gap at the prior end -- it asks a T1ce dictionary to reconstruct a
state that is partly T1. Subtracting mu_1 x_1 removes it exactly, for free, using only schedule
constants. No amount of t-weighting can substitute: the correction needs a NEGATIVE multiple of
x_1 analyzed through D_D, and no positively-weighted sum of fidelity terms can produce it.

WHY mu_0 MULTIPLIES AND NEVER DIVIDES. The whitened observation (x_t - mu_1 x_1)/mu_0 is the
statistically natural anchor, but mu_0 -> 1/(n+1) at the prior end, so forming it explicitly
amplifies the input ~1000x. Keeping mu_0 on the operator side instead leaves every quantity
bounded: r -> 0 and the target gradient picks up mu_0^2, so the term self-annihilates exactly
where the bridge state carries no information about x_0. The "trust weighting" is therefore not
a hand-chosen coefficient -- it falls out of the model, and a step size in [0, 1] suffices.

ENDPOINT LIMITS (see tests):
    k -> 0        mu_0 -> 1, r -> x_0        full-strength ISTA against the target -> identity
    k -> n-1      mu_0 -> 0, r -> 0          target term vanishes; z is set by the prior fidelity
                                             alone and x0_hat = D_D z is pure coupled-dictionary
                                             cross-modal synthesis from x_1.

NOT AN OPTIMIZER. The two fidelity blocks get SEPARATE learned steps (eta_k, nu_k), so a layer is
a block-preconditioned proximal step, not a literal proximal-gradient step on J_k -- there is no
single scalar for the prox to inherit. That is fine because the threshold is learned directly and
absorbs whatever the factor should have been, but it does mean a learned tau is NOT an estimate of
lambda_m. The scaffold fixes the FORM of each layer; it is not minimized.

Calling convention matches the rest of the repo's denoisers so sb.base.predict_x0 works unchanged:

    x0_hat, z = net(y, E=Identity(), sigma=std_fwd)

with `y = cat([x_t, cond])` -- exactly what predict_x0 builds. `E` is accepted and ignored (pure
translation has no forward operator). `sigma` is the schedule's std_fwd, from which the bridge
coefficients are recovered by table lookup, so THE SCHEDULE PARAMETERS HERE MUST MATCH cfg["i2sb"]
(kind / tau / n_points / beta_max) -- `assert_schedule_matches` checks that against a schedule
object. Pass `step=` instead of `sigma=` to bypass the lookup.

REQUIRES a conditioning channel: the bridge prior x_1 must be in `cond` (set the loader's
`cond_idx` to include it and point `prior_idx` at its position), so C = 1 + len(cond_idx) >= 2.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseUnrolledModel, set_weight
from models.components import ST, Conv2d, ConvTranspose2d
from models.sb_schedule import BridgeScheduleMixin, horner as _horner
from operators.padding import unpad
from operators.projections import uball_project
from solvers.eigen import power_method


class SBCDLNet(BridgeScheduleMixin, BaseUnrolledModel):
    """Unrolled two-fidelity sparse coding for the Schrodinger bridge.

    Per layer k, with r the debiased bridge residual and c the (DC-removed) conditioning stack:

        g_D = mu_0 * A_D[k]( mu_0 * B_D[k] z - r )        target fidelity
        g_P =         A_P[k]( B_P[k] z - c )              prior fidelity
        z  <- ST( z - eta_k * g_D - nu_k * g_P ; tau_k )

    and the readout is x0_hat = B_D[0] z + dc, mirroring CDLNet's `D = B[0]` alias.

    Parameters
    ----------
    K, M, P, s     unrolled depth, atoms, filter side, stride -- as in CDLNet.
    C              input width, = 1 + len(cond_idx). Must be >= 2 (the prior lives in cond).
    prior_idx      which CONDITIONING channel is the bridge prior x_1 (0-based within cond).
                   With cond_idx=[0,1,3] = [FLAIR,T1,T2] and x1_idx=1 (T1), this is 1.
    t0             constant term of the threshold polynomial (CDLNet's t0).
    deg_eta        degree of the step polynomials in s = log(sigma_eff). 0 = a plain learned
                   scalar per layer (the recommended starting point).
    deg_tau        degree of the threshold polynomial in the normalized sigma_eff. 1 reproduces
                   CDLNet's affine-in-sigma threshold; 2 additionally spans the sigma^2 law.
    kind, tau, n_points, beta_max
                   the bridge schedule -- MUST match cfg["i2sb"], since sigma is inverted through
                   it to recover (mu_0, mu_1, sigma_eff).
    """

    def __init__(self, K=30, M=169, P=7, s=2, C=2, prior_idx=0, t0=0.0,
                 deg_eta=0, deg_tau=1,
                 kind="brownian", tau=0.19, n_points=1000, beta_max=0.3,
                 init=True, complex=False):
        super().__init__()

        if C < 2:
            raise ValueError(
                f"SBCDLNet needs the bridge prior x_1 as a conditioning channel, so "
                f"C = 1 + len(cond_idx) >= 2; got C={C}. Set the loader's cond_idx to include "
                f"the contrast used as x1_idx.")
        self.n_cond = C - 1
        if not 0 <= prior_idx < self.n_cond:
            raise ValueError(
                f"prior_idx={prior_idx} out of range for {self.n_cond} conditioning channel(s). "
                f"It indexes cond (0-based), not the stored contrasts.")

        self.K, self.M, self.P, self.s, self.C = K, M, P, s, C
        self.prior_idx = int(prior_idx)

        # ---- two dictionary pairs: D = target domain (1 channel), P = prior/conditioning ----
        mk_A = lambda cin: nn.ModuleList(
            [Conv2d(cin, M, P, stride=s, bias=False, complex=complex) for _ in range(K)])
        mk_B = lambda cout: nn.ModuleList(
            [ConvTranspose2d(M, cout, P, stride=s, bias=False, complex=complex) for _ in range(K)])
        self.A_D, self.B_D = mk_A(1), mk_B(1)
        self.A_P, self.B_P = mk_A(self.n_cond), mk_B(self.n_cond)

        self.D = self.B_D[0]        # alias, as CDLNet does: the readout dictionary

        # ---- learned coefficients ----
        # Steps are sigmoid-squashed, so they live in (0,1) and cannot destabilize the iteration.
        # Init at pre-activation 0 -> eta = nu = 0.5: spectral_init makes each pair's ISTA step 1,
        # so a HALF step on each of the two fidelities is the combined-step analogue of CDLNet's
        # single unit step. Starting both at ~1 would double the effective step at mu_0 = 1.
        self.a_eta = nn.Parameter(torch.zeros(K, deg_eta + 1))
        self.a_nu = nn.Parameter(torch.zeros(K, deg_eta + 1))
        # Per-atom threshold, shaped like CDLNet's t = (K, deg+1, M, 1, 1).
        t = torch.zeros(K, deg_tau + 1, M, 1, 1)
        t[:, 0] = float(t0)
        self.t = nn.Parameter(t)

        # ---- schedule tables (buffers: saved with the checkpoint, moved by .to()) ----
        self._init_bridge_tables(kind=kind, tau=tau, n_points=n_points, beta_max=beta_max)

        self.init_filters()
        if init:
            self.spectral_init()

    # `visualization.filters` renders net.A / net.B and returns {} for models without them, so
    # without these the filter logging in train_i2sb would silently be a no-op. They expose the
    # TARGET pair (the readout dictionary) -- what `filters/A_stage_*` shows is D, not P. These
    # are properties, not assigned attributes, so nothing gets registered (and duplicated in the
    # state_dict) a second time.
    @property
    def A(self):
        return self.A_D

    @property
    def B(self):
        return self.B_D

    # -----------------------------------------------------------------
    # initialization / projection, run once per dictionary pair
    # -----------------------------------------------------------------
    def _init_pair(self, A, B, cin, dtype):
        W = torch.randn(self.M, cin, self.P, self.P, dtype=dtype)
        for k in range(self.K):
            set_weight(A[k], W)
            set_weight(B[k], W.conj())

    def init_filters(self, dtype=torch.cfloat):
        self._init_pair(self.A_D, self.B_D, 1, dtype)
        self._init_pair(self.A_P, self.B_P, self.n_cond, dtype)

    @torch.no_grad()
    def _spectral_init_pair(self, A, B, cin):
        """Scale (A, B) so ||B A||_2 = 1, i.e. that pair's ISTA step is 1 -- run per pair so
        eta and nu are both interpretable as a FRACTION of an exact ISTA step."""
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
        self._spectral_init_pair(self.A_D, self.B_D, 1)
        self._spectral_init_pair(self.A_P, self.B_P, self.n_cond)

    @torch.no_grad()
    def project_filters(self):
        for A, B in ((self.A_D, self.B_D), (self.A_P, self.B_P)):
            for k in range(self.K):
                set_weight(A[k], uball_project(A[k].weight))
                set_weight(B[k], uball_project(B[k].weight))

    @torch.no_grad()
    def project(self):
        # Nonnegative threshold coefficients keep tau >= 0 AND nondecreasing in sigma_eff (more
        # noise -> more shrinkage), the same constraint CDLNet's t.clamp_(0.) imposes.
        self.t.clamp_(0.0)
        self.project_filters()

    # -----------------------------------------------------------------
    # logging hook (visualization/params.py picks this up automatically)
    # -----------------------------------------------------------------
    @torch.no_grad()
    def param_logs(self, probes=(0.0, 0.5, 1.0)):
        """The step sizes and threshold this net will ACTUALLY use, at a few bridge positions.
        See BridgeScheduleMixin._sb_param_logs for why the raw coefficients are not enough."""
        return self._sb_param_logs({
            "eta": lambda sl, sh: [torch.sigmoid(_horner(self.a_eta[j], sl)) for j in range(self.K)],
            "nu": lambda sl, sh: [torch.sigmoid(_horner(self.a_nu[j], sl)) for j in range(self.K)],
            "tau": lambda sl, sh: [_horner(self.t[j], sh) for j in range(self.K)],
        }, probes=probes)

    # -----------------------------------------------------------------
    # forward
    # -----------------------------------------------------------------
    def forward(self, y, E=None, sigma=None, step=None):
        """`y = cat([x_t, cond], dim=1)`, the tensor sb.base.predict_x0 builds. `E` is accepted
        for signature parity with the repo's denoisers and ignored. Returns (x0_hat, z)."""
        # split, debias, DC-correct and pad -- all shared with SBGroupCDL
        r, c, dc, pad, mu0, s_log, s_hat = self.bridge_inputs(y, sigma=sigma, step=step)

        z = torch.zeros_like(self.A_D[0](r))
        for k in range(self.K):
            eta = torch.sigmoid(_horner(self.a_eta[k], s_log))     # (B,1,1,1) in (0,1)
            nu = torch.sigmoid(_horner(self.a_nu[k], s_log))
            # NO relu here. Nonnegativity is enforced by project() clamping the COEFFICIENTS
            # after every optimizer step (train_i2sb calls it each iteration), exactly as CDLNet
            # does with t.clamp_(0.). Clamping the OUTPUT instead would put the common t0 = 0
            # start right on relu's kink, where the subgradient is 0 -- the threshold would then
            # receive no gradient at any degree and stay pinned at zero for the whole run.
            tau = _horner(self.t[k], s_hat)                        # (B,M,1,1)

            g_D = mu0 * self.A_D[k](mu0 * self.B_D[k](z) - r)      # target fidelity
            g_P = self.A_P[k](self.B_P[k](z) - c)                  # prior fidelity
            z = ST(z - eta * g_D - nu * g_P, tau)

        x0_hat = unpad(self.B_D[0](z), pad) + dc
        return x0_hat, z
