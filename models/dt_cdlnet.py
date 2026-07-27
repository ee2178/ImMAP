"""
Domain-Transfer CDLNet (DT-CDLNet): an unrolled FISTA network with an additive
enhancement map delta-S.

Model
-----
Paired (x, y) with only y observable at inference (y = source contrasts, x = target
contrast). The target is the base synthesis plus a sparse, one-sided ENHANCEMENT map:

    x = D_{z->x} z + dS ,      dS >= 0

Training-time objective (x IS observed), jointly convex because the coupling
[D_x  I][z; dS] is LINEAR, not bilinear -- which is why this is FISTA on a stacked
variable rather than iPALM:

    min_{z, dS}  1/2 ||y - D_{z->y} z||_2^2  +  gamma/2 ||x - D_{z->x} z - dS||_2^2
                 + lam ||z||_1  +  lam_S ||dS||_1  +  i_{>=0}(dS)

Stacking w = [z; dS], K = [[D_y, 0], [sqrt(g) D_x, sqrt(g) I]], b = [y; sqrt(g) x], the
smooth part is one quadratic and the nonsmooth part is block-separable:

    prox_{eta g}(w) = [ soft_{eta lam}(z) ; max(dS - eta lam_S, 0) ]

so one FISTA step (with inertia zbar^k = z^k + beta^k (z^k - z^{k-1})) is

    r_y = D_y z - y ,   r_x = D_x z + dS - x
    z^{k+1}  = soft_{eta lam}( zbar^k - eta D_y^T r_y - eta gamma D_x^T r_x )
    dS^{k+1} = ReLU( dS^k - eta_S gamma r_x - eta_S lam_S )

The obstruction, and the unrolling
----------------------------------
r_x contains x, which vanishes at inference. Expanding the dS step isolates exactly what
is lost:

    dS^k - eta_S gamma r_x = (1 - eta_S gamma) dS^k  +  eta_S gamma (x - D_x zbar^k)
                             \_____ leak _____/        \___ enhancement residual ___/

The leak is computable; ONLY the enhancement residual becomes a learned operator. In the
z-branch the analogous unavailable piece is gamma D_x^T x, which we DROP (v1) -- so the
z-branch is exactly CDLNet and the ablation baseline is this network with the dS branch
removed. The unrolled layer:

    r^k      = B^k zbar^k - y
    z^{k+1}  = soft_{tau^k}( zbar^k - A^k r^k )
    dS^{k+1} = ReLU( (1 - rho^k) dS^k + E^k z^{k+1} - tauS^k )
    x_hat    = D_x z^K + dS^K

E^k is a synthesis convolution (code -> target image) standing in for the enhancement
residual; rho^k in [0,1] is a learned leak; tauS^k is a learned ReLU bias. The dS step uses
z^{k+1} (not zbar^k) -- the Gauss-Seidel ordering PALM uses, which is free and strictly
better information.

Parameters: {A^k, B^k, tau^k, E^k, rho^k, tauS^k, beta^k} plus D_x -- one convolution and
three scalars per layer over CDLNet.

Initialization makes layer 0 an exact algorithm step: A^k = eta D_y^T, B^k = D_y (both from
CDLNet's spectral init), E^k = eta_S gamma D_x, rho^k = eta_S gamma, tauS^k = eta_S lam_S,
beta^k on the FISTA schedule.

On rho0 (the step size eta_S gamma). For the ADDITIVE model L_dS = gamma exactly, so the
prox-gradient step that jumps straight to the dS-subproblem optimum is eta_S = 1/gamma, i.e.
rho = 1. That is mathematically the "exact" choice, but it sets the leak (1 - rho) to ZERO,
which makes the dS branch MEMORYLESS: dS^{k+1} = ReLU(E^k z^{k+1} - tauS^k) depends only on
the last layer, so E^0..E^{K-2} receive no gradient at initialization and sit unused unless
rho later moves off 1. The default rho0 = 0.5 keeps the accumulator live so every layer's E^k
contributes from step one; set rho0 = 1.0 if you want the literal one-step-to-optimum form.

Known quirk inherited from CDLNet: z^0 = 0, so B^0 is applied to zero inside the loop and is
inert there. In plain CDLNet B[0] doubles as the output dictionary D, but here the read-out
uses D_x, so B[0] only gets gradient through the SOURCE reconstruction -- i.e. only when
lam_src > 0. With lam_src = 0 it is effectively an unused parameter.

Note on `gamma`
---------------
Once E^k and rho^k are untied and learnable, an explicit gamma is REDUNDANT -- it is
absorbed into them. `learn_gamma=True` is still offered because it gives one interpretable
"enhancement drive" scalar shared across layers: it multiplies BOTH the injection and the
leak, preserving the algorithmic relation that each is proportional to eta_S gamma, while
E^k carries only the shape. It is a no-op at initialization (gamma = 1).

dS is one-sided (ReLU), so this model is REAL-valued only; a one-sided threshold has no
meaning for complex codes.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cdlnet import CDLNet
from models.components import ST, ConvTranspose2d
from operators.projections import uball_project
from operators.padding import unpad
from preprocessing.image import pre_process, post_process
from operators import Identity


def _fista_betas(K):
    """The classical FISTA inertia schedule beta_k = (t_k - 1)/t_{k+1}, beta_0 = 0."""
    betas, t = [], 1.0
    for _ in range(K):
        t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
        betas.append((t - 1.0) / t_next)
        t = t_next
    return torch.tensor(betas, dtype=torch.float32)


class DTCDLNet(CDLNet):
    """Unrolled FISTA for  x = D_x z + dS  with dS >= 0 sparse (see module docstring).

    Parameters
    ----------
    K, M, P, s, C, t0, adaptive, init
        Passed to :class:`CDLNet` -- they configure the shared sparse-coding trunk
        (analysis A, source synthesis B, thresholds t). `C` = number of SOURCE channels.
    Cx : int or None
        Number of TARGET channels (default `C`). The dS map has this many channels.
    rho0 : float
        Initial leak parameter rho^k = eta_S gamma, in [0, 1]. rho = 1 is the exact
        one-step-to-optimum step size but makes the dS branch memoryless (only the last
        layer's E^k matters, the rest start with zero gradient); the default 0.5 keeps the
        accumulator live across layers. See the module docstring.
    tauS0 : float
        Initial ReLU bias tauS^k = eta_S lam_S. 0.0 = no thresholding at init.
    momentum : bool
        Learn the FISTA inertia beta^k (initialized to the classical schedule). False
        pins beta = 0, giving an ISTA trunk.
    learn_gamma, gamma0 : bool, float
        Optional shared "enhancement drive" scalar; see the module docstring (redundant
        with untied E/rho, and a no-op at gamma0 = 1).
    """

    def __init__(self, K=3, M=64, P=7, s=1, C=1, t0=0, adaptive=False,
                 init=True, complex=False, Cx=None,
                 rho0=0.5, tauS0=0.0, momentum=True, learn_gamma=False, gamma0=1.0):
        if complex:
            raise ValueError(
                "DTCDLNet is real-valued only: the dS branch uses a ONE-SIDED (ReLU) prox, "
                "which has no meaning for complex codes. Build with complex=False.")
        # Builds A, B, t and runs the (spectral) init for the source sparse-coding trunk.
        super().__init__(K=K, M=M, P=P, s=s, C=C, t0=t0,
                         adaptive=adaptive, init=init, complex=False)

        self.Cx = C if Cx is None else Cx
        self.momentum = bool(momentum)
        self.learn_gamma = bool(learn_gamma)

        # z -> y (source) synthesis: B[k] inside the unrolling; alias B[0] for readability.
        self.Dy = self.B[0]

        # z -> x (target) synthesis: the algorithm's D_x, applied once at read-out.
        self.Dx = ConvTranspose2d(M, self.Cx, P, stride=s, bias=False, complex=False)

        # E^k: code -> target image, the learned surrogate for the enhancement residual.
        self.E_S = nn.ModuleList([
            ConvTranspose2d(M, self.Cx, P, stride=s, bias=False, complex=False)
            for _ in range(K)
        ])
        # E^k = eta_S gamma D_x = rho0 * D_x  makes layer 0 an exact algorithm step.
        for k in range(K):
            self.E_S[k].weight = rho0 * self.Dx.weight

        self.rho = nn.Parameter(rho0 * torch.ones(K, 1, 1, 1))     # leak,       in [0, 1]
        self.tauS = nn.Parameter(tauS0 * torch.ones(K, 1, 1, 1))   # ReLU bias,  >= 0
        self.beta = nn.Parameter(_fista_betas(K).reshape(K, 1, 1, 1)) if self.momentum else None
        self.gamma = nn.Parameter(torch.tensor(float(gamma0))) if self.learn_gamma else None

    # ------------------------------------------------------------------
    def forward(self,
                y,                    # source-domain measurement
                E=Identity(),         # forward operator (Identity for pure transfer)
                sigma=None,           # noise level (only used if adaptive)
                return_source=False,  # also return the source reconstruction y_hat
                ):
        EHy = E.H(y)
        yp, params = pre_process(EHy, self.s)
        pad = params[1]                       # keep: post_process pops `params`
        c = 0 if sigma is None or not self.adaptive else sigma

        z = torch.zeros_like(self.A[0](yp))
        z_prev = z
        dS = None                             # lazily shaped from the first E^k decode

        for k in range(self.K):
            # FISTA inertia on the code
            zbar = z if self.beta is None else z + self.beta[k] * (z - z_prev)

            # ---- z-branch: exactly CDLNet (the dropped gamma D_x^T x term lives here) ----
            z_next = ST(
                zbar - self.A[k](E.H(E(self.B[k](zbar)) - yp)),
                self.t[k, :1] + c * self.t[k, 1:2],
            )

            # ---- dS-branch: leaky accumulator + one-sided threshold ----
            # Gauss-Seidel: uses the JUST-updated z^{k+1}, not zbar^k.
            inj = self.E_S[k](z_next)
            rho = self.rho[k]
            if self.gamma is not None:        # shared drive: both terms scale with eta_S gamma
                inj = self.gamma * inj
                rho = self.gamma * rho
            dS = inj if dS is None else (1.0 - rho) * dS + inj
            dS = F.relu(dS - self.tauS[k])

            z_prev, z = z, z_next

        # ---- read-out:  x_hat = D_x z^K + dS^K ----
        x_hat = post_process(self.Dx(z) + dS, list(params))
        dS_img = unpad(dS, pad)               # residual: unpad only, NO mean added back

        if return_source:
            y_hat = post_process(self.Dy(z), list(params))
            return x_hat, dS_img, y_hat, z
        return x_hat, dS_img, z

    # ------------------------------------------------------------------
    @torch.no_grad()
    def project(self):
        # Clamps thresholds t and projects A, B (hence Dy = B[0]) onto the unit ball.
        super().project()
        self.tauS.clamp_(0.0)                 # it is a threshold
        self.rho.clamp_(0.0, 1.0)             # leak (1 - rho) must stay a convex weight
        if self.beta is not None:
            self.beta.clamp_(0.0, 1.0)        # FISTA inertia
        if self.gamma is not None:
            self.gamma.clamp_(0.0)
        # D_x is a synthesis dictionary -> same unit-ball discipline as B. E^k is a learned
        # surrogate operator (not a dictionary whose atoms set an ISTA step size), so it is
        # left free -- constraining it would cap how strongly dS can be driven.
        self.Dx.weight = uball_project(self.Dx.weight)
