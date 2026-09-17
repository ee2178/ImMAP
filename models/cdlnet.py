import torch
import torch.nn as nn
import torch.nn.functional as F

from models.components import ST, Conv2d, ConvTranspose2d
from models.base import BaseUnrolledModel
from operators.padding import unpad
from preprocessing.image import pre_process, post_process


class CDLNet(BaseUnrolledModel):
    def __init__(self, K=3, M=64, P=7, s=1, C=1, t0=0, adaptive=False, init=True, complex = True):
        super().__init__()

        self.K, self.M, self.P, self.s, self.C = K, M, P, s, C
        self.adaptive = adaptive

        self.A = nn.ModuleList([
            Conv2d(C, M, P, stride=s, bias=False, complex = complex)
            for _ in range(K)
        ])

        self.B = nn.ModuleList([
            ConvTranspose2d(M, C, P, stride=s, bias=False, complex = complex)
            for _ in range(K)
        ])

        self.D = self.B[0] # alias D to B[0], otherwise unused as z0 is 0

        self.t = nn.Parameter(t0 * torch.ones(K, 2, M, 1, 1))

        self.init_filters()

        if init:
            self.spectral_init()

    def forward(self, 
                y,          # Measurement
                E,          # Operator
                sigma=None  # Noise Level (optional)
                ):

        if getattr(E, "nonlinear", False):
            return self._forward_nonlinear(y, E, sigma)

        EHy = E.H(y)

        yp, params = pre_process(EHy, self.s)

        c = 0 if sigma is None or not self.adaptive else sigma
        
        # Initialization of z^(0)
        z = torch.zeros_like(self.A[0](yp))

        # K ISTA iterations, now with operator E
        for k in range(self.K):
            z = ST(
                z - self.A[k](E.H(E(self.B[k](z)) - yp)),
                self.t[k, :1] + c * self.t[k, 1:2],
            )

        x_hat = post_process(self.B[0](z), params)
        return x_hat, z

    def _forward_nonlinear(self, y, E, sigma=None):
        """The same K ISTA layers for a NONLINEAR operator (operators/learned.py):

            z <- ST(z - A_k grad f(B_k z),  tau_k),     grad f = E.data_grad(x, y)

        Three things differ from the linear path, all forced by E having no point-free adjoint:

          * the gradient is ONE call at the current iterate. `E.H(E(x)) - E.H(y)` would take
            J_E^T at two different points (E^H y has no point at all).
          * the start is `E.init(y)`, the operator's own image-domain estimate, not E^H y.
          * pre_process's mean removal and padding do NOT commute with a learned E, which was
            trained on real intensities at the true frame size. So each gradient is evaluated
            on x = unpad(B_k z) + mean, and zero-padded back: the exact gradient w.r.t. the
            padded iterate (pad rows carry no data term).

        `sigma` for the adaptive threshold is E.noise_level(y) when the operator defines one --
        the std of the estimate the layers are refining -- else the caller's sigma.
        """
        yp, params = pre_process(E.init(y), self.s)
        xmean, pad = params
        if hasattr(E, "noise_level"):
            sigma = E.noise_level(y)
        c = 0 if sigma is None or not self.adaptive else sigma

        z = torch.zeros_like(self.A[0](yp))
        for k in range(self.K):
            x = unpad(self.B[k](z), pad) + xmean
            g = F.pad(E.data_grad(x, y), pad)                 # zeros in the pad band
            z = ST(z - self.A[k](g), self.t[k, :1] + c * self.t[k, 1:2])

        x_hat = post_process(self.B[0](z), params)
        return x_hat, z

    @torch.no_grad()
    def project(self):
        self.t.clamp_(0.0)
        self.project_filters()

