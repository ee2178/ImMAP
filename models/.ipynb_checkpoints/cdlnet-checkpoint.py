import torch
import torch.nn as nn

from models.components import ST, ComplexConvTranspose2d
from models.base import BaseUnrolledModel
from preprocessing.image import pre_process, post_process


class CDLNet(BaseUnrolledModel):
    def __init__(self, K=3, M=64, P=7, s=1, C=1, t0=0, adaptive=False, init=True):
        super().__init__()

        self.K, self.M, self.P, self.s, self.C = K, M, P, s, C
        self.adaptive = adaptive

        self.A = nn.ModuleList([
            nn.Conv2d(C, M, P, stride=s, padding=(P-1)//2, bias=False, dtype=torch.cfloat)
            for _ in range(K)
        ])

        self.B = nn.ModuleList([
            ComplexConvTranspose2d(M, C, P, stride=s, bias=False)
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

        EHy = E.H(y)

        yp, params = pre_process(EHy, self.s)

        c = 0 if sigma is None or not self.adaptive else sigma
        
        # Initialization of z^(0)
        z = torch.zeros_like(self.A[0](yp))

        # K ISTA iterations, now with operator E
        for k in range(self.K):
            z = ST(
                z - self.A[k](EH(E(self.B[k](z)) - yp)),
                self.t[k, :1] + c * self.t[k, 1:2],
            )

        x_hat = post_process(self.B[0](z), params)
        return x_hat, z

    @torch.no_grad()
    def project(self):
        self.t.clamp_(0.0)
        self.project_filters()

