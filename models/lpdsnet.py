import torch
import torch.nn as nn

from models.components import CLIP, Conv2d, ConvTranspose2d
from preprocessing.image import pre_process, post_process
from models.base import BaseUnrolledModel

class LPDSNet(BaseUnrolledModel):
    """
    Standard learned primal-dual style unrolled model
    (clean base reconstruction model)
    """

    def __init__(
        self,
        K=3,
        M=64,
        P=7,
        s=1,
        C=1,
        l0=1e-3,
        eta_0=0.5,
        theta_0=0.0,
        adaptive=False,
        init=True,
    ):
        super().__init__()

        self.K, self.M, self.P, self.s, self.C = K, M, P, s, C
        self.adaptive = adaptive

        # -----------------------------
        # Operators
        # -----------------------------
        self.A = nn.ModuleList([
            Conv2d(C, M, P, stride=s, bias=False)
            for _ in range(K)
        ])
        
        self.B = nn.ModuleList([
            ConvTranspose2d(M, C, P, stride=s, bias=False)
            for _ in range(K)
        ])

        self.D = self.B[0] # alias D to B[0], otherwise unused as z0 is 0
        
        # Noise adaptive thresholds
        self.l = nn.Parameter(
            torch.cat(
                (
                    l0 * torch.ones(K, 1, M, 1, 1),
                    torch.zeros(K, 1, M, 1, 1),
                ),
                dim=1,
            )
        )

        self.eta = nn.Parameter(eta_0 * torch.ones(K, 1))
        self.theta = nn.Parameter(theta_0 * torch.ones(K, 1))

        # init shared weights
        self.init_filters()
        if init:
            self.spectral_init()

    def forward(self, y, E, sigma=None):
        # Refactor this to just take in arbitrary constructed E, assumed to be of my Operator class.  
        EHy = E.H(y)

        yp, params = pre_process(EHy, self.s, mask=1)

        c = 0 if sigma is None or not self.adaptive else sigma

        x_prev = torch.zeros_like(EHy)
        z = torch.zeros_like(self.A[0](x_prev))

        for k in range(self.K):

            x = x_prev - self.eta[k] * (E.H(E(x_prev)) - yp + self.B[k](z))
            x = x + self.theta[k] * (x - x_prev)

            z = CLIP(
                z + self.A[k](x),
                self.l[k, :1] + c * self.l[k, 1:2],
            )

            x_prev = x

        return post_process(x, params), z

    @torch.no_grad()
    def project(self):
        self.l.clamp_(0.0)
        self.eta.clamp_(0.0)
        self.theta.clamp_(0.0, 1.0)
        self.project_filters()
