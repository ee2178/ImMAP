import torch
import torch.nn as nn

from models.components import CLIP, Conv2d, ConvTranspose2d, LearnablePolynomial
from preprocessing.image import pre_process_pair, post_process
from models.base import BaseUnrolledModel
from operators import Identity


class DiffLPDSNet(BaseUnrolledModel):
    """
    Diffusion / SSDU / double-noise LPDS variant
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
        poly_coeffs=torch.tensor([-1.1, 0, 0]),
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

        self.l = nn.Parameter(
            torch.cat(
                (
                    l0 * torch.ones(K, 1, M, 1, 1),
                    torch.zeros(K, 1, M, 1, 1),
                ),
                dim=1,
            )
        )

        self.l_2 = nn.Parameter(torch.zeros(K, 1, M, 1, 1))

        # Eta and beta are now going to be learnable polynomials that are functions of the image domain noise power. 
        self.eta = nn.ModuleList([LearnablePolynomial(coeffs = poly_coeffs) for k in range(K)])
        # We have a beta to control step size on (x_hat-x)
        self.beta = nn.ModuleList([LearnablePolynomial(coeffs = poly_coeffs) for k in range(K)])

        # Extrapolation remains the same
        self.theta = nn.Parameter(theta_0 * torch.ones(K, 1))

        self.init_filters()
        if init:
            self.spectral_init()

    def forward(
        self,
        y,
        x_init,     # For diffusion inference, we REQUIRE this input
        E,          # Measurement operator
        E_z = None, # Optional SSDU Masking operator
        sigma=None,
        sigma_t=None,
    ):
        # Refactor to take in an arbitrary E, EH operator
        if E_z is None:
            E_z = Identity()

        EHy = E.H(y)

        yp, x_init_p, params = pre_process_pair(EHy, x_init, self.s, mask=1)

        c1 = 0 if sigma is None else sigma
        c2 = 0 if sigma_t is None else sigma_t

        x_prev = torch.zeros_like(EHy)
        z = torch.zeros_like(self.A[0](x_prev))

        for k in range(self.K):
            
            eta_k = torch.sigmoid(self.eta[k](c2))
            beta_k = torch.sigmoid(self.beta[k](c2))

            x = (
                x_prev
                - eta_k * (E.H(E(x_prev)) - yp)
                - 0.25 * self.B[k](z)
                - beta_k * E_z.H(E_z(x_prev - x_init_p))
            )

            x = x + self.theta[k] * (x - x_prev)

            z = CLIP(
                z + self.A[k](x),
                self.l[k, :1] + c1 * self.l[k, 1:2] + c2 * self.l_2[k],
            )

            x_prev = x

        return post_process(x, params), z

    @torch.no_grad()
    def project(self):
        self.l.clamp_(0.0)
        self.l_2.clamp_(0.0)
        self.theta.clamp_(0.0, 1.0)
        self.project_filters()
