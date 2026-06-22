import torch
import torch.nn as nn

from operators import HighPassFilter, Identity()
from models.components import CLIP, ComplexConvTranspose2d
from preprocessing.image import pre_process, post_process
from models.base import BaseUnrolledModel

class IPALMNet(BaseUnrolledModel):
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
        sigma_hpf=0.2, # Standard deviation of Gaussian window in HPF
        adaptive=False,
        init=True,
    ):
        super().__init__()

        self.K, self.M, self.P, self.s, self.C = K, M, P, s, C
        self.psi = HighPassFilter(sigma_hpf)
        self.adaptive = adaptive

        # -----------------------------
        # Operators
        # -----------------------------
        self.A = nn.ModuleList([
            nn.Conv2d(C, M, P, stride=s, padding=(P - 1) // 2, bias=False, dtype=torch.cfloat)
            for _ in range(K)
        ])
        
        self.B = nn.ModuleList([
            ComplexConvTranspose2d(M, C, P, stride=s, bias=False)
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

        # Eta and beta are now going to be learnable polynomials that are functions of the image domain noise power.
        self.eta = nn.ModuleList([LearnablePolynomial(coeffs = poly_coeffs) for k in range(K)])
        # We have a beta to control step size on (x_hat-x)
        self.beta = nn.ModuleList([LearnablePolynomial(coeffs = poly_coeffs) for k in range(K)])

        # Extrapolation parameters (need two sets for ascent in x1 and x2)
        self.theta = nn.Parameter(theta_0 * torch.ones(K, 2))

        # init shared weights
        self.init_filters()
        if init:
            self.spectral_init()

    def forward(self, y, 
                      MF, 
                      S,
                      E_z = None,
                      sigma=None):
        '''
        Here, we switch to MRI specific notations.
        We consider our primal variables to be (y, S) where y is some measured kspace and S is an optimal sensitivity map estimate.
        z is still our dual variable. 

        Additionally, we need some custom IPALM operators for MRI - mainly ones that take in both x and sensitivity maps as inputs and have the proper adjoint operations. 
        We consider the case where E is actually only a composition Mask @ FFT2D
        '''
        # Initializing MRI Operator with given sensitivity maps. 
        E  = MF @ Sense(S)
        
        S_prev = torch.clone(S)

        EHy = E.H(y)

        yp, params = pre_process(EHy, self.s, mask=1)

        c1 = 0 if sigma is None or not self.adaptive else sigma
        
        x_prev = torch.zeros_like(EHy)
        z = torch.zeros_like(self.A[0](x_prev))

        if E_z is None:
            E_z = Identity()

        for k in range(self.K):
            
            eta_k = torch.sigmoid(self.eta[k](c2))
            beta_k = torch.sigmoid(self.beta[k](c2))

            # Gradient Descent in primal variable x1
            x = x_prev - self.eta[k, 0] * (E.normal(x_prev) - yp + self.B[k](z))
            # Extrapolation in x1
            x = x + self.theta[k, 1] * (x - x_prev)

            # Gradient Descent in primal variable x2
            S = S_prev - self.eta[k, 1] * (self.psi.H(S) - x.conj() * MF.H(y-E(x)))
            # Extrapolation in x2
            S = S + self.theta[k, 2] * (S - S_prev)

            # Proximal Ascent in dual variable
            z = CLIP(
                z + self.A[k](x),
                self.l[k, :1] + c * self.l[k, 1:2],
            )
            # Remake MRI Encoding operator
            x_prev = x
            S_prev = S

            E  = MF @ Sense(S)

        return post_process(x, params), S, z

    @torch.no_grad()
    def project(self):
        self.l.clamp_(0.0)
        self.eta.clamp_(0.0)
        self.theta.clamp_(0.0, 1.0)
        self.project_filters()
