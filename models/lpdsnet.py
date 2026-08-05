import torch
import torch.nn as nn

from models.components import CLIP, Conv2d, ConvTranspose2d
from preprocessing.image import pre_process, post_process
from preprocessing.kspace import kspace_post_process, kspace_pre_process
from models.base import BaseUnrolledModel

class LPDSNet(BaseUnrolledModel):
    """
    Standard learned primal-dual style unrolled model
    (clean base reconstruction model)

    `preproc`
    ---------
    `"image"` (the original behaviour, still the default) subtracts a plain
    `mean(E^H y)`. That is right for denoising and wrong for reconstruction: the
    primal step below is a gradient step on `1/2||y - Ex||^2`, and writing
    `x = v + mu 1` gives `grad_v = E^H E v - (E^H y - mu E^H E 1)`, so the DC
    image has to travel through `E^H E`. See preprocessing/kspace.py for the
    derivation and the choice of `mu`.

    `"kspace"` uses that surrogate, and pads the operator alongside `y~` instead
    of padding only the image -- the original code allocated `x_prev` at the
    UNPADDED size while `yp` was padded, so it silently required a zero pad
    (true at s=2 on 320x320, not in general).
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
        preproc="image",
    ):
        super().__init__()

        self.K, self.M, self.P, self.s, self.C = K, M, P, s, C
        self.adaptive = adaptive
        if preproc not in ("image", "kspace"):
            raise ValueError(
                f"preproc must be 'image' (denoising) or 'kspace' "
                f"(reconstruction); got {preproc!r}")
        self.preproc = preproc

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
        if self.preproc == "kspace":
            # Returns the operator too: padding grows its mask and coil maps, so
            # the loop below must use the one that matches y~.
            yp, E, params = kspace_pre_process(y, E, self.s)
            post = kspace_post_process
        else:
            yp, params = pre_process(E.H(y), self.s, mask=1)
            post = post_process

        c = 0 if sigma is None or not self.adaptive else sigma

        # zeros_like(yp), not zeros_like(E^H y): the iterate has to live on the
        # same (padded) grid as the surrogate it is compared against.
        x_prev = torch.zeros_like(yp)
        z = torch.zeros_like(self.A[0](x_prev))

        for k in range(self.K):

            x = x_prev - self.eta[k] * (E.H(E(x_prev)) - yp + self.B[k](z))
            x = x + self.theta[k] * (x - x_prev)

            z = CLIP(
                z + self.A[k](x),
                self.l[k, :1] + c * self.l[k, 1:2],
            )

            x_prev = x

        return post(x, params), z

    @torch.no_grad()
    def project(self):
        self.l.clamp_(0.0)
        self.eta.clamp_(0.0)
        self.theta.clamp_(0.0, 1.0)
        self.project_filters()
