import torch
import torch.nn as nn
import torch.nn.functional as F

from models.components import CLIP, Conv2d, ConvTranspose2d
from operators.padding import unpad
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
        magnitude_output=False,
    ):
        super().__init__()

        self.K, self.M, self.P, self.s, self.C = K, M, P, s, C
        self.adaptive = adaptive
        # Return |x| instead of the complex iterate. For magnitude images (raw MR data): the
        # weights stay complex, but everything downstream -- loss, metrics, and an I2SB sampler
        # that mixes the prediction back into a REAL bridge state -- sees a real image.
        self.magnitude_output = bool(magnitude_output)
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
        if getattr(E, "nonlinear", False):
            x, z = self._forward_nonlinear(y, E, sigma)
            return (x.abs() if self.magnitude_output else x), z
        x, z = self._forward_linear(y, E, sigma)
        return (x.abs() if self.magnitude_output else x), z

    def _forward_nonlinear(self, y, E, sigma=None):
        """The same K primal-dual layers for a NONLINEAR operator (operators/learned.py):

            x <- x - eta_k (grad f(x) + B_k z),   x <- x + theta_k (x - x_prev),
            z <- CLIP(z + A_k x, l_k)

        As in CDLNet._forward_nonlinear, the gradient is ONE call at the current iterate, the
        start is E.init(y), and E is evaluated on the un-preprocessed image unpad(x) + mean.

        What is new is that the iterate is COMPLEX (the weights are) while E, the bridge state and
        T1 are real magnitude images. The data term is taken on the MAGNITUDE, f(|x|), so

            grad_x f(|x|) = sgn(x) * E.data_grad(|x|, y)          (sgn(x) = x / |x|, 0 at 0)

        -- the data fixes |x| and leaves the phase to the prior. With magnitude_output=True the
        net returns |x|, the same quantity the data term constrains.

        The iterate starts at the operator's estimate E.init(y) itself (mean-removed), not at 0:
        the linear path's zero start is only reached through E^H y on its first step, which a
        nonlinear operator does not provide.
        """
        yp, params = pre_process(E.init(y), self.s, mask=1)
        xmean, pad = params
        if hasattr(E, "noise_level"):
            sigma = E.noise_level(y)
        c = 0 if sigma is None or not self.adaptive else sigma

        x_prev = yp
        z = torch.zeros_like(self.A[0](x_prev))
        for k in range(self.K):
            u = unpad(x_prev, pad) + xmean                   # absolute image, maybe complex
            if u.is_complex():
                g = u.sgn() * E.data_grad(u.abs(), y)
            else:
                g = E.data_grad(u, y)
            g = F.pad(g, pad)                                # zeros in the pad band

            x = x_prev - self.eta[k] * (g + self.B[k](z))
            x = x + self.theta[k] * (x - x_prev)
            z = CLIP(z + self.A[k](x), self.l[k, :1] + c * self.l[k, 1:2])
            x_prev = x

        return post_process(x, params), z

    def _forward_linear(self, y, E, sigma=None):
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
