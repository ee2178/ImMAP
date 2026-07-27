"""
Domain-Transfer CDLNet (DT-CDLNet).

A convolutional-dictionary unrolled network for *domain transfer* (image-to-image
translation between two acquisition contrasts / modalities), built directly on top
of CDLNet.

Idea
----
CDLNet infers a sparse code z from a source measurement y by unrolled ISTA on the
convolutional BPDN problem, then synthesizes the image with a single dictionary
D = B[0] (z -> y). DT-CDLNet keeps that sparse-coding stage *unchanged* -- z is a
shared, contrast-agnostic content code -- but decodes it into a DIFFERENT domain
with a SECOND, independent synthesis dictionary:

    preprocess:  y~ = pre(E^H y)                     # source domain, mean-subtracted + padded
    z^(0) = 0
    for k = 0..K-1:                                  # identical to CDLNet
        z^(k+1) = ST( z^(k) - A^(k)( E^H( E( B^(k) z^(k) ) - y~ ) ) ; t^(k) )
    x_hat = post( Dx z^(K) )                         # z -> x   (TARGET domain)
    y_hat = post( Dy z^(K) )   (optional)            # z -> y   (SOURCE domain)

So the two "separate dictionaries" are:
  * Dy := B[0]  -- the z -> y (source) synthesis, already learned by the CDLNet
                   sparse-coding stage and used inside the unrolling.
  * Dx          -- a z -> x (target) synthesis, a brand-new dictionary that is
                   fully INDEPENDENT of Dy in this variant.

`forward(..., return_source=True)` returns y_hat as well, for a source-reconstruction
(cycle/consistency) loss alongside the target-transfer loss.

Deferred (see the "additive delS" plan)
---------------------------------------
  * Additive coupling  Dx = Dy + delS : parameterize the target dictionary as the
    source dictionary plus a learned perturbation, instead of two free dictionaries.
    Dx is warm-started here as a copy of Dy precisely so that this reduces to
    delS = 0 with no change in initial behaviour.
  * Target-domain DC/mean handling : the target decode is currently un-padded and
    re-centered with the SOURCE mean/pad (post_process with the source params),
    which is exact for y_hat and a placeholder for x_hat when the two domains have
    different DC statistics.
"""

import torch

from models.cdlnet import CDLNet
from models.components import ST, ConvTranspose2d
from operators.projections import uball_project
from preprocessing.image import pre_process, post_process
from operators import Identity

class DTCDLNet(CDLNet):
    """CDLNet with two synthesis dictionaries (z -> y and z -> x) for domain transfer.

    Parameters
    ----------
    K, M, P, s, C, t0, adaptive, init, complex
        Passed straight through to :class:`CDLNet` -- these configure the shared
        sparse-coding stage (analysis A, source synthesis B, thresholds t) exactly
        as in a standard CDLNet. `C` is the number of SOURCE channels.
    Cx : int or None
        Number of TARGET channels for the z -> x dictionary. Defaults to `C`
        (same channel count in both domains, e.g. single-channel MRI contrasts).
    """

    def __init__(self, K=3, M=64, P=7, s=1, C=1, t0=0, adaptive=False,
                 init=True, complex=True, Cx=None):
        # Builds A, B, t and runs (spectral) init for the source sparse-coding stage.
        super().__init__(K=K, M=M, P=P, s=s, C=C, t0=t0,
                         adaptive=adaptive, init=init, complex=complex)

        self.Cx = C if Cx is None else Cx

        # z -> y (source) decoder. B[0] already plays this role inside CDLNet, both
        # in the unrolling and as the reconstruction dictionary; alias it as Dy.
        self.Dy = self.B[0]

        # z -> x (target) decoder: a SEPARATE dictionary, independent of Dy.
        self.Dx = ConvTranspose2d(M, self.Cx, P, stride=s, bias=False, complex=complex)

        # Warm-start Dx from the (already spectrally-normalized) source dictionary
        # when the channel counts match, so training begins at an identity transfer
        # (x_hat == y_hat) and cleanly reduces to the future additive variant
        # (Dx = Dy + delS with delS = 0). If Cx != C the shapes differ, so we keep
        # the default ConvTranspose2d initialization.
        if self.Cx == C:
            self.Dx.weight = self.B[0].weight

    def forward(self,
                y,                    # Source-domain measurement
                E = Identity(),       # Forward operator (Identity for pure transfer)
                sigma=None,           # Noise level (optional; only if adaptive)
                return_source=False,  # Also return the source reconstruction y_hat
                ):
        EHy = E.H(y)

        yp, params = pre_process(EHy, self.s)

        c = 0 if sigma is None or not self.adaptive else sigma

        # ------------------------------------------------------------------
        # Shared sparse code z, inferred from the SOURCE measurement.
        # This block is identical to CDLNet: the data term E^H(E(B z) - y~)
        # lives in the source domain, so B is the z -> y synthesis.
        # ------------------------------------------------------------------
        z = torch.zeros_like(self.A[0](yp))

        for k in range(self.K):
            z = ST(
                z - self.A[k](E.H(E(self.B[k](z)) - yp)),
                self.t[k, :1] + c * self.t[k, 1:2],
            )

        # ------------------------------------------------------------------
        # Decode the shared code into the TARGET domain (z -> x).
        # pass a copy of params so post_process (which pops) can be reused below.
        # ------------------------------------------------------------------
        x_hat = post_process(self.Dx(z), list(params))

        if return_source:
            # Source reconstruction (z -> y) for a consistency / cycle loss.
            y_hat = post_process(self.Dy(z), list(params))
            return x_hat, y_hat, z

        return x_hat, z

    @torch.no_grad()
    def project(self):
        # Clamps thresholds t and projects A, B (hence Dy = B[0]) onto the unit ball.
        super().project()
        # Apply the same unit-ball constraint to the independent target dictionary.
        self.Dx.weight = uball_project(self.Dx.weight)
