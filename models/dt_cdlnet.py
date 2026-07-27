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
  * Dx          -- a z -> x (target) synthesis.

`forward(..., return_source=True)` returns y_hat as well, for a source-reconstruction
(cycle/consistency) loss alongside the target-transfer loss.

Coupling between the two dictionaries (`coupling`)
--------------------------------------------------
"independent" (default)
    Dx is a free dictionary, unrelated to Dy (warm-started as a copy of Dy when the
    channel counts match, so training begins at an identity transfer).

"additive"
    Dx = Dy[:, src_idx] + delS : the target dictionary IS the source dictionary
    (restricted to the source channel(s) `coupling_src_idx`) plus a learned
    perturbation delS, initialized to ZERO. By linearity of the transposed
    convolution this makes the target decode

        x_hat = Dy[:, src_idx] z  +  delS z
              = (reconstruction of the source channel)  +  (learned enhancement)

    i.e. exactly the ADDITIVE image model  x = y_src + S,  with the enhancement map
    S = delS z produced by a dedicated dictionary. For T1ce-from-T1 set
    `coupling_src_idx` to the position of T1 in the source stack: the model then
    predicts T1ce as its own T1 reconstruction plus S. delS = 0 at init, so training
    starts from a pure copy of the source channel and only has to learn S.

    NOTE: because x_hat now depends on Dy, the TARGET loss also backpropagates into
    Dy = B[0], which is part of the unrolled sparse-coding stage. That coupling is
    the point, but it means the target loss shapes the source dictionary even when
    lam_src = 0.

The target decode is un-padded and re-centered with the SOURCE mean/pad
(post_process with the source params). That is exact for y_hat, and for x_hat under
"additive" it is consistent for the Dy[:, src_idx] part (which lives in the source
mean's frame); the residual DC of S is absorbed by delS.
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
    coupling : {"independent", "additive"}
        How the target dictionary relates to the source dictionary (see module docstring).
        "independent" (default) keeps the original behaviour; "additive" parameterizes
        Dx = Dy[:, coupling_src_idx] + delS with delS initialized to zero.
    coupling_src_idx : int, list[int] or None
        Only used when coupling="additive". Which SOURCE channel(s) of Dy the target is
        anchored to, indexed in the order the network receives them (i.e. positions within
        the loader's `input_idx`, NOT stored BraTS channel indices). Must have length `Cx`.
        None means "all source channels" and then requires Cx == C.

        Example: input_idx [0, 1, 3] feeds the net [FLAIR, T1, T2], so T1 is network
        channel 1 -> coupling_src_idx = 1 for a T1ce = T1 + S model.
    """

    def __init__(self, K=3, M=64, P=7, s=1, C=1, t0=0, adaptive=False,
                 init=True, complex=True, Cx=None, coupling="independent",
                 coupling_src_idx=None):
        # Builds A, B, t and runs (spectral) init for the source sparse-coding stage.
        super().__init__(K=K, M=M, P=P, s=s, C=C, t0=t0,
                         adaptive=adaptive, init=init, complex=complex)

        if coupling not in ("independent", "additive"):
            raise ValueError(
                f"coupling must be 'independent' or 'additive', got {coupling!r}")
        self.Cx = C if Cx is None else Cx
        self.coupling = coupling

        # z -> y (source) decoder. B[0] already plays this role inside CDLNet, both
        # in the unrolling and as the reconstruction dictionary; alias it as Dy.
        self.Dy = self.B[0]

        if coupling == "independent":
            # z -> x (target) decoder: a SEPARATE dictionary, independent of Dy.
            self.Dx = ConvTranspose2d(M, self.Cx, P, stride=s, bias=False, complex=complex)
            self.delS = None
            self.src_idx = None

            # Warm-start Dx from the (already spectrally-normalized) source dictionary
            # when the channel counts match, so training begins at an identity transfer
            # (x_hat == y_hat). If Cx != C the shapes differ, so we keep the default init.
            if self.Cx == C:
                self.Dx.weight = self.B[0].weight
        else:
            # Dx = Dy[:, src_idx] + delS. Dx is DERIVED, so there is no free target
            # dictionary module; only the perturbation delS is a parameter.
            self.src_idx = self._resolve_src_idx(coupling_src_idx, C)
            self.Dx = None
            self.delS = ConvTranspose2d(M, self.Cx, P, stride=s, bias=False, complex=complex)
            # delS = 0 -> the model starts as an exact copy of the source channel(s),
            # so it only ever has to learn the enhancement S on top of that.
            self.delS.weight = torch.zeros_like(self.delS.weight)

    def _resolve_src_idx(self, coupling_src_idx, C):
        """Validate/normalize the source-channel selector into a list of length Cx."""
        if coupling_src_idx is None:
            if self.Cx != C:
                raise ValueError(
                    f"coupling='additive' with coupling_src_idx=None couples ALL source "
                    f"channels and so needs Cx == C, but Cx={self.Cx} and C={C}. Pass "
                    f"coupling_src_idx (length Cx={self.Cx}) to pick which source "
                    f"channel(s) the target is anchored to.")
            return list(range(C))

        idx = [coupling_src_idx] if isinstance(coupling_src_idx, int) else list(coupling_src_idx)
        if len(idx) != self.Cx:
            raise ValueError(
                f"coupling_src_idx must have length Cx={self.Cx}, got {len(idx)}: {idx}.")
        if any((not isinstance(i, int)) or i < 0 or i >= C for i in idx):
            raise ValueError(
                f"coupling_src_idx entries must be ints in [0, C={C}), got {idx}. They index "
                f"the SOURCE channels as fed to the network (positions within input_idx).")
        return idx

    @torch.no_grad()
    def effective_target_weight(self):
        """The z -> x dictionary actually applied, for inspection/visualization.
        Detached (built from the `.weight` property, which returns `.data`)."""
        if self.coupling == "independent":
            return self.Dx.weight
        return self.Dy.weight[:, self.src_idx] + self.delS.weight

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
        # The source decode is needed for y_hat, and also for the additive coupling.
        y_dec = self.Dy(z) if (return_source or self.coupling == "additive") else None

        if self.coupling == "additive":
            # Dx = Dy[:, src_idx] + delS, applied via LINEARITY of the transposed conv:
            #   (Dy[:, src_idx] + delS) z  ==  Dy(z)[:, src_idx] + delS(z)
            # (slicing output channels commutes with the conv). Doing it this way keeps
            # autograd flowing into BOTH Dy and delS -- materializing the summed weight
            # via the `.weight` property would detach it, since that property returns .data.
            x_dec = y_dec[:, self.src_idx] + self.delS(z)
        else:
            x_dec = self.Dx(z)

        x_hat = post_process(x_dec, list(params))

        if return_source:
            # Source reconstruction (z -> y) for a consistency / cycle loss.
            y_hat = post_process(y_dec, list(params))
            return x_hat, y_hat, z

        return x_hat, z

    @torch.no_grad()
    def project(self):
        # Clamps thresholds t and projects A, B (hence Dy = B[0]) onto the unit ball.
        super().project()
        if self.coupling == "independent":
            # Apply the same unit-ball constraint to the independent target dictionary.
            self.Dx.weight = uball_project(self.Dx.weight)
        else:
            # delS is not itself a dictionary -- the object that must stay on the unit ball
            # is the EFFECTIVE target dictionary Dy[:, src_idx] + delS (same constraint set
            # as the independent branch). Project it, then store the residual back in delS.
            # Runs AFTER super().project(), so Dy is already normalized; at init delS = 0
            # makes this an exact no-op.
            Dy_src = self.Dy.weight[:, self.src_idx]
            self.delS.weight = uball_project(Dy_src + self.delS.weight) - Dy_src
