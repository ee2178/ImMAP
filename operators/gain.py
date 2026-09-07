# -*- coding: utf-8 -*-
"""
ChannelGain -- a per-sample, per-channel real scaling, as a linear operator.

Written for the Schrodinger-bridge multigrid nets (models/sb_multigrid.py), where it is what
turns SBCDLNet's TWO-fidelity objective into an ordinary ONE-fidelity CDL problem, so the whole
multigrid V-cycle machinery applies unchanged. The reduction:

    SBCDLNet step   g = mu_0 A_D(mu_0 B_D z - r)  +  A_P(B_P z - c)
    LISTA step      g = A( E^H E (B z) - y~ )                       (models/lista.py)

Stack the two dictionary pairs into ONE (1 + n_cond)-channel pair -- A = [A_D, A_P],
B = [B_D; B_P], which is exactly what a single Conv2d/ConvTranspose2d of that width computes,
since a convolution already sums over input channels. Then with

    E   = ChannelGain(diag(mu_0, 1, ..., 1))      so  E^H E = diag(mu_0^2, 1, ..., 1)
    y   = [r ; c]                                 so  y~ = E^H y = [mu_0 r ; c]

the LISTA step becomes

    A( diag(mu_0^2, 1..) B z - [mu_0 r ; c] )
      = A_D(mu_0^2 B_D z - mu_0 r) + A_P(B_P z - c)
      = mu_0 A_D(mu_0 B_D z - r)   + A_P(B_P z - c)                 == the SBCDLNet step.

Verified to float32 rounding in tests; the two-fidelity structure is therefore not a special case
the V-cycle has to know about, it is a choice of measurement operator. The bridge's endpoint
behaviour survives intact: mu_0 -> 0 at the prior end sends the target block of the gradient to
zero quadratically, so the code there is set by the prior fidelity alone.

The single learned step this implies (SBCDLNet gives its two fidelities SEPARATE learned steps
eta_k, nu_k) is a real difference -- see models/sb_multigrid.py's docstring.

COMMUTES WITH RESAMPLING. The gain is constant in space, so it commutes exactly with restriction
and prolongation and its Galerkin coarsening is itself. `commutes_with_resample = True` tells
`operators.resample.galerkin` to skip building `E . P`, which would otherwise wrap the coarse
Gram in a spurious R.P smoothing (and cost two resamples per coarse Gram apply).
"""

import torch

from operators.base import Operator


class ChannelGain(Operator):
    """Multiply channel c of every sample by `gains[b, c]`.

    Parameters
    ----------
    gains : Tensor broadcastable to (B, C, 1, 1)
        Real, non-negative by convention (nothing enforces it; a negative gain is a well-defined
        operator, it just has no meaning here).

    The operator is real and diagonal, so it is its own adjoint and its Gram is the elementwise
    square. Nothing is learned -- `gains` is data, recomputed per forward from the bridge step.
    """

    #: spatially constant, so galerkin() can return it unchanged (see the module docstring)
    commutes_with_resample = True

    def __init__(self, gains):
        g = torch.as_tensor(gains)
        if g.dim() == 1:                      # (C,) -> broadcast over the batch
            g = g.view(1, -1, 1, 1)
        while g.dim() < 4:
            g = g.unsqueeze(-1)
        self.gains = g

    def forward(self, x):
        return self.gains.to(device=x.device, dtype=x.dtype) * x

    def adjoint(self, x):
        return self.forward(x)                # real diagonal -> self-adjoint

    def gram(self, x):
        g = self.gains.to(device=x.device, dtype=x.dtype)
        return (g * g) * x

    def __repr__(self):
        g = self.gains
        return (f"ChannelGain(shape={tuple(g.shape)}, "
                f"range=[{float(g.min()):.4g}, {float(g.max()):.4g}])")


def bridge_gain(mu0, n_cond, like=None):
    """The SB measurement operator at one bridge step: `diag(mu_0, 1, ..., 1)`.

    `mu_0` is the target-fidelity trust weight from the schedule, shaped (B, 1, 1, 1); the
    conditioning channels are passed through unscaled because the prior fidelity does not weaken
    along the bridge.
    """
    mu0 = mu0.reshape(-1, 1, 1, 1)
    ones = torch.ones(mu0.shape[0], int(n_cond), 1, 1,
                      device=mu0.device, dtype=mu0.dtype)
    return ChannelGain(torch.cat([mu0, ones], dim=1))
