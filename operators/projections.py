import torch


# Which axes `uball_project` reduces over, by name.
#
#   "slice"  each (out, in) spatial slice has unit norm.  This is what the
#            repo and Sljiva have always done (`dims=1:ndims(W)-2` there, the
#            spatial axes in Julia's (kW, kH, in, out) layout).
#   "atom"   each ATOM has unit norm -- axis 0 is the coefficient index for
#            both `Conv2d` (out, in, kH, kW) and `ConvTranspose2d`
#            (in, out, kH, kW), so reducing everything but axis 0 is the
#            per-atom constraint in either direction.
#
# The two COINCIDE whenever axis 1 has length 1, which is every `C -> M` conv
# in CDLNet, LPDSNet and both multigrid families -- Sljiva never builds an
# analysis with more than C input channels, so its projection has only ever run
# the degenerate case.  They differ first at levels >= 2 of the multilevel nets,
# where "slice" admits atoms a factor sqrt(M_{l-1}) larger (~7x at 48 input
# channels, ~10x at 96) and that factor compounds down the cascade.
#
# Neither is the operator unit ball: measured on random filters at P=7, s=2,
# `||A||_2` lands at ~8 (1->169), ~12 (48->96, slice) and ~1.7 (48->96, atom)
# against a target of 1.  Only `spectral_normalize` sets that, and only at init.
PROJ_DIMS = {"slice": (2, 3), "atom": (1, 2, 3)}


def proj_dims(mode):
    """Axes for `uball_project`, from a `PROJ_DIMS` key."""
    try:
        return PROJ_DIMS[mode]
    except KeyError:
        raise ValueError(
            "proj_mode must be one of %s; got %r"
            % (sorted(PROJ_DIMS), mode)) from None


def uball_project(W, dim=(2, 3)):
    """
    Project tensor onto the unit ball along specified dimensions.

    Parameters
    ----------
    W : torch.Tensor
        Input tensor.
    dim : tuple
        Dimensions over which to compute the norm.  See `PROJ_DIMS`.

    Returns
    -------
    torch.Tensor
        Projected tensor.
    """
    # Explicit sum of squares rather than `torch.linalg.norm`: that caps `dim`
    # at length 2 (torch 1.12), which rules out the 3-axis "atom" reduction.
    # `abs()` first keeps this correct for complex weights -- `W.pow(2)` would
    # square the complex value, not its modulus.
    normW = W.abs().pow(2).sum(dim=dim, keepdim=True).sqrt()

    return W * torch.clamp(1 / normW, max=1)
