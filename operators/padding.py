import numpy as np
import torch.nn.functional as F


def calc_pad_1d(L, M):
    """
    Compute symmetric padding for a 1D signal.

    Parameters
    ----------
    L : int
        Signal length.
    M : int
        Desired divisor.

    Returns
    -------
    list[int]
        [left_pad, right_pad]
    """
    if L % M == 0:
        return [0, 0]

    Lprime = int(np.ceil(L / M) * M)
    Ldiff = Lprime - L

    return [
        int(np.floor(Ldiff / 2)),
        int(np.ceil(Ldiff / 2)),
    ]


def calc_pad_2d(H, W, M):
    """
    Compute symmetric padding for a 2D image.

    Parameters
    ----------
    H : int
        Image height.
    W : int
        Image width.
    M : int
        Desired divisor.

    Returns
    -------
    tuple
        (left, right, top, bottom)
    """
    return (
        *calc_pad_1d(W, M),
        *calc_pad_1d(H, M),
    )


def conv_pad(x, ks, mode="reflect"):
    """
    Pad tensor for same-sized convolution.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    ks : int
        Kernel size.
    mode : str
        Padding mode.

    Returns
    -------
    torch.Tensor
        Padded tensor.
    """
    pad = (
        int(np.floor((ks - 1) / 2)),
        int(np.ceil((ks - 1) / 2)),
    )

    return F.pad(x, (*pad, *pad), mode=mode)


def unpad(x, pad):
    """
    Remove 2D padding from tensor.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    pad : tuple
        (left, right, top, bottom)

    Returns
    -------
    torch.Tensor
        Unpadded tensor.
    """
    left, right, top, bottom = pad

    if bottom == 0 and right > 0:
        return x[..., top:, left:-right]

    elif bottom > 0 and right == 0:
        return x[..., top:-bottom, left:]

    elif bottom == 0 and right == 0:
        return x[..., top:, left:]

    else:
        return x[..., top:-bottom, left:-right]
