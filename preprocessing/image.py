import torch

from operators.padding import calc_pad_2d
from operators.padding import unpad
import torch.nn.functional as F

def apply_pre_process(x, params):
    xmean, pad = params
    return F.pad(x - xmean, pad, mode="reflect")

def pre_process(x, stride, mask=1):
    """
    Mean subtraction + stride-compatible padding.

    Parameters
    ----------
    x : torch.Tensor
        Input image (B, C, H, W)
    stride : int
        Stride for downstream conv operators
    mask : torch.Tensor or float
        Optional weighting mask

    Returns
    -------
    x_out : torch.Tensor
        Preprocessed image
    params : list
        Stored statistics for inversion
    """

    params = []

    # -----------------------------
    # Mean subtraction
    # -----------------------------
    if torch.is_tensor(mask):
        xmean = x.sum(dim=(1, 2, 3), keepdim=True) / mask.sum(
            dim=(1, 2, 3), keepdim=True
        )
    else:
        xmean = x.mean(dim=(1, 2, 3), keepdim=True)

    x = mask * (x - xmean)
    params.append(xmean)

    # -----------------------------
    # Padding
    # -----------------------------
    pad = calc_pad_2d(*x.shape[2:], stride)

    x = F.pad(x, pad, mode="reflect")

    if torch.is_tensor(mask):
        mask = F.pad(mask, pad, mode="reflect")

    params.append(pad)

    return x, params


def pre_process_pair(x1, x2, stride, mask=1):
    """
    Joint preprocessing:
    - shared mean subtraction
    - shared padding

    Useful for: SSDU, diffusion conditioning, paired losses
    """

    params = []

    # -----------------------------
    # Joint mean
    # -----------------------------
    if torch.is_tensor(mask):
        total = (x1 + x2).sum(dim=(1, 2, 3), keepdim=True)
        denom = 2 * mask.sum(dim=(1, 2, 3), keepdim=True)
        xmean = total / denom
    else:
        xmean = (
            x1.mean(dim=(1, 2, 3), keepdim=True)
            + x2.mean(dim=(1, 2, 3), keepdim=True)
        ) / 2

    x1 = mask * (x1 - xmean)
    x2 = mask * (x2 - xmean)

    params.append(xmean)

    # -----------------------------
    # Padding
    # -----------------------------
    pad = calc_pad_2d(*x1.shape[2:], stride)

    x1 = F.pad(x1, pad, mode="reflect")
    x2 = F.pad(x2, pad, mode="reflect")

    if torch.is_tensor(mask):
        mask = F.pad(mask, pad, mode="reflect")

    params.append(pad)

    return x1, x2, params


def post_process(x, params):
    """
    Reverse preprocessing:
    - unpad
    - add mean back
    """

    pad = params.pop()
    xmean = params.pop()

    x = unpad(x, pad)
    x = x + xmean

    return x
