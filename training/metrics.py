import torch
import torch.nn.functional as F

from utils.transforms import gaussian_window


# ============================================================
# Helper utilities
# ============================================================

def joint_normalize(x, y):
    """
    Joint min-max normalization.

    Parameters
    ----------
    x : torch.Tensor
    y : torch.Tensor

    Returns
    -------
    x_norm : torch.Tensor
    y_norm : torch.Tensor
    """

    x = x.squeeze()
    y = y.squeeze()

    xy = torch.cat((x, y), dim=0)

    min_xy = xy.min()
    max_xy = xy.max()

    x_norm = (x - min_xy) / (max_xy - min_xy + 1e-12)
    y_norm = (y - min_xy) / (max_xy - min_xy + 1e-12)

    return x_norm, y_norm


# ============================================================
# PSNR
# ============================================================

def psnr(
    gt,
    pred,
    data_range=1.0,
    eps=1e-12,
):
    """
    Complex-valued PSNR.

    Parameters
    ----------
    data_range : float
        Peak-to-peak range of the signal. 1.0 (the default) reproduces the previous
        `-10*log10(mse)`. Use 2.0 for data in [-1, 1], and the measured
        max-min for anything else -- PSNR is only comparable between runs that
        assume the SAME range, and getting it wrong shifts every number by
        20*log10(data_range) dB (6.02 dB for 1 -> 2).

    Returns
    -------
    torch.Tensor
        Scalar tensor.
    """

    mse = torch.mean(
        (gt - pred).abs() ** 2
    )

    return 10 * torch.log10(
        data_range ** 2 / (mse + eps)
    )


# ============================================================
# NRMSE
# ============================================================

def nrmse(
    gt,
    pred,
    eps=1e-12,
):
    """
    Normalized RMSE using joint dynamic range.

    Deliberately takes no `data_range`: it divides by the range MEASURED from
    (gt, pred), so it is already scale-free and a nominal range would double-count.

    Returns
    -------
    torch.Tensor
        Scalar tensor.
    """

    rmse = torch.sqrt(
        torch.mean(
            (gt - pred).abs() ** 2
        )
    )

    xy = torch.cat(
        (gt, pred),
        dim=0,
    )

    dyn_range = (
        xy.max() - xy.min()
    )

    return rmse / (
        dyn_range + eps
    )


# ============================================================
# SSIM
# ============================================================

def ssim(
    gt,
    pred,
    data_range=1.0,
    window_size=11,
    K1=1e-2,
    K2=3e-2,
    C1=None,
    C2=None,
):
    """
    Complex-valued SSIM using magnitude images.

    Parameters
    ----------
    gt : torch.Tensor
        Shape (B, C, H, W)

    pred : torch.Tensor
        Shape (B, C, H, W)

    data_range : float
        Peak-to-peak range of the signal. The stability constants are
        C1 = (K1*data_range)^2, C2 = (K2*data_range)^2 -- SSIM depends on the
        range just as PSNR does, and leaving them at their data_range=1 values on
        [-1, 1] data quietly changes what is being measured (the constants stop
        being small relative to the local variances they regularize).
        data_range=1.0 reproduces the previous C1=(1e-2)^2, C2=(3e-2)^2.

    C1, C2 : float, optional
        Explicit overrides; when given, `data_range` / K1 / K2 are ignored.

    Returns
    -------
    torch.Tensor
        Shape (B,)
    """

    C1 = (K1 * data_range) ** 2 if C1 is None else C1
    C2 = (K2 * data_range) ** 2 if C2 is None else C2

    gt_mag = gt.abs()
    pred_mag = pred.abs()

    width = (window_size - 1) // 2

    window = gaussian_window(width).to(gt.device)

    window = window.expand(
        gt.shape[1],
        1,
        window_size,
        window_size,
    )

    pad = window_size // 2

    mu_x = F.conv2d(
        gt_mag,
        window,
        padding=pad,
        groups=gt.shape[1],
    )

    mu_y = F.conv2d(
        pred_mag,
        window,
        padding=pad,
        groups=gt.shape[1],
    )

    mu_x2 = mu_x ** 2
    mu_y2 = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x2 = (
        F.conv2d(
            gt_mag * gt_mag,
            window,
            padding=pad,
            groups=gt.shape[1],
        ) - mu_x2
    )

    sigma_y2 = (
        F.conv2d(
            pred_mag * pred_mag,
            window,
            padding=pad,
            groups=gt.shape[1],
        ) - mu_y2
    )

    sigma_xy = (
        F.conv2d(
            gt_mag * pred_mag,
            window,
            padding=pad,
            groups=gt.shape[1],
        ) - mu_xy
    )

    numerator = (
        (2 * mu_xy + C1)
        * (2 * sigma_xy + C2)
    )

    denominator = (
        (mu_x2 + mu_y2 + C1)
        * (sigma_x2 + sigma_y2 + C2)
    )

    ssim_map = numerator / (
        denominator + 1e-12
    )

    return ssim_map.mean(
        dim=(1, 2, 3)
    )


def compute_metrics(gt, recon, psnr_only=False, data_range=1.0):
    """PSNR / NRMSE / SSIM. `data_range` is the signal's peak-to-peak range and feeds
    PSNR and SSIM (NRMSE measures its own -- see nrmse). Default 1.0 keeps every
    existing caller bit-identical; pass 2.0 for data in [-1, 1]."""
    if psnr_only:
        return {
            "psnr": psnr(gt, recon, data_range=data_range),
        }
    else:
        return {
            "psnr": psnr(gt, recon, data_range=data_range),
            "nrmse": nrmse(gt, recon),
            "ssim": ssim(gt, recon, data_range=data_range).mean(),
        }

