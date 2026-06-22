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
    eps=1e-12,
):
    """
    Complex-valued PSNR.

    Returns
    -------
    torch.Tensor
        Scalar tensor.
    """

    mse = torch.mean(
        (gt - pred).abs() ** 2
    )

    return -10 * torch.log10(
        mse + eps
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
    window_size=11,
    C1=(1e-2) ** 2,
    C2=(3e-2) ** 2,
):
    """
    Complex-valued SSIM using magnitude images.

    Parameters
    ----------
    gt : torch.Tensor
        Shape (B, C, H, W)

    pred : torch.Tensor
        Shape (B, C, H, W)

    Returns
    -------
    torch.Tensor
        Shape (B,)
    """

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


def compute_metrics(gt, recon, psnr_only = False):
    if psnr_only:
        return {
            "psnr": psnr(gt, recon),
        }
    else:
        return {
            "psnr": psnr(gt, recon),
            "nrmse": nrmse(gt, recon),
            "ssim": ssim(gt, recon).mean(),
        }

