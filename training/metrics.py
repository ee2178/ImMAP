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


def _as_weight(mask, like):
    """A broadcast float weight from a boolean (or float) `organ_mask`.

    Returns None for `mask=None` so every metric has ONE unmasked path rather
    than an all-ones weight that would still change the arithmetic.

    The mask is the support of the coil sensitivities -- `datasets/fastmri`
    returns `smaps.abs().sum(1, keepdim=True) > 0` -- so it is (B, 1, H, W) and
    broadcasts over a multi-channel image.
    """
    if mask is None:
        return None
    dtype = like.real.dtype if torch.is_complex(like) else like.dtype
    w = mask.to(dtype=dtype, device=like.device)
    return w.expand_as(like) if w.shape != like.shape else w


def _nan_if_empty(value, area):
    """`value`, or NaN where `area` is 0.

    An empty organ mask means a broken sensitivity map, not a legitimate slice.
    NaN leaves a visible gap in the curve; returning 0 would read as a training
    collapse, and averaging it into the val mean would bury the cause. Written
    with torch.where rather than a Python `if` so it costs no device sync --
    this runs once per validation batch.
    """
    return torch.where(area > 0, value, value.new_full((), float("nan")))


# ============================================================
# PSNR
# ============================================================

def psnr(
    gt,
    pred,
    data_range=1.0,
    eps=1e-12,
    mask=None,
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

    mask : torch.Tensor, optional
        Binary organ mask. The MSE is then taken over MASK PIXELS ONLY -- the
        denominator is the mask area, not the pixel count. Zeroing the pair and
        calling the unmasked version instead would divide the same numerator by
        the FULL count and report an artificially high PSNR that scales with
        how much background the slice happens to contain, which is exactly the
        dependence masking is meant to remove.

        Masked and unmasked PSNR are not comparable to each other. A run with
        the mask on cannot be put in the same table as one without it.

    Returns
    -------
    torch.Tensor
        Scalar tensor. NaN if the mask is empty.
    """

    se = (gt - pred).abs() ** 2
    w = _as_weight(mask, se)

    if w is None:
        return 10 * torch.log10(data_range ** 2 / (se.mean() + eps))

    area = w.sum()
    mse = (se * w).sum() / area.clamp_min(1)

    return _nan_if_empty(
        10 * torch.log10(data_range ** 2 / (mse + eps)), area
    )


# ============================================================
# NRMSE
# ============================================================

def nrmse(
    gt,
    pred,
    eps=1e-12,
    mask=None,
):
    """
    Normalized RMSE using joint dynamic range.

    Deliberately takes no `data_range`: it divides by the range MEASURED from
    (gt, pred), so it is already scale-free and a nominal range would double-count.

    Parameters
    ----------
    mask : torch.Tensor, optional
        Binary organ mask. BOTH halves are restricted to it: the RMSE averages
        over mask pixels, and the dynamic range is the span of the masked
        pixels. Restricting only the numerator would divide a masked error by
        an unmasked span and quietly re-admit the background -- which for a
        knee slice is most of the image, and whose min sets the bottom of the
        range.

        Takes real (magnitude) input, as it already did: `.max()` on a complex
        tensor raises.

    Returns
    -------
    torch.Tensor
        Scalar tensor. NaN if the mask is empty.
    """

    se = (gt - pred).abs() ** 2
    w = _as_weight(mask, se)

    xy = torch.cat(
        (gt, pred),
        dim=0,
    )

    if w is None:
        rmse = torch.sqrt(torch.mean(se))
        dyn_range = xy.max() - xy.min()
        return rmse / (dyn_range + eps)

    area = w.sum()
    rmse = torch.sqrt((se * w).sum() / area.clamp_min(1))

    ww = torch.cat((w, w), dim=0)
    hi = torch.where(ww > 0, xy, xy.new_full((), float("-inf"))).max()
    lo = torch.where(ww > 0, xy, xy.new_full((), float("inf"))).min()

    return _nan_if_empty(rmse / (hi - lo + eps), area)


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
    mask=None,
    erode_mask=True,
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

    mask : torch.Tensor, optional
        Binary organ mask restricting the AVERAGE ONLY. The SSIM map is still
        computed from the unmasked images, so the local means and variances are
        the true ones and only their spatial average is taken over the organ.

        This is deliberately NOT `ssim(gt * mask, pred * mask)`. Zeroing both
        inputs fills the background with pixels that agree exactly, and a
        constant region scores SSIM ~ 1, so the masked-input version would
        REPORT A HIGHER SSIM the more background a slice has -- the opposite of
        what is wanted.

    erode_mask : bool
        Shrink the averaging region by the window radius, so only windows lying
        ENTIRELY inside the organ are counted. Without it a masked SSIM is not
        actually background-independent: a window centred one pixel inside the
        boundary still reads `window_size // 2` pixels of background into its
        local mean and variance, so corrupting the background moves the map
        INSIDE the mask. Measured at 0.936 -> 0.861 on a disc phantom when the
        background was replaced with noise at half the signal amplitude; with
        erosion the same change moves nothing.

        The cost is `window_size // 2` pixels of real organ at the rim (5 by
        default), which is where reconstruction error is often largest -- so
        this is a trade, not a free fix. Set False to keep the rim and accept
        the leak. PSNR and NRMSE are pointwise and need no such allowance,
        which is why they use the mask as given: the two metrics are therefore
        averaged over slightly different regions, each the right one for its
        own definition.

    Returns
    -------
    torch.Tensor
        Shape (B,). NaN for any sample whose mask is empty.
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

    if mask is None:
        return ssim_map.mean(
            dim=(1, 2, 3)
        )

    w = _as_weight(mask, ssim_map).contiguous()

    if erode_mask and width > 0:
        # Min-filter == erosion by the window radius.
        #
        # The explicit F.pad is load-bearing. max_pool2d's own `padding` pads
        # with -inf, so `-max_pool2d(-w, padding=width)` would leave the IMAGE
        # border un-eroded -- a mask running to the edge would keep windows
        # that SSIM itself computed against its conv's zero padding. Padding
        # the mask with 0 first says "outside the image is background", which
        # is the same thing the conv assumed.
        w = F.pad(w, (width,) * 4, value=0.0)
        w = -F.max_pool2d(-w, kernel_size=2 * width + 1, stride=1)

    area = w.sum(dim=(1, 2, 3))

    return _nan_if_empty(
        (ssim_map * w).sum(dim=(1, 2, 3)) / area.clamp_min(1), area
    )


def compute_metrics(gt, recon, psnr_only=False, data_range=1.0, mask=None):
    """PSNR / NRMSE / SSIM. `data_range` is the signal's peak-to-peak range and feeds
    PSNR and SSIM (NRMSE measures its own -- see nrmse). Default 1.0 keeps every
    existing caller bit-identical; pass 2.0 for data in [-1, 1].

    `mask` is the `organ_mask` the recon loader returns (the coil-sensitivity
    support). Passing it restricts all three metrics to that region -- PSNR and
    NRMSE by averaging their error there, SSIM by averaging its map over an
    ERODED copy of it; see each function for why those are three different
    operations rather than one. `mask=None` is
    bit-identical to before.

    Masked and unmasked numbers are NOT interchangeable. Do not compare a run
    trained and scored with the mask on against one scored without it."""
    if psnr_only:
        return {
            "psnr": psnr(gt, recon, data_range=data_range, mask=mask),
        }
    else:
        return {
            "psnr": psnr(gt, recon, data_range=data_range, mask=mask),
            "nrmse": nrmse(gt, recon, mask=mask),
            "ssim": ssim(gt, recon, data_range=data_range, mask=mask).mean(),
        }

