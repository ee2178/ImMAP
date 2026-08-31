import torch
import numpy as np
import torchvision
import lpips

# ---------------------------------------------------------------------------
# VGG16 block4_conv3 feature-MSE perceptual loss
# Faithful port of the TF reference `custom_perceptualLoss`
# (CCL-Synthetis/Synthesis/synthesis_losses.py). NOT the same as lpips-vgg:
# single layer, raw feature MSE, no learned weights.
# ---------------------------------------------------------------------------
_VGG_CACHE = {}
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _get_vgg_relu43(device):
    """VGG16 (ImageNet) truncated at relu4_3 == Keras 'block4_conv3' output.
    torchvision `features[:23]` keeps blocks 1-4 up to and including relu4_3
    (index 23 is block4's maxpool, excluded). Frozen + eval, cached per device.
    Needs the ImageNet VGG16 weights (downloaded/cached on first use)."""
    key = str(device)
    model = _VGG_CACHE.get(key)
    if model is None:
        try:
            vgg = torchvision.models.vgg16(
                weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
        except AttributeError:                      # older torchvision API
            vgg = torchvision.models.vgg16(pretrained=True)
        model = vgg.features[:23].eval()
        for p in model.parameters():
            p.requires_grad_(False)
        _VGG_CACHE[key] = model
    return model.to(device)


def _imagenet_prep(t):
    """Per-sample min-max to [0,1] then ImageNet mean/std. t: (B, 3, H, W)."""
    lo = t.amin(dim=(1, 2, 3), keepdim=True)
    hi = t.amax(dim=(1, 2, 3), keepdim=True)
    t = (t - lo) / (hi - lo).clamp_min(1e-8)
    mean = t.new_tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
    std = t.new_tensor(_IMAGENET_STD).view(1, 3, 1, 1)
    return (t - mean) / std


def vgg_feature_loss(x, y, sigma=None, weight=0.5, size_normalize=True, imagenet_norm=False):
    """Single-layer VGG16 (block4_conv3 / relu4_3) feature-MSE perceptual loss.

    Faithful port of the TF reference: for each output channel, replicate grayscale
    -> RGB, extract block4_conv3 features, take the feature MSE, optionally divide by
    the feature-map size (the reference's 1/(H*W*C) factor), scale by `weight`, and
    sum over channels.

    x, y : (B, C, H, W). Brain masking is applied by the training loop before this call.
    weight         : reference perceptualLoss_weight (0.5).
    size_normalize : reference multiplies the (already mean-reduced) MSE by 1/(H*W*C);
                     keep True to match it exactly, False for a plain mean feature-MSE.
    imagenet_norm  : the reference feeds RAW (un-preprocessed) intensities to VGG
                     (default False). True per-sample min-maxes to [0,1] then applies
                     ImageNet mean/std (how the backbone was trained) -- better
                     conditioned, but a deviation from the reference.
    """
    net = _get_vgg_relu43(x.device)
    total = x.new_zeros(())
    for c in range(x.shape[1]):
        xc = x[:, c:c + 1].repeat(1, 3, 1, 1)
        yc = y[:, c:c + 1].repeat(1, 3, 1, 1)
        if imagenet_norm:
            xc, yc = _imagenet_prep(xc), _imagenet_prep(yc)
        fx, fy = net(xc), net(yc)
        mse = ((fx - fy) ** 2).mean()               # mean over all elements (Keras MSE)
        if size_normalize:
            _, CH, H, W = fx.shape
            mse = mse / (H * W * CH)                 # reference's extra 1/(H*W*C)
        total = total + weight * mse
    return total

def complex_mse(x, y, sigma):
    return torch.mean((x - y).abs() ** 2)


def magnitude_mse(x, y, sigma):
    return torch.mean((x.abs() - y.abs()) ** 2)


def sigma_scaled_complex_mse(x, y, sigma):
    return torch.mean((sigma + 1e-3) ** (-2) * (x - y).abs() ** 2)

def magnitude_l1(x, y, sigma):
    return torch.mean(torch.abs(x - y))

def complex_nl1_nl2(x, y, sigma, eps=1e-8):
    diff = x - y

    nl1 = (
        (diff.abs() + eps).mean(dim=(1, 2, 3))
        / (x.abs() + eps).mean(dim=(1, 2, 3))
    ).mean()

    nl2 = (
        diff.pow(2).mean(dim=(1, 2, 3))
        / (x.pow(2).mean(dim=(1, 2, 3)) + eps)
    ).sqrt().mean()

    return nl1 + nl2


def mag_nl1_nl2(x, y, sigma, eps=1e-8):
    return complex_nl1_nl2(x.abs(), y.abs(), sigma, eps=eps)

def mse_vgg(x, y, sigma=None, vgg_weight=1.0, mse_weight=1.0,
            size_normalize=False, imagenet_norm=False):
    """MSE + `vgg_weight` * VGG(relu4_3) feature-MSE -- the perception/distortion combination.

    WHY THIS EXISTS SEPARATELY FROM `vgg-feature`. The TF reference is PURE perceptual: its
    `Recon_perceptualLoss` returns `perceptualLoss_weight * style_loss` and nothing else (there
    is no MAE term, despite what the config comment there says). `vgg_feature_loss` ports that
    faithfully and should stay that way.

    That objective is a poor fit for an I2SB regressor. VGG features are largely invariant to a
    global affine change of intensity, so a pure perceptual loss does not pin the ABSOLUTE scale
    of x0_hat -- and the reverse bridge mixes x0_hat with the running state x_t on an absolute
    scale at every step (see sb.base.reverse_sample). A drift the perceptual loss cannot see
    therefore compounds over NFE steps. The pixel term is what anchors it.

    SETTING vgg_weight. The two terms are not naturally commensurate, and the reference's
    `size_normalize` makes the gap enormous: it divides the feature MSE by H*W*CH, which at a
    192px input is ~1/295000, so the perceptual term would contribute nothing next to a pixel
    MSE. This function therefore defaults to `size_normalize=False` (a plain mean feature-MSE).
    Even then the scale is data-dependent -- use `vgg_mse_balance` below, which train_i2sb prints
    on the first step, and pick `vgg_weight` from the ratio it reports rather than guessing.
    """
    mse = torch.mean((x - y).abs() ** 2)
    vgg = vgg_feature_loss(x, y, weight=1.0, size_normalize=size_normalize,
                           imagenet_norm=imagenet_norm)
    return mse_weight * mse + vgg_weight * vgg


@torch.no_grad()
def vgg_mse_balance(x, y, size_normalize=False, imagenet_norm=False):
    """(mse, vgg, vgg_weight_for_parity) on one batch -- the numbers needed to set `vgg_weight`.

    The third value is mse/vgg: the weight at which the two terms contribute equally on this
    batch. A sensible perceptual run sits BELOW it (the pixel term should still dominate), but
    knowing where parity is beats guessing across five orders of magnitude.
    """
    mse = float(torch.mean((x - y).abs() ** 2))
    vgg = float(vgg_feature_loss(x, y, weight=1.0, size_normalize=size_normalize,
                                 imagenet_norm=imagenet_norm))
    return mse, vgg, (mse / vgg if vgg > 0 else float("nan"))


LOSS_REGISTRY = {
    "complex-mse": complex_mse,
    "magnitude-mse": magnitude_mse,
    "magnitude-l1":magnitude_l1,
    "sigma-scaled-complex-mse": sigma_scaled_complex_mse,
    "complex-nl1-nl2": complex_nl1_nl2,
    "magnitude-nl1-nl2": mag_nl1_nl2,
    "vgg-feature": vgg_feature_loss,   # PURE perceptual -- the faithful TF port
    "mse-vgg": mse_vgg,                # MSE + vgg_weight * perceptual (see mse_vgg's docstring)
}

# Losses that accept extra keyword arguments from cfg["training"]["loss_params"]. Everything
# else takes only (x, y, sigma), so passing params to them is a config error worth catching.
LOSS_PARAM_KEYS = {
    "vgg-feature": {"weight", "size_normalize", "imagenet_norm"},
    "mse-vgg": {"vgg_weight", "mse_weight", "size_normalize", "imagenet_norm"},
}


# ---------------------------------------------------------------------------
# Per-pixel region weighting (enhancing-tumor upweighting)
# ---------------------------------------------------------------------------
# The elementwise error each mean-reduced loss above averages. Only losses that ARE a
# per-pixel mean can be weighted per pixel; the ratio and perceptual entries in
# LOSS_REGISTRY are not, and weighted_loss refuses them rather than approximating.
POINTWISE_REGISTRY = {
    "complex-mse":   lambda x, y: (x - y).abs() ** 2,
    "magnitude-mse": lambda x, y: (x.abs() - y.abs()) ** 2,
    "magnitude-l1":  lambda x, y: (x - y).abs(),
}


def weighted_loss(loss_type, pred, target, region, weight):
    """`loss_type` with pixels inside `region` counted `weight` times as heavily:

        L = mean_i( w_i * err_i ) ,      w_i = 1 + (weight - 1) * region_i

    i.e. `mean(err) over non-region  +  weight * mean(err) over region`, both divided by
    the SAME total pixel count. `weight = 1` is exactly the unweighted loss -- the term is
    a no-op by construction rather than by a branch, and an empty region needs no special
    case.

    Replaces the area-normalized `training.common.region_loss`, which computed a MEAN OVER
    THE REGION (`sum_region(err) / |region|`). That form's gradient inside the region
    scales as 1/|region|, and enhancing tumor is well under 1% of a slice and absent from
    many, so the per-pixel gradient swung ~2 orders of magnitude batch to batch (measured:
    the ET term's gradient norm ran 2x the whole-brain term's on a large tumor and 190x on
    a 1-pixel one). `clip_grad_norm_` cannot undo that -- it rescales the sum, preserving
    the ratio -- so the update DIRECTION was set by a handful of pixels, at full Adam step
    size. Here the per-pixel gradient is `w_i * d(err_i)/d(pred_i) / N`: bounded, and
    independent of how much tumor the batch happens to contain.

    It also makes every batch minimize the SAME objective. The old version returned 0 on
    ET-free batches, so those steps descended a different function -- the estimator was not
    just noisy but inconsistent.

    Scale note: `weight` is PER PIXEL, so the region's share of the objective is
    approximately `f*weight / (1 - f + f*weight)` for a region area fraction `f`. ET runs
    around f = 0.003, so weight = 330 puts ~50% of the objective on the tumor (what the old
    `lam_et = 1.0` did, via the 1/|region| normalization), weight = 50 puts ~13% there, and
    weight = 1 is off. Expect to tune in the tens-to-hundreds, not around 1.

    `region` broadcasts against `pred` (a (B, 1, H, W) mask over a (B, C, H, W) prediction
    is the intended use). Both `pred` and `target` are expected to be brain-masked already,
    exactly as with the unweighted loss.
    """
    if weight == 1:
        return LOSS_REGISTRY[loss_type](pred, target, None)
    if loss_type not in POINTWISE_REGISTRY:
        raise ValueError(
            f"loss_type {loss_type!r} is not a per-pixel mean, so it cannot be weighted "
            f"per pixel; got weight={weight}. Use one of "
            f"{sorted(POINTWISE_REGISTRY)}, or set the region weight to 1.")
    w = 1.0 + (weight - 1.0) * region.expand_as(pred)
    return torch.mean(w * POINTWISE_REGISTRY[loss_type](pred, target))
