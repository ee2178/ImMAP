import torch
import numpy as np
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

LOSS_REGISTRY = {
    "complex-mse": complex_mse,
    "magnitude-mse": magnitude_mse,
    "magnitude-l1":magnitude_l1, 
    "sigma-scaled-complex-mse": sigma_scaled_complex_mse,
    "complex-nl1-nl2": complex_nl1_nl2,
    "magnitude-nl1-nl2": mag_nl1_nl2,
    "vgg-feature": vgg_feature_loss,
}
