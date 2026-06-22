import torch
import numpy as np
import lpips

# Cache models by net name so we build AlexNet/VGG only once.
_LPIPS_CACHE = {}


def _get_lpips(net, device):
    model = _LPIPS_CACHE.get(net)
    if model is None:
        model = lpips.LPIPS(net=net)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        _LPIPS_CACHE[net] = model
    # No-op when already on `device`; keeps lazy init device-agnostic.
    return model.to(device)


def _reduce(t, percentile):
    """Per (batch, channel) scale, shape (N, C, 1, 1)."""
    flat = t.flatten(start_dim=2)  # (N, C, H*W)
    if percentile >= 1.0:
        val = flat.amax(dim=-1)
    else:
        val = torch.quantile(flat, percentile, dim=-1)
    return val.unsqueeze(-1).unsqueeze(-1)


def _prepare_lpips_inputs(x, y, scale=None, percentile=0.99):
    x = x.abs()
    y = y.abs()

    if scale is None:
        # Joint-to-target: reference both to the target's stat, detached so the
        # normalizer is a fixed reference and gradients don't flow through it.
        scale = _reduce(y, percentile).clamp_min(1e-8).detach()
    else:
        # Fixed constant (e.g. dataset-level dynamic range).
        scale = torch.as_tensor(scale, dtype=x.dtype, device=x.device)

    # Clamp to [0, 1]: joint scaling can push values past 1, and with
    # percentile < 1 the target's brightest pixels exceed `scale` too.
    x = (x / scale).clamp(0, 1)
    y = (y / scale).clamp(0, 1)

    # Single channel -> RGB (LPIPS backbones expect 3 channels).
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
        y = y.repeat(1, 3, 1, 1)

    # [0, 1] -> [-1, 1] (default LPIPS normalize=False expects this range).
    x = 2 * x - 1
    y = 2 * y - 1
    return x, y


# `sigma` is kept for signature compatibility with the loss registry; unused.
def lpips_alex(x, y, sigma=None, scale=None, percentile=0.99):
    x, y = _prepare_lpips_inputs(x, y, scale=scale, percentile=percentile)
    model = _get_lpips("alex", x.device)
    return model(x, y).mean()


def lpips_vgg(x, y, sigma=None, scale=None, percentile=0.99):
    x, y = _prepare_lpips_inputs(x, y, scale=scale, percentile=percentile)
    model = _get_lpips("vgg", x.device)
    return model(x, y).mean()


def complex_mse(x, y, sigma):
    return torch.mean((x - y).abs() ** 2)


def magnitude_mse(x, y, sigma):
    return torch.mean((x.abs() - y.abs()) ** 2)


def sigma_scaled_complex_mse(x, y, sigma):
    return torch.mean((sigma + 1e-3) ** (-2) * (x - y).abs() ** 2)

def magnitude_l1(x, y, sigma):
    return torch.mean(torch.abs(x.abs() - y.abs()))

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
    "lpips-alex": lpips_alex,
    "lpips-vgg": lpips_vgg,
}
