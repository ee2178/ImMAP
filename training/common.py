# training/common.py

import os
import json
import numpy as np
import torch
from physics.nle import whiten
from operators.fourier import ifftc
from models import build_model

def grad_norm(params):
    """
    Compute the ℓ2 norm of gradients.
    """

    total_norm = 0.0

    for p in params:

        if p.grad is None:
            continue

        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2

    return total_norm ** 0.5


def get_lr(optimizer):
    """
    Return learning rates for all parameter groups.
    """

    return [pg["lr"] for pg in optimizer.param_groups]


def set_lr(optimizer, lr):
    """
    Set optimizer learning rate(s).

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
    lr : float or list
    """

    if not isinstance(lr, (list, tuple, np.ndarray)):
        lr = [lr] * len(optimizer.param_groups)

    for i, pg in enumerate(optimizer.param_groups):
        pg["lr"] = lr[i]


def save_ckpt(
    path,
    model=None,
    step=None,
    optimizer=None,
    scheduler=None,
):
    """
    Save checkpoint.
    """

    def get_state_dict(obj):

        if obj is None:
            return None

        return obj.state_dict()

    torch.save(
        {
            "step": step,
            "model_state_dict": get_state_dict(model),
            "optimizer_state_dict": get_state_dict(optimizer),
            "scheduler_state_dict": get_state_dict(scheduler),
        },
        path,
    )


def load_ckpt(
    path,
    model=None,
    optimizer=None,
    scheduler=None,
    device="cpu",
):
    """
    Load checkpoint.
    """

    ckpt = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    def load_state_dict(obj, key):

        if obj is None:
            return obj

        state_key = f"{key}_state_dict"

        if state_key in ckpt and ckpt[state_key] is not None:

            print(f"Loading {key} state dict...")
            obj.load_state_dict(ckpt[state_key])

        return obj

    model = load_state_dict(model, "model")
    optimizer = load_state_dict(optimizer, "optimizer")
    scheduler = load_state_dict(scheduler, "scheduler")

    step = ckpt.get("step", 0) + 1

    return model, optimizer, scheduler, step


def save_args(
    args,
    save_dir,
    ckpt_path=None,
    filename="args.json",
):
    """
    Save experiment arguments/configuration.
    """

    args = dict(args)

    if ckpt_path is not None:

        if "paths" not in args:
            args["paths"] = {}

        args["paths"]["ckpt"] = ckpt_path

    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, filename)

    with open(save_path, "w") as f:
        json.dump(args, f, indent=4, sort_keys=True)


def count_parameters(model):
    """
    Count trainable parameters.
    """

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

### LOADING MODEL FROM CONFIG
def load_model(config_path, device="cpu"):
    """
    Load trained model from config + checkpoint.
    """

    # Load config
    with open(config_path, "r") as f:
        cfg = json.load(f)

    # Rebuild architecture
    model = build_model(cfg).to(device)

    # Load checkpoint
    ckpt = torch.load(
        cfg["paths"]["ckpt"],
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(ckpt["model_state_dict"])

    model.eval()

    return model


### MRI Recon/ ImMAP2.5 Specific
def apply_loss_mask(image, recon, organ_mask, use_mask):
    if use_mask:
        image = image * organ_mask
        recon = recon * organ_mask
    return image, recon


### Region-restricted supervision (enhancing-tumor / ET masks)
def region_loss(loss_fn, pred, target, region, eps=1e-8):
    """`loss_fn` restricted to a binary `region`, normalized by that region's AREA.

    Zeroing outside the region and calling a mean-reduced loss would still divide by the
    FULL pixel count. Enhancing tumor covers well under 1% of a slice, so the term would
    arrive ~100x smaller than it reads and the weight in front of it would be meaningless.
    Rescaling by numel/|region| converts that back into a mean over the region, so a
    weight of 1.0 means "count the ET region as heavily as the whole brain".

    Exact for the mean-reduced losses (magnitude-l1, complex-mse, magnitude-mse); for the
    ratio and perceptual entries in LOSS_REGISTRY it is a monotone reweighting, not a
    literal regional mean.

    Returns 0 when the region is empty. That is not an edge case -- plenty of slices
    contain no enhancing tumor -- and 0/0 would otherwise poison the batch's gradient.
    """
    region = region.expand_as(pred)
    area = region.sum()
    if float(area) <= 0:
        return pred.new_zeros(())
    return loss_fn(pred * region, target * region, None) * (region.numel() / area.clamp_min(eps))


def region_psnr(gt, pred, region, eps=1e-12):
    """PSNR with the MSE taken over `region` pixels only (0 if the region is empty).

    This is the number to watch when tuning lam_et. Global PSNR does respond to ET-only
    error, but weakly -- it averages over the ~99.7% of the slice that is not tumor, so a
    change that costs it a few hundredths of a dB costs et_psnr several dB (measured at
    roughly 20-250x more movement, the ratio growing as the error gets smaller).
    """
    region = region.expand_as(gt)
    area = region.sum()
    if float(area) <= 0:
        return gt.new_zeros(())
    mse = (((gt - pred) * region).abs() ** 2).sum() / area.clamp_min(eps)
    return -10 * torch.log10(mse + eps)


def snr_loss_weight(std_fwd, mode="uniform"):
    """Per-sample I2SB loss weight as a function of the forward std sigma_t = std_fwd, normalized to
    batch-mean 1 (so the loss scale, and thus LR / backtrack_thresh, stays comparable across modes):

        "uniform" -> 1
        "snr"     -> 1 / sigma_t^2   (emphasize LOW sigma_t ~ t=0; == the eps objective's implicit
                                      weight -- parameterization="eps" is the numerically stabler route)
        "t1"      -> sigma_t^2       (emphasize HIGH sigma_t ~ t=1; trains the initial reverse steps)

    std_fwd: (B, ...) or (B,) tensor of per-sample sigma_t. Returns a (B,) weight.
    """
    s2 = (std_fwd.reshape(std_fwd.shape[0], -1)[:, 0] ** 2).clamp_min(1e-12)   # (B,) sigma_t^2
    if mode == "uniform":
        return torch.ones_like(s2)
    if mode == "snr":
        w = 1.0 / s2
    elif mode == "t1":
        w = s2
    else:
        raise ValueError(f"loss_weight must be 'uniform', 'snr', or 't1'; got {mode!r}")
    return w / w.mean().clamp_min(1e-12)


def prepare_measurement(
    image, kspace, mask, smaps,
    kspace_type,
    noise_std,
    noise_dist,
    whiten_kspace,
):
    extra = {}

    if kspace_type == "simulated":
        y, sigma_n = mri_awgn(image, mask, smaps, noise_std, noise_dist)

    elif kspace_type == "measurement":

        if whiten_kspace:
            # Have to whiten from masked kspace
            kspace_w, smaps_w, Sigma_n, Zinv = whiten(mask*kspace, smaps)

            y = mask * kspace_w
            sigma_n = Sigma_n.max()

            extra["Zinv"] = Zinv
            extra["smaps"] = smaps_w
            # Regenerate image from fully sampled kspace
            extra["image_w"] = torch.sum(smaps_w.conj() * ifftc(kspace), dim=1, keepdim=True)

        else:
            # If using measurement kspace and not whitening, fix sigma_n = 0.01
            y = mask * kspace
            sigma_n = 0.01
            extra["smaps"] = smaps

    else:
        raise ValueError(f"Unknown kspace_type: {kspace_type}")

    sigma_n = torch.as_tensor(sigma_n, device=image.device, dtype=torch.float32)
    return y, sigma_n, extra
