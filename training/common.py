# training/common.py

import os
import re
from decimal import Decimal, InvalidOperation
import json
import numpy as np
import torch
import yaml
from physics.nle import whiten
from operators.fourier import ifftc
from operators.noise import mri_awgn
from operators.truncate import Truncate, embed_operator
from models import build_model

# ===========================================================================
#  Per-step device syncs
# ===========================================================================
# `loss.item()` blocks until every queued CUDA kernel has finished. Calling it
# inside the training loop -- once to accumulate the epoch mean and again to
# refresh tqdm's postfix -- drains the queue TWICE per step, so the CPU cannot
# run ahead and enqueue the next step's work. The GPU then sits idle between
# steps, which is what a sawtooth in `nvidia-smi` utilisation actually is.
#
# Two fixes, applied to every loop in this package:
#   * accumulate `running_loss` as a 0-d CUDA tensor (`loss.detach()`) and
#     convert ONCE at the end of the epoch;
#   * refresh the progress bar every `POSTFIX_EVERY` steps rather than every
#     step. It is cosmetic, and one sync per 50 steps costs nothing.
POSTFIX_EVERY = 50


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


# write_config and its float-spelling helpers now live in training/config_io.py,
# which imports nothing from this repo and nothing from torch. They are
# re-exported here because `from training.common import write_config` is what
# the rest of the repo (and the sbatch bodies) already say.
from training.config_io import (            # noqa: F401
    _PLAIN_MAX_LEN,
    _RAW,
    _plain_float,
    _plain_floats,
    write_config,
)


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
#
# NOTE: the training loops NO LONGER use region_loss. ET supervision is now a per-pixel
# weight inside the ordinary loss (`et_weight` -> training.losses.weighted_loss), because
# the 1/|region| factor below makes the per-pixel gradient scale as 1/area -- measured at a
# ~6500x swing across realistic ET areas, which let a handful of pixels set the update
# direction for every parameter. Both functions are kept for the analysis notebooks
# (lam_et_sweep, synthesis_testbench), which use them as METRICS, where per-batch area
# normalization is exactly what you want.
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

    This is the number to watch when tuning et_weight. Global PSNR does respond to ET-only
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


def embed_for_net(net, E, image, pad_hw):
    """`(E @ Truncate, T)` for one batch, with the size contract checked.

    The loader decides H'/W' from its own `pad_multiple`; the NETWORK is what
    actually constrains them. Checking here turns a config that disagrees into
    a named error instead of letting `kspace_pre_process` quietly fall back to
    resampling the mask -- the failure this embedding exists to remove.

    Lives here, and not in `training/recon.py` where it started, because
    `evaluation/tasks.py` has to build the SAME operator: an adapter that
    re-derived the embedded size from `pad_stride` alone would silently ignore
    the loader's `pad_multiple` and score a different operator than training
    validated on. Both callers now share one definition.
    """
    stride = int(getattr(net, "pad_stride", 1) or 1)
    hw = tuple(image.shape[-2:])

    if pad_hw is None:                       # loader predates the 5th return
        return embed_operator(E, hw, stride)

    p = torch.as_tensor(pad_hw).reshape(-1, 2)
    if p.shape[0] > 1 and not bool((p == p[0]).all()):
        raise ValueError(
            f"the batch mixes embedded sizes ({p.tolist()}); a batch must be "
            f"uniform for a single operator to cover it.")
    big = (int(p[0, 0]), int(p[0, 1]))

    if big[0] % stride or big[1] % stride:
        raise ValueError(
            f"data.pad_multiple gives {big[0]}x{big[1]}, which {type(net).__name__} "
            f"cannot use: it needs a multiple of pad_stride={stride}. Set "
            f"data.<split>.pad_multiple to {stride} (or a multiple of it).")
    if big[0] < hw[0] or big[1] < hw[1]:
        raise ValueError(f"embedded size {big} is smaller than the image {hw}.")

    T = Truncate(big, hw)
    return (E if T.is_identity else E @ T), T


def prepare_measurement(
    image, kspace, mask, smaps,
    kspace_type,
    noise_std,
    noise_dist,
    whiten_kspace,
    generator=None,
):
    """Build (y, sigma, extra) for one reconstruction batch.

    `extra["smaps"]` is ALWAYS the set of maps `y` is consistent with, and the
    caller must build its encoding operator from those -- the simulated branch
    RSS-normalizes them (so sigma is the image-domain noise std) and the
    whitening branch replaces them outright.

    `generator` seeds the noise draw; pass one during validation so the same
    realization is seen every epoch.
    """
    extra = {}

    if kspace_type == "simulated":
        # Fully synthetic measurement: the clean coil-combined image pushed
        # through Sense -> Fourier -> mask, with AWGN in the coil-image domain.
        # The measured `kspace` argument is deliberately unused.
        y, sigma_n, smaps_n = mri_awgn(
            image, mask, smaps, noise_std, noise_dist, generator=generator,
        )
        extra["smaps"] = smaps_n

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

    # Always (B, 1, 1, 1): that is the shape a Polynomial threshold broadcasts
    # against and the only one `MGCDLNet(resize_noise=True)` will resize.
    sigma_n = torch.as_tensor(sigma_n, device=image.device, dtype=torch.float32)
    sigma_n = sigma_n.reshape(-1, 1, 1, 1).expand(image.shape[0], 1, 1, 1).contiguous()
    return y, sigma_n, extra
