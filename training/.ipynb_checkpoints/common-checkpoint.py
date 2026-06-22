# training/common.py

import os
import json
import numpy as np
import torch
from physics.nle import whiten
from operators.fourier import ifftc

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

    step = ckpt.get("step", 0)

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

### MRI Recon/ ImMAP2.5 Specific
def apply_loss_mask(image, recon, organ_mask, use_mask):
    if use_mask:
        image = image * organ_mask
        recon = recon * organ_mask
    return image, recon


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
