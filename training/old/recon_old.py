import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils

from tqdm import tqdm
from training.losses import LOSS_REGISTRY
from training.metrics import compute_metrics
from training.common import (
    save_ckpt,
    load_ckpt,
    get_lr,
    set_lr,
    apply_loss_mask,
    prepare_measurement,
)
from visualization.filters import get_filter_grids
from physics.mask import get_mask_cached as get_mask
from operators import Mask, FFT2D, Sense


def train_recon(
    net,
    opt,
    sched,
    train_loader,
    val_loader,
    device,
    ### MRI Arguments
    R=8,
    acs_lines=24,
    kspace_type="measurement",
    whiten_kspace=False,
    mask_dist = "uniform",
    noise_std=(0.0, 0.01),
    noise_dist="uniform",
    ### Loss 
    loss_type="complex-mse",
    use_organ_mask=False,
    ### Fit
    max_steps=100000,
    clip_grad=1.0,
    ### Logging
    log_every=50,
    val_every=1000,
    save_every=5000,
    start_step=0,
    save_dir=None,
    ckpt=None,
    save_ckpt_fn=save_ckpt,
    backtrack_thresh=None,
    backtrack_factor=0.8,
    wandb=None,
):
    net.to(device)
    net.train()

    loss_fn = LOSS_REGISTRY[loss_type]

    train_iter = iter(train_loader)

    pbar = tqdm(
        total=max_steps,
        initial=start_step,
        desc="TRAIN",
        dynamic_ncols=True,
    )

    best_psnr = -1e9

    for step in range(start_step, max_steps):

        try:
            kspace, smaps, image, organ_mask = next(train_iter)

        except StopIteration:
            train_iter = iter(train_loader)
            kspace, smaps, image, organ_mask = next(train_iter)

        kspace = kspace.to(device, non_blocking=True)
        smaps = smaps.to(device, non_blocking=True)
        image = image.to(device, non_blocking=True)
        organ_mask = organ_mask.to(device, non_blocking=True)
        mask = get_mask(
            image,
            R=R,
            acs_lines=acs_lines,
            mode=mask_dist
        )

        E = Mask(mask) @ FFT2D() @ Sense(smaps)

        y, sigma_n, extra = prepare_measurement(
            image=image,
            kspace=kspace,
            mask=mask,
            smaps=smaps,
            kspace_type=kspace_type,
            # Synthetic noise parameters only for sim kspace
            noise_std=noise_std,
            noise_dist=noise_dist,
            whiten_kspace=whiten_kspace,
        )

        if whiten_kspace:
            smaps = extra["smaps"]

        opt.zero_grad(set_to_none=True)

        recon, _ = net(
            y,
            E=E,
            sigma=sigma_n,
        )

        if whiten_kspace and "Zinv" in extra:
            recon = extra["Zinv"] * recon

        image_l, recon_l = apply_loss_mask(
            image,
            recon,
            organ_mask,
            use_organ_mask,
        )

        loss = loss_fn(
            image_l,
            recon_l,
            sigma_n,
        )

        loss.backward()

        if clip_grad is not None:
            nn.utils.clip_grad_norm_(
                net.parameters(),
                clip_grad,
            )

        opt.step()

        if hasattr(net, "project"):
            net.project()

        if sched is not None:
            sched.step()

        if step % log_every == 0:
            with torch.no_grad():
                train_metrics = compute_metrics(
                    image.abs(),
                    recon.abs(),
                )
            log_dict = {
                "train/loss": loss.item(),
                "train/lr": get_lr(opt)[0],
                **{
                    f"train/{k}": v.detach().item()
                    for k, v in train_metrics.items()
                },
            }

            if wandb is not None:
                wandb.log(log_dict, step=step)
            else:
                print(log_dict)
            # Update progress bar
            pbar.update(log_every)
            pbar.set_postfix(
                loss=f"{loss.item():.2e}",
                psnr=f"{train_metrics['psnr'].detach():.2f}",
            )


        # Validation Loop
        if step % val_every == 0 and step > 0:

            net.eval()

            val_metrics = {
                "psnr": torch.tensor(0.0, device=device),
                "ssim": torch.tensor(0.0, device=device),
                "nrmse": torch.tensor(0.0, device=device),
                "loss": torch.tensor(0.0, device=device),
            }

            num_val_batches = 0

            with torch.no_grad():

                for kspace_v, smaps_v, image_v, organ_mask_v in val_loader:

                    kspace_v = kspace_v.to(device, non_blocking=True)
                    smaps_v = smaps_v.to(device, non_blocking=True)
                    image_v = image_v.to(device, non_blocking=True)
                    organ_mask_v = organ_mask_v.to(device, non_blocking=True)

                    mask_v = get_mask(
                        image_v,
                        R=R,
                        acs_lines=acs_lines,
                        mode=mask_dist
                    )

                    E_v = Mask(mask_v) @ FFT2D() @ Sense(smaps_v)

                    y_v, sigma_v, extra_v = prepare_measurement(
                        image=image_v,
                        kspace=kspace_v,
                        mask=mask_v,
                        smaps=smaps_v,
                        kspace_type=kspace_type,
                        noise_std=noise_std,
                        noise_dist=noise_dist,
                        whiten_kspace=whiten_kspace,
                    )

                    if whiten_kspace:
                        smaps_v = extra_v["smaps"]

                    recon_v, _ = net(
                        y_v,
                        E = E_v,
                        sigma = sigma_v,
                    )

                    if whiten_kspace and "Zinv" in extra_v:
                        recon_v = extra_v["Zinv"] * recon_v

                    m = compute_metrics(
                        image_v.abs(),
                        recon_v.abs(),
                    )

                    val_metrics["psnr"] += m["psnr"].detach()
                    val_metrics["ssim"] += m["ssim"].detach()
                    val_metrics["nrmse"] += m["nrmse"].detach()

                    image_l_v, recon_l_v = apply_loss_mask(
                        image_v,
                        recon_v,
                        organ_mask_v,
                        use_organ_mask,
                    )

                    val_loss = loss_fn(
                        image_l_v,
                        recon_l_v,
                        sigma_v,
                    )

                    val_metrics["loss"] += val_loss

                    num_val_batches += 1

                ### Sample Image Logging for Wandb
                if wandb:
                    # Log final validation image to wandb
                    gt_img = image_v[:1].abs()
                    recon_img = recon_v[:1].abs()

                    grid = torch.cat([gt_img, recon_img], dim=0)

                    # Normalize
                    grid = grid - grid.min()
                    grid = grid / grid.max()
                    
                    wandb.log(
                    {
                        "val/recon_example": wandb.Image(
                            vutils.make_grid(grid, nrow=2)
                        )
                    },
                    step=step,
                    )

            mean_metrics = {
                k: (v.detach() / num_val_batches).item()
                for k, v in val_metrics.items()
            }

            if wandb is not None:
                wandb.log(
                    {
                        f"val/{k}": v
                        for k, v in mean_metrics.items()
                    },
                    step=step,
                )

                wandb.log(
                    get_filter_grids(net),
                    step=step,
                )

            else:
                print(f"[VAL] step={step} {mean_metrics}")

            # =================================================
            # BACKTRACKING
            # =================================================

            if (
                backtrack_thresh is not None
                and mean_metrics["psnr"] + backtrack_thresh < best_psnr
            ):

                print("Validation dropped — backtracking")

                ckpt_path = os.path.join(
                    save_dir,
                    "net.ckpt",
                )

                net, opt, sched, _ = load_ckpt(
                    ckpt_path,
                    model=net,
                    optimizer=opt,
                    scheduler=sched,
                    device=device,
                )

                old_lr = np.array(get_lr(opt))
                new_lr = old_lr * backtrack_factor

                set_lr(opt, new_lr)

                print("Updated LR:", new_lr)

            net.train()

        # =====================================================
        # PERIODIC CHECKPOINT
        # =====================================================

        if step % save_every == 0 and step > 0:

            if save_ckpt_fn is not None:

                ckpt_path = os.path.join(
                    save_dir,
                    "net.ckpt",
                )

                save_ckpt_fn(
                    ckpt_path,
                    model=net,
                    step=step,
                    optimizer=opt,
                    scheduler=sched,
                )
        

    pbar.close()
