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
from physics.mask import gen_ssdu_mask
from operators import Mask, FFT2D, Sense, Identity
from operators.noise import awgn

def train_joint_denoising_recon(
    net,
    opt,
    sched,
    train_loader,
    val_loader,
    device,
    # ---------------------------------------------------------
    # MRI physics
    # ---------------------------------------------------------
    R=8,
    acs_lines=24,
    kspace_type="measurement",
    mask_dist="uniform",
    whiten_kspace=False,
    noise_std=(0.0, 0.01),
    noise_dist="uniform",
    # ---------------------------------------------------------
    # Image-domain corruption
    # ---------------------------------------------------------
    image_noise_std=(0.0, 0.1),
    image_noise_dist="uniform",
    image_noise_log_base=10,
    # ---------------------------------------------------------
    # Loss
    # ---------------------------------------------------------
    loss_type="complex-mse",
    use_organ_mask=False,
    sigma_scaling=False,
    # ---------------------------------------------------------
    # Optimization
    # ---------------------------------------------------------
    max_steps=100000,
    clip_grad=1.0,
    # ---------------------------------------------------------
    # Logging
    # ---------------------------------------------------------
    log_every=50,
    val_every=1000,
    # ---------------------------------------------------------
    # Checkpointing
    # ---------------------------------------------------------
    start_step=0,
    save_every=5000,
    save_dir=None,
    save_ckpt_fn=save_ckpt,
    ckpt=None,
    # ---------------------------------------------------------
    # Initialization from Network Outputs
    # ---------------------------------------------------------
    init_type=None,
    denoiser_path=None,
    recon_path=None,
    # ---------------------------------------------------------
    # SSDU Masking Params
    # ---------------------------------------------------------
    ssdu_masking=False,
    ssdu_base_accel=2,
    ssdu_acs=10,
    ssdu_rho=(0.2, 0.2),
    # ---------------------------------------------------------
    # Backtracking
    # ---------------------------------------------------------
    backtrack_thresh=None,
    backtrack_factor=0.8,
    # ---------------------------------------------------------
    # External logging
    # ---------------------------------------------------------
    wandb=None,
):
    """
    ImMAP2.5 training loop.

    Requires a DiffLPDS-style network that takes:
        recon = net(y, E, sigma_n, x_init, sigma_t)
    """

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

        # =====================================================
        # TRAIN BATCH
        # =====================================================

        try:
            kspace, smaps, image, organ_mask = next(train_iter)

        except StopIteration:
            train_iter = iter(train_loader)
            kspace, smaps, image, organ_mask = next(train_iter)

        kspace = kspace.to(device, non_blocking=True)
        smaps = smaps.to(device, non_blocking=True)
        image = image.to(device, non_blocking=True)
        organ_mask = organ_mask.to(device, non_blocking=True)

        # =====================================================
        # MASK + OPERATOR
        # =====================================================

        mask = get_mask(
            image,
            R=R,
            acs_lines=acs_lines,
            mode=mask_dist
        )

        E = Mask(mask) @ FFT2D() @ Sense(smaps)

        # =====================================================
        # (OPTIONAL) SSDU MASK + OPERATOR
        # =====================================================
        
        if ssdu_masking is True:
            D = Mask(gen_ssdu_mask(image[0,0].shape, acs_lines, ssdu_base_accel, ssdu_acs, ssdu_rho, device = device)) @ FFT2D() @ Sense(torch.ones_like(image))
        else:
            D = Identity()

        # =====================================================
        # KSPACE MEASUREMENT
        # =====================================================

        y, sigma_n, extra = prepare_measurement(
            image=image,
            kspace=kspace,
            mask=mask,
            smaps=smaps,
            kspace_type=kspace_type,
            noise_std=noise_std,
            noise_dist=noise_dist,
            whiten_kspace=whiten_kspace,
        )

        if whiten_kspace:
            smaps = extra["smaps"]

        # =====================================================
        # IMAGE-DOMAIN NOISE
        # =====================================================

        if whiten_kspace and "image_w" in extra:
            x_clean = extra["image_w"]
        else:
            x_clean = image

        x_init, sigma_t = awgn(
            x_clean,
            image_noise_std,
            dist=image_noise_dist,
            log_base=image_noise_log_base
        )

        # =====================================================
        # FORWARD
        # =====================================================

        opt.zero_grad(set_to_none=True)

        recon, _ = net(
            y,
            E=E,
            sigma=sigma_n,
            E_z=D,
            x_init=x_init,
            sigma_t=sigma_t,
        )

        if whiten_kspace and "Zinv" in extra:
            recon = extra["Zinv"] * recon

        # =====================================================
        # LOSS
        # =====================================================

        image_l, recon_l = apply_loss_mask(
            image,
            recon,
            organ_mask,
            use_organ_mask,
        )

        loss = loss_fn(
            image_l,
            recon_l,
            sigma_t,
        )

        if sigma_scaling:
            # Only works cuz we force batch size 1
            loss = loss * sigma_t.squeeze().pow(-2)

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
        # =====================================================
        # LOGGING
        # =====================================================

        if step % log_every == 0:
            train_metrics = compute_metrics(
                image.abs(),
                recon.abs(),
            )
            log_dict = {
                "train/loss": loss.item(),
                "train/lr": get_lr(opt)[0],
                **{
                    f"train/{k}": v
                    for k, v in train_metrics.items()
                },
            }

            if wandb is not None:
                wandb.log(log_dict, step=step)
            else:
                print(log_dict)
            
        # =====================================================
        # VALIDATION
        # =====================================================

        if step % val_every == 0 and step > 0:

            net.eval()

            # Running sums ON GPU
            val_metrics = {
                "psnr": torch.zeros(1, device=device),
                "ssim": torch.zeros(1, device=device),
                "nrmse": torch.zeros(1, device=device),
                "loss": torch.zeros(1, device=device),
            }
            n_samples = 0

            with torch.no_grad():

                for kspace_v, smaps_v, image_v, organ_mask_v in val_loader:

                    kspace_v = kspace_v.to(device, non_blocking=True)
                    smaps_v = smaps_v.to(device, non_blocking=True)
                    image_v = image_v.to(device, non_blocking=True)
                    organ_mask_v = organ_mask_v.to(device, non_blocking=True)

                    batch_size = image_v.shape[0]

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

                    if whiten_kspace and "image_w" in extra_v:
                        x_clean_v = extra_v["image_w"]
                    else:
                        x_clean_v = image_v

                    x_init_v, sigma_t_v = awgn(
                        x_clean_v,
                        image_noise_std,
                        dist=image_noise_dist,
                        log_base=image_noise_log_base
                    )

                    if ssdu_masking is True:
                        D_v = (
                            Mask(
                                gen_ssdu_mask(
                                    image_v[0, 0].shape,
                                    acs_lines,
                                    ssdu_base_accel,
                                    ssdu_acs,
                                    ssdu_rho,
                                    device=device
                                )
                            )
                            @ FFT2D()
                            @ Sense(torch.ones_like(image_v))
                        )
                    else:
                        D_v = Identity()

                    recon_v, _ = net(
                        y_v,
                        E = E_v,
                        sigma=sigma_v,
                        E_z=D_v,
                        x_init=x_init_v,
                        sigma_t=sigma_t_v,
                    )

                    if whiten_kspace and "Zinv" in extra_v:
                        recon_v = extra_v["Zinv"] * recon_v

                    # Compute metrics (dict of GPU tensors)
                    metrics_v = compute_metrics(
                        image_v.abs(),
                        recon_v.abs(),
                    )

                    image_l_v, recon_l_v = apply_loss_mask(
                        image_v,
                        recon_v,
                        organ_mask_v,
                        use_organ_mask,
                    )

                    metrics_v["loss"] = loss_fn(
                        image_l_v,
                        recon_l_v,
                        sigma_t_v,
                    )

                    # Running weighted sums
                    for k, v in metrics_v.items():
                        val_metrics[k] += v * batch_size

                    n_samples += batch_size

                # Compute averages (still tensors)
                mean_metrics = {
                    k: v / n_samples
                    for k, v in val_metrics.items()
                }

                if wandb:
                    # Log final validation image
                    gt_img = image_v[:1].abs()
                    recon_img = recon_v[:1].abs()
                    noisy_img = x_init_v[:1].abs()

                    grid = torch.cat([gt_img, noisy_img, recon_img], dim=0)

                    # Normalize
                    grid = grid - grid.min()
                    grid = grid / grid.max()

                    wandb.log({
                        "val/jdr_example": wandb.Image(
                            vutils.make_grid(grid, nrow=3),
                            caption=f"sigma={sigma_t_v.item():.2f}"
                        ),
                        **{f"val/{k}": v.item() for k, v in mean_metrics.items()},
                    }, step=step)

                    wandb.log(
                        get_filter_grids(net),
                        step=step,
                    )

                else:
                    print(
                        f"[VAL] step={step} "
                        + " ".join(
                            [f"{k}={v.item():.4f}" for k, v in mean_metrics.items()]
                        )
                    )

            net.train()
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
                    f"net.ckpt",
                )

                save_ckpt_fn(
                    ckpt_path,
                    model=net,
                    step=step,
                    optimizer=opt,
                    scheduler=sched,
                )

        # =====================================================
        # PROGRESS BAR
        # =====================================================
        # Don't update progress bar if using wandb, no longer necessary. 
        '''
        pbar.update(1)

        pbar.set_postfix(
            loss=f"{loss.item():.2e}",
            psnr=f"{train_metrics['psnr']:.2f}",
        )
        '''

    pbar.close()
