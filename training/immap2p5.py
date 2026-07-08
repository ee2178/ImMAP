import os
import math
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
from operators import Mask, FFT2D, Sense, Identity, SSDUMask
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
    k=1,
    # ---------------------------------------------------------
    # Loss
    # ---------------------------------------------------------
    loss_type="complex-mse",
    use_organ_mask=False,
    sigma_scaling=False,
    # ---------------------------------------------------------
    # Optimization (epoch-based)
    # ---------------------------------------------------------
    num_epochs=1000,          # total epochs (was: max_steps)
    steps_per_epoch=100,      # an "epoch" == this many gradient steps
    clip_grad=1.0,
    # ---------------------------------------------------------
    # Logging
    # ---------------------------------------------------------
    val_every_epochs=10,      # validate every N epochs (was: val_every steps)
    # ---------------------------------------------------------
    # Checkpointing
    # ---------------------------------------------------------
    start_epoch=0,            # was: start_step
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
    # Backtracking (now on averaged LOSS, not PSNR)
    # ---------------------------------------------------------
    backtrack_thresh=None,
    backtrack_factor=0.8,
    # ---------------------------------------------------------
    # External logging
    # ---------------------------------------------------------
    wandb=None,
):
    """
    ImMAP2.5 training loop (epoch-based).

    Requires a DiffLPDS-style network that takes:
        recon = net(y, E, sigma_n, E_z, x_init, sigma_t)
    """

    net.to(device)
    net.train()

    loss_fn = LOSS_REGISTRY[loss_type]
    ckpt_path = os.path.join(save_dir, "net.ckpt")

    best_loss = float("inf")        # lower is better -> start high
    backtrack_enabled = backtrack_thresh is not None

    # Build the SSDU operator ONCE; it reshuffles its own mask on each forward.
    # (The @-composition below is rebuilt per step so Sense(ones_like) tracks
    #  the current image shape, which varies when crop_size is null.)
    if ssdu_masking:
        ssdu_op = SSDUMask(
            base_accel=ssdu_base_accel,
            base_acs=ssdu_acs,
            rho=ssdu_rho,
            acs_lines=acs_lines,
            device=device,
        )
    else:
        ssdu_op = None

    train_iter = iter(train_loader)
    total_steps = num_epochs * steps_per_epoch
    pbar = tqdm(
        total=total_steps,
        initial=start_epoch * steps_per_epoch,
        desc="TRAIN",
        dynamic_ncols=True,
    )

    for epoch in range(start_epoch, num_epochs):
        net.train()
        running_loss = 0.0
        n_batches = 0

        # ==================================================================
        # ONE EPOCH == steps_per_epoch GRADIENT STEPS
        # ==================================================================
        for _ in range(steps_per_epoch):

            try:
                kspace, smaps, image, organ_mask = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                kspace, smaps, image, organ_mask = next(train_iter)

            kspace = kspace.to(device, non_blocking=True)
            smaps = smaps.to(device, non_blocking=True)
            image = image.to(device, non_blocking=True)
            organ_mask = organ_mask.to(device, non_blocking=True)

            # ---- mask + forward operator ----
            mask = get_mask(image, R=R, acs_lines=acs_lines, mode=mask_dist)
            E = Mask(mask) @ FFT2D() @ Sense(smaps)

            # ---- (optional) SSDU operator (one fresh split per step) ----
            if ssdu_masking:
                ssdu_op.shuffle_mask(image)   # reshuffle once, before the net runs
                D = ssdu_op @ FFT2D() @ Sense(torch.ones_like(image))
            else:
                D = Identity()

            # ---- k-space measurement ----
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

            # ---- image-domain noise ----
            if whiten_kspace and "image_w" in extra:
                x_clean = extra["image_w"]
            else:
                x_clean = image

            x_init, sigma_t = awgn(
                x_clean,
                image_noise_std,
                dist=image_noise_dist,
                k=k,
            )

            # ---- forward ----
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

            # ---- loss ----
            image_l, recon_l = apply_loss_mask(
                image, recon, organ_mask, use_organ_mask,
            )

            loss = loss_fn(image_l, recon_l, sigma_t)

            if sigma_scaling:
                # Only works because we force batch size 1
                loss = loss * sigma_t.squeeze().pow(-2)

            loss.backward()

            if clip_grad is not None:
                nn.utils.clip_grad_norm_(net.parameters(), clip_grad)

            opt.step()

            if hasattr(net, "project"):
                net.project()

            if sched is not None:
                sched.step()

            running_loss += float(loss.item())
            n_batches += 1

            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.2e}", epoch=epoch)

        global_step = (epoch + 1) * steps_per_epoch

        # ==================================================================
        # END-OF-EPOCH: averaged-loss backtracking + checkpoint + logging
        # ==================================================================
        avg_loss = running_loss / max(n_batches, 1)
        nonfinite = not math.isfinite(avg_loss)

        # Metrics on the last batch -- for LOGGING ONLY.
        with torch.no_grad():
            train_metrics = compute_metrics(image.abs(), recon.abs())
        train_metrics = {key: float(val.detach()) for key, val in train_metrics.items()}

        # Non-finite loss always attempts a protective restore (never
        # checkpoint NaN over a good model). Threshold check only when enabled.
        should_backtrack = nonfinite or (
            backtrack_enabled and avg_loss > best_loss + backtrack_thresh
        )

        if should_backtrack:
            reason = "non-finite loss" if nonfinite else (
                f"avg loss {avg_loss:.3e} > best {best_loss:.3e} + {backtrack_thresh}"
            )
            print(f"[epoch {epoch}] {reason} — backtracking")

            if os.path.exists(ckpt_path) and math.isfinite(best_loss):
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
                # best_loss unchanged: the on-disk ckpt is still the best model.
            else:
                raise RuntimeError(
                    f"Backtrack requested at epoch {epoch} but no valid checkpoint "
                    f"exists yet (best_loss={best_loss})."
                )

        # Save ONLY on genuine improvement -> on-disk ckpt is always the best.
        elif save_ckpt_fn is not None and avg_loss < best_loss:
            save_ckpt_fn(
                ckpt_path,
                model=net,
                step=global_step,
                optimizer=opt,
                scheduler=sched,
            )
            best_loss = avg_loss

        # ---- train logging (skip wandb if this epoch went non-finite) ----
        if wandb is not None and not nonfinite:
            log_dict = {
                "train/loss": avg_loss,
                "train/lr": get_lr(opt)[0],
                "train/epoch": epoch,
                **{f"train/{k}": v for k, v in train_metrics.items()},
            }
            wandb.log(log_dict, step=global_step)
        elif wandb is None:
            print({"epoch": epoch, "avg_loss": avg_loss, **train_metrics})

        # ==================================================================
        # VALIDATION (log-only, every val_every_epochs epochs)
        # ==================================================================
        if val_every_epochs and (epoch + 1) % val_every_epochs == 0:

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

                    mask_v = get_mask(image_v, R=R, acs_lines=acs_lines, mode=mask_dist)
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
                        k=k,
                    )
                    '''
                    if ssdu_masking:
                        ssdu_op.shuffle_mask(image_v)
                        D_v = ssdu_op @ FFT2D() @ Sense(torch.ones_like(image_v))
                    else:
                        D_v = Identity()
                    '''

                    # We want D_v to be Identity() during inference
                    D_v = Identity()

                    recon_v, _ = net(
                        y_v,
                        E=E_v,
                        sigma=sigma_v,
                        E_z=D_v,
                        x_init=x_init_v,
                        sigma_t=sigma_t_v,
                    )

                    if whiten_kspace and "Zinv" in extra_v:
                        recon_v = extra_v["Zinv"] * recon_v

                    metrics_v = compute_metrics(image_v.abs(), recon_v.abs())

                    image_l_v, recon_l_v = apply_loss_mask(
                        image_v, recon_v, organ_mask_v, use_organ_mask,
                    )
                    metrics_v["loss"] = loss_fn(image_l_v, recon_l_v, sigma_t_v)

                    for key, val in metrics_v.items():
                        val_metrics[key] += val * batch_size

                    n_samples += batch_size

                mean_metrics = {
                    k: v / n_samples
                    for k, v in val_metrics.items()
                }
 
                if wandb:
                    gt_img = image_v[:1].abs()
                    recon_img = recon_v[:1].abs()
                    noisy_img = x_init_v[:1].abs()

                    # GT | Noisy | Recon, shared scale from GT+Noisy (unchanged)
                    grid = torch.cat([gt_img, noisy_img, recon_img], dim=0)
                    grid = grid - grid[0:2].min()
                    grid = grid / grid[0:2].max().clamp(min=1e-8)

                    # Residual on its own symmetric scale: 0.5 = zero error, 0/1 = -/+ max|error|
                    res = (gt_img - recon_img).abs()
                    res = res / res.max().clamp(min=1e-8)
                    
                    wandb.log({
                        "val/jdr_example": wandb.Image(
                            vutils.make_grid(grid, nrow=3),
                            caption=f"sigma={sigma_t_v.flatten()[0].item():.2f}"
                        ),
                        "val/jdr_residual": wandb.Image(
                            vutils.make_grid(res, nrow=1),
                            caption="GT - Recon"
                        ),
                        **{f"val/{k}": v.item() for k, v in mean_metrics.items()},
                    }, step=global_step)
                    wandb.log(get_filter_grids(net), step=global_step)


                else:
                    print(
                        f"[VAL] epoch={epoch} "
                        + " ".join(
                            [f"{k}={v.item():.4f}" for k, v in mean_metrics.items()]
                        )
                    )

            net.train()

    pbar.close()
