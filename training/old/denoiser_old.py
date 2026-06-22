import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils
import math

from tqdm import tqdm
from datasets.fastmri.common import load_fastmri_data
from operators import Identity
from operators.noise import awgn
from training.common import save_ckpt, load_ckpt, get_lr, set_lr
from training.losses import LOSS_REGISTRY
from training.metrics import compute_metrics, psnr
from visualization.filters import get_filter_grids


def train_denoiser(
    net, opt, sched, device,
    train_loader,
    val_loader,
    wandb=None,
    start_step=0,
    max_steps=100000,
    noise_std=(0.0, 0.1),
    noise_dist="uniform",
    noise_log_base=100,
    loss_type="complex-mse",
    clip_grad=1.0,
    log_every=50,
    val_every=1000,
    backtrack_thresh=5,
    backtrack_factor=0.8,
    save_ckpt_fn=save_ckpt,
    save_dir=None,
    ckpt=None,
    psnr_only=True,
):
    net.to(device)
    net.train()

    loss_fn = LOSS_REGISTRY[loss_type]
    E = Identity()

    best_psnr = 1e-9

    train_iter = iter(train_loader)
    pbar = tqdm(total=max_steps, initial=start_step, desc="TRAIN", dynamic_ncols=True)

    for step in range(start_step, max_steps):

        try:
            # Grab next batch, only returns batched image
            gt = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            gt = next(train_iter)

        gt = gt.to(device).unsqueeze(1)

        opt.zero_grad()
        noisy, sigma = awgn(gt, noise_std, dist=noise_dist, log_base=noise_log_base)
        # we need to pass in different arguments if network is a UNet unfortunately. Short term cheat:
        if net.__class__.__name__ == "Unet" or net.__class__.__name__ == "NormUnet":
            # Norm Unet requires some rather nasty image formatting
            recon = net(torch.view_as_real(noisy))
            # Convert to complex valued formatting
            recon = torch.view_as_complex(recon.contiguous())
        else:
            recon, _ = net(noisy, E=E, sigma=sigma)
        loss = loss_fn(gt, recon, sigma)   # FIXED (sigma included everywhere)
        loss.backward()

        if clip_grad is not None:
            nn.utils.clip_grad_norm_(net.parameters(), clip_grad)

        opt.step()
        if hasattr(net, "project"): net.project()
        if sched is not None: sched.step()

        # Computing metrics every literation is killing GPU utilization
        # train_metrics = compute_metrics(gt.abs(), recon.abs()) # COMPUTE METRICS ON MAGNITUDE IMAGES

        if step % log_every == 0:
            # Compute metrics every log_every iterations instead
            train_metrics = compute_metrics(gt.abs(), recon.abs(), psnr_only=psnr_only)
            train_metrics = {k: float(v.detach()) for k, v in train_metrics.items()}
            psnr = train_metrics["psnr"]
            
            nonfinite = not math.isfinite(psnr)
            # Perform a backtracking check every log_every iterations
            if nonfinite or (train_metrics["psnr"] + backtrack_thresh < best_psnr):
                print("PSNR dropped — backtracking")

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
            # Every log_every iterations, save our checkpoint as well. Shouldn't cost too much.
            elif save_ckpt_fn:
                save_ckpt_fn(
                    os.path.join(save_dir, "net.ckpt"),
                    model=net,
                    optimizer=opt,
                    scheduler=sched,
                    step=step,
                )
                best_psnr = train_metrics["psnr"]
            
            # Update progress every log_every steps (saves on cpu sync)
            pbar.update(log_every)
            pbar.set_postfix(
                loss=f"{loss.item():.2e}",
                psnr=f"{train_metrics['psnr']:.2f}"
            ) 
            # Log in wandb if not NaN:
            if nonfinite is False:
                log_dict = {
                    "train/loss": float(loss.item()),
                    "train/lr": opt.param_groups[0]["lr"],
                    **{f"train/{k}": v for k, v in train_metrics.items()},
                }
                wandb.log(log_dict, step=step) if wandb else print(log_dict)

        # ======================================================
        # VALIDATION
        # ======================================================
        if step % val_every == 0 and step > 0:
            net.eval()
            # Running sums ON GPU
            val_metrics_agg = {
                "psnr": torch.zeros(1, device=device),
                "ssim": torch.zeros(1, device=device),
                "nrmse": torch.zeros(1, device=device),
                "loss": torch.zeros(1, device=device),
            }
            n_samples = 0

            with torch.no_grad():
                for gt_v in val_loader:
                    gt_v = gt_v.to(device, non_blocking=True).unsqueeze(1)
                    batch_size = gt_v.shape[0]

                    noisy_v, sigma_v = awgn(
                        gt_v,
                        noise_std,
                        dist=noise_dist,
                        log_base=noise_log_base
                    )

                    # Forward
                    if net.__class__.__name__ in ["Unet", "NormUnet"]:
                        recon_v = net(torch.view_as_real(noisy_v))
                        recon_v = torch.view_as_complex(recon_v.contiguous())
                    else:
                        recon_v, _ = net(noisy_v, E=E, sigma=sigma_v)

                    # Compute metrics (dict of GPU tensors)
                    metrics_v = compute_metrics(gt_v.abs(), recon_v.abs(), psnr_only=psnr_only)
                    metrics_v["loss"] = loss_fn(gt_v, recon_v, sigma_v)

                    # Running weighted sums
                    for k, v in metrics_v.items():
                        val_metrics_agg[k] += v.detach() * batch_size

                    n_samples += batch_size

                # Compute means (still tensors)
                mean_metrics = {
                    k: v / n_samples
                    for k, v in val_metrics_agg.items()
                }

                if wandb:
                    # Log final validation image
                    gt_img = gt_v[:1].abs()
                    recon_img = recon_v[:1].abs()
                    noisy_img = noisy_v[:1].abs()

                    grid = torch.cat([gt_img, noisy_img, recon_img], dim=0)

                    # Normalize
                    grid = grid - grid.min()
                    grid = grid / grid.max()

                    wandb.log({
                        "val/denoising_example": wandb.Image(
                            vutils.make_grid(grid, nrow=3),
                            caption=f"Gnd Truth | Noisy sigma={sigma_v.item():.2f} | Output"
                        ),
                        **{f"val/{k}": v.item() for k, v in mean_metrics.items()},
                    }, step=step)

                else:
                    print(
                        f"[VAL] step={step} "
                        + " ".join(
                            [f"{k}={v.item():.4f}" for k, v in mean_metrics.items()]
                        )
                    )

            net.train()
                    
            # ======================================================
            # FILTER LOGGING (NOW ALWAYS DURING VALIDATION)
            # ======================================================
            if wandb:
                try:
                    # Will only work on Unrolled Nets
                    wandb.log(get_filter_grids(net), step=step)
                except (AttributeError, NotImplementedError, AssertionError):
                    pass
            net.train()

    pbar.close()
