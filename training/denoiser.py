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
    start_epoch=0,
    num_epochs=50,            # total epochs to train (was: max_steps)
    steps_per_epoch=100,      # an "epoch" == this many gradient steps
    val_every_epochs=10,      # run validation every N epochs (was: val_every steps)
    noise_std=(0.0, 0.1),
    noise_dist="uniform",
    loss_type="complex-mse",
    clip_grad=1.0,
    # NOTE: backtrack_thresh is now in *loss units*, not dB. Lower loss is better,
    # so we backtrack when avg_loss rises above best_loss by more than this margin.
    # The old default of 5 was a PSNR margin and is meaningless here — tune to your
    # loss scale. (For a scale-free version, see the relative-threshold note below.)
    backtrack_thresh=1.0,
    backtrack_factor=0.9,
    save_ckpt_fn=save_ckpt,
    save_dir=None,
    ckpt=None,
    psnr_only=True,
):
    net.to(device)
    net.train()

    loss_fn = LOSS_REGISTRY[loss_type]
    E = Identity()

    best_loss = float("inf")        # lower is better -> start high
    ckpt_path = os.path.join(save_dir, "net.ckpt")

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
                # Grab next batch, only returns batched image
                gt = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                gt = next(train_iter)
            # Send to GPU
            gt = gt.to(device, non_blocking = True)

            opt.zero_grad()
            noisy, sigma = awgn(gt, noise_std, dist=noise_dist)

            # UNet variants need a different I/O format. Short term cheat:
            if net.__class__.__name__ in ("Unet", "NormUnet"):
                recon = net(torch.view_as_real(noisy))
                recon = torch.view_as_complex(recon.contiguous())
            else:
                recon, _ = net(noisy, E=E, sigma=sigma)

            loss = loss_fn(gt, recon, sigma)
            loss.backward()

            if clip_grad is not None:
                nn.utils.clip_grad_norm_(net.parameters(), clip_grad)

            opt.step()
            if hasattr(net, "project"): net.project()
            if sched is not None: sched.step()

            running_loss += float(loss.item())
            n_batches += 1

            # Smooth per-step progress; heavy logging happens at epoch end.
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.2e}", epoch=epoch)

        # Steps completed so far -> use as the wandb step axis.
        global_step = (epoch + 1) * steps_per_epoch

        # ==================================================================
        # END-OF-EPOCH: averaged-loss backtracking + checkpoint + logging
        # ==================================================================
        avg_loss = running_loss / max(n_batches, 1)
        nonfinite = not math.isfinite(avg_loss)

        # Metrics on the last batch — for LOGGING ONLY. PSNR no longer drives
        # backtracking; the averaged loss does.
        train_metrics = compute_metrics(gt.abs(), recon.abs(), psnr_only=psnr_only)
        train_metrics = {k: float(v.detach()) for k, v in train_metrics.items()}

        # ---- backtracking decision (on averaged loss) ----
        if nonfinite or (avg_loss > best_loss + backtrack_thresh):
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
                # best_loss is left unchanged: the on-disk ckpt is still the best model.
            else:
                # Nothing good to restore yet (e.g. blew up before the first save).
                raise RuntimeError(
                    f"Backtrack requested at epoch {epoch} but no valid checkpoint "
                    f"exists yet (best_loss={best_loss})."
                )

        # ---- otherwise save ONLY on genuine improvement ----
        elif save_ckpt_fn and avg_loss < best_loss:
            save_ckpt_fn(
                ckpt_path,
                model=net,
                optimizer=opt,
                scheduler=sched,
                step=global_step,
            )
            best_loss = avg_loss

        # ---- logging (skip wandb if this epoch went non-finite) ----
        if wandb and not nonfinite and (avg_loss < best_loss + backtrack_thresh):
            log_dict = {
                "train/loss": avg_loss,
                "train/lr": opt.param_groups[0]["lr"],
                "train/epoch": epoch,
                **{f"train/{k}": v for k, v in train_metrics.items()},
            }
            wandb.log(log_dict, step=global_step)
        elif not wandb:
            print({"epoch": epoch, "avg_loss": avg_loss, **train_metrics})

        # ==================================================================
        # VALIDATION (every val_every_epochs epochs)
        # ==================================================================
        if val_every_epochs and (epoch + 1) % val_every_epochs == 0:
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
                    gt_v = gt_v.to(device, non_blocking=True)
                    batch_size = gt_v.shape[0]

                    noisy_v, sigma_v = awgn(
                        gt_v,
                        noise_std,
                        dist=noise_dist,
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
                            caption=f"Gnd Truth | Noisy sigma={sigma_v.flatten()[0].item():.2f} | Output"
                        ),
                        **{f"val/{k}": v.item() for k, v in mean_metrics.items()},
                    }, step=global_step)

                else:
                    print(
                        f"[VAL] epoch={epoch} "
                        + " ".join(
                            [f"{k}={v.item():.4f}" for k, v in mean_metrics.items()]
                        )
                    )

            net.train()

            # ==============================================================
            # FILTER LOGGING (ALWAYS DURING VALIDATION)
            # ==============================================================
            if wandb:
                try:
                    # Will only work on Unrolled Nets
                    wandb.log(get_filter_grids(net), step=global_step)
                except (AttributeError, NotImplementedError, AssertionError):
                    pass
            net.train()

    pbar.close()
