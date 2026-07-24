"""
Domain-transfer synthesis training loop for DT-CDLNet.

A thin variant of train_synthesis (training/synthesis.py). Same paired-data pipeline
(X = source contrast(s), y = target contrast, organ_mask), same backtracking / metrics /
logging / checkpointing. Three differences, all forced by the CDLNet-family model:

  1. Forward interface. DT-CDLNet is `x_hat, y_hat, z = net(X, E=Identity, return_source=True)`
     (an unrolled model, not a plain `pred = net(X)`): x_hat is the z -> x TARGET decode,
     y_hat is the z -> y SOURCE reconstruction from the same shared code z.
  2. Source-consistency term. loss = L(x_hat, y) + lam_src * L(y_hat, X). The second term
     supervises the source dictionary Dy = B[0] and the sparse-coding stage directly; set
     lam_src = 0 to train the target path only. (This is also where the future additive
     `delS` coupling will be exercised.)
  3. Projection. `net.project()` runs after every optimizer step to enforce the CDLNet
     constraints (thresholds >= 0, dictionaries on the unit ball). train_synthesis omits
     this because U-Nets are unconstrained; unrolled models require it.

DT-CDLNet is real-valued here (build with complex=False), matching the z-scored BraTS data.
There is no CCL backbone to warm-start from, so this loop has no `pretrained` hook.
"""

import os
import math

import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils
from tqdm import tqdm

from operators import Identity
from training.common import save_ckpt, load_ckpt, get_lr, set_lr, apply_loss_mask
from training.metrics import compute_metrics
from training.losses import LOSS_REGISTRY

# For reducing LR on plateau
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train_dt_synthesis(
    net, opt, sched, device,
    train_loader,
    val_loader,
    wandb=None,
    start_epoch=0,
    num_epochs=200,
    steps_per_epoch=200,
    val_every_epochs=10,
    clip_grad=1.0,
    backtrack_thresh=0.5,
    backtrack_factor=0.9,
    loss_type="magnitude-l1",
    use_mask=True,
    psnr_only=False,
    lam_src=1.0,                     # weight of the source-reconstruction consistency term
    save_dir=None,
    ckpt=None,                       # signature parity; resume handled in main()
    save_ckpt_fn=save_ckpt,
):
    net.to(device)

    loss_fn = LOSS_REGISTRY[loss_type]
    E = Identity()                   # pure transfer: no measurement operator

    net.train()
    best_loss = float("inf")
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, "net.ckpt")

    train_iter = iter(train_loader)
    total_steps = num_epochs * steps_per_epoch
    pbar = tqdm(total=total_steps, initial=start_epoch * steps_per_epoch,
                desc="DT-SYNTHESIS", dynamic_ncols=True)

    for epoch in range(start_epoch, num_epochs):
        net.train()
        running_loss, running_src, n_batches = 0.0, 0.0, 0

        for _ in range(steps_per_epoch):
            try:
                X, y, organ_mask = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                X, y, organ_mask = next(train_iter)
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            organ_mask = organ_mask.to(device, non_blocking=True)

            opt.zero_grad()
            # x_hat: (B, Cx, H, W) target decode;  y_hat: (B, C, H, W) source reconstruction
            x_hat, y_hat, z = net(X, E=E, return_source=True)

            # organ_mask is (B, 1, H, W) and broadcasts over channels
            y_t,  x_hat_m = apply_loss_mask(y, x_hat, organ_mask, use_mask)
            X_s,  y_hat_m = apply_loss_mask(X, y_hat, organ_mask, use_mask)

            tgt_loss = loss_fn(x_hat_m, y_t, None)
            src_loss = loss_fn(y_hat_m, X_s, None)
            loss = tgt_loss + lam_src * src_loss

            loss.backward()
            if clip_grad is not None:
                nn.utils.clip_grad_norm_(net.parameters(), clip_grad)
            opt.step()
            if hasattr(net, "project"):          # CDLNet-family constraint projection
                net.project()

            if sched is not None and not isinstance(sched, ReduceLROnPlateau):
                sched.step()

            running_loss += float(loss.item())
            running_src += float(src_loss.item())
            n_batches += 1
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.3e}", src=f"{src_loss.item():.3e}", epoch=epoch)

        global_step = (epoch + 1) * steps_per_epoch
        avg_loss = running_loss / max(n_batches, 1)
        avg_src = running_src / max(n_batches, 1)
        nonfinite = not math.isfinite(avg_loss)

        train_metrics = compute_metrics(y_t, x_hat_m, psnr_only=psnr_only)
        train_metrics = {k: float(v.detach()) for k, v in train_metrics.items()}

        # ---- averaged-loss backtracking (on the TOTAL training loss) ----
        if nonfinite or (avg_loss > best_loss + backtrack_thresh):
            reason = "non-finite loss" if nonfinite else (
                f"avg loss {avg_loss:.3e} > best {best_loss:.3e} + {backtrack_thresh}")
            print(f"[epoch {epoch}] {reason} — backtracking")
            if os.path.exists(ckpt_path) and math.isfinite(best_loss):
                net, opt, sched, _ = load_ckpt(ckpt_path, model=net, optimizer=opt,
                                               scheduler=sched, device=device)
                new_lr = np.array(get_lr(opt)) * backtrack_factor
                set_lr(opt, new_lr)
                print("Updated LR:", new_lr)
            else:
                raise RuntimeError(f"Backtrack at epoch {epoch} but no valid checkpoint "
                                   f"(best_loss={best_loss}).")
        elif save_ckpt_fn and avg_loss < best_loss:
            save_ckpt_fn(ckpt_path, model=net, optimizer=opt, scheduler=sched, step=global_step)
            best_loss = avg_loss

        # ---- logging ----
        if wandb and not nonfinite:
            wandb.log({"train/loss": avg_loss, "train/src_loss": avg_src,
                       "train/lr": opt.param_groups[0]["lr"], "train/epoch": epoch,
                       **{f"train/{k}": v for k, v in train_metrics.items()}}, step=global_step)
        elif not wandb:
            print({"epoch": epoch, "avg_loss": avg_loss, "avg_src": avg_src, **train_metrics})

        # ---- validation ----
        if val_loader is not None and val_every_epochs and (epoch + 1) % val_every_epochs == 0:
            net.eval()
            agg = {"psnr": 0.0, "ssim": 0.0, "nrmse": 0.0, "loss": 0.0, "src_loss": 0.0}
            n_samples = 0
            with torch.no_grad():
                for Xv, yv, organ_maskv in val_loader:
                    Xv = Xv.to(device, non_blocking=True)
                    yv = yv.to(device, non_blocking=True)
                    organ_maskv = organ_maskv.to(device, non_blocking=True)
                    x_hat_v, y_hat_v, zv = net(Xv, E=E, return_source=True)
                    bs = Xv.shape[0]

                    yv_t, x_hat_vm = apply_loss_mask(yv, x_hat_v, organ_maskv, use_mask)
                    Xv_s, y_hat_vm = apply_loss_mask(Xv, y_hat_v, organ_maskv, use_mask)

                    # DO NOT CALL .ABS(), WE HAVE NEGATIVE NUMBERS (z-scored data)
                    mets = compute_metrics(yv_t, x_hat_vm, psnr_only=psnr_only)
                    mets = {k: float(v.detach()) for k, v in mets.items()}
                    src_l = float(loss_fn(y_hat_vm, Xv_s, None).item())
                    mets["src_loss"] = src_l
                    mets["loss"] = float(loss_fn(x_hat_vm, yv_t, None).item()) + lam_src * src_l
                    for k in agg:
                        if k in mets:
                            agg[k] += mets[k] * bs
                    n_samples += bs
            mean_metrics = {k: v / max(n_samples, 1) for k, v in agg.items()}
            val_loss = mean_metrics["loss"]      # <-- capture for the scheduler

            if wandb:
                gt_img = yv_t[:1]; pred_img = x_hat_vm[:1]; in_img = Xv[:1, :1]
                mask = organ_maskv[:1]

                # Input(ch0) | target GT | target Pred, shared scale from input+GT
                grid = torch.cat([in_img, gt_img, pred_img], dim=0)
                grid = grid - grid[0:2].min(); grid = grid / grid[0:2].max().clamp(min=1e-8)
                grid = mask * grid
                # Residual on its own scale
                res = (gt_img - pred_img).abs()
                res = res / res.max().clamp(min=1e-8)

                wandb.log({
                    "val/example": wandb.Image(vutils.make_grid(grid, nrow=3),
                                               caption="Source(ch0) | Target GT | Target Pred"),
                    "val/residual": wandb.Image(vutils.make_grid(res, nrow=1),
                                                caption="| GT - Pred |"),
                    **{f"val/{k}": v for k, v in mean_metrics.items()},
                }, step=global_step)
            else:
                print(f"[VAL] epoch={epoch} " +
                      " ".join(f"{k}={v:.4f}" for k, v in mean_metrics.items()))
            net.train()

            # ReduceLROnPlateau: metric-driven, one step per epoch on the val loss
            if isinstance(sched, ReduceLROnPlateau) and val_loss is not None:
                sched.step(val_loss)
    pbar.close()
    return net
