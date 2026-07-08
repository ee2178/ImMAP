"""
Contrast-synthesis training loop. Same structure as train_denoiser, minus the denoiser
specifics. Step: pred = net(X); loss = recon_loss(pred, y, brain_mask).
X = multi-contrast input (e.g. T1,T2,FLAIR), y = target (e.g. T1ce).

`pretrained` warm-starts the network from a CCL backbone (ccl_encoder.pt) with a partial,
shape-tolerant load. It is applied ONLY on a fresh run (start_epoch == 0) so it never
overrides a resumed checkpoint. Reuses training.common + training.metrics like the others.
"""

import os
import math

import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils
from tqdm import tqdm

from training.common import save_ckpt, load_ckpt, get_lr, set_lr, apply_loss_mask
from training.metrics import compute_metrics
from training.losses import LOSS_REGISTRY

# For reducing LR on plateau
from torch.optim.lr_scheduler import ReduceLROnPlateau

def brain_mask(x, eps=1e-6):
    """Foreground from z-scored input: background is the per-image-per-channel min (constant)."""
    bg = x.amin(dim=(-1, -2), keepdim=True)
    return (x > bg + eps).any(dim=1, keepdim=True).float()       # (B, 1, H, W)

def load_pretrained_backbone(net, path, device):
    """Partial, shape-tolerant warm-start. Reads a 'backbone' payload (ccl_encoder.pt) or a
    full 'model_state_dict'; loads every key whose shape matches, reports the rest. strict=False
    forgives the missing task head (final.*) and any projection-head keys from a full ckpt."""
    ck = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ck, dict) and "backbone" in ck:
        state = ck["backbone"]
    elif isinstance(ck, dict) and "model_state_dict" in ck:
        state = ck["model_state_dict"]
    else:
        state = ck
    own = net.state_dict()
    keep = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    shape_skipped = [k for k, v in state.items() if k in own and own[k].shape != v.shape]
    not_in_model = [k for k in state if k not in own]          # e.g. proj.* from a full ckpt
    own.update(keep)
    net.load_state_dict(own, strict=False)
    random_keys = [k for k in own if k not in state]           # e.g. final.* task head
    print(f"[pretrain] {path}: loaded {len(keep)}/{len(state)} | "
          f"random (not in ckpt): {random_keys[:6]} | "
          f"shape-skipped: {shape_skipped} | ignored-from-ckpt: {not_in_model[:6]}")

def train_synthesis(
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
    loss_type="l1",
    use_mask=True,
    psnr_only=False,
    pretrained=None,                 # path to ccl_encoder.pt; applied only when start_epoch == 0
    save_dir=None,
    ckpt=None,                       # signature parity; resume handled in main()
    save_ckpt_fn=save_ckpt,
):
    net.to(device)

    loss_fn = LOSS_REGISTRY[loss_type]

    # warm-start from pretraining ONLY on a fresh run (not when resuming)
    if pretrained and start_epoch == 0:
        load_pretrained_backbone(net, pretrained, device)

    net.train()
    best_loss = float("inf")
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, "net.ckpt")

    train_iter = iter(train_loader)
    total_steps = num_epochs * steps_per_epoch
    pbar = tqdm(total=total_steps, initial=start_epoch * steps_per_epoch,
                desc="SYNTHESIS", dynamic_ncols=True)

    for epoch in range(start_epoch, num_epochs):
        net.train()
        running_loss, n_batches = 0.0, 0

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
            pred = net(X)                       # (B, 1, H, W)
            
            y, pred = apply_loss_mask(
                y, pred, organ_mask, use_mask,
            )

            loss = loss_fn(pred, y, None)
            loss.backward()
            if clip_grad is not None:
                nn.utils.clip_grad_norm_(net.parameters(), clip_grad)
            opt.step()

            if sched is not None and not isinstance(sched, ReduceLROnPlateau):
                sched.step()

            running_loss += float(loss.item())
            n_batches += 1
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.3e}", epoch=epoch)

        global_step = (epoch + 1) * steps_per_epoch
        avg_loss = running_loss / max(n_batches, 1)
        nonfinite = not math.isfinite(avg_loss)
        
        # y_m = (y+2)/4
        # pred_m = (pred+2)/4
        train_metrics = compute_metrics(y, pred, psnr_only=psnr_only)
        train_metrics = {k: float(v.detach()) for k, v in train_metrics.items()}

        # ---- averaged-loss backtracking ----
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
            wandb.log({"train/loss": avg_loss, "train/lr": opt.param_groups[0]["lr"],
                       "train/epoch": epoch,
                       **{f"train/{k}": v for k, v in train_metrics.items()}}, step=global_step)
        elif not wandb:
            print({"epoch": epoch, "avg_loss": avg_loss, **train_metrics})

        # ---- validation ----
        if val_loader is not None and val_every_epochs and (epoch + 1) % val_every_epochs == 0:
            net.eval()
            agg = {"psnr": 0.0, "ssim": 0.0, "nrmse": 0.0, "loss": 0.0}
            n_samples = 0
            with torch.no_grad():
                for Xv, yv, organ_maskv in val_loader:
                    Xv = Xv.to(device, non_blocking=True)
                    yv = yv.to(device, non_blocking=True)
                    organ_maskv = organ_maskv.to(device, non_blocking=True)
                    pv = net(Xv)
                    bs = Xv.shape[0]
                    # Apply mask
                    yv, pv = apply_loss_mask(
                        yv, pv, organ_maskv, use_mask,
                    )
                    # DO NOT CALL .ABS(), WE HAVE NEGATIVE NUMBERS
                    # Generally speaking, our data is mean 0 variance 1, so for the purposes of metric computations, We can do the following
                    # yv_m = (yv + 2)/4
                    # pv_m = (pv + 2)/4
                    mets = compute_metrics(yv, pv, psnr_only=psnr_only)
                    mets = {k: float(v.detach()) for k, v in mets.items()}
                    mets["loss"] = float(loss_fn(pv, yv, None).item())
                    for k in agg:
                        if k in mets:
                            agg[k] += mets[k] * bs
                    n_samples += bs
            mean_metrics = {k: v / max(n_samples, 1) for k, v in agg.items()}
            val_loss = mean_metrics["loss"]      # <-- capture for the scheduler
                        
            if wandb:
                gt_img = yv[:1]; pred_img = pv[:1]; in_img = Xv[:1, :1]
                mask = organ_maskv[:1]

                # Input | GT | Pred, shared scale from input+GT (unchanged)
                grid = torch.cat([in_img, gt_img, pred_img], dim=0)
                grid = grid - grid[0:2].min(); grid = grid / grid[0:2].max().clamp(min=1e-8)

                # Mask our grid after normalization
                grid = mask*grid
                # Residual on its own symmetric scale: 0.5 = zero error, 0/1 = -/+ max|error|
                res = (gt_img - pred_img).abs()
                res = res / res.max().clamp(min=1e-8)

                wandb.log({
                    "val/example": wandb.Image(vutils.make_grid(grid, nrow=3),
                                               caption="Input(ch0) | T1ce GT | Predicted"),
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
