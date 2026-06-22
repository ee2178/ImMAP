"""
CCL pretraining loop, structured like train_denoiser.

Differences from the denoiser loop: no awgn / Identity operator / view_as_real branch /
PSNR-SSIM; the step is feats = net(X); loss = ccl(feats, y_true). The averaged-loss
backtracking, checkpointing (via training.common), and wandb logging are unchanged.

The backbone (built by build_model) must return an embedding map (B, proj_dim, H', W').
On every improvement it also exports the backbone weights to ccl_encoder.pt for the
downstream segmentation finetuning stage (projection head dropped).

Place the loss at training/ccl_loss.py.
"""

import os
import math

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from training.common import save_ckpt, load_ckpt, get_lr, set_lr
from training.ccl_loss import ConstrainedContrastiveLoss


def train_ccl(
    net, opt, sched, device,
    train_loader,
    val_loader,
    wandb=None,
    start_epoch=0,
    num_epochs=100,
    steps_per_epoch=200,
    val_every_epochs=5,
    clip_grad=1.0,
    backtrack_thresh=0.5,        # in LOSS units (lower is better), not dB
    backtrack_factor=0.9,
    # --- CCL loss hyperparameters (from cfg["training"]) ---
    patch_size=4,
    topk=100,
    num_samples_loss_eval=100,
    temperature=0.1,
    contrastive_loss_type=2,
    partial_decoder=True,
    use_mask_sampling=True,
    # --- paths (from cfg["paths"]) ---
    save_dir=None,
    ckpt=None,                   # accepted for signature parity; resume handled in main()
    save_ckpt_fn=save_ckpt,
):
    net.to(device)
    net.train()

    ccl = ConstrainedContrastiveLoss(
        patch_size=patch_size, topk=topk,
        num_samples_loss_eval=num_samples_loss_eval, temperature=temperature,
        contrastive_loss_type=contrastive_loss_type,
        use_mask_sampling=use_mask_sampling, partial_decoder=partial_decoder,
    ).to(device)

    # resolution invariant: backbone stride must match the constraint downsampling
    ds_factor = getattr(net, "downsample_factor", None)
    if partial_decoder and ds_factor is not None and ds_factor != patch_size:
        raise ValueError(f"net.downsample_factor ({ds_factor}) != patch_size ({patch_size}); "
                         f"the constraint maps and the feature grid would not align.")

    best_loss = float("inf")
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, "net.ckpt")
    enc_path = os.path.join(save_dir, "ccl_encoder.pt")

    train_iter = iter(train_loader)
    total_steps = num_epochs * steps_per_epoch
    pbar = tqdm(total=total_steps, initial=start_epoch * steps_per_epoch,
                desc="CCL-PRETRAIN", dynamic_ncols=True)

    for epoch in range(start_epoch, num_epochs):
        net.train()
        running_loss = 0.0
        n_batches = 0

        # ============== ONE EPOCH == steps_per_epoch GRADIENT STEPS ==============
        for _ in range(steps_per_epoch):
            try:
                X, y_true = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                X, y_true = next(train_iter)
            X = X.to(device, non_blocking=True)
            y_true = y_true.to(device, non_blocking=True)

            opt.zero_grad()
            feats = net(X)                       # (B, proj_dim, H', W')
            loss = ccl(feats, y_true)
            loss.backward()
            if clip_grad is not None:
                nn.utils.clip_grad_norm_(net.parameters(), clip_grad)
            opt.step()
            if sched is not None:
                sched.step()

            running_loss += float(loss.item())
            n_batches += 1
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.3e}", epoch=epoch)

        global_step = (epoch + 1) * steps_per_epoch
        avg_loss = running_loss / max(n_batches, 1)
        nonfinite = not math.isfinite(avg_loss)

        # ============== averaged-loss backtracking ==============
        if nonfinite or (avg_loss > best_loss + backtrack_thresh):
            reason = "non-finite loss" if nonfinite else (
                f"avg loss {avg_loss:.3e} > best {best_loss:.3e} + {backtrack_thresh}")
            print(f"[epoch {epoch}] {reason} — backtracking")
            if os.path.exists(ckpt_path) and math.isfinite(best_loss):
                net, opt, sched, _ = load_ckpt(
                    ckpt_path, model=net, optimizer=opt, scheduler=sched, device=device)
                new_lr = np.array(get_lr(opt)) * backtrack_factor
                set_lr(opt, new_lr)
                print("Updated LR:", new_lr)
            else:
                raise RuntimeError(f"Backtrack requested at epoch {epoch} but no valid "
                                   f"checkpoint exists yet (best_loss={best_loss}).")

        # ============== save only on genuine improvement ==============
        elif save_ckpt_fn and avg_loss < best_loss:
            save_ckpt_fn(ckpt_path, model=net, optimizer=opt, scheduler=sched, step=global_step)
            if hasattr(net, "backbone_state_dict"):
                torch.save({"backbone": net.backbone_state_dict(),
                            "downsample_factor": ds_factor}, enc_path)
            best_loss = avg_loss

        # ============== logging ==============
        if wandb and not nonfinite:
            wandb.log({"train/loss": avg_loss,
                       "train/lr": opt.param_groups[0]["lr"],
                       "train/epoch": epoch}, step=global_step)
        elif not wandb:
            print({"epoch": epoch, "avg_loss": avg_loss})

        # ============== validation (same CCL loss on held-out subjects) ==============
        if val_loader is not None and val_every_epochs and (epoch + 1) % val_every_epochs == 0:
            net.eval()
            vtot, vn = 0.0, 0
            with torch.no_grad():
                for Xv, yv in val_loader:
                    Xv = Xv.to(device, non_blocking=True)
                    yv = yv.to(device, non_blocking=True)
                    vtot += float(ccl(net(Xv), yv).item()) * Xv.shape[0]
                    vn += Xv.shape[0]
            vloss = vtot / max(vn, 1)
            if wandb:
                wandb.log({"val/loss": vloss}, step=global_step)
            else:
                print(f"[VAL] epoch={epoch} loss={vloss:.6f}")
            net.train()

    pbar.close()
    return net
