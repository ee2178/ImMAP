"""
Domain-transfer synthesis training loop for DT-CDLNet.

A thin variant of train_synthesis (training/synthesis.py). Same paired-data pipeline
(X = source contrast(s), y = target contrast, organ_mask, et), same backtracking / metrics /
logging / checkpointing. Three differences, all forced by the CDLNet-family model:

  1. Forward interface. DT-CDLNet is
     `x_hat, dS, y_hat, z = net(X, E=Identity, return_source=True)` (an unrolled model, not a
     plain `pred = net(X)`): x_hat = D_x z^K + dS^K is the TARGET decode, dS is the additive
     ENHANCEMENT map, y_hat is the z -> y SOURCE reconstruction from the same shared code z.
  2. Source-consistency term. loss = L(x_hat, y) + lam_src * L(y_hat, X). The second term
     supervises the source dictionary Dy = B[0] and the sparse-coding stage directly; set
     lam_src = 0 (the default in the config) to train the target path only.
     NOTE: dS is deliberately UNSUPERVISED -- no l1 penalty, no direct ET-mask term. The
     only signal it gets is the image-domain loss on x_hat, so it is free to place
     enhancement wherever that loss wants it. `et_weight` does NOT change this: it reweights
     x_hat inside the enhancing tumor, which reaches dS only through the read-out, exactly
     as the plain reconstruction loss does. The two known failure modes (dS collapsing to
     0, or dS absorbing everything so the model degenerates to a generic residual
     connection) are therefore invisible to the loss, and are left to VISUAL inspection --
     the validation step logs a dS panel on its own scale and the base decode D_x z beside
     the full prediction. Building the model with use_dS=False removes the branch outright
     (the ablation baseline); its dS panel is then uniformly zero.
  3. Projection. `net.project()` runs after every optimizer step to enforce the CDLNet
     constraints (thresholds >= 0, leak in [0,1], dictionaries on the unit ball).
     train_synthesis omits this because U-Nets are unconstrained; unrolled models require it.

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
from visualization.params import get_param_logs
from training.losses import LOSS_REGISTRY, weighted_loss

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
    et_weight=1.0,                   # per-pixel weight on enhancing-tumor pixels of the
                                     # x_hat loss (1.0 = off); needs et_mask: true in the
                                     # data config. See training.losses.weighted_loss --
                                     # this is a PER-PIXEL weight, so the useful range is
                                     # tens to hundreds, not around 1.
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

    et_warned = False
    for epoch in range(start_epoch, num_epochs):
        net.train()
        running_loss, running_src, n_batches = 0.0, 0.0, 0

        for _ in range(steps_per_epoch):
            try:
                X, y, organ_mask, et = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                X, y, organ_mask, et = next(train_iter)
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            organ_mask = organ_mask.to(device, non_blocking=True)
            et = et.to(device, non_blocking=True)

            opt.zero_grad()
            # x_hat: (B, Cx, H, W) target decode;  dS: (B, Cx, H, W) enhancement map;
            # y_hat: (B, C, H, W) source reconstruction
            x_hat, dS, y_hat, z = net(X, E=E, return_source=True)

            # organ_mask is (B, 1, H, W) and broadcasts over channels
            y_t,  x_hat_m = apply_loss_mask(y, x_hat, organ_mask, use_mask)
            X_s,  y_hat_m = apply_loss_mask(X, y_hat, organ_mask, use_mask)

            # ET weighting applies to the TARGET reconstruction only -- same form as
            # train_synthesis, so DT-CDLNet and the U-Net stay directly comparable. dS is
            # still not supervised: it gets the ET signal only through x_hat.
            tgt_loss = weighted_loss(loss_type, x_hat_m, y_t, et, et_weight)
            src_loss = loss_fn(y_hat_m, X_s, None)
            loss = tgt_loss + lam_src * src_loss

            if et_weight != 1 and not et_warned and float(et.sum()) == 0:
                et_warned = True
                print("[dt-synthesis] WARNING: et_weight != 1 but the first batch's ET mask "
                      "is entirely zero. Set et_mask: true in the data config (and make sure "
                      "the h5 has an 'et' dataset), or the weighting is a no-op.")

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
                       **{f"train/{k}": v for k, v in train_metrics.items()}},
                      step=global_step)
        elif not wandb:
            print({"epoch": epoch, "avg_loss": avg_loss, "avg_src": avg_src, **train_metrics})

        # ---- validation ----
        if val_loader is not None and val_every_epochs and (epoch + 1) % val_every_epochs == 0:
            net.eval()
            agg = {"psnr": 0.0, "ssim": 0.0, "nrmse": 0.0, "loss": 0.0, "src_loss": 0.0}
            n_samples = 0
            with torch.no_grad():
                for Xv, yv, organ_maskv, etv in val_loader:
                    Xv = Xv.to(device, non_blocking=True)
                    yv = yv.to(device, non_blocking=True)
                    organ_maskv = organ_maskv.to(device, non_blocking=True)
                    etv = etv.to(device, non_blocking=True)
                    x_hat_v, dS_v, y_hat_v, zv = net(Xv, E=E, return_source=True)
                    bs = Xv.shape[0]

                    yv_t, x_hat_vm = apply_loss_mask(yv, x_hat_v, organ_maskv, use_mask)
                    Xv_s, y_hat_vm = apply_loss_mask(Xv, y_hat_v, organ_maskv, use_mask)

                    # DO NOT CALL .ABS(), WE HAVE NEGATIVE NUMBERS (z-scored data)
                    mets = compute_metrics(yv_t, x_hat_vm, psnr_only=psnr_only)
                    mets = {k: float(v.detach()) for k, v in mets.items()}
                    src_l = float(loss_fn(y_hat_vm, Xv_s, None).item())
                    mets["src_loss"] = src_l
                    # same total the training step optimizes, so val/loss stays comparable
                    mets["loss"] = (float(weighted_loss(loss_type, x_hat_vm, yv_t,
                                                        etv, et_weight).item())
                                    + lam_src * src_l)
                    for k in agg:
                        if k in mets:
                            agg[k] += mets[k] * bs
                    n_samples += bs
            mean_metrics = {k: v / max(n_samples, 1) for k, v in agg.items()}
            val_loss = mean_metrics["loss"]      # <-- capture for the scheduler

            if wandb:
                gt_img = yv_t[:1]; pred_img = x_hat_vm[:1]; in_img = Xv[:1, :1]
                mask = organ_maskv[:1]
                dS_img = (mask * dS_v[:1])                 # enhancement map, brain-masked
                base_img = pred_img - dS_img               # D_x z alone (the base branch)

                # Source(ch0) | base D_x z | target Pred | target GT, shared scale from input+GT
                grid = torch.cat([in_img, base_img, pred_img, gt_img], dim=0)
                lo = grid[[0, 3]].min(); hi = grid[[0, 3]].max()
                grid = mask * ((grid - lo) / (hi - lo).clamp(min=1e-8)).clamp(0, 1)
                # Residual on its own scale
                res = (gt_img - pred_img).abs()
                res = res / res.max().clamp(min=1e-8)
                # dS on ITS OWN scale (it is one-sided and usually small + sparse, so the
                # shared window above would render it as flat black)
                dS_show = dS_img / dS_img.max().clamp(min=1e-8)

                wandb.log({
                    "val/example": wandb.Image(
                        vutils.make_grid(grid, nrow=4),
                        caption="Source(ch0) | base D_x z | Pred (base + dS) | Target GT"),
                    "val/residual": wandb.Image(vutils.make_grid(res, nrow=1),
                                                caption="| GT - Pred |"),
                    "val/dS": wandb.Image(
                        vutils.make_grid(dS_show, nrow=1),
                        caption=f"enhancement map dS  (max={float(dS_img.max()):.3f})"),
                    # ET mask next to dS: the question this panel answers is whether the
                    # enhancement the model produces lands where the tumor actually is.
                    "val/et": wandb.Image(vutils.make_grid(etv[:1], nrow=1),
                                          caption="ET mask"),
                    **{f"val/{k}": v for k, v in mean_metrics.items()},
                }, step=global_step)
                # Parameter values ride the VALIDATION cadence: walking the model costs a
                # host transfer per tensor, and thresholds / step sizes drift on the
                # timescale of training, not of an epoch. Grads are still populated here --
                # every loop zero_grads at the TOP of the next step -- so grad_norm reports
                # the last training step's gradients rather than being absent.
                wandb.log(get_param_logs(net), step=global_step)
            else:
                print(f"[VAL] epoch={epoch} " +
                      " ".join(f"{k}={v:.4f}" for k, v in mean_metrics.items()))
            net.train()

            # ReduceLROnPlateau: metric-driven, one step per epoch on the val loss
            if isinstance(sched, ReduceLROnPlateau) and val_loss is not None:
                sched.step(val_loss)
    pbar.close()
    return net
