"""
Contrast-synthesis training loop. Same structure as train_denoiser, minus the denoiser
specifics. Step: pred = net(X); loss = recon_loss(pred, y, brain_mask).
X = multi-contrast input (e.g. T1,T2,FLAIR), y = target (e.g. T1ce).

RESIDUAL MODE (`residual_mode=True`)
-----------------------------------
Instead of regressing the target itself, the network is supervised on the DIFFERENCE
between the target and one input channel -- the "anchor" -- picked by `residual_src_idx`:

    src    = X[:, residual_src_idx]              # the T1 anchor, already z-scored
    target = y - src                             # e.g. T1ce - T1, what the net predicts
    T1ce-domain estimate = pred + src

The point is where the loss spends its budget. Regressing T1ce directly, almost all of
the error mass is bulk anatomy that T1 already contains, so the enhancement -- the part
that actually distinguishes the two contrasts -- is a small fraction of the objective.
T1ce - T1 is near zero except at strong edges and in the enhancing tumor, so the loss is
concentrated exactly there.

Only the forward pass changes. Both sides of the loss are brain-masked as usual, and
because the anchor is masked identically, adding it back commutes with the mask:
mask*(y - src) + mask*src == mask*y. So metrics are reported in the ORDINARY T1ce domain
(`psnr`, `ssim`, `nrmse`) and stay directly comparable to a non-residual run, with the
residual-domain versions logged alongside as `delta_*`. `val/residual` (|GT - Pred|) is
identical in both domains -- the anchor cancels -- so it is unchanged.

Validation additionally logs the ground-truth and predicted residuals side by side under
`val/delta`, on a shared symmetric diverging scale (white at zero, red positive, blue
negative) -- a gray colormap would render a mostly-zero signed map as flat mid-gray and
hide the sign entirely.

ET-WEIGHTED SUPERVISION (`et_weight != 1`)
------------------------------------------
The loader's fourth item is the enhancing-tumor mask, and `et_weight` counts pixels inside
it that many times as heavily in the ordinary loss:

    loss = mean_i( w_i * err_i ) ,     w_i = 1 + (et_weight - 1) * et_i

`et_weight = 1` is exactly the unweighted loss. This is a PER-PIXEL weight (see
training.losses.weighted_loss), so the tumor's share of the objective is roughly
`f*et_weight / (1 - f + f*et_weight)` at ET area fraction f ~ 0.003: et_weight = 50 puts
~13% of the objective on the tumor, et_weight = 330 puts ~50% there. Tune in the tens to
hundreds, not around 1.

It deliberately does NOT normalize by the ET area. A region MEAN -- which is what the
previous `lam_et` * region_loss form computed -- has a gradient scaling as 1/|ET|, and
since enhancing tumor is well under 1% of a slice and missing from many, that made the
per-pixel gradient swing ~2 orders of magnitude between batches and let a handful of
pixels set the update direction for every parameter (clip_grad_norm_ rescales the sum, so
it preserves that ratio rather than fixing it). Requires `et_mask: true` in the data
config; a one-time warning fires if et_weight != 1 and the first batch's mask is empty.

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
from training.losses import LOSS_REGISTRY, weighted_loss

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


def predict(net, X):
    """`pred = net(X)`, tolerating the CDLNet-family multi-output signature.

    Unrolled models return a tuple whose FIRST element is the image estimate
    (DTCDLNet: (x_hat, dS, z); GroupCDL: (x_hat, z)) and default their forward operator
    to Identity, so a bare call works. U-Nets return the tensor itself. Plain CDLNet is
    not usable here -- its `E` argument has no default.
    """
    out = net(X)
    return out[0] if isinstance(out, (tuple, list)) else out


def anchor_channel(X, idx):
    """The residual anchor (e.g. T1) as (B, 1, H, W).

    The bounds check is not decoration: `X[:, i:i+1]` with an out-of-range `i` returns an
    EMPTY tensor rather than raising, which would surface as a silent NaN loss instead of
    a config error.
    """
    if not 0 <= idx < X.shape[1]:
        raise IndexError(f"residual_src_idx={idx} is out of range for an input with "
                         f"{X.shape[1]} channels")
    return X[:, idx:idx + 1]


def diverging_rgb(x, vmax=None, eps=1e-8):
    """Signed (B, 1, H, W) map -> (B, 3, H, W) RGB on a diverging blue-white-red scale
    (matplotlib 'bwr'): white at 0, red positive, blue negative, clamped to [-vmax, vmax].

    Pass an explicit `vmax` to put several maps on ONE shared scale -- without that, a
    per-map max makes a good prediction and a bad one look equally saturated.
    """
    x = x[:, :1]
    if vmax is None:
        vmax = x.abs().amax()
    t = (x / max(float(vmax), eps)).clamp(-1.0, 1.0)
    pos, neg = t.clamp(min=0.0), (-t).clamp(min=0.0)
    return torch.cat([1.0 - neg, 1.0 - neg - pos, 1.0 - pos], dim=1)


def synthesis_metrics(target, pred, src, organ_mask, use_mask, psnr_only=False):
    """Metrics for one already-brain-masked (target, pred) pair.

    Plain mode (`src is None`): metrics on the network's own target.
    Residual mode: `delta_*` measure the supervised residual directly, and the unprefixed
    `psnr`/`ssim`/`nrmse` are lifted back to the T1ce domain by adding the identically
    masked anchor to both sides -- i.e. exactly the numbers a non-residual run reports.
    """
    if src is None:
        mets = compute_metrics(target, pred, psnr_only=psnr_only)
    else:
        mets = {f"delta_{k}": v
                for k, v in compute_metrics(target, pred, psnr_only=psnr_only).items()}
        src = src * organ_mask if use_mask else src    # mask*(y-src) + mask*src == mask*y
        mets.update(compute_metrics(target + src, pred + src, psnr_only=psnr_only))

        # rms(predicted residual) / rms(true residual). This is the collapse detector: the
        # degenerate solution in residual mode is to predict 0 everywhere, which reconstructs
        # T1ce as exactly the T1 anchor -- a plausible-looking image that PSNR and SSIM both
        # reward, because T1 already carries all the bulk anatomy. 1.0 = right magnitude,
        # -> 0 = the net has stopped predicting enhancement at all.
        mets["resid_ratio"] = (pred.pow(2).mean().sqrt()
                               / target.pow(2).mean().sqrt().clamp_min(1e-8))
    return mets


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
    residual_mode=False,             # supervise on y - X[:, residual_src_idx] (see docstring)
    residual_src_idx=0,              # which INPUT channel is the anchor; with
                                     # input_idx=[0,1,3] (flair,t1,t2) the T1 anchor is 1
    et_weight=1.0,                   # per-pixel weight on enhancing-tumor pixels (1.0 =
                                     # off); needs et_mask: true in the data config. See
                                     # training.losses.weighted_loss -- this is a PER-PIXEL
                                     # weight, so the useful range is tens to hundreds.
    pretrained=None,                 # path to ccl_encoder.pt; applied only when start_epoch == 0
    save_dir=None,
    ckpt=None,                       # signature parity; resume handled in main()
    save_ckpt_fn=save_ckpt,
):
    net.to(device)

    if loss_type not in LOSS_REGISTRY:      # fail at setup, not on the first step
        raise ValueError(f"unknown loss_type {loss_type!r}; "
                         f"expected one of {sorted(LOSS_REGISTRY)}")

    # warm-start from pretraining ONLY on a fresh run (not when resuming)
    if pretrained and start_epoch == 0:
        load_pretrained_backbone(net, pretrained, device)

    if residual_mode:
        print(f"[synthesis] residual mode: supervising on target - X[:, {residual_src_idx}]")

    net.train()
    best_loss = float("inf")
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, "net.ckpt")

    train_iter = iter(train_loader)
    total_steps = num_epochs * steps_per_epoch
    pbar = tqdm(total=total_steps, initial=start_epoch * steps_per_epoch,
                desc="SYNTHESIS", dynamic_ncols=True)

    et_warned = False
    for epoch in range(start_epoch, num_epochs):
        net.train()
        running_loss, n_batches = 0.0, 0

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

            # residual mode: the net predicts y - src, not y. Everything downstream
            # (mask, loss, backtracking) is unchanged; `src` carries the anchor forward
            # so metrics and images can be lifted back to the T1ce domain.
            src = anchor_channel(X, residual_src_idx) if residual_mode else None
            target = y - src if residual_mode else y

            opt.zero_grad()
            pred = predict(net, X)              # (B, 1, H, W)

            target_m, pred_m = apply_loss_mask(
                target, pred, organ_mask, use_mask,
            )

            # ET pixels counted et_weight times as heavily inside the ordinary loss;
            # et_weight = 1 is the plain unweighted loss.
            loss = weighted_loss(loss_type, pred_m, target_m, et, et_weight)

            if et_weight != 1 and not et_warned and float(et.sum()) == 0:
                et_warned = True
                print("[synthesis] WARNING: et_weight != 1 but the first batch's ET mask is "
                      "entirely zero. Set et_mask: true in the data config (and make sure "
                      "the h5 has an 'et' dataset), or the weighting is a no-op.")

            loss.backward()
            if clip_grad is not None:
                nn.utils.clip_grad_norm_(net.parameters(), clip_grad)
            opt.step()
            if hasattr(net, "project"):         # CDLNet-family constraint projection
                net.project()

            if sched is not None and not isinstance(sched, ReduceLROnPlateau):
                sched.step()

            running_loss += float(loss.item())
            n_batches += 1
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.3e}", epoch=epoch)

        global_step = (epoch + 1) * steps_per_epoch
        avg_loss = running_loss / max(n_batches, 1)
        nonfinite = not math.isfinite(avg_loss)

        train_metrics = synthesis_metrics(target_m, pred_m, src, organ_mask,
                                          use_mask, psnr_only=psnr_only)
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
            wandb.log({"train/loss": avg_loss,
                       "train/lr": opt.param_groups[0]["lr"], "train/epoch": epoch,
                       **{f"train/{k}": v for k, v in train_metrics.items()}}, step=global_step)
        elif not wandb:
            print({"epoch": epoch, "avg_loss": avg_loss, **train_metrics})

        # ---- validation ----
        if val_loader is not None and val_every_epochs and (epoch + 1) % val_every_epochs == 0:
            net.eval()
            agg = {"psnr": 0.0, "ssim": 0.0, "nrmse": 0.0, "loss": 0.0}
            if residual_mode:
                agg.update({"delta_psnr": 0.0, "delta_ssim": 0.0, "delta_nrmse": 0.0,
                            "resid_ratio": 0.0})
            n_samples = 0
            with torch.no_grad():
                for Xv, yv, organ_maskv, etv in val_loader:
                    Xv = Xv.to(device, non_blocking=True)
                    yv = yv.to(device, non_blocking=True)
                    organ_maskv = organ_maskv.to(device, non_blocking=True)
                    etv = etv.to(device, non_blocking=True)

                    src_v = anchor_channel(Xv, residual_src_idx) if residual_mode else None
                    target_v = yv - src_v if residual_mode else yv

                    pv = predict(net, Xv)
                    bs = Xv.shape[0]
                    # Apply mask
                    target_vm, pv_m = apply_loss_mask(
                        target_v, pv, organ_maskv, use_mask,
                    )
                    # DO NOT CALL .ABS(), WE HAVE NEGATIVE NUMBERS
                    # Generally speaking, our data is mean 0 variance 1, so for the purposes of metric computations, We can do the following
                    # yv_m = (yv + 2)/4
                    # pv_m = (pv + 2)/4
                    mets = synthesis_metrics(target_vm, pv_m, src_v, organ_maskv,
                                             use_mask, psnr_only=psnr_only)
                    mets = {k: float(v.detach()) for k, v in mets.items()}
                    # same total the training step optimizes, so val/loss stays comparable
                    mets["loss"] = float(weighted_loss(loss_type, pv_m, target_vm,
                                                       etv, et_weight).item())
                    for k in agg:
                        if k in mets:
                            agg[k] += mets[k] * bs
                    n_samples += bs
            mean_metrics = {k: v / max(n_samples, 1) for k, v in agg.items()}
            val_loss = mean_metrics["loss"]      # <-- capture for the scheduler

            if wandb:
                mask = organ_maskv[:1]
                in_img = Xv[:1, residual_src_idx:residual_src_idx + 1] if residual_mode \
                    else Xv[:1, :1]
                if residual_mode:
                    # lift both sides back to T1ce; the anchor is masked to match
                    src_m = (src_v[:1] * mask) if use_mask else src_v[:1]
                    gt_img = target_vm[:1] + src_m
                    pred_img = pv_m[:1] + src_m
                else:
                    gt_img = target_vm[:1]; pred_img = pv_m[:1]

                # Input | GT | Pred, shared scale from input+GT (unchanged)
                grid = torch.cat([in_img, gt_img, pred_img], dim=0)
                grid = grid - grid[0:2].min(); grid = grid / grid[0:2].max().clamp(min=1e-8)

                # Mask our grid after normalization
                grid = mask*grid
                # Residual on its own symmetric scale: 0.5 = zero error, 0/1 = -/+ max|error|
                # (identical in both modes -- the anchor cancels out of GT - Pred)
                res = (gt_img - pred_img).abs()
                res = res / res.max().clamp(min=1e-8)

                in_cap = f"T1 anchor(ch{residual_src_idx})" if residual_mode else "Input(ch0)"
                log = {
                    "val/example": wandb.Image(vutils.make_grid(grid, nrow=3),
                                               caption=f"{in_cap} | T1ce GT | Predicted"),
                    "val/residual": wandb.Image(vutils.make_grid(res, nrow=1),
                                                caption="| GT - Pred |"),
                    **{f"val/{k}": v for k, v in mean_metrics.items()},
                }

                if residual_mode:
                    # The supervised quantity itself, GT vs prediction, on ONE shared
                    # symmetric scale so over-/under-shoot is readable at a glance. The ET
                    # mask rides along as a third panel on the same scale (it is 0/1, so it
                    # renders solid red on white): without it you cannot tell whether a weak
                    # prediction is missing the tumor or the tumor is simply not in view.
                    d_gt, d_pred = target_vm[:1], pv_m[:1]
                    vmax = torch.maximum(d_gt.abs().amax(), d_pred.abs().amax())
                    delta = torch.cat([diverging_rgb(d_gt, vmax),
                                       diverging_rgb(d_pred, vmax),
                                       diverging_rgb(etv[:1], 1.0)], dim=0)
                    rr = mean_metrics.get("resid_ratio", float("nan"))
                    log["val/delta"] = wandb.Image(
                        vutils.make_grid(delta, nrow=3),
                        caption=f"residual (T1ce - T1): GT | Pred | ET mask  "
                                f"[bwr, white=0, ±{float(vmax):.3f}] "
                                f"rms(pred)/rms(gt)={rr:.3f}")

                wandb.log(log, step=global_step)

            else:
                print(f"[VAL] epoch={epoch} " +
                      " ".join(f"{k}={v:.4f}" for k, v in mean_metrics.items()))
            net.train()
                    # ReduceLROnPlateau: metric-driven, one step per epoch on the val loss

            if isinstance(sched, ReduceLROnPlateau) and val_loss is not None:
                sched.step(val_loss)
    pbar.close()
    return net
