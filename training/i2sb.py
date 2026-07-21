"""
I2SB training loop for a denoiser regressor (CDLNet, GroupCDL, ...). Same skeleton as
train_synthesis (epoch = steps_per_epoch gradient steps, averaged-loss backtracking,
checkpoint-on-improvement, wandb logging) but the step is the Schrodinger-bridge regression:

    x0, x1, cond  <- batch          (x0=T1ce target, x1=T1 prior, cond=FLAIR/T1/T2)
    step ~ U{0..interval-1}
    xt = forward_sample(step, x0, x1)                            # bridge interpolant
    out = net(cat[xt, cond]; sigma = forward_std(step))[:, :1]   # predict the clean endpoint x0
    loss = MSE(out, x0)                                          # in the T1ce domain

The schedule design lives in sb/base.py and the bridge algorithm helpers there too; this file only
drives them. The network always predicts x0 directly (the only parameterization).

Conditioning is a single toggle: set the data loader's `cond_idx` and the model's `C` together
(C == 1 + len(cond_idx)). We assert they agree so a mismatch fails loudly rather than silently
mis-slicing channels.
"""

import os
import math

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from tqdm import tqdm

from torch.optim.lr_scheduler import ReduceLROnPlateau

from training.common import save_ckpt, load_ckpt, get_lr, set_lr, apply_loss_mask, snr_loss_weight
from training.losses import LOSS_REGISTRY
from training.metrics import compute_metrics
from sb.base import build_schedule, n_steps, forward_sample, forward_std, predict_x0
from sb.i2sb import i2sb_sample
from visualization.filters import get_filter_grids


def masked_mse(pred, target, organ_mask, use_mask):
    """MSE in the (masked) target domain. Mirrors train_synthesis: multiply by the mask,
    then mean over all elements."""
    target, pred = apply_loss_mask(target, pred, organ_mask, use_mask)
    return F.mse_loss(pred, target)


def _apply_loss(loss_fn, target, pred, organ_mask, use_mask, sigma):
    """Mask (optionally) then evaluate a training.losses registry loss. Registry convention is
    loss_fn(target, pred, sigma) (see train_denoiser); masked complex-mse reproduces masked_mse.
    The network predicts x0 directly, so the loss always sees IMAGES (pred_x0 vs x0) and a
    perceptual loss (e.g. vgg-feature) is meaningful."""
    target, pred = apply_loss_mask(target, pred, organ_mask, use_mask)
    return loss_fn(target, pred, sigma)


def _x0_weighted_mse(x0, pred_x0, mask, use_mask, std_fwd, loss_weight):
    """Per-sample masked x0-MSE reweighted by snr_loss_weight(sigma_t) -- the item-3 t-weighting.
    'snr' reproduces the I2SB eps objective as an x0 loss (toward t=0); 't1' biases toward t=1. This
    is MSE-based, so it bypasses loss_type (VGG etc.)."""
    tgt, prd = apply_loss_mask(x0, pred_x0, mask, use_mask)
    per_sample = ((prd - tgt).abs() ** 2).flatten(1).mean(dim=1)          # (B,) per-sample MSE
    return (snr_loss_weight(std_fwd, loss_weight) * per_sample).mean()


def _split_batch(batch, device):
    """Batch is (x0, x1, cond, mask). cond may have 0 channels (conditioning off) -> None."""
    x0, x1, cond, mask = batch
    x0 = x0.to(device, non_blocking=True)
    x1 = x1.to(device, non_blocking=True)
    mask = mask.to(device, non_blocking=True)
    cond = cond.to(device, non_blocking=True)
    if cond.shape[1] == 0:
        cond = None
    return x0, x1, cond, mask


def train_i2sb(
    net, opt, sched, device,
    train_loader,
    val_loader,
    wandb=None,
    start_epoch=0,
    # ---- generic loop (cfg["training"]) ----
    num_epochs=300,
    steps_per_epoch=200,
    val_every_epochs=10,
    clip_grad=1.0,
    backtrack_thresh=0.5,
    backtrack_factor=0.9,
    use_mask=True,
    loss_type="complex-mse",         # any key in training.losses.LOSS_REGISTRY (e.g. "vgg-feature")
    loss_weight="uniform",           # per-sample t-weighting: "uniform" | "snr" (~t=0) | "t1" (~t=1)
    psnr_only=False,
    # ---- I2SB method (cfg["i2sb"]) ----
    kind="brownian",                 # schedule: "brownian" (tau-parameterized) or "i2sb" (paper)
    tau=0.19,                        # brownian: peak bridge-noise std; max forward_std (sigma) = 2*tau
    n_points=1000,                   # number of discrete bridge steps (paper's "interval")
    beta_max=0.3,                    # i2sb: peak-diffusivity knob of the faithful paper schedule
    deterministic=False,             # drop the bridge / posterior noise (the OT-ODE limit)
    posterior="ddpm",                # reverse update: "ddpm" (moving average) | "interpolant" (x1<->x0_hat)
    clip_denoise=False,
    val_mode="single_pass",          # "single_pass" (one random-step denoise) or "full_recon"
    val_seed=None,
    val_nfe=20,                      # only used when val_mode == "full_recon"
    target_channels=1,
    # ---- paths (cfg["paths"]) ----
    save_dir=None,
    ckpt=None,                       # signature parity; model resume handled in main()
    save_ckpt_fn=save_ckpt,
    **_unused,
):
    net.to(device)
    net.train()

    if loss_type not in LOSS_REGISTRY:
        raise ValueError(f"loss_type {loss_type!r} not in LOSS_REGISTRY {sorted(LOSS_REGISTRY)}.")
    loss_fn = LOSS_REGISTRY[loss_type]

    # bridge schedule: "brownian" = the constant t(1-t) Brownian bridge (tau = peak noise std),
    # "i2sb" = the faithful paper schedule (mirrored-quadratic betas via beta_max). For a fully
    # custom schedule, build your own betas -> sb.base.from_betas(betas).
    bridge = build_schedule(kind=kind, tau=tau, n_points=n_points, beta_max=beta_max, device=device)
    interval = n_steps(bridge)

    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, "net.ckpt")

    # sanity: conditioning channels must match the network's input width (C = 1 + n_cond)
    _assert_cond_matches_model(net, train_loader, device, target_channels)

    best_loss = float("inf")
    train_iter = iter(train_loader)
    total_steps = num_epochs * steps_per_epoch
    pbar = tqdm(total=total_steps, initial=start_epoch * steps_per_epoch,
                desc="I2SB", dynamic_ncols=True)

    for epoch in range(start_epoch, num_epochs):
        net.train()
        running_loss, n_batches = 0.0, 0

        for _ in range(steps_per_epoch):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            x0, x1, cond, mask = _split_batch(batch, device)

            # ----- sample a bridge point and regress the clean endpoint x0 -----
            b = x0.shape[0]
            step = torch.randint(0, interval, (b,), device=device)
            xt = forward_sample(bridge, step, x0, x1, deterministic=deterministic)
            std_fwd = forward_std(bridge, step, xdim=x0.shape[1:])   # (B,1,1,1) noise level

            opt.zero_grad()
            pred_x0 = predict_x0(net, xt, std_fwd, cond=cond, target_channels=target_channels)
            loss = (_apply_loss(loss_fn, x0, pred_x0, mask, use_mask, std_fwd)
                    if loss_weight == "uniform"
                    else _x0_weighted_mse(x0, pred_x0, mask, use_mask, std_fwd, loss_weight))

            loss.backward()
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clip_grad)
            opt.step()
            # Important for our unrolled models
            if hasattr(net, "project"): net.project()
            if sched is not None and not isinstance(sched, ReduceLROnPlateau):
                sched.step()

            running_loss += float(loss.item())
            n_batches += 1
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.3e}", epoch=epoch)

        global_step = (epoch + 1) * steps_per_epoch
        avg_loss = running_loss / max(n_batches, 1)
        nonfinite = not math.isfinite(avg_loss)

        # one-step denoise metrics (pred_x0 vs x0), masked like the loss
        x0_m, pred_m = apply_loss_mask(x0, pred_x0, mask, use_mask)
        train_metrics = compute_metrics(x0_m, pred_m, psnr_only=psnr_only)
        train_metrics = {k: float(v.detach()) for k, v in train_metrics.items()}

        # ---- averaged-loss backtracking (matches the other loops) ----
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
            val_loss = _validate(
                net, bridge, val_loader, device, interval=interval, val_mode=val_mode,
                val_seed=val_seed, use_mask=use_mask, deterministic=deterministic,
                posterior=posterior, clip_denoise=clip_denoise, val_nfe=val_nfe,
                target_channels=target_channels, psnr_only=psnr_only, loss_fn=loss_fn,
                loss_weight=loss_weight, wandb=wandb, global_step=global_step,
            )
            if isinstance(sched, ReduceLROnPlateau) and val_loss is not None:
                sched.step(val_loss)

    pbar.close()
    return net


def _assert_cond_matches_model(net, loader, device, target_channels):
    """Peek one batch: the network's input width self.C must equal target_channels + n_cond."""
    n_cond = 0
    try:
        cond = next(iter(loader))[2]
        n_cond = int(cond.shape[1])
    except Exception:
        pass
    expected_C = target_channels + n_cond
    model_C = getattr(net, "C", None)
    if model_C is not None and model_C != expected_C:
        raise ValueError(
            f"Conditioning/model mismatch: data provides {n_cond} cond channel(s) so the net "
            f"needs C={expected_C}, but model.C={model_C}. Set model.params.C = 1 + len(cond_idx) "
            f"(or cond_idx=[] and C=1 to disable conditioning)."
        )


@torch.no_grad()
def _validate(net, bridge, val_loader, device, *, interval, val_mode, val_seed,
              use_mask, deterministic, posterior, clip_denoise, val_nfe,
              target_channels, psnr_only, loss_fn, loss_weight, wandb, global_step):
    """Validate. Two modes:
      "single_pass" (default) -- draw one random step per batch, run ONE network forward, and
                                  score the single-pass pred_x0 (mirrors the training objective;
                                  cheap). Steps use a fixed seed so metrics are comparable epoch
                                  to epoch.
      "full_recon"            -- run the full val_nfe-step reverse sampler (end-to-end recon).
    """
    net.eval()
    agg = {"loss": 0.0, "psnr": 0.0, "ssim": 0.0, "nrmse": 0.0}
    n_samples = 0
    last = None
    if val_seed is not None:
        gen = torch.Generator(device=device).manual_seed(val_seed)   # fixed val steps -> comparable
    else:
        gen = None
    for batch in val_loader:
        x0, x1, cond, mask = _split_batch(batch, device)
        bs = x0.shape[0]

        if val_mode == "full_recon":
            pred, _, _ = i2sb_sample(
                net, x1, bridge, cond=cond, nfe=val_nfe, deterministic=deterministic,
                posterior=posterior, clip_denoise=clip_denoise,
                target_channels=target_channels, log_count=1, verbose=False,
            )
            sigma_v = torch.zeros(bs, 1, 1, 1, device=device)   # end-to-end: no single step
            loss = _apply_loss(loss_fn, x0, pred, mask, use_mask, sigma_v)
            xt, step = x1, None
        else:  # single_pass: one random step, one forward (same as a training step)
            step = torch.randint(0, interval, (bs,), generator=gen, device=device)
            xt = forward_sample(bridge, step, x0, x1, deterministic=deterministic)
            std_fwd = forward_std(bridge, step, xdim=x0.shape[1:])
            pred = predict_x0(net, xt, std_fwd, cond=cond, target_channels=target_channels)
            loss = (_apply_loss(loss_fn, x0, pred, mask, use_mask, std_fwd)
                    if loss_weight == "uniform"
                    else _x0_weighted_mse(x0, pred, mask, use_mask, std_fwd, loss_weight))

        x0_m, pred_m = apply_loss_mask(x0, pred, mask, use_mask)
        mets = compute_metrics(x0_m, pred_m, psnr_only=psnr_only)
        agg["loss"] += float(loss) * bs
        for k in ("psnr", "ssim", "nrmse"):
            if k in mets:
                agg[k] += float(mets[k].detach()) * bs
        n_samples += bs
        last = (x1, xt, x0_m, pred_m, mask, step)

    mean_metrics = {k: v / max(n_samples, 1) for k, v in agg.items()}

    if wandb and last is not None:
        x1, xt, x0_m, pred_m, mask, step = last
        if val_mode == "single_pass":
            cols = [x1[:1], xt[:1], pred_m[:1], x0_m[:1]]
            cap = f"T1 prior | x_t (step={int(step[0])}) | single-pass pred_x0 | T1ce GT"
        else:
            cols = [x1[:1], pred_m[:1], x0_m[:1]]
            cap = f"T1 prior | I2SB recon (nfe={val_nfe}) | T1ce GT"
        ref = torch.cat([cols[0], cols[-1]], dim=0)          # scale from prior + GT (clean)
        lo = float(ref.amin()); hi = max(float(ref.amax()), lo + 1e-8)
        grid = mask[:1] * torch.cat([((c - lo) / (hi - lo)).clamp(0, 1) for c in cols], dim=0)
        res = (x0_m[:1] - pred_m[:1]).abs(); res = res / res.max().clamp(min=1e-8)
        wandb.log({
            "val/example": wandb.Image(vutils.make_grid(grid, nrow=len(cols)), caption=cap),
            "val/residual": wandb.Image(vutils.make_grid(res, nrow=1), caption="| GT - pred |"),
            **{f"val/{k}": v for k, v in mean_metrics.items()},
        }, step=global_step)
        # learned dictionary filters (works for real or complex CDLNet); no-op if absent
        try:
            wandb.log(get_filter_grids(net), step=global_step)
        except (AttributeError, NotImplementedError, AssertionError):
            pass
    elif not wandb:
        print(f"[VAL {val_mode}] " + " ".join(f"{k}={v:.4f}" for k, v in mean_metrics.items()))

    net.train()
    return mean_metrics["loss"]   # lower is better -> valid for ReduceLROnPlateau
