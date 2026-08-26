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

Two optional loss weightings compose on top of the plain objective (see `_bridge_loss`):
`et_weight` upweights enhancing-tumor pixels (needs the loader's `et_mask: true`), and
`loss_weight` reweights each sample by its noise level. Both default to off.
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
from training.losses import LOSS_REGISTRY, POINTWISE_REGISTRY, weighted_loss
from training.metrics import compute_metrics
from sb.base import build_schedule, n_steps, forward_sample, forward_std, predict_x0
from sb.i2sb import i2sb_sample
from visualization.filters import get_filter_grids
from visualization.params import get_param_logs


def masked_mse(pred, target, organ_mask, use_mask):
    """MSE in the (masked) target domain. Mirrors train_synthesis: multiply by the mask,
    then mean over all elements."""
    target, pred = apply_loss_mask(target, pred, organ_mask, use_mask)
    return F.mse_loss(pred, target)


def _bridge_loss(loss_fn, loss_type, x0, pred_x0, mask, use_mask, et, et_weight,
                 std_fwd, loss_weight):
    """The x0-regression objective, with two INDEPENDENT weightings composed on top of it:

        per-pixel   w_pix = 1 + (et_weight - 1) * et      enhancing-tumor upweighting
        per-sample  w_t   = snr_loss_weight(sigma_t)      the t-weighting ("snr" / "t1")

        loss = mean_i( w_t[i] * mean_j( w_pix[i,j] * err[i,j] ) )

    With both off this is just the masked registry loss `loss_fn(target, pred, sigma)`, so existing
    runs are bit-identical. With et_weight != 1 and loss_weight == "uniform" it is exactly
    losses.weighted_loss (same
    mean-over-all-pixels-of-w*err form), so the et_weight SCALE carries over from train_synthesis
    unchanged -- tune it in the tens-to-hundreds, not around 1 (see weighted_loss's docstring: ET
    covers ~0.3% of a slice, so weight 50 puts ~13% of the objective on the tumor and 330 ~50%).

    Note what each weighting costs: the full LOSS_REGISTRY (vgg-feature, sigma-scaled, the ratio
    losses) is only available when BOTH are off, because neither a perceptual loss nor a ratio loss
    is a per-pixel mean. Any weighting therefore drops to the pointwise error, and `loss_type` must
    name a POINTWISE_REGISTRY entry."""
    tgt, prd = apply_loss_mask(x0, pred_x0, mask, use_mask)

    if loss_weight == "uniform":
        if et_weight == 1:
            return loss_fn(tgt, prd, std_fwd)            # untouched path: whole registry available
        return weighted_loss(loss_type, prd, tgt, et, et_weight)

    # per-sample t-weighting: reduce per sample FIRST, then weight, so w_t is a per-sample scalar
    if loss_type not in POINTWISE_REGISTRY:
        raise ValueError(
            f"loss_weight={loss_weight!r} needs a per-pixel loss, but loss_type {loss_type!r} is "
            f"not one. Use one of {sorted(POINTWISE_REGISTRY)}, or loss_weight='uniform'.")
    err = POINTWISE_REGISTRY[loss_type](prd, tgt)
    if et_weight != 1:
        err = (1.0 + (et_weight - 1.0) * et.expand_as(err)) * err
    per_sample = err.flatten(1).mean(dim=1)                                # (B,)
    return (snr_loss_weight(std_fwd, loss_weight) * per_sample).mean()


def _split_batch(batch, device):
    """Batch is (x0, x1, cond, mask) or (x0, x1, cond, mask, et) when the loader has et_mask=True.
    cond may have 0 channels (conditioning off) -> None; et is None when absent."""
    x0, x1, cond, mask = batch[:4]
    et = batch[4] if len(batch) > 4 else None
    x0 = x0.to(device, non_blocking=True)
    x1 = x1.to(device, non_blocking=True)
    mask = mask.to(device, non_blocking=True)
    cond = cond.to(device, non_blocking=True)
    if cond.shape[1] == 0:
        cond = None
    if et is not None:
        et = et.to(device, non_blocking=True)
    return x0, x1, cond, mask, et


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
    et_weight=1.0,                   # per-pixel enhancing-tumor weight; 1 = off. Needs the loader's
                                     # et_mask: true. Tune in the tens-to-hundreds (see _bridge_loss).
    psnr_only=False,
    data_range=1.0,                  # peak-to-peak range of the data, for PSNR and SSIM. 2.0 for
                                     # data in [-1, 1]. Must match the loader's actual scaling or
                                     # every reported dB is offset by 20*log10(data_range).
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

    # Some regressors carry their OWN copy of the schedule because they need more than sigma:
    # SBCDLNet inverts std_fwd to recover (mu_0, mu_1, sigma_eff) for its two-fidelity step. Its
    # model.params therefore duplicate cfg["i2sb"], and a disagreement would look up the wrong
    # bridge coefficients at every step -- silently, training happily to convergence on nonsense.
    # This is the sweep failure mode in particular: overriding cfg["i2sb"]["beta_max"] without
    # also overriding model.params.beta_max. Fail at startup instead.
    if hasattr(net, "assert_schedule_matches"):
        net.assert_schedule_matches(bridge)

    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, "net.ckpt")

    # sanity: conditioning width vs the model, and that an ET run actually has ET masks
    _assert_batch_matches_config(net, train_loader, device, target_channels, et_weight, loss_type)

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
            x0, x1, cond, mask, et = _split_batch(batch, device)

            # ----- sample a bridge point and regress the clean endpoint x0 -----
            b = x0.shape[0]
            step = torch.randint(0, interval, (b,), device=device)
            xt = forward_sample(bridge, step, x0, x1, deterministic=deterministic)
            std_fwd = forward_std(bridge, step, xdim=x0.shape[1:])   # (B,1,1,1) noise level

            opt.zero_grad()
            pred_x0 = predict_x0(net, xt, std_fwd, cond=cond, target_channels=target_channels)
            loss = _bridge_loss(loss_fn, loss_type, x0, pred_x0, mask, use_mask, et, et_weight,
                                std_fwd, loss_weight)

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
        train_metrics = compute_metrics(x0_m, pred_m, psnr_only=psnr_only, data_range=data_range)
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
                       **{f"train/{k}": v for k, v in train_metrics.items()}},
                      step=global_step)
        elif not wandb:
            print({"epoch": epoch, "avg_loss": avg_loss, **train_metrics})

        # ---- validation ----
        if val_loader is not None and val_every_epochs and (epoch + 1) % val_every_epochs == 0:
            val_loss = _validate(
                net, bridge, val_loader, device, interval=interval, val_mode=val_mode,
                val_seed=val_seed, use_mask=use_mask, deterministic=deterministic,
                posterior=posterior, clip_denoise=clip_denoise, val_nfe=val_nfe,
                target_channels=target_channels, psnr_only=psnr_only, loss_fn=loss_fn,
                loss_type=loss_type, loss_weight=loss_weight, et_weight=et_weight,
                data_range=data_range, wandb=wandb, global_step=global_step,
            )
            if isinstance(sched, ReduceLROnPlateau) and val_loss is not None:
                sched.step(val_loss)

    pbar.close()
    return net


def _assert_batch_matches_config(net, loader, device, target_channels, et_weight, loss_type):
    """Peek one batch and fail loudly on the two config mismatches that would otherwise be silent:
    the network's input width (C = target_channels + n_cond), and an ET-weighted run whose loader
    is not returning ET masks (which would train the ordinary objective under an ET config)."""
    batch = None
    n_cond = 0
    try:
        batch = next(iter(loader))
        n_cond = int(batch[2].shape[1])
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

    if et_weight == 1:
        return
    if loss_type not in POINTWISE_REGISTRY:
        raise ValueError(
            f"et_weight={et_weight} needs a per-pixel loss, but loss_type {loss_type!r} is not "
            f"one. Use one of {sorted(POINTWISE_REGISTRY)}, or set et_weight: 1.")
    if batch is not None and len(batch) < 5:
        raise ValueError(
            f"et_weight={et_weight} but the loader returned a {len(batch)}-tuple (no ET mask). "
            f"Set et_mask: true in BOTH data.train and data.val, or set et_weight: 1.")
    if batch is not None and float(batch[4].sum()) == 0:
        print(f"[i2sb] WARNING: et_weight={et_weight} but the first batch's ET mask is entirely "
              f"zero. That is normal for a batch of tumor-free slices, but if it persists check "
              f"the h5 'et' dataset.")


@torch.no_grad()
def _validate(net, bridge, val_loader, device, *, interval, val_mode, val_seed,
              use_mask, deterministic, posterior, clip_denoise, val_nfe,
              target_channels, psnr_only, loss_fn, loss_type, loss_weight, et_weight,
              data_range, wandb, global_step):
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
    et_sse, et_n = 0.0, 0.0      # pooled over the whole split, not averaged per batch: most
                                 # batches contain little or no tumor, so a mean of per-batch
                                 # ET-PSNRs would be dominated by the emptiest ones
    last = None
    if val_seed is not None:
        gen = torch.Generator(device=device).manual_seed(val_seed)   # fixed val steps -> comparable
    else:
        gen = None
    for batch in val_loader:
        x0, x1, cond, mask, et = _split_batch(batch, device)
        bs = x0.shape[0]

        if val_mode == "full_recon":
            pred, _, _ = i2sb_sample(
                net, x1, bridge, cond=cond, nfe=val_nfe, deterministic=deterministic,
                posterior=posterior, clip_denoise=clip_denoise,
                target_channels=target_channels, log_count=1, verbose=False,
            )
            # end-to-end: there is no single step, so no t-weighting either (sigma is 0 and the
            # per-sample weight would be meaningless) -- but ET weighting still applies.
            sigma_v = torch.zeros(bs, 1, 1, 1, device=device)
            loss = _bridge_loss(loss_fn, loss_type, x0, pred, mask, use_mask, et, et_weight,
                                sigma_v, "uniform")
            xt, step = x1, None
        else:  # single_pass: one random step, one forward (same as a training step)
            step = torch.randint(0, interval, (bs,), generator=gen, device=device)
            xt = forward_sample(bridge, step, x0, x1, deterministic=deterministic)
            std_fwd = forward_std(bridge, step, xdim=x0.shape[1:])
            pred = predict_x0(net, xt, std_fwd, cond=cond, target_channels=target_channels)
            loss = _bridge_loss(loss_fn, loss_type, x0, pred, mask, use_mask, et, et_weight,
                                std_fwd, loss_weight)

        x0_m, pred_m = apply_loss_mask(x0, pred, mask, use_mask)
        mets = compute_metrics(x0_m, pred_m, psnr_only=psnr_only, data_range=data_range)
        agg["loss"] += float(loss) * bs
        for k in ("psnr", "ssim", "nrmse"):
            if k in mets:
                agg[k] += float(mets[k].detach()) * bs
        if et is not None:
            et_sse += float((et * (x0_m - pred_m).abs() ** 2).sum())
            et_n += float(et.sum())
        n_samples += bs
        last = (x1, xt, x0_m, pred_m, mask, step)

    mean_metrics = {k: v / max(n_samples, 1) for k, v in agg.items()}
    # ET PSNR, area-normalized over the split, on the same data_range as metrics.psnr.
    # This is the number et_weight is meant to move; whole-brain psnr barely registers a change
    # confined to ~0.3% of the voxels.
    if et_n > 0:
        mean_metrics["et_psnr"] = 10.0 * math.log10(data_range ** 2 / (et_sse / et_n + 1e-12))

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
            # Parameter values ride the VALIDATION cadence: walking the model costs a
            # host transfer per tensor, and thresholds / step sizes drift on the
            # timescale of training, not of an epoch. Grads are still populated here --
            # every loop zero_grads at the TOP of the next step -- so grad_norm reports
            # the last training step's gradients rather than being absent.
            wandb.log(get_param_logs(net), step=global_step)
            wandb.log(get_filter_grids(net), step=global_step)
        except (AttributeError, NotImplementedError, AssertionError):
            pass
    elif not wandb:
        print(f"[VAL {val_mode}] " + " ".join(f"{k}={v:.4f}" for k, v in mean_metrics.items()))

    net.train()
    return mean_metrics["loss"]   # lower is better -> valid for ReduceLROnPlateau
