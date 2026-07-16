"""
I2SB training loop for a CDLNet denoiser. Same skeleton as train_synthesis (epoch =
steps_per_epoch gradient steps, averaged-loss backtracking, checkpoint-on-improvement,
wandb logging) but the step is the Schrodinger-bridge regression:

    x0, x1, cond  <- batch          (x0=T1ce target, x1=T1 prior, cond=FLAIR/T1/T2)
    step ~ U{0..interval-1}
    xt = q_sample(step, x0, x1)                                   # bridge interpolant
    out = CDLNet(cat[xt, cond]; sigma = std_fwd(step))[:, :1]     # denoise xt
    loss = MSE(out, x0)               (parameterization="x0",  in the T1ce domain)
         | MSE(out, (xt-x0)/std_fwd)  (parameterization="eps", faithful Eq 12)

An EMA of the weights is maintained (I2SB relies on it for sample quality) and used for the
validation sampling pass. The schedule design lives in physics/bbridge.py and the bridge
algorithm functions in diffusion/i2sb.py; this file only drives them.

Conditioning is a single toggle: set the data loader's `cond_idx` and the model's `C` together
(C == 1 + len(cond_idx)). We assert they agree so a mismatch fails loudly rather than silently
mis-slicing channels.
"""

import os
import math
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from tqdm import tqdm

from torch.optim.lr_scheduler import ReduceLROnPlateau

from training.common import save_ckpt, load_ckpt, get_lr, set_lr, apply_loss_mask
from training.metrics import compute_metrics
from physics.bbridge import build_bridge, n_steps
from diffusion.i2sb import (
    cdlnet_pred, i2sb_sample, q_sample, compute_label, compute_pred_x0, get_std_fwd,
)
from visualization.filters import get_filter_grids


# ---------------------------------------------------------------------------
# EMA (tiny; avoids a torch_ema dependency). Shadows the module's Parameters in place,
# so load_ckpt's in-place load during backtracking keeps the references valid.
# ---------------------------------------------------------------------------
class EMA:
    def __init__(self, parameters, decay):
        self.decay = decay
        self.params = [p for p in parameters if p.requires_grad]
        self.shadow = [p.detach().clone() for p in self.params]

    @torch.no_grad()
    def update(self):
        for s, p in zip(self.shadow, self.params):
            s.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @contextmanager
    def average_parameters(self):
        backup = [p.detach().clone() for p in self.params]
        for p, s in zip(self.params, self.shadow):
            p.data.copy_(s)
        try:
            yield
        finally:
            for p, b in zip(self.params, backup):
                p.data.copy_(b)

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, sd):
        self.decay = sd["decay"]
        for s, v in zip(self.shadow, sd["shadow"]):
            s.copy_(v.to(s.device))


def masked_mse(pred, target, organ_mask, use_mask):
    """MSE in the (masked) target domain. Mirrors train_synthesis: multiply by the mask,
    then mean over all elements."""
    target, pred = apply_loss_mask(target, pred, organ_mask, use_mask)
    return F.mse_loss(pred, target)


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
    psnr_only=False,
    # ---- I2SB method (cfg["i2sb"]) ----
    bridge_type="brownian",          # "brownian" (tau-parameterized) or "i2sb" (paper baseline)
    tau=0.19,                        # brownian: peak bridge-noise std; max std_fwd (sigma) = 2*tau
    n_points=1000,                   # number of discrete bridge steps (paper's "interval")
    bridge_shape="constant",         # brownian: "constant" (t(1-t)) or "symmetric" (paper profile)
    beta_max=0.3,                    # i2sb: peak-diffusivity knob of the faithful paper schedule
    ot_ode=False,
    parameterization="x0",
    clip_denoise=False,
    ema_decay=0.99,
    val_mode="single_pass",          # "single_pass" (one random-step denoise) or "full_recon"
    val_seed = None,
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

    # bridge schedule toggle: "brownian" = tau-parameterized (t(1-t) or paper profile), "i2sb" =
    # the faithful paper schedule (symmetric quadratic betas via beta_max) for baseline comparison.
    # For a fully custom schedule, build your own betas -> physics.bbridge.bridge_schedule(betas).
    bridge = build_bridge(bridge_type=bridge_type, n_points=n_points, device=device,
                          tau=tau, shape=bridge_shape, beta_max=beta_max)
    interval = n_steps(bridge)
    ema = EMA(net.parameters(), decay=ema_decay)

    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, "net.ckpt")
    ema_path = os.path.join(save_dir, "ema.pt")

    # resume EMA alongside the model (main() already reloaded model/opt/sched via load_ckpt)
    if start_epoch > 0 and os.path.exists(ema_path):
        ema.load_state_dict(torch.load(ema_path, map_location=device))
        print(f"[i2sb] resumed EMA from {ema_path}")

    # sanity: conditioning channels must match CDLNet's input width (C = 1 + n_cond)
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

            # ----- sample a bridge point and regress -----
            b = x0.shape[0]
            step = torch.randint(0, interval, (b,), device=device)
            xt = q_sample(bridge, step, x0, x1, ot_ode=ot_ode)
            std_fwd = get_std_fwd(bridge, step, xdim=x0.shape[1:])   # (B,1,1,1) noise level

            opt.zero_grad()
            out = cdlnet_pred(net, xt, std_fwd, cond=cond, target_channels=target_channels)

            if parameterization == "x0":
                pred_x0 = out
                loss = masked_mse(pred_x0, x0, mask, use_mask)
            elif parameterization == "eps":
                label = compute_label(bridge, step, x0, xt)
                loss = masked_mse(out, label, mask, use_mask)
                pred_x0 = compute_pred_x0(bridge, step, xt, out)
            else:
                raise ValueError(f"Unknown parameterization {parameterization!r}")

            loss.backward()
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clip_grad)
            opt.step()
            # Important for our unrolled models
            if hasattr(net, "project"): net.project()
            ema.update()
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
                if os.path.exists(ema_path):
                    ema.load_state_dict(torch.load(ema_path, map_location=device))
                new_lr = np.array(get_lr(opt)) * backtrack_factor
                set_lr(opt, new_lr)
                print("Updated LR:", new_lr)
            else:
                raise RuntimeError(f"Backtrack at epoch {epoch} but no valid checkpoint "
                                   f"(best_loss={best_loss}).")
        elif save_ckpt_fn and avg_loss < best_loss:
            save_ckpt_fn(ckpt_path, model=net, optimizer=opt, scheduler=sched, step=global_step)
            torch.save(ema.state_dict(), ema_path)
            best_loss = avg_loss

        # ---- logging ----
        if wandb and not nonfinite:
            wandb.log({"train/loss": avg_loss, "train/lr": opt.param_groups[0]["lr"],
                       "train/epoch": epoch,
                       **{f"train/{k}": v for k, v in train_metrics.items()}}, step=global_step)
        elif not wandb:
            print({"epoch": epoch, "avg_loss": avg_loss, **train_metrics})

        # ---- validation: sample with EMA weights ----
        if val_loader is not None and val_every_epochs and (epoch + 1) % val_every_epochs == 0:
            val_loss = _validate(
                net, ema, bridge, val_loader, device, interval=interval, val_mode=val_mode, val_seed = val_seed,
                use_mask=use_mask, ot_ode=ot_ode, parameterization=parameterization,
                clip_denoise=clip_denoise, val_nfe=val_nfe, target_channels=target_channels,
                psnr_only=psnr_only, wandb=wandb, global_step=global_step,
            )
            if isinstance(sched, ReduceLROnPlateau) and val_loss is not None:
                sched.step(val_loss)

    pbar.close()
    return net


def _assert_cond_matches_model(net, loader, device, target_channels):
    """Peek one batch: CDLNet's input width self.C must equal target_channels + n_cond."""
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
            f"Conditioning/model mismatch: data provides {n_cond} cond channel(s) so CDLNet "
            f"needs C={expected_C}, but model.C={model_C}. Set model.params.C = 1 + len(cond_idx) "
            f"(or cond_idx=[] and C=1 to disable conditioning)."
        )


@torch.no_grad()
def _validate(net, ema, bridge, val_loader, device, *, interval, val_mode, val_seed, 
                use_mask, ot_ode, parameterization, clip_denoise, val_nfe, 
                target_channels, psnr_only, wandb, global_step):
    """Validate with EMA weights. Two modes:
      "single_pass" (default) -- draw one random step per batch, run ONE network forward, and
                                  score the single-pass pred_x0 (mirrors the training objective;
                                  cheap). Steps use a fixed seed so metrics are comparable epoch
                                  to epoch.
      "full_recon"            -- run the full val_nfe-step DDPM sampler (end-to-end recon quality).
    """
    net.eval()
    agg = {"loss": 0.0, "psnr": 0.0, "ssim": 0.0, "nrmse": 0.0}
    n_samples = 0
    last = None
    if val_seed is not None:
        gen = torch.Generator(device=device).manual_seed(val_seed)   # fixed val steps -> comparable
    else:
        gen = None
    with ema.average_parameters():
        for batch in val_loader:
            x0, x1, cond, mask = _split_batch(batch, device)
            bs = x0.shape[0]

            if val_mode == "full_recon":
                pred, _, _ = i2sb_sample(
                    net, x1, bridge, cond=cond, nfe=val_nfe, ot_ode=ot_ode,
                    parameterization=parameterization, clip_denoise=clip_denoise,
                    target_channels=target_channels, log_count=1, verbose=False,
                )
                loss = masked_mse(pred, x0, mask, use_mask)
                xt, step = x1, None
            else:  # single_pass: one random step, one forward (same as a training step)
                step = torch.randint(0, interval, (bs,), generator=gen, device=device)
                xt = q_sample(bridge, step, x0, x1, ot_ode=ot_ode)
                std_fwd = get_std_fwd(bridge, step, xdim=x0.shape[1:])
                out = cdlnet_pred(net, xt, std_fwd, cond=cond, target_channels=target_channels)
                if parameterization == "x0":
                    pred = out
                    loss = masked_mse(pred, x0, mask, use_mask)
                else:
                    label = compute_label(bridge, step, x0, xt)
                    loss = masked_mse(out, label, mask, use_mask)
                    pred = compute_pred_x0(bridge, step, xt, out)

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
