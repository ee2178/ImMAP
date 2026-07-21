"""
Latent I2SB training loop. A NEW regressor R predicts the T1ce latent z0, decoded once by the frozen
T1ce dictionary. Same skeleton as train_i2sb (epoch = steps_per_epoch gradient steps, averaged-loss
backtracking, checkpoint-on-improvement, wandb logging, single_pass / full_recon validation).

`bridge_domain` selects between two approaches (this is the only knob that changes the data flow;
D_t1ce is a plain frozen T1ce denoiser, C=1, used as encoder for z0 and as decoder in both):

  "latent"  -- True Latent I2SB. The bridge lives ENTIRELY in the M-channel latent, decode once.
      D_joint : frozen GroupCDL, C = n_cond   (a PLAIN denoiser on the conditioning contrasts, e.g.
                [FLAIR, T1, T2]; NO x_t channel -- we only ever encode it once, at the bridge start).
      R       : C = 2M (sees cat[z_t, z1]).
          z0 = D_t1ce(x0).z ;  z1 = D_joint(cond).z            # cond-only prior latent = bridge start
          z_t = forward_sample(z0, z1)                         # LATENT bridge + latent noise
          z0_hat = R(cat[z_t, z1] ; sigma) ; x0_hat = D_t1ce.D(z0_hat) + dc

  "image"   -- Latent Regression, Image-Domain Bridge. The bridge is in the T1ce IMAGE domain; the
      joint dict re-encodes the image bridge point each step so R stays unconditional (no channel
      expansion for conditioning).
      D_joint : frozen GroupCDL, C = 1 + n_cond   (trained on the IMAGE bridge stack [x_t, cond]).
      R       : C = M (sees z_t only).
          x_t = forward_sample(x0, x1)                         # IMAGE bridge
          z_t = D_joint(cat[x_t, cond] ; sigma).z              # 4-ch joint encode at sigma(t)
          z0_hat = R(z_t ; sigma) ; x0_hat = D_t1ce.D(z0_hat) + dc

Both frozen dicts map into the SAME latent (identical GroupCDL M and sc). The per-step forward pass
lives in _latent_forward; the full_recon samplers in sb/latent_i2sb.py. R always predicts z0 directly
(the only parameterization); the loss is latent, image, or a mix (see _combine_loss).
"""

import os
import math

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from tqdm import tqdm

from torch.optim.lr_scheduler import ReduceLROnPlateau

from training.common import (save_ckpt, load_ckpt, get_lr, set_lr, apply_loss_mask, load_model,
                             snr_loss_weight)
from training.losses import LOSS_REGISTRY
from training.metrics import compute_metrics
from sb.base import build_schedule, n_steps, forward_sample, forward_std
from sb.latent_i2sb import (encode, decode, latent_regress, _decode_dc,
                            latent_i2sb_sample, latent_i2sb_sample_imgdomain)
from visualization.filters import get_filter_grids


def _apply_loss(loss_fn, target, pred, mask, use_mask, sigma):
    """Mask (optionally) then evaluate a training.losses registry loss. The registry convention
    is loss_fn(target, pred, sigma) (see train_denoiser's loss_fn(gt, recon, sigma)); masked
    complex-mse == the old masked_mse since mask in {0,1}."""
    target, pred = apply_loss_mask(target, pred, mask, use_mask)
    return loss_fn(target, pred, sigma)


def _combine_loss(loss_fn, loss_mode, latent_weight, image_weight, loss_weight, latent_loss,
                  z0, z0_hat, x0, pred_x0, mask, use_mask, sigma):
    """Latent-I2SB training loss. R must predict the latent endpoint z0 (else the reverse posterior
    drifts on the decoder's null space -> collapse). Modes: 'latent' (loss on z0, UNMASKED over the
    full code), 'image' (registry loss on the decoded image, masked), 'mixed' (weighted sum).

    `latent_loss` picks the z0 metric:
      'l2' -- MSE, minimizes to the conditional MEAN -> a DENSE, small-magnitude code (the mean of a
              group-sparse signal averages away its spikes) -> wrong sparsity + a faded decode.
      'l1' -- minimizes to the conditional MEDIAN -> promotes sparsity (exact zeros) and, being
              robust to large values, preserves the big active codes -> matches z0's sparsity, less fade.
    `loss_weight` != 'uniform' reweights each sample by snr_loss_weight(sigma_t) (MSE-based, so the
    image term also drops to MSE there). Scales differ a lot (code std ~O(0.05) vs image std ~O(0.5)),
    so in 'mixed' weight the latent term up. Metrics/visualizations always stay image-domain."""
    if loss_mode not in ("latent", "image", "mixed"):
        raise ValueError(f"loss_mode {loss_mode!r} must be 'latent', 'image', or 'mixed'.")
    w = None if loss_weight == "uniform" else snr_loss_weight(sigma, loss_weight)   # (B,) or None

    total = z0_hat.new_zeros(())
    if loss_mode in ("latent", "mixed"):
        d = z0_hat - z0                                       # per-sample z0 loss (unmasked)
        lat = d.abs().flatten(1).mean(dim=1) if latent_loss == "l1" \
            else (d ** 2).flatten(1).mean(dim=1)              # (B,)
        total = total + latent_weight * (lat.mean() if w is None else (w * lat).mean())
    if loss_mode in ("image", "mixed"):
        if w is None:                                        # registry loss (keeps VGG etc. available)
            total = total + image_weight * _apply_loss(loss_fn, x0, pred_x0, mask, use_mask, sigma)
        else:                                                # per-sample weighted MSE
            tgt, prd = apply_loss_mask(x0, pred_x0, mask, use_mask)
            img = ((prd - tgt) ** 2).flatten(1).mean(dim=1)
            total = total + image_weight * (w * img).mean()
    return total


def _split_batch(batch, device):
    """Batch is (x0, x1, cond, mask). cond must be non-empty (the joint dictionary is
    conditioned on FLAIR/T1/T2), so unlike train_i2sb we keep it as a tensor."""
    x0, x1, cond, mask = batch
    x0 = x0.to(device, non_blocking=True)
    x1 = x1.to(device, non_blocking=True)
    mask = mask.to(device, non_blocking=True)
    cond = cond.to(device, non_blocking=True)
    return x0, x1, cond, mask


# ---------------------------------------------------------------------------
# frozen-dictionary loading
# ---------------------------------------------------------------------------
def _load_frozen_dict(config_path, device, backend=None):
    """Rebuild a trained GroupCDL dictionary from its SAVED config.json (which carries
    init=false and paths.ckpt), load its weights, freeze, eval, and compile flex if used.

    `backend` overrides the saved config's attn_backend. GroupCDL's weights are backend-agnostic
    (attn_backend only changes HOW the circulant attention is applied, not the parameters), so a
    dict trained on "flex" can be run on "gather" — the pure-eager path that needs NO torch.compile.
    Set backend="gather" to skip flex's first-iteration Triton compile (big win interactively);
    leave None to keep the trained backend (compiled flex is faster once warm)."""
    net = load_model(config_path, device=device)          # build_model + load ckpt + eval()
    for p in net.parameters():
        p.requires_grad_(False)
    net.eval()
    if backend is not None:
        net.attn_backend = backend
    if getattr(net, "attn_backend", None) == "flex":
        net.compile_flex()
    return net


def _latent_forward(bridge_domain, D_joint, R, D_t1ce, x0, x1, cond, bridge, step, deterministic,
                    M, decode_dc):
    """One latent-I2SB forward pass, shared by training and single_pass validation. Returns
    (z0, z0_hat, pred_x0, std) -- z0 is the target latent (for a latent loss), pred_x0 the decoded
    T1ce (for the image loss + metrics), std = forward_std(step), a (B,1,1,1) tensor that broadcasts
    to image OR latent tensors alike.

      bridge_domain="latent": z1 = encode(D_joint, cond) [cond-only dict]; z_t = forward_sample(z0, z1)
                              in the LATENT; z0_hat = R(cat[z_t, z1]) (conditional).
      bridge_domain="image" : x_t = forward_sample(x0, x1) in the IMAGE domain; z_t = encode(D_joint,
                              [x_t, cond]; std) [4-ch dict]; z0_hat = R(z_t) (unconditional -- the
                              conditioning is baked into z_t)."""
    z0 = encode(D_t1ce, x0)                                     # target latent (both modes)
    std = forward_std(bridge, step, xdim=x0.shape[1:])          # (B,1,1,1)
    if bridge_domain == "latent":
        z1 = encode(D_joint, cond)                             # prior latent (cond-only dict)
        z_t = forward_sample(bridge, step, z0, z1, deterministic=deterministic)
        z0_hat = latent_regress(R, z_t, z1, std, M)            # R conditions on z1
    else:  # "image"
        x_t = forward_sample(bridge, step, x0, x1, deterministic=deterministic)   # IMAGE bridge point
        z_t = encode(D_joint, torch.cat([x_t, cond], dim=1), sigma=std)           # 4-ch joint encode
        z0_hat = latent_regress(R, z_t, None, std, M)          # R unconditional
    pred_x0 = decode(D_t1ce, z0_hat, dc=_decode_dc(x1, decode_dc))
    return z0, z0_hat, pred_x0, std


# ---------------------------------------------------------------------------
# sanity: the shared-latent invariants the whole design rests on
# ---------------------------------------------------------------------------
def _assert_latent_shapes(bridge_domain, D_joint, R, D_t1ce, loader, target_channels):
    n_cond = 0
    try:
        n_cond = int(next(iter(loader))[2].shape[1])
    except Exception:
        pass

    if D_t1ce.C != target_channels:
        raise ValueError(f"T1ce dict C={D_t1ce.C} != target_channels {target_channels}.")
    if D_joint.M != D_t1ce.M or D_joint.sc != D_t1ce.sc:
        raise ValueError(
            f"Dictionaries do not share a latent shape: joint (M={D_joint.M}, sc={D_joint.sc}) "
            f"vs t1ce (M={D_t1ce.M}, sc={D_t1ce.sc}). Retrain them with identical M and sc.")
    M = D_joint.M
    if bridge_domain == "latent":
        # cond-only joint dict (NO x_t channel); the latent bridge conditions R on z1 = cat[z_t, z1]
        if D_joint.C != n_cond:
            raise ValueError(
                f"bridge_domain='latent' needs a CONDITIONING-ONLY joint dict: D_joint.C={D_joint.C} "
                f"must == n_cond={n_cond} (a plain denoiser on the {n_cond} conditioning contrasts, "
                f"NO x_t channel). Point joint_dict_config at that dict.")
        if R.C != 2 * M:
            raise ValueError(
                f"bridge_domain='latent': R.C={R.C} must == 2*M={2*M} (R sees cat[z_t, z1], M={M}).")
    else:  # "image"
        expected = target_channels + n_cond
        if D_joint.C != expected:
            raise ValueError(
                f"bridge_domain='image' needs a joint dict on the IMAGE bridge stack [x_t, cond]: "
                f"D_joint.C={D_joint.C} must == target_channels+n_cond={expected}.")
        if R.C != M:
            raise ValueError(
                f"bridge_domain='image': R.C={R.C} must == M={M} (R sees z_t only; the conditioning "
                f"is baked into z_t by the joint encode).")


# ---------------------------------------------------------------------------
# training loop
# ---------------------------------------------------------------------------
def train_latent_i2sb(
    R, opt, sched, device,
    train_loader,
    val_loader,
    wandb=None,
    start_epoch=0,
    # ---- frozen dictionaries (cfg["dicts"]) ----
    joint_dict_config=None,          # path to trained_nets/.../GroupCDL_Dict_Joint/config.json
    t1ce_dict_config=None,           # path to trained_nets/.../GroupCDL_Dict_T1ce/config.json
    dict_backend=None,               # None = keep trained backend; "gather" = compile-free eager
    # ---- generic loop (cfg["training"]) ----
    num_epochs=300,
    steps_per_epoch=200,
    val_every_epochs=10,
    clip_grad=1.0,
    backtrack_thresh=0.5,
    backtrack_factor=0.9,
    use_mask=True,
    loss_type="complex-mse",         # image-loss key in training.losses.LOSS_REGISTRY (image/mixed)
    loss_mode="latent",              # "latent" (MSE on z0) | "image" | "mixed"
    latent_weight=1.0,               # weight on the z0 loss term (latent / mixed)
    image_weight=1.0,                # weight on the decoded-image loss term (image / mixed)
    loss_weight="uniform",           # per-sample t-weighting: "uniform" | "snr" (~t=0) | "t1" (~t=1)
    latent_loss="l2",                # z0 metric: "l2" (dense mean) | "l1" (sparsity-promoting median)
    psnr_only=False,
    # ---- I2SB method (cfg["i2sb"]) ----
    kind="brownian",                 # schedule: "brownian" (tau-parameterized) or "i2sb" (paper)
    tau=0.19,                        # NOTE: latent-scale; re-tune (image-domain value is wrong)
    n_points=1000,
    beta_max=0.3,
    deterministic=False,             # drop the bridge / posterior noise (the OT-ODE limit)
    posterior="ddpm",                # reverse update: "ddpm" (moving average) | "interpolant"
    bridge_domain="latent",          # "latent" (bridge in the latent; cond-only joint dict; R cond., R.C=2M)
                                     # | "image" (bridge in the image; 4-ch [x_t,cond] joint dict; R uncond., R.C=M)
    val_mode="single_pass",
    val_seed=None,
    val_nfe=20,
    target_channels=1,
    decode_dc="x1_mean",             # "x1_mean" | "none": DC re-added at the frozen decode
    # ---- paths (cfg["paths"]) ----
    save_dir=None,
    ckpt=None,                       # signature parity; R resume handled in main()
    save_ckpt_fn=save_ckpt,
    **_unused,
):
    if joint_dict_config is None or t1ce_dict_config is None:
        raise ValueError("cfg['dicts'] must give joint_dict_config and t1ce_dict_config "
                         "(paths to each frozen dictionary's saved config.json).")
    if loss_type not in LOSS_REGISTRY:
        raise ValueError(f"loss_type {loss_type!r} not in LOSS_REGISTRY {sorted(LOSS_REGISTRY)}.")
    if loss_mode not in ("latent", "image", "mixed"):
        raise ValueError(f"loss_mode {loss_mode!r} must be 'latent', 'image', or 'mixed'.")
    if bridge_domain not in ("latent", "image"):
        raise ValueError(f"bridge_domain {bridge_domain!r} must be 'latent' or 'image'.")
    loss_fn = LOSS_REGISTRY[loss_type]

    R.to(device)
    R.train()

    # ---- frozen convolutional dictionaries ----
    D_joint = _load_frozen_dict(joint_dict_config, device, backend=dict_backend)
    D_t1ce = _load_frozen_dict(t1ce_dict_config, device, backend=dict_backend)
    M = D_joint.M
    print(f"[latent-i2sb] dicts: joint C={D_joint.C} M={D_joint.M} sc={D_joint.sc} | "
          f"t1ce C={D_t1ce.C} M={D_t1ce.M} sc={D_t1ce.sc} | R C={R.C} M={R.M} attn={R.attn_backend} "
          f"| dict_attn={D_joint.attn_backend} bridge_domain={bridge_domain} "
          f"loss={loss_mode}(img={loss_type}, λz={latent_weight}, λx={image_weight})")
    _assert_latent_shapes(bridge_domain, D_joint, R, D_t1ce, train_loader, target_channels)

    # Bridge schedule. tau/std_sb units depend on bridge_domain: "latent" -> LATENT-code units (re-tune
    # to the code magnitude; the image-domain value is wrong), "image" -> T1ce-image units (as train_i2sb).
    bridge = build_schedule(kind=kind, tau=tau, n_points=n_points, beta_max=beta_max, device=device)
    interval = n_steps(bridge)

    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, "net.ckpt")

    best_loss = float("inf")
    train_iter = iter(train_loader)
    total_steps = num_epochs * steps_per_epoch
    pbar = tqdm(total=total_steps, initial=start_epoch * steps_per_epoch,
                desc="LATENT-I2SB", dynamic_ncols=True)

    for epoch in range(start_epoch, num_epochs):
        R.train()
        running_loss, n_batches = 0.0, 0

        for _ in range(steps_per_epoch):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            x0, x1, cond, mask = _split_batch(batch, device)

            # ----- one latent-I2SB forward pass (per bridge_domain) -----
            b = x0.shape[0]
            step = torch.randint(0, interval, (b,), device=device)
            opt.zero_grad()
            z0, z0_hat, pred_x0, std_fwd = _latent_forward(
                bridge_domain, D_joint, R, D_t1ce, x0, x1, cond, bridge, step, deterministic, M, decode_dc)
            loss = _combine_loss(loss_fn, loss_mode, latent_weight, image_weight, loss_weight,
                                 latent_loss, z0, z0_hat, x0, pred_x0, mask, use_mask, std_fwd)

            loss.backward()
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(R.parameters(), clip_grad)
            opt.step()
            if hasattr(R, "project"): R.project()      # only R is trained; dicts stay frozen
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
                R, opt, sched, _ = load_ckpt(ckpt_path, model=R, optimizer=opt,
                                             scheduler=sched, device=device)
                new_lr = np.array(get_lr(opt)) * backtrack_factor
                set_lr(opt, new_lr)
                print("Updated LR:", new_lr)
            else:
                raise RuntimeError(f"Backtrack at epoch {epoch} but no valid checkpoint "
                                   f"(best_loss={best_loss}).")
        elif save_ckpt_fn and avg_loss < best_loss:
            save_ckpt_fn(ckpt_path, model=R, optimizer=opt, scheduler=sched, step=global_step)
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
                D_joint, R, D_t1ce, bridge, val_loader, device, M=M, interval=interval,
                loss_fn=loss_fn, val_mode=val_mode, val_seed=val_seed, use_mask=use_mask,
                deterministic=deterministic, posterior=posterior, bridge_domain=bridge_domain,
                decode_dc=decode_dc, val_nfe=val_nfe, psnr_only=psnr_only,
                loss_mode=loss_mode, latent_weight=latent_weight, image_weight=image_weight,
                loss_weight=loss_weight, latent_loss=latent_loss, wandb=wandb, global_step=global_step,
            )
            if isinstance(sched, ReduceLROnPlateau) and val_loss is not None:
                sched.step(val_loss)

    pbar.close()
    return R


@torch.no_grad()
def _validate(D_joint, R, D_t1ce, bridge, val_loader, device, *, M, interval, loss_fn,
              val_mode, val_seed, use_mask, deterministic, posterior, bridge_domain,
              decode_dc, val_nfe, psnr_only, loss_mode, latent_weight, image_weight, loss_weight,
              latent_loss, wandb, global_step):
    """Validate R. Two modes mirror train_i2sb:
      "single_pass" -- one random latent-bridge step, one latent regression + decode (matches
                       training; cheap). "full_recon" -- the val_nfe-step latent reverse bridge
                       with a single final decode (end-to-end synthesis quality)."""
    R.eval()
    agg = {"loss": 0.0, "psnr": 0.0, "ssim": 0.0, "nrmse": 0.0}
    n_samples = 0
    last = None
    gen = torch.Generator(device=device).manual_seed(val_seed) if val_seed is not None else None

    for batch in val_loader:
        x0, x1, cond, mask = _split_batch(batch, device)
        bs = x0.shape[0]

        if val_mode == "full_recon":
            sampler = latent_i2sb_sample if bridge_domain == "latent" else latent_i2sb_sample_imgdomain
            pred = sampler(D_joint, R, D_t1ce, x1, cond, bridge, M=M, nfe=val_nfe,
                           deterministic=deterministic, posterior=posterior,
                           decode_dc=decode_dc, verbose=False)
            sigma = torch.zeros(bs, 1, 1, 1, device=device)     # end-to-end: no single step
            # full_recon has no single z0_hat -> report the image-domain recon error regardless
            # of loss_mode (this is the end-to-end quality the LR scheduler should track).
            loss = _apply_loss(loss_fn, x0, pred, mask, use_mask, sigma)
            xt, step = x1, None
        else:  # single_pass
            step = torch.randint(0, interval, (bs,), generator=gen, device=device)
            z0, z0_hat, pred, std_fwd = _latent_forward(
                bridge_domain, D_joint, R, D_t1ce, x0, x1, cond, bridge, step, deterministic, M, decode_dc)
            loss = _combine_loss(loss_fn, loss_mode, latent_weight, image_weight, loss_weight,
                                 latent_loss, z0, z0_hat, x0, pred, mask, use_mask, std_fwd)
            xt = x1  # for display: latent zt is not an image; show the prior instead

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
            cols = [x1[:1], pred_m[:1], x0_m[:1]]
            cap = f"T1 prior | single-pass pred_x0 (step={int(step[0])}) | T1ce GT"
        else:
            cols = [x1[:1], pred_m[:1], x0_m[:1]]
            cap = f"T1 prior | latent-I2SB recon (nfe={val_nfe}) | T1ce GT"
        ref = torch.cat([cols[0], cols[-1]], dim=0)
        lo = float(ref.amin()); hi = max(float(ref.amax()), lo + 1e-8)
        grid = mask[:1] * torch.cat([((c - lo) / (hi - lo)).clamp(0, 1) for c in cols], dim=0)
        res = (x0_m[:1] - pred_m[:1]).abs(); res = res / res.max().clamp(min=1e-8)
        wandb.log({
            "val/example": wandb.Image(vutils.make_grid(grid, nrow=len(cols)), caption=cap),
            "val/residual": wandb.Image(vutils.make_grid(res, nrow=1), caption="| GT - pred |"),
            **{f"val/{k}": v for k, v in mean_metrics.items()},
        }, step=global_step)
        # learned latent dictionary R (real-valued GroupCDL); no-op if R has no filter banks
        try:
            wandb.log(get_filter_grids(R), step=global_step)
        except (AttributeError, NotImplementedError, AssertionError):
            pass
    elif not wandb:
        print(f"[VAL {val_mode}] " + " ".join(f"{k}={v:.4f}" for k, v in mean_metrics.items()))

    R.train()
    return mean_metrics["loss"]
