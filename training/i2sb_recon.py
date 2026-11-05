"""
I2SB training loop for accelerated MRI reconstruction (task "i2sb_recon").

The fastMRI recon loader's batch (kspace, smaps, image, mask, pad_hw) is turned into a bridge pair
ON THE DEVICE, exactly as training/recon.py builds its measurement -- the undersampling mask and
the noise are drawn per step, so x1 cannot come from the loader:

    y, sigma_n   <- prepare_measurement(image, kspace, mask, smaps)     (recon.py's measurement)
    E            =  Mask @ FFT2D @ Sense(smaps)
    x0           =  the fully-sampled target, in E's phase convention (`x0_source`, below)
    x1           =  E^H y                       the bridge start (sb.i2sb_recon.bridge_start)
    step ~ U{0..n-1},  x_t = forward_sample(step, x0, x1)
    x0_hat       =  net(x_t, sigma=std_fwd, x1=x1, y=y, E=E, sigma_n=sigma_n)
    loss         =  loss_fn(x0, x0_hat)         (optionally t-weighted, see i2sb._bridge_loss)

Everything after the measurement is train_i2sb's skeleton: epochs of steps_per_epoch optimizer
steps, gradient accumulation, EMA (validation and net_ema.ckpt use the average), averaged-loss
backtracking and checkpoint-on-improvement of net.ckpt. On top of it, the best validation PSNR is
kept as net_bestval.ckpt, since the train loss of a bridge regressor (a single random step) is a
poor proxy for the quality of the sampled reconstruction.

The measurement settings (cfg["mri"], and noise_std / val_noise_std / val_seed) mean the same as
in train_recon, so an i2sb_recon run and a one-shot recon run on the same config blocks see the
same masks, noise and organ mask, and their numbers share a table.

THE TARGET MUST SHARE THE OPERATOR'S PHASE CONVENTION (`x0_source`, cfg["i2sb"]). Under the
measured protocol (kspace_type "measurement_awgn" + online_smaps), E is built from maps estimated
from THIS measurement's ACS, while the loader's stored `image` is the combination under the
OFFLINE maps. ESPIRiT fixes each map set's phase only up to its own per-pixel reference, and the
two disagree on some slices: on a local brain volume, |image| matched the online-map combination
exactly but its phase was off by -1.8 rad on average (std ~1 rad, spatially varying) on 3 of 8
slices. A magnitude-loss one-shot net never sees that; a COMPLEX bridge would be asked to
transport x1 into an arbitrary phase convention it cannot infer from y. So:

    "operator"  (default) x0 = S_online^H F^-1 kspace when the operator's maps are online
                estimates, i.e. the fully-sampled image E's own maps produce; otherwise
                (simulated k-space, where the measurement IS built from `image`) x0 = image
    "stored"    x0 = image always (reproduces the mismatch; for comparison only)

METRICS are always taken on the MAGNITUDE against the stored `image` -- the ground truth the
one-shot grid is scored on -- which "operator" leaves unchanged wherever the offline maps have
support (their |.| agree there; outside it the stored image is exactly 0).

Validation (`val_mode`, cfg["i2sb"]):
    "single_pass"   one random bridge step per slice, one forward -- the training objective
    "full_recon"    the full `val_nfe`-step reverse sampler, i.e. the reconstruction itself
Both log val/zf_* (the zero-filled x1 scored the same way) as the floor to beat.
"""

import contextlib
import math
import os
import time

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from operators import Mask, FFT2D, Sense
from operators.fourier import ifftc
from physics.mask import get_mask_cached as get_mask, resolve_acs_lines
from sb.base import build_schedule, n_steps, forward_sample, forward_std
from sb.i2sb_recon import X1_SOURCES, bridge_start, predict_x0_recon, i2sb_recon_sample
from training.common import (EMA, POSTFIX_EVERY, apply_loss_mask, embed_for_net, get_lr,
                             load_ckpt, prepare_measurement, save_ckpt, save_ema_ckpt)
from training.common import backtrack as do_backtrack, resync_schedule
from training.i2sb import _bridge_loss
from training.losses import LOSS_REGISTRY
from training.metrics import compute_metrics
from training.recon import _check_paired, _ckpt_field, _guide, _unpack, guide_display
from visualization.image import recon_panel
from visualization.params import get_param_logs
from visualization.wandb_image import wandb_image


BESTVAL_CKPT = "net_bestval.ckpt"
X0_SOURCES = ("operator", "stored")


def _measure(batch, net, device, *, R, acs_lines, mask_dist, mask_offset, center_frac,
             adjust_accel, kspace_type, noise_std, noise_dist, online_smaps, online_smaps_kws,
             x1_source, x0_source="operator", guide_R=1, guide_acs_lines=0,
             guide_mask_dist="uniform", guide_offset=0, generator=None):
    """One recon batch -> (x0, x1, y, E, sigma_n, organ_mask, gt, guide), all on `device`.

    x0 is the bridge target (see `x0_source` in the module docstring); gt is the stored image,
    the ground truth metrics are scored against. guide is what the net takes from a paired
    loader's side k-space (training/recon.py::_guide: undersampled per guide_R, routed by the
    net's guide_domain), or None."""
    kspace, smaps, image, organ_mask, pad_hw, side = _unpack(batch)
    if side is not None:
        _check_paired(net, kspace_type, online_smaps)
    kspace = kspace.to(device, non_blocking=True)
    smaps = smaps.to(device, non_blocking=True)
    image = image.to(device, non_blocking=True)
    organ_mask = organ_mask.to(device, non_blocking=True)

    mask = get_mask(image, R=R, acs_lines=acs_lines, mode=mask_dist, offset=mask_offset,
                    center_frac=center_frac, adjust_accel=adjust_accel)
    y, sigma_n, extra = prepare_measurement(
        image=image, kspace=kspace, mask=mask, smaps=smaps,
        kspace_type=kspace_type, noise_std=noise_std, noise_dist=noise_dist,
        whiten_kspace=False, generator=generator,
        online_smaps=online_smaps,
        online_smaps_kws=dict(online_smaps_kws or {},
                              acs_lines=resolve_acs_lines(image.shape[-1], acs_lines,
                                                          center_frac)),
        need_smaps=getattr(net, "uses_operator_smaps", True),
    )
    E = Mask(mask) @ FFT2D() @ Sense(extra["smaps"])
    # The bridge lives on the measured grid. A net that needs a larger one (pad_stride > 1)
    # would have to embed x_t and x1 too; refuse it here rather than mix grids.
    E, T = embed_for_net(net, E, image, pad_hw)
    if not T.is_identity:
        raise NotImplementedError(
            f"{type(net).__name__} needs the image-domain embedding (pad_stride="
            f"{getattr(net, 'pad_stride', 1)}); bridge regressors run on the measured grid. "
            f"Set data.*.pad_multiple to 1.")
    x1 = bridge_start(y, E, x1_source)
    x0 = image
    if x0_source == "operator" and online_smaps and kspace_type != "simulated":
        x0 = Sense(extra["smaps"]).adjoint(ifftc(kspace))
    guide = _guide(net, side, extra["smaps"], T, device, guide_R, guide_acs_lines,
                   guide_mask_dist, guide_offset)
    return x0, x1, y, E, sigma_n, organ_mask, image, guide


def train_i2sb_recon(
    net, opt, sched, device,
    train_loader,
    val_loader,
    wandb=None,
    start_epoch=0,
    # ---- generic loop (cfg["training"]) ----
    num_epochs=1000,
    steps_per_epoch=200,
    val_every_epochs=10,
    clip_grad=1.0,
    backtrack_thresh=None,           # None = backtrack on a non-finite loss only
    backtrack_factor=0.9,
    backtrack_count=0,
    best_loss=float("inf"),
    loss_type="complex-mse",         # complex, not magnitude: the bridge state carries phase
    loss_weight="uniform",           # per-sample t-weighting: "uniform" | "snr" | "t1"
    loss_weight_max=None,
    use_organ_mask=False,            # mask the loss AND the metrics/panel, as train_recon does
    accum_steps=1,
    ema_decay=None,
    ema_warmup=True,
    noise_std=(0.0, 0.01),           # measurement noise, as train_recon
    noise_dist="uniform",
    val_noise_std=None,
    val_seed=None,                   # fixes the val noise realization AND the single_pass steps
    val_slices=None,                 # fixed random subset of the val split; None = all of it
    residual_gain=4.0,
    # ---- measurement (cfg["mri"]) ----
    R=8,
    acs_lines=24,
    kspace_type="simulated",
    whiten_kspace=False,
    online_smaps=None,
    online_smaps_kws=None,
    mask_dist="uniform",
    mask_offset=0,
    center_frac=None,
    adjust_accel=False,
    val_kspace_type=None,
    guide_R=1,                       # paired data only: side-scan acceleration, 1 = fully
    guide_acs_lines=0,               # sampled; see training/recon.py::undersample_guide
    guide_mask_dist="uniform",
    # ---- I2SB method (cfg["i2sb"]) ----
    kind="brownian",
    tau=0.1,
    n_points=1000,
    beta_max=0.3,
    deterministic=False,
    posterior="ddpm",
    x1_source="adjoint",
    x0_source="operator",            # bridge target in E's phase convention; see module doc
    val_mode="full_recon",
    val_nfe=20,
    # ---- paths (cfg["paths"]) ----
    save_dir=None,
    ckpt=None,
    save_ckpt_fn=save_ckpt,
):
    net.to(device)
    net.train()

    if loss_type not in LOSS_REGISTRY:
        raise ValueError(f"loss_type {loss_type!r} not in LOSS_REGISTRY {sorted(LOSS_REGISTRY)}")
    loss_fn = LOSS_REGISTRY[loss_type]
    if whiten_kspace:
        raise NotImplementedError("i2sb_recon does not whiten: the bridge pair would have to "
                                  "be built in the whitened domain")
    if x0_source not in X0_SOURCES:
        raise ValueError(f"x0_source {x0_source!r} not in {X0_SOURCES}")
    if x1_source not in X1_SOURCES:
        raise ValueError(f"x1_source {x1_source!r} not in {X1_SOURCES}")
    if val_mode not in ("single_pass", "full_recon"):
        raise ValueError(f"val_mode must be 'single_pass' or 'full_recon', got {val_mode!r}")
    if accum_steps < 1:
        raise ValueError(f"accum_steps must be >= 1, got {accum_steps}")

    bridge = build_schedule(kind=kind, tau=tau, n_points=n_points, beta_max=beta_max,
                            device=device)
    interval = n_steps(bridge)
    if hasattr(net, "assert_schedule_matches"):
        net.assert_schedule_matches(bridge)

    meas = dict(R=R, acs_lines=acs_lines, mask_dist=mask_dist, mask_offset=mask_offset,
                center_frac=center_frac, adjust_accel=adjust_accel, noise_dist=noise_dist,
                online_smaps=online_smaps, online_smaps_kws=online_smaps_kws,
                x1_source=x1_source, x0_source=x0_source, guide_R=guide_R,
                guide_acs_lines=guide_acs_lines, guide_mask_dist=guide_mask_dist)

    ema = None
    if ema_decay:
        ema = EMA(net.parameters(), decay=float(ema_decay), use_num_updates=bool(ema_warmup))
        print(f"[i2sb_recon] EMA decay={float(ema_decay)} warmup={bool(ema_warmup)}")
    if accum_steps > 1:
        print(f"[i2sb_recon] gradient accumulation x{accum_steps}")

    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, "net.ckpt")
    ema_ckpt_path = os.path.join(save_dir, "net_ema.ckpt")
    bestval_path = os.path.join(save_dir, BESTVAL_CKPT)

    best_val_psnr = -float("inf")
    if start_epoch > 0:
        resume_from = ckpt if ckpt else ckpt_path
        if ema is not None and os.path.exists(resume_from):
            load_ckpt(resume_from, ema=ema, device=device)
            print(f"[i2sb_recon] resumed EMA from {resume_from} at {ema.num_updates} updates")
        if os.path.exists(bestval_path):
            v = _ckpt_field(bestval_path, "best_val_loss")
            if v is not None and math.isfinite(v):
                best_val_psnr = -float(v)            # stored negated: lower is better on disk

    if val_loader is not None and val_slices:
        from training.forward_op import fixed_val_subset
        n_full = len(val_loader.dataset)
        val_loader = fixed_val_subset(val_loader, int(val_slices),
                                      0 if val_seed is None else int(val_seed))
        print(f"[i2sb_recon] validating on a fixed subset: {len(val_loader.dataset)}/{n_full}")

    if backtrack_count:
        print(f"resuming at backtrack_count={backtrack_count}: LR -> "
              f"{resync_schedule(opt, sched, backtrack_count, backtrack_factor)}")

    train_iter = iter(train_loader)
    pbar = tqdm(total=num_epochs * steps_per_epoch, initial=start_epoch * steps_per_epoch,
                desc="I2SB-recon", dynamic_ncols=True)

    for epoch in range(start_epoch, num_epochs):
        net.train()
        running_loss, n_batches = 0.0, 0

        for _ in range(steps_per_epoch):
            opt.zero_grad(set_to_none=True)
            step_loss = 0.0
            for _ in range(accum_steps):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)
                x0, x1, y, E, sigma_n, organ_mask, gt, guide = _measure(
                    batch, net, device, kspace_type=kspace_type, noise_std=noise_std,
                    guide_offset="random", **meas)

                b = x0.shape[0]
                step = torch.randint(0, interval, (b,), device=device)
                x_t = forward_sample(bridge, step, x0, x1, deterministic=deterministic)
                std_fwd = forward_std(bridge, step, xdim=x0.shape[1:])
                pred = predict_x0_recon(net, x_t, std_fwd, x1=x1, y=y, E=E, sigma_n=sigma_n,
                                        guide=guide)
                loss = _bridge_loss(loss_fn, loss_type, x0, pred, organ_mask, use_organ_mask,
                                    None, 1.0, std_fwd, loss_weight,
                                    loss_weight_max=loss_weight_max)
                (loss / accum_steps).backward()
                step_loss = step_loss + loss.detach() / accum_steps

            if clip_grad is not None:
                nn.utils.clip_grad_norm_(net.parameters(), clip_grad)
            opt.step()
            if ema is not None:
                ema.update()
            if hasattr(net, "project"):
                net.project()
            if sched is not None and not isinstance(sched, ReduceLROnPlateau):
                sched.step()

            running_loss += step_loss
            n_batches += 1
            pbar.update(1)
            if n_batches % POSTFIX_EVERY == 0:
                pbar.set_postfix(loss=f"{loss.item():.3e}", epoch=epoch)

        global_step = (epoch + 1) * steps_per_epoch
        avg_loss = float(running_loss) / max(n_batches, 1)
        nonfinite = not math.isfinite(avg_loss)

        with torch.no_grad():
            x0_m, pred_m = apply_loss_mask(gt, pred, organ_mask, use_organ_mask)
            train_metrics = {k: float(v) for k, v in compute_metrics(
                x0_m.abs(), pred_m.abs(),
                mask=organ_mask if use_organ_mask else None).items()}

        over = backtrack_thresh is not None and avg_loss > best_loss + backtrack_thresh
        if nonfinite or over:
            print(f"[epoch {epoch}] "
                  f"{'non-finite loss' if nonfinite else f'avg loss {avg_loss:.3e} > best {best_loss:.3e} + {backtrack_thresh}'}"
                  f" -- backtracking")
            if os.path.exists(ckpt_path) and math.isfinite(best_loss):
                backtrack_count, new_lr = do_backtrack(
                    ckpt_path, model=net, optimizer=opt, scheduler=sched, device=device,
                    backtrack_count=backtrack_count, backtrack_factor=backtrack_factor)
                print(f"backtrack #{backtrack_count} -> LR {new_lr}")
            else:
                raise RuntimeError(f"Backtrack at epoch {epoch} but no valid checkpoint "
                                   f"(best_loss={best_loss}).")
        elif save_ckpt_fn and avg_loss < best_loss:
            save_ckpt_fn(ckpt_path, model=net, optimizer=opt, scheduler=sched,
                         step=global_step, backtrack_count=backtrack_count,
                         best_loss=avg_loss, ema=ema)
            if ema is not None:
                save_ema_ckpt(ema_ckpt_path, ema, net, step=global_step, best_loss=avg_loss)
            best_loss = avg_loss

        if wandb and not nonfinite:
            wandb.log({"train/loss": avg_loss, "train/lr": get_lr(opt)[0], "train/epoch": epoch,
                       **{f"train/{k}": v for k, v in train_metrics.items()}}, step=global_step)
        elif not wandb:
            print({"epoch": epoch, "avg_loss": avg_loss, **train_metrics})

        if val_loader is not None and val_every_epochs and (epoch + 1) % val_every_epochs == 0:
            t_val = time.time()
            val_ctx = ema.average_parameters() if ema is not None else contextlib.nullcontext()
            with val_ctx:
                val = _validate(
                    net, bridge, val_loader, device, interval=interval, val_mode=val_mode,
                    val_nfe=val_nfe, val_seed=val_seed, deterministic=deterministic,
                    posterior=posterior, loss_fn=loss_fn, loss_type=loss_type,
                    use_organ_mask=use_organ_mask, residual_gain=residual_gain,
                    kspace_type=val_kspace_type or kspace_type,
                    noise_std=noise_std if val_noise_std is None else val_noise_std,
                    meas=meas, wandb=wandb, global_step=global_step, epoch=epoch)
                if save_ckpt_fn and math.isfinite(val["psnr"]) and val["psnr"] > best_val_psnr:
                    best_val_psnr = val["psnr"]
                    # under the EMA context, so this is the averaged net -- the one validated
                    save_ckpt_fn(bestval_path, model=net, step=global_step,
                                 backtrack_count=backtrack_count, best_loss=best_loss,
                                 best_val_loss=-best_val_psnr)
            print(f"[epoch {epoch}] validation ({val_mode}) took {time.time() - t_val:.0f}s: "
                  f"PSNR {val['psnr']:.2f} dB (zero-filled {val['zf_psnr']:.2f})")
            if isinstance(sched, ReduceLROnPlateau):
                sched.step(val["loss"])

    pbar.close()
    return net


@torch.no_grad()
def _validate(net, bridge, val_loader, device, *, val_seed, **kws):
    """-> mean {"loss", "psnr", "ssim", "nrmse", "zf_psnr", "zf_ssim", "zf_nrmse"} over the split.

    With `val_seed`, the bridge / sampler noise is seeded too (forward_sample and reverse_sample
    draw from the global RNG), inside fork_rng so the training stream is left where it was."""
    if val_seed is None:
        return _validate_pass(net, bridge, val_loader, device, gen=None, **kws)
    devices = [device] if torch.device(device).type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(val_seed))
        gen = torch.Generator(device=device).manual_seed(int(val_seed))
        return _validate_pass(net, bridge, val_loader, device, gen=gen, **kws)


def _validate_pass(net, bridge, val_loader, device, *, gen, interval, val_mode, val_nfe,
                   deterministic, posterior, loss_fn, loss_type, use_organ_mask, residual_gain,
                   kspace_type, noise_std, meas, wandb, global_step, epoch):
    net.eval()
    keys = ("loss", "psnr", "ssim", "nrmse", "zf_psnr", "zf_ssim", "zf_nrmse")
    agg = {k: 0.0 for k in keys}
    n = 0
    last = None

    for batch in val_loader:
        x0, x1, y, E, sigma_n, organ_mask, gt, guide = _measure(
            batch, net, device, kspace_type=kspace_type, noise_std=noise_std, generator=gen,
            guide_offset=0, **meas)
        bs = x0.shape[0]
        if val_mode == "full_recon":
            pred, _, _ = i2sb_recon_sample(net, y, E, bridge, sigma_n=sigma_n, x1=x1,
                                           nfe=val_nfe, deterministic=deterministic,
                                           posterior=posterior, verbose=False, guide=guide)
            std_fwd = torch.zeros(bs, 1, 1, 1, device=device)
        else:
            step = torch.randint(0, interval, (bs,), generator=gen, device=device)
            x_t = forward_sample(bridge, step, x0, x1, deterministic=deterministic)
            std_fwd = forward_std(bridge, step, xdim=x0.shape[1:])
            pred = predict_x0_recon(net, x_t, std_fwd, x1=x1, y=y, E=E, sigma_n=sigma_n,
                                    guide=guide)
        # unweighted in both modes, so val/loss is comparable across loss_weight settings
        loss = _bridge_loss(loss_fn, loss_type, x0, pred, organ_mask, use_organ_mask,
                            None, 1.0, std_fwd, "uniform")

        mmask = organ_mask if use_organ_mask else None
        m = compute_metrics(gt.abs(), pred.abs(), mask=mmask)
        mz = compute_metrics(gt.abs(), x1.abs(), mask=mmask)
        agg["loss"] += float(loss) * bs
        for k in ("psnr", "ssim", "nrmse"):
            agg[k] += float(m[k]) * bs
            agg[f"zf_{k}"] += float(mz[k]) * bs
        n += bs
        last = (gt, pred, x1, mmask, sigma_n, guide, _unpack(batch)[5])

    mean = {k: v / max(n, 1) for k, v in agg.items()}

    if wandb and last is not None:
        x0, pred, x1, mmask, sigma_n, guide, side = last
        img, rec, zf = (x0, pred, x1) if mmask is None else (x0 * mmask, pred * mmask, x1 * mmask)
        gtiles = guide_display(net, guide, side, meas.get("guide_R", 1))
        gimgs = [(g if mmask is None else g * mmask)[:1] for _, g in gtiles]
        panel, pstats = recon_panel(img[:1], rec[:1], zero_filled=zf[:1], gain=residual_gain,
                                    guides=gimgs)
        how = f"I2SB recon, nfe={val_nfe}" if val_mode == "full_recon" else "single-pass x0_hat"
        caption = (f"epoch {epoch} | zero-filled x1 | {how} | ground truth | "
                   f"|residual| x{pstats['gain']:g}"
                   + "".join(f" | {lab} (own scale)" for lab, _ in gtiles)
                   + f"  (sigma_n={float(sigma_n.reshape(-1)[0]):.4f})")
        wandb.log({"val/recon_example": wandb_image(panel, caption=caption),
                   **{f"val/{k}": v for k, v in mean.items()}}, step=global_step)
        try:
            wandb.log(get_param_logs(net), step=global_step)
        except (AttributeError, NotImplementedError, AssertionError):
            pass
    elif not wandb:
        print(f"[VAL {val_mode}] " + " ".join(f"{k}={v:.4f}" for k, v in mean.items()))

    net.train()
    return mean
