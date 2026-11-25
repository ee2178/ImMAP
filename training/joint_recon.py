"""
JOINT reconstruction of the target and an accelerated guide, task "joint_recon" (2026-11-18).

For nets that reconstruct BOTH contrasts (models/joint_lggs.py::JointLGGSNet): the forward returns
`(x_hat, {"guide_recon": g_hat})`, and both are supervised,

    loss = l(x_hat, x*) + guide_loss_weight * l(g_hat, g*)

with l the run's `loss_type` (the recon registry, e.g. magnitude-nl1-nl2), x* the stored target
image and g* the FULLY-SAMPLED guide coil-combined under the operator's maps -- the maps the net's
guide operator uses, so g_hat and g* share a coil combination. Supervising the guide keeps the
guide branch an honest reconstruction rather than a free latent.

The measurement is train_recon's exactly (training/recon.py): the target mask, measured k-space +
AWGN, online ESPIRiT maps, the image-domain embedding, and the guide undersampled per guide_R --
its line phase redrawn every step, `guide_val_offset` at validation (R_g / 2: complementary to
the target's lines).

Validation on a fixed subset: PSNR / SSIM of |x_hat| against the target AND of |g_hat| against
the full guide (val/guide_*), next to the zero-filled floors (val/zf_*, val/guide_zf_*). A step
whose loss or gradient is not finite is skipped (counted in train/skipped). Checkpoints at
validation only: net.ckpt (latest, resumable), net_bestval.ckpt (best TARGET val PSNR).
"""

import math
import os
import time

import numpy as np
import torch

from operators import FFT2D, Mask, Sense
from operators.fourier import ifftc
from training.common import embed_for_net, get_lr, prepare_measurement, save_ckpt
from training.forward_op import fixed_val_subset
from training.losses import LOSS_REGISTRY
from training.metrics import compute_metrics
from training.recon import _guide, _unpack, get_mask

BESTVAL_CKPT = "net_bestval.ckpt"


def _measure(batch, net, device, *, R, acs_lines, mask_dist, mask_offset, center_frac,
             adjust_accel, kspace_type, noise_std, noise_dist, online_smaps, online_smaps_kws,
             guide_R, guide_acs_lines, guide_mask_dist, guide_offset, generator=None):
    """One batch -> (y, E, T, sigma_n, side_guide, x_star, g_star, x_zf, g_zf)."""
    from physics.mask import resolve_acs_lines
    kspace, smaps, image, _organ, pad_hw, side = _unpack(batch)
    if side is None or isinstance(side, dict):
        raise ValueError("joint_recon needs a paired loader with ONE side slice (guide_slices 1)")
    kspace, smaps, image = (t.to(device, non_blocking=True) for t in (kspace, smaps, image))
    side = side.to(device, non_blocking=True)
    mask = get_mask(image, R=R, acs_lines=acs_lines, mode=mask_dist, offset=mask_offset,
                    center_frac=center_frac, adjust_accel=adjust_accel)
    y, sigma_n, extra = prepare_measurement(
        image=image, kspace=kspace, mask=mask, smaps=smaps, kspace_type=kspace_type,
        noise_std=noise_std, noise_dist=noise_dist, whiten_kspace=False, generator=generator,
        online_smaps=online_smaps,
        online_smaps_kws=dict(online_smaps_kws or {},
                              acs_lines=resolve_acs_lines(image.shape[-1], acs_lines,
                                                          center_frac)))
    maps = extra["smaps"]
    E0 = Mask(mask) @ FFT2D() @ Sense(maps)
    E, T = embed_for_net(net, E0, image, pad_hw)
    if guide_offset == "random" and float(guide_R) > 1:
        guide_offset = int(torch.randint(int(round(float(guide_R))), (1,), generator=generator,
                                         device=generator.device if generator else "cpu"))
    side_guide = _guide(net, side, maps, T, device, guide_R, guide_acs_lines, guide_mask_dist,
                        guide_offset)
    g_star = Sense(maps).adjoint(ifftc(side))                  # the full guide, operator maps
    g_zf = Sense(maps).adjoint(ifftc(side_guide)) if side_guide.dim() == 4 else None
    sigma = torch.as_tensor(sigma_n, device=device, dtype=torch.float32).reshape(-1, 1, 1, 1)
    return y, E, T, sigma, side_guide, image, g_star, E0.adjoint(y), g_zf


def _run(net, y, E, T, sigma, side_guide):
    out, extra = net(y, E=E, sigma=sigma, guide=side_guide)
    if not isinstance(extra, dict) or "guide_recon" not in extra:
        raise ValueError(f"{type(net).__name__} returns no 'guide_recon': joint_recon needs a "
                         f"net that reconstructs the guide too (e.g. JointLGGS)")
    return T.forward(out), T.forward(extra["guide_recon"])


def train_joint_recon(
    net, opt, sched, device, train_loader, val_loader, wandb=None, start_epoch=0,
    # ---- loop (cfg["training"]) ----
    num_epochs=6000,
    steps_per_epoch=50,
    val_every_epochs=100,
    val_slices=None,
    guide_loss_weight=1.0,           # lambda_g on the guide reconstruction's loss
    loss_type="magnitude-nl1-nl2",
    clip_grad=None,
    noise_std=(0.0, 0.001),
    noise_dist="uniform",
    val_noise_std=0.0,
    val_seed=0,
    metric_range=1.0,                # PSNR/SSIM peak fixed at 1 (normalized data); "gt_max" per slice
    use_organ_mask=False,            # accepted for the recon configs' keys; not used
    backtrack_thresh=None,           # (train.py's resume contract; no backtracking here)
    backtrack_factor=0.9,
    backtrack_count=0,
    best_loss=float("inf"),
    # ---- measurement (cfg["mri"]) ----
    R=8, acs_lines=None, mask_dist="uniform", mask_offset=0, kspace_type="measurement_awgn",
    whiten_kspace=False, center_frac=None, adjust_accel=False, online_smaps="espirit",
    online_smaps_kws=None, val_kspace_type=None, guide_R=1, guide_acs_lines=0,
    guide_mask_dist="uniform", guide_val_offset=0,
    # ---- paths (cfg["paths"]) ----
    save_dir=None, ckpt=None, save_ckpt_fn=save_ckpt,
):
    if whiten_kspace:
        raise NotImplementedError("joint_recon does not whiten")
    if use_organ_mask:
        raise NotImplementedError("joint_recon scores the full frame (use_organ_mask false)")
    if float(guide_loss_weight) < 0:
        raise ValueError(f"guide_loss_weight must be >= 0, got {guide_loss_weight}")
    loss_fn = LOSS_REGISTRY[loss_type]
    net.to(device).train()
    print(f"[joint_recon] {type(net).__name__} -- "
          f"{sum(p.numel() for p in net.parameters())} parameters, guide R {guide_R}, "
          f"lambda_g {float(guide_loss_weight):g}")
    meas = dict(R=R, acs_lines=acs_lines, mask_dist=mask_dist, mask_offset=mask_offset,
                center_frac=center_frac, adjust_accel=adjust_accel, noise_dist=noise_dist,
                online_smaps=online_smaps, online_smaps_kws=online_smaps_kws, guide_R=guide_R,
                guide_acs_lines=guide_acs_lines, guide_mask_dist=guide_mask_dist)
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, "net.ckpt")
    bestval_path = os.path.join(save_dir, BESTVAL_CKPT)
    best_val_psnr = -float("inf")
    if start_epoch > 0 and os.path.exists(bestval_path):
        v = torch.load(bestval_path, map_location="cpu", weights_only=False).get("best_val_loss")
        if v is not None and math.isfinite(v):
            best_val_psnr = -float(v)
    if val_loader is not None and val_slices:
        val_loader = fixed_val_subset(val_loader, int(val_slices), int(val_seed or 0))

    train_iter = iter(train_loader)
    for epoch in range(start_epoch, num_epochs):
        net.train()
        run = dict(loss=0.0, target=0.0, guide=0.0)
        n_ok, n_skip, t0 = 0, 0, time.time()
        for _ in range(steps_per_epoch):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            y, E, T, sigma, sg, x_star, g_star, _, _ = _measure(
                batch, net, device, kspace_type=kspace_type, noise_std=noise_std,
                guide_offset="random", **meas)
            x_hat, g_hat = _run(net, y, E, T, sigma, sg)
            l_t = loss_fn(x_star, x_hat, sigma)
            l_g = loss_fn(g_star, g_hat, sigma)
            loss = l_t + float(guide_loss_weight) * l_g
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(
                net.parameters(), clip_grad if clip_grad is not None else float("inf"))
            if not (torch.isfinite(loss) and torch.isfinite(gn)):
                n_skip += 1                     # a non-finite step never reaches the weights
                opt.zero_grad(set_to_none=True)
                continue
            opt.step()
            if hasattr(net, "project"):
                net.project()
            if sched is not None:
                sched.step()
            run["loss"] += float(loss.detach())
            run["target"] += float(l_t.detach())
            run["guide"] += float(l_g.detach())
            n_ok += 1
        global_step = (epoch + 1) * steps_per_epoch
        if n_ok == 0:
            raise RuntimeError(f"[epoch {epoch}] every step was non-finite")
        avg = {k: v / n_ok for k, v in run.items()}
        best_loss = min(best_loss, avg["loss"])
        if wandb:
            wandb.log({"train/loss": avg["loss"], "train/loss_target": avg["target"],
                       "train/loss_guide": avg["guide"], "train/skipped": n_skip,
                       "train/lr": get_lr(opt)[0], "train/epoch": epoch}, step=global_step)

        at_val = bool(val_every_epochs) and (epoch + 1) % val_every_epochs == 0
        val = None
        if val_loader is not None and at_val:
            t_val = time.time()
            val = _validate(net, val_loader, device,
                            dict(meas, kspace_type=val_kspace_type or kspace_type),
                            val_noise_std, val_seed, guide_val_offset, metric_range, wandb,
                            global_step, epoch)
            print(f"[epoch {epoch}] val target PSNR {val['psnr']:.2f} dB (zf {val['zf_psnr']:.2f})"
                  f", guide PSNR {val['guide_psnr']:.2f} dB (zf {val['guide_zf_psnr']:.2f}) "
                  f"-- {time.time() - t_val:.0f}s")
            if save_ckpt_fn and math.isfinite(val["psnr"]) and val["psnr"] > best_val_psnr:
                best_val_psnr = val["psnr"]
                save_ckpt_fn(bestval_path, model=net, step=global_step,
                             backtrack_count=backtrack_count, best_loss=best_loss,
                             best_val_loss=-best_val_psnr)
        if save_ckpt_fn and (at_val or epoch + 1 == num_epochs):
            save_ckpt_fn(ckpt_path, model=net, optimizer=opt, scheduler=sched, step=global_step,
                         backtrack_count=backtrack_count, best_loss=best_loss,
                         best_val_loss=None if val is None else -float(val["psnr"]))
    return net


@torch.no_grad()
def _validate(net, loader, device, meas, noise_std, seed, guide_offset, metric_range, wandb,
              global_step, epoch):
    net.eval()
    gen = torch.Generator(device=device).manual_seed(int(seed or 0))
    keys = ("psnr", "ssim", "zf_psnr", "zf_ssim", "guide_psnr", "guide_ssim", "guide_zf_psnr",
            "guide_zf_ssim")
    acc = {k: 0.0 for k in keys}
    n, panel = 0, None
    for batch in loader:
        y, E, T, sigma, sg, x_star, g_star, x_zf, g_zf = _measure(
            batch, net, device, noise_std=noise_std, generator=gen, guide_offset=guide_offset,
            **meas)
        x_hat, g_hat = _run(net, y, E, T, sigma, sg)
        b = x_star.shape[0]
        for pre, est, ref in (("", x_hat, x_star), ("zf_", x_zf, x_star),
                              ("guide_", g_hat, g_star), ("guide_zf_", g_zf, g_star)):
            if est is None:
                continue
            m = compute_metrics(ref.abs(), est.abs(), data_range=metric_range)
            acc[pre + "psnr"] += float(m["psnr"]) * b
            acc[pre + "ssim"] += float(m["ssim"]) * b
        n += b
        if panel is None:
            ims = [v[0, 0].abs().float().cpu().numpy() for v in
                   (x_zf, x_hat, x_star, g_zf if g_zf is not None else g_star, g_hat, g_star)]
            sx, sg_ = max(ims[2].max(), 1e-12), max(ims[5].max(), 1e-12)
            top = np.concatenate([ims[0] / sx, ims[1] / sx, ims[2] / sx], axis=1)
            bot = np.concatenate([ims[3] / sg_, ims[4] / sg_, ims[5] / sg_], axis=1)
            panel = np.concatenate([top, bot], axis=0)
    net.train()
    res = {k: v / max(n, 1) for k, v in acc.items()}
    if wandb:
        log = {f"val/{k}": v for k, v in res.items()}
        if panel is not None and hasattr(wandb, "Image"):
            log["val/recon_example"] = wandb.Image(
                np.clip(panel, 0, 1),
                caption=f"epoch {epoch} | top: target zero-filled | recon | ground truth; "
                        f"bottom: guide zero-filled | recon | fully sampled")
        wandb.log(log, step=global_step)
    else:
        print("[VAL joint] " + " ".join(f"{k}={v:.4f}" for k, v in res.items()))
    return res
