"""
Pretraining a guide CONTRAST TRANSFER net (models/contrast_transfer.py), task "contrast_transfer".

The net maps the guide, exactly as a guided recon net receives it, onto the fully-sampled target:

    batch -> training/i2sb_recon.py::_measure (the recon measurement: mask, noise, online maps)
    guide  = the side scan (PD) coil-combined under the OPERATOR's maps   (recon.py::_guide)
    target = the stored fully-sampled image (PDFS)          -- "mag" loss: |T(guide)| vs |target|
             or, for loss_on "complex", the target in the operator's phase convention (x0)
    loss   = L1 (or MSE) over the image

so the frozen net later sees in LGGS precisely the inputs it was trained on. The measurement
matters only through the maps the guide is combined with (online ESPIRiT on the undersampled
target): the transfer itself never sees y.

Validation on a fixed subset, as the recon loops: PSNR / SSIM of |T(guide)| against the target
(metric_range as everywhere), and of the RAW guide after the best per-slice scalar gain
(val/guide_psnr) -- the floor any transfer has to beat. Checkpoints at validation only: net.ckpt
(latest, resumable), net_bestval.ckpt (best val PSNR; what LGGS loads).
"""

import math
import os
import time

import numpy as np
import torch

from training.common import get_lr, save_ckpt
from training.forward_op import fixed_val_subset
from training.metrics import compute_metrics

LOSSES = ("l1", "mse")
LOSS_ON = ("mag", "complex")
BESTVAL_CKPT = "net_bestval.ckpt"


def _pair(batch, net, device, meas, noise_std, generator=None, guide_offset=0):
    from training.i2sb_recon import _measure
    x0, _, _, _, _, _, gt, guide = _measure(batch, net, device, noise_std=noise_std,
                                            generator=generator, guide_offset=guide_offset,
                                            x1_source="adjoint", **meas)
    if guide is None:
        raise ValueError("contrast_transfer needs a paired loader (the side scan is its input)")
    if guide.dim() == 5:
        if guide.shape[1] != 1:
            raise ValueError("contrast_transfer takes one guide slice (guide_slices = 1)")
        guide = guide[:, 0]
    return guide, gt, x0


def _loss(out, gt, x0, loss_type, loss_on):
    d = (out.abs() - gt.abs()) if loss_on == "mag" else (out - x0)
    return d.abs().mean() if loss_type == "l1" else d.abs().pow(2).mean()


def _scaled_guide(guide, gt):
    """|guide| times the least-squares scalar gain onto |gt|, per slice: the raw guide's best."""
    g, t = guide.abs().flatten(1), gt.abs().flatten(1)
    a = (g * t).sum(1) / g.pow(2).sum(1).clamp_min(1e-12)
    return guide.abs() * a.view(-1, 1, 1, 1)


def train_contrast_transfer(
    net, opt, sched, device, train_loader, val_loader, wandb=None, start_epoch=0,
    # ---- loop (cfg["training"]) ----
    num_epochs=200,
    steps_per_epoch=100,
    val_every_epochs=10,
    val_slices=None,
    clip_grad=None,
    loss_type="l1",
    loss_on="mag",
    noise_std=(0.0, 0.01),
    noise_dist="uniform",
    val_noise_std=0.0,
    val_seed=0,
    metric_range="gt_max",
    backtrack_count=0,                # carried for train.py's resume contract; no backtracking
    best_loss=float("inf"),
    # ---- measurement (cfg["mri"]) ----
    R=8, acs_lines=None, mask_dist="uniform", mask_offset=0, kspace_type="measurement_awgn",
    whiten_kspace=False, center_frac=None, adjust_accel=False, online_smaps="espirit",
    online_smaps_kws=None, val_kspace_type=None, guide_R=1, guide_acs_lines=0,
    guide_mask_dist="uniform",
    # ---- paths (cfg["paths"]) ----
    save_dir=None, ckpt=None, save_ckpt_fn=save_ckpt,
):
    if loss_type not in LOSSES:
        raise ValueError(f"loss_type {loss_type!r} not in {LOSSES}")
    if loss_on not in LOSS_ON:
        raise ValueError(f"loss_on {loss_on!r} not in {LOSS_ON}")
    if whiten_kspace:
        raise NotImplementedError("contrast_transfer does not whiten")
    net.to(device).train()
    n_par = sum(p.numel() for p in net.parameters())
    print(f"[contrast_transfer] {type(net).__name__}({net.extra_repr()}) -- {n_par} parameters")
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
        run_loss, t0 = 0.0, time.time()
        for _ in range(steps_per_epoch):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            guide, gt, x0 = _pair(batch, net, device, dict(meas, kspace_type=kspace_type),
                                  noise_std, guide_offset="random")
            out = net(guide)
            loss = _loss(out, gt, x0, loss_type, loss_on)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clip_grad)
            opt.step()
            if sched is not None:
                sched.step()
            run_loss += float(loss.detach())
        global_step = (epoch + 1) * steps_per_epoch
        avg = run_loss / steps_per_epoch
        if not math.isfinite(avg):
            raise RuntimeError(f"[epoch {epoch}] non-finite contrast-transfer loss")
        best_loss = min(best_loss, avg)
        if wandb:
            wandb.log({"train/loss": avg, "train/lr": get_lr(opt)[0], "train/epoch": epoch},
                      step=global_step)
        print(f"[epoch {epoch}] loss {avg:.4e} ({time.time() - t0:.0f}s)")

        at_val = bool(val_every_epochs) and (epoch + 1) % val_every_epochs == 0
        val = None
        if val_loader is not None and at_val:
            val = _validate(net, val_loader, device, dict(meas, kspace_type=val_kspace_type or
                                                          kspace_type),
                            val_noise_std, val_seed, metric_range, loss_type, loss_on, wandb,
                            global_step)
            print(f"[epoch {epoch}] val PSNR {val['psnr']:.2f} dB, SSIM {val['ssim']:.4f} "
                  f"(guide, best gain: {val['guide_psnr']:.2f} dB / {val['guide_ssim']:.4f})")
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
def _validate(net, loader, device, meas, noise_std, seed, metric_range, loss_type, loss_on,
              wandb, global_step):
    net.eval()
    gen = torch.Generator(device=device).manual_seed(int(seed or 0))
    acc = dict(loss=0.0, psnr=0.0, ssim=0.0, guide_psnr=0.0, guide_ssim=0.0)
    n, panel = 0, None
    for batch in loader:
        guide, gt, x0 = _pair(batch, net, device, meas, noise_std, generator=gen)
        out = net(guide)
        b = gt.shape[0]
        acc["loss"] += float(_loss(out, gt, x0, loss_type, loss_on)) * b
        m = compute_metrics(gt.abs(), out.abs(), data_range=metric_range)
        mg = compute_metrics(gt.abs(), _scaled_guide(guide, gt), data_range=metric_range)
        acc["psnr"] += float(m["psnr"]) * b
        acc["ssim"] += float(m["ssim"]) * b
        acc["guide_psnr"] += float(mg["psnr"]) * b
        acc["guide_ssim"] += float(mg["ssim"]) * b
        n += b
        if panel is None:
            g, o, t = (v[0, 0].abs().float().cpu().numpy() for v in (guide, out, gt))
            panel = np.concatenate([g / max(g.max(), 1e-12), o / max(t.max(), 1e-12),
                                    t / max(t.max(), 1e-12),
                                    np.abs(o - t) / max(t.max(), 1e-12) * 5], axis=1)
    net.train()
    res = {k: v / max(n, 1) for k, v in acc.items()}
    if wandb:
        log = {f"val/{k}": v for k, v in res.items()}
        if panel is not None and hasattr(wandb, "Image"):
            log["val/example"] = wandb.Image(np.flipud(np.clip(panel, 0, 1)),
                                             caption="guide | transfer | target | |error| x5")
        wandb.log(log, step=global_step)
    return res
