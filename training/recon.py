import os
import math
import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm
from training.losses import LOSS_REGISTRY
from training.metrics import compute_metrics
from training.common import (
    save_ckpt,
    load_ckpt,
    get_lr,
    set_lr,
    apply_loss_mask,
    prepare_measurement,
)
from visualization.filters import get_filter_grids
from visualization.image import recon_panel, residual_kspace
from physics.mask import get_mask_cached as get_mask
from operators import Mask, FFT2D, Sense


def train_recon(
    net,
    opt,
    sched,
    train_loader,
    val_loader,
    device,
    ### MRI Arguments
    R=8,
    acs_lines=24,
    kspace_type="measurement",
    whiten_kspace=False,
    mask_dist="uniform",
    mask_offset=0,
    noise_std=(0.0, 0.01),
    noise_dist="uniform",
    # Validation noise. `val_noise_std=None` redraws from `noise_std` like
    # training does; a number pins it (use the midpoint of the training range,
    # as Sljiva's closure does -- `genobs(clo, Val{true}(), s)` evaluates at
    # `mean(clo.noise_level)`). `val_seed` fixes the realization, so val curves
    # move because the model moved, not because the noise did.
    val_noise_std=None,
    val_seed=None,
    ### Loss
    loss_type="complex-mse",
    use_organ_mask=False,
    ### Fit (epoch-based)
    num_epochs=1000,          # total epochs (was: max_steps)
    steps_per_epoch=100,      # an "epoch" == this many gradient steps
    clip_grad=1.0,
    ### Logging
    val_every_epochs=10,      # validate every N epochs (was: val_every steps)
    # Fixed amplification for the residual panel. Fixed, NOT auto-scaled: an
    # auto-scaled residual always fills the dynamic range, so an artifact that
    # is shrinking over training looks identical to one that is not. Raise it
    # to inspect a residual that has become faint.
    residual_gain=4.0,
    start_epoch=0,            # was: start_step
    save_dir=None,
    ckpt=None,
    save_ckpt_fn=save_ckpt,
    # NOTE: backtrack_thresh is now an averaged-LOSS margin (lower is better),
    # not a PSNR margin. None disables the threshold check (but a non-finite
    # loss will still trigger a protective restore). Tune to your loss scale.
    backtrack_thresh=None,
    backtrack_factor=0.8,
    wandb=None,
):
    net.to(device)
    net.train()

    loss_fn = LOSS_REGISTRY[loss_type]
    ckpt_path = os.path.join(save_dir, "net.ckpt")

    best_loss = float("inf")        # lower is better -> start high
    backtrack_enabled = backtrack_thresh is not None

    train_iter = iter(train_loader)
    total_steps = num_epochs * steps_per_epoch
    pbar = tqdm(
        total=total_steps,
        initial=start_epoch * steps_per_epoch,
        desc="TRAIN",
        dynamic_ncols=True,
    )

    for epoch in range(start_epoch, num_epochs):
        net.train()
        running_loss = 0.0
        n_batches = 0

        # ==================================================================
        # ONE EPOCH == steps_per_epoch GRADIENT STEPS
        # ==================================================================
        for _ in range(steps_per_epoch):

            try:
                kspace, smaps, image, organ_mask = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                kspace, smaps, image, organ_mask = next(train_iter)

            kspace = kspace.to(device, non_blocking=True)
            smaps = smaps.to(device, non_blocking=True)
            image = image.to(device, non_blocking=True)
            organ_mask = organ_mask.to(device, non_blocking=True)

            mask = get_mask(image, R=R, acs_lines=acs_lines, mode=mask_dist,
                            offset=mask_offset)

            # prepare_measurement FIRST, then E from the maps it hands back:
            # the simulated branch RSS-normalizes them and the whitening branch
            # replaces them, so building E from the raw `smaps` would leave the
            # encoding operator inconsistent with y.
            y, sigma_n, extra = prepare_measurement(
                image=image,
                kspace=kspace,
                mask=mask,
                smaps=smaps,
                kspace_type=kspace_type,
                noise_std=noise_std,
                noise_dist=noise_dist,
                whiten_kspace=whiten_kspace,
            )

            smaps = extra["smaps"]
            E = Mask(mask) @ FFT2D() @ Sense(smaps)

            opt.zero_grad(set_to_none=True)

            recon, _ = net(y, E=E, sigma=sigma_n)

            if whiten_kspace and "Zinv" in extra:
                recon = extra["Zinv"] * recon

            image_l, recon_l = apply_loss_mask(
                image, recon, organ_mask, use_organ_mask,
            )

            loss = loss_fn(image_l, recon_l, sigma_n)
            loss.backward()

            if clip_grad is not None:
                nn.utils.clip_grad_norm_(net.parameters(), clip_grad)

            opt.step()

            if hasattr(net, "project"):
                net.project()

            if sched is not None:
                sched.step()

            running_loss += float(loss.item())
            n_batches += 1

            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.2e}", epoch=epoch)

        global_step = (epoch + 1) * steps_per_epoch

        # ==================================================================
        # END-OF-EPOCH: averaged-loss backtracking + checkpoint + logging
        # ==================================================================
        avg_loss = running_loss / max(n_batches, 1)
        nonfinite = not math.isfinite(avg_loss)

        # Metrics on the last batch -- for LOGGING ONLY.
        with torch.no_grad():
            train_metrics = compute_metrics(image.abs(), recon.abs())
        train_metrics = {k: float(v.detach()) for k, v in train_metrics.items()}

        # A non-finite loss always attempts a restore (protective: never let a
        # NaN update get checkpointed over a good model). The threshold check
        # only fires when backtracking is enabled.
        should_backtrack = nonfinite or (
            backtrack_enabled and avg_loss > best_loss + backtrack_thresh
        )

        if should_backtrack:
            reason = "non-finite loss" if nonfinite else (
                f"avg loss {avg_loss:.3e} > best {best_loss:.3e} + {backtrack_thresh}"
            )
            print(f"[epoch {epoch}] {reason} — backtracking")

            if os.path.exists(ckpt_path) and math.isfinite(best_loss):
                net, opt, sched, _ = load_ckpt(
                    ckpt_path,
                    model=net,
                    optimizer=opt,
                    scheduler=sched,
                    device=device,
                )
                old_lr = np.array(get_lr(opt))
                new_lr = old_lr * backtrack_factor
                set_lr(opt, new_lr)
                print("Updated LR:", new_lr)
                # best_loss unchanged: the on-disk ckpt is still the best model.
            else:
                raise RuntimeError(
                    f"Backtrack requested at epoch {epoch} but no valid checkpoint "
                    f"exists yet (best_loss={best_loss})."
                )

        # Save ONLY on genuine improvement -> the on-disk ckpt is always the best.
        elif save_ckpt_fn is not None and avg_loss < best_loss:
            save_ckpt_fn(
                ckpt_path,
                model=net,
                step=global_step,
                optimizer=opt,
                scheduler=sched,
            )
            best_loss = avg_loss

        # ---- train logging (skip wandb if this epoch went non-finite) ----
        if wandb is not None and not nonfinite:
            log_dict = {
                "train/loss": avg_loss,
                "train/lr": get_lr(opt)[0],
                "train/epoch": epoch,
                **{f"train/{k}": v for k, v in train_metrics.items()},
            }
            wandb.log(log_dict, step=global_step)
        elif wandb is None:
            print({"epoch": epoch, "avg_loss": avg_loss, **train_metrics})

        # ==================================================================
        # VALIDATION (log-only, every val_every_epochs epochs)
        # ==================================================================
        if val_every_epochs and (epoch + 1) % val_every_epochs == 0:

            net.eval()

            val_metrics = {
                "psnr": torch.tensor(0.0, device=device),
                "ssim": torch.tensor(0.0, device=device),
                "nrmse": torch.tensor(0.0, device=device),
                "loss": torch.tensor(0.0, device=device),
            }
            num_val_batches = 0

            # Re-seeded every validation pass, so epoch N and epoch N+1 see the
            # SAME noise and (for a random mask_dist) the same sampling pattern.
            val_gen = None
            if val_seed is not None:
                val_gen = torch.Generator(device=device).manual_seed(int(val_seed))

            val_std = noise_std if val_noise_std is None else val_noise_std

            with torch.no_grad():

                for kspace_v, smaps_v, image_v, organ_mask_v in val_loader:

                    kspace_v = kspace_v.to(device, non_blocking=True)
                    smaps_v = smaps_v.to(device, non_blocking=True)
                    image_v = image_v.to(device, non_blocking=True)
                    organ_mask_v = organ_mask_v.to(device, non_blocking=True)

                    mask_v = get_mask(image_v, R=R, acs_lines=acs_lines,
                                      mode=mask_dist, offset=mask_offset)

                    y_v, sigma_v, extra_v = prepare_measurement(
                        image=image_v,
                        kspace=kspace_v,
                        mask=mask_v,
                        smaps=smaps_v,
                        kspace_type=kspace_type,
                        noise_std=val_std,
                        noise_dist=noise_dist,
                        whiten_kspace=whiten_kspace,
                        generator=val_gen,
                    )

                    smaps_v = extra_v["smaps"]
                    E_v = Mask(mask_v) @ FFT2D() @ Sense(smaps_v)

                    recon_v, _ = net(y_v, E=E_v, sigma=sigma_v)

                    if whiten_kspace and "Zinv" in extra_v:
                        recon_v = extra_v["Zinv"] * recon_v

                    m = compute_metrics(image_v.abs(), recon_v.abs())

                    val_metrics["psnr"] += m["psnr"].detach()
                    val_metrics["ssim"] += m["ssim"].detach()
                    val_metrics["nrmse"] += m["nrmse"].detach()

                    image_l_v, recon_l_v = apply_loss_mask(
                        image_v, recon_v, organ_mask_v, use_organ_mask,
                    )

                    val_loss = loss_fn(image_l_v, recon_l_v, sigma_v)
                    val_metrics["loss"] += val_loss

                    num_val_batches += 1

                ### Sample Image Logging for Wandb
                if wandb:
                    # The zero-filled adjoint is the reference the network has
                    # to beat: it is what the undersampling artifact looks like
                    # BEFORE reconstruction, so "did the net remove it or move
                    # it around" is answerable at a glance.
                    zf_v = E_v.adjoint(y_v)
                    if whiten_kspace and "Zinv" in extra_v:
                        zf_v = extra_v["Zinv"] * zf_v

                    panel, pstats = recon_panel(
                        image_v, recon_v, zero_filled=zf_v, gain=residual_gain,
                    )

                    # metrics of the DISPLAYED slice, not the val-set mean --
                    # a caption quoting the set mean next to one image is a
                    # standing invitation to misread the picture.
                    m1 = compute_metrics(image_v[:1].abs(), recon_v[:1].abs())
                    sig = float(torch.as_tensor(sigma_v).reshape(-1)[0])

                    caption = (
                        f"epoch {epoch} | zero-filled | recon | ground truth | "
                        f"|residual| x{pstats['gain']:g}\n"
                        f"R={R} acs={acs_lines} mask={mask_dist} sigma={sig:.4f} "
                        f"| this slice: PSNR {float(m1['psnr']):.2f} dB, "
                        f"SSIM {float(m1['ssim']):.4f}, "
                        f"NRMSE {float(m1['nrmse']):.4f}\n"
                        f"residual: peak {100 * pstats['res_max']:.1f}%, "
                        f"rms {100 * pstats['res_rms']:.2f}% of peak signal "
                        f"(panel saturates at {100 / pstats['gain']:.0f}%)"
                    )

                    wandb.log(
                        {
                            "val/recon_example": wandb.Image(panel, caption=caption),
                            "val/residual_kspace": wandb.Image(
                                residual_kspace(image_v, recon_v),
                                caption=(
                                    f"epoch {epoch}: log|FFT(recon - gt)|. "
                                    f"Energy laid out along the UNSAMPLED "
                                    f"phase-encode lines (a comb of period R={R} "
                                    f"for an equispaced mask, scattered for a "
                                    f"random one) = unrecovered undersampling: a "
                                    f"data-consistency / capacity problem. Energy "
                                    f"piled at the band edges and corners = "
                                    f"grid-locked structure at a coarse level's "
                                    f"Nyquist, i.e. the multigrid transfer pair or "
                                    f"a strided-ConvTranspose checkerboard -- more "
                                    f"depth will NOT remove that one. Flat floor = "
                                    f"noise."
                                ),
                            ),
                            "val/residual_rms_frac": pstats["res_rms"],
                            "val/residual_peak_frac": pstats["res_max"],
                        },
                        step=global_step,
                    )

            mean_metrics = {
                k: (v.detach() / num_val_batches).item()
                for k, v in val_metrics.items()
            }

            if wandb is not None:
                wandb.log(
                    {f"val/{k}": v for k, v in mean_metrics.items()},
                    step=global_step,
                )
                wandb.log(get_filter_grids(net), step=global_step)
            else:
                print(f"[VAL] epoch={epoch} {mean_metrics}")

            net.train()

    pbar.close()
