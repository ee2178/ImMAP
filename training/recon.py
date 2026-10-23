import os
import math
import time
import numpy as np
import torch
from visualization.wandb_image import wandb_image
import torch.nn as nn

from tqdm import tqdm
from training.losses import LOSS_REGISTRY
from training.metrics import compute_metrics
from visualization.params import get_param_logs
from training.common import (
    POSTFIX_EVERY,
    save_ckpt,
    load_ckpt,
    get_lr,
    set_lr,
    apply_loss_mask,
    prepare_measurement,
    embed_for_net,
)

# Backtracking bookkeeping. `backtrack` is the whole restore-and-drop-the-LR
# operation; `resync_schedule` re-applies an already-earned reduction on resume.
from training.common import backtrack as do_backtrack, resync_schedule
from training.step_profile import StepProfiler
from visualization.filters import get_filter_grids
from models.level_trace import (LEVEL_LOSS_OUTERS, _pd_chains, _transfer_kws,
                                collect_level_outputs, supervised_level_images)
from visualization.image import levels_panel, recon_panel, upsample_repeat
from physics.mask import get_mask_cached as get_mask, resolve_acs_lines
from operators import Mask, FFT2D, Sense


# `_embed` moved to training/common.py so evaluation/tasks.py can build the
# identical operator; the alias keeps this module's call sites unchanged.
_embed = embed_for_net

# NON-FINITE STEPS ARE SKIPPED, NOT APPLIED (2026-10-13; Sljiva's train.jl
# guards `isnan(loss)` and `isnan(gradnorm)` before `Optimisers.update!`). A
# step whose loss or gradient norm is NaN / inf used to go into the optimizer:
# the weights and Adam's moments turned NaN, the rest of the epoch was wasted
# on them, and the epoch-end check then restored the best checkpoint -- often
# far older than the bad step -- at a lower learning rate. One bad batch is not
# a diverged run, so it is now dropped and training carries on from where it
# was.
#
# More than this share of an epoch's steps skipped means the WEIGHTS are the
# problem, not a batch, and the epoch is handed to the existing restore.
MAX_SKIPPED_FRACTION = 0.1

# RELATIVE BACKTRACK THRESHOLD (2026-10-13; Sljiva's `backtrack_factor`, there
# 100-500). An epoch whose average loss exceeds this many times the best
# epoch's is restored from the checkpoint at a lower learning rate. Before
# this, with `backtrack_thresh` None in every generated config, only a
# NON-FINITE epoch was -- a run that blew up to a large finite loss, or
# collapsed to an all-zero output, just kept training.
#
# A RATIO, because the losses used here are normalised by the target (nl1 + nl2:
# an all-zero output scores ~2 whatever the anatomy), so one number serves every
# cell and no loss scale has to be known in advance -- the reason the absolute
# margin was never set. 2 is deliberately loose: the "best" it is compared with
# is the luckiest of many noisy 50-step batch-1 averages, and a false alarm
# costs an epoch and 10% of the learning rate for good; a genuinely diverged
# run is far past it (a collapse from a best of 0.27 is 7x).
#
# 500, SLJIVA'S VALUE, since 2026-10-14 (was 2). At 2 the check fired on the
# ordinary epoch-to-epoch swings of early training, on runs that were not
# diverging -- each false alarm a restore and a permanent learning-rate cut
# (x `backtrack_factor`, 0.9 in the generated configs). Sljiva compares the
# same thing every epoch (train.jl:455, `|avg_loss / bestloss| > factor`) with
# `backtrack_factor: 500` from config/train.yaml (the train.jl default is 100).
# NB the names cross: there `backtrack_factor` is this RATIO; here it is the
# learning-rate multiplier.
#
# 100 since 2026-10-19 (Sljiva's train.jl default; was 500). At 500 two
# MG-Group cells (M=169 brain R12, M=169 knee R8) diverged and never came back:
# the blow-up sat under 500x the best and was never restored. 100 is still far
# above the early-training swings that made 2 fire on healthy runs.
#
# It will NOT catch a collapse to an all-zero output (~2 on these normalised
# losses, ~7x a best of 0.27) -- the validation check below is what catches a
# model that has gone bad without its training loss exploding. A non-finite
# epoch still restores either way (that path does not use the ratio). `None` in
# a config's `training.backtrack_ratio` turns the check off.
BACKTRACK_RATIO = 100.0

# VALIDATION BACKTRACKING (2026-10-19). A validation pass whose loss exceeds
# this many times the best validation loss so far -- or is non-finite -- is
# restored from the BEST-VALIDATION checkpoint (`net_bestval.ckpt`, written on
# every improving validation pass) at a lower learning rate, exactly as the
# training check restores `net.ckpt`.
#
# Its own checkpoint, because `net.ckpt` is the best TRAINING-loss model, and
# that can be the very model whose validation blew up: the training check then
# restores it faithfully and validation stays bad -- what MG-Group runs showed.
# After a validation restore `net.ckpt` is rewritten with the restored weights
# (and the training loss recorded beside them), so a later training backtrack
# or a requeue lands on the good model too, not back on the bad one.
#
# 2, not the training check's 100: the validation loss has none of the epoch
# loss's noise -- the same slices, the same seeded noise and masks every pass
# (`val_seed`) -- so it moves only when the weights do, and a doubling between
# passes 100 epochs apart is a model that got worse, not an unlucky batch.
# `None` in a config's `training.val_backtrack_ratio` turns it off (a
# non-finite validation loss still restores).
VAL_BACKTRACK_RATIO = 2.0
BESTVAL_CKPT = "net_bestval.ckpt"


def _ckpt_field(path, key):
    """One bookkeeping value out of a checkpoint, or None."""
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:                                   # torch < 1.13
        ck = torch.load(path, map_location="cpu")
    return ck.get(key)


@torch.no_grad()
def _levels_log(net, y, E, sigma, T, zinv, mask, gt):
    """The `log_levels` strip -> `(panel, [(h, w) per level, finest first])`,
    or None when `net` has no per-level images.

    Every level's image is brought to the embedded grid by pixel repetition,
    then treated exactly like the recon in the panel beside it: cropped by `T`,
    un-whitened by `zinv`, masked by `mask` -- so the finest tile IS the recon
    tile of val/recon_example."""
    fn = getattr(net, "forward_level_images", None)
    if fn is None or getattr(net, "levels", 1) < 2:
        return None
    _, imgs = fn(y, E=E, sigma=sigma)
    if not imgs:
        return None
    full = imgs[0].shape[-2:]
    out = []
    for x in imgs:
        x = T.forward(upsample_repeat(x, full))
        if zinv is not None:
            x = zinv * x
        out.append(x if mask is None else x * mask)
    return levels_panel(out, gt), [tuple(x.shape[-2:]) for x in imgs]


def _level_loss_weights(net, weight, outers):
    """`level_loss_weight` -> one weight per coarse level (L1 first), or [] when
    off. Refuses a net with no primal-dual V-cycle: the option would silently
    do nothing there."""
    ws = list(weight) if isinstance(weight, (list, tuple)) else [weight]
    if not any(float(w) for w in ws):
        return []
    if outers not in LEVEL_LOSS_OUTERS:
        raise ValueError(f"level_loss_outers must be one of {LEVEL_LOSS_OUTERS}, got {outers!r}")
    if not _pd_chains(net):
        raise ValueError(f"level_loss_weight is set but {type(net).__name__} has no "
                         f"primal-dual V-cycle: per-level supervision is for multigrid "
                         f"MGLPDS nets only")
    n = int(getattr(net, "levels", 1)) - 1
    if len(ws) == 1:
        ws = ws * n
    if len(ws) != n:
        raise ValueError(f"level_loss_weight has {len(ws)} weights for {n} coarse levels")
    if any(float(w) < 0 for w in ws):
        raise ValueError(f"level_loss_weight must be >= 0, got {ws}")
    return [float(w) for w in ws]


def _level_loss(net, levels, recon_e, x0, image, organ_mask, T, zinv, use_mask, weights,
                loss_fn, sigma):
    """sum_l w_l * mean over the collected V-cycles of loss_fn(R^l gt, x_l), on
    the embedded grid, both sides weighted by R^l of the crop support (and of
    the organ mask when it is on) -- the coarse grids cannot be cropped by T."""
    from operators.resample import restrict

    kws = _transfer_kws(net)
    imgs = supervised_level_images(net, levels, recon_e, x0)
    if zinv is not None and torch.is_tensor(zinv) and zinv.dim() == 4 and \
            zinv.shape[-2:] != (1, 1):
        raise NotImplementedError("per-level supervision with a spatial whitening map")
    g = T.adjoint(image)
    w = T.adjoint(torch.ones_like(image)).real
    if use_mask:
        w = w * T.adjoint(organ_mask.to(image.dtype)).real
    total, level = 0.0, 0
    for l in sorted(imgs):
        while level < l:
            g, w, level = restrict(g, **kws), restrict(w, **kws), level + 1
        wl = w.clamp(0.0, 1.0)
        terms = []
        for x in imgs[l]:
            if zinv is not None:
                x = zinv * x
            terms.append(loss_fn(g * wl, x * wl, sigma))
        total = total + weights[l - 1] * (sum(terms) / len(terms))
    return total


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
    # Estimate the coil maps from the ACS of each measurement instead of
    # taking the dataset's precomputed ones -- what Sljiva's `genobs` does.
    # None keeps the precomputed maps. See physics/online_smaps.py.
    online_smaps=None,
    online_smaps_kws=None,
    mask_dist="uniform",
    mask_offset=0,
    # Sampling heuristics from Sljiva/src/mask.jl. `adjust_accel=True` makes
    # the ACS come out of the acceleration budget instead of being added to it
    # -- without it a nominal R=16 with 20 ACS lines samples at an EFFECTIVE
    # R of 8.2. `center_frac` is its ACS-as-a-fraction convention.
    center_frac=None,
    adjust_accel=False,
    noise_std=(0.0, 0.01),
    noise_dist="uniform",
    # Validation noise. `val_noise_std=None` redraws from `noise_std` like
    # training does; a number pins it (use the midpoint of the training range,
    # as Sljiva's closure does -- `genobs(clo, Val{true}(), s)` evaluates at
    # `mean(clo.noise_level)`). `val_seed` fixes the realization, so val curves
    # move because the model moved, not because the noise did.
    val_noise_std=None,
    val_seed=None,
    # How the VALIDATION measurement is made; None = the training `kspace_type`,
    # which is what the generator writes since 2026-10-22 (configs generated
    # 2026-10-20 .. 2026-10-22 carry "simulated"; scripts/set_val_kspace.py
    # removes it from a run). "simulated" pushes the SENSE ground truth through
    # the STORED maps, mask and AWGN, so the noise in y is exactly
    # `val_noise_std` -- measured k-space already carries the scanner's own
    # noise on top of what is added. The operator still gets its maps the
    # training way (online ESPIRiT from the measurement's ACS).
    val_kspace_type=None,
    ### Loss
    loss_type="complex-mse",
    # `organ_mask` is the coil-sensitivity support the recon loader returns.
    # This ONE flag governs all three places it can act, because splitting it
    # would let the loss optimise one region while the curves scored another:
    #   * the loss           (image * mask vs recon * mask)
    #   * PSNR / NRMSE / SSIM (restricted to the region -- see training.metrics)
    #   * the wandb panel     (masked images, so the picture matches the number)
    # Off by default: turning it on changes what every metric MEANS, so runs
    # either side of the switch are not comparable.
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
    # Multigrid nets only: also log `val/levels`, the image every grid level
    # holds after the LAST V-cycle (coarsest -> finest, then the ground truth),
    # coarse levels pixel-repeated up to full size, on val/recon_example's
    # greyscale and mask. One extra traced forward per validation, on the slice
    # the recon panel shows. Ignored (with one notice) by nets without levels.
    log_levels=False,
    # PER-LEVEL SUPERVISION, multigrid MGLPDS nets only (2026-10-23). Off at 0.
    # Adds, to the output loss, `level_loss_weight` x the same loss between
    # each COARSE level's final iterate (what it hands back up, in image units)
    # and R^l(ground truth) -- restricted with the net's own transfer, on the
    # embedded grid, weighted by the restricted crop support (and the
    # restricted organ mask when `use_organ_mask`). Without it nothing asks a
    # coarse level to hold an image: the loss sees only the output, and the
    # trained coarse levels end each V-cycle mid-oscillation.
    #   level_loss_weight  a number (every coarse level that weight) or a list,
    #                      one weight per coarse level, L1 first
    #   level_loss_outers  "last": the final V-cycle only; "all": every
    #                      V-cycle, averaged per level
    # The TRAINING loss (and so the training backtrack) includes the level
    # terms; the VALIDATION loss stays the output's alone -- it picks the
    # best-validation checkpoint, which must be about the reconstruction.
    level_loss_weight=0.0,
    level_loss_outers="last",
    start_epoch=0,            # was: start_step
    save_dir=None,
    ckpt=None,
    save_ckpt_fn=save_ckpt,
    # NOTE: backtrack_thresh is now an averaged-LOSS margin (lower is better),
    # not a PSNR margin. None disables the threshold check (but a non-finite
    # loss will still trigger a protective restore). Tune to your loss scale.
    backtrack_thresh=None,
    # The RELATIVE check: restore when the epoch's average loss exceeds
    # `backtrack_ratio` x the best epoch's (see BACKTRACK_RATIO). None -- the
    # default -- turns it off. Independent of `backtrack_thresh`; either one
    # triggers a restore.
    backtrack_ratio=BACKTRACK_RATIO,
    # The VALIDATION check: restore the best-validation checkpoint when a
    # validation pass's loss exceeds `val_backtrack_ratio` x the best one's
    # (see VAL_BACKTRACK_RATIO). None turns it off.
    val_backtrack_ratio=VAL_BACKTRACK_RATIO,
    backtrack_factor=0.8,
    # Backtracking bookkeeping, threaded in from train.py so it survives a
    # requeue. `backtrack_count` sets the LR amplitude (lr0 * factor**count) and
    # `best_loss` is the bar the on-disk checkpoint already cleared -- resetting
    # either one on resume is what let a backtracked run climb back to full LR.
    backtrack_count=0,
    best_loss=float("inf"),
    # The validation bar. NOT threaded from train.py: it is read back from
    # net_bestval.ckpt below on a resume, which is the file it describes.
    best_val_loss=float("inf"),
    wandb=None,
):
    net.to(device)
    net.train()

    loss_fn = LOSS_REGISTRY[loss_type]
    ckpt_path = os.path.join(save_dir, "net.ckpt")
    bestval_path = os.path.join(save_dir, BESTVAL_CKPT)
    # A best-val checkpoint is trusted only on a RESUME. On a fresh start one
    # left in the directory belongs to an earlier run (FORCE_RESTART does not
    # delete files) and must neither set the bar nor be restored from; it is
    # overwritten by this run's first validation pass.
    bestval_ok = start_epoch > 0 and os.path.exists(bestval_path)
    if bestval_ok:
        v = _ckpt_field(bestval_path, "best_val_loss")
        if v is not None and math.isfinite(v):
            best_val_loss = min(best_val_loss, float(v))
            print(f"resuming with best validation loss {best_val_loss:.4e} "
                  f"({BESTVAL_CKPT})")

    def metric_mask(m):
        """The mask to hand `compute_metrics` / the panel, or None when off.

        One place decides whether masking is live, so the loss, the numbers and
        the picture can never end up disagreeing about which region this run is
        about. `apply_loss_mask` reads the same flag.
        """
        return m if use_organ_mask else None

    # best_loss is a PARAMETER now, carried in from the checkpoint: see
    # the note in the signature.
    backtrack_enabled = backtrack_thresh is not None

    train_iter = iter(train_loader)
    total_steps = num_epochs * steps_per_epoch
    pbar = tqdm(
        total=total_steps,
        initial=start_epoch * steps_per_epoch,
        desc="TRAIN",
        dynamic_ncols=True,
    )

    # `train.py` rebuilds `sched` from the config on every launch, so its
    # base_lrs are the run's ORIGINAL learning rate. Re-apply the reductions
    # this run has already earned, or a requeue silently undoes every backtrack.
    if backtrack_count:
        print(f"resuming at backtrack_count={backtrack_count}: LR -> "
              f"{resync_schedule(opt, sched, backtrack_count, backtrack_factor)}")

    # One filter log at the start, so every later image has something to be
    # compared with. Images only: no spread / drift statistics, and so no
    # snapshot of the whole model held for the length of the run.
    if wandb is not None:
        wandb.log(get_filter_grids(net), step=start_epoch * steps_per_epoch)

    # PROFILE_STEPS=N in the environment times the first N steps per phase
    # (training/step_profile.py). Off by default, and then every `prof.*` call
    # below returns at once.
    prof = StepProfiler(device)
    levels_off = False        # set once a net turns out to have no level images
    level_weights = _level_loss_weights(net, level_loss_weight, level_loss_outers)
    if level_weights:
        print(f"per-level supervision: weights {level_weights} on levels "
              f"{list(range(1, len(level_weights) + 1))}, V-cycles: {level_loss_outers}")

    for epoch in range(start_epoch, num_epochs):
        net.train()
        running_loss = 0.0
        n_batches = 0             # steps APPLIED: what the average is over
        n_skipped = 0             # steps dropped for a non-finite loss / gradient
        last_ok = None            # (image, recon, organ_mask) of the last applied step
        profiling_epoch = prof.on

        # ==================================================================
        # ONE EPOCH == steps_per_epoch GRADIENT STEPS
        # ==================================================================
        for _ in range(steps_per_epoch):

            prof.start()
            try:
                kspace, smaps, image, organ_mask, pad_hw = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                kspace, smaps, image, organ_mask, pad_hw = next(train_iter)

            kspace = kspace.to(device, non_blocking=True)
            smaps = smaps.to(device, non_blocking=True)
            image = image.to(device, non_blocking=True)
            organ_mask = organ_mask.to(device, non_blocking=True)
            prof.mark("data")

            mask = get_mask(image, R=R, acs_lines=acs_lines, mode=mask_dist,
                            offset=mask_offset, center_frac=center_frac,
                            adjust_accel=adjust_accel)

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
                online_smaps=online_smaps,
                online_smaps_kws=dict(online_smaps_kws or {},
                                      acs_lines=resolve_acs_lines(
                                          image.shape[-1], acs_lines,
                                          center_frac)),
                need_smaps=getattr(net, "uses_operator_smaps", True),
            )

            smaps = extra["smaps"]
            E = Mask(mask) @ FFT2D() @ Sense(smaps)

            # Solve on a grid the network can halve cleanly, measure on the one
            # the scanner used. T crops back and its adjoint zero-pads, so
            # E @ T is exact -- no resampled mask, no reflect-padded maps.
            E, T = _embed(net, E, image, pad_hw)
            prof.mark("measure")

            opt.zero_grad(set_to_none=True)

            if level_weights:
                with collect_level_outputs(net, level_loss_outers) as lv:
                    recon_e, (x0, _) = net(y, E=E, sigma=sigma_n)
            else:
                recon_e, _ = net(y, E=E, sigma=sigma_n)
            recon = T.forward(recon_e)
            prof.mark("forward")

            zinv = extra.get("Zinv") if whiten_kspace else None
            if zinv is not None:
                recon = zinv * recon

            image_l, recon_l = apply_loss_mask(
                image, recon, organ_mask, use_organ_mask,
            )

            loss = loss_fn(image_l, recon_l, sigma_n)
            if level_weights:
                loss = loss + _level_loss(net, lv.levels, recon_e, x0, image, organ_mask,
                                          T, zinv, use_organ_mask, level_weights,
                                          loss_fn, sigma_n)
            loss.backward()
            prof.mark("backward")

            # The gradient norm BEFORE clipping. With no clip configured the
            # norm is still needed for the check below; an infinite max_norm
            # computes it and scales the gradients by exactly 1.
            grad_norm = nn.utils.clip_grad_norm_(
                net.parameters(),
                clip_grad if clip_grad is not None else float("inf"))

            # Apply the step only if both are finite (see MAX_SKIPPED_FRACTION).
            # This is the step's one device sync: the host waits here for the
            # backward pass it has just queued.
            step_ok = bool(torch.isfinite(loss.detach()) & torch.isfinite(grad_norm))

            if step_ok:
                opt.step()
            else:
                opt.zero_grad(set_to_none=True)    # nothing of this step survives
                n_skipped += 1
            prof.mark("opt")

            if step_ok and hasattr(net, "project"):
                net.project()
            prof.mark("project")

            # The schedule is a function of the step COUNT, skipped or not, so
            # that it still spans exactly the configured number of steps.
            if sched is not None:
                sched.step()
            prof.mark("opt")

            # one eval-mode forward on this batch, timed apart from the step
            prof.infer(net, y, E, sigma_n)
            prof.end(image.shape[-2:], embedded=not T.is_identity)

            if step_ok:
                running_loss += loss.detach()      # on-device; no sync
                n_batches += 1
                last_ok = (image, recon.detach(), organ_mask)

            pbar.update(1)
            if step_ok and n_batches % POSTFIX_EVERY == 0:
                pbar.set_postfix(loss=f"{loss.item():.2e}", epoch=epoch)

        global_step = (epoch + 1) * steps_per_epoch

        # ==================================================================
        # END-OF-EPOCH: averaged-loss backtracking + checkpoint + logging
        # ==================================================================
        t_epoch_end = time.perf_counter()
        # Averaged over the steps that were APPLIED. A skipped step's loss is
        # non-finite by definition, and one of those in the sum would turn a
        # dropped batch into a restore -- the thing skipping it avoids.
        avg_loss = float(running_loss) / max(n_batches, 1)
        too_many = n_skipped > MAX_SKIPPED_FRACTION * steps_per_epoch
        nonfinite = n_batches == 0 or too_many or not math.isfinite(avg_loss)
        if n_skipped:
            print(f"[epoch {epoch}] skipped {n_skipped} of {steps_per_epoch} steps: "
                  f"non-finite loss or gradient, not applied to the weights")

        # Metrics on the last APPLIED batch -- for LOGGING ONLY.
        train_metrics = {}
        if last_ok is not None:
            image_m, recon_m, organ_mask_m = last_ok
            with torch.no_grad():
                train_metrics = compute_metrics(
                    image_m.abs(), recon_m.abs(), mask=metric_mask(organ_mask_m))
            train_metrics = {k: float(v.detach()) for k, v in train_metrics.items()}

        # A non-finite epoch always attempts a restore (protective: never let a
        # NaN update get checkpointed over a good model). The two finite checks
        # compare with the best epoch so far: by a ratio (on unless
        # `backtrack_ratio` is None) and by an absolute margin (off unless
        # `backtrack_thresh` is set). Neither can fire before a first
        # checkpoint exists -- best_loss is +inf until then.
        over_ratio = (backtrack_ratio is not None and math.isfinite(best_loss)
                      and avg_loss > backtrack_ratio * best_loss)
        over_margin = backtrack_enabled and avg_loss > best_loss + backtrack_thresh
        should_backtrack = nonfinite or over_ratio or over_margin

        if should_backtrack:
            reason = (
                f"{n_skipped} of {steps_per_epoch} steps non-finite "
                f"(more than {MAX_SKIPPED_FRACTION:.0%})" if (too_many or n_batches == 0)
                else "non-finite loss" if nonfinite else
                f"avg loss {avg_loss:.3e} > {backtrack_ratio:g} x best {best_loss:.3e}"
                if over_ratio else
                f"avg loss {avg_loss:.3e} > best {best_loss:.3e} + {backtrack_thresh}"
            )
            print(f"[epoch {epoch}] {reason} — backtracking")

            if os.path.exists(ckpt_path) and math.isfinite(best_loss):
                backtrack_count, new_lr = do_backtrack(
                    ckpt_path,
                    model=net,
                    optimizer=opt,
                    scheduler=sched,
                    device=device,
                    backtrack_count=backtrack_count,
                    backtrack_factor=backtrack_factor,
                )
                print(f"backtrack #{backtrack_count} -> LR {new_lr}")
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
                backtrack_count=backtrack_count,
                best_loss=avg_loss,
                best_val_loss=best_val_loss,
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

        if profiling_epoch:
            # Not a step, but the progress bar charges it to one: this is the
            # periodic stall behind a "varying" it/s.
            print(f"[profile] epoch {epoch}: end-of-epoch work (train metrics + "
                  f"checkpoint{'' if best_loss == avg_loss else ' [not saved]'} + "
                  f"logging) took {1e3 * (time.perf_counter() - t_epoch_end):.0f} ms",
                  flush=True)

        # ==================================================================
        # VALIDATION (log-only, every val_every_epochs epochs)
        # ==================================================================
        if val_every_epochs and (epoch + 1) % val_every_epochs == 0:

            net.eval()
            t_val = time.perf_counter()

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

                for (kspace_v, smaps_v, image_v, organ_mask_v,
                     pad_hw_v) in val_loader:

                    kspace_v = kspace_v.to(device, non_blocking=True)
                    smaps_v = smaps_v.to(device, non_blocking=True)
                    image_v = image_v.to(device, non_blocking=True)
                    organ_mask_v = organ_mask_v.to(device, non_blocking=True)

                    mask_v = get_mask(image_v, R=R, acs_lines=acs_lines,
                                      mode=mask_dist, offset=mask_offset,
                                      center_frac=center_frac,
                                      adjust_accel=adjust_accel)

                    y_v, sigma_v, extra_v = prepare_measurement(
                        image=image_v,
                        kspace=kspace_v,
                        mask=mask_v,
                        smaps=smaps_v,
                        kspace_type=val_kspace_type or kspace_type,
                        noise_std=val_std,
                        noise_dist=noise_dist,
                        whiten_kspace=whiten_kspace,
                        online_smaps=online_smaps,
                        online_smaps_kws=dict(online_smaps_kws or {},
                                              acs_lines=resolve_acs_lines(
                                                  image_v.shape[-1], acs_lines,
                                                  center_frac)),
                        need_smaps=getattr(net, "uses_operator_smaps", True),
                        generator=val_gen,
                    )

                    smaps_v = extra_v["smaps"]
                    E_v = Mask(mask_v) @ FFT2D() @ Sense(smaps_v)
                    E_v, T_v = _embed(net, E_v, image_v, pad_hw_v)

                    recon_v, _ = net(y_v, E=E_v, sigma=sigma_v)
                    recon_v = T_v.forward(recon_v)

                    if whiten_kspace and "Zinv" in extra_v:
                        recon_v = extra_v["Zinv"] * recon_v

                    m = compute_metrics(
                        image_v.abs(), recon_v.abs(),
                        mask=metric_mask(organ_mask_v))

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
                    zf_v = T_v.forward(E_v.adjoint(y_v))
                    if whiten_kspace and "Zinv" in extra_v:
                        zf_v = extra_v["Zinv"] * zf_v

                    # Show the same pixels the metrics scored. An unmasked
                    # panel next to a masked PSNR invites reading a background
                    # artifact as the cause of a number that never saw it --
                    # and the residual tile, which is where such an artifact is
                    # most visible, would be the most misleading of the four.
                    # `p_mask` also zeroes the ZERO-FILLED tile: it is the
                    # reference the recon is judged against, so masking one and
                    # not the other would make the comparison unfair by eye.
                    p_mask = metric_mask(organ_mask_v)
                    img_p, rec_p, zf_p = (
                        (image_v, recon_v, zf_v) if p_mask is None else
                        (image_v * p_mask, recon_v * p_mask, zf_v * p_mask))

                    panel, pstats = recon_panel(
                        img_p, rec_p, zero_filled=zf_p, gain=residual_gain,
                    )

                    # metrics of the DISPLAYED slice, not the val-set mean --
                    # a caption quoting the set mean next to one image is a
                    # standing invitation to misread the picture.
                    m1 = compute_metrics(
                        image_v[:1].abs(), recon_v[:1].abs(),
                        mask=None if p_mask is None else p_mask[:1])
                    sig = float(torch.as_tensor(sigma_v).reshape(-1)[0])

                    caption = (
                        f"epoch {epoch} | zero-filled | recon | ground truth | "
                        f"|residual| x{pstats['gain']:g}\n"
                        f"R={R} acs={resolve_acs_lines(image_v.shape[-1], acs_lines, center_frac)} "
                        f"mask={mask_dist} sigma={sig:.4f} "
                        f"| this slice{' [organ mask]' if p_mask is not None else ''}: "
                        f"PSNR {float(m1['psnr']):.2f} dB, "
                        f"SSIM {float(m1['ssim']):.4f}, "
                        f"NRMSE {float(m1['nrmse']):.4f}"
                    )

                    wandb.log(
                        {"val/recon_example": wandb_image(panel, caption=caption)},
                        step=global_step,
                    )

                    if log_levels and not levels_off:
                        lp = _levels_log(net, y_v, E_v, sigma_v, T_v,
                                         extra_v.get("Zinv") if whiten_kspace else None,
                                         p_mask, img_p)
                        if lp is None:
                            levels_off = True
                            print("log_levels: this net has no per-level images "
                                  "(not an MGLPDS V-cycle) -- val/levels is not logged")
                        else:
                            lpanel, sizes = lp
                            wandb.log({"val/levels": wandb_image(
                                lpanel,
                                caption=(f"epoch {epoch} | last V-cycle, each grid level: "
                                         + " | ".join(f"L{l} {h}x{w}" for l, (h, w)
                                                      in reversed(list(enumerate(sizes))))
                                         + " | ground truth\ncoarse levels pixel-repeated "
                                         "to full size; same greyscale as recon_example"
                                         + (" [organ mask]" if p_mask is not None else "")))},
                                step=global_step)

            mean_metrics = {
                k: (v.detach() / num_val_batches).item()
                for k, v in val_metrics.items()
            }

            if wandb is not None:
                wandb.log(
                    {f"val/{k}": v for k, v in mean_metrics.items()},
                    step=global_step,
                )
                # Parameter values ride the VALIDATION cadence, not the epoch one, and for the
                # same reason get_filter_grids does: both walk the whole model and pay a host
                # transfer per tensor, and neither answers a question that needs answering
                # every epoch -- thresholds and step sizes drift on the timescale of training.
                wandb.log(get_param_logs(net), step=global_step)
                wandb.log(get_filter_grids(net), step=global_step)
            else:
                print(f"[VAL] epoch={epoch} {mean_metrics}")
            # The bar does not advance during validation, so tqdm reads it as
            # one very slow step and its smoothed it/s sags afterwards.
            print(f"[VAL] epoch={epoch}: validation + logging took "
                  f"{time.perf_counter() - t_val:.1f} s over {num_val_batches} "
                  f"slices", flush=True)

            # ---- validation backtracking (see VAL_BACKTRACK_RATIO) ----
            vloss = float(mean_metrics["loss"])
            v_nonfinite = not math.isfinite(vloss)
            v_over = (val_backtrack_ratio is not None and math.isfinite(best_val_loss)
                      and vloss > val_backtrack_ratio * best_val_loss)
            if v_nonfinite or v_over:
                why = ("non-finite val loss" if v_nonfinite else
                       f"val loss {vloss:.3e} > {val_backtrack_ratio:g} x best val "
                       f"{best_val_loss:.3e}")
                src = bestval_path if bestval_ok else ckpt_path
                if os.path.exists(src):
                    print(f"[VAL] epoch={epoch}: {why} -- backtracking to "
                          f"{os.path.basename(src)}")
                    backtrack_count, new_lr = do_backtrack(
                        src, model=net, optimizer=opt, scheduler=sched,
                        device=device, backtrack_count=backtrack_count,
                        backtrack_factor=backtrack_factor)
                    print(f"val backtrack #{backtrack_count} -> LR {new_lr}")
                    # net.ckpt may hold the model that just failed validation:
                    # replace it with the restored one, and its bar with the
                    # training loss that model had, so neither a training
                    # backtrack nor a requeue goes back to the bad weights.
                    if src == bestval_path:
                        b = _ckpt_field(bestval_path, "best_loss")
                        if b is not None and math.isfinite(b):
                            best_loss = float(b)
                    if save_ckpt_fn is not None:
                        save_ckpt_fn(ckpt_path, model=net, step=global_step,
                                     optimizer=opt, scheduler=sched,
                                     backtrack_count=backtrack_count,
                                     best_loss=best_loss, best_val_loss=best_val_loss)
                else:
                    print(f"[VAL] epoch={epoch}: {why}, but there is no checkpoint "
                          f"to restore yet -- continuing")
            elif vloss < best_val_loss:
                best_val_loss = vloss
                if save_ckpt_fn is not None:
                    # The training loss stored with it is that of the weights
                    # in hand: the epoch average, or -- when this epoch was
                    # itself restored -- the checkpoint's.
                    save_ckpt_fn(bestval_path, model=net, step=global_step,
                                 optimizer=opt, scheduler=sched,
                                 backtrack_count=backtrack_count,
                                 best_loss=best_loss if should_backtrack else avg_loss,
                                 best_val_loss=best_val_loss)
                    bestval_ok = True

            net.train()

    pbar.close()
