"""
Latent I2SB training loop (design A: the Schrodinger bridge lives ENTIRELY in the M-channel
latent, and we decode to the T1ce image only once). Same skeleton as train_i2sb (epoch =
steps_per_epoch gradient steps, EMA of the trainable weights, averaged-loss backtracking,
checkpoint-on-improvement, wandb logging, single_pass / full_recon validation).

Three convolutional dictionaries, two frozen:

    D_joint  : frozen GroupCDL, C = 1 + n_cond   (trained on [bridge_mean, FLAIR, T1, T2])
    R        : the NEW GroupCDL, C = M (uncond) or 2M (cond)   (trained here; the regressor)
    D_t1ce   : frozen GroupCDL, C = 1            (trained on T1ce)

Both frozen dictionaries map into the SAME M-channel latent (identical GroupCDL M and sc, so
the two sparse codes z share a shape). D_joint is used only as an ENCODER (its sparse code z),
D_t1ce as both ENCODER (for the target endpoint z0) and DECODER (its synthesis dictionary
D = B[0], a transposed conv).

Per training step (everything except the final decode is in the latent domain):

    x0, x1, cond, mask <- batch          (x0 = T1ce target, x1 = T1 prior, cond = FLAIR/T1/T2)
    z0 = D_t1ce( x0 ).z                                   # frozen encode -> target latent
    z1 = D_joint( cat[x1, cond] ).z                       # frozen encode -> prior  latent
    step ~ U{0..interval-1}
    zt   = q_sample(step, z0, z1)                         # LATENT bridge interpolant + latent noise
    z0h  = R( zt [|| z1] ; sigma=std_fwd ).recon[:, :M]   # NEW net: latent -> latent regression
    x0h  = D_t1ce.D( z0h ) + dc                           # frozen transposed-conv decode -> T1ce
    loss = loss_fn(x0, x0h ; sigma=std_fwd)               # IMAGE-domain (T1ce) loss, masked

The reverse sampler (full_recon validation) runs the DDPM posterior in the LATENT domain --
diffusion/i2sb.py:ddpm_sampling is space-agnostic, so we hand it z1 as the start and a pred_x0
callable that returns the latent z0 estimate; only ONE decode happens, on the final latent.
This is the inference-time win over an image-domain bridge: the attention-heavy D_joint encode
runs once (not per step), and R is the only per-step network.

Conditioning toggle (`conditional`):
    True  -- R sees cat[zt, z1] every step (the prior latent as a conditioning stack); R.C = 2M,
             and we keep recon[:, :M] as the z0 estimate (mirrors cdlnet_pred's x_hat[:, :1]).
    False -- R sees zt only; R.C = M. Conditioning then enters only through the z1 bridge endpoint.

Loss: any key from training.losses.LOSS_REGISTRY via `loss_type` (default "complex-mse", which for
the real T1ce equals MSE). Masking is applied in front of the loss (use_mask), so masked
complex-mse reproduces the old masked_mse exactly. Only parameterization="x0" is meaningful --
R's decoded output IS the T1ce estimate, so the loss is in the target image domain (NOT latent).

DC: D_t1ce.D is the bare synthesis operator (GroupCDL reconstructs zero-mean codes as
`recon = D z + mu`). decode_dc="x1_mean" re-adds the per-image mean of the prior x1 -- available
identically at train and inference -- so no train/test mismatch. "none" lets R + the dictionary
carry the DC. NOTE: the bridge noise (std_sb) is now in LATENT units; `tau` almost certainly
needs re-tuning to the latent's scale (image-domain tau is wrong here).

Gradient flow: the frozen encodes run under no_grad; gradients backprop through the frozen
D_t1ce.D (constant filters, requires_grad=False) into R only.
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

from operators import Identity
from training.common import (save_ckpt, load_ckpt, get_lr, set_lr, apply_loss_mask, load_model,
                             snr_loss_weight)
from training.losses import LOSS_REGISTRY
from training.metrics import compute_metrics
from physics.bbridge import build_bridge, n_steps, space_indices
from diffusion.i2sb import q_sample, get_std_fwd, ddpm_sampling
from visualization.filters import get_filter_grids


# ---------------------------------------------------------------------------
# EMA (identical to training/i2sb.py). Shadows the module's Parameters in place so
# load_ckpt's in-place load during backtracking keeps the references valid.
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


def _apply_loss(loss_fn, target, pred, mask, use_mask, sigma):
    """Mask (optionally) then evaluate a training.losses registry loss. The registry convention
    is loss_fn(target, pred, sigma) (see train_denoiser's loss_fn(gt, recon, sigma)); masked
    complex-mse == the old masked_mse since mask in {0,1}."""
    target, pred = apply_loss_mask(target, pred, mask, use_mask)
    return loss_fn(target, pred, sigma)


def _combine_loss(loss_fn, loss_mode, latent_weight, image_weight, loss_weight,
                  z0, z0_hat, x0, pred_x0, mask, use_mask, sigma):
    """Latent-I2SB training loss. The reverse posterior lives in the latent, so R must predict the
    latent endpoint z0: an image-only loss leaves z0_hat free in the decoder's (huge) null space
    (z0_hat decodes to x0 yet is ~orthogonal to z0), and the sampler then collapses. Modes:
      'latent' -- MSE(z0_hat, z0), UNMASKED over the full code (the fix; the bridge noise is added
                  unmasked, so the whole code is the relevant population).
      'image'  -- the registry loss on the decoded image, masked (legacy; under-determined).
      'mixed'  -- latent_weight * latent + image_weight * image.
    NOTE the two terms live on very different scales (code std ~O(0.05) vs image std ~O(0.5), so raw
    image MSE is ~100x the latent MSE); in 'mixed' weight the latent term up so it is not swamped.
    `loss_weight` != 'uniform' reweights EACH SAMPLE by snr_loss_weight(sigma_t) ('t1' -> emphasize
    t=1) and is MSE-based, so it bypasses loss_type. Metrics/visualizations stay image-domain."""
    if loss_mode not in ("latent", "image", "mixed"):
        raise ValueError(f"loss_mode {loss_mode!r} must be 'latent', 'image', or 'mixed'.")

    if loss_weight == "uniform":                              # batch-reduced registry losses
        total = z0_hat.new_zeros(())
        if loss_mode in ("latent", "mixed"):
            total = total + latent_weight * F.mse_loss(z0_hat, z0)
        if loss_mode in ("image", "mixed"):
            total = total + image_weight * _apply_loss(loss_fn, x0, pred_x0, mask, use_mask, sigma)
        return total

    # per-sample sigma_t weighting (MSE-based): weight each sample's loss by snr_loss_weight
    w = snr_loss_weight(sigma, loss_weight)                   # (B,)
    total = z0_hat.new_zeros(())
    if loss_mode in ("latent", "mixed"):
        lat = ((z0_hat - z0) ** 2).flatten(1).mean(dim=1)     # (B,) per-sample latent MSE (unmasked)
        total = total + latent_weight * (w * lat).mean()
    if loss_mode in ("image", "mixed"):
        tgt, prd = apply_loss_mask(x0, pred_x0, mask, use_mask)
        img = ((prd - tgt) ** 2).flatten(1).mean(dim=1)       # (B,) per-sample image MSE
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


# ---------------------------------------------------------------------------
# the latent composition  (encode -> regress -> decode), all frozen except R
# ---------------------------------------------------------------------------
def encode(D, img, sigma=0.0):
    """Frozen encode of a (clean) image into D's M-channel sparse code z. The clean endpoints
    are encoded at sigma=0 (the bridge stochasticity is injected later, in latent space)."""
    H, W = img.shape[-2:]
    if H % D.sc or W % D.sc:
        raise ValueError(
            f"Input {H}x{W} must be divisible by the dictionary stride sc={D.sc} (the decode "
            f"calls D_t1ce.D directly, bypassing GroupCDL.forward's internal pad/crop). Set "
            f"crop_size / center_crop to a multiple of {D.sc}.")
    with torch.no_grad():
        _, z = D(img, E=Identity(), sigma=sigma)
    return z.detach()


def decode(D_t1ce, z0_hat, dc=None):
    """Frozen decode of a predicted code back to the T1ce image via the synthesis dictionary
    D = B[0] (a transposed convolution). Gradients flow through this fixed operator into R."""
    x = D_t1ce.D(z0_hat)
    if dc is not None:
        x = x + dc
    return x


def _decode_dc(x1, decode_dc):
    """DC term re-added at decode. 'x1_mean' = per-image mean of the prior (available at train
    AND inference); 'none' = 0. Mirrors GroupCDL's own mean(dim=(1,2,3)) convention."""
    if decode_dc in (None, "none"):
        return None
    if decode_dc == "x1_mean":
        return x1.mean(dim=(1, 2, 3), keepdim=True)
    raise ValueError(f"unknown decode_dc {decode_dc!r} (use 'x1_mean' or 'none')")


def latent_regress(R, zt, cond_lat, sigma, M):
    """R's latent -> latent step: E[z0 | zt (, z1)]. cond_lat is z1 (conditional) or None. We
    keep the first M channels of the reconstruction as the z0 estimate (the zt slot)."""
    net_in = zt if cond_lat is None else torch.cat([zt, cond_lat], dim=1)
    recon, _ = R(net_in, E=Identity(), sigma=sigma)
    return recon[:, :M]


@torch.no_grad()
def latent_i2sb_sample(D_joint, R, D_t1ce, x1, cond, bridge, *, conditional, M, nfe=20,
                       ot_ode=False, decode_dc="x1_mean", verbose=False):
    """Reverse sampling z1 -> z0 IN THE LATENT DOMAIN, then a single decode to the T1ce image.
    Reuses diffusion/i2sb.py:ddpm_sampling (space-agnostic) with z1 as the start; only the final
    latent is decoded. Returns the T1ce estimate (B, 1, H, W)."""
    z1 = encode(D_joint, torch.cat([x1, cond], dim=1))
    cond_lat = z1 if conditional else None
    interval = n_steps(bridge)
    nfe = nfe or interval - 1
    assert 0 < nfe < interval
    steps = space_indices(interval, nfe + 1)

    def pred_x0_fn(zt, step):
        step_t = torch.full((zt.shape[0],), step, device=bridge.device, dtype=torch.long)
        std_fwd = get_std_fwd(bridge, step_t, xdim=zt.shape[1:])
        return latent_regress(R, zt, cond_lat, std_fwd, M)

    zs, _ = ddpm_sampling(bridge, steps, pred_x0_fn, z1, ot_ode=ot_ode,
                          log_steps=[0], verbose=verbose)
    z0_hat = zs[:, 0].to(bridge.device)
    return decode(D_t1ce, z0_hat, dc=_decode_dc(x1, decode_dc))


# ---------------------------------------------------------------------------
# sanity: the shared-latent invariants the whole design rests on
# ---------------------------------------------------------------------------
def _assert_latent_shapes(D_joint, R, D_t1ce, loader, target_channels, conditional):
    n_cond = 0
    try:
        n_cond = int(next(iter(loader))[2].shape[1])
    except Exception:
        pass

    expected_joint_C = target_channels + n_cond
    if D_joint.C != expected_joint_C:
        raise ValueError(
            f"Joint dict C={D_joint.C} but data gives {n_cond} cond channel(s) so it needs "
            f"C={expected_joint_C} (= target_channels {target_channels} + n_cond). Check the "
            f"joint dict's config and the data cond_idx.")
    if D_t1ce.C != target_channels:
        raise ValueError(f"T1ce dict C={D_t1ce.C} != target_channels {target_channels}.")
    if D_joint.M != D_t1ce.M or D_joint.sc != D_t1ce.sc:
        raise ValueError(
            f"Dictionaries do not share a latent shape: joint (M={D_joint.M}, sc={D_joint.sc}) "
            f"vs t1ce (M={D_t1ce.M}, sc={D_t1ce.sc}). Retrain them with identical M and sc.")
    # R bridges IN that latent: input width is M (unconditional) or 2M (conditional, cat[zt, z1])
    expected_R_C = 2 * D_joint.M if conditional else D_joint.M
    if R.C != expected_R_C:
        raise ValueError(
            f"Regressor R.C={R.C} but conditional={conditional} needs R.C={expected_R_C} "
            f"({'2*M for cat[zt, z1]' if conditional else 'M for zt only'}, M={D_joint.M}). "
            f"Set the R model C accordingly.")


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
    latent_weight=1.0,               # weight on the z0 MSE term (latent / mixed)
    image_weight=1.0,                # weight on the decoded-image loss term (image / mixed)
    loss_weight="uniform",           # per-sample t-weighting: "uniform" | "snr" (~t=0) | "t1" (~t=1)
    psnr_only=False,
    # ---- I2SB method (cfg["i2sb"]) ----
    bridge_type="brownian",
    tau=0.19,                        # NOTE: latent-scale; re-tune (image-domain value is wrong)
    n_points=1000,
    bridge_shape="constant",
    beta_max=0.3,
    ot_ode=False,
    parameterization="x0",
    conditional=True,                # R sees cat[zt, z1] (True, R.C=2M) or zt only (False, R.C=M)
    ema_decay=0.99,
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
    if parameterization != "x0":
        raise ValueError(
            f"latent I2SB supports parameterization='x0' only (the decoded output IS the T1ce "
            f"estimate; the loss is image-domain). Got {parameterization!r}.")
    if joint_dict_config is None or t1ce_dict_config is None:
        raise ValueError("cfg['dicts'] must give joint_dict_config and t1ce_dict_config "
                         "(paths to each frozen dictionary's saved config.json).")
    if loss_type not in LOSS_REGISTRY:
        raise ValueError(f"loss_type {loss_type!r} not in LOSS_REGISTRY {sorted(LOSS_REGISTRY)}.")
    if loss_mode not in ("latent", "image", "mixed"):
        raise ValueError(f"loss_mode {loss_mode!r} must be 'latent', 'image', or 'mixed'.")
    loss_fn = LOSS_REGISTRY[loss_type]

    R.to(device)
    R.train()

    # ---- frozen convolutional dictionaries ----
    D_joint = _load_frozen_dict(joint_dict_config, device, backend=dict_backend)
    D_t1ce = _load_frozen_dict(t1ce_dict_config, device, backend=dict_backend)
    M = D_joint.M
    print(f"[latent-i2sb] dicts: joint C={D_joint.C} M={D_joint.M} sc={D_joint.sc} | "
          f"t1ce C={D_t1ce.C} M={D_t1ce.M} sc={D_t1ce.sc} | R C={R.C} M={R.M} attn={R.attn_backend} "
          f"| dict_attn={D_joint.attn_backend} conditional={conditional} "
          f"loss={loss_mode}(img={loss_type}, λz={latent_weight}, λx={image_weight})")
    _assert_latent_shapes(D_joint, R, D_t1ce, train_loader, target_channels, conditional)

    # LATENT bridge schedule. tau/std_sb are now in latent units -- almost certainly needs
    # re-tuning to the code magnitude; the image-domain value is only a placeholder.
    bridge = build_bridge(bridge_type=bridge_type, n_points=n_points, device=device,
                          tau=tau, shape=bridge_shape, beta_max=beta_max)
    interval = n_steps(bridge)
    ema = EMA(R.parameters(), decay=ema_decay)

    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, "net.ckpt")
    ema_path = os.path.join(save_dir, "ema.pt")

    if start_epoch > 0 and os.path.exists(ema_path):
        ema.load_state_dict(torch.load(ema_path, map_location=device))
        print(f"[latent-i2sb] resumed EMA from {ema_path}")

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

            # ----- encode endpoints, sample a LATENT bridge point, regress -----
            z0 = encode(D_t1ce, x0)                           # target latent  (frozen)
            z1 = encode(D_joint, torch.cat([x1, cond], dim=1))  # prior latent (frozen)
            cond_lat = z1 if conditional else None

            b = x0.shape[0]
            step = torch.randint(0, interval, (b,), device=device)
            zt = q_sample(bridge, step, z0, z1, ot_ode=ot_ode)      # latent interpolant + noise
            std_fwd = get_std_fwd(bridge, step, xdim=z0.shape[1:])  # (B,1,1,1) bridge level

            opt.zero_grad()
            z0_hat = latent_regress(R, zt, cond_lat, std_fwd, M)    # (B, M, Q1, Q2)
            pred_x0 = decode(D_t1ce, z0_hat, dc=_decode_dc(x1, decode_dc))   # (B, 1, N1, N2) for metrics
            loss = _combine_loss(loss_fn, loss_mode, latent_weight, image_weight, loss_weight,
                                 z0, z0_hat, x0, pred_x0, mask, use_mask, std_fwd)

            loss.backward()
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(R.parameters(), clip_grad)
            opt.step()
            if hasattr(R, "project"): R.project()      # only R is trained; dicts stay frozen
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
                R, opt, sched, _ = load_ckpt(ckpt_path, model=R, optimizer=opt,
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
            save_ckpt_fn(ckpt_path, model=R, optimizer=opt, scheduler=sched, step=global_step)
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
                D_joint, R, D_t1ce, ema, bridge, val_loader, device, M=M, interval=interval,
                loss_fn=loss_fn, val_mode=val_mode, val_seed=val_seed, use_mask=use_mask,
                ot_ode=ot_ode, conditional=conditional, decode_dc=decode_dc, val_nfe=val_nfe,
                psnr_only=psnr_only, loss_mode=loss_mode, latent_weight=latent_weight,
                image_weight=image_weight, loss_weight=loss_weight, wandb=wandb, global_step=global_step,
            )
            if isinstance(sched, ReduceLROnPlateau) and val_loss is not None:
                sched.step(val_loss)

    pbar.close()
    return R


@torch.no_grad()
def _validate(D_joint, R, D_t1ce, ema, bridge, val_loader, device, *, M, interval, loss_fn,
              val_mode, val_seed, use_mask, ot_ode, conditional, decode_dc, val_nfe, psnr_only,
              loss_mode, latent_weight, image_weight, loss_weight, wandb, global_step):
    """Validate with EMA weights on R. Two modes mirror train_i2sb:
      "single_pass" -- one random latent-bridge step, one latent regression + decode (matches
                       training; cheap). "full_recon" -- the val_nfe-step latent reverse bridge
                       with a single final decode (end-to-end synthesis quality)."""
    R.eval()
    agg = {"loss": 0.0, "psnr": 0.0, "ssim": 0.0, "nrmse": 0.0}
    n_samples = 0
    last = None
    gen = torch.Generator(device=device).manual_seed(val_seed) if val_seed is not None else None

    with ema.average_parameters():
        for batch in val_loader:
            x0, x1, cond, mask = _split_batch(batch, device)
            bs = x0.shape[0]

            if val_mode == "full_recon":
                pred = latent_i2sb_sample(D_joint, R, D_t1ce, x1, cond, bridge,
                                          conditional=conditional, M=M, nfe=val_nfe,
                                          ot_ode=ot_ode, decode_dc=decode_dc, verbose=False)
                sigma = torch.zeros(bs, 1, 1, 1, device=device)     # end-to-end: no single step
                # full_recon has no single z0_hat -> report the image-domain recon error regardless
                # of loss_mode (this is the end-to-end quality the LR scheduler should track).
                loss = _apply_loss(loss_fn, x0, pred, mask, use_mask, sigma)
                xt, step = x1, None
            else:  # single_pass
                z0 = encode(D_t1ce, x0)
                z1 = encode(D_joint, torch.cat([x1, cond], dim=1))
                cond_lat = z1 if conditional else None
                step = torch.randint(0, interval, (bs,), generator=gen, device=device)
                zt = q_sample(bridge, step, z0, z1, ot_ode=ot_ode)
                std_fwd = get_std_fwd(bridge, step, xdim=z0.shape[1:])
                z0_hat = latent_regress(R, zt, cond_lat, std_fwd, M)
                pred = decode(D_t1ce, z0_hat, dc=_decode_dc(x1, decode_dc))
                loss = _combine_loss(loss_fn, loss_mode, latent_weight, image_weight, loss_weight,
                                     z0, z0_hat, x0, pred, mask, use_mask, std_fwd)
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
