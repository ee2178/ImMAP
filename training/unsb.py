"""
Unpaired Neural Schrodinger Bridge training loop (sb/unsb.py, models/unsb_nets.py).

Same skeleton as train_i2sb -- an epoch is `steps_per_epoch` optimizer steps, the regressor `net`
and its optimizer / scheduler come from train.py, cosine-style schedulers step once per optimizer
step -- but the step is the adversarial one:

    x0, x1, cond <- batch                    (x0 = CT1 target, x1 = T1 source; cond = FLAIR/T1/T2/...)
    stage j ~ U{0..K-1}
    x_t   = chain(x1, j stages)              no grad: the model's OWN chain, not forward_sample
    x_hat = G(x_t, step_j, cond)             the one grad-carrying call
    D step   LSGAN(D(x0_real), D(x_hat.detach()))          D sees (image, cond) when cond_d
    E step   MINE critic on (x_t, x_hat) pairs             only if `entropy`
    G step   lambda_gan * adv + lambda_sb * tau * (transport - (1-t) * ET) + lambda_nce * PatchNCE

`x0` is used as the REAL sample for D and never compared with `x_hat` pixelwise, which is the whole
point on T1/CT1 pairs that are not well registered. `pairing: "shuffle"` goes further and gives D a
real CT1 from a DIFFERENT case in the batch (with that case's own conditioning).

WHAT IS DELIBERATELY DIFFERENT FROM train_i2sb
  * NO loss backtracking and NO best-loss checkpoint. An adversarial G loss is not monotone -- it
    RISES when D gets better -- so "loss went up, restore the old weights" would undo progress.
    Instead: `net.ckpt` (+ `unsb_nets.ckpt`) is the resumable latest state, written every
    `save_every_epochs`; `net_best.ckpt` is the validated weights (EMA if on) with the best val PSNR.
    A non-finite loss aborts rather than being silently skipped. Caveat: val PSNR is measured against
    the real (imperfectly registered) CT1, so it is a sanity check on selection, not a quality oracle.
  * Critics/NCE state lives OUTSIDE net.ckpt, in `unsb_nets.ckpt` (D, E, the PatchEncoder/PatchNCE,
    and their optimizers). Losing it on a requeue would restart the adversarial game, so resume
    loads it and says so when it is missing.
  * The critics' learning rates FOLLOW the regressor's schedule (same ratio to their base lr), so a
    cosine-annealed G is never facing a constant-lr D.
  * No gradient accumulation and no learned DC; the entropy critic needs batch >= 2.

LOGGING follows the repo preference: val/psnr, val/ssim, val/nrmse and IMAGES. Three panels: the
chain (source | each stage's prediction | GT), the residual, and -- when conditioning exists -- the
cond-swap panel: the same source translated with its own conditioning and with another patient's
(same noise), the direct check that G uses what D is supposed to make it use. The adversarial loss
terms are OFF unless `log_gan_terms: true` (then logged under gan/), because a GAN is the one place
you do want to see D vs G.
"""

import contextlib
import math
import os
import time

import torch
import torchvision.utils as vutils
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sb.base import build_schedule, n_steps
from sb.unsb import (unsb_steps, sample_stage, make_g_fn, unsb_forward, unsb_sample, d_loss,
                     e_loss, g_loss, make_nce_fn, set_requires_grad)
from models.unsb_nets import build_critics, PatchEncoder, FeatureTap, PatchNCE
from training.common import save_ckpt, load_ckpt, load_ckpt_meta, EMA, apply_loss_mask, POSTFIX_EVERY
from training.i2sb import _split_batch, _assert_batch_matches_config, DISPLAY_VMIN, DISPLAY_VMAX
from training.metrics import compute_metrics
from visualization.image import orient_tensor
from visualization.wandb_image import wandb_image


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def _seeded(seed, device):
    """Run a block with the RNG forked and (if `seed` is given) seeded, then put the training RNG
    back untouched. Validation noise is then comparable between epochs, and validating never
    perturbs the training stream."""
    if seed is None:
        yield
        return
    dev = torch.device(device)
    devices = []
    if dev.type == "cuda":
        devices = [dev.index if dev.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        yield


def _group_params(*mods):
    return [p for m in mods if m is not None for p in m.parameters()]


def _sync_lrs(opt, base_lr_g, others):
    """Scale every (optimizer, base_lr) in `others` by the regressor's CURRENT lr / its base lr."""
    g = opt.param_groups[0]
    ratio = g["lr"] / max(float(base_lr_g), 1e-30)
    for o, base in others:
        for pg in o.param_groups:
            pg["lr"] = base * ratio


def _assert_unsb_batch(net, loader, device, target_channels, entropy):
    """Peek one batch; fail loudly on the mismatches that would otherwise train the wrong thing.
    Returns (n_cond, example) with example = (x1, cond, guide) for building the NCE tap."""
    batch = next(iter(loader))
    x0, x1, cond, mask, et, guide = _split_batch(batch, device)
    if entropy and x1.shape[0] < 2:
        raise ValueError("unsb.entropy needs data.train.batch_size >= 2 (the entropy critic draws "
                         "its negatives from the rest of the batch). Raise the batch size or set "
                         "entropy: false.")
    prior_idx = getattr(net, "prior_idx", None)
    if prior_idx is not None:
        # SB-CDL nets debias with mu_1 * x_1 read from cond[prior_idx]. If that channel is not the
        # source the chain starts from, every step is debiased against the wrong image -- silently.
        if cond is None or not torch.allclose(cond[:, prior_idx:prior_idx + 1], x1, atol=1e-5):
            raise ValueError(
                f"{type(net).__name__}.prior_idx={prior_idx} must index the SOURCE image x1 inside "
                f"cond, but cond[:, {prior_idx}] != x1. Set model.params.prior_idx to the position "
                f"of x1_idx within data.*.cond_idx.")
    return (0 if cond is None else int(cond.shape[1])), (x1, cond, guide)


def _panel(cols, lo, hi, mask, orient):
    grid = torch.cat([((c - lo) / (hi - lo)).clamp(0, 1) for c in cols], dim=0)
    if mask is not None:
        grid = mask * grid
    return orient_tensor(grid, orient)


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------
def train_unsb(
    net, opt, sched, device,
    train_loader,
    val_loader,
    wandb=None,
    start_epoch=0,
    # ---- generic loop (cfg["training"]) ----
    num_epochs=300,
    steps_per_epoch=200,
    val_every_epochs=10,
    save_every_epochs=5,
    clip_grad=1.0,
    use_mask=False,
    psnr_only=False,
    val_slices=None,
    display_window=None,
    display_orient=None,
    display_mask=None,
    data_range=1.0,
    # ---- bridge + chain (cfg["unsb"]) ----
    kind="brownian",                 # schedule, as in the i2sb block. tau_paper = (2 * tau)**2
    tau=0.1,
    n_points=1000,
    beta_max=0.3,
    num_stages=5,                    # K: G evaluations per sample. Train and sample with the SAME K.
    grid="uniform",                  # "uniform" (paper text) | "official" (the released code's clock)
    train_noise_scale=1.0,           # bridge-noise multiplier inside the TRAINING chain
    deterministic=False,             # drop the bridge noise at validation sampling
    target_channels=1,
    # ---- discriminator / entropy critic ----
    cond_d=True,                     # THE conditioning toggle: does D see cond? (G always does)
    pairing="same",                  # real CT1 for D: "same" case as the source | "shuffle" another
    entropy=True,                    # entropy critic E (needs batch >= 2); false = transport only
    ndf=64,
    n_layers_d=3,
    spectral_norm=False,
    lr_d=None,                       # None -> the regressor's base lr
    lr_e=None,
    betas_d=(0.5, 0.999),
    # ---- loss weights ----
    lambda_gan=1.0,
    lambda_sb=1.0,
    lambda_nce=1.0,
    # ---- PatchNCE (structure). nce=false -> no structure term ----
    nce=False,
    nce_extractor="encoder",         # "encoder" (PatchEncoder: any G) | "tap" (G's own named layers)
    nce_layers=None,                 # tap only, e.g. ["input_blocks.2", "input_blocks.5"]
    nce_ch=32,
    nce_levels=3,
    nce_dim=256,
    nce_patches=256,
    nce_temperature=0.07,
    nce_mask=True,                   # sample patches inside the brain mask
    # ---- EMA of the regressor ----
    ema_decay=None,
    ema_warmup=True,
    # ---- validation ----
    val_nfe=None,                    # None -> num_stages
    val_seed=0,
    val_swap_cond=True,              # the cond-swap panel
    log_gan_terms=False,
    # ---- paths ----
    save_dir=None,
    ckpt=None,
    **_unused,
):
    net.to(device)
    net.train()
    if isinstance(sched, ReduceLROnPlateau):
        raise ValueError("train_unsb does not support ReduceLROnPlateau: there is no monotone "
                         "validation loss to plateau on (an adversarial loss is not one).")
    if pairing not in ("same", "shuffle"):
        raise ValueError(f"pairing {pairing!r} must be 'same' or 'shuffle'")
    if nce and nce_extractor not in ("encoder", "tap"):
        raise ValueError(f"nce_extractor {nce_extractor!r} must be 'encoder' or 'tap'")
    if nce and nce_extractor == "tap" and not nce_layers:
        raise ValueError("nce_extractor='tap' needs nce_layers (names from net.named_modules())")
    if lambda_gan <= 0:
        raise ValueError("lambda_gan must be > 0: the discriminator is the only term that defines "
                         "the target contrast")

    bridge = build_schedule(kind=kind, tau=tau, n_points=n_points, beta_max=beta_max, device=device)
    if hasattr(net, "assert_schedule_matches"):
        net.assert_schedule_matches(bridge)
    steps = unsb_steps(bridge, num_stages, grid)
    val_nfe = num_stages if val_nfe is None else int(val_nfe)

    _assert_batch_matches_config(net, train_loader, device, target_channels, 1.0, "mse")
    n_cond, (ex_x1, ex_cond, ex_guide) = _assert_unsb_batch(
        net, train_loader, device, target_channels, entropy)
    if cond_d and n_cond == 0:
        print("[unsb] cond_d=True but the loader provides no conditioning: D will be unconditional")
        cond_d = False

    # ----- critics, NCE extractor, and their optimizers ------------------------------------
    D, E = build_critics(target_channels, n_cond, cond_d=cond_d, entropy=entropy, ndf=ndf,
                         n_layers=n_layers_d, spectral_norm=spectral_norm)
    D.to(device)
    E = None if E is None else E.to(device)

    extractor = pnce = None
    if nce:
        if nce_extractor == "encoder":
            extractor = PatchEncoder(target_channels, ch=nce_ch, n_levels=nce_levels).to(device)
            channels = extractor.channels()
        else:
            extractor = FeatureTap(net, nce_layers)
            sigma0 = torch.zeros(ex_x1.shape[0], 1, 1, 1, device=device)
            channels = extractor.channels(ex_x1, ex_cond, sigma0)
        pnce = PatchNCE(channels, nc=nce_dim, num_patches=nce_patches,
                        temperature=nce_temperature).to(device)

    g_group = opt.param_groups[0]
    base_lr_g = float(g_group.get("initial_lr", g_group["lr"]))
    lr_d = base_lr_g if lr_d is None else float(lr_d)
    lr_e = base_lr_g if lr_e is None else float(lr_e)
    opt_d = torch.optim.Adam(D.parameters(), lr=lr_d, betas=tuple(betas_d))
    opt_e = None if E is None else torch.optim.Adam(E.parameters(), lr=lr_e, betas=tuple(betas_d))
    # The encoder (if own-weights) and PatchNCE's MLPs are trained by the NCE loss alone, alongside
    # the regressor, on its schedule and its Adam betas. A FeatureTap adds no parameters.
    aux_params = _group_params(pnce, extractor if isinstance(extractor, PatchEncoder) else None)
    opt_aux = (torch.optim.Adam(aux_params, lr=base_lr_g, betas=tuple(opt.defaults.get("betas", (0.9, 0.999))))
               if aux_params else None)
    lr_followers = [(opt_d, lr_d)] + ([(opt_e, lr_e)] if opt_e else []) \
        + ([(opt_aux, base_lr_g)] if opt_aux else [])

    ema = None
    if ema_decay:
        ema = EMA(net.parameters(), decay=float(ema_decay), use_num_updates=bool(ema_warmup))
        print(f"[unsb] EMA decay={float(ema_decay)} over {len(ema.shadow)} tensors -- validation and "
              f"net_best.ckpt use the average; net.ckpt keeps the live weights for resume")

    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, "net.ckpt")
    nets_path = os.path.join(save_dir, "unsb_nets.ckpt")
    best_path = os.path.join(save_dir, "net_best.ckpt")

    best_psnr = -float("inf")
    if os.path.exists(best_path):
        _, b = load_ckpt_meta(best_path)
        best_psnr = -b if math.isfinite(b) else -float("inf")

    def save_nets():
        # critics FIRST: a kill between the two writes then leaves net.ckpt (what train.py resumes
        # from) one save OLDER than the critics, never newer.
        state = {"D": D.state_dict(), "opt_d": opt_d.state_dict(),
                 "E": None if E is None else E.state_dict(),
                 "opt_e": None if opt_e is None else opt_e.state_dict(),
                 "pnce": None if pnce is None else pnce.state_dict(),
                 "encoder": extractor.state_dict() if isinstance(extractor, PatchEncoder) else None,
                 "opt_aux": None if opt_aux is None else opt_aux.state_dict(),
                 "meta": {"n_cond": n_cond, "cond_d": bool(cond_d), "entropy": bool(entropy),
                          "nce": bool(nce), "nce_extractor": nce_extractor if nce else None}}
        tmp = nets_path + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, nets_path)

    def load_nets():
        st = torch.load(nets_path, map_location=device, weights_only=False)
        m = st["meta"]
        want = {"n_cond": n_cond, "cond_d": bool(cond_d), "entropy": bool(entropy),
                "nce": bool(nce), "nce_extractor": nce_extractor if nce else None}
        if m != want:
            raise ValueError(f"{nets_path} was written for {m} but this run is {want}; the critics "
                             f"would not fit. Use a new save_dir, or restore the matching config.")
        D.load_state_dict(st["D"]); opt_d.load_state_dict(st["opt_d"])
        if E is not None:
            E.load_state_dict(st["E"]); opt_e.load_state_dict(st["opt_e"])
        if pnce is not None:
            pnce.load_state_dict(st["pnce"])
        if isinstance(extractor, PatchEncoder):
            extractor.load_state_dict(st["encoder"])
        if opt_aux is not None:
            opt_aux.load_state_dict(st["opt_aux"])

    if start_epoch > 0:
        resume_from = ckpt if ckpt else ckpt_path
        if ema is not None and os.path.exists(resume_from):
            load_ckpt(resume_from, ema=ema, device=device)
        if os.path.exists(nets_path):
            load_nets()
            print(f"[unsb] resumed D{'/E' if E is not None else ''}{'/NCE' if nce else ''} "
                  f"from {nets_path}")
        else:
            print(f"[unsb] WARNING: resuming at epoch {start_epoch} but {nets_path} is missing; "
                  f"the critics restart from scratch and the adversarial game restarts with them")

    if val_loader is not None and val_slices:
        from training.forward_op import fixed_val_subset
        n_full = len(val_loader.dataset)
        val_loader = fixed_val_subset(val_loader, int(val_slices),
                                      0 if val_seed is None else int(val_seed))
        print(f"[unsb] validating on a fixed subset: {len(val_loader.dataset)}/{n_full} val slices")

    print(f"[unsb] K={num_stages} stages on grid {grid!r} (bridge steps {steps}); tau_paper="
          f"{float(bridge.std_fwd[-1]) ** 2:.4g}; cond_d={cond_d} pairing={pairing} entropy={entropy} "
          f"nce={nce_extractor if nce else 'off'}; lambdas gan={lambda_gan} sb={lambda_sb} "
          f"nce={lambda_nce if nce else 0}")

    train_iter = iter(train_loader)
    pbar = tqdm(total=num_epochs * steps_per_epoch, initial=start_epoch * steps_per_epoch,
                desc="UNSB", dynamic_ncols=True)
    g_params = list(net.parameters()) + aux_params

    for epoch in range(start_epoch, num_epochs):
        net.train()
        running = {}                                       # on-device sums; ONE sync per epoch
        n_batches = 0

        for _ in range(steps_per_epoch):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            x0, x1, cond, mask, et, guide = _split_batch(batch, device)

            # ----- one on-policy sample of the chain -----
            g_fn = make_g_fn(net, bridge, x1, cond=cond, target_channels=target_channels,
                             guide=guide)
            state = unsb_forward(g_fn, bridge, x1, steps, sample_stage(num_stages),
                                 noise_scale=train_noise_scale)

            if pairing == "shuffle" and x0.shape[0] > 1:       # a real CT1 from ANOTHER case
                x0_real = x0.roll(1, dims=0)
                cond_real = None if cond is None else cond.roll(1, dims=0)
            else:
                x0_real, cond_real = x0, cond

            # ----- D (and E) on the detached prediction -----
            set_requires_grad(D, True)
            opt_d.zero_grad(set_to_none=True)
            loss_d = d_loss(D, state, x0_real, cond=cond, cond_real=cond_real, cond_d=cond_d)
            loss_d.backward()
            opt_d.step()
            loss_e = None
            if E is not None:
                set_requires_grad(E, True)
                opt_e.zero_grad(set_to_none=True)
                loss_e = e_loss(E, state)
                loss_e.backward()
                opt_e.step()

            # ----- G (+ NCE head), with the critics frozen so gradients reach only the generator -----
            set_requires_grad([m for m in (D, E) if m is not None], False)
            opt.zero_grad(set_to_none=True)
            if opt_aux is not None:
                opt_aux.zero_grad(set_to_none=True)
            nce_fn = (make_nce_fn(extractor, pnce, bridge, cond=cond,
                                  mask=mask if nce_mask else None) if nce else None)
            total, terms = g_loss(D, E, state, bridge, x1, cond=cond, nce=nce_fn, cond_d=cond_d,
                                  lambda_gan=lambda_gan, lambda_sb=lambda_sb, lambda_nce=lambda_nce)
            total.backward()
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(g_params, clip_grad)
            opt.step()
            if opt_aux is not None:
                opt_aux.step()
            if ema is not None:
                ema.update()
            if hasattr(net, "project"):
                net.project()                              # the unrolled nets' nonnegativity clamps
            if sched is not None:
                sched.step()
            _sync_lrs(opt, base_lr_g, lr_followers)

            vals = {"g": total.detach(), "d": loss_d.detach(), **terms}
            if loss_e is not None:
                vals["e"] = loss_e.detach()
            for k, v in vals.items():
                running[k] = running.get(k, 0.0) + v
            n_batches += 1
            pbar.update(1)
            if n_batches % POSTFIX_EVERY == 0:
                pbar.set_postfix(g=f"{total.item():.3e}", d=f"{loss_d.item():.3e}", epoch=epoch)

        global_step = (epoch + 1) * steps_per_epoch
        means = {k: float(v) / max(n_batches, 1) for k, v in running.items()}     # the ONE sync
        if not all(math.isfinite(v) for v in means.values()):
            raise RuntimeError(
                f"[epoch {epoch}] non-finite loss {means}. Adversarial training has no loss "
                f"backtracking (a rising G loss is normal), so this is a hard stop; resume from "
                f"{ckpt_path} with a lower lr or spectral_norm: true.")

        x0_m, hat_m = apply_loss_mask(x0, state.x_hat.detach(), mask, use_mask)
        train_metrics = {k: float(v.detach()) for k, v in
                         compute_metrics(x0_m, hat_m, psnr_only=psnr_only,
                                         data_range=data_range).items()}

        if wandb:
            log = {"train/loss": means["g"], "train/lr": opt.param_groups[0]["lr"],
                   "train/epoch": epoch, **{f"train/{k}": v for k, v in train_metrics.items()}}
            if log_gan_terms:
                log.update({f"gan/{k}": v for k, v in means.items()})
            wandb.log(log, step=global_step)
        else:
            print({"epoch": epoch, "g": means["g"], "d": means["d"], **train_metrics})

        # ----- resumable state (not "best": see the module docstring) -----
        if save_every_epochs and (epoch + 1) % save_every_epochs == 0:
            save_nets()
            save_ckpt(ckpt_path, model=net, optimizer=opt, scheduler=sched, step=global_step,
                      ema=ema)

        # ----- validation + best-by-PSNR -----
        if val_loader is not None and val_every_epochs and (epoch + 1) % val_every_epochs == 0:
            t_val = time.time()
            val_ctx = ema.average_parameters() if ema is not None else contextlib.nullcontext()
            with val_ctx:
                mets = _validate(
                    net, bridge, val_loader, device, num_stages=num_stages, grid=grid,
                    nfe=val_nfe, deterministic=deterministic, target_channels=target_channels,
                    use_mask=use_mask, psnr_only=psnr_only, data_range=data_range, wandb=wandb,
                    global_step=global_step, display_window=display_window,
                    display_orient=display_orient, display_mask=display_mask, seed=val_seed,
                    swap_cond=val_swap_cond)
                if mets["psnr"] > best_psnr:
                    best_psnr = mets["psnr"]
                    # inside the EMA context: the saved weights are the VALIDATED ones
                    save_ckpt(best_path, model=net, step=global_step, best_loss=-best_psnr)
                    print(f"[epoch {epoch}] new best val psnr {best_psnr:.3f} -> {best_path}")
            val_sec = time.time() - t_val
            print(f"[epoch {epoch}] validation took {val_sec:.0f}s on {len(val_loader.dataset)} slices")
            if wandb:
                wandb.log({"val/seconds": val_sec}, step=global_step)

    pbar.close()
    save_nets()
    save_ckpt(ckpt_path, model=net, optimizer=opt, scheduler=sched,
              step=num_epochs * steps_per_epoch, ema=ema)
    return net


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def _swap_cond(cond, other, keep_idx):
    """Another patient's conditioning, keeping the SOURCE channel (SB-CDL nets read x1 out of cond
    at `prior_idx`; swapping that too would contradict the chain's own start state)."""
    out = other.clone()
    if keep_idx is not None:
        out[:, keep_idx] = cond[:, keep_idx]
    return out


@torch.no_grad()
def _validate(net, bridge, val_loader, device, *, num_stages, grid, nfe, deterministic,
              target_channels, use_mask, psnr_only, data_range, wandb, global_step,
              display_window=None, display_orient=None, display_mask=None, seed=0,
              swap_cond=True):
    """Full K-stage (or `nfe`-stage) translation of every val slice, scored against the real CT1.

    The metrics are against an imperfectly registered GT, so read them as a sanity check and a
    selection signal, not as the objective -- the UNSB generator was never asked to match that
    image pixelwise."""
    net.eval()
    agg = {"psnr": 0.0, "ssim": 0.0, "nrmse": 0.0}
    n_samples = 0
    last = probe = other = None
    keep_idx = getattr(net, "prior_idx", None)

    for batch in val_loader:
        x0, x1, cond, mask, et, guide = _split_batch(batch, device)
        bs = x0.shape[0]
        with _seeded(seed, device):
            recon, _, hats = unsb_sample(net, x1, bridge, cond=cond, num_stages=num_stages,
                                         nfe=nfe, grid=grid, deterministic=deterministic,
                                         target_channels=target_channels, guide=guide)
        x0_m, rec_m = apply_loss_mask(x0, recon, mask, use_mask)
        mets = compute_metrics(x0_m, rec_m, psnr_only=psnr_only, data_range=data_range)
        for k in agg:
            if k in mets:
                agg[k] += float(mets[k].detach()) * bs
        n_samples += bs
        last = (x1, cond, guide, x0, recon, hats, mask)

        if swap_cond and cond is not None:                  # collect a first sample + another's cond
            if probe is None:
                probe = (x1[:1], cond[:1], None if guide is None else guide[:1])
                if bs > 1:
                    other = (cond[1:2], None if guide is None else guide[1:2])
            elif other is None:
                other = (cond[:1], None if guide is None else guide[:1])

    mean_metrics = {k: v / max(n_samples, 1) for k, v in agg.items() if n_samples}

    if wandb and last is not None:
        x1, cond, guide, x0, recon, hats, mask = last
        show_masked = use_mask if display_mask is None else bool(display_mask)
        lo, hi = ((DISPLAY_VMIN, DISPLAY_VMAX) if display_window is None
                  else (float(display_window[0]), float(display_window[1])))
        m1 = mask[:1].cpu() if show_masked else None
        chain = hats[:1].flip(1)[0]                          # (nfe, C, H, W), source -> target order
        cols = [x1[:1].cpu(), *[c[None] for c in chain], x0[:1].cpu()]
        res = (x0[:1] - recon[:1]).abs().cpu()
        res = res / res.max().clamp(min=1e-8)
        log = {
            "val/example": wandb_image(
                vutils.make_grid(_panel(cols, lo, hi, m1, display_orient), nrow=len(cols)),
                caption=f"x1 (source) | stage 1..{chain.shape[0]} predictions | CT1 GT "
                        f"(nfe={chain.shape[0]})"),
            "val/residual": wandb_image(
                vutils.make_grid(orient_tensor(res, display_orient), nrow=1),
                caption="| GT - recon |  (GT is imperfectly registered)"),
            **{f"val/{k}": v for k, v in mean_metrics.items()},
        }
        if swap_cond and probe is not None and other is not None:
            p_x1, p_cond, p_guide = probe
            o_cond, o_guide = other
            sw_cond = _swap_cond(p_cond, o_cond, keep_idx)
            with _seeded(seed, device):
                own, _, _ = unsb_sample(net, p_x1, bridge, cond=p_cond, num_stages=num_stages,
                                        nfe=nfe, grid=grid, deterministic=deterministic,
                                        target_channels=target_channels, guide=p_guide)
            with _seeded(seed, device):                      # same noise: only the cond differs
                swp, _, _ = unsb_sample(net, p_x1, bridge, cond=sw_cond, num_stages=num_stages,
                                        nfe=nfe, grid=grid, deterministic=deterministic,
                                        target_channels=target_channels,
                                        guide=o_guide if o_guide is not None else p_guide)
            diff = (swp - own).abs().cpu()
            diff = diff / diff.max().clamp(min=1e-8)
            log["val/cond_swap"] = wandb_image(
                vutils.make_grid(torch.cat([
                    _panel([p_x1.cpu(), own.cpu(), swp.cpu()], lo, hi, None, display_orient),
                    orient_tensor(diff, display_orient)]), nrow=4),
                caption="source | own cond | ANOTHER patient's cond (same noise) | |difference|. "
                        "A black difference = G ignores its conditioning.")
        wandb.log(log, step=global_step)
    elif not wandb:
        print("[VAL] " + " ".join(f"{k}={v:.4f}" for k, v in mean_metrics.items()))

    net.train()
    return mean_metrics
