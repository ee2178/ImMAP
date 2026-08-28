#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Finetune RAM on fastMRI multicoil brain / knee, under THIS repo's conventions.

    python scripts/finetune_ram.py --anatomy brain --volumes 8 --slices 4
    python scripts/finetune_ram.py --anatomy knee --R 8 --max-iter 50 --out trained_nets/ram

Why not just call `ram.finetune`
--------------------------------
`ram/models/ram.py::finetune` binds ONE `physics` to the whole dataset and hands
it to `dinv.Trainer`. Multicoil MRI has PER-SLICE sensitivity maps, so a single
physics cannot describe a dataset of slices -- using one slice's maps for all of
them would be a different (and easier, and wrong) problem.

So the loop is written out here, with every hyperparameter and every loss taken
from their `finetune` defaults so the only intentional difference is the
per-sample physics:

    losses     SureGaussianLoss(sigma) + EILoss(Shift(shift_max=.1), weight=.1)
    optimiser  Adam
    lr         1e-4
    max_iter   50          (EPOCHS in their code, not gradient steps)
    batch_size 1           (forced here: physics changes per sample)
    early stop on the validation loss

Pass --use-ram-finetune to call their wrapper instead, which is only correct
when every sample shares one set of maps (`--volumes 1 --slices 1`).

THE NOISE CONVENTION -- the reason this script exists
-----------------------------------------------------
RAM/deepinv assume `y = A(x) + n`: noise on the MEASUREMENT. This repo adds it
in the COIL-IMAGE domain, before the transform (`operators/noise.py::mri_awgn`):

    y = mask . F(s.x + sigma.n)

`fftc` is orthonormal, so `F(sigma.n)` is again white with std sigma and the two
agree on the SAMPLED entries. They differ where it matters:

    ours     unsampled k-space bins are EXACTLY ZERO
    theirs   unsampled bins contain pure noise

A network trained on the second sees noise where ours sees nothing. So this
script never calls `physics(x)` to synthesise data -- every measurement comes
from `mri_awgn`, and the deepinv physics exists only to describe A and to carry
`sigma` for the SURE loss.

One honest caveat that follows: SURE's Gaussian estimator assumes white noise
across the whole measurement vector. Under our convention the unsampled bins are
deterministically zero, so the estimate is exact on the sampled support and
biased over the zeros. `--noise-loss splitting` avoids the assumption entirely
and is the safer choice if the SURE runs look unstable.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from operators import FFT2D, Mask, Sense
from operators.noise import mri_awgn
from physics.mask import make_acc_mask

# `training.metrics` is imported lazily inside `evaluate`: `training/__init__`
# pulls in torchvision, and `--help` should work without the full env.

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ===========================================================================
#  data
# ===========================================================================
def resolve_roots(anatomy, split, config=None):
    """`(smap_root, scale_fac, R, acs)` from a generated config."""
    cfg_path = config or os.path.join(ROOT, "config", anatomy, "mg", "mglpds_R8.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    d = cfg["data"][split]
    root = d["smap_root"]
    if not os.path.isabs(root):
        root = os.path.normpath(os.path.join(ROOT, root))
    return root, float(d["scale_fac"]), cfg["mri"], cfg


def load_slices(smap_root, n_volumes, n_slices, scale_fac, device):
    """`[(x, smaps), ...]`, each normalised to unit max magnitude.

    Normalisation is per slice and RECORDED: RAM is a foundation model trained
    on data in roughly [0, 1], and the preprocessed volumes carry scale_fac
    (2000 for brain). Metrics are computed after normalisation, so every method
    is scored on the same scale.
    """
    import h5py
    files = sorted(glob.glob(os.path.join(smap_root, "*.h5")))
    if not files:
        raise SystemExit(f"[ram] no volumes under {smap_root}")

    out = []
    for path in files[:n_volumes]:
        with h5py.File(path, "r") as f:
            n = f["image"].shape[0]
            take = list(range(min(n_slices, n)))
            image = np.asarray(f["image"][take])
            smaps = np.asarray(f["smaps"][take])
        for k in range(image.shape[0]):
            x = torch.from_numpy(image[k]).to(torch.complex64).reshape(1, 1, *image.shape[-2:])
            s = torch.from_numpy(smaps[k]).to(torch.complex64)
            s = s.reshape(1, -1, *s.shape[-2:])
            scale = float(x.abs().max())
            if scale <= 0:
                continue
            out.append((x.to(device) / scale, s.to(device), os.path.basename(path), k))
    return out


def make_measurement(x, smaps, R, acs, sigma, device, seed=None):
    """OUR convention: noise in the coil-image domain, then transform, then mask."""
    H, W = x.shape[-2:]
    mask = make_acc_mask((H, W), R, acs_lines=acs, device=device)
    while mask.dim() < 4:
        mask = mask.unsqueeze(0)
    E = Mask(mask) @ FFT2D() @ Sense(smaps)
    gen = None
    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(int(seed))
    # mri_awgn adds sigma*n to s.x BEFORE fftc and the mask -- unsampled bins
    # come out exactly zero. This is the whole point; do not replace it with
    # `physics(x)`.
    y, _, _ = mri_awgn(x, mask, smaps, sigma, "uniform", generator=gen)
    return y, mask, E


# ===========================================================================
#  deepinv plumbing
# ===========================================================================
def build_physics(mask, smaps, sigma, device):
    """`MultiCoilMRI` describing A, carrying sigma for SURE.

    Never used to GENERATE data -- see the module docstring. If deepinv's
    signature has moved, this is the one place to fix.
    """
    import deepinv as dinv
    H, W = mask.shape[-2:]
    physics = dinv.physics.MultiCoilMRI(
        mask=mask.squeeze(1), coil_maps=smaps, img_size=(H, W), device=device)
    physics.noise_model = dinv.physics.GaussianNoise(sigma)
    return physics


def to_dinv(xc):
    """(B, 1, H, W) complex -> (B, 2, H, W) real, deepinv's complex layout."""
    return torch.cat([xc.real, xc.imag], dim=1)


def from_dinv(xr):
    if torch.is_complex(xr):
        return xr
    if xr.shape[1] == 2:
        return torch.complex(xr[:, :1], xr[:, 1:2])
    return xr.to(torch.complex64)          # magnitude-only output


def kspace_to_dinv(y):
    """ImMAP (B, C, H, W) complex k-space -> deepinv's real-view layout."""
    return torch.stack([y.real, y.imag], dim=2)      # (B, C, 2, H, W)


# ===========================================================================
#  finetuning
# ===========================================================================
def build_losses(sigma, noise_loss, transform):
    """Exactly what `ram.finetune` builds, so only the physics differs."""
    import deepinv as dinv
    losses = []
    if noise_loss == "SURE":
        losses.append(dinv.loss.SureGaussianLoss(sigma))
    elif noise_loss == "splitting":
        losses.append(dinv.loss.SplittingLoss(split_ratio=0.9))
    elif noise_loss == "noiseless":
        losses.append(dinv.loss.MCLoss())
    else:
        raise SystemExit(f"[ram] unknown --noise-loss {noise_loss!r}")
    if transform and transform != "none":
        t = (dinv.transform.Shift(shift_max=0.1) if transform == "shift"
             else dinv.transform.Rotate(multiples=90))
        losses.append(dinv.loss.EILoss(t, weight=0.1))
    return losses


@torch.no_grad()
def evaluate(model, samples, args, device):
    """Mean PSNR / SSIM / NRMSE over `samples`, on the same axis the grid uses."""
    from training.metrics import compute_metrics
    tot, n = {"psnr": 0.0, "ssim": 0.0, "nrmse": 0.0}, 0
    for i, (x, smaps, _, _) in enumerate(samples):
        y, mask, E = make_measurement(x, smaps, args.R, args.acs, args.sigma,
                                      device, seed=1000 + i)
        physics = build_physics(mask, smaps, args.sigma, device)
        est = from_dinv(model(kspace_to_dinv(y), physics=physics))
        m = compute_metrics(x.abs(), est.abs())
        for k in tot:
            tot[k] += float(m[k])
        n += 1
    return {k: v / max(n, 1) for k, v in tot.items()}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--anatomy", choices=("brain", "knee"), default="brain")
    p.add_argument("--split", choices=("train", "val"), default="train")
    p.add_argument("--eval-split", choices=("train", "val"), default="val")
    p.add_argument("--config", default=None, help="config to read roots/mri from")
    p.add_argument("--volumes", type=int, default=8)
    p.add_argument("--slices", type=int, default=4, help="slices per volume")
    p.add_argument("--eval-volumes", type=int, default=4)
    p.add_argument("--R", type=int, default=None, help="default: the config's")
    p.add_argument("--acs", type=int, default=None)
    p.add_argument("--sigma", type=float, default=None,
                   help="OUR convention: coil-image-domain std. Default: the "
                        "config's val_noise_std. Fixed, not sampled -- SURE "
                        "needs one level, while the grid trains over a range.")
    # their finetune() defaults, verbatim
    p.add_argument("--max-iter", type=int, default=50, help="EPOCHS, as in ram")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--noise-loss", choices=("SURE", "splitting", "noiseless"),
                   default="SURE")
    p.add_argument("--transform", choices=("shift", "rotate", "none"),
                   default="shift")
    p.add_argument("--patience", type=int, default=10,
                   help="early stop after this many epochs without improvement")
    p.add_argument("--use-ram-finetune", action="store_true",
                   help="call ram.finetune instead; only correct when every "
                        "sample shares one set of coil maps")
    p.add_argument("--out", default="trained_nets/ram")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    smap_root, scale_fac, mri, cfg = resolve_roots(args.anatomy, args.split, args.config)
    args.R = args.R if args.R is not None else int(mri["R"])
    args.acs = args.acs if args.acs is not None else int(mri["acs_lines"])
    if args.sigma is None:
        args.sigma = float(cfg["training"]["val_noise_std"])

    print(f"[ram] {args.anatomy} {args.split}  R={args.R}  acs={args.acs}  "
          f"sigma={args.sigma}  (coil-image domain, our convention)")
    print(f"[ram] {smap_root}")

    train = load_slices(smap_root, args.volumes, args.slices, scale_fac, device)
    eval_root, _, _, _ = resolve_roots(args.anatomy, args.eval_split, args.config)
    held = load_slices(eval_root, args.eval_volumes, 1, scale_fac, device)
    print(f"[ram] {len(train)} finetuning slices, {len(held)} held-out slices")
    if not train:
        raise SystemExit("[ram] no training slices found.")

    from ram import RAM
    model = RAM(device=str(device))

    before = evaluate(model, held, args, device)
    print(f"[ram] zero-shot   PSNR {before['psnr']:.2f}  SSIM {before['ssim']:.4f}  "
          f"NRMSE {before['nrmse']:.4f}")

    if args.use_ram_finetune:
        # Only sound when one physics covers every sample.
        if len(train) > 1:
            raise SystemExit(
                "[ram] --use-ram-finetune binds ONE physics to the whole "
                "dataset, but coil maps are per slice. Use --volumes 1 "
                "--slices 1, or drop the flag for the per-sample loop.")
        from ram import finetune
        x, smaps, _, _ = train[0]
        y, mask, _ = make_measurement(x, smaps, args.R, args.acs, args.sigma, device)
        physics = build_physics(mask, smaps, args.sigma, device)
        model = finetune(model, kspace_to_dinv(y), physics, max_iter=args.max_iter,
                         noise_loss=args.noise_loss, transform=args.transform,
                         lr=args.lr, device=str(device))
    else:
        model = finetune_per_sample(model, train, held, args, device)

    after = evaluate(model, held, args, device)
    print(f"[ram] finetuned   PSNR {after['psnr']:.2f}  SSIM {after['ssim']:.4f}  "
          f"NRMSE {after['nrmse']:.4f}")
    print(f"[ram] delta       PSNR {after['psnr'] - before['psnr']:+.2f} dB")

    out_dir = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    out_dir = os.path.join(out_dir, f"{args.anatomy}_R{args.R}")
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(),
                "args": vars(args), "zero_shot": before, "finetuned": after},
               os.path.join(out_dir, "ram_finetuned.ckpt"))
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump({"args": vars(args), "zero_shot": before, "finetuned": after},
                  f, indent=2)
    print(f"[ram] wrote {out_dir}")
    return 0


def finetune_per_sample(model, train, held, args, device):
    """`ram.finetune`'s recipe, with the physics rebuilt for each slice."""
    import deepinv as dinv                                       # noqa: F401

    losses = build_losses(args.sigma, args.noise_loss, args.transform)
    print("[ram] losses: " + str([l.__class__.__name__ for l in losses]))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # An 80/20 split of the finetuning slices, mirroring theirs. `held` is a
    # SEPARATE set used only for the before/after report, so early stopping
    # never sees the numbers that get quoted.
    n_val = max(1, len(train) // 5)
    tr, va = train[:-n_val], train[-n_val:]
    if not tr:
        tr, va = train, train
    print(f"[ram] {len(tr)} train / {len(va)} val slices for early stopping")

    def epoch_loss(samples, train_mode):
        model.train(train_mode)
        total, n = 0.0, 0
        for i, (x, smaps, _, _) in enumerate(samples):
            y, mask, _ = make_measurement(x, smaps, args.R, args.acs, args.sigma,
                                          device, seed=None if train_mode else i)
            physics = build_physics(mask, smaps, args.sigma, device)
            yd = kspace_to_dinv(y)
            with torch.set_grad_enabled(train_mode):
                x_hat = model(yd, physics=physics)
                loss = sum(l(x_net=x_hat, y=yd, physics=physics, model=model)
                           for l in losses)
                if train_mode:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
            total += float(loss.detach())
            n += 1
        return total / max(n, 1)

    best, best_state, bad = float("inf"), None, 0
    for ep in range(args.max_iter):
        t0 = time.perf_counter()
        tl = epoch_loss(tr, True)
        vl = epoch_loss(va, False)
        flag = ""
        if vl < best - 1e-6:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            flag = "  *"
        else:
            bad += 1
        print(f"[ram] epoch {ep:>3}/{args.max_iter}  train {tl:.5f}  "
              f"val {vl:.5f}  ({time.perf_counter() - t0:.1f}s){flag}")
        if bad >= args.patience:
            print(f"[ram] early stop: {args.patience} epochs without improvement")
            break

    if best_state is not None:
        model.load_state_dict(best_state)      # return the BEST, as theirs does
    model.eval()
    return model


if __name__ == "__main__":
    raise SystemExit(main())
