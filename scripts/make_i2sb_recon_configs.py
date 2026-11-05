#!/usr/bin/env python3
"""
Configs for the I2SB reconstruction baselines (task "i2sb_recon"), derived from the mg recon grid.

The measurement -- data roots, slice ranges, mask, ACS, protocol, online maps, noise range, val
noise and seed -- is NOT written here. It is read from what scripts/make_mg_recon_configs.py
generates TODAY for that anatomy and acceleration (its `lpdsnet` cell), so an i2sb_recon run and a
one-shot run of the grid see the same masks, noise and maps and their numbers share a table. The
mg launcher regenerates its configs on every submit (torch/_mg_recon_body.sh); so does
torch/i2sb_recon_unet.sbatch, through this script. Copying a committed mg config instead would
silently go stale the next time the grid's protocol changes (it did, 2026-09-22: legacy ->
measured).

What IS set here: the regressor (the I2SB paper's ADM UNet, models/sb_recon_unet.py), the bridge
schedule, the I2SB training loop's knobs, and two data overrides the bridge needs:
    pad_multiple 1   the bridge runs on the measured grid (training/i2sb_recon.py refuses an
                     embedding); SBUnet pads internally
    num_workers 8

    python scripts/make_i2sb_recon_configs.py                         # brain + knee, R=8
    python scripts/make_i2sb_recon_configs.py --anatomy brain --accels 4 8 16
    python scripts/make_i2sb_recon_configs.py --list-cells            # index, anatomy, R, model
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

ANATOMIES = ("brain", "knee")
ACCELS = (8,)
MODELS = ("unet",)

# Brownian-bridge temperature: tau = RMS(x0 - x1)/2 over the object (the rule of
# scripts/calibrate_nyumets_bridge.py), x0 in the operator's phase convention (x0_source
# "operator"). Brain: measured on ONE local volume (file_brain_AXT2_202_2020322, slices 0-7) under
# the measured protocol at R=8 -- RMS 0.217 inside the head, per-slice 0.18-0.24, insensitive to
# sigma_n. Knee: UNMEASURED placeholder. Re-measure for other accelerations.
TAU = {"brain": 0.11, "knee": 0.1}

# Inherited from the mg grid's training block, in addition to the whole data / mri blocks.
TRAINING_KEYS = ("noise_std", "noise_dist", "val_noise_std", "val_seed", "use_organ_mask",
                 "val_kspace_type")

UNET = dict(cond_x1=True, model_channels=64, num_res_blocks=2, channel_mult=[1, 2, 3, 4],
            attention_resolutions=[40], image_size=320, num_heads=4, num_head_channels=32,
            dropout=0.0, use_scale_shift_norm=True, resblock_updown=False, conv_resample=True,
            t0=0.0001, T=1.0)


def cells(anatomies=ANATOMIES, accels=ACCELS, models=MODELS):
    return [(a, r, m) for a in anatomies for r in accels for m in models]


def grid_config(anatomy, r):
    """The mg grid's lpdsnet cell for (anatomy, R), as its generator writes it now."""
    with tempfile.TemporaryDirectory() as d:
        cmd = [sys.executable, os.path.join(REPO, "scripts", "make_mg_recon_configs.py"),
               "--out", d, "--anatomy", anatomy, "--accels", str(r), "--only", "lpdsnet"]
        subprocess.run(cmd, check=True, cwd=REPO, stdout=subprocess.DEVNULL)
        path = os.path.join(d, anatomy, "mg", f"lpdsnet_R{r}.json")
        with open(path) as f:
            return json.load(f)


def make_config(anatomy, r, model, tau=None):
    if model != "unet":
        raise ValueError(f"only the UNet baseline is configured, got {model!r}")
    base = grid_config(anatomy, r)
    if base.get("task") != "recon":
        raise SystemExit(f"mg grid cell for {anatomy} R{r} has task {base.get('task')!r}")
    tau = TAU[anatomy] if tau is None else float(tau)
    sched = {"kind": "brownian", "tau": tau, "n_points": 1000, "beta_max": 0.3}

    data = base["data"]
    for split in data.values():
        split["pad_multiple"] = 1
        split["num_workers"] = 8

    name = f"{model}_R{r}"
    return {
        "task": "i2sb_recon",
        "experiment": {"name": f"I2SB_{model}_{anatomy}_R{r}"},
        "model": {"type": "SBReconUnet", "params": {**UNET, **sched}},
        "i2sb": {**sched, "deterministic": False, "posterior": "ddpm",
                 "x1_source": "adjoint", "x0_source": "operator",
                 "val_mode": "full_recon", "val_nfe": 20},
        "data": data,
        "mri": base["mri"],
        "training": {
            "num_epochs": 1500, "steps_per_epoch": 200, "val_every_epochs": 25,
            "accum_steps": 4, "ema_decay": 0.99, "ema_warmup": True,
            **{k: base["training"][k] for k in TRAINING_KEYS if k in base["training"]},
            "val_slices": 100,
            "loss_type": "complex-mse", "loss_weight": "uniform",
            "clip_grad": 1.0, "backtrack_thresh": None, "backtrack_factor": 0.9},
        "paths": {"save_dir": f"trained_nets/i2sb_recon/{anatomy}/{name}", "ckpt": None},
        "optimizer": {"type": "Adam", "params": {"lr": 1.0e-4}},
        "scheduler": {"type": "CosineAnnealingLR",
                      "params": {"eta_min": 1.0e-6, "T_max": 300000}},
        "wandb": {"project": "i2sb_recon", "id": None},
    }


def config_path(root, anatomy, r, model):
    return os.path.join(root, anatomy, "i2sb", f"{model}_R{r}.json")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="config", help="config root (default: config)")
    p.add_argument("--anatomy", nargs="*", choices=ANATOMIES, default=list(ANATOMIES))
    p.add_argument("--accels", nargs="*", type=int, default=list(ACCELS))
    p.add_argument("--tau", type=float, default=None,
                   help="override the per-anatomy TAU (written to i2sb AND model.params)")
    p.add_argument("--list-cells", action="store_true",
                   help="print 'index<TAB>anatomy<TAB>R<TAB>model' and exit")
    args = p.parse_args()

    todo = cells(args.anatomy, args.accels)
    if args.list_cells:
        for i, (a, r, m) in enumerate(todo):
            print(f"{i}\t{a}\tR{r}\t{m}")
        return
    for a, r, m in todo:
        cfg = make_config(a, r, m, tau=args.tau)
        path = config_path(args.out, a, r, m)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="\n") as f:
            json.dump(cfg, f, indent=4)
            f.write("\n")
        mri = cfg["mri"]
        print(f"{path}: {mri['kspace_type']}, R={mri['R']}, "
              f"noise {cfg['training']['noise_std']}, tau={cfg['i2sb']['tau']}")


if __name__ == "__main__":
    main()
