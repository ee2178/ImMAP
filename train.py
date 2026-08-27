import yaml
import torch
import wandb
import sys
import os
import json
import copy

from models import build_model
from training import (
    train_denoiser,
    train_recon,
    train_joint_denoising_recon,
    train_ipalm,
    train_ccl,
    train_synthesis,
    train_dt_synthesis,
    train_i2sb,
    train_latent_i2sb
)
from training.common import load_ckpt, write_config

import datasets                       # triggers registration via __init__
from datasets.registry import build_loader


def build_optimizer(model, cfg):
    opt_cfg = cfg["optimizer"]

    if opt_cfg["type"] == "Adam":
        return torch.optim.Adam(
            model.parameters(),
            **opt_cfg["params"]
        )

    raise ValueError(f"Unknown optimizer {opt_cfg['type']}")

def build_scheduler(optimizer, cfg):
    sched_cfg = cfg["scheduler"]

    if sched_cfg["type"] == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            **sched_cfg["params"]
        )

    if sched_cfg["type"] == "StepLR":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            **sched_cfg["params"]
        )

    if sched_cfg["type"] == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            **sched_cfg["params"]
        )

    if sched_cfg["type"] in ("Constant", "None", None):
        # LR stays at the optimizer's initial value; step() is a no-op.
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda _: 1.0
        )

    raise ValueError(f"Unknown scheduler {sched_cfg['type']}")

def save_config(cfg):
    """
    Save experiment config as JSON.

    Any `init` flag under the model config is forced to False in the saved
    copy, so reloading this config alongside a checkpoint won't re-initialize
    over the loaded weights. The in-memory cfg for the current run is unchanged.
    """
    save_dir = cfg["paths"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    cfg_to_save = copy.deepcopy(cfg)

    def disable_init(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "init":
                    obj[k] = False
                else:
                    disable_init(v)
        elif isinstance(obj, list):
            for item in obj:
                disable_init(item)

    disable_init(cfg_to_save.get("model", {}))

    # write_config, not json.dump: this file is re-read with yaml.safe_load on resume, and
    # json's `1e-06` comes back as a string there (see training.common.write_config).
    cfg_save_path = write_config(cfg_to_save, os.path.join(save_dir, "config.json"))

    print(f"Saved config to {cfg_save_path}")


def main(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # cuDNN autotuning. Measured +17% on MGLPDSNet (130.51 -> 108.05 ms, L40S,
    # 320x320 / 16 coils / batch 1), essentially ALL of it from one shape: the
    # coarsest level's ConvTranspose at 40x40, where the default heuristic
    # picks 196 us/call against autotuning's 35. Only the multigrid models have
    # small-spatial / high-channel transposed convs, which is why the flat
    # baseline never showed it.
    #
    # Autotuning re-benchmarks per distinct input shape, so it is a win only
    # when shapes are stable. The image-domain embedding makes them stable:
    # every sample is snapped up to a multiple of the model's pad_stride, so a
    # dataset of varied fastMRI matrix sizes collapses to a handful of grids.
    # Set `training.cudnn_benchmark: false` if you ever feed genuinely
    # unbounded sizes.
    torch.backends.cudnn.benchmark = bool(
        cfg.get("training", {}).get("cudnn_benchmark", True))
    # Init model from scratch if no ckpt provided
    model = build_model(cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    # torch.autograd.set_detect_anomaly(True)   # debug only — remove after
    # --------------------------------------------------
    # Optionally resume from checkpoint
    # --------------------------------------------------
    start_step = 0
    ckpt_path = cfg.get("paths", {}).get("ckpt", None)

    # `paths.init_ckpt`: weights-only WARM START, as opposed to `paths.ckpt`,
    # which is a full resume (optimizer, scheduler and step all restored).
    # Transferring a model trained on a different problem -- an easier
    # acceleration, a different anatomy -- wants none of that: a fresh Adam and
    # the whole cosine schedule, from step 0. Passing optimizer=None below is
    # what makes it weights-only; load_ckpt skips whatever it is handed None
    # for, so the source checkpoint keeps its optimizer state untouched and does
    # not need rewriting first.
    #
    # POPPED, not read in place: cfg["paths"] is splatted as **cfg["paths"] into
    # every train_* function, and none of them accepts this key -- leaving it
    # would be a TypeError at dispatch, in every task branch.
    #
    # Ignored once `ckpt` exists, so a run that has started (train.py stamps
    # paths.ckpt into its saved config) never re-applies the warm start over its
    # own progress on a requeue.
    init_ckpt = cfg.get("paths", {}).pop("init_ckpt", None)

    if ckpt_path:
        print(f"Loading checkpoint from {ckpt_path}")

        model, optimizer, scheduler, start_step = load_ckpt(
            path=ckpt_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )

        print(f"Resuming from step {start_step}")

    elif init_ckpt:
        print(f"Warm-starting weights from {init_ckpt}")
        load_ckpt(path=init_ckpt, model=model, device=device)
        print("Fresh optimizer and scheduler; starting from step 0.")

    # compile the fused flex kernel once, on the final model object
    if getattr(model, "attn_backend", None) == "flex":
        model.compile_flex()
    # --------------------------------------------------
    # Data loaders
    # --------------------------------------------------

    train_loader = build_loader(cfg["data"]["train"], shuffle=True,  drop_last=True)
    val_loader   = build_loader(cfg["data"]["val"],   shuffle=False, drop_last=False)

    # --------------------------------------------------
    # Initialize wandb
    # --------------------------------------------------
    wandb.init(
        project=cfg["wandb"]["project"],
        resume="allow",
        id=cfg["wandb"]["id"],
        name=cfg["experiment"]["name"],
        config=cfg,
    )

    # Persist the wandb run id so the saved config can resume this exact run.
    # On a fresh run id starts as None and wandb assigns one; capture it here.
    cfg["wandb"]["id"] = wandb.run.id

    task = cfg["task"]

    # Saving configs
    # Rewrite checkpoint path so the saved config resumes from net.ckpt
    cfg["paths"]["ckpt"] = os.path.join(
        cfg["paths"]["save_dir"],
        "net.ckpt"
    )

    # Save args (now carries wandb id + ckpt path -> resume-ready)
    save_config(cfg)

    # --------------------------------------------------
    # Training dispatch
    # --------------------------------------------------
    if task == "denoiser":
        steps_per_epoch = cfg["training"]["steps_per_epoch"]

        train_denoiser(
            net=model,
            opt=optimizer,
            sched=scheduler,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            wandb=wandb,
            start_epoch=start_step // steps_per_epoch,
            **cfg["training"],
            **cfg["paths"],
        )

    elif task == "recon":
        steps_per_epoch = cfg["training"]["steps_per_epoch"]

        train_recon(
            net=model,
            opt=optimizer,
            sched=scheduler,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            wandb=wandb,
            start_epoch=start_step // steps_per_epoch,
            **cfg["training"],
            **cfg["mri"],
            **cfg["paths"],
        )

    elif task == "immap":
        steps_per_epoch = cfg["training"]["steps_per_epoch"]

        train_joint_denoising_recon(
            net=model,
            opt=optimizer,
            sched=scheduler,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            wandb=wandb,
            start_epoch=start_step // steps_per_epoch,
            **cfg["training"],
            **cfg["mri"],
            **cfg["paths"],
        )

    elif task == "ipalm":
        # NOTE: train_ipalm is still step-based (not yet converted to epochs).
        train_ipalm(
            net=model,
            opt=optimizer,
            sched=scheduler,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            wandb=wandb,
            start_step=start_step,
            **cfg["training"],
            **cfg["mri"],
            **cfg["paths"],
        )
    elif task == "ccl_pretrain":
        steps_per_epoch = cfg["training"]["steps_per_epoch"]
        
        train_ccl(
            net=model, 
            opt=optimizer, 
            sched=scheduler, 
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            wandb=wandb,
            start_epoch=start_step//steps_per_epoch,
            **cfg["training"],
            **cfg["paths"],
            )
    elif task == "synthesis":
        steps_per_epoch = cfg["training"]["steps_per_epoch"]

        train_synthesis(
            net=model,
            opt=optimizer,
            sched=scheduler,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            wandb=wandb,
            start_epoch=start_step//steps_per_epoch,
            **cfg["training"],
            **cfg["paths"],
            )
    elif task == "dt_synthesis":
        steps_per_epoch = cfg["training"]["steps_per_epoch"]

        train_dt_synthesis(
            net=model,
            opt=optimizer,
            sched=scheduler,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            wandb=wandb,
            start_epoch=start_step // steps_per_epoch,
            **cfg["training"],
            **cfg["paths"],
            )
    elif task == "i2sb":
        steps_per_epoch = cfg["training"]["steps_per_epoch"]

        train_i2sb(
            net=model,
            opt=optimizer,
            sched=scheduler,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            wandb=wandb,
            start_epoch=start_step//steps_per_epoch,
            **cfg["training"],
            **cfg["i2sb"],
            **cfg["paths"],
            )
    elif task == "latent_i2sb":
        steps_per_epoch = cfg["training"]["steps_per_epoch"]

        # `model` (built by build_model) is the NEW regressor R; the two convolutional
        # dictionaries are loaded frozen inside the loop from cfg["dicts"] config paths.
        train_latent_i2sb(
            R=model,
            opt=optimizer,
            sched=scheduler,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            wandb=wandb,
            start_epoch=start_step//steps_per_epoch,
            **cfg["dicts"],
            **cfg["training"],
            **cfg["i2sb"],
            **cfg["paths"],
            )
    else:
        raise ValueError(f"Unknown task {task}")


if __name__ == "__main__":
    main(sys.argv[1])
