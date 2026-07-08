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
    train_synthesis
)
from training.common import load_ckpt

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

    cfg_save_path = os.path.join(save_dir, "config.json")
    with open(cfg_save_path, "w") as f:
        json.dump(cfg_to_save, f, indent=4)

    print(f"Saved config to {cfg_save_path}")


def main(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
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
    else:
        raise ValueError(f"Unknown task {task}")


if __name__ == "__main__":
    main(sys.argv[1])
