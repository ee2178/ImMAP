#!/usr/bin/env python3
"""
Rebuild a run's resume checkpoint (net.ckpt) from its intact WEIGHTS-ONLY checkpoint, after
net.ckpt was truncated by a failed write ("unexpected pos ... vs ..." from torch's zip writer).

    python scripts/salvage_ckpt.py trained_nets/guided_recon/knee/s3/sbtgvn_c8_adjoint_R8_g1_tau0.09
    python scripts/salvage_ckpt.py <run_dir> --source net_bestval.ckpt

A truncated net.ckpt cannot be repaired: torch writes the tensors one after another, and the ones
after the failure point were never written (re-indexing the zip only exposes that). What the run
still has is net_ema.ckpt (written right after every successful net.ckpt, so it is the latest
averaged weights before the crash), net_last.ckpt and net_bestval.ckpt. From one of them this
writes a net.ckpt that the launcher's normal resume picks up:

    model        the source's weights (EMA-averaged: resuming from the smoothed point)
    step         the source's step -> training continues from the next epoch
    scheduler    built from the run's config and ADVANCED to that step, so the learning rate
                 continues on its schedule rather than restarting at its maximum
    optimizer    FRESH Adam: its moment estimates are not in any weights-only checkpoint; they
                 rebuild within a few hundred steps
    EMA          restarts from the loaded weights (the trainer warns; expected here)
    best_loss    the source's, so the first resumed epoch does not count as an improvement
                 unconditionally; backtrack_count 0 unless the source records one

The corrupted file is kept beside it as net.ckpt.corrupt-<timestamp>. Refuses to touch a net.ckpt
that still loads, unless --force.
"""

import argparse
import os
import sys
import time
import warnings

import torch
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

from models import build_model                                       # noqa: E402
from training.common import load_ckpt, save_ckpt                     # noqa: E402


def _loads(path):
    try:
        torch.load(path, map_location="cpu", weights_only=False)
        return True
    except Exception:                                                # noqa: BLE001
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--source", default=None,
                    help="weights-only checkpoint in run_dir (default: the first that loads of "
                         "net_ema.ckpt, net_last.ckpt, net_bestval.ckpt)")
    ap.add_argument("--force", action="store_true", help="replace a net.ckpt that still loads")
    args = ap.parse_args()

    import train                                                     # build_optimizer/_scheduler
    target = os.path.join(args.run_dir, "net.ckpt")
    if os.path.exists(target) and _loads(target) and not args.force:
        raise SystemExit(f"{target} loads fine; nothing to salvage (pass --force to rebuild it)")
    names = [args.source] if args.source else ["net_ema.ckpt", "net_last.ckpt",
                                               "net_bestval.ckpt"]
    source = next((os.path.join(args.run_dir, n) for n in names
                   if os.path.exists(os.path.join(args.run_dir, n))
                   and _loads(os.path.join(args.run_dir, n))), None)
    if source is None:
        raise SystemExit(f"no intact weights-only checkpoint among {names} in {args.run_dir}")

    with open(os.path.join(args.run_dir, "config.json")) as f:
        cfg = yaml.safe_load(f)                                     # what train.py does
    meta = torch.load(source, map_location="cpu", weights_only=False)
    step = int(meta.get("step") or 0)
    model = build_model(cfg)
    load_ckpt(source, model=model)
    opt = train.build_optimizer(model, cfg)
    sched = train.build_scheduler(opt, cfg)
    if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
        raise SystemExit("ReduceLROnPlateau has no step-indexed schedule to advance; not handled")
    with warnings.catch_warnings():                                 # "scheduler before optimizer"
        warnings.simplefilter("ignore")
        for _ in range(step):
            sched.step()

    if os.path.exists(target):
        kept = f"{target}.corrupt-{time.strftime('%Y%m%d_%H%M%S')}"
        os.replace(target, kept)
        print(f"kept the corrupted checkpoint as {kept}")
    save_ckpt(target, model=model, step=step, optimizer=opt, scheduler=sched,
              backtrack_count=int(meta.get("backtrack_count") or 0),
              best_loss=meta.get("best_loss"), best_val_loss=meta.get("best_val_loss"))
    print(f"wrote {target} from {os.path.basename(source)}: step {step}, learning rate "
          f"{opt.param_groups[0]['lr']:.3e}, fresh Adam, EMA restarts on resume")


if __name__ == "__main__":
    main()
