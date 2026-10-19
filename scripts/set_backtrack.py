#!/usr/bin/env python3
"""
Change the backtrack thresholds of EXISTING runs, so a resume trains under them.

    python scripts/set_backtrack.py trained_nets/mg_recon/brain/mggroup169v6_R12 \
        trained_nets/mg_recon/knee/mggroup169v6_R8 --new-wandb-run
    python scripts/set_backtrack.py <run dir> ... --dry-run

Why this exists: train.py writes `training.backtrack_ratio` (and, since
2026-10-19, `training.val_backtrack_ratio`) into a run's config.json the first
time it runs, and a resume reads config.json -- so a run launched under the old
default of 500 keeps 500 on every resume, whatever training/recon.py now says.
This rewrites the two keys in place (the old file is kept as
config.json.bak-<timestamp>) and reports where the run will resume from.

--new-wandb-run clears `wandb.id`, so the resume logs to a NEW wandb run. Use it
when the checkpoint is far behind where the run got to -- a diverged run, whose
last improving epoch may be thousands of epochs back. wandb drops every log
whose step is below the run's highest one, so resuming into the same wandb run
would log nothing until the restarted run caught up with the diverged one.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time

import torch


def _none_or_float(s):
    return None if s.lower() in ("none", "null", "off") else float(s)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="run directories (holding config.json)")
    p.add_argument("--ratio", type=_none_or_float, default=100.0,
                   help="training.backtrack_ratio (default 100; 'none' = off)")
    p.add_argument("--val-ratio", type=_none_or_float, default=2.0,
                   help="training.val_backtrack_ratio (default 2; 'none' = off)")
    p.add_argument("--new-wandb-run", action="store_true",
                   help="clear wandb.id so the resume logs to a new wandb run")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    for run in a.runs:
        cfg_path = os.path.join(run, "config.json")
        if not os.path.exists(cfg_path):
            print(f"[skip] {run}: no config.json -- nothing to resume (the launcher "
                  f"starts it fresh from config.gen.json)")
            continue
        with open(cfg_path) as f:
            cfg = json.load(f)
        t = cfg["training"]
        before = (t.get("backtrack_ratio", "unset"), t.get("val_backtrack_ratio", "unset"),
                  cfg.get("wandb", {}).get("id"))
        t["backtrack_ratio"] = a.ratio
        t["val_backtrack_ratio"] = a.val_ratio
        if a.new_wandb_run:
            cfg.setdefault("wandb", {})["id"] = None

        where = "no net.ckpt -- will start fresh"
        ckpt = os.path.join(run, "net.ckpt")
        if os.path.exists(ckpt):
            try:
                ck = torch.load(ckpt, map_location="cpu", weights_only=False)
            except TypeError:                               # torch < 1.13
                ck = torch.load(ckpt, map_location="cpu")
            ep = int(ck.get("step") or 0) // int(t["steps_per_epoch"])
            where = (f"resumes at epoch {ep}/{t['num_epochs']}, best train loss "
                     f"{ck.get('best_loss')}, backtrack_count {ck.get('backtrack_count')}")
            if ep >= int(t["num_epochs"]):
                where += "  ** COMPLETE: the launcher will SKIP it **"

        print(f"{run}\n  backtrack_ratio {before[0]} -> {a.ratio}, "
              f"val_backtrack_ratio {before[1]} -> {a.val_ratio}"
              + (f", wandb.id {before[2]} -> new run" if a.new_wandb_run else "")
              + f"\n  {where}")
        if a.dry_run:
            continue
        shutil.copy2(cfg_path, f"{cfg_path}.bak-{stamp}")
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=4, sort_keys=True)   # as save_config writes it
    if a.dry_run:
        print("\n(dry run: nothing written)")


if __name__ == "__main__":
    main()
