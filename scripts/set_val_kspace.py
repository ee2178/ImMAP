#!/usr/bin/env python3
"""
Switch EXISTING runs to validate on measured k-space, like they train.

    python scripts/set_val_kspace.py --dry-run          # every run under trained_nets/mg_recon
    python scripts/set_val_kspace.py                    # apply
    python scripts/set_val_kspace.py trained_nets/mg_recon/knee/mg121v6_R8 ...

Configs generated 2026-10-20 .. 2026-10-22 carry `training.val_kspace_type:
"simulated"`: those runs train on measured k-space + AWGN but validate on
simulated k-space, so their wandb val curves (and best-validation checkpoint,
and validation backtracking) are on a different, easier measurement than every
earlier run's. A resume reads the run's own config.json, so regenerating
configs does not change them -- this does.

For each run with a `val_kspace_type` different from its training kspace_type:
  * the key is removed from config.json (validation then follows training);
    the old file is kept as config.json.bak-<timestamp>;
  * net_bestval.ckpt, if present, is RENAMED to net_bestval.<old type>.ckpt.
    Its stored best validation loss was measured on the other measurement, and
    measured losses are higher: kept, it would make the validation backtrack
    (training/recon.py, VAL_BACKTRACK_RATIO) compare measured losses against a
    simulated best. Renamed, the next validation pass sets a fresh bar.
net.ckpt (the best TRAINING-loss checkpoint, what a resume continues from) is
untouched. The wandb val curve will step at the switch -- note the epoch.

Do this while no job is running on the run: a running job keeps the setting it
started with and would write its own config.json over the edit.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import time


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="*", help="run dirs (default: every run under --root)")
    p.add_argument("--root", default="trained_nets/mg_recon")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)

    runs = a.runs or sorted(d for d in glob.glob(os.path.join(a.root, "*", "*"))
                            if os.path.isfile(os.path.join(d, "config.json")))
    stamp = time.strftime("%Y%m%d-%H%M%S")
    changed = 0
    for run in runs:
        cfg_path = os.path.join(run, "config.json")
        if not os.path.isfile(cfg_path):
            print(f"[skip] {run}: no config.json")
            continue
        with open(cfg_path) as f:
            cfg = json.load(f)
        t = cfg.get("training", {})
        train_k = cfg.get("mri", {}).get("kspace_type", "simulated")
        val_k = t.get("val_kspace_type")
        if not val_k or val_k == train_k:
            continue
        changed += 1
        bv = os.path.join(run, "net_bestval.ckpt")
        moved = os.path.join(run, f"net_bestval.{val_k}.ckpt")
        print(f"{run}\n  validates on {val_k} -> {train_k} (as it trains)"
              + (f"\n  net_bestval.ckpt -> {os.path.basename(moved)}" if os.path.exists(bv) else ""))
        if a.dry_run:
            continue
        shutil.copy2(cfg_path, f"{cfg_path}.bak-{stamp}")
        del t["val_kspace_type"]
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=4, sort_keys=True)     # as save_config writes it
        if os.path.exists(bv):
            os.replace(bv, moved)
    print(f"\n{changed} run(s) {'would be ' if a.dry_run else ''}switched to measured "
          f"validation; {len(runs) - changed} already validate as they train.")


if __name__ == "__main__":
    main()
