#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train/val split of the NYUMets h5 set as SYMLINK DIRECTORIES, grouped BY PATIENT.

Same output contract as datasets/BraTS/make_split_symlinks.py -- two sibling dirs whose
entries are relative symlinks to the real session folders --

    <source>_train/<patient>_<session> -> <source>/<patient>_<session>
    <source>_val/<patient>_<session>   -> <source>/<patient>_<session>

so I2SBDataset.index_img_from_root reads either one directly and nothing is copied.

THE DIFFERENCE FROM THE BraTS SPLIT, and the reason this file exists: NYUMets is
longitudinal, so `<source>` holds one directory PER SESSION and a patient owns several of
them. Splitting at the directory level -- which is all the BraTS script knows how to do --
would put visits 1 and 3 of the same patient in train and visit 2 in val. The same anatomy,
often weeks apart, on both sides of the split: val PSNR would read high for reasons that
have nothing to do with generalization. So randomization happens over PATIENTS and every
session of a patient follows its patient.

The patient key is the directory name up to the first underscore, matching the
`<patient>_<session>` names preprocessing/nyumets_h5.py writes.

Split is deterministic given --seed and recorded to <source>_split_record.json. Refuses to
overwrite unless --force, and even then removes only symlinks.

Run it as a FILE, not `python -m`: datasets/__init__.py eagerly imports every registered
loader (torchvision, wandb, ...) and this script needs none of them.

    python datasets/NYUMets/make_split_symlinks.py \
        --source ~/scratch/datasets/NYUMets_h5 --val_frac 0.30 --seed 0
"""

import os
import json
import random
import argparse
import datetime
from collections import defaultdict


def _clean_link_dir(d):
    """Make d an empty dir, removing ONLY symlinks. Refuse if it holds real entries."""
    if os.path.islink(d):
        raise SystemExit(f"{d} is itself a symlink; refusing.")
    if os.path.isdir(d):
        for name in os.listdir(d):
            p = os.path.join(d, name)
            if os.path.islink(p):
                os.unlink(p)
            else:
                raise SystemExit(f"Refusing to overwrite: {p} is not a symlink.")
    else:
        os.makedirs(d, exist_ok=True)


def patient_of(dirname):
    """'NYU0123_2019-05-05' -> 'NYU0123'."""
    return dirname.split("_", 1)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="dir of <patient>_<session> folders")
    ap.add_argument("--train_name", default=None, help="default: <source>_train")
    ap.add_argument("--val_name", default=None, help="default: <source>_val")
    ap.add_argument("--val_frac", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    source = os.path.abspath(args.source)
    train_dir = args.train_name or source + "_train"
    val_dir = args.val_name or source + "_val"
    record = source + "_split_record.json"

    sessions = sorted(d for d in os.listdir(source)
                      if os.path.isdir(os.path.join(source, d)) and not d.endswith(".h5"))
    if not sessions:
        raise SystemExit(f"no session directories under {source}")

    by_patient = defaultdict(list)
    for s in sessions:
        by_patient[patient_of(s)].append(s)
    patients = sorted(by_patient)

    if os.path.exists(record) and not args.force:
        raise SystemExit(f"{record} exists; pass --force to redo the split.")

    rng = random.Random(args.seed)
    shuffled = patients[:]
    rng.shuffle(shuffled)
    n_val = int(round(args.val_frac * len(shuffled)))
    val_patients = set(shuffled[:n_val])

    _clean_link_dir(train_dir)
    _clean_link_dir(val_dir)

    counts = {"train": 0, "val": 0}
    for p in patients:
        dst_root = val_dir if p in val_patients else train_dir
        split = "val" if p in val_patients else "train"
        for s in by_patient[p]:
            os.symlink(os.path.relpath(os.path.join(source, s), dst_root),
                       os.path.join(dst_root, s))
            counts[split] += 1

    with open(record, "w") as f:
        json.dump({"created": datetime.datetime.now().isoformat(timespec="seconds"),
                   "source": source, "seed": args.seed, "val_frac": args.val_frac,
                   "grouped_by": "patient",
                   "n_patients": len(patients), "n_sessions": len(sessions),
                   "val_patients": sorted(val_patients),
                   "train_patients": sorted(set(patients) - val_patients)}, f, indent=2)

    print(f"{len(patients)} patients / {len(sessions)} sessions")
    print(f"  train: {len(patients) - len(val_patients):>4} patients  "
          f"{counts['train']:>5} sessions -> {train_dir}")
    print(f"  val  : {len(val_patients):>4} patients  {counts['val']:>5} sessions -> {val_dir}")
    print(f"  record -> {record}")


if __name__ == "__main__":
    main()
