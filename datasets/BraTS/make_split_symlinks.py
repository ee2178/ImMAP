#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-time, reproducible 70/30 train/val split as SYMLINK DIRECTORIES.

Creates two sibling directories next to the source dataset:
    <source>_train/<subject> -> <source>/<subject>     (relative symlink)
    <source>_val/<subject>   -> <source>/<subject>
so you can point your pipeline at a clean train path and val path. The original
data is never moved or copied; each subject folder (niftis + in-place constraint-map
H5s) is reachable through its symlink.

Split is at the SUBJECT level, deterministic given --seed, and recorded to
<source>_split_record.json. Refuses to overwrite an existing split unless --force
(and even with --force it only ever removes symlinks, never real files/dirs).

Usage:
    python make_split_symlinks.py \
        --source /home/ee2178/scratch/ee2178/datasets/BraTS/BraTS2021_DataSet \
        --val_frac 0.30 --seed 0
"""

import os
import json
import random
import argparse
import datetime


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="dataset dir with one folder per subject")
    ap.add_argument("--train_name", default=None, help="default: <source>_train")
    ap.add_argument("--val_name", default=None, help="default: <source>_val")
    ap.add_argument("--val_frac", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    source = os.path.abspath(args.source.rstrip("/"))
    parent = os.path.dirname(source)
    base = os.path.basename(source)
    train_dir = args.train_name or os.path.join(parent, base + "_train")
    val_dir = args.val_name or os.path.join(parent, base + "_val")
    record_path = os.path.join(parent, base + "_split_record.json")

    if os.path.exists(record_path) and not args.force:
        rec = json.load(open(record_path))
        raise SystemExit(
            f"Split record already exists: {record_path}\n"
            f"  seed={rec['seed']} val_frac={rec['val_frac']} "
            f"n_train={rec['n_train']} n_val={rec['n_val']}\n"
            f"Refusing to overwrite (use --force). The split is preserved.")

    subjects = sorted(d for d in os.listdir(source)
                      if os.path.isdir(os.path.join(source, d)))
    rng = random.Random(args.seed)
    order = list(range(len(subjects)))
    rng.shuffle(order)
    n_val = round(len(subjects) * args.val_frac)
    val_pos = set(order[:n_val])
    train_subj = [subjects[i] for i in range(len(subjects)) if i not in val_pos]
    val_subj = [subjects[i] for i in range(len(subjects)) if i in val_pos]

    _clean_link_dir(train_dir)
    _clean_link_dir(val_dir)

    def link(subj, dst_dir):
        target = os.path.relpath(os.path.join(source, subj), start=dst_dir)  # relative for portability
        os.symlink(target, os.path.join(dst_dir, subj))

    for s in train_subj:
        link(s, train_dir)
    for s in val_subj:
        link(s, val_dir)

    record = {
        "source": source,
        "train_dir": train_dir,
        "val_dir": val_dir,
        "seed": args.seed,
        "val_frac": args.val_frac,
        "split_level": "subject",
        "link_type": "relative_symlink",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "n_total": len(subjects),
        "n_train": len(train_subj),
        "n_val": len(val_subj),
        "train_subjects": train_subj,
        "val_subjects": val_subj,
    }
    with open(record_path, "w") as f:
        json.dump(record, f, indent=2)

    print(f"Train: {len(train_subj)} subjects -> {train_dir}")
    print(f"Val  : {len(val_subj)} subjects -> {val_dir}")
    print(f"Record: {record_path}")


if __name__ == "__main__":
    main()
