#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N-way split of a dataset directory into SYMLINK DIRECTORIES, with optional grouping.

Dataset-agnostic. Takes a directory whose entries are one folder per case and produces one
sibling directory per split, whose entries are relative symlinks to the real folders:

    <source>_train/<case> -> <source>/<case>
    <source>_val/...
    <source>_test/...

so any loader that scans a root of case folders (I2SBDataset.index_img_from_root,
CCLPretrainDataset, ...) reads any split directly and nothing is copied. Pass --dest to
nest them as <dest>/{train,val,test} instead of siblings.

GROUPING (--group) is the part that must match the dataset, and getting it wrong is silent:

    none      one case per folder; randomize over folders.        BraTS: BraTS2021_00061
    prefix    group by the name up to the first underscore.       NYUMets: NYU0123_2019-05-05
    regex     group by --group-regex (capture group 1 if present).

Grouping exists for LONGITUDINAL data. NYUMets holds one folder per SESSION and a patient
owns several, so splitting at the folder level would put visits 1 and 3 of a patient in
train and visit 2 in val -- the same anatomy, often weeks apart, on both sides of the split,
and val/test scores that read high for reasons unrelated to generalization.

Do not carry a --group setting between datasets. `prefix` on BraTS maps every subject
(BraTS2021_00061, BraTS2021_00062, ...) to the single key "BraTS2021", which would drop the
entire dataset into one group and one split. The script prints the group count and refuses a
split that collapses to a single group.

Patient counts use LARGEST-REMAINDER allocation, so the splits always sum to exactly the
number of patients -- no subject is silently dropped or double-assigned by rounding. Note
that SESSION counts will not match the requested ratios as closely as patient counts do,
because patients contribute different numbers of visits; the printout shows both.

Split is deterministic given --seed and recorded to <source>_split_record.json. Refuses to
overwrite unless --force, and even then removes only symlinks.

Run it as a FILE, not `python -m`: datasets/__init__.py eagerly imports every registered
loader (torchvision, wandb, ...) and this script needs none of them.

    python datasets/NYUMets/make_split_symlinks.py \
        --source ../datasets/NYUMets_h5 --splits train=0.6,val=0.3,test=0.1 --seed 0

    # or from a JSON config: {"splits": {"train": 0.6, "val": 0.3, "test": 0.1}, "seed": 0}
    python datasets/NYUMets/make_split_symlinks.py --source ... --config split.json
"""

import os
import re
import json
import math
import random
import argparse
import datetime
from collections import defaultdict

DEFAULT_SPLITS = {"train": 0.6, "val": 0.3, "test": 0.1}


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


def make_group_key(mode, pattern=None):
    """-> f(dirname) giving the key that must not straddle two splits."""
    if mode == "none":
        return lambda name: name
    if mode == "prefix":
        return lambda name: name.split("_", 1)[0]
    if mode == "regex":
        if not pattern:
            raise SystemExit("--group regex needs --group-regex")
        rx = re.compile(pattern)

        def key(name):
            m = rx.search(name)
            if not m:
                raise SystemExit(f"--group-regex {pattern!r} does not match {name!r}")
            return m.group(1) if m.groups() else m.group(0)
        return key
    raise SystemExit(f"--group must be none|prefix|regex, got {mode!r}")


def parse_splits(spec):
    """'train=0.6,val=0.3,test=0.1' -> {name: frac}, order preserved."""
    out = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit(f"--splits entry {part!r} is not name=fraction")
        name, _, frac = part.partition("=")
        name = name.strip()
        if name in out:
            raise SystemExit(f"--splits names a split twice: {name!r}")
        try:
            out[name] = float(frac)
        except ValueError:
            raise SystemExit(f"--splits fraction for {name!r} is not a number: {frac!r}")
    if not out:
        raise SystemExit("--splits parsed to nothing")
    return out


def allocate(n, fracs):
    """Largest-remainder allocation of n items over `fracs`. Sums to exactly n.

    Plain rounding would over- or under-allocate and silently drop or duplicate a patient;
    flooring then handing the leftovers to the largest fractional parts keeps the totals
    exact and the proportions as close as integers allow.
    """
    raw = [n * f for f in fracs]
    base = [int(math.floor(r)) for r in raw]
    leftover = n - sum(base)
    order = sorted(range(len(raw)), key=lambda i: (raw[i] - base[i], raw[i]), reverse=True)
    for i in order[:leftover]:
        base[i] += 1
    return base


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="dir with one folder per case")
    ap.add_argument("--group", default=None, choices=["none", "prefix", "regex"],
                    help="what must not straddle splits: none (per folder, BraTS) | "
                         "prefix (up to first '_', NYUMets patient) | regex. Default none.")
    ap.add_argument("--group-regex", default=None, dest="group_regex",
                    help="with --group regex; capture group 1 is the key if present")
    ap.add_argument("--splits", default=None,
                    help="inline spec, e.g. train=0.6,val=0.3,test=0.1 (default) -- "
                         "overrides --config")
    ap.add_argument("--config", default=None,
                    help='JSON with {"splits": {...}, "seed": int} ')
    ap.add_argument("--dest", default=None,
                    help="nest splits as <dest>/<name> instead of sibling <source>_<name>")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    file_cfg = {}
    if args.config:
        with open(args.config) as f:
            file_cfg = json.load(f)

    splits = (parse_splits(args.splits) if args.splits
              else dict(file_cfg.get("splits") or DEFAULT_SPLITS))
    seed = args.seed if args.seed is not None else int(file_cfg.get("seed", 0))
    group = args.group or file_cfg.get("group", "none")
    group_regex = args.group_regex or file_cfg.get("group_regex")

    total = sum(splits.values())
    if abs(total - 1.0) > 1e-6:
        raise SystemExit(f"split fractions sum to {total:g}, not 1.0: {splits}")
    if any(v < 0 for v in splits.values()):
        raise SystemExit(f"negative split fraction: {splits}")

    source = os.path.abspath(args.source)
    dirs = {name: (os.path.join(args.dest, name) if args.dest else f"{source}_{name}")
            for name in splits}
    record = source + "_split_record.json"

    sessions = sorted(d for d in os.listdir(source)
                      if os.path.isdir(os.path.join(source, d)))
    if not sessions:
        raise SystemExit(f"no session directories under {source}")

    key_of = make_group_key(group, group_regex)
    by_patient = defaultdict(list)
    for s in sessions:
        by_patient[key_of(s)].append(s)
    patients = sorted(by_patient)

    # A grouping rule meant for another dataset collapses everything into one key and would
    # hand the whole set to a single split -- silently, since the symlinks still get made.
    if len(patients) == 1 and len(sessions) > 1:
        raise SystemExit(
            f"--group {group} maps all {len(sessions)} folders to one key "
            f"({patients[0]!r}); that would put the entire dataset in one split. "
            f"Use --group none for one-folder-per-case data.")

    if os.path.exists(record) and not args.force:
        raise SystemExit(f"{record} exists; pass --force to redo the split.")

    names = list(splits)
    counts = allocate(len(patients), [splits[n] for n in names])
    for n, c in zip(names, counts):
        if splits[n] > 0 and c == 0:
            print(f"  !! split '{n}' asked for {splits[n]:.0%} but only "
                  f"{len(patients)} groups exist -- it gets ZERO")

    rng = random.Random(seed)
    shuffled = patients[:]
    rng.shuffle(shuffled)

    assignment, at = {}, 0
    for n, c in zip(names, counts):
        for p in shuffled[at:at + c]:
            assignment[p] = n
        at += c

    for d in dirs.values():
        _clean_link_dir(d)

    n_sess = defaultdict(int)
    for p in patients:
        split = assignment[p]
        dst_root = dirs[split]
        for s in by_patient[p]:
            os.symlink(os.path.relpath(os.path.join(source, s), dst_root),
                       os.path.join(dst_root, s))
            n_sess[split] += 1

    with open(record, "w") as f:
        json.dump({"created": datetime.datetime.now().isoformat(timespec="seconds"),
                   "source": source, "seed": seed, "splits": splits,
                   "grouped_by": group, "group_regex": group_regex,
                   "n_patients": len(patients), "n_sessions": len(sessions),
                   "dirs": dirs,
                   "patients": {n: sorted(p for p, s in assignment.items() if s == n)
                                for n in names}}, f, indent=2)

    print(f"{len(patients)} groups / {len(sessions)} folders   "
          f"group={group}   seed {seed}")
    print(f"  {'split':<8} {'want':>6} {'groups':>10} {'folders':>10}  dir")
    for n, c in zip(names, counts):
        print(f"  {n:<8} {splits[n]:>5.0%} {c:>10} {n_sess[n]:>10}  {dirs[n]}")
    print(f"  record -> {record}")


if __name__ == "__main__":
    main()
