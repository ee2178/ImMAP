#!/usr/bin/env python3
"""
Score trained runs on EVERY volume of the val split, per slice and per volume.

    python scripts/eval_per_volume.py --anatomy knee --run lpdsnet_R8
    python scripts/eval_per_volume.py --anatomy knee --run varnetmaps_R8 lpdsnet_R8 \
        mg81v6_R8 mg169v6_R8 --metrics psnr ssim nrmse lpips
    python scripts/eval_per_volume.py --combine          # merge everything written
    python scripts/eval_per_volume.py --anatomy knee --run lpdsnet_R8 --recompute nrmse

`--recompute nrmse` re-scores the rows a run ALREADY HAS for that metric only,
leaving every other column as it is -- for a metric whose definition changed
(NRMSE, 2026-10-20). It reruns inference (same per-slice noise, so the same
reconstructions) but skips the other metrics, LPIPS included. Resumable: its
progress goes to <run>.recompute-<metrics>.csv and is merged into the run's
file when every row is done. Each run's meta.json records the definition of
every metric it holds (`metric_versions`); a normal pass refuses to extend a
file whose values came from an older definition, and --combine leaves such
values out (blank) rather than mixing two definitions in one column.

For the main comparison's significance tests (scripts/wilcoxon_main.py), which
pair the nets VOLUME BY VOLUME. `scripts/evaluate.py` reports a mean and std
over one slice per volume; a paired test needs the per-volume values
themselves, over the whole val split.

Output (under --out, default results/main/):

    <anatomy>/<run>.slices.csv    one row per (volume, slice); written as the
                                  run goes, so an interrupted job resumes
    <anatomy>/<run>.meta.json     what the rows were computed under
    slices.csv, volumes.csv       --combine: every run merged; a volume's value
                                  is the MEAN of its slices' values

Same forward pass and metrics as evaluate.py and dump_eval.py
--------------------------------------------------------------
The adapter in evaluation/tasks.py and the per-sample metrics in
evaluation/metrics.py, unchanged, and dump_eval's `load_run` (which sets the
run's complex-conv mode and planar state). PSNR / SSIM / NRMSE are restricted to
the organ mask iff the run trained with `use_organ_mask`; LPIPS is never masked
(see evaluation.metrics.lpips).

What is held fixed across runs
------------------------------
* THE VOLUMES: the full val list, sorted, from the dataset's own file filter.
* THE SLICES: by default the range the run TRAINED on (`data.train`
  start_slice:end_slice -- knee 12:25, brain 0:8). The val config itself serves
  one slice per volume, too few for a per-volume value; slices outside the
  training range are ones no net saw. `--slices ''` takes whole volumes.
* THE NOISE: seeded PER SLICE from (seed, volume, slice) rather than from one
  generator walked through the set. Every run sees the same realisation on the
  same slice regardless of which volumes it was given or where a resumed job
  restarted. (So the noise differs from dump_eval's, which walks one generator;
  the two agree in distribution, not draw for draw.)
* SIGMA: pinned at each run's val_noise_std unless --sigma is given.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
import zlib

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import datasets                              # noqa: F401,E402  registers the loaders
from datasets.registry import build_loader   # noqa: E402
from evaluation.metrics import REGISTRY as METRIC_REGISTRY  # noqa: E402
from evaluation.metrics import build_metrics, metric_version, warm_up  # noqa: E402
from evaluation.tasks import (EVAL_KSPACE_TYPE, build_adapter, default_sigma,  # noqa: E402
                              force_organ_mask, metric_mask_source, val_kspace_type)

# Display names of the main comparison. Keyed by CELL (the run name without
# its _R<r> suffix). MGLPDS-Big is `mg169v6`; `mglpds` was brain's stopgap until
# 2026-10-21, and where both are present the plots keep mg169v6
# (wilcoxon_main.PREFERRED_CELLS). A run not listed keeps its cell name.
LABELS = {
    "varnetmaps": "E2E-VarNet",
    "lpdsnet": "LPDSNet",
    "mg81v6": "MGLPDS-Small",
    "mg169v6": "MGLPDS-Big",
    "mglpds": "MGLPDS-Big",
    "ram": "RAM",                # exp13: the trained RAM, random init, our protocol
    "ramdrunet": "RAM-DRUNet",   # the same net with a DRUNet-initialised trunk (not trained)
}

DEFAULT_METRICS = ["psnr", "ssim", "nrmse", "lpips"]
META_KEYS = ("run", "anatomy", "R", "step", "sigma", "seed", "slices", "val_kspace_type",
             "mask_source",
             "metrics", "n_volumes", "use_organ_mask", "rediscretize_embedded",
             "transfer_filter", "volume_select")


def cell_of(run):
    """`mg169v6_R12` -> `mg169v6`."""
    head, sep, tail = run.rpartition("_R")
    return head if sep and tail.isdigit() else run


def label_of(run):
    return LABELS.get(cell_of(run), cell_of(run))


def slice_seed(seed, volume, sl):
    """A per-slice seed: identical for every run, independent of visiting order."""
    return (int(seed) * 1_000_003 + zlib.crc32(f"{volume}:{sl}".encode())) % (2 ** 63)


def sample_volumes(names, n, seed=0):
    """`n` volumes drawn uniformly at random (fixed `seed`) from `names`, in
    sorted order -- the same draw for every run. The default first-N of a
    sorted list is one acquisition site for brain (file_brain_AXT2_200_*);
    a sweep that means to say something about the val set wants a spread."""
    names = sorted(names)
    if not n or n >= len(names):
        return names
    import numpy as np
    pick = np.random.default_rng(int(seed)).choice(len(names), int(n), replace=False)
    return [names[i] for i in sorted(pick)]


def train_slices(cfg):
    t = cfg["data"]["train"]
    lo, hi = t.get("start_slice", 0), t.get("end_slice")
    return f"{lo}:{'' if hi is None else hi}"


def _write_meta(path, meta):
    tmp = path + ".partial"
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=1)
    os.replace(tmp, path)


def _done_volumes(csv_path, n_slices_of):
    """Volumes already COMPLETE in a previous pass; a half-written one is redone."""
    if not os.path.exists(csv_path):
        return set(), []
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    seen = {}
    for r in rows:
        seen.setdefault(r["volume"], set()).add(int(r["slice"]))
    done = {v for v, s in seen.items() if len(s) == n_slices_of.get(v, -1)}
    return done, [r for r in rows if r["volume"] in done]


def _setup(run_dir, anatomy, args, metric_names, device):
    """Load the run and build its meta, loader and output paths."""
    import dump_eval as de

    run = os.path.basename(os.path.normpath(run_dir))
    dest = os.path.join(args.out, anatomy)
    os.makedirs(dest, exist_ok=True)

    cfg, model, n_params, step = de.load_run(run_dir, device)
    # --transfer-filter: the grid-transfer filter of a multigrid net swapped at
    # inference (same weights; operators/resample.py::set_transfer_filter), e.g.
    # "none" = the sinc off. Nets without grid transfers are scored as they are.
    from operators.resample import set_transfer_filter, transfer_filter_of
    filt_trained = transfer_filter_of(model) or ""
    filt = filt_trained
    if getattr(args, "transfer_filter", None):
        if filt_trained:
            set_transfer_filter(model, args.transfer_filter)
            filt = args.transfer_filter
            print(f"  [{run}] transfer filter {filt_trained} -> {filt} (inference swap)")
        else:
            print(f"  [{run}] no grid transfers: --transfer-filter does not apply")
    # Rows written before `val_kspace_type` was recorded were measured the
    # training way -- what the run's config said then (it had no val key).
    legacy = {"val_kspace_type": cfg.get("mri", {}).get("kspace_type", "simulated"),
              # ...and scored on whatever mask the run itself used
              "mask_source": metric_mask_source(cfg),
              # ...and with the transfer the net was built with
              "transfer_filter": filt_trained,
              # ...on the first N (or all) volumes
              "volume_select": "first"}
    if getattr(args, "mask", "organ") == "organ":
        force_organ_mask(cfg)
    if args.val_kspace_type:
        cfg.setdefault("training", {})["val_kspace_type"] = args.val_kspace_type
    sigma = args.sigma if args.sigma is not None else default_sigma(cfg)
    slices = args.slices if args.slices is not None else train_slices(cfg)
    lo, hi = de.parse_slices(slices)
    if getattr(args, "sample_volumes", None):
        if args.volumes or args.n_volumes:
            raise SystemExit("--sample-volumes picks the volumes itself; drop --volumes / --n-volumes")
        volumes = sample_volumes(de.resolve_volumes(cfg["data"]["val"], None, None),
                                 args.sample_volumes, args.sample_seed)
        volume_select = f"sample{args.sample_volumes}@{args.sample_seed}"
    else:
        volumes = de.resolve_volumes(cfg["data"]["val"], args.volumes, args.n_volumes)
        volume_select = "first"

    meta = dict(run=run, anatomy=anatomy, R=cfg.get("mri", {}).get("R"),
                step=int(step or 0), sigma=float(sigma), seed=int(args.seed),
                slices=slices, metrics=list(metric_names), n_volumes=len(volumes),
                use_organ_mask=bool(cfg.get("training", {}).get("use_organ_mask", False)),
                mask_source=metric_mask_source(cfg),
                label=label_of(run), n_params=int(n_params),
                val_source={k: cfg["data"]["val"].get(k) for k in
                            ("name", "anatomy", "kspace_root", "smap_root", "scale_fac")},
                rediscretize_embedded=os.environ.get("IMMAP_REDISCRETIZE_EMBEDDED", "1"),
                val_kspace_type=val_kspace_type(cfg),
                transfer_filter=filt,
                volume_select=volume_select,
                metric_versions={m: metric_version(m) for m in metric_names})

    val = dict(cfg["data"]["val"])
    val.update(enumerate_slices=True, start_slice=lo, end_slice=hi,
               volumes=volumes, batch_size=1, num_workers=args.workers)
    loader = build_loader(val, shuffle=False, drop_last=False)
    return dict(run=run, cfg=cfg, model=model, meta=meta, sigma=sigma, loader=loader,
                legacy=legacy,
                csv=os.path.join(dest, f"{run}.slices.csv"),
                meta_path=os.path.join(dest, f"{run}.meta.json"), dest=dest)


def _score(S, todo, metric_names, args, device, write_row, flush):
    """Push dataset items `todo` through the net; `write_row` gets each scored row."""
    ds, loader, cfg = S["loader"].dataset, S["loader"], S["cfg"]
    metrics = build_metrics(metric_names)
    adapter = build_adapter(cfg["task"])
    gen = torch.Generator(device=device)
    # Only the items still to do are loaded -- a Subset keeps the loader's
    # workers and collation without reading the finished volumes again.
    sub = torch.utils.data.Subset(ds, todo)
    sub_loader = torch.utils.data.DataLoader(
        sub, batch_size=1, shuffle=False, num_workers=args.workers,
        collate_fn=loader.collate_fn, pin_memory=device.type == "cuda")

    running = {m: [] for m in metric_names}
    for j, batch in enumerate(sub_loader):
        vol, sl = ds.item_id(todo[j])
        gen.manual_seed(slice_seed(args.seed, vol, sl))
        gt, recon, mmask = adapter(S["model"], batch, cfg, device, S["sigma"], gen)
        row = {"volume": vol, "slice": sl}
        for name, fn in metrics.items():
            v = float(fn(gt, recon, mask=mmask).reshape(-1)[0])
            row[name] = f"{v:.6g}"
            running[name].append(v)
        write_row(row)
        if (j + 1) % 200 == 0 or j + 1 == len(todo):
            flush()
            shown = "  ".join(
                f"{m}={METRIC_REGISTRY[m]['fmt'].format(sum(x) / len(x))}"
                for m, x in running.items())
            print(f"    {j + 1:>6}/{len(todo)}  {shown}", flush=True)


def _was(old_meta, key, S):
    """`key` as an older meta.json recorded it, or its value from before it was recorded."""
    return old_meta.get(key, S["legacy"].get(key))


def _stale(old_meta, names):
    """Metrics among `names` whose stored values used another definition."""
    have = old_meta.get("metric_versions", {})
    return [m for m in names if have.get(m, "v1") != metric_version(m)]


@torch.no_grad()
def eval_run(run_dir, anatomy, args, metric_names, device):
    S = _setup(run_dir, anatomy, args, metric_names, device)
    run, meta, csv_path, meta_path = S["run"], S["meta"], S["csv"], S["meta_path"]

    # Resume only what was computed under the SAME settings; anything else
    # (a newer checkpoint, another sigma) starts the run's file over.
    if os.path.exists(meta_path) and not args.overwrite:
        with open(meta_path) as f:
            old = json.load(f)
        changed = [k for k in META_KEYS if _was(old, k, S) != meta[k]]
        if "transfer_filter" in changed:
            # Starting over would DELETE the other filter's results for this run.
            raise SystemExit(
                f"[{run}] {meta_path} was scored with transfer filter "
                f"{_was(old, 'transfer_filter', S)!r}, this pass uses {meta['transfer_filter']!r}. "
                f"Give a filter swap its own --out (main_eval.sbatch's TRANSFER_FILTER does).")
        if changed:
            print(f"  [{run}] settings changed ({', '.join(changed)}) -- starting over")
            if os.path.exists(csv_path):
                os.remove(csv_path)
        elif os.path.exists(csv_path) and _stale(old, metric_names):
            # Extending this file would put two definitions in one column.
            stale = _stale(old, metric_names)
            raise SystemExit(
                f"[{run}] {csv_path} holds {', '.join(stale)} under an older "
                f"definition. Re-score it first:  --recompute {' '.join(stale)}  "
                f"(or --overwrite to redo every metric).")
    elif os.path.exists(csv_path) and (args.overwrite or not os.path.exists(meta_path)):
        os.remove(csv_path)                  # rows of unknown provenance
    _write_meta(meta_path, meta)

    ds = S["loader"].dataset
    n_slices_of = {}
    for f, sl in ds.index:
        v = os.path.splitext(f)[0]
        n_slices_of[v] = n_slices_of.get(v, 0) + 1

    done, kept = _done_volumes(csv_path, n_slices_of)
    todo = [i for i, (f, _) in enumerate(ds.index)
            if os.path.splitext(f)[0] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"  [{run}] {label_of(run)}  R={meta['R']}  sigma={S['sigma']:g}  "
          f"slices {meta['slices']}  {meta['n_volumes']} volumes / {len(ds)} slices; "
          f"{len(done)} volumes already done")
    if not todo:
        return csv_path

    # Rewrite the file with only the complete volumes, then append.
    fields = ["volume", "slice"] + list(metric_names)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows({k: r[k] for k in fields} for r in kept)

    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        _score(S, todo, metric_names, args, device, w.writerow, f.flush)

    del S["model"]
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return csv_path


@torch.no_grad()
def recompute_run(run_dir, anatomy, args, names, device):
    """Re-score the rows a run already has, for metrics `names` only."""
    S = _setup(run_dir, anatomy, args, names, device)
    run, meta, csv_path, meta_path = S["run"], S["meta"], S["csv"], S["meta_path"]
    if not (os.path.exists(csv_path) and os.path.exists(meta_path)):
        raise SystemExit(f"[{run}] nothing to recompute: no {csv_path} -- run it normally")
    with open(meta_path) as f:
        old = json.load(f)
    # The reconstructions must be the ones the kept columns came from.
    changed = [k for k in META_KEYS if k != "metrics" and _was(old, k, S) != meta[k]]
    if changed:
        raise SystemExit(
            f"[{run}] settings changed since its rows were computed ({', '.join(changed)}); "
            f"re-scoring one column would mix two setups. Rerun it with --overwrite.")
    missing = [m for m in names if m not in old["metrics"]]
    if missing:
        raise SystemExit(f"[{run}] has no {', '.join(missing)} column to recompute")

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    fields = list(rows[0].keys()) if rows else ["volume", "slice"] + old["metrics"]
    want = {(r["volume"], int(r["slice"])) for r in rows}

    part_path = os.path.join(S["dest"], f"{run}.recompute-{'-'.join(names)}.csv")
    got = {}
    if os.path.exists(part_path):
        with open(part_path, newline="") as f:
            for r in csv.DictReader(f):
                got[(r["volume"], int(r["slice"]))] = r
    ds = S["loader"].dataset
    ids = [ds.item_id(i) for i in range(len(ds))]
    lost = want - set(ids)
    if lost:
        raise SystemExit(f"[{run}] {len(lost)} stored rows are not in the current val set "
                         f"(e.g. {sorted(lost)[:2]}); rerun it with --overwrite")
    todo = [i for i, k in enumerate(ids) if k in want and k not in got]
    if args.limit:
        todo = todo[:args.limit]
    print(f"  [{run}] recompute {', '.join(names)}: {len(want)} rows, "
          f"{len(got)} already redone, {len(todo)} to go")

    if todo:
        new_file = not os.path.exists(part_path)
        with open(part_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["volume", "slice"] + list(names))
            if new_file:
                w.writeheader()

            def keep(row):
                w.writerow(row)
                got[(row["volume"], int(row["slice"]))] = row
            _score(S, todo, names, args, device, keep, f.flush)

    left = want - set(got)
    if left:
        print(f"  [{run}] {len(left)} rows still to recompute -- run again")
        return csv_path

    # Every row redone: swap the columns in, then record the new definitions.
    for r in rows:
        new = got[(r["volume"], int(r["slice"]))]
        for m in names:
            r[m] = new[m]
    tmp = f"{csv_path}.tmp{os.getpid()}"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, csv_path)
    old.setdefault("metric_versions", {})
    old["metric_versions"].update({m: metric_version(m) for m in names})
    _write_meta(meta_path, old)
    os.remove(part_path)
    print(f"  [{run}] {', '.join(names)} recomputed for all {len(rows)} rows")
    del S["model"]
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return csv_path


def combine(out_root):
    """Merge every <anatomy>/<run>.slices.csv into slices.csv and volumes.csv."""
    metas = sorted(glob.glob(os.path.join(out_root, "*", "*.meta.json")))
    if not metas:
        raise SystemExit(f"nothing to combine under {out_root}")
    s_rows, v_rows, metric_cols = [], [], []
    for mp in metas:
        with open(mp) as f:
            meta = json.load(f)
        cp = mp[:-len(".meta.json")] + ".slices.csv"
        if not os.path.exists(cp):
            continue
        with open(cp, newline="") as f:
            rows = list(csv.DictReader(f))
        ms = [m for m in meta["metrics"]]
        metric_cols += [m for m in ms if m not in metric_cols]
        # Values made under an older definition stay OUT of the merged files
        # (blank), so one column never holds two definitions.
        stale = _stale(meta, ms)
        if stale:
            print(f"  [{meta['run']}] {', '.join(stale)}: older definition -- left "
                  f"blank; re-score with --recompute {' '.join(stale)}")
            ms = [m for m in ms if m not in stale]
        # The label from today's LABELS, not the one stored when the run was
        # scored: renaming a net must not need its rows scored again.
        meta["label"] = label_of(meta["run"])
        key = dict(anatomy=meta["anatomy"], R=meta["R"], net=meta["label"],
                   run=meta["run"])
        per_vol = {}
        for r in rows:
            s_rows.append(dict(key, volume=r["volume"], slice=r["slice"],
                               **{m: r[m] for m in ms}))
            per_vol.setdefault(r["volume"], []).append(r)
        for vol, rs in per_vol.items():
            v_rows.append(dict(key, volume=vol, n_slices=len(rs), **{
                m: f"{math.fsum(float(r[m]) for r in rs) / len(rs):.6g}" for m in ms}))
        print(f"  {meta['anatomy']:<6} {meta['run']:<16} {meta['label']:<13} "
              f"{len(per_vol):>4} volumes / {len(rows):>5} slices "
              f"(of {meta['n_volumes']} volumes)")

    kinds = {}
    for mp in metas:
        with open(mp) as f:
            m = json.load(f)
        kinds.setdefault(m.get("val_kspace_type", "unrecorded (before 2026-10-20)"),
                         []).append(f"{m['anatomy']}/{m['run']}")
    if len(kinds) > 1:
        print("  [WARNING] runs here were measured differently -- their numbers are "
              "not comparable; keep each kind in its own --out:")
        for k, runs in sorted(kinds.items()):
            print(f"      {k}: {', '.join(runs)}")

    for name, rows, lead in (("slices.csv", s_rows, ["volume", "slice"]),
                             ("volumes.csv", v_rows, ["volume", "n_slices"])):
        path = os.path.join(out_root, name)
        # Written aside and renamed: torch/main_eval.sbatch merges at the end of
        # every task, so the file may be read (or copied) while another task is
        # rewriting it.
        tmp = f"{path}.tmp{os.getpid()}"
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["anatomy", "R", "net", "run"] + lead
                               + metric_cols, restval="")
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, path)
        print(f"wrote {len(rows)} rows to {path}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-root", default="trained_nets/mg_recon",
                   help="holds <anatomy>/<run>/{config.json,net.ckpt}")
    p.add_argument("--anatomy", choices=["brain", "knee"])
    p.add_argument("--run", nargs="*", default=[],
                   help="run directory NAMES under <runs-root>/<anatomy>")
    p.add_argument("--out", default="results/main")
    p.add_argument("--metrics", nargs="*", default=DEFAULT_METRICS,
                   help=f"subset of {sorted(METRIC_REGISTRY)}")
    p.add_argument("--slices", help="'lo:hi' per volume; default = the run's "
                                    "training range; '' = whole volumes")
    p.add_argument("--volumes", nargs="*", help="explicit volume names (smoke tests)")
    p.add_argument("--n-volumes", type=int, default=None,
                   help="first N volumes only (smoke tests); default all")
    p.add_argument("--sample-volumes", type=int, metavar="N",
                   help="N volumes drawn at random (fixed --sample-seed) from the whole "
                        "val split, the same for every run -- for sweeps that score a "
                        "subset; the first N of brain is one acquisition site")
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument("--sigma", type=float)
    p.add_argument("--mask", choices=["organ", "run"], default="organ",
                   help="'organ' (default): score every run inside the organ mask "
                        "from its stored coil maps (background out of every metric; "
                        "LPIPS compares with the background zeroed). 'run': the mask "
                        "setting each run trained with")
    p.add_argument("--val-kspace-type", choices=["simulated", "measurement_awgn"],
                   default=EVAL_KSPACE_TYPE,
                   help=f"how the evaluation measurement is made (default "
                        f"{EVAL_KSPACE_TYPE}: measured k-space + AWGN, as the nets train, "
                        f"whatever the run's config says). 'simulated': the SENSE ground "
                        f"truth through the stored maps, mask and AWGN, so the noise is "
                        f"exactly sigma"),
    p.add_argument("--transfer-filter",
                   help="swap the grid-transfer filter of multigrid runs at inference "
                        "(same weights): box, bilinear, ..., or none = the sinc off; "
                        "default: as trained. Use its own --out.")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, help="stop after N slices (smoke test)")
    p.add_argument("--overwrite", action="store_true",
                   help="discard this run's previous rows instead of resuming")
    p.add_argument("--recompute", nargs="+", metavar="METRIC",
                   help="re-score these metrics on the rows each run already has, "
                        "keeping every other column (e.g. --recompute nrmse)")
    p.add_argument("--combine", action="store_true",
                   help="after evaluating (or alone), merge every run under --out")
    args = p.parse_args()

    if args.run:
        if not args.anatomy:
            raise SystemExit("--run needs --anatomy")
        device = torch.device(args.device or
                              ("cuda" if torch.cuda.is_available() else "cpu"))
        warm_up(args.recompute or args.metrics, device)   # LPIPS loads now, not mid-run
        for run in args.run:
            run_dir = os.path.join(args.runs_root, args.anatomy, run)
            if not (os.path.exists(os.path.join(run_dir, "config.json"))
                    and os.path.exists(os.path.join(run_dir, "net.ckpt"))):
                print(f"[skip] {run_dir}: no config.json + net.ckpt")
                continue
            if args.recompute:
                build_metrics(args.recompute)    # a typo fails before any inference
                recompute_run(run_dir, args.anatomy, args, args.recompute, device)
            else:
                eval_run(run_dir, args.anatomy, args, args.metrics, device)
    elif not args.combine:
        raise SystemExit("nothing to do: give --anatomy and --run, or --combine")

    if args.combine:
        combine(args.out)


if __name__ == "__main__":
    main()
