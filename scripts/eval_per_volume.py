#!/usr/bin/env python3
"""
Score trained runs on EVERY volume of the val split, per slice and per volume.

    python scripts/eval_per_volume.py --anatomy knee --run lpdsnet_R8
    python scripts/eval_per_volume.py --anatomy knee --run varnetmaps_R8 lpdsnet_R8 \
        mg81v6_R8 mg169v6_R8 --metrics psnr ssim nrmse lpips
    python scripts/eval_per_volume.py --combine          # merge everything written

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
from evaluation.metrics import build_metrics, warm_up       # noqa: E402
from evaluation.tasks import build_adapter, default_sigma    # noqa: E402

# Display names of the main comparison. Keyed by CELL (the run name without
# its _R<r> suffix). Brain's MGLPDS-Big is the full-length `mglpds` run (Galerkin
# coarse Gram); knee's is `mg169v6`. A run not listed keeps its cell name.
LABELS = {
    "varnetmaps": "E2E-VarNet",
    "lpdsnet": "LPDSNet",
    "mg81v6": "MGLPDS-Small",
    "mg169v6": "MGLPDS-Big",
    "mglpds": "MGLPDS-Big",
}

DEFAULT_METRICS = ["psnr", "ssim", "nrmse", "lpips"]
META_KEYS = ("run", "anatomy", "R", "step", "sigma", "seed", "slices",
             "metrics", "n_volumes", "use_organ_mask", "rediscretize_embedded")


def cell_of(run):
    """`mg169v6_R12` -> `mg169v6`."""
    head, sep, tail = run.rpartition("_R")
    return head if sep and tail.isdigit() else run


def label_of(run):
    return LABELS.get(cell_of(run), cell_of(run))


def slice_seed(seed, volume, sl):
    """A per-slice seed: identical for every run, independent of visiting order."""
    return (int(seed) * 1_000_003 + zlib.crc32(f"{volume}:{sl}".encode())) % (2 ** 63)


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


@torch.no_grad()
def eval_run(run_dir, anatomy, args, metric_names, device):
    import dump_eval as de

    run = os.path.basename(os.path.normpath(run_dir))
    dest = os.path.join(args.out, anatomy)
    os.makedirs(dest, exist_ok=True)
    csv_path = os.path.join(dest, f"{run}.slices.csv")
    meta_path = os.path.join(dest, f"{run}.meta.json")

    cfg, model, n_params, step = de.load_run(run_dir, device)
    sigma = args.sigma if args.sigma is not None else default_sigma(cfg)
    slices = args.slices if args.slices is not None else train_slices(cfg)
    lo, hi = de.parse_slices(slices)
    volumes = de.resolve_volumes(cfg["data"]["val"], args.volumes, args.n_volumes)

    meta = dict(run=run, anatomy=anatomy, R=cfg.get("mri", {}).get("R"),
                step=int(step or 0), sigma=float(sigma), seed=int(args.seed),
                slices=slices, metrics=list(metric_names), n_volumes=len(volumes),
                use_organ_mask=bool(cfg.get("training", {}).get("use_organ_mask", False)),
                label=label_of(run), n_params=int(n_params),
                val_source={k: cfg["data"]["val"].get(k) for k in
                            ("name", "anatomy", "kspace_root", "smap_root", "scale_fac")},
                rediscretize_embedded=os.environ.get("IMMAP_REDISCRETIZE_EMBEDDED", "1"))

    # Resume only what was computed under the SAME settings; anything else
    # (a newer checkpoint, another sigma) starts the run's file over.
    if os.path.exists(meta_path) and not args.overwrite:
        with open(meta_path) as f:
            old = json.load(f)
        if any(old.get(k) != meta[k] for k in META_KEYS):
            changed = [k for k in META_KEYS if old.get(k) != meta[k]]
            print(f"  [{run}] settings changed ({', '.join(changed)}) -- starting over")
            if os.path.exists(csv_path):
                os.remove(csv_path)
    elif os.path.exists(csv_path) and (args.overwrite or not os.path.exists(meta_path)):
        os.remove(csv_path)                  # rows of unknown provenance
    _write_meta(meta_path, meta)

    val = dict(cfg["data"]["val"])
    val.update(enumerate_slices=True, start_slice=lo, end_slice=hi,
               volumes=volumes, batch_size=1, num_workers=args.workers)
    loader = build_loader(val, shuffle=False, drop_last=False)
    ds = loader.dataset
    n_slices_of = {}
    for f, sl in ds.index:
        v = os.path.splitext(f)[0]
        n_slices_of[v] = n_slices_of.get(v, 0) + 1

    done, kept = _done_volumes(csv_path, n_slices_of)
    todo = [i for i, (f, _) in enumerate(ds.index)
            if os.path.splitext(f)[0] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"  [{run}] {label_of(run)}  R={meta['R']}  sigma={sigma:g}  slices {slices}  "
          f"{len(volumes)} volumes / {len(ds)} slices; {len(done)} volumes already done")
    if not todo:
        return csv_path

    # Rewrite the file with only the complete volumes, then append.
    fields = ["volume", "slice"] + list(metric_names)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows({k: r[k] for k in fields} for r in kept)

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
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        for j, batch in enumerate(sub_loader):
            vol, sl = ds.item_id(todo[j])
            gen.manual_seed(slice_seed(args.seed, vol, sl))
            gt, recon, mmask = adapter(model, batch, cfg, device, sigma, gen)
            row = {"volume": vol, "slice": sl}
            for name, fn in metrics.items():
                v = float(fn(gt, recon, mask=mmask).reshape(-1)[0])
                row[name] = f"{v:.6g}"
                running[name].append(v)
            w.writerow(row)
            if (j + 1) % 200 == 0 or j + 1 == len(todo):
                f.flush()
                shown = "  ".join(
                    f"{m}={METRIC_REGISTRY[m]['fmt'].format(sum(x) / len(x))}"
                    for m, x in running.items())
                print(f"    {j + 1:>6}/{len(todo)}  {shown}", flush=True)

    del model
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
    p.add_argument("--sigma", type=float)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, help="stop after N slices (smoke test)")
    p.add_argument("--overwrite", action="store_true",
                   help="discard this run's previous rows instead of resuming")
    p.add_argument("--combine", action="store_true",
                   help="after evaluating (or alone), merge every run under --out")
    args = p.parse_args()

    if args.run:
        if not args.anatomy:
            raise SystemExit("--run needs --anatomy")
        device = torch.device(args.device or
                              ("cuda" if torch.cuda.is_available() else "cpu"))
        warm_up(args.metrics, device)        # LPIPS weights load now, not mid-run
        for run in args.run:
            run_dir = os.path.join(args.runs_root, args.anatomy, run)
            if not (os.path.exists(os.path.join(run_dir, "config.json"))
                    and os.path.exists(os.path.join(run_dir, "net.ckpt"))):
                print(f"[skip] {run_dir}: no config.json + net.ckpt")
                continue
            eval_run(run_dir, args.anatomy, args, args.metrics, device)
    elif not args.combine:
        raise SystemExit("nothing to do: give --anatomy and --run, or --combine")

    if args.combine:
        combine(args.out)


if __name__ == "__main__":
    main()
