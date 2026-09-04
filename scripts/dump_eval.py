#!/usr/bin/env python3
"""
Dump per-volume reconstructions to HDF5, for the figure viewer to read.

    python scripts/dump_eval.py --runs trained_nets/mg_recon/brain --slices 0:16
    python scripts/dump_eval.py --runs trained_nets/mg_recon/brain \
        --only lpdsnet_R8 mglpds_R8 mggrouplpds_R8 --n-volumes 6 --slices 4:20

Why a dump exists at all
------------------------
`scripts/evaluate.py` scores runs and throws the images away; the comparison
notebooks keep the images but rebuild every network in process, which is far
too slow to scrub slices through. Everything downstream of this script --
`figures/viewer.py`, and any figure built from its picks -- reads ONLY HDF5 +
numpy, so it needs neither torch nor a GPU and runs on a laptop against an
rsync'd copy of the output.

The forward pass is `evaluation/tasks.py`'s adapter, unchanged, so the images
here and the numbers in `results/*.csv` come from the same code path. If that
stops being true the viewer starts lying, which is the whole reason the dump
does not reimplement the measurement.

What is held fixed
------------------
Exactly what `evaluate.py` holds fixed, for the same reasons: one seeded noise
generator reset per run, a pinned sigma rather than a sampled one, and
`shuffle=False`. Two things this script adds:

* THE SAME VOLUMES for every run. The dataset's file list comes from
  `os.listdir`, whose order is not guaranteed, so the volume selection is
  sorted and resolved ONCE and then passed to every run explicitly. Columns of
  a comparison figure that showed different patients would be worse than
  useless.
* WHOLE SLICE RANGES. The val split serves one slice per volume
  (`start_slice: 0, end_slice: 1`); a viewer needs a stack to scrub, so the
  loader is rebuilt here with `enumerate_slices=True`.

Output
------
`<run_dir>/eval_dump/<volume>.h5` by default, or mirrored under `--out`:

    reference       (n, H, W) complex64   ground truth
    recon           (n, H, W) complex64   network output
    zero_filled     (n, H, W) complex64   E^H y -- the artifact to be removed
    organ_mask      (n, H, W) uint8       coil-support anatomy mask
    sampling_mask   (n, H, W) uint8       k-space pattern
    slice_index     (n,)      int32       source slice number in the volume
    <metric>_slice  (n,)      float32     one value per slice

with the run, sigma, seed, R and mask policy in the file attributes. Arrays are
(nslice, H, W) row-major with no transpose needed -- unlike the Julia-written
evals this tooling was modelled on.

A dump is only comparable to another dump made at the same sigma, seed and R;
`figures/common.py` checks those attributes when it loads a set of columns and
refuses to put mismatched ones side by side.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datasets                              # noqa: F401  registers the loaders
from datasets.registry import build_loader
from evaluation.metrics import DEFAULT_METRICS, REGISTRY as METRIC_REGISTRY
from evaluation.metrics import build_metrics, warm_up
from evaluation.tasks import build_adapter, default_sigma

DUMP_VERSION = 1


# ---------------------------------------------------------------------------
# run discovery
# ---------------------------------------------------------------------------
def find_runs(root, only=None):
    """Every directory under `root` holding both a config.json and a net.ckpt.

    `only` filters by the run's path relative to `root` -- a substring match, so
    `--only R8` takes every R=8 cell and `--only mglpds_R8` takes one.
    """
    out = []
    for dirpath, _, filenames in os.walk(root):
        if "config.json" in filenames and "net.ckpt" in filenames:
            out.append(dirpath)
    out = sorted(out)
    if only:
        out = [d for d in out
               if any(o in os.path.relpath(d, root).replace(os.sep, "/")
                      for o in only)]
    return out


def load_run(run_dir, device):
    from models import build_model            # imported late: pulls in CUDA bits

    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)

    model = build_model(cfg).to(device)
    ckpt = torch.load(os.path.join(run_dir, "net.ckpt"), map_location=device,
                      weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    if getattr(model, "attn_backend", None) == "flex" and device.type == "cuda":
        model.compile_flex()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return cfg, model, n_params, ckpt.get("step")


def out_dir(run_dir, runs_root, out_root):
    """Where this run's volumes go: beside the checkpoint, or mirrored under --out."""
    if out_root is None:
        return os.path.join(run_dir, "eval_dump")
    rel = os.path.relpath(run_dir, runs_root)
    return os.path.join(out_root, "." if rel == "." else rel)


# ---------------------------------------------------------------------------
# volume selection
# ---------------------------------------------------------------------------
def parse_slices(spec):
    """"4:20" -> (4, 20); "8" -> (0, 8); "" / None -> (0, None) = whole volume."""
    if not spec:
        return 0, None
    if ":" not in spec:
        return 0, int(spec)
    lo, _, hi = spec.partition(":")
    return (int(lo) if lo else 0), (int(hi) if hi else None)


def resolve_volumes(val_cfg, names, n_volumes):
    """The volume list every run will be given, sorted for reproducibility.

    Built by instantiating the dataset once and reading its file list rather
    than globbing, so the anatomy filter (`CORPD_FBK` / `T2`) is applied by the
    same code that applies it during training.
    """
    if names:
        return list(names)

    probe = dict(val_cfg)
    probe.update(batch_size=1, num_workers=0)
    ds = build_loader(probe, shuffle=False, drop_last=False).dataset
    files = getattr(ds, "files", None)
    if files is None:
        raise SystemExit(
            f"{type(ds).__name__} has no `.files`, so the dump cannot resolve a "
            f"volume list; pass --volumes explicitly, or give the dataset a "
            f"`files` attribute and an `item_id(idx)` method")
    vols = sorted(os.path.splitext(f)[0] for f in files)
    if n_volumes:
        vols = vols[:n_volumes]
    return vols


def same_source(a, b):
    """Do two val configs read the same images? Columns must, or the figure lies."""
    keys = ("name", "anatomy", "kspace_root", "smap_root", "scale_fac")
    return all(a.get(k) == b.get(k) for k in keys)


# ---------------------------------------------------------------------------
# the dump itself
# ---------------------------------------------------------------------------
def _np(x):
    """(B=1, C=1, H, W) torch tensor -> (H, W) numpy, complex preserved."""
    a = x.detach().cpu().numpy()
    while a.ndim > 2 and a.shape[0] == 1:
        a = a[0]
    if a.ndim > 2:                            # multi-channel: take the first
        a = a[0]
    return a


def write_volume(path, buf, attrs):
    import h5py

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        for k, v in attrs.items():
            f.attrs[k] = v
        for k, v in buf.items():
            if not v:
                continue
            arr = np.stack(v)
            # Compression matters here: a 32-slice brain volume is ~80 MB of
            # complex64 across the three image stacks, and these get copied off
            # the cluster by hand.
            f.create_dataset(k, data=arr, compression="gzip", compression_opts=4)


@torch.no_grad()
def dump_run(run_dir, cfg, model, n_params, step, args, volumes, metrics,
             sigma, dest):
    """Push every (volume, slice) through the net and write one HDF5 per volume."""
    lo, hi = parse_slices(args.slices)

    val = dict(cfg["data"]["val"])
    val.update(enumerate_slices=True, start_slice=lo, end_slice=hi,
               volumes=volumes, batch_size=1, num_workers=args.workers)
    loader = build_loader(val, shuffle=False, drop_last=False)
    ds = loader.dataset

    adapter = build_adapter(cfg["task"])
    device = next(model.parameters()).device
    # Reset per run, exactly as evaluate.py does, so run 0 and run 7 see the
    # same noise realisation on the same slice.
    gen = torch.Generator(device=device).manual_seed(int(args.seed))

    mri = cfg.get("mri", {})
    base_attrs = {
        "dump_version": DUMP_VERSION,
        "run": os.path.basename(os.path.normpath(run_dir)),
        "run_dir": run_dir,
        "task": cfg["task"],
        "model_type": cfg["model"]["type"],
        "experiment": cfg.get("experiment", {}).get("name", ""),
        "anatomy": val.get("anatomy", ""),
        "R": mri.get("R", 0),
        "acs_lines": mri.get("acs_lines", 0),
        "mask_dist": mri.get("mask_dist", ""),
        "sigma": float(sigma),
        "seed": int(args.seed),
        "scale_fac": float(val.get("scale_fac", 1.0) or 1.0),
        "use_organ_mask": bool(cfg.get("training", {}).get("use_organ_mask", False)),
        "n_params": int(n_params),
        "step": int(step or 0),
        "metrics": list(metrics),
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
    }

    buf, cur, written = {}, None, 0

    def flush():
        nonlocal buf, cur
        if cur is None:
            return
        write_volume(os.path.join(dest, f"{cur}.h5"), buf,
                     dict(base_attrs, volume=cur))
        buf, cur = {}, None

    for i, batch in enumerate(loader):
        if args.limit and i >= args.limit:
            break
        vol, sl = ds.item_id(i)
        if vol != cur:
            flush()
            cur = vol

        extras = {}
        gt, recon, metric_mask = adapter(model, batch, cfg, device, sigma, gen,
                                         extras=extras)

        buf.setdefault("reference", []).append(_np(gt).astype(np.complex64))
        buf.setdefault("recon", []).append(_np(recon).astype(np.complex64))
        if "zero_filled" in extras:
            buf.setdefault("zero_filled", []).append(
                _np(extras["zero_filled"]).astype(np.complex64))
        if "sampling_mask" in extras:
            buf.setdefault("sampling_mask", []).append(
                (np.abs(_np(extras["sampling_mask"])) > 0).astype(np.uint8))
        # The ANATOMY mask is taken from the batch, not from the adapter: the
        # adapter returns None for it whenever the run did not train with it,
        # but the viewer still wants it to crop the FOV and to window the
        # display. Whether it entered the metrics is a separate question, and
        # `use_organ_mask` in the attrs is what answers it.
        if cfg["task"] == "recon" and len(batch) >= 4:
            buf.setdefault("organ_mask", []).append(
                (np.abs(_np(batch[3])) > 0).astype(np.uint8))
        buf.setdefault("slice_index", []).append(np.int32(sl))

        for name, fn in metrics.items():
            v = fn(gt, recon, mask=metric_mask).reshape(-1)[0]
            buf.setdefault(f"{name}_slice", []).append(np.float32(v.item()))
        written += 1

    flush()
    return written


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default="trained_nets/mg_recon/brain",
                   help="root of the trained-run tree (or one run directory)")
    p.add_argument("--only", nargs="*",
                   help="substring filter on the run path, e.g. --only R8")
    p.add_argument("--out", help="mirror the run tree here instead of writing "
                                 "into each <run>/eval_dump/")
    p.add_argument("--slices", default="0:16",
                   help="slice range per volume, 'lo:hi' (hi exclusive); "
                        "'' means the whole volume")
    p.add_argument("--volumes", nargs="*",
                   help="explicit volume names (no .h5); default is the first "
                        "--n-volumes in sorted order")
    p.add_argument("--n-volumes", type=int, default=8,
                   help="how many volumes to dump when --volumes is not given")
    p.add_argument("--metrics", nargs="*", default=DEFAULT_METRICS,
                   help=f"subset of {sorted(METRIC_REGISTRY)}")
    p.add_argument("--sigma", type=float,
                   help="noise level; default is each run's val_noise_std. "
                        "One value only -- a dump is a single operating point, "
                        "and columns at different sigmas are not comparable")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, help="stop after N slices (smoke test)")
    args = p.parse_args()

    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    metrics = build_metrics(args.metrics)
    warm_up(args.metrics, device)             # LPIPS downloads now, not mid-sweep

    runs = find_runs(args.runs, args.only)
    if not runs:
        raise SystemExit(
            f"no runs with both config.json and net.ckpt under {args.runs}"
            + (f" matching {args.only}" if args.only else ""))

    # One volume list for the whole sweep, resolved from the first run's val
    # config and checked against the rest.
    with open(os.path.join(runs[0], "config.json")) as f:
        first_val = json.load(f)["data"]["val"]
    volumes = resolve_volumes(first_val, args.volumes, args.n_volumes)
    print(f"{len(runs)} run(s) | {len(volumes)} volume(s) | slices "
          f"{args.slices or 'all'} | device {device}")
    print(f"volumes: {', '.join(volumes)}")

    total = 0
    for run_dir in runs:
        rel = os.path.relpath(run_dir, args.runs)
        try:
            cfg, model, n_params, step = load_run(run_dir, device)
        except Exception as e:                # a bad cell must not sink the sweep
            print(f"[skip] {rel}: {type(e).__name__}: {e}")
            continue

        if not same_source(cfg["data"]["val"], first_val):
            print(f"[skip] {rel}: reads a different val set than {runs[0]} -- "
                  f"dumping both into one comparison would put different "
                  f"patients in different columns")
            del model
            continue

        sigma = args.sigma if args.sigma is not None else default_sigma(cfg)
        dest = out_dir(run_dir, args.runs, args.out)
        n = dump_run(run_dir, cfg, model, n_params, step, args, volumes,
                     metrics, sigma, dest)
        total += n
        print(f"  {rel:<28} sigma={sigma:<7g} {n:>4} slices -> {dest}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not total:
        raise SystemExit("nothing was written")
    print(f"\nwrote {total} slices across {len(runs)} run(s)")


if __name__ == "__main__":
    main()
