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

    reference       (n, H, W) complex64   ground truth          } all three x organ_mask
    recon           (n, H, W) complex64   network output        } (attr images_masked)
    zero_filled     (n, H, W) complex64   E^H y -- the artifact to be removed
    guide           (n, H, W) complex64   paired data only: the side scan, fully sampled,
                                          coil-combined with the operator's maps
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
import csv
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
from evaluation.tasks import (EVAL_KSPACE_TYPE, build_adapter, default_sigma,
                              force_organ_mask, val_kspace_type)

DUMP_VERSION = 1
# tasks whose dumps are images of the anatomy (organ mask stored, images masked to it)
IMAGE_TASKS = ("recon", "i2sb_recon")


def _dc_setting(s):
    """--dc ITERS:LAM -> (iters, lam)."""
    it, _, lam = s.partition(":")
    return int(it), float(lam)


# ---------------------------------------------------------------------------
# run discovery
# ---------------------------------------------------------------------------
def find_runs(root, only=None, exact=False, ckpt="net.ckpt"):
    """Every directory under `root` holding both a config.json and the checkpoint `ckpt`.

    `only` filters by the run's path relative to `root` -- a substring match, so
    `--only R8` takes every R=8 cell and `--only mglpds_R8` takes one. With
    `exact`, an entry must BE the run directory's name: `mg81v6_R12` then leaves
    out `mg81v6_R12_500ep` and any other tagged probe of the same cell.
    """
    out = []
    for dirpath, _, filenames in os.walk(root):
        if "config.json" in filenames and ckpt in filenames:
            out.append(dirpath)
    out = sorted(out)
    if only and exact:
        out = [d for d in out if os.path.basename(os.path.normpath(d)) in set(only)]
    elif only:
        out = [d for d in out
               if any(o in os.path.relpath(d, root).replace(os.sep, "/")
                      for o in only)]
    return out


def for_checkpoint(cfg):
    """The config to BUILD a net whose weights a checkpoint will then replace:
    an initialisation that fetches pretrained weights (RAM's `init_from`, the
    DRUNet trunk) is dropped -- the checkpoint overwrites every one of those
    weights, and building it would need the download cache on the node."""
    params = cfg.get("model", {}).get("params", {})
    if params.get("init_from") is None:
        return cfg
    out = dict(cfg, model=dict(cfg["model"], params=dict(params, init_from=None)))
    return out


def load_run(run_dir, device, ckpt="net.ckpt"):
    """`ckpt`: net.ckpt (recon runs: lowest TRAINING loss) or net_bestval.ckpt (best val)."""
    from models import build_model            # imported late: pulls in CUDA bits

    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)
    from models.components import set_complex_mode
    set_complex_mode(cfg.get("training", {}).get("complex_conv"))
    # per run: a dump spans runs trained with and without the key
    from models.mg_lpds import set_planar_state
    set_planar_state(bool(cfg.get("training", {}).get("planar_state", False)))

    model = build_model(for_checkpoint(cfg)).to(device)
    if cfg.get("task") == "i2sb_recon":
        # the weights validation SAMPLED with: net.ckpt's EMA when it carries one
        from training.i2sb_recon import load_sampling_weights
        ckpt = load_sampling_weights(model, os.path.join(run_dir, ckpt), device)
    else:
        ckpt = torch.load(os.path.join(run_dir, ckpt), map_location=device,
                          weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    if getattr(model, "attn_backend", None) == "flex" and device.type == "cuda":
        model.compile_flex()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return cfg, model, n_params, ckpt.get("step")


def out_dir(run_dir, runs_root, out_root):
    """Where this run's volumes go: beside the checkpoint, or mirrored under --out.

    Always `.../eval_dump/`, so alias_scan.py and alias_detect.py can read a
    mirrored tree as they read the real one."""
    if out_root is None:
        return os.path.join(run_dir, "eval_dump")
    rel = os.path.relpath(run_dir, runs_root)
    return os.path.normpath(os.path.join(out_root, rel, "eval_dump"))


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


def read_flagged(path, top=0):
    """{volume: {source slices}} of the flagged rows of an alias_check.py slices_R<R>.csv,
    the `top` highest ranked if set."""
    with open(path, newline="") as f:
        rows = sorted((r for r in csv.DictReader(f) if int(r["flagged"])),
                      key=lambda r: int(r["rank"]))
    out = {}
    for r in rows[:top] if top else rows:
        out.setdefault(r["volume"], set()).add(int(r["slice"]))
    return out


def same_source(a, b):
    """Do two val configs read the same images? Columns must, or the figure lies."""
    keys = ("name", "anatomy", "kspace_root", "smap_root", "scale_fac",
            "split_csv", "kspace_roots")          # the last two: paired (guided) knee data
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
    cg_lam = getattr(args, "cg_sense_lam", None)      # callers without the option: no CG-SENSE
    keep_slices = getattr(args, "keep_slices", None)  # {volume: slices}: the net runs on no others
    # Optional, for scripts/scale_sweep.py: `call(net, y, E, sigma) -> image` replaces the
    # plain forward (the recon adapter's hook), and `on_item(volume, slice)` is told which
    # slice the next forward is.
    call = getattr(args, "call", None)
    on_item = getattr(args, "on_item", None)

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
        # -1 = derived from center_frac (null in the config); HDF5 attributes
        # cannot hold None. The three keys after it are what, together with R,
        # decide the sampling pattern and the problem -- a measured run and a
        # legacy run must never look alike here.
        "acs_lines": (-1 if mri.get("acs_lines") is None
                      else int(mri["acs_lines"])),
        "center_frac": float(mri.get("center_frac") or 0.0),
        "adjust_accel": bool(mri.get("adjust_accel", False)),
        # The measurement the IMAGES came from (validation k-space), which is
        # what a figure compares -- not necessarily what the net trained on.
        "kspace_type": str(val_kspace_type(cfg)),
        "online_smaps": str(mri.get("online_smaps") or ""),
        "mask_dist": mri.get("mask_dist", ""),
        "sigma": float(sigma),
        "seed": int(args.seed),
        "scale_fac": float(val.get("scale_fac", 1.0) or 1.0),
        "use_organ_mask": bool(cfg.get("training", {}).get("use_organ_mask", False)),
        "images_masked": bool(cfg["task"] in IMAGE_TASKS),
        "n_params": int(n_params),
        "step": int(step or 0),
        "metrics": list(metrics),
        # -1 = no CG-SENSE stored (HDF5 attributes cannot hold None)
        "cg_sense_lam": (-1.0 if cg_lam is None else float(cg_lam)),
        "ckpt": str(getattr(args, "ckpt", "net.ckpt")),
        # paired (guided) data: how the guide the net took was made. The stored `guide`
        # dataset is the FULLY-SAMPLED side scan either way (evaluation/tasks.py::recon).
        "guide_R": float(mri.get("guide_R") or 1),
        "guide_val_offset": float(mri.get("guide_val_offset") or 0),
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    if cfg["task"] == "i2sb_recon":
        # the sampler the images came from (evaluation/tasks.py::i2sb_recon: cfg["eval"] over
        # the run's own i2sb settings)
        i2, ev = cfg["i2sb"], cfg.get("eval", {}) or {}
        base_attrs.update(
            nfe=int(ev.get("nfe") or i2.get("val_nfe", 20)),
            dc_iters=int(ev["dc_iters"] if "dc_iters" in ev else i2.get("dc_iters", 0)),
            dc_lam=float(ev["dc_lam"] if "dc_lam" in ev else i2.get("dc_lam", 7e-5)),
            sampler="inversion" if ev.get("inversion") else "i2sb_recon",
            tau=float(i2["tau"]), x1_source=str(i2.get("x1_source", "adjoint")))

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
        if keep_slices is not None and sl not in keep_slices.get(vol, ()):
            continue
        if vol != cur:
            flush()
            cur = vol

        extras = {}
        if cg_lam is not None:
            extras.update(cg_sense_lam=cg_lam,
                          cg_sense_iters=getattr(args, "cg_sense_iters", 100),
                          cg_sense_stored_maps=bool(getattr(args, "cg_sense_stored", False)))
        if on_item is not None:
            on_item(vol, sl)
        gt, recon, metric_mask = adapter(model, batch, cfg, device, sigma, gen,
                                         extras=extras, **({"call": call} if call else {}))

        # The ANATOMY mask is taken from the batch, not from the adapter: the
        # adapter returns None for it whenever the run did not train with it,
        # but the viewer still wants it to crop the FOV and to window the
        # display. Whether it entered the metrics is a separate question, and
        # `use_organ_mask` in the attrs is what answers it.
        organ = None
        if cfg["task"] in IMAGE_TASKS and len(batch) >= 4:
            organ = (np.abs(_np(batch[3])) > 0).astype(np.uint8)
            buf.setdefault("organ_mask", []).append(organ)
        # Every stored IMAGE is masked to the anatomy (`images_masked` attr):
        # what a network leaves in the background is not part of the problem,
        # and the metrics below never see it. They are computed from the
        # unmasked tensors with the metric mask, so this changes no number.
        keep = (lambda a: a) if organ is None else (lambda a: a * organ)
        buf.setdefault("reference", []).append(keep(_np(gt)).astype(np.complex64))
        buf.setdefault("recon", []).append(keep(_np(recon)).astype(np.complex64))
        if "zero_filled" in extras:
            buf.setdefault("zero_filled", []).append(
                keep(_np(extras["zero_filled"])).astype(np.complex64))
        if "guide" in extras:
            buf.setdefault("guide", []).append(
                keep(_np(extras["guide"])).astype(np.complex64))
        if "cg_sense" in extras:
            buf.setdefault("cg_sense", []).append(
                keep(_np(extras["cg_sense"])).astype(np.complex64))
        if "cg_sense_stored" in extras:
            buf.setdefault("cg_sense_stored", []).append(
                keep(_np(extras["cg_sense_stored"])).astype(np.complex64))
        if "sampling_mask" in extras:
            buf.setdefault("sampling_mask", []).append(
                (np.abs(_np(extras["sampling_mask"])) > 0).astype(np.uint8))
        if "bridge_start" in extras:               # i2sb: x1, where the sampler started
            buf.setdefault("bridge_start", []).append(
                keep(_np(extras["bridge_start"])).astype(np.complex64))
        alt = extras.get("alt_recons") or {}       # i2sb: e.g. "nodc", the plain sampler
        for k, v in alt.items():
            buf.setdefault(f"recon_{k}", []).append(keep(_np(v)).astype(np.complex64))
        buf.setdefault("slice_index", []).append(np.int32(sl))

        for name, fn in metrics.items():
            v = fn(gt, recon, mask=metric_mask).reshape(-1)[0]
            buf.setdefault(f"{name}_slice", []).append(np.float32(v.item()))
            for k, a in alt.items():
                v = fn(gt, a, mask=metric_mask).reshape(-1)[0]
                buf.setdefault(f"{name}_{k}_slice", []).append(np.float32(v.item()))
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
    p.add_argument("--exact", action="store_true",
                   help="--only entries are run directory NAMES, matched exactly "
                        "(skips tagged probes such as <cell>_500ep)")
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
    p.add_argument("--mask", choices=["organ", "run"], default="organ",
                   help="'organ' (default): the stored per-slice metrics use the "
                        "organ mask from the stored coil maps for every run; 'run': "
                        "each run's own setting")
    p.add_argument("--val-kspace-type", choices=["simulated", "measurement_awgn"],
                   default=EVAL_KSPACE_TYPE,
                   help=f"how the measurement is made (default {EVAL_KSPACE_TYPE}: "
                        f"measured k-space + AWGN, as the nets train, whatever the "
                        f"run's config says)")
    p.add_argument("--cg-sense-lam", type=float,
                   help="also store `cg_sense`: Tikhonov CG-SENSE, (E^H E + lam I) x "
                        "= E^H y, on the same measurement the net saw")
    p.add_argument("--cg-sense-iters", type=int, default=100,
                   help="CG iterations for --cg-sense-lam")
    p.add_argument("--cg-sense-stored", action="store_true",
                   help="with --cg-sense-lam, also store `cg_sense_stored`: the same solve "
                        "through the STORED maps (the reference's) instead of the online "
                        "estimate, to separate map mismatch from aliasing")
    p.add_argument("--ckpt", default="net.ckpt",
                   help="checkpoint file in each run dir: net.ckpt (recon runs: lowest "
                        "training loss) or net_bestval.ckpt (best validation)")
    # i2sb_recon runs only (evaluation/tasks.py::i2sb_recon); default: each run's own settings
    p.add_argument("--nfe", type=int,
                   help="i2sb runs: sampler NFE (default: the run's val_nfe)")
    p.add_argument("--dc", type=_dc_setting, metavar="ITERS:LAM",
                   help="i2sb runs: CG data consistency on every x0 prediction (I2SB-Recon), "
                        "e.g. 7:7e-5; 0:0 = the plain sampler (default: the run's dc_iters / "
                        "dc_lam). With CG on, the plain sampler on the same noise draws is "
                        "stored too, as `recon_nodc` (+ <metric>_nodc_slice)")
    p.add_argument("--no-nodc", action="store_true",
                   help="i2sb runs: do not also store the plain sampler (halves the time)")
    p.add_argument("--inversion", action="store_true",
                   help="i2sb runs: I2SB-Inversion's recon -> inversion -> recon instead")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, help="stop after N slices (smoke test)")
    p.add_argument("--flags", help="an alias_check.py slices_R<R>.csv: run the net only on its "
                                   "flagged slices, in place of --volumes and --slices")
    p.add_argument("--flags-top", type=int, default=0, help="--flags: only the top N flagged slices")
    args = p.parse_args()

    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    metrics = build_metrics(args.metrics)
    warm_up(args.metrics, device)             # LPIPS downloads now, not mid-sweep

    runs = find_runs(args.runs, args.only, exact=args.exact, ckpt=args.ckpt)
    if args.exact and args.only:
        missing = sorted(set(args.only) - {os.path.basename(os.path.normpath(r))
                                           for r in runs})
        if missing:
            print(f"[dump] no trained run named: {' '.join(missing)}  (skipped)")
    if not runs:
        raise SystemExit(
            f"no runs with both config.json and {args.ckpt} under {args.runs}"
            + (f" matching {args.only}" if args.only else ""))

    # One volume list for the whole sweep, resolved from the first run's val
    # config and checked against the rest.
    with open(os.path.join(runs[0], "config.json")) as f:
        first_val = json.load(f)["data"]["val"]
    args.keep_slices = None
    if args.flags:
        args.keep_slices = read_flagged(args.flags, args.flags_top)
        if not args.keep_slices:
            raise SystemExit(f"nothing flagged in {args.flags}")
        args.volumes = sorted(args.keep_slices)
        _sl = [s for v in args.keep_slices.values() for s in v]
        args.slices = f"{min(_sl)}:{max(_sl) + 1}"
    volumes = resolve_volumes(first_val, args.volumes, args.n_volumes)
    print(f"{len(runs)} run(s) | {len(volumes)} volume(s) | slices "
          f"{args.slices or 'all'} | device {device}")
    print(f"volumes: {', '.join(volumes)}")

    total = 0
    for run_dir in runs:
        rel = os.path.relpath(run_dir, args.runs)
        try:
            cfg, model, n_params, step = load_run(run_dir, device, args.ckpt)
        except Exception as e:                # a bad cell must not sink the sweep
            print(f"[skip] {rel}: {type(e).__name__}: {e}")
            continue

        if not same_source(cfg["data"]["val"], first_val):
            print(f"[skip] {rel}: reads a different val set than {runs[0]} -- "
                  f"dumping both into one comparison would put different "
                  f"patients in different columns")
            del model
            continue

        if args.mask == "organ":
            force_organ_mask(cfg)        # metrics inside the stored-map organ mask
        if args.val_kspace_type:
            cfg.setdefault("training", {})["val_kspace_type"] = args.val_kspace_type
        if cfg["task"] == "i2sb_recon":
            ev = cfg.setdefault("eval", {})
            if args.nfe:
                ev["nfe"] = args.nfe
            if args.dc is not None:
                ev["dc_iters"], ev["dc_lam"] = args.dc
            ev["also_nodc"] = not args.no_nodc
            ev["inversion"] = bool(args.inversion)
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
