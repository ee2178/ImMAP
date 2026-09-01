#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build one HDF5 per (patient, session) from NYUMets, for sessions carrying the full
FLAIR / T1 / CT1 / T2 quartet. Sessions missing any of the four are skipped.

Layout mirrors preprocessing/cmap.py, so the existing BraTS loaders read these unchanged:

    <out>/<patient>/<patient>_<session>_img.h5
        img_raw          (n, H, W, C) float32   unnormalized intensities
        img_median_mad   (n, H, W, C) float32   per-contrast median/MAD, within brain
        img              -> soft link to img_median_mad (an alias, zero extra bytes)
        mask             (n, H, W, 1) uint8     brain mask
        slice_index      (n,)         int32     maps n back to the original RAS z

Channel order is [FLAIR, T1, T1ce, T2] -- the BraTS order, so `contrast_idx` and every
existing config keep their meaning. NYUMets writes the enhanced T1 as CT1; the stored
LABEL stays T1ce.

Normalization is median/MAD (robust centre AND scale), computed per contrast over the
IN-BRAIN voxels of the whole volume via cmap.normalize_masked -- same statistics and
same background-zero convention as the BraTS h5s. Following cmap, the stats come from
the percentile-CLIPPED volume while `img_raw` stores the UNCLIPPED data, so inverting
img_median_mad -> raw is exact only away from the clipped tails.

    python -m preprocessing.nyumets_h5 --root ../datasets/NYUMets/data/imaging/patientId \
                                       --out ~/scratch/datasets/NYUMets_h5 --crop 224
"""

import os
import re
import csv
import glob
import argparse

import numpy as np
import torch
import h5py
import nibabel as nib

from preprocessing.cmap import (
    normalize_masked, channelwise_percentile_clip, center_crop_spatial, norm_key,
)

CONTRASTS = ("FLAIR", "T1", "T1ce", "T2")          # stored channel order (BraTS order)
MODE = "median-mad"

# First match wins. FLAIR before T2 ("T2_FLAIR" is FLAIR); CT1 before T1 (t1 is a substring
# of ct1). The CT1 guard is a lookbehind, not \bct1\b -- '_' is a word character, so \b never
# fires between the '_' and the 'c' in 'NYU0001_CT1.nii.gz'.
RULES = (
    ("FLAIR", r"flair"),
    ("T1ce",  r"(?<![a-z0-9])c[\W_]?t1(?!\d)"),
    ("T1",    r"(?<![a-z0-9])t1(?!\d)"),
    ("T2",    r"(?<![a-z0-9])t2(?!\d)"),
)
RULES = tuple((lab, re.compile(p, re.I)) for lab, p in RULES)
SKIP_RE = re.compile(r"(seg|label|lesion|mask|roi|contour)", re.I)
DATE_RE = re.compile(r"((?:19|20)\d{2})[-_.]?(\d{2})[-_.]?(\d{2})")


def find_sessions(root):
    """-> {(patient, session): {contrast: path}}, keeping only complete quartets."""
    out = {}
    for pid in sorted(e.name for e in os.scandir(root) if e.is_dir()):
        for path in glob.glob(os.path.join(root, pid, "**", "*.nii*"), recursive=True):
            rel = os.path.relpath(path, os.path.join(root, pid)).replace(os.sep, "/")
            if SKIP_RE.search(rel):
                continue
            lab = next((l for l, rx in RULES if rx.search(rel)), None)
            if lab is None:
                continue
            m = DATE_RE.search(rel)
            ses = "-".join(m.groups()) if m else rel.split("/")[0] or "single"
            out.setdefault((pid, ses), {}).setdefault(lab, path)
    return {k: v for k, v in sorted(out.items()) if all(c in v for c in CONTRASTS)}


def load_ras(path):
    """(H, W, D) float32 in canonical RAS, plus the affine. No resampling."""
    img = nib.as_closest_canonical(nib.load(path))
    v = np.asanyarray(img.dataobj)
    while v.ndim > 3:
        v = v[..., 0]
    return np.asarray(v, dtype=np.float32), img.affine


def brain_mask(img, bg_frac):
    """(D,C,H,W) -> (D,H,W) bool.

    NYUMets is not skull-stripped, so there is no `> 0` rule to lean on. Scale each
    contrast by its own p99.5, AVERAGE across contrasts, then threshold. Averaging rather
    than intersecting keeps CSF, which is dark on T1 but bright on T2/FLAIR and would be
    cut by an all-contrasts-must-agree rule. This is a proxy -- swap in HD-BET or
    SynthStrip masks later if the tissue statistics need to be exact.
    """
    acc = torch.zeros((img.shape[0],) + tuple(img.shape[2:]), dtype=torch.float32)
    for c in range(img.shape[1]):
        v = img[:, c].float()
        flat = v.reshape(-1)
        step = max(1, flat.numel() // 2_000_000)          # torch.quantile caps around 16M
        hi = torch.quantile(flat[::step], 0.995).clamp_min(1e-6)
        acc += v / hi
    return (acc / img.shape[1]) > bg_frac


def build(files, cfg):
    """-> (raw_hwc, norm_hwc, mask_hwc, slice_idx, stats, orig_depth, affine)."""
    vols, affines = zip(*(load_ras(files[c]) for c in CONTRASTS))
    if len({v.shape for v in vols}) != 1:
        raise ValueError(f"shapes differ: {[v.shape for v in vols]} -- not on a common grid")
    if cfg.require_affine and not all(np.allclose(a, affines[0], atol=1e-3) for a in affines):
        raise ValueError("affines differ -- contrasts are not co-registered")

    img = torch.from_numpy(np.stack([v.transpose(2, 0, 1) for v in vols], axis=1))  # (D,C,H,W)
    fg = brain_mask(img, cfg.bg_frac)
    clipped = (channelwise_percentile_clip(img, fg, cfg.clip[0], cfg.clip[1])
               if cfg.clip else img)
    norm, stats = normalize_masked(clipped, fg, mode=MODE)

    raw, orig_depth = img, img.shape[0]
    if cfg.crop:
        raw = center_crop_spatial(raw, cfg.crop, h_axis=2, w_axis=3)
        norm = center_crop_spatial(norm, cfg.crop, h_axis=2, w_axis=3)
        fg = center_crop_spatial(fg, cfg.crop, h_axis=1, w_axis=2)

    frac = fg.reshape(fg.shape[0], -1).float().mean(dim=1).numpy()
    keep = frac >= cfg.min_brain_frac
    if cfg.start is not None or cfg.end is not None:
        z = np.arange(orig_depth)
        keep &= (z >= (cfg.start or 0)) & (z < (cfg.end if cfg.end is not None else orig_depth))
    idx = np.where(keep)[0]
    if idx.size == 0:
        raise ValueError("no slice passed min_brain_frac")

    to_hwc = lambda t: np.transpose(t.numpy()[idx], (0, 2, 3, 1))
    return (to_hwc(raw), to_hwc(norm), fg.numpy()[idx][..., None],
            idx, stats, orig_depth, affines[0])


def write_h5(path, raw, norm, mask, idx, stats, orig_depth, affine, key, files, cfg):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    comp = "gzip" if cfg.compress else None
    chunk = lambda a: (1,) + a.shape[1:]
    with h5py.File(path, "w") as f:
        f.create_dataset("img_raw", data=raw.astype(np.float32),
                         chunks=chunk(raw), compression=comp)
        f.create_dataset(norm_key(MODE), data=norm.astype(np.float32),
                         chunks=chunk(norm), compression=comp)
        # `img` is the name every existing loader reaches for; a soft link keeps that working
        # without a second copy of the pixels on disk.
        f["img"] = h5py.SoftLink("/" + norm_key(MODE))
        f.create_dataset("mask", data=mask.astype(np.uint8),
                         chunks=chunk(mask), compression=comp)
        f.create_dataset("slice_index", data=idx.astype(np.int32))
        f.attrs["subject_id"] = f"{key[0]}_{key[1]}"
        f.attrs["patient"], f.attrs["session"] = key
        f.attrs["contrasts"] = ",".join(CONTRASTS)
        f.attrs["normalize"] = MODE
        f.attrs[f"norm_stats_{MODE.replace('-', '_')}"] = stats      # (C, 2) centre/scale
        f.attrs["norm_stats"] = stats
        f.attrs["clip_percentiles"] = np.asarray(cfg.clip or (-1, -1), dtype=np.float32)
        f.attrs["crop_size"] = int(cfg.crop) if cfg.crop else -1
        f.attrs["orig_depth"] = int(orig_depth)
        f.attrs["min_brain_frac"] = float(cfg.min_brain_frac)
        f.attrs["mask_rule"] = f"mean_c(v/p99.5_c) > {cfg.bg_frac}"
        f.attrs["affine"] = np.asarray(affine, dtype=np.float32)
        f.attrs["source_files"] = ",".join(os.path.basename(files[c]) for c in CONTRASTS)
        f.attrs["axis_order"] = "N,H,W,C ; slice_index maps N back to the canonical-RAS z"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="patient-ID level, e.g. .../imaging/patientId")
    ap.add_argument("--out", required=True)
    ap.add_argument("--crop", type=int, default=0, help="center crop in-plane; 0 = none")
    ap.add_argument("--clip", type=float, nargs=2, default=(0.5, 99.5),
                    help="foreground percentile clip before the stats; '--clip 0 0' disables")
    ap.add_argument("--bg-frac", type=float, default=0.05, dest="bg_frac")
    ap.add_argument("--min-brain-frac", type=float, default=0.02, dest="min_brain_frac")
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--require-affine", action="store_true", dest="require_affine",
                    help="skip sessions whose contrasts are not on one affine")
    ap.add_argument("--compress", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    cfg = ap.parse_args()
    if cfg.clip and float(cfg.clip[1]) <= 0:
        cfg.clip = None

    sessions = find_sessions(cfg.root)
    if cfg.limit:
        sessions = dict(list(sessions.items())[:cfg.limit])
    print(f"{len(sessions)} complete {'/'.join(CONTRASTS)} sessions under {cfg.root}")

    rows, shapes, skipped = [], {}, []
    for i, (key, files) in enumerate(sessions.items(), 1):
        path = os.path.join(cfg.out, key[0], f"{key[0]}_{key[1]}_img.h5")
        if os.path.exists(path) and not cfg.overwrite:
            continue
        try:
            raw, norm, mask, idx, stats, depth, aff = build(files, cfg)
        except Exception as err:
            skipped.append((key, str(err)))
            print(f"  !! {key[0]}/{key[1]}: {err}")
            continue
        write_h5(path, raw, norm, mask, idx, stats, depth, aff, key, files, cfg)
        shapes[raw.shape[1:3]] = shapes.get(raw.shape[1:3], 0) + 1
        rows.append({"patient": key[0], "session": key[1], "path": path,
                     "n_slices": len(idx), "orig_depth": depth,
                     "H": raw.shape[1], "W": raw.shape[2]})
        if i % 25 == 0:
            print(f"  {i}/{len(sessions)}  {len(rows)} written, {len(skipped)} skipped")

    os.makedirs(cfg.out, exist_ok=True)
    with open(os.path.join(cfg.out, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["patient", "session", "path", "n_slices",
                                          "orig_depth", "H", "W"])
        w.writeheader()
        w.writerows(rows)

    print(f"\n{len(rows)} written, {len(skipped)} skipped -> {cfg.out}")
    print("in-plane sizes:", dict(shapes))
    if not cfg.crop and len(shapes) > 1:
        print("!! ragged in-plane sizes -- pass --crop to get a single shape before training")


if __name__ == "__main__":
    main()
