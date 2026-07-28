#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline constraint-map generation for MR-contrast guided contrastive learning (PyTorch).

Per subject:
    load multi-contrast  ->  stack (slices, contrast, H, W)
    ->  channel-wise percentile clip  ->  WITHIN-BRAIN normalization (z-score or minmax)
    ->  pixel-wise PCA  ->  TV denoise (FISTA)  ->  MiniBatchKMeans over the WHOLE volume
    ->  integer constraint map aligned to the image slices.

Output: ONE HDF5 per subject, written into that subject's own folder (in place by default),
named to match the BraTS file convention:
    <case>/<case>_constraint_map_K<K>.h5   ['param']       (n, H, W, 1)  int16
                                           ['slice_index']  (n,)         int32  -> original z
    <case>/<case>_img.h5  (optional)       ['img']          (n, H, W, C) float  normalized
                                           ['img_raw']      (n, H, W, C) float  unnormalized (raw)
                                           ['mask']         (n, H, W, 1) uint8  brain mask
                                           ['seg']          (n, H, W, 1) int16  BraTS tumor labels
                                           ['et']           (n, H, W, 1) uint8  enhancing-tumor mask
                                           ['slice_index']  (n,)         int32
Plus a manifest CSV at the subdir root listing every subject and its kept-slice count.

`seg` is the raw BraTS segmentation (1 = necrotic core, 2 = edema, 4 = enhancing tumor;
3 unused), carried through the exact same crop + slice selection as the images so it stays
voxel-aligned. `et` is the derived binary enhancing-tumor mask, `isin(seg, cfg.et_labels)`
-- stored explicitly because it is what the training losses consume, with `seg` kept
alongside so tumor-core / whole-tumor masks can be derived later without regenerating.
Both are written only when `save_seg` is on AND the subject has a *_seg.nii.gz.

Configuration is read from a YAML file (default ./cmap_config.yaml), not CLI flags.

Depends on the user's repo modules:
    preprocessing.pca.pca_pixelwise
    solvers.tvd.tvd_fista
    preprocessing.kmeans.minibatch_kmeans
"""

import os
import sys
import csv
import glob
import argparse
from types import SimpleNamespace

import yaml
import numpy as np
import nibabel as nib
import torch
import h5py


# ----------------------------------------------------------------------------
# IO + preprocessing helpers
# ----------------------------------------------------------------------------
def load_case(case_dir, contrasts):
    """Load one case as a dict {contrast: np.ndarray (H, W, D)}. No preprocessing."""
    def _load(tag):
        matches = glob.glob(os.path.join(case_dir, f"*_{tag}.nii.gz"))
        if not matches:
            raise FileNotFoundError(f"No '*_{tag}.nii.gz' in {case_dir}")
        return np.asarray(nib.load(matches[0]).get_fdata(), dtype=np.float32)
    return {c: _load(c) for c in contrasts}


def load_seg(case_dir, tag="seg"):
    """Load one case's BraTS segmentation as (H, W, D) int16, or None if absent.

    Returned as integer labels, NOT a binary mask: nearest-integer rounding because
    get_fdata() hands back float64 and label 4 must survive as exactly 4. Missing seg is
    a normal condition (inference-only cohorts ship without it), so this returns None
    rather than raising -- the caller decides.
    """
    matches = glob.glob(os.path.join(case_dir, f"*_{tag}.nii.gz"))
    if not matches:
        return None
    seg = np.asarray(nib.load(matches[0]).get_fdata())
    return np.rint(seg).astype(np.int16)


def normalize_masked(img, fg, mode="zscore", eps=1e-8):
    """Per-channel normalization computed ONLY within the brain mask; background -> 0.
    img (D,C,H,W), fg (D,H,W) bool. Returns (out, stats) where stats[c] = (mean,std) for
    zscore or (lo,range) for minmax, so the transform is invertible later.

    Note: a linear "contrast stretch" to [0,1] before a z-score is a no-op (z-score is
    affine-invariant), so we offer the stretch as an ALTERNATIVE normalization (minmax),
    not an extra step. The nonlinear clipping that actually matters is done separately."""
    out = img.clone()
    C = img.shape[1]
    stats = np.zeros((C, 2), dtype=np.float32)
    for c in range(C):
        vals = img[:, c][fg]
        if vals.numel() == 0:
            out[:, c] = 0.0
            continue
        if mode == "zscore":
            mean = vals.mean()
            std = vals.std().clamp_min(eps)
            out[:, c] = (img[:, c] - mean) / std
            stats[c] = (float(mean), float(std))
        elif mode == "minmax":
            lo = vals.amin()
            rng = (vals.amax() - lo).clamp_min(eps)
            out[:, c] = ((img[:, c] - lo) / rng).clamp(0.0, 1.0)
            stats[c] = (float(lo), float(rng))
        else:
            raise ValueError(f"normalize must be 'zscore' or 'minmax', got {mode}")
    out = out * fg.unsqueeze(1)            # background -> 0 sentinel (consistent across channels)
    return out, stats


def channelwise_percentile_clip(img, fg_mask, p_lo, p_hi, max_samples=2_000_000):
    """Clip each contrast to its [p_lo, p_hi] foreground percentiles. img (D,C,H,W)."""
    out = img.clone()
    for c in range(img.shape[1]):
        vals = img[:, c][fg_mask]
        if vals.numel() == 0:
            continue
        if vals.numel() > max_samples:
            sel = torch.randint(0, vals.numel(), (max_samples,), device=vals.device)
            vals = vals[sel]
        lo = torch.quantile(vals.float(), p_lo / 100.0)
        hi = torch.quantile(vals.float(), p_hi / 100.0)
        out[:, c] = out[:, c].clamp(min=lo.item(), max=hi.item())
    return out


def center_crop_spatial(a, size, h_axis, w_axis):
    """Center-crop a numpy array to size x size along the given spatial axes."""
    H, W = a.shape[h_axis], a.shape[w_axis]
    h0, w0 = (H - size) // 2, (W - size) // 2
    sl = [slice(None)] * a.ndim
    sl[h_axis] = slice(h0, h0 + size)
    sl[w_axis] = slice(w0, w0 + size)
    return a[tuple(sl)]


# ----------------------------------------------------------------------------
# Per-subject constraint map
# ----------------------------------------------------------------------------
@torch.no_grad()
def constraint_map_for_subject(case_dir, cfg, pca_fn, tvd_fn, kmeans_fn):
    """
    Returns (param, image, image_raw, fg, seg, stats):
        param     : (D, H, W) int16      constraint map (background = 0 if mask_background)
        image     : (D, C, H, W) float32 NORMALIZED image (within-brain zscore/minmax, bg=0)
        image_raw : (D, C, H, W) float32 UNNORMALIZED, UNCLIPPED raw intensities (bg=0)
        fg        : (D, H, W) bool        brain mask (from raw intensities)
        seg       : (D, H, W) int16       BraTS tumor labels, or None if the case has no seg
        stats     : (C, 2) float32        per-channel normalization stats (invertible)

    Clustering (PCA/TVD/kmeans) runs on the NORMALIZED image so the constraint maps are
    unchanged; the unnormalized image is carried alongside for tasks (e.g. I2SB translation)
    that need the true inter-contrast intensity relationship preserved.
    """
    device = cfg.device

    data = load_case(case_dir, cfg.contrasts)
    x = np.stack([data[c] for c in cfg.contrasts], axis=0)      # (C, H, W, D)
    x = torch.from_numpy(x).float().to(device)
    x = x.permute(3, 0, 1, 2).contiguous()                      # (D, C, H, W)

    fg = (x.abs().sum(dim=1) > 0)                               # (D, H, W) bool, from RAW

    # tumor labels, permuted to match the (D, H, W) image layout. Never normalized,
    # clipped or masked -- these are labels, and any interpolation would invent classes.
    seg = None
    if getattr(cfg, "save_seg", True):
        seg = load_seg(case_dir)
        if seg is not None:
            seg = np.transpose(seg, (2, 0, 1))                  # (H, W, D) -> (D, H, W)

    x_raw = (x * fg.unsqueeze(1)).contiguous()                  # UNnormalized, UNclipped, bg -> 0
    x = channelwise_percentile_clip(x, fg, cfg.clip_lo, cfg.clip_hi)
    x, stats = normalize_masked(x, fg, mode=getattr(cfg, "normalize", "zscore"))

    pca_out = pca_fn(x, n_components=cfg.n_pca)                  # cluster on the NORMALIZED image
    x_pc = pca_out[0] if isinstance(pca_out, (tuple, list)) else pca_out

    tvd_out = tvd_fn(x_pc.unsqueeze(1), lam=cfg.tvd_lam, eta=cfg.tvd_eta,
                     maxit=cfg.tvd_maxit, tol=cfg.tvd_tol, verbose=False, isotropic=True)
    x_den = tvd_out[0] if isinstance(tvd_out, (tuple, list)) else tvd_out
    x_den = x_den.squeeze(1)
    if torch.is_complex(x_den):
        x_den = x_den.real
    x_den = x_den.contiguous()                                  # (D, n_pca, H, W)

    D, P, H, W = x_den.shape
    feats = x_den.permute(0, 2, 3, 1).reshape(-1, P)

    param = torch.zeros(D * H * W, dtype=torch.int64, device=device)
    if cfg.mask_background:
        fg_flat = fg.reshape(-1)
        labels, _ = kmeans_fn(feats[fg_flat], n_clusters=cfg.n_clusters)
        param[fg_flat] = labels.to(torch.int64) + 1             # 0 reserved for background
    else:
        labels, _ = kmeans_fn(feats, n_clusters=cfg.n_clusters)
        param = labels.to(torch.int64)

    param = param.reshape(D, H, W).to(torch.int16).cpu().numpy()
    image = x.cpu().numpy().astype(np.float32)
    image_raw = x_raw.cpu().numpy().astype(np.float32)
    fg = fg.cpu().numpy()
    return param, image, image_raw, fg, seg, stats


# ----------------------------------------------------------------------------
# Per-subject HDF5 writing
# ----------------------------------------------------------------------------
def et_labels(cfg):
    """Segmentation labels counted as ENHANCING TUMOR. BraTS 2021 uses 4 (1 = necrotic
    core, 2 = edema, 3 unused); later BraTS releases relabel ET to 3, hence the config
    knob rather than a hard-coded constant."""
    return [int(v) for v in getattr(cfg, "et_labels", [4])]


def _common_attrs(f, case, slice_idx, orig_depth, stats, cfg):
    f.create_dataset("slice_index", data=slice_idx.astype(np.int32))
    f.attrs["subject_id"] = case
    f.attrs["n_clusters"] = int(cfg.n_clusters)
    f.attrs["contrasts"] = ",".join(cfg.contrasts)
    f.attrs["mask_background"] = bool(cfg.mask_background)
    f.attrs["crop_size"] = int(cfg.crop_size) if cfg.crop_size else -1
    f.attrs["orig_depth"] = int(orig_depth)
    f.attrs["slice_range"] = [int(getattr(cfg, "start_slice", -1) or -1),
                              int(getattr(cfg, "end_slice", -1) or -1)]
    f.attrs["normalize"] = getattr(cfg, "normalize", "zscore")
    f.attrs["norm_stats"] = stats                 # (C,2): (mean,std) or (lo,range) per channel
    f.attrs["has_raw_image"] = bool(getattr(cfg, "save_raw_image", True))
    f.attrs["et_labels"] = np.asarray(et_labels(cfg), dtype=np.int16)
    f.attrs["axis_order"] = "N,H,W,C ; slice_index maps N back to the original nifti z-axis"


def save_subject(out_dir, case, image_hwc, image_raw_hwc, param_hwc, mask_hwc, seg_hwc,
                 slice_idx, orig_depth, stats, cfg):
    os.makedirs(out_dir, exist_ok=True)
    comp = "gzip" if cfg.compress else None

    cmap_path = os.path.join(out_dir, f"{case}_constraint_map_K{cfg.n_clusters}.h5")
    with h5py.File(cmap_path, "w") as f:
        f.create_dataset("param", data=param_hwc.astype(np.int16),
                         chunks=(1,) + param_hwc.shape[1:], compression=comp)
        _common_attrs(f, case, slice_idx, orig_depth, stats, cfg)

    img_path = ""
    if cfg.save_image:
        img_path = os.path.join(out_dir, f"{case}_img.h5")
        with h5py.File(img_path, "w") as f:
            f.create_dataset("img", data=image_hwc.astype(cfg.img_dtype),         # NORMALIZED
                             chunks=(1,) + image_hwc.shape[1:], compression=comp)
            if getattr(cfg, "save_raw_image", True) and image_raw_hwc is not None:
                f.create_dataset("img_raw", data=image_raw_hwc.astype(cfg.img_dtype),  # UNNORMALIZED
                                 chunks=(1,) + image_raw_hwc.shape[1:], compression=comp)
            f.create_dataset("mask", data=mask_hwc.astype(np.uint8),     # brain mask (n,H,W,1)
                             chunks=(1,) + mask_hwc.shape[1:], compression=comp)
            if seg_hwc is not None:
                f.create_dataset("seg", data=seg_hwc.astype(np.int16),   # BraTS labels (n,H,W,1)
                                 chunks=(1,) + seg_hwc.shape[1:], compression=comp)
                et = np.isin(seg_hwc, np.asarray(et_labels(cfg)))        # enhancing tumor
                f.create_dataset("et", data=et.astype(np.uint8),
                                 chunks=(1,) + et.shape[1:], compression=comp)
            _common_attrs(f, case, slice_idx, orig_depth, stats, cfg)
    return cmap_path, img_path


def prepare_arrays(param, image, image_raw, fg, seg, cfg):
    """Crop + brain-fraction slice selection; return (img_hwc, img_raw_hwc, param_hwc,
    mask_hwc, seg_hwc, slice_idx, orig_depth). A slice is kept iff brain covers >=
    min_brain_frac of the FOV (subsumes the old 'any foreground' rule and drops near-empty
    top/bottom slices). `seg` rides through the identical crop + `keep` selection as the
    images -- that alignment is the whole point of doing it here rather than at load time.
    seg_hwc is None when the subject had no segmentation."""
    orig_depth = param.shape[0]
    if cfg.crop_size:
        image = center_crop_spatial(image, cfg.crop_size, h_axis=2, w_axis=3)
        image_raw = center_crop_spatial(image_raw, cfg.crop_size, h_axis=2, w_axis=3)
        param = center_crop_spatial(param, cfg.crop_size, h_axis=1, w_axis=2)
        fg = center_crop_spatial(fg, cfg.crop_size, h_axis=1, w_axis=2)
        if seg is not None:
            seg = center_crop_spatial(seg, cfg.crop_size, h_axis=1, w_axis=2)

    D = fg.shape[0]
    z = np.arange(D)
    # positional slice band on the ORIGINAL z-axis: [start_slice, end_slice)
    in_range = np.ones(D, dtype=bool)
    start = getattr(cfg, "start_slice", None)
    end = getattr(cfg, "end_slice", None)
    if start is not None:
        in_range &= (z >= int(start))
    if end is not None:
        in_range &= (z < int(end))
    # fine trim: brain fraction within the band
    if cfg.drop_empty_slices:
        frac = fg.reshape(D, -1).mean(axis=1)                   # brain fraction per slice
        keep = in_range & (frac >= float(getattr(cfg, "min_brain_frac", 0.0)))
    else:
        keep = in_range
    slice_idx = np.where(keep)[0]

    img_hwc = np.transpose(image[keep], (0, 2, 3, 1))           # (n, H, W, C) normalized
    img_raw_hwc = np.transpose(image_raw[keep], (0, 2, 3, 1))   # (n, H, W, C) unnormalized
    param_hwc = param[keep][..., np.newaxis]                    # (n, H, W, 1)
    mask_hwc = fg[keep][..., np.newaxis]                        # (n, H, W, 1) bool
    seg_hwc = None if seg is None else seg[keep][..., np.newaxis]   # (n, H, W, 1) int16
    return img_hwc, img_raw_hwc, param_hwc, mask_hwc, seg_hwc, slice_idx, orig_depth


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def load_config(path):
    with open(path, "r") as f:
        return SimpleNamespace(**yaml.safe_load(f))


def out_dir_for(case, src_dir, cfg):
    if cfg.output_root:
        return os.path.join(cfg.output_root, case)
    return src_dir                                              # in place


def process_split(cases, subdir_path, cfg, fns, manifest_rows):
    pca_fn, tvd_fn, kmeans_fn = fns
    n_ok = n_fail = n_skip = n_noseg = total_slices = 0
    for i, case in enumerate(cases):
        src_dir = os.path.join(subdir_path, case)
        out_dir = out_dir_for(case, src_dir, cfg)
        cmap_path = os.path.join(out_dir, f"{case}_constraint_map_K{cfg.n_clusters}.h5")

        if cfg.skip_existing and os.path.exists(cmap_path):
            with h5py.File(cmap_path, "r") as f:
                n = f["param"].shape[0]
            manifest_rows.append([case, n, cmap_path,
                                  os.path.join(out_dir, f"{case}_img.h5") if cfg.save_image else ""])
            n_skip += 1
            print(f"[{i + 1}/{len(cases)}] {case}: skip (exists, {n} slices)")
            continue

        try:
            param, image, image_raw, fg, seg, stats = constraint_map_for_subject(src_dir, cfg, pca_fn, tvd_fn, kmeans_fn)
            img_hwc, img_raw_hwc, param_hwc, mask_hwc, seg_hwc, slice_idx, orig_depth = prepare_arrays(
                param, image, image_raw, fg, seg, cfg)
            cpath, ipath = save_subject(out_dir, case, img_hwc, img_raw_hwc, param_hwc, mask_hwc,
                                        seg_hwc, slice_idx, orig_depth, stats, cfg)
            manifest_rows.append([case, param_hwc.shape[0], cpath, ipath])
            n_ok += 1
            total_slices += param_hwc.shape[0]
            if getattr(cfg, "save_seg", True) and seg_hwc is None:
                n_noseg += 1
                # counted and reported at the end: a run where every case lands here
                # produces h5s with no 'et', and ET-weighted training will refuse to start
                print(f"[{i + 1}/{len(cases)}] {case}: no *_seg.nii.gz -- no 'seg'/'et' written")
            et_frac = "" if seg_hwc is None else \
                f" et={float(np.isin(seg_hwc, et_labels(cfg)).mean()):.4f}"
            print(f"[{i + 1}/{len(cases)}] {case}: {param_hwc.shape[0]}/{orig_depth} slices "
                  f"-> {os.path.basename(cpath)}{et_frac}")
        except Exception as e:
            n_fail += 1
            print(f"[{i + 1}/{len(cases)}] {case}: FAILED ({type(e).__name__}: {e})")
    print(f"  -> split done: ok={n_ok}, skipped={n_skip}, failed={n_fail}, "
          f"new slices={total_slices}")
    if n_noseg:
        print(f"  -> WARNING: {n_noseg}/{n_ok} subjects had no *_seg.nii.gz, so their h5 has "
              f"no 'et' dataset. ET-weighted training will raise on those subjects.")


def add_seg_to_existing(cases, subdir_path, cfg):
    """Backfill 'seg' + 'et' into ALREADY-GENERATED <case>_img.h5 files, in place.

    Motivation: the segmentation is the only thing missing from an existing dataset, and a
    full regeneration would redo PCA + TV denoising + k-means per subject to reproduce
    constraint maps that would come out identical. This path touches nothing else.

    Alignment is reconstructed from what the h5 already records -- the `crop_size` attr and
    the `slice_index` dataset (original nifti z per stored slice) -- so the labels land on
    exactly the slices the images came from. The stored (H, W) is verified against the
    cropped seg before anything is written; a mismatch means the h5 was made with different
    settings and the case is skipped rather than silently misaligned.
    """
    reasons = {"ok": 0, "no_h5": 0, "already_had_et": 0, "no_seg_nifti": 0, "failed": 0}
    first_err = None
    # print the RESOLVED path once: the split dirs are symlink farms, so this is what tells
    # you whether you are about to write the same files the training config reads.
    print(f"  backfill target: {subdir_path}\n"
          f"  resolves to    : {os.path.realpath(subdir_path)}\n"
          f"  {len(cases)} case(s), et_labels={et_labels(cfg)}")
    for i, case in enumerate(cases):
        src_dir = os.path.join(subdir_path, case)
        img_path = os.path.join(out_dir_for(case, src_dir, cfg), f"{case}_img.h5")
        if not os.path.exists(img_path):
            print(f"[{i + 1}/{len(cases)}] {case}: no {os.path.basename(img_path)} -- skip")
            reasons["no_h5"] += 1
            continue
        try:
            with h5py.File(img_path, "r+") as f:
                if "et" in f and not cfg.overwrite_seg:
                    print(f"[{i + 1}/{len(cases)}] {case}: 'et' exists -- skip")
                    reasons["already_had_et"] += 1
                    continue
                seg = load_seg(src_dir)
                if seg is None:
                    print(f"[{i + 1}/{len(cases)}] {case}: no *_seg.nii.gz in {src_dir} -- skip")
                    reasons["no_seg_nifti"] += 1
                    continue
                seg = np.transpose(seg, (2, 0, 1))                  # (H,W,D) -> (D,H,W)

                crop = int(f.attrs.get("crop_size", -1))
                if crop and crop > 0:
                    seg = center_crop_spatial(seg, crop, h_axis=1, w_axis=2)
                slice_idx = np.asarray(f["slice_index"]).astype(np.int64)
                seg = seg[slice_idx][..., np.newaxis]               # (n, H, W, 1)

                want = tuple(f["mask"].shape)
                if seg.shape != want:
                    raise ValueError(f"seg {seg.shape} != stored mask {want}; the h5 was "
                                     f"written with different crop/slice settings")

                labels = et_labels(cfg)
                et = np.isin(seg, np.asarray(labels)).astype(np.uint8)
                for key, data in (("seg", seg.astype(np.int16)), ("et", et)):
                    if key in f:
                        del f[key]
                    f.create_dataset(key, data=data, chunks=(1,) + data.shape[1:],
                                     compression="gzip" if cfg.compress else None)
                f.attrs["et_labels"] = np.asarray(labels, dtype=np.int16)
            reasons["ok"] += 1
            print(f"[{i + 1}/{len(cases)}] {case}: +seg{seg.shape} +et "
                  f"(et fraction {float(et.mean()):.4f})")
        except Exception as e:
            reasons["failed"] += 1
            if first_err is None:
                first_err = e
            print(f"[{i + 1}/{len(cases)}] {case}: FAILED ({type(e).__name__}: {e})")

    print("  -> backfill: " + ", ".join(f"{k}={v}" for k, v in reasons.items()))
    if reasons["ok"] == 0 and reasons["already_had_et"] == 0:
        # Exiting 0 here is how this silently "succeeded" while writing nothing, leaving the
        # failure to surface much later as a KeyError inside the training loop. Fail loudly,
        # at the point where the cause is still on screen.
        hint = ""
        if reasons["no_h5"]:
            hint = ("No *_img.h5 under this path. Check data_root/train_subdir in the config, "
                    "or pass --root with the directory your TRAINING config reads.")
        elif reasons["no_seg_nifti"]:
            hint = ("The case folders have no *_seg.nii.gz. Segmentations ship only with the "
                    "BraTS training cohort -- confirm they were downloaded alongside the "
                    "contrast volumes, and that et_labels matches this BraTS release.")
        elif isinstance(first_err, OSError):
            hint = ("h5py could not open the files read-write. On a networked filesystem this "
                    "is usually HDF5 file locking: retry with HDF5_USE_FILE_LOCKING=FALSE set, "
                    "and check the files are not read-only.")
        elif isinstance(first_err, ValueError):
            hint = ("Geometry mismatch: the seg does not line up with the stored images, so "
                    "the h5 was written with different crop/slice settings than this config. "
                    "Match crop_size / start_slice / end_slice to the generating run.")
        raise SystemExit(f"ERROR: backfill wrote 0 files under {subdir_path}. {hint}")


def write_manifest(subdir_path, rows, cfg):
    path = os.path.join(subdir_path, f"constraint_maps_manifest_K{cfg.n_clusters}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject_id", "n_slices", "constraint_map_path", "img_path"])
        w.writerows(rows)
    print(f"Manifest written: {path}  ({len(rows)} subjects)")


def main():
    ap = argparse.ArgumentParser(description="Offline constraint-map generation for CCL")
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "cmap_config.yaml"))
    ap.add_argument("--add-seg-only", action="store_true",
                    help="do NOT regenerate: only backfill 'seg' + 'et' into existing "
                         "<case>_img.h5 files (see add_seg_to_existing)")
    ap.add_argument("--overwrite-seg", action="store_true",
                    help="with --add-seg-only, replace existing 'seg'/'et' datasets")
    ap.add_argument("--root", action="append", default=None, metavar="DIR",
                    help="with --add-seg-only: back-fill THIS directory instead of "
                         "data_root/<train_subdir>. Repeatable. Give it the exact roots your "
                         "TRAINING config reads (e.g. .../BraTS2021_DataSet_train and "
                         "..._val) -- the config's train_subdir is the pre-split source, "
                         "which is a different path even when the split dirs symlink into it.")
    args = ap.parse_args()
    cfg = load_config(args.config)
    cfg.overwrite_seg = args.overwrite_seg

    if cfg.repo_root:
        sys.path.insert(0, cfg.repo_root)
        os.chdir(cfg.repo_root)

    # The backfill reads niftis and writes h5 -- no PCA/TVD/k-means, no GPU. Importing them
    # anyway would make it fail on a login node for reasons unrelated to what it is doing.
    fns = None
    if not args.add_seg_only:
        from preprocessing.pca import pca_pixelwise as pca_fn
        from solvers.tvd import tvd_fista as tvd_fn
        from preprocessing.kmeans import minibatch_kmeans as kmeans_fn
        fns = (pca_fn, tvd_fn, kmeans_fn)
        cfg.device = torch.device(cfg.device)

    def run(subdir_path):
        if not os.path.isdir(subdir_path):
            raise SystemExit(f"ERROR: not a directory: {subdir_path}")
        cases = cfg.cases if cfg.cases else sorted(
            d for d in os.listdir(subdir_path) if os.path.isdir(os.path.join(subdir_path, d)))
        if args.add_seg_only:
            print(f"\n=== {subdir_path}: {len(cases)} subjects (BACKFILL seg/et only) ===")
            add_seg_to_existing(cases, subdir_path, cfg)
            return
        print(f"\n=== {subdir_path}: {len(cases)} subjects "
              f"(K={cfg.n_clusters}, crop={cfg.crop_size}, save_image={cfg.save_image}, "
              f"in_place={cfg.output_root is None}) ===")
        rows = []
        process_split(cases, subdir_path, cfg, fns, rows)
        if cfg.write_manifest:
            write_manifest(subdir_path, rows, cfg)

    if args.root:
        if not args.add_seg_only:
            raise SystemExit("ERROR: --root is only supported with --add-seg-only.")
        for r in args.root:
            run(os.path.abspath(os.path.expanduser(r)))
    else:
        run(os.path.join(cfg.data_root, cfg.train_subdir))
        if cfg.process_test:
            run(os.path.join(cfg.data_root, cfg.test_subdir))
    print("\nAll done.")


if __name__ == "__main__":
    main()
