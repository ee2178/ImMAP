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
                                           ['slice_index']  (n,)         int32
Plus a manifest CSV at the subdir root listing every subject and its kept-slice count.

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
    Returns (param, image, image_raw, fg, stats):
        param     : (D, H, W) int16      constraint map (background = 0 if mask_background)
        image     : (D, C, H, W) float32 NORMALIZED image (within-brain zscore/minmax, bg=0)
        image_raw : (D, C, H, W) float32 UNNORMALIZED, UNCLIPPED raw intensities (bg=0)
        fg        : (D, H, W) bool        brain mask (from raw intensities)
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
    return param, image, image_raw, fg, stats


# ----------------------------------------------------------------------------
# Per-subject HDF5 writing
# ----------------------------------------------------------------------------
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
    f.attrs["axis_order"] = "N,H,W,C ; slice_index maps N back to the original nifti z-axis"


def save_subject(out_dir, case, image_hwc, image_raw_hwc, param_hwc, mask_hwc,
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
            _common_attrs(f, case, slice_idx, orig_depth, stats, cfg)
    return cmap_path, img_path


def prepare_arrays(param, image, image_raw, fg, cfg):
    """Crop + brain-fraction slice selection; return (img_hwc, img_raw_hwc, param_hwc,
    mask_hwc, slice_idx, orig_depth). A slice is kept iff brain covers >= min_brain_frac of
    the FOV (subsumes the old 'any foreground' rule and drops near-empty top/bottom slices)."""
    orig_depth = param.shape[0]
    if cfg.crop_size:
        image = center_crop_spatial(image, cfg.crop_size, h_axis=2, w_axis=3)
        image_raw = center_crop_spatial(image_raw, cfg.crop_size, h_axis=2, w_axis=3)
        param = center_crop_spatial(param, cfg.crop_size, h_axis=1, w_axis=2)
        fg = center_crop_spatial(fg, cfg.crop_size, h_axis=1, w_axis=2)

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
    return img_hwc, img_raw_hwc, param_hwc, mask_hwc, slice_idx, orig_depth


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
    n_ok = n_fail = n_skip = total_slices = 0
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
            param, image, image_raw, fg, stats = constraint_map_for_subject(src_dir, cfg, pca_fn, tvd_fn, kmeans_fn)
            img_hwc, img_raw_hwc, param_hwc, mask_hwc, slice_idx, orig_depth = prepare_arrays(
                param, image, image_raw, fg, cfg)
            cpath, ipath = save_subject(out_dir, case, img_hwc, img_raw_hwc, param_hwc, mask_hwc,
                                        slice_idx, orig_depth, stats, cfg)
            manifest_rows.append([case, param_hwc.shape[0], cpath, ipath])
            n_ok += 1
            total_slices += param_hwc.shape[0]
            print(f"[{i + 1}/{len(cases)}] {case}: {param_hwc.shape[0]}/{orig_depth} slices "
                  f"-> {os.path.basename(cpath)}")
        except Exception as e:
            n_fail += 1
            print(f"[{i + 1}/{len(cases)}] {case}: FAILED ({type(e).__name__}: {e})")
    print(f"  -> split done: ok={n_ok}, skipped={n_skip}, failed={n_fail}, "
          f"new slices={total_slices}")


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
    cfg = load_config(ap.parse_args().config)

    if cfg.repo_root:
        sys.path.insert(0, cfg.repo_root)
        os.chdir(cfg.repo_root)
    from preprocessing.pca import pca_pixelwise as pca_fn
    from solvers.tvd import tvd_fista as tvd_fn
    from preprocessing.kmeans import minibatch_kmeans as kmeans_fn
    fns = (pca_fn, tvd_fn, kmeans_fn)

    cfg.device = torch.device(cfg.device)

    def run(subdir):
        subdir_path = os.path.join(cfg.data_root, subdir)
        cases = cfg.cases if cfg.cases else sorted(
            d for d in os.listdir(subdir_path) if os.path.isdir(os.path.join(subdir_path, d)))
        print(f"\n=== {subdir}: {len(cases)} subjects "
              f"(K={cfg.n_clusters}, crop={cfg.crop_size}, save_image={cfg.save_image}, "
              f"in_place={cfg.output_root is None}) ===")
        rows = []
        process_split(cases, subdir_path, cfg, fns, rows)
        if cfg.write_manifest:
            write_manifest(subdir_path, rows, cfg)

    run(cfg.train_subdir)
    if cfg.process_test:
        run(cfg.test_subdir)
    print("\nAll done.")


if __name__ == "__main__":
    main()
