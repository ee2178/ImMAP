# -*- coding: utf-8 -*-
"""
CCL pretraining dataset — PyTorch port of `generate_clusters`.

Reads the per-subject HDF5 files written by generate_constraint_maps.py. The index can
be built from EITHER:
  * a directory root  (cfg.root)     -> scans subject subfolders (use with the symlink
                                        train/val dirs from make_split_symlinks.py), or
  * a manifest CSV    (cfg.manifest) -> the constraint_maps_manifest_K*.csv

Each __getitem__ returns one slice as:
    X      : (C, H, W) float32         multi-contrast image (selected contrasts)
    y_true : (Hc, Wc, 2) float32       [...,0]=majority constraint label, [...,1]=anchor mask
where Hc = H // patch_size, Wc = W // patch_size. Default collation -> (B, C, H, W) and
(B, Hc, Wc, 2), the signature ConstrainedContrastiveLoss.forward expects.
"""

import os
import csv
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

import h5py


# ----------------------------------------------------------------------------
# numeric helpers (numpy; run inside dataloader workers)
# ----------------------------------------------------------------------------
def patch_majority(arr, patch, num_classes):
    """Majority (mode) label per non-overlapping patch x patch block. arr: (H, W) int."""
    if patch == 1:
        return arr.astype(np.int64)
    H, W = arr.shape
    Hc, Wc = H // patch, W // patch
    arr = arr[:Hc * patch, :Wc * patch]
    blocks = (arr.reshape(Hc, patch, Wc, patch)
                 .transpose(0, 2, 1, 3)
                 .reshape(Hc, Wc, patch * patch))
    counts = (blocks[..., None] == np.arange(num_classes)).sum(axis=2)
    return counts.argmax(axis=-1).astype(np.int64)


def make_sampling_mask(fg_patch, n_samples, rng):
    """Pick up to n_samples random foreground patches as anchors. fg_patch: (Hc, Wc) {0,1}."""
    Hc, Wc = fg_patch.shape
    flat = fg_patch.reshape(-1)
    fg_idx = np.where(flat > 0)[0]
    rng.shuffle(fg_idx)
    sel = fg_idx[:n_samples]
    m = np.zeros(flat.shape[0], dtype=np.int64)
    m[sel] = 1
    return m.reshape(Hc, Wc)


# ----------------------------------------------------------------------------
# index builders
# ----------------------------------------------------------------------------
def _subject_files(subject_dir, n_clusters=None):
    """Return (cmap_path, img_path) for one subject folder, or (None, None) if no map."""
    cms = sorted(glob.glob(os.path.join(subject_dir, "*_constraint_map_K*.h5")))
    if n_clusters is not None:
        cms = [p for p in cms if p.endswith(f"_K{n_clusters}.h5")]
    if not cms:
        return None, None
    imgs = sorted(glob.glob(os.path.join(subject_dir, "*_img.h5")))
    return cms[0], (imgs[0] if imgs else "")


def index_from_root(root, n_clusters=None):
    """Scan a directory of subject folders (symlinks ok). Returns lists + per-file n_slices."""
    cmap_paths, img_paths, n_slices = [], [], []
    for subj in sorted(os.listdir(root)):
        sdir = os.path.join(root, subj)
        if not os.path.isdir(sdir):
            continue
        cmap, img = _subject_files(sdir, n_clusters)
        if cmap is None:
            continue
        with h5py.File(cmap, "r") as f:
            n = int(f["param"].shape[0])
        cmap_paths.append(cmap)
        img_paths.append(img)
        n_slices.append(n)
    if not cmap_paths:
        raise RuntimeError(f"No constraint-map h5 files found under {root}")
    return cmap_paths, img_paths, n_slices


def index_from_manifest(manifest_csv):
    cmap_paths, img_paths, n_slices = [], [], []
    with open(manifest_csv) as f:
        for row in csv.DictReader(f):
            cmap_paths.append(row["constraint_map_path"])
            img_paths.append(row["img_path"])
            n_slices.append(int(row["n_slices"]))
    return cmap_paths, img_paths, n_slices


# ----------------------------------------------------------------------------
# dataset
# ----------------------------------------------------------------------------
class CCLPretrainDataset(Dataset):
    def __init__(self, cfg):
        """
        cfg attributes:
            root  OR manifest       (one is required; root takes precedence)
            n_clusters              (int, optional) disambiguates the K of the map file
            patch_size              (int)   input->feature downsampling factor (>=1)
            num_samples_loss_eval   (int)   anchors per slice
            contrast_idx            (list)  stored channels fed to the network
            foreground_from_label   (bool)  brain = (param > 0) [exact] vs X[...,0] > min
        """
        self.patch_size = int(getattr(cfg, "patch_size", 1))
        self.num_samples = int(getattr(cfg, "num_samples_loss_eval", 100))
        self.contrast_idx = list(getattr(cfg, "contrast_idx", [0, 1, 2, 3]))
        self.fg_from_label = bool(getattr(cfg, "foreground_from_label", True))

        root = getattr(cfg, "root", None)
        manifest = getattr(cfg, "manifest", None)
        n_clusters = getattr(cfg, "n_clusters", None)
        if root:
            self.cmap_paths, self.img_paths, n_slices = index_from_root(root, n_clusters)
        elif manifest:
            self.cmap_paths, self.img_paths, n_slices = index_from_manifest(manifest)
        else:
            raise ValueError("CCLPretrainDataset needs cfg.root or cfg.manifest")

        if any(not p for p in self.img_paths):
            raise NotImplementedError(
                "Some subjects have no <subject>_img.h5 (save_image was false). Regenerate "
                "with save_image: true, or request the nii-reloading dataset variant.")

        file_id, local = [], []
        for fi, n in enumerate(n_slices):
            file_id.extend([fi] * n)
            local.extend(range(n))
        self.file_id = np.asarray(file_id, dtype=np.int64)
        self.local = np.asarray(local, dtype=np.int64)
        self._cmap_h, self._img_h = {}, {}     # lazy per-worker handles

    def __len__(self):
        return self.file_id.shape[0]

    def _handle(self, cache, path):
        h = cache.get(path)
        if h is None:
            h = h5py.File(path, "r")
            cache[path] = h
        return h

    def __getitem__(self, idx):
        fi = int(self.file_id[idx])
        li = int(self.local[idx])
        img_ds = self._handle(self._img_h, self.img_paths[fi])["img"]
        cmap_ds = self._handle(self._cmap_h, self.cmap_paths[fi])["param"]

        x = np.asarray(img_ds[li])[..., self.contrast_idx]      # (H, W, C)
        param = np.squeeze(np.asarray(cmap_ds[li]), axis=-1)    # (H, W) int

        if self.fg_from_label:
            fg = (param > 0).astype(np.int64)
        else:
            x0 = x[..., 0]
            fg = (x0 > x0.min()).astype(np.int64)

        p = self.patch_size
        label_ds = patch_majority(param, p, int(param.max()) + 1)
        fg_ds = patch_majority(fg, p, 2)

        rng = np.random.default_rng()
        samp = make_sampling_mask(fg_ds, self.num_samples, rng)

        X = np.ascontiguousarray(np.transpose(x, (2, 0, 1)), dtype=np.float32)   # (C, H, W)
        y_true = np.stack([label_ds, samp], axis=-1).astype(np.float32)          # (Hc, Wc, 2)
        return torch.from_numpy(X), torch.from_numpy(y_true)
