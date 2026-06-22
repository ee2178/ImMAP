# -*- coding: utf-8 -*-
"""
Contrast-synthesis dataset. Reads the SAME per-subject *_img.h5 files written by
generate_constraint_maps.py (which store all contrasts), and returns:
    X : (len(input_idx), H, W)   e.g. T1, T2, FLAIR
    y : (len(target_idx), H, W)  e.g. T1ce (contrast-enhanced)

Stored channel order follows cmap_config.contrasts (default [flair, t1, t1ce, t2]):
    flair=0, t1=1, t1ce=2, t2=3  ->  input (T1,T2,FLAIR)=[1,3,0], target (T1ce)=[2].

Index is built from a directory `root` (scan subject folders; use the symlink train/val
dirs) or a `manifest` CSV. Optional joint random flips for augmentation.
"""

import os
import csv
import glob
import numpy as np
import torch
import torchvision.transforms as transforms

from torch.utils.data import Dataset

import h5py


def index_img_from_root(root):
    img_paths, n_slices = [], []
    for subj in sorted(os.listdir(root)):
        sdir = os.path.join(root, subj)
        if not os.path.isdir(sdir):
            continue
        imgs = sorted(glob.glob(os.path.join(sdir, "*_img.h5")))
        if not imgs:
            continue
        with h5py.File(imgs[0], "r") as f:
            n = int(f["img"].shape[0])
        img_paths.append(imgs[0])
        n_slices.append(n)
    if not img_paths:
        raise RuntimeError(f"No *_img.h5 found under {root}")
    return img_paths, n_slices


def index_img_from_manifest(manifest_csv):
    img_paths, n_slices = [], []
    with open(manifest_csv) as f:
        for row in csv.DictReader(f):
            if not row["img_path"]:
                raise NotImplementedError("manifest has no img_path (save_image was false).")
            img_paths.append(row["img_path"])
            n_slices.append(int(row["n_slices"]))
    return img_paths, n_slices


class SynthesisDataset(Dataset):
    def __init__(self, cfg):
        """
        cfg attributes:
            root OR manifest        (one required; root preferred)
            input_idx               (list) stored channels used as network input
            target_idx              (list) stored channels used as target
        """
        self.input_idx = list(getattr(cfg, "input_idx", [1, 3, 0]))   # T1, T2, FLAIR
        self.target_idx = list(getattr(cfg, "target_idx", [2]))       # T1ce

        # Make loader capable of handling random transforms
        tfms = []

        center_crop = getattr(cfg, "center_crop", None)
        crop_size = getattr(cfg, "crop_size", None)

        if center_crop is not None:
            tfms.append(transforms.CenterCrop(center_crop))

        if crop_size is not None:
            tfms.append(transforms.RandomCrop(crop_size))

        if getattr(cfg, "random_flips", False):
            tfms += [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]

        self.transform = transforms.Compose(tfms)

        root = getattr(cfg, "root", None)
        manifest = getattr(cfg, "manifest", None)
        if root:
            self.img_paths, n_slices = index_img_from_root(root)
        elif manifest:
            self.img_paths, n_slices = index_img_from_manifest(manifest)
        else:
            raise ValueError("SynthesisDataset needs cfg.root or cfg.manifest")

        file_id, local = [], []
        for fi, n in enumerate(n_slices):
            file_id.extend([fi] * n)
            local.extend(range(n))
        self.file_id = np.asarray(file_id, dtype=np.int64)
        self.local = np.asarray(local, dtype=np.int64)
        self._img_h = {}

    def __len__(self):
        return self.file_id.shape[0]

    def _handle(self, path):
        h = self._img_h.get(path)
        if h is None:
            h = h5py.File(path, "r")
            self._img_h[path] = h
        return h

    def __getitem__(self, idx):
        fi = int(self.file_id[idx])
        li = int(self.local[idx])
        img = np.asarray(self._handle(self.img_paths[fi])["img"][li])   # (H, W, C)
        brain_mask = np.asarray(self._handle(self.img_paths[fi])["mask"][li])
        eps = 1e-8

        # Send to torch right away
        X = torch.from_numpy(np.transpose(img[..., self.input_idx], (2, 0, 1)))           # (Cin, H, W)
        y = torch.from_numpy(np.transpose(img[..., self.target_idx], (2, 0, 1)))          # (Cout, H, W)
        brain_mask = torch.from_numpy(np.transpose(brain_mask, (2, 0, 1)))
        
        xy = torch.cat([X, y], dim=0)
        # Apply transform consistently to X and y
        if self.transform is not None:
            xy = self.transform(xy)

        X = xy[:len(self.input_idx)]
        y = xy[len(self.input_idx):]

        return X, y, brain_mask
