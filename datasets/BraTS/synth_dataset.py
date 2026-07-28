# -*- coding: utf-8 -*-
"""
Contrast-synthesis dataset. Reads the SAME per-subject *_img.h5 files written by
generate_constraint_maps.py (which store all contrasts), and returns:
    X    : (len(input_idx), H, W)   e.g. T1, T2, FLAIR
    y    : (len(target_idx), H, W)  e.g. T1ce (contrast-enhanced)
    mask : (1, H, W)                brain mask
    et   : (1, H, W)                enhancing-tumor mask; all zeros unless et_mask=True

Stored channel order follows cmap_config.contrasts (default [flair, t1, t1ce, t2]):
    flair=0, t1=1, t1ce=2, t2=3  ->  input (T1,T2,FLAIR)=[1,3,0], target (T1ce)=[2].

Index is built from a directory `root` (scan subject folders; use the symlink train/val
dirs) or a `manifest` CSV. Optional joint random flips for augmentation.

Intensity scaling
-----------------
`scales` applies per-STORED-channel division exactly as I2SBDataset does -- same semantics,
same indexing, so a value that works there works here. The h5 is written z-scored, which
puts most voxels in roughly [-2, 2]; scales=[2,2,2,2] compresses that to about [-1, 1].
None (the default) leaves the data untouched, i.e. the previous behaviour. The division is
applied to the full stored array BEFORE channel selection, so `scales` is indexed by stored
channel and not by position within input_idx.

ET mask
-------
`et_mask=True` reads the 'et' dataset written by preprocessing/cmap.py and returns it as a
fourth item, for ET-weighted supervision (`lam_et` in the training loops). It raises if the
h5 predates that change rather than substituting zeros, which would silently turn the ET
loss term into a no-op. Backfill an existing dataset with:
    python preprocessing/cmap.py --config config/BraTS/cmap.yaml --add-seg-only

Geometry
--------
X, y, mask and et are concatenated into ONE tensor before the random crop / flips and split
afterwards, so every random parameter is shared and the four stay pixel-aligned. (The mask
previously bypassed the transform, which since crop_size started being forwarded left it at
the uncropped size and made `image * organ_mask` a shape error.)
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
            scales                  (list) per-STORED-channel divisor; None = no scaling
            et_mask                 (bool) return the enhancing-tumor mask (needs 'et' in h5)
        """
        self.input_idx = list(getattr(cfg, "input_idx", [1, 3, 0]))   # T1, T2, FLAIR
        self.target_idx = list(getattr(cfg, "target_idx", [2]))       # T1ce
        self.et_mask = bool(getattr(cfg, "et_mask", False))

        scales = getattr(cfg, "scales", None)
        self.scales = None if scales is None else np.asarray(scales, dtype=np.float32)
        if self.scales is not None:
            need = max(self.input_idx + self.target_idx) + 1
            if self.scales.size < need:
                raise ValueError(f"scales has {self.scales.size} entries but stored channel "
                                 f"{need - 1} is used by input_idx/target_idx")
            if not np.all(self.scales > 0):
                raise ValueError(f"scales must all be > 0 (they are divisors), got {scales}")

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

        self.transform = transforms.Compose(tfms) if tfms else None

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
        h = self._handle(self.img_paths[fi])
        img = np.asarray(h["img"][li])                                  # (H, W, C)
        brain_mask = np.asarray(h["mask"][li])                          # (H, W, 1)

        # per-contrast scaling, on stored-channel indices, before channel selection.
        # Divide (not multiply) to match I2SBDataset.
        if self.scales is not None:
            img = img / self.scales[None, None, :]

        def chw(a):
            return torch.from_numpy(
                np.ascontiguousarray(np.transpose(a, (2, 0, 1)), dtype=np.float32))

        X = chw(img[..., self.input_idx])                               # (Cin, H, W)
        y = chw(img[..., self.target_idx])                              # (Cout, H, W)
        brain_mask = chw(brain_mask)                                    # (1, H, W)

        if self.et_mask:
            if "et" not in h:
                raise KeyError(
                    f"'et' not in {self.img_paths[fi]} (keys={list(h.keys())}). This h5 "
                    f"predates ET-mask storage. Backfill it in place with: python "
                    f"preprocessing/cmap.py --config config/BraTS/cmap.yaml --add-seg-only")
            et = chw(np.asarray(h["et"][li]))                           # (1, H, W) 0/1
        else:
            et = torch.zeros_like(brain_mask)

        # Apply transform consistently to X, y, mask and et -- one stack, one set of
        # random parameters, so the crop and flips cannot desynchronize them.
        if self.transform is not None:
            nX, ny = X.shape[0], y.shape[0]
            stacked = self.transform(torch.cat([X, y, brain_mask, et], dim=0))
            X = stacked[:nX]
            y = stacked[nX:nX + ny]
            brain_mask = stacked[nX + ny:nX + ny + 1]
            et = stacked[nX + ny + 1:]

        return X, y, brain_mask, et
