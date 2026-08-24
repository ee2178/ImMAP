# -*- coding: utf-8 -*-
"""
I2SB dataset. Reads the SAME per-subject *_img.h5 files as SynthesisDataset, but is built
for the Schrodinger bridge: it returns the two 1-channel bridge endpoints plus optional
conditioning, and it applies ADJUSTABLE PER-CONTRAST SCALE FACTORS instead of any channel
normalization.

    x0   : (1, H, W)          target contrast      (default T1ce, stored idx 2)
    x1   : (1, H, W)          prior / bridge start (default T1,  stored idx 1)
    cond : (n_cond, H, W)     conditioning stack   (default FLAIR,T1,T2 = [0,1,3]); () if off
    mask : (1, H, W)          brain mask
    et   : (1, H, W)          enhancing-tumor mask -- ONLY when et_mask=True (see below)

ET MASK. `et_mask=True` appends a FIFTH item, the enhancing-tumor mask, for `et_weight` in
train_i2sb. It is appended rather than always returned (unlike SynthesisDataset, which always
returns 4) because this dataset's 4-tuple is already unpacked positionally in several places --
datasets/BraTS/stack_dataset.py and the i2sb notebooks -- so an unconditional fifth element would
break them. Default False keeps every existing config and notebook on the old 4-tuple.

Stored channel order follows cmap_config.contrasts: [flair, t1, t1ce, t2] -> flair=0, t1=1,
t1ce=2, t2=3.

Normalization: NONE is applied here. Each stored contrast c is DIVIDED by scales[c]
(default 1.0). This lets you dial the relative dynamic range of each contrast directly, which
matters for I2SB because the bridge endpoints (x0, x1) and the absolute noise-schedule stds
must live on a compatible scale. If your *_img.h5 was written z-scored, the scales rescale that;
if you regenerated it raw (cmap normalize: none), the scales map raw intensities into range.
"""

import os
import csv
import glob
import numpy as np
import torch
import torchvision.transforms as transforms

from torch.utils.data import Dataset

import h5py

from datasets.BraTS.synth_dataset import filter_by_key


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


class I2SBDataset(Dataset):
    def __init__(self, cfg):
        """
        cfg attributes:
            root OR manifest    (one required; root preferred)
            x0_idx    (int)     stored channel used as the bridge target   (T1ce=2)
            x1_idx    (int)     stored channel used as the bridge prior     (T1=1)
            cond_idx  (list)    stored channels used as conditioning; [] disables conditioning
            scales    (list)    per-STORED-channel DIVISORS (len >= max stored idx + 1); the
                                image is divided by these, so 3.0 scales intensities DOWN by 3;
                                None -> all ones. NO other normalization is applied.
            image_key (str)     h5 dataset to read: "img" (normalized) or "img_raw"
                                (unnormalized; requires the generator's save_raw_image: true).
            et_mask   (bool)    also return the enhancing-tumor mask as a 5th item (needs 'et'
                                in the h5); subjects without it are dropped up front.
            center_crop, crop_size, random_flips   (optional geometric augmentation)
        """
        self.x0_idx = int(getattr(cfg, "x0_idx", 2))     # T1ce
        self.x1_idx = int(getattr(cfg, "x1_idx", 1))     # T1
        self.cond_idx = list(getattr(cfg, "cond_idx", [0, 1, 3]))   # FLAIR, T1, T2
        # which stored image to read: "img" (normalized) or "img_raw" (raw: no clip/normalize).
        # img_raw preserves the true inter-contrast intensities and needs save_raw_image: true.
        self.image_key = str(getattr(cfg, "image_key", "img"))

        # bridge prior x1: "contrast" -> stored channel x1_idx (default T1); "synth" -> the
        # precomputed e2e synthesis output (h5 dataset `yhat_key`, from synth_precompute.py) for the
        # yhat -> T1ce bridge. yhat is stored in the "img" (z-scored) space and gets scales[x0_idx]
        # applied below, exactly like x0, so the two share a scale.
        self.x1_source = str(getattr(cfg, "x1_source", "contrast"))
        self.yhat_key = str(getattr(cfg, "yhat_key", "yhat"))
        if self.x1_source not in ("contrast", "synth"):
            raise ValueError(f"x1_source must be 'contrast' or 'synth', got {self.x1_source!r}")

        # enhancing-tumor mask, for train_i2sb's et_weight. Off by default (keeps the 4-tuple).
        self.et_mask = bool(getattr(cfg, "et_mask", False))

        scales = getattr(cfg, "scales", None)
        self.scales = None if scales is None else np.asarray(scales, dtype=np.float32)

        # geometric augmentation applied jointly to x0/x1/cond/mask (see __getitem__)
        tfms = []
        if getattr(cfg, "center_crop", None) is not None:
            tfms.append(transforms.CenterCrop(cfg.center_crop))
        if getattr(cfg, "crop_size", None) is not None:
            tfms.append(transforms.RandomCrop(cfg.crop_size))
        if getattr(cfg, "random_flips", False):
            tfms += [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
        self.transform = transforms.Compose(tfms) if tfms else None

        root = getattr(cfg, "root", None)
        manifest = getattr(cfg, "manifest", None)
        if root:
            self.img_paths, n_slices = index_img_from_root(root)
        elif manifest:
            self.img_paths, n_slices = index_img_from_manifest(manifest)
        else:
            raise ValueError("I2SBDataset needs cfg.root or cfg.manifest")

        # BraTS ships subjects with no segmentation nifti, so they have no 'et' and never will.
        # Drop them up front (as SynthesisDataset does) -- otherwise et_weight silently degrades
        # to 1 on those slices, i.e. a different objective on an unknown fraction of the data.
        if self.et_mask:
            n_before = len(self.img_paths)
            self.img_paths, n_slices, dropped = filter_by_key(self.img_paths, n_slices, "et")
            if not self.img_paths:
                raise RuntimeError(
                    f"et_mask=True but NONE of the {n_before} subjects under "
                    f"{root or manifest} have an 'et' dataset. Backfill them in place with: "
                    f"python preprocessing/cmap.py --config config/BraTS/cmap.yaml --add-seg-only")
            if dropped:
                print(f"[I2SBDataset] et_mask: excluded {len(dropped)}/{n_before} subject(s) "
                      f"({len(dropped) / n_before:.1%}) with no 'et' in their h5")

        file_id, local = [], []
        for fi, n in enumerate(n_slices):
            file_id.extend([fi] * n)
            local.extend(range(n))
        self.file_id = np.asarray(file_id, dtype=np.int64)
        self.local = np.asarray(local, dtype=np.int64)
        self._img_h = {}                                  # lazy per-worker handles

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
        if self.image_key not in h:
            raise KeyError(
                f"'{self.image_key}' not in {self.img_paths[fi]} (keys={list(h.keys())}). "
                f"Regenerate with save_raw_image: true to get 'img_raw', or set image_key: 'img'.")
        img = np.asarray(h[self.image_key][li])           # (H, W, Cstored)
        mask = np.asarray(h["mask"][li])                  # (H, W, 1)

        # per-contrast scaling (the ONLY intensity transform); no channel normalization
        if self.scales is not None:
            # We choose to divide by the scale factor in practice. 
            img = img / self.scales[None, None, :]

        def chw(a):
            return torch.from_numpy(np.ascontiguousarray(np.transpose(a, (2, 0, 1)), dtype=np.float32))

        x0 = chw(img[..., [self.x0_idx]])                 # (1, H, W)
        if self.x1_source == "synth":
            if self.yhat_key not in h:
                raise KeyError(
                    f"'{self.yhat_key}' not in {self.img_paths[fi]} (keys={list(h.keys())}). Run "
                    f"datasets/BraTS/synth_precompute.py first, or set x1_source='contrast'.")
            yhat = np.asarray(h[self.yhat_key][li]).astype(np.float32)   # (H, W) z-scored, unscaled
            if self.scales is not None:
                yhat = yhat / float(self.scales[self.x0_idx])           # match x0's per-contrast scale
            x1 = torch.from_numpy(np.ascontiguousarray(yhat[None]))     # (1, H, W)
        else:
            x1 = chw(img[..., [self.x1_idx]])             # (1, H, W)  stored contrast (default T1)
        cond = chw(img[..., self.cond_idx]) if self.cond_idx else torch.zeros(0, *img.shape[:2])
        mask = chw(mask)                                  # (1, H, W)

        if self.et_mask:
            if "et" not in h:
                raise KeyError(
                    f"'et' not in {self.img_paths[fi]} (keys={list(h.keys())}). This h5 "
                    f"predates ET-mask storage. Backfill it in place with: python "
                    f"preprocessing/cmap.py --config config/BraTS/cmap.yaml --add-seg-only")
            et = chw(np.asarray(h["et"][li]))             # (1, H, W) 0/1
        else:
            et = None

        # joint geometric transform: stack -> transform -> split (keeps alignment)
        if self.transform is not None:
            n0, n1, ncond = x0.shape[0], x1.shape[0], cond.shape[0]
            parts = [x0, x1, cond, mask] + ([et] if et is not None else [])
            stacked = self.transform(torch.cat(parts, dim=0))
            x0 = stacked[:n0]
            x1 = stacked[n0:n0 + n1]
            cond = stacked[n0 + n1:n0 + n1 + ncond]
            mask = stacked[n0 + n1 + ncond:n0 + n1 + ncond + 1]
            if et is not None:
                et = stacked[n0 + n1 + ncond + 1:]

        return (x0, x1, cond, mask) if et is None else (x0, x1, cond, mask, et)
