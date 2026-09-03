# -*- coding: utf-8 -*-
"""
NYUMets guided-I2SB dataset: the usual bridge endpoints plus a GUIDE image.

Reads the same per-session h5 files as datasets/BraTS/i2sb_dataset.py (written by
preprocessing/nyumets_h5.py), one directory per (patient, study):

    x0    (1, H, W)          target contrast            default CT1  (stored idx 2)
    x1    (1, H, W)          bridge prior               default T1   (stored idx 1)
    cond  (n_cond, H, W)     conditioning stack         default FLAIR, T1, T2
    mask  (1, H, W)          brain mask
    guide (n_guides, H, W)   the guide -- ONLY when guide_mode != "none"

The guide is APPENDED rather than always returned, following the `et_mask` precedent in
I2SBDataset: `train_i2sb` and several notebooks unpack the 4-tuple positionally, so an
unconditional fifth element would break them. guide_mode="none" is therefore byte-identical
to the plain synthesis dataset.

GUIDE MODES
-----------
    none            no guide; plain contrast synthesis (the 4-tuple)
    far_slice       same study, same contrast, at least `min_slice_gap` slices away
    central_slice   same study, same contrast, the volume's central slice
    other_study     a DIFFERENT study of the SAME patient, same contrast
    same_slice      the target slice itself -- an ORACLE, for sanity checks only

`min_slice_gap` exists because an adjacent slice is nearly the answer: neighbouring slices of
the same contrast are so correlated that the guide branch would let the network copy rather
than synthesise. Default 5.

WHAT THE GUIDE IS NOT ALIGNED TO. `other_study` guides come from a separate acquisition with
its own affine, so they are NOT registered to the target. They carry the patient's enhancement
morphology, not voxel correspondence. Slice indices across studies mean nothing in common,
which is why `guide_slice` defaults to the guide volume's centre rather than the target's index.
Prior-vs-later ordering is deliberately NOT modelled: NYUMets study ids are random 10-digit
`image_id`s carrying no time, and ordering needs the release's CSV tables joined on image_id
(`time_from_gk_days`). Until those are on disk, "any other study" is the honest formulation.

Scaling matches I2SBDataset exactly: each stored contrast is DIVIDED by `scales[c]`, and no
channel normalization is applied. The guide is divided by `scales[guide_idx]`, so it shares a
scale with whichever contrast it is drawn from.
"""

import os
import glob

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import h5py

GUIDE_MODES = ("none", "far_slice", "central_slice", "other_study", "same_slice")


def index_sessions(root):
    """-> (paths, n_slices, patients) for every <session>/*_img.h5 under `root`."""
    paths, n_slices, patients = [], [], []
    for sub in sorted(os.listdir(root)):
        sdir = os.path.join(root, sub)
        if not os.path.isdir(sdir):
            continue
        hits = sorted(glob.glob(os.path.join(sdir, "*_img.h5")))
        if not hits:
            continue
        with h5py.File(hits[0], "r") as f:
            n = int(f["img"].shape[0])
            # written by nyumets_h5; the directory prefix is the fallback for older files
            pid = f.attrs.get("patient", sub.split("_", 1)[0])
        paths.append(hits[0])
        n_slices.append(n)
        patients.append(str(pid))
    if not paths:
        raise RuntimeError(f"No */*_img.h5 found under {root}")
    return paths, n_slices, patients


class NYUMetsGuidedDataset(Dataset):
    def __init__(self, cfg):
        self.x0_idx = int(getattr(cfg, "x0_idx", 2))            # CT1
        self.x1_idx = int(getattr(cfg, "x1_idx", 1))            # T1
        self.cond_idx = list(getattr(cfg, "cond_idx", [0, 1, 3]))
        self.image_key = str(getattr(cfg, "image_key", "img_median_mad"))

        self.guide_mode = str(getattr(cfg, "guide_mode", "none"))
        if self.guide_mode not in GUIDE_MODES:
            raise ValueError(f"guide_mode must be one of {GUIDE_MODES}, got {self.guide_mode!r}")
        # default: guide with the same contrast as the target (CT1 -> the enhancement pattern)
        self.guide_idx = int(getattr(cfg, "guide_idx", self.x0_idx))
        self.n_guides = int(getattr(cfg, "n_guides", 1))
        self.min_slice_gap = int(getattr(cfg, "min_slice_gap", 5))
        self.guide_slice = str(getattr(cfg, "guide_slice", "central"))   # other_study only
        if self.guide_slice not in ("central", "random"):
            raise ValueError(f"guide_slice must be 'central' or 'random', got {self.guide_slice!r}")
        # deterministic guides make val/test reproducible: the choice is derived from the sample
        # index instead of an RNG, so the same item yields the same guide on every epoch.
        self.deterministic = bool(getattr(cfg, "deterministic", False))

        scales = getattr(cfg, "scales", None)
        self.scales = None if scales is None else np.asarray(scales, dtype=np.float32)

        tfms = []
        if getattr(cfg, "center_crop", None) is not None:
            tfms.append(transforms.CenterCrop(cfg.center_crop))
        if getattr(cfg, "crop_size", None) is not None:
            tfms.append(transforms.RandomCrop(cfg.crop_size))
        if getattr(cfg, "random_flips", False):
            tfms += [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
        self.transform = transforms.Compose(tfms) if tfms else None

        root = getattr(cfg, "root", None)
        if not root:
            raise ValueError("NYUMetsGuidedDataset needs cfg.root")
        paths, n_slices, patients = index_sessions(root)

        # ---- drop files that cannot supply the requested guide, and say so --------------
        keep = list(range(len(paths)))
        if self.guide_mode == "far_slice":
            # a partner at distance >= gap exists for EVERY z iff the volume has more than
            # `gap` slices; below that the mode is undefined for at least the middle slices
            keep = [i for i in keep if n_slices[i] > self.min_slice_gap]
            if len(keep) < len(paths):
                print(f"[NYUMetsGuided] far_slice: dropped {len(paths)-len(keep)}/{len(paths)} "
                      f"session(s) with <= min_slice_gap={self.min_slice_gap} slices")
        elif self.guide_mode == "other_study":
            n_by_pat = {}
            for i in keep:
                n_by_pat[patients[i]] = n_by_pat.get(patients[i], 0) + 1
            keep = [i for i in keep if n_by_pat[patients[i]] >= 2]
            if len(keep) < len(paths):
                print(f"[NYUMetsGuided] other_study: dropped {len(paths)-len(keep)}/{len(paths)} "
                      f"session(s) whose patient has only one study in this split")
        if not keep:
            raise RuntimeError(f"guide_mode={self.guide_mode!r} left no usable sessions under {root}")

        self.img_paths = [paths[i] for i in keep]
        self.n_slices = [n_slices[i] for i in keep]
        self.patients = [patients[i] for i in keep]

        # file indices of the OTHER studies of the same patient, for other_study
        by_pat = {}
        for i, p in enumerate(self.patients):
            by_pat.setdefault(p, []).append(i)
        self.siblings = [[j for j in by_pat[p] if j != i] for i, p in enumerate(self.patients)]

        file_id, local = [], []
        for fi, n in enumerate(self.n_slices):
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

    def _pick(self, n, idx, salt):
        """One index in [0, n). Uses torch's RNG, which DataLoader seeds per worker -- numpy's
        is not seeded per worker, so np.random here would hand every worker the same stream."""
        if n <= 0:
            raise ValueError("nothing to pick from")
        if self.deterministic:
            return int((idx * 1103515245 + salt * 12345) % n)      # stable, index-derived
        return int(torch.randint(n, (1,)).item())

    def _guide_slices(self, fi, li, idx):
        """-> list of (file_index, slice_index) of length n_guides."""
        out = []
        for g in range(self.n_guides):
            if self.guide_mode == "same_slice":
                out.append((fi, li))
            elif self.guide_mode == "central_slice":
                out.append((fi, self.n_slices[fi] // 2))
            elif self.guide_mode == "far_slice":
                n = self.n_slices[fi]
                far = [z for z in range(n) if abs(z - li) >= self.min_slice_gap]
                out.append((fi, far[self._pick(len(far), idx, g + 1)]))
            elif self.guide_mode == "other_study":
                sib = self.siblings[fi]
                gf = sib[self._pick(len(sib), idx, g + 1)]
                gn = self.n_slices[gf]
                gz = gn // 2 if self.guide_slice == "central" else self._pick(gn, idx, g + 101)
                out.append((gf, gz))
            else:
                raise AssertionError(self.guide_mode)
        return out

    def _read(self, fi, li):
        h = self._handle(self.img_paths[fi])
        if self.image_key not in h:
            raise KeyError(f"'{self.image_key}' not in {self.img_paths[fi]} "
                           f"(keys={list(h.keys())})")
        img = np.asarray(h[self.image_key][li])           # (H, W, Cstored)
        if self.scales is not None:
            img = img / self.scales[None, None, :]
        return img, h

    def __getitem__(self, idx):
        fi = int(self.file_id[idx])
        li = int(self.local[idx])
        img, h = self._read(fi, li)
        mask = np.asarray(h["mask"][li])                  # (H, W, 1)

        def chw(a):
            return torch.from_numpy(
                np.ascontiguousarray(np.transpose(a, (2, 0, 1)), dtype=np.float32))

        x0 = chw(img[..., [self.x0_idx]])
        x1 = chw(img[..., [self.x1_idx]])
        cond = chw(img[..., self.cond_idx]) if self.cond_idx else torch.zeros(0, *img.shape[:2])
        mask = chw(mask)

        guide = None
        if self.guide_mode != "none":
            planes = []
            for gf, gz in self._guide_slices(fi, li, idx):
                gimg, _ = self._read(gf, gz)
                planes.append(gimg[..., self.guide_idx])
            guide = chw(np.stack(planes, axis=-1))        # (n_guides, H, W)

        # joint geometric transform: stack -> transform -> split (keeps alignment).
        # The guide rides the SAME crop/flip. For within-volume guides that preserves spatial
        # correspondence with the target, which is what a guided prox needs; for other_study
        # there is no correspondence to preserve either way.
        if self.transform is not None:
            n0, n1, ncond = x0.shape[0], x1.shape[0], cond.shape[0]
            parts = [x0, x1, cond, mask] + ([guide] if guide is not None else [])
            stacked = self.transform(torch.cat(parts, dim=0))
            x0 = stacked[:n0]
            x1 = stacked[n0:n0 + n1]
            cond = stacked[n0 + n1:n0 + n1 + ncond]
            mask = stacked[n0 + n1 + ncond:n0 + n1 + ncond + 1]
            if guide is not None:
                guide = stacked[n0 + n1 + ncond + 1:]

        return (x0, x1, cond, mask) if guide is None else (x0, x1, cond, mask, guide)
