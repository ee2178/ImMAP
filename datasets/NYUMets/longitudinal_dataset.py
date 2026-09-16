# -*- coding: utf-8 -*-
"""
NYUMets guided-I2SB dataset: the usual bridge endpoints plus GUIDE images.

Reads the per-session h5 files written by preprocessing/nyumets_h5.py, one directory per
(patient, study):

    x0    (1, H, W)          target contrast            default CT1  (stored idx 2)
    x1    (1, H, W)          bridge prior               default T1   (stored idx 1)
    cond  (n_cond, H, W)     conditioning stack
    mask  (1, H, W)          brain mask
    guide (G, 1, H, W)       guide planes -- ONLY when guide_mode != "none"

`guide_as_cond=True` instead APPENDS the guide planes to cond (after cond_idx, in guide order)
and returns the 4-TUPLE: for regressors with no guided prox, e.g. an SBUnet whose C grows by G.
Unlike the guided prox, input channels CAN carry intensities straight into the prediction.

RETURN LAYOUT. guide_mode="none" returns the plain 4-TUPLE, byte-identical to the BraTS "i2sb"
loader. Any guided mode returns a DICT {"x0","x1","cond","mask","guide"}. It is a dict and not a
5-tuple on purpose: train_i2sb reads tuple position 4 as the ET mask, so a guide there would be
consumed as `et` and, under et_weight != 1, would weight the loss by the guide image. Keys cannot
collide, and training/i2sb.py::_batch_parts accepts both layouts.

GUIDE MODES -- a string, or a LIST of them whose guide planes are concatenated in order:

    none            no guide; plain contrast synthesis
    same_session    other contrasts of the SAME slice (`guide_contrasts`, default T1, T2, FLAIR).
                    Co-registered by construction -- same session, common acquisition support.
    other_study     `guide_idx` (default CT1) from a DIFFERENT study of the SAME patient
    far_slice       `guide_idx`, same study, >= `min_slice_gap` slices away
    central_slice   `guide_idx`, same study, the volume's central slice
    same_slice      `guide_idx` at the target slice itself -- an ORACLE, for sanity checks only

So ["same_session", "other_study"] gives G = len(guide_contrasts) + n_guides planes: the session
contrasts first, then the longitudinal CT1(s). That combination is what makes a longitudinal run
comparable to a session-only one -- the difference between them is then the CT1 guide and nothing
else.

`other_study` AND ITS LIMITS. Study ids are random 10-digit `image_id`s with no time, and the
imaging carries no acquisition date, so "prior" vs "later" cannot be told apart. Any other study
of the patient is used. Those studies are NOT registered to the target: separate sessions,
separate patient positioning. `guide_slice` picks which slice of the guide volume to use:

    matched   (default) the same RELATIVE depth: z_g = round(z / (n - 1) * (n_g - 1)). Both volumes'
              stored slices were selected by brain fraction, so each spans the brain from inferior
              to superior and relative depth lands near the same anatomical level. This is an
              approximation, not registration -- slice spacing, coverage and head tilt all differ.
    central   the guide volume's central slice
    random    any slice

In-plane misalignment is left to the model: the guided prox searches a nonlocal `guide_window`.

`min_slice_gap` exists because an adjacent slice of the same contrast is nearly the answer.

SAMPLING uses torch's RNG, which DataLoader seeds per worker. numpy's is not seeded per worker, so
np.random here would hand every worker the identical guide stream. `deterministic=True` derives
every choice from the sample index instead, so val/test see the same guide every epoch.

Scaling matches I2SBDataset exactly: each stored contrast is DIVIDED by `scales[c]`; each guide
plane is divided by the scale of the contrast it was drawn from.
"""

import os
import glob

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import h5py

GUIDE_MODES = ("none", "same_session", "other_study", "far_slice", "central_slice", "same_slice")
GUIDE_SLICES = ("matched", "central", "random")


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
            pid = f.attrs.get("patient", sub.split("_", 1)[0])
        if isinstance(pid, bytes):
            pid = pid.decode()
        paths.append(hits[0])
        n_slices.append(n)
        patients.append(str(pid))
    if not paths:
        raise RuntimeError(f"No */*_img.h5 found under {root}")
    return paths, n_slices, patients


def _as_modes(guide_mode):
    modes = [guide_mode] if isinstance(guide_mode, str) else list(guide_mode)
    bad = [m for m in modes if m not in GUIDE_MODES]
    if bad:
        raise ValueError(f"guide_mode entries must be in {GUIDE_MODES}; got {bad}")
    if "none" in modes and len(modes) > 1:
        raise ValueError(f"'none' cannot be combined with other guide modes: {modes}")
    return [m for m in modes if m != "none"]


class NYUMetsGuidedDataset(Dataset):
    def __init__(self, cfg):
        self.x0_idx = int(getattr(cfg, "x0_idx", 2))            # CT1
        self.x1_idx = int(getattr(cfg, "x1_idx", 1))            # T1
        self.cond_idx = list(getattr(cfg, "cond_idx", [1]))
        self.image_key = str(getattr(cfg, "image_key", "img_median_mad"))

        self.modes = _as_modes(getattr(cfg, "guide_mode", "none"))
        self.guide_idx = int(getattr(cfg, "guide_idx", self.x0_idx))
        gc = getattr(cfg, "guide_contrasts", None)
        self.guide_contrasts = [1, 3, 0] if gc is None else [int(c) for c in gc]   # T1, T2, FLAIR
        self.n_guides = int(getattr(cfg, "n_guides", 1))
        self.min_slice_gap = int(getattr(cfg, "min_slice_gap", 5))
        self.guide_slice = str(getattr(cfg, "guide_slice", "matched"))
        if self.guide_slice not in GUIDE_SLICES:
            raise ValueError(f"guide_slice must be one of {GUIDE_SLICES}, got {self.guide_slice!r}")
        if "same_session" in self.modes and not self.guide_contrasts:
            raise ValueError("guide_mode 'same_session' needs a non-empty guide_contrasts")
        self.deterministic = bool(getattr(cfg, "deterministic", False))
        # Append the guide planes to `cond` instead of returning them separately. For regressors
        # with no guided prox (SBUnet, SBCDLNet): the guide becomes ordinary input channels, the
        # batch stays the plain 4-tuple, and the model's C must grow by n_guide_planes.
        self.guide_as_cond = bool(getattr(cfg, "guide_as_cond", False))
        if self.guide_as_cond and not self.modes:
            raise ValueError("guide_as_cond=True needs a guide_mode other than 'none'")

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

        # ---- drop sessions that cannot supply a requested guide, and say how many ----
        keep = list(range(len(paths)))
        if "far_slice" in self.modes:
            before = len(keep)
            # a partner at distance >= gap exists for EVERY z iff the volume has > gap slices
            keep = [i for i in keep if n_slices[i] > self.min_slice_gap]
            if len(keep) < before:
                print(f"[NYUMetsGuided] far_slice: dropped {before-len(keep)}/{before} session(s) "
                      f"with <= min_slice_gap={self.min_slice_gap} slices")
        if "other_study" in self.modes:
            before = len(keep)
            n_by_pat = {}
            for i in keep:
                n_by_pat[patients[i]] = n_by_pat.get(patients[i], 0) + 1
            keep = [i for i in keep if n_by_pat[patients[i]] >= 2]
            if len(keep) < before:
                print(f"[NYUMetsGuided] other_study: dropped {before-len(keep)}/{before} "
                      f"session(s) whose patient has only one study in this split")
        if not keep:
            raise RuntimeError(f"guide_mode={self.modes} left no usable sessions under {root}")

        self.img_paths = [paths[i] for i in keep]
        self.n_slices = [n_slices[i] for i in keep]
        self.patients = [patients[i] for i in keep]

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

    @property
    def n_guide_planes(self):
        """G, the number of guide planes per sample -- fixed for a given configuration."""
        n = 0
        for m in self.modes:
            n += len(self.guide_contrasts) if m == "same_session" else self.n_guides
        return n

    def __len__(self):
        return self.file_id.shape[0]

    def _handle(self, path):
        h = self._img_h.get(path)
        if h is None:
            h = h5py.File(path, "r")
            self._img_h[path] = h
        return h

    def _pick(self, n, idx, salt):
        """One index in [0, n) -- torch RNG (seeded per worker), or index-derived if deterministic."""
        if n <= 0:
            raise ValueError("nothing to pick from")
        if self.deterministic:
            return int((idx * 1103515245 + salt * 12345) % n)
        return int(torch.randint(n, (1,)).item())

    def _guide_planes(self, fi, li, idx):
        """-> list of (file_index, slice_index, stored_contrast), in guide order."""
        out = []
        for mi, mode in enumerate(self.modes):
            salt0 = 1000 * (mi + 1)
            if mode == "same_session":
                out += [(fi, li, c) for c in self.guide_contrasts]
                continue
            for g in range(self.n_guides):
                salt = salt0 + g
                if mode == "same_slice":
                    out.append((fi, li, self.guide_idx))
                elif mode == "central_slice":
                    out.append((fi, self.n_slices[fi] // 2, self.guide_idx))
                elif mode == "far_slice":
                    far = [z for z in range(self.n_slices[fi]) if abs(z - li) >= self.min_slice_gap]
                    out.append((fi, far[self._pick(len(far), idx, salt)], self.guide_idx))
                elif mode == "other_study":
                    sib = self.siblings[fi]
                    gf = sib[self._pick(len(sib), idx, salt)]
                    n, gn = self.n_slices[fi], self.n_slices[gf]
                    if self.guide_slice == "matched":
                        gz = int(round(li / max(n - 1, 1) * (gn - 1)))
                    elif self.guide_slice == "central":
                        gz = gn // 2
                    else:
                        gz = self._pick(gn, idx, salt + 500)
                    out.append((gf, min(max(gz, 0), gn - 1), self.guide_idx))
                else:
                    raise AssertionError(mode)
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
        if self.modes:
            planes = []
            for gf, gz, gc in self._guide_planes(fi, li, idx):
                gimg = img if (gf == fi and gz == li) else self._read(gf, gz)[0]
                planes.append(gimg[..., gc])
            guide = chw(np.stack(planes, axis=-1))        # (G, H, W)

        # Joint geometric transform: stack -> transform -> split. The guide rides the SAME
        # crop/flip -- for same-session guides that preserves exact registration with the target,
        # which the guided prox depends on; for other_study it is harmless.
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

        if guide is None:
            return x0, x1, cond, mask
        if self.guide_as_cond:
            # guide planes go AFTER cond_idx, in guide order: cond = [cond_idx..., guides...]
            return x0, x1, torch.cat([cond, guide], dim=0), mask
        # (G, H, W) -> (G, 1, H, W), so a batch collates to (B, G, 1, H, W): the stacked layout
        # models/guided_prox.as_guide_list documents. A (B, G, H, W) batch is AMBIGUOUS there --
        # as_guide_list reads any 4-D tensor as ONE guide with G channels, which a C=1 analysis
        # conv then rejects (or, at G=1, silently accepts for the wrong reason).
        return {"x0": x0, "x1": x1, "cond": cond, "mask": mask, "guide": guide.unsqueeze(1)}
