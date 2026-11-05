"""
Paired fastMRI knee data for GUIDED reconstruction: PDFS main contrast, PD side information.

The pairs are TGVN's patient-level splits (config/knee/guided/splits/{train,val,test}.csv,
copied from sodicksonlab/TGVN data_splits/fastmri; columns `pd,pdfs`). They draw from fastMRI's
multicoil train AND val sets combined, so every file is looked up in each of `kspace_roots`.
Slices are paired BY INDEX within a pair, as TGVN does; pairs whose volumes disagree on slice
count, matrix size or coil count are dropped at construction (and counted), since a guide off
by a slice or on another grid is not side information.

The TARGET (`target`):
  "sense"  (decided 2026-11-05) the OFFLINE SENSE combination of the fully-sampled MAIN k-space,
           `image` = sum_c conj(s_c) c_c, read from `smap_root` -- one h5 per PDFS volume with
           `smaps` (S, C, H, W) and `image` (S, H, W), as scripts/make_espirit_smaps.py writes
           them (torch/espirit_knee_pairs.sbatch: ESPIRiT, ACS 32, from the full k-space). The
           image is scaled by `scale_fac` like the k-space; `smaps` are those maps.
  "rss"    the RSS of the fully-sampled main k-space (TGVN's own target, `reconstruction_rss`),
           complex with zero phase; `smaps` are then a flat unit-RSS PLACEHOLDER.
Either way the offline maps define only the GROUND TRUTH: under the measured protocol the
operator's maps come from the measurement's own ACS (`mri.online_smaps`), and training/recon.py
refuses paired data with simulated k-space or with an operator-map model and no online maps.
  guide    the FULLY-SAMPLED side (PD) k-space, scaled like the main. The recon loop turns it
           into whatever the model consumes (models with guide_domain "kspace", e.g. TGVN, take
           it as is; image-guided models get its coil combination under the operator's maps).

Each item is the recon loader's 5-tuple plus the side k-space:

    (kspace, smaps, image, organ_mask, pad_hw, side_kspace)

Sampling follows FastMRIDataset: one item per PAIR with the slice drawn in [start_slice,
end_slice) at __getitem__, or every slice with `enumerate_slices`.
"""

import csv
import os
import random

import h5py
import torch
from torch.utils.data import Dataset, DataLoader

from datasets.registry import register_loader
from operators.fourier import ifftc
from operators.truncate import embedded_size
from physics.object_mask import rss_object_mask

MAIN_ACQ, SIDE_ACQ = "CORPDFS_FBK", "CORPD_FBK"


def read_pairs(csv_path):
    """[(pd_fname, pdfs_fname)] from a TGVN split CSV (header `pd,pdfs`)."""
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or set(rows[0]) != {"pd", "pdfs"}:
        raise ValueError(f"{csv_path}: expected columns 'pd,pdfs'")
    return [(os.path.basename(r["pd"]), os.path.basename(r["pdfs"])) for r in rows]


class PairedKneeDataset(Dataset):
    def __init__(self, split_csv, kspace_roots, scale_fac=5e3, start_slice=0, end_slice=None,
                 enumerate_slices=False, pad_multiple=1, organ_mask_source="none",
                 organ_mask_kws=None, check_acquisition=True, target="rss", smap_root=None):
        if target not in ("rss", "sense"):
            raise ValueError(f"target must be 'rss' or 'sense', got {target!r}")
        if target == "sense" and not smap_root:
            raise ValueError("target='sense' needs smap_root: the offline maps + SENSE images "
                             "(torch/espirit_knee_pairs.sbatch writes them)")
        if organ_mask_source not in ("none", "rss", "smaps"):
            raise ValueError(f"organ_mask_source must be 'none', 'rss' or 'smaps', got "
                             f"{organ_mask_source!r}")
        if organ_mask_source == "smaps" and target != "sense":
            raise ValueError("organ_mask_source='smaps' needs the offline maps (target='sense')")
        self.target, self.smap_root = target, smap_root
        self.roots = [kspace_roots] if isinstance(kspace_roots, str) else list(kspace_roots)
        self.scale_fac = float(scale_fac)
        self.start_slice, self.end_slice = int(start_slice), end_slice
        self.pad_multiple = int(pad_multiple)
        self.organ_mask_source = organ_mask_source
        self.organ_mask_kws = dict(organ_mask_kws or {})

        self.pairs, dropped, missing = [], [], []
        for pd_name, pdfs_name in read_pairs(split_csv):
            pd_path, pdfs_path = self._find(pd_name), self._find(pdfs_name)
            with h5py.File(pd_path, "r") as a, h5py.File(pdfs_path, "r") as b:
                ka, kb = a["kspace"].shape, b["kspace"].shape
                if check_acquisition:
                    acq = (a.attrs.get("acquisition", ""), b.attrs.get("acquisition", ""))
                    if acq != (SIDE_ACQ, MAIN_ACQ):
                        raise ValueError(f"{pd_name}/{pdfs_name}: acquisitions {acq}, expected "
                                         f"({SIDE_ACQ!r}, {MAIN_ACQ!r}) -- pd column = side")
            if ka != kb:
                dropped.append((pd_name, pdfs_name, ka, kb))
                continue
            map_path = None
            if target == "sense":
                map_path = os.path.join(smap_root, pdfs_name)
                if not os.path.exists(map_path):
                    missing.append(pdfs_name)
                    continue
                with h5py.File(map_path, "r") as m:
                    if m["smaps"].shape != kb or m["image"].shape != (kb[0],) + kb[2:]:
                        raise ValueError(f"{map_path}: smaps {m['smaps'].shape} / image "
                                         f"{m['image'].shape} do not match the k-space {kb}")
            self.pairs.append((pd_path, pdfs_path, int(ka[0]), map_path))
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} PDFS volume(s) of {os.path.basename(split_csv)} have no offline "
                f"maps in {smap_root}, e.g. {missing[:3]}: run torch/espirit_knee_pairs.sbatch")
        self.dropped = dropped
        if dropped:
            print(f"[fastmri_paired] {os.path.basename(split_csv)}: dropped {len(dropped)} of "
                  f"{len(dropped) + len(self.pairs)} pairs whose PD/PDFS k-space shapes differ "
                  f"(first: {dropped[0]})")
        if not self.pairs:
            raise ValueError(f"no usable pairs in {split_csv}")

        self.index = [(i, None) for i in range(len(self.pairs))]
        if enumerate_slices:
            self.index = []
            for i, (_, _, n, _) in enumerate(self.pairs):
                hi = n if self.end_slice is None else min(int(self.end_slice), n)
                self.index += [(i, s) for s in range(min(self.start_slice, hi), hi)]

    def _find(self, name):
        for r in self.roots:
            p = os.path.join(r, name)
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"{name} is in none of {self.roots}")

    def __len__(self):
        return len(self.index)

    def item_id(self, idx):
        i, sl = self.index[idx]
        if sl is None:
            raise RuntimeError("item_id() needs enumerate_slices=True")
        return os.path.splitext(os.path.basename(self.pairs[i][1]))[0], int(sl)

    def _slice(self, n):
        lo = min(self.start_slice, n - 1)
        hi = n if self.end_slice is None else min(int(self.end_slice), n)
        return lo if hi <= lo else random.randint(lo, hi - 1)

    def __getitem__(self, idx):
        i, sl = self.index[idx]
        pd_path, pdfs_path, n, map_path = self.pairs[i]
        sl = self._slice(n) if sl is None else sl
        with h5py.File(pdfs_path, "r") as f:
            kspace = torch.from_numpy(f["kspace"][sl]) * self.scale_fac
        with h5py.File(pd_path, "r") as f:
            side = torch.from_numpy(f["kspace"][sl]) * self.scale_fac

        C, H, W = kspace.shape
        rss = ifftc(kspace).abs().pow(2).sum(dim=0, keepdim=True).sqrt()      # (1, H, W)
        if self.target == "sense":
            with h5py.File(map_path, "r") as f:
                smaps = torch.from_numpy(f["smaps"][sl]).to(torch.complex64)
                image = (torch.from_numpy(f["image"][sl])[None] * self.scale_fac
                         ).to(torch.complex64)
        else:
            image = rss.to(torch.complex64)
            smaps = torch.full((C, H, W), C ** -0.5, dtype=torch.complex64)   # placeholder
        if self.organ_mask_source == "none":
            organ = torch.ones((1, H, W), dtype=torch.bool)
        elif self.organ_mask_source == "smaps":
            organ = smaps.abs().sum(dim=0, keepdim=True) > 0                  # map support
        else:
            organ = rss_object_mask(rss, **self.organ_mask_kws)
        pad_hw = torch.tensor(embedded_size((H, W), self.pad_multiple), dtype=torch.long)
        return kspace, smaps, image, organ, pad_hw, side


@register_loader("fastmri_paired")
def get_paired_knee_loader(split_csv, kspace_roots, batch_size=1, shuffle=True, drop_last=True,
                           scale_fac=5e3, start_slice=0, end_slice=None, enumerate_slices=False,
                           pad_multiple=1, organ_mask_source="none", organ_mask_kws=None,
                           check_acquisition=True, num_workers=8, task="recon", target="rss",
                           smap_root=None):
    """`task` is accepted (the recon configs carry it) and must be "recon"."""
    if task != "recon":
        raise ValueError(f"fastmri_paired serves recon batches only, got task={task!r}")
    ds = PairedKneeDataset(split_csv, kspace_roots, scale_fac=scale_fac,
                           start_slice=start_slice, end_slice=end_slice,
                           enumerate_slices=enumerate_slices, pad_multiple=pad_multiple,
                           organ_mask_source=organ_mask_source, organ_mask_kws=organ_mask_kws,
                           check_acquisition=check_acquisition, target=target,
                           smap_root=smap_root)
    num_workers = int(num_workers)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                      num_workers=num_workers, pin_memory=True,
                      persistent_workers=num_workers > 0,
                      **({"prefetch_factor": 4} if num_workers > 0 else {}))
