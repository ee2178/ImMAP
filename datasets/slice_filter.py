"""
Restrict a slice-indexed h5 dataset to one UNIVERSAL range of original slice indices.

Both slice datasets (datasets/BraTS/i2sb_dataset.I2SBDataset and
datasets/NYUMets/longitudinal_dataset.NYUMetsGuidedDataset) index a flat list of slices as
(file_id[i], local[i]) into `img_paths`. `filter_by_slice_range` keeps the slices whose ORIGINAL
index -- the `slice_index` dataset preprocessing/nyumets_h5.py writes, i.e. the z of the canonical
RAS volume before the build dropped near-empty end slices -- lies in [lo, hi).

Why a fixed range rather than a per-slice brain-coverage threshold: a patient's studies are
registered to each other, so one original index is one anatomical level in EVERY study. A universal
range therefore keeps the same anatomy everywhere, and the slice of another study that matches a
target slice is simply the one with the same original index (NYUMetsGuidedDataset guide_slice
"index"). [40, 110) was chosen from the brain-coverage-vs-index plot over the NYUMets train split.

Cost: one small `slice_index` read per volume when the dataset is constructed.
"""

import h5py
import numpy as np


def slice_indices(path):
    """(n_kept,) original slice index of every stored slice of one h5 volume."""
    with h5py.File(path, "r") as f:
        if "slice_index" not in f:
            raise KeyError(f"slice_range needs a 'slice_index' dataset in {path} "
                           f"(written by preprocessing/nyumets_h5.py)")
        return np.asarray(f["slice_index"])


def filter_by_slice_range(img_paths, file_id, local, slice_range, tag=""):
    """-> (file_id, local) restricted to slices with lo <= original index < hi.

    `slice_range` None (or empty) keeps everything."""
    if slice_range is None or len(slice_range) == 0:
        return file_id, local
    if len(slice_range) != 2 or not slice_range[0] < slice_range[1]:
        raise ValueError(f"slice_range must be [lo, hi) with lo < hi, got {slice_range}")
    lo, hi = int(slice_range[0]), int(slice_range[1])
    zi = [slice_indices(p) for p in img_paths]
    z = np.fromiter((zi[f][l] for f, l in zip(file_id, local)), dtype=np.int64, count=len(file_id))
    keep = (z >= lo) & (z < hi)
    if not keep.any():
        raise RuntimeError(f"{tag + ': ' if tag else ''}slice_range [{lo}, {hi}) keeps no slice "
                           f"(original indices span {z.min()}..{z.max()})")
    kept_files = len(set(file_id[keep].tolist()))
    print(f"[{tag or 'dataset'}] slice_range [{lo}, {hi}): kept {int(keep.sum())}/{keep.size} slices "
          f"({100 * keep.mean():.0f}%) from {kept_files}/{len(img_paths)} volumes")
    return file_id[keep], local[keep]
