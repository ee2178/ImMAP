"""
Drop mostly-background slices from a slice-indexed h5 dataset.

Both slice datasets (datasets/BraTS/i2sb_dataset.I2SBDataset and
datasets/NYUMets/longitudinal_dataset.NYUMetsGuidedDataset) index a flat list of slices as
(file_id[i], local[i]) into `img_paths`. `filter_by_brain_frac` shrinks that index to the slices
whose stored brain `mask` covers at least `min_brain_frac` of the frame.

This is the SAME quantity preprocessing/nyumets_h5.py thresholds at build time
(`--min-brain-frac`, default 0.02, applied to the same foreground it saves as `mask`), so raising
it here is equivalent to having built with a higher value -- without rebuilding the h5 files.

Cost: every volume's mask is read once, in the main process, when the dataset is constructed.
"""

import h5py
import numpy as np


def slice_brain_fracs(path, mask_key="mask"):
    """(n_slices,) fraction of the frame covered by the mask, for one h5 volume."""
    with h5py.File(path, "r") as f:
        m = np.asarray(f[mask_key])                   # (N, H, W, 1) uint8
    m = m.reshape(m.shape[0], -1)
    return (m > 0).mean(axis=1)


def filter_by_brain_frac(img_paths, file_id, local, min_brain_frac, mask_key="mask", tag=""):
    """-> (file_id, local) restricted to slices with brain frac >= min_brain_frac."""
    if not min_brain_frac or min_brain_frac <= 0:
        return file_id, local
    fracs = [slice_brain_fracs(p, mask_key) for p in img_paths]
    per_slice = np.fromiter((fracs[f][z] for f, z in zip(file_id, local)), dtype=np.float64,
                            count=len(file_id))
    keep = per_slice >= float(min_brain_frac)
    if not keep.any():
        raise RuntimeError(f"{tag}min_brain_frac={min_brain_frac} drops every slice "
                           f"(max brain frac {per_slice.max():.3f})")
    q = np.percentile(per_slice, [10, 50, 90])
    print(f"[{tag or 'dataset'}] min_brain_frac={min_brain_frac:g}: kept {int(keep.sum())}/{keep.size} "
          f"slices ({100 * keep.mean():.0f}%); brain frac p10/p50/p90 = {q[0]:.2f}/{q[1]:.2f}/{q[2]:.2f}")
    return file_id[keep], local[keep]
