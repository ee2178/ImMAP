# -*- coding: utf-8 -*-
"""Guide contrast lists and slice windows in NYUMetsGuidedDataset (guide_idx=[...], guide_window).

Every stored pixel ENCODES where it came from: value = 1000 * study + 10 * slice + contrast. A guide
plane can then be decoded back to (study, slice, contrast) and checked against the layout the
docstring promises, without trusting the code that built it.
"""
import os
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from datasets.NYUMets.longitudinal_dataset import NYUMetsGuidedDataset

N_SLICES, HW = 12, 8


def _write(root, patient, study, sid):
    case = f"{patient}_{study}"
    d = os.path.join(root, case)
    os.makedirs(d)
    z = np.arange(N_SLICES, dtype=np.float32)[:, None, None, None]
    c = np.arange(4, dtype=np.float32)[None, None, None, :]
    img = np.broadcast_to(1000 * sid + 10 * z + c, (N_SLICES, HW, HW, 4)).astype(np.float32)
    with h5py.File(os.path.join(d, f"{case}_img.h5"), "w") as f:
        f["img_raw"] = img
        f["img"] = h5py.SoftLink("/img_raw")
        f["mask"] = np.ones((N_SLICES, HW, HW, 1), np.uint8)
        f["slice_index"] = np.arange(N_SLICES, dtype=np.int32)
        f.attrs["patient"] = patient


@pytest.fixture()
def root(tmp_path):
    _write(str(tmp_path), "P", "a", 1)
    _write(str(tmp_path), "P", "b", 2)
    return str(tmp_path)


def ds(root, **over):
    cfg = dict(root=root, image_key="img_raw", x0_idx=2, x1_idx=1, cond_idx=[0, 1, 3],
               guide_mode="other_study", guide_idx=2, guide_slice="index", guide_as_cond=True,
               deterministic=True)
    cfg.update(over)
    return NYUMetsGuidedDataset(SimpleNamespace(**cfg))


def decode(plane):
    v = int(round(float(plane[0, 0])))
    return v // 1000, (v % 1000) // 10, v % 10          # study, slice, contrast


def item_for(d, study_id, z):
    """Index of the sample (study, local slice z)."""
    for i in range(len(d)):
        li = int(d.local[i])
        if li == z and decode(d[i][0][0])[0] == study_id:  # x0 (1, H, W) carries the study id
            return i
    raise AssertionError("sample not found")


def test_single_ct1_is_unchanged(root):
    d = ds(root)
    assert d.n_guide_planes == 1
    _, _, cond, _ = d[item_for(d, 1, 5)]
    assert cond.shape[0] == 3 + 1
    assert decode(cond[3]) == (2, 5, 2)                  # other study, same slice, CT1


def test_pair_is_t1_then_ct1_of_the_other_study(root):
    d = ds(root, guide_idx=[1, 2])
    assert d.n_guide_planes == 2
    _, _, cond, _ = d[item_for(d, 1, 5)]
    assert cond.shape[0] == 3 + 2
    assert [decode(p) for p in cond[3:]] == [(2, 5, 1), (2, 5, 2)]


@pytest.mark.parametrize("k", [1, 2])
def test_window_is_offset_major_contrast_minor_from_one_study(root, k):
    d = ds(root, guide_idx=[1, 2], guide_window=k)
    assert d.n_guide_planes == (2 * k + 1) * 2
    _, _, cond, _ = d[item_for(d, 1, 5)]
    assert cond.shape[0] == 3 + (2 * k + 1) * 2
    want = [(2, 5 + o, c) for o in range(-k, k + 1) for c in (1, 2)]
    assert [decode(p) for p in cond[3:]] == want


def test_window_repeats_the_edge_slice(root):
    d = ds(root, guide_idx=[1, 2], guide_window=2)
    _, _, cond, _ = d[item_for(d, 1, 0)]
    got = [decode(p)[1] for p in cond[3:]]
    assert got == [0, 0, 0, 0, 0, 0, 1, 1, 2, 2]        # z-2, z-1 clamp to 0


def test_cond_channels_come_first(root):
    d = ds(root, guide_idx=[1, 2], guide_window=1)
    _, _, cond, _ = d[item_for(d, 1, 5)]
    assert [decode(p) for p in cond[:3]] == [(1, 5, 0), (1, 5, 1), (1, 5, 3)]


def test_window_needs_other_study(root):
    with pytest.raises(ValueError, match="guide_window"):
        ds(root, guide_mode="far_slice", guide_window=1)
