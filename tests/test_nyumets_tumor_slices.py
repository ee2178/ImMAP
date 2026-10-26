# -*- coding: utf-8 -*-
"""scripts/nyumets_tumor_slices.py: a segmentation on its OWN grid lands on the lesion in h5 space.

A native study (canonical index space, a real-looking affine) has a bright lesion on CT1; its
segmentation is written on a coarser grid with a different affine. The h5 study is made the way the
build makes it -- centre crop / pad to `crop`, then a fit-style resample by a rotation + scale +
shift `reg_matrix` -- and the mapped segmentation must cover the bright lesion in the stored CT1.
"""
import math
import os
import sys
import types

import h5py
import numpy as np
import pytest

sys.modules.setdefault("nibabel", types.ModuleType("nibabel"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import nyumets_tumor_slices as T  # noqa: E402
from datasets.NYUMets.inference_registration import warp  # noqa: E402

Hn, Wn, D, CROP = 44, 52, 36, 48          # native in-plane 44 x 52: cropped on W, padded on H
CENTRE = np.array([20.0, 30.0, 18.0])     # lesion centre, native (h, w, z)
R = 4.0


def native_affine():
    A = np.diag([0.9, 0.9, 1.2, 1.0])
    A[:3, 3] = [-20.0, -25.0, -30.0]
    return A


def native_ct1():
    h, w, z = np.meshgrid(np.arange(Hn), np.arange(Wn), np.arange(D), indexing="ij")
    head = (((h - 22) / 19) ** 2 + ((w - 26) / 22) ** 2 + ((z - 18) / 16) ** 2) < 1
    les = ((h - CENTRE[0]) ** 2 + (w - CENTRE[1]) ** 2 + (z - CENTRE[2]) ** 2) < R * R
    return (head * 100.0 + les * 300.0).astype(np.float32), les


def seg_on_its_own_grid():
    """The lesion, rasterised on a 2x coarser grid whose affine differs from the images'."""
    A_seg = np.diag([1.8, 1.8, 2.4, 1.0])
    A_seg[:3, 3] = [-21.0, -24.0, -31.0]
    world_c = native_affine() @ np.r_[CENTRE, 1.0]
    i, j, k = np.meshgrid(np.arange(30), np.arange(30), np.arange(24), indexing="ij")
    P = A_seg @ np.stack([i.ravel(), j.ravel(), k.ravel(), np.ones(i.size)])
    d = np.linalg.norm((P[:3].T - world_c[:3]) / np.array([0.9, 0.9, 1.2]), axis=1)
    return (d < R).reshape(30, 30, 24).astype(np.uint8) * 7, A_seg      # label value 7


def center_crop_or_pad(a, size):
    """(H, W, ...) -> (size, size, ...), the build's rule: offset (size - n) // 2 per axis."""
    out = np.zeros((size, size) + a.shape[2:], a.dtype)
    for ax in (0, 1):
        pass
    oh, ow = (size - a.shape[0]) // 2, (size - a.shape[1]) // 2
    sh, sw = max(-oh, 0), max(-ow, 0)
    dh, dw = max(oh, 0), max(ow, 0)
    nh, nw = min(a.shape[0] - sh, size - dh), min(a.shape[1] - sw, size - dw)
    out[dh:dh + nh, dw:dw + nw] = a[sh:sh + nh, sw:sw + nw]
    return out


def reg_matrix():
    t = math.radians(6.0)
    A = np.array([[math.cos(t), -math.sin(t), 0], [math.sin(t), math.cos(t), 0], [0, 0, 1]]) * 1.04
    c = np.array([24.0, 24.0, 18.0])
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = A, c - A @ c + np.array([1.5, -2.0, 1.0])
    return M


@pytest.fixture
def study(tmp_path, monkeypatch):
    ct1, _ = native_ct1()
    cropped = center_crop_or_pad(ct1, CROP)                          # (CROP, CROP, D) (h, w, z)
    vol = np.transpose(cropped, (2, 0, 1))[..., None]                # (D, H, W, 1)
    M = reg_matrix()
    out = warp(vol, M, (D, CROP, CROP))[..., 0]                      # the h5 grid
    keep = np.arange(8, 30)                                          # slice filtering
    img = np.zeros((len(keep), CROP, CROP, 4), np.float32)
    img[..., 2] = out[keep]
    img[..., 1] = (out[keep] > 50) * 100.0
    case = "P1_S1"
    d = tmp_path / "split" / case
    os.makedirs(d)
    with h5py.File(d / (case + "_img.h5"), "w") as f:
        f.create_dataset("img_raw", data=img)
        f.create_dataset("mask", data=(img[..., 1:2] > 0).astype(np.uint8))
        f.create_dataset("slice_index", data=keep.astype(np.int32))
        f.attrs.update(patient="P1", session="S1", reg_method="fit", reg_matrix=M.astype(np.float32),
                       crop_size=CROP, native_size_source=np.int32([Hn, Wn]),
                       affine_source=native_affine().astype(np.float32))
    seg, A_seg = seg_on_its_own_grid()
    sp = tmp_path / "nifti" / "P1" / "studyId" / "S1"
    os.makedirs(sp)
    open(sp / T.SEG_NAME, "w").close()                               # exists; content mocked
    monkeypatch.setattr(T, "load_seg", lambda path: (seg, A_seg))
    return str(d / (case + "_img.h5")), str(tmp_path / "nifti")


def test_the_segmentation_lands_on_the_bright_lesion(study):
    h5path, nroot = study
    seen = {}
    L = T.study_lesions(h5path, nroot, seen=seen)
    bright = L["ct1"] > 250                                          # the lesion, as stored
    les = L["lesion"]
    iou = (les & bright).sum() / max((les | bright).sum(), 1)
    g = T.geometry_check(L)
    assert seen == {0: 1, 7: 1}
    assert bright.sum() > 50 and iou > 0.6, (int(bright.sum()), int(les.sum()), iou)
    assert g["ct1_ratio"] > 2.0 and g["in_brain"] > 0.95, g


def test_a_wrong_label_finds_nothing_and_no_file_is_unknown(study, tmp_path):
    h5path, nroot = study
    assert not T.study_lesions(h5path, nroot, labels=[3])["lesion"].any()
    assert T.study_lesions(h5path, str(tmp_path / "elsewhere")) is None


def test_orient_matches_the_dump(study):
    from visualization.image import orient_image
    a = np.random.default_rng(0).random((3, 5, 7))
    for mode in ("radiological", "neurological", None):
        want = np.stack([orient_image(x, mode) for x in a])
        assert np.array_equal(T.orient(a, mode), want)
