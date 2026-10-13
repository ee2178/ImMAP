# -*- coding: utf-8 -*-
"""Does `--reg-method fit` recover a known ROTATION, SCALE and shift -- what no other method can?

Same discipline as test_nyumets_search_register.py: the fixture is ANALYTIC. One phantom is
defined as a function of the world point, and each study samples it on its own grid -- here a
grid that is rotated, scaled and shifted about the head centre -- so nothing is resampled to
build it and the expectations come from the construction, never from the code under test.

A study built with world = P(index) holds at its index i what the reference holds at P(i). The
transform that takes a REFERENCE index to the study index showing the same anatomy is therefore
inv(P), and that is what `reg_matrix` must come out as (in (h, w, z) order).
"""
import math
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.modules.setdefault("nibabel", types.ModuleType("nibabel"))

from preprocessing.nyumets_h5 import (  # noqa: E402
    Volume, fit_affine, fit_matrix, masked_ncc, register_patient,
)

H = W = 64
D_REF, D_OTHER = 48, 46
C0 = np.array([24.0, 32.0, 32.0])                   # head centre, world (z, y, x)
_rng = np.random.default_rng(0)
_BLOBS = [(_rng.uniform(10, 38), _rng.uniform(16, 48), _rng.uniform(16, 48),
           _rng.uniform(1.5, 5), _rng.uniform(-1, 1, size=4)) for _ in range(150)]


def phantom(z, y, x):
    head = (((y - 32) / 26.0) ** 2 + ((x - 32) / 21.0) ** 2 + ((z - 24) / 20.0) ** 2) < 1
    ch = np.zeros((4,) + np.shape(z))
    for cz, cy, cx, s2, w in _BLOBS:
        g = np.exp(-((z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2) / (2 * s2))
        ch += w[:, None, None, None] * g
    # NOT multiplied by the head: the build stores unmasked intensities (mask_output=False), and
    # a cliff at the mask edge would make every correlation below a statement about that edge
    return ch + 1.0, head


def pose(yaw=0.0, pitch=0.0, scale=(1.0, 1.0, 1.0), shift=(0.0, 0.0, 0.0)):
    """4x4 P in (z, y, x): world = R S (index - C0) + C0 + shift. yaw turns the axial plane
    (about z); pitch tilts it (about x), so a slice of the study is not a slice of the world."""
    cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
    Rz = np.array([[1, 0, 0], [0, cy, -sy], [0, sy, cy]])            # mixes (y, x)
    Rx = np.array([[cp, -sp, 0], [sp, cp, 0], [0, 0, 1]])            # mixes (z, y)
    A = Rz @ Rx @ np.diag(scale)
    P = np.eye(4)
    P[:3, :3] = A
    P[:3, 3] = C0 - A @ C0 + np.asarray(shift, dtype=float)
    return P


def study(depth, P=None):
    z, y, x = np.mgrid[0:depth, 0:H, 0:W].astype(np.float64)
    if P is not None:
        pts = np.stack([z, y, x, np.ones_like(z)])
        z, y, x = np.tensordot(P[:3], pts, axes=1)
    ch, head = phantom(z, y, x)
    img = torch.from_numpy(np.ascontiguousarray(ch.transpose(1, 0, 2, 3)).astype(np.float32))
    fg = torch.from_numpy(head)
    return Volume(raw=img.clone() * 1000 + 500, norm=img.clone(), fg=fg.clone(), fov=fg.clone(),
                  stats=None, orig_depth=depth, affine=np.eye(4), lost={}, native_hw=(H, W))


def cfg(**over):
    c = SimpleNamespace(reg_method="fit", reg_contrast="T1", reg_device="cpu", reg_stride=2,
                        reg_max_dz=12, reg_dof=9, reg_levels=(2, 1), reg_iters=(150, 60),
                        reg_min_cover=0.9)
    for k, v in over.items():
        setattr(c, k, v)
    return c


def to_hwz(P):
    """A (z, y, x) 4x4 as the (h, w, z) one `reg_matrix` uses (h = y, w = x)."""
    perm = [1, 2, 0, 3]
    return P[np.ix_(perm, perm)]


def ncc_with_ref(vols):
    """Masked T1 correlation with the reference, index against index (an unregistered study is
    shallower, so it is zero-padded at the end first -- padding is outside its mask)."""
    b, mb = vols[1].norm[:, 1], vols[1].fg
    if b.shape[0] < D_REF:
        n = D_REF - b.shape[0]
        b = torch.cat([b, torch.zeros((n,) + b.shape[1:])])
        mb = torch.cat([mb, torch.zeros((n,) + mb.shape[1:], dtype=torch.bool)])
    return masked_ncc(vols[0].norm[:, 1], b, vols[0].fg, mb)


CASES = {
    "in-plane rotation": dict(yaw=6.0),
    "tilt": dict(pitch=5.0),
    "z scale": dict(scale=(1.08, 1.0, 1.0)),
    "everything": dict(yaw=-5.0, pitch=4.0, scale=(0.94, 1.04, 1.03), shift=(3.0, -4.0, 2.5)),
}


@pytest.mark.parametrize("name", list(CASES))
def test_fit_recovers_the_pose_and_aligns(name):
    P = pose(**CASES[name])
    before = ncc_with_ref([study(D_REF), study(D_OTHER, P)])
    vols = [study(D_REF), study(D_OTHER, P)]
    offsets, valid = register_patient(vols, cfg(), ref=0)
    reg = vols[1].reg
    assert reg["method"] == "fit" and reg["fit_ok"], reg
    after = ncc_with_ref(vols)
    assert after > 0.985 and after >= before, (name, before, after)
    if name != "z scale":                # an 8% z-scale barely moves a correlation; see dA below
        assert after > before + 0.03, (name, before, after)
    assert abs(reg["ncc"] - masked_ncc(vols[0].norm[:, 1], vols[1].norm[:, 1], vols[0].fg,
                                       vols[1].fg, stride=2)) < 1e-6

    # the transform itself: reference index -> study index is inv(P)
    want = to_hwz(np.linalg.inv(P))
    got = np.asarray(reg["matrix"], dtype=np.float64)
    assert np.abs(got[:3, :3] - want[:3, :3]).max() < 0.01, (name, got, want)
    # where the head centre lands, in voxels
    c = np.array([C0[1], C0[2], C0[0], 1.0])
    assert np.abs((got @ c) - (want @ c)).max() < 0.2, (name, got @ c, want @ c)

    # everything was resampled onto the REFERENCE grid, masks included
    for arr in (vols[1].raw, vols[1].norm):
        assert tuple(arr.shape) == (D_REF, 4, H, W)
    assert tuple(vols[1].fg.shape) == (D_REF, H, W) and vols[1].fg.dtype == torch.bool
    assert valid[0].all() and valid[1].sum() >= 10 and len(valid[1]) == D_REF


def test_a_translation_only_method_cannot_do_this():
    """The control: the same tilted, scaled study under `centroid` stays misaligned."""
    P = pose(**CASES["everything"])
    vols = [study(D_REF), study(D_OTHER, P)]
    register_patient(vols, cfg(reg_method="centroid"), ref=0)
    left = ncc_with_ref(vols)
    vols = [study(D_REF), study(D_OTHER, P)]
    register_patient(vols, cfg(), ref=0)
    assert left < 0.9 and ncc_with_ref(vols) > left + 0.08, left


def test_rigid_dof_cannot_absorb_a_scale_but_nine_can():
    P = pose(scale=(1.1, 1.0, 1.0))
    out = {}
    for dof in (6, 9):
        vols = [study(D_REF), study(D_OTHER, P)]
        register_patient(vols, cfg(reg_dof=dof), ref=0)
        M = np.asarray(vols[1].reg["matrix"], dtype=np.float64)
        out[dof] = abs(M[2, 2] - 1 / 1.1)                # the z-scale entry, (h, w, z) order
    assert out[9] < 0.01 and out[6] > 0.05, out


def test_an_already_aligned_study_is_left_essentially_alone():
    vols = [study(D_REF), study(D_OTHER)]
    register_patient(vols, cfg(), ref=0)
    M = np.asarray(vols[1].reg["matrix"], dtype=np.float64)
    assert np.abs(M[:3, :3] - np.eye(3)).max() < 0.005 and np.abs(M[:3, 3]).max() < 0.2
    assert ncc_with_ref(vols) > 0.995


def test_a_fit_that_does_not_help_is_rejected():
    """With no optimiser steps the fit cannot beat its start, so the study must come out on
    the centroid translation alone, flagged."""
    P = pose(shift=(4.0, 0.0, 0.0))
    vols = [study(D_REF), study(D_OTHER, P)]
    register_patient(vols, cfg(reg_iters=(0, 0)), ref=0)
    reg = vols[1].reg
    assert reg["fit_ok"] is False
    M = np.asarray(reg["matrix"])
    assert np.allclose(M[:3, :3], np.eye(3)) and np.allclose(M[:3, 3], np.round(M[:3, 3]))
    assert ncc_with_ref(vols) > 0.97                     # a pure shift: the translation suffices


def test_fit_matrix_rotates_about_the_centre():
    par = {"rot": torch.tensor([0.0, 0.0, 0.3]), "logs": torch.zeros(3),
           "shift": torch.zeros(3), "free": torch.zeros(3, 3)}
    c = torch.tensor([10.0, 20.0, 5.0])
    M = fit_matrix(par, c, torch.zeros(3), 9)
    assert torch.allclose(M[:3, :3] @ c + M[:3, 3], c, atol=1e-5)       # the centre stays put
    assert torch.allclose(M[:3, :3] @ M[:3, :3].T, torch.eye(3), atol=1e-5)


def test_fit_affine_reports_its_own_improvement():
    P = pose(yaw=5.0)
    a, b = study(D_REF), study(D_OTHER, P)
    M, info = fit_affine(a.norm[:, 1], a.fg, b.norm[:, 1], b.fg, (0, 0, 0), dof=6,
                         levels=(2,), iters=(120,))
    # overlap counts voxels WHOLLY inside both masks, so the one-voxel shell of a small pooled
    # head is excluded and it sits well below 1 even when the brains coincide
    assert info["ncc"] > info["ncc_init"] + 0.02 and info["overlap"] > 0.6
