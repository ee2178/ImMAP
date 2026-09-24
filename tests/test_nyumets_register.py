# -*- coding: utf-8 -*-
"""Does affine registration actually undo a known rigid transform, rotation included?

The expectations here are written from the CONSTRUCTION, never from the implementation. That
distinction has already caught two sign errors in this module: when a test's numbers come out of
the code under test, a wrong sign agrees with itself.

THE FIXTURE IS ANALYTIC, and that is load-bearing. The obvious way to build a second "session"
is to resample the first one by a known transform -- but that zero-pads the corners and applies
an interpolation, so the two sessions differ by the fixture's own losses on top of the geometry,
and no registration can recover data the fixture destroyed. Measured: such a fixture leaves a
residual floor of ~7% of the signal on a 6.5 degree tilt, and the registration then hits that
floor EXACTLY -- a correct result that reads as a failure against any absolute threshold.

So instead there is one phantom defined as a smooth function of the WORLD point, and each
session samples it on its own grid through its own affine. Nothing is interpolated to build the
fixture, both sessions see the same anatomy, and the only error left afterwards is the
registration's own, which is what these tests are for.
"""
import math
import sys
import types

import numpy as np
import pytest
import torch

sys.modules.setdefault("nibabel", types.ModuleType("nibabel"))

from preprocessing.nyumets_h5 import (          # noqa: E402
    Volume, affine_matrix, register_patient, stored_to_world,
)

CROP, D, CI = 64, 32, 1          # T1 is stored channel 1
CTR = ((CROP - 1) / 2.0, (CROP - 1) / 2.0, (D - 1) / 2.0)     # world mm, since A_ref = I


def phantom(ph, pw, pz):
    """Intensity at WORLD (h, w, z) in mm. SMOOTH everywhere.

    Smooth on purpose: a hard-edged phantom is not band-limited, so two grids sampling it at
    different sub-voxel offsets disagree at every boundary and that disagreement would be
    mistaken for misalignment. Softened shells plus a position-locked texture give plenty of
    rotation-sensitive structure with none of that.
    """
    h, w, z = ph - CTR[0], pw - CTR[1], pz - CTR[2]
    r = ((h / 26.0) ** 2 + (w / 22.0) ** 2 + (z / 13.0) ** 2).sqrt()
    head = 0.5 * (1.0 - torch.tanh((r - 1.0) * 5.0))          # soft outer boundary
    shell = 0.9 * torch.exp(-(((r - 0.33) / 0.10) ** 2))      # soft bright ring
    # Texture as a function of the WORLD point, so both sessions see the identical field.
    # Periods of ~8-10 mm: fine enough that a few degrees of rotation visibly changes the image
    # (a 30 mm period barely does -- measured, it left a 7 degree in-plane rotation looking only
    # 4% different), coarse enough to stay well sampled on a 1 mm grid.
    tex = (torch.sin(h * 0.68) * torch.cos(w * 0.61)
           + 0.6 * torch.sin(z * 0.55 + h * 0.37)
           + 0.4 * torch.cos(h * 0.44 - w * 0.72))
    # a smooth blob well off the rotation axis, which any rotation moves a long way
    blob = 1.1 * torch.exp(-(((h - 13.0) / 6.0) ** 2 + ((w + 10.0) / 6.0) ** 2
                             + ((z - 3.0) / 5.0) ** 2))
    return ((head * (1.0 + 0.35 * tex + blob) + shell * head).clamp_min(0.0) * 1200.0)


def sample_on_grid(affine, dhw=(D, CROP, CROP)):
    """(D,H,W) of the phantom as session with `affine` would record it. No interpolation."""
    nz, nh, nw = dhw
    a = torch.as_tensor(np.asarray(affine), dtype=torch.float32)
    z = torch.arange(nz).float().view(-1, 1, 1)
    h = torch.arange(nh).float().view(1, -1, 1)
    w = torch.arange(nw).float().view(1, 1, -1)
    # stored index order is (h, w, z) -- see stored_to_world
    # each term broadcasts to (nz, nh, nw), so all three come out full-shape
    ph, pw, pz = (a[r, 0] * h + a[r, 1] * w + a[r, 2] * z + a[r, 3] for r in range(3))
    return phantom(ph, pw, pz)


def rigid(deg_about_z, deg_about_w, shift_hwz):
    """A 4x4 rigid transform in (h, w, z) index order, about the volume centre."""
    a, b = math.radians(deg_about_z), math.radians(deg_about_w)
    rz = np.array([[math.cos(a), -math.sin(a), 0.0],
                   [math.sin(a), math.cos(a), 0.0],
                   [0.0, 0.0, 1.0]])                      # in-plane: mixes h and w
    rw = np.array([[math.cos(b), 0.0, -math.sin(b)],
                   [0.0, 1.0, 0.0],
                   [math.sin(b), 0.0, math.cos(b)]])      # tilt: mixes h and z
    rot = rz @ rw
    c = np.asarray(CTR, dtype=float)
    t = np.eye(4)
    t[:3, :3] = rot
    t[:3, 3] = c - rot @ c + np.asarray(shift_hwz, dtype=float)
    return t


def make_vol(base, affine, native_hw=(CROP, CROP)):
    raw = base.unsqueeze(1).repeat(1, 4, 1, 1)
    fg = base > 600.0
    norm = torch.empty_like(raw)
    stats = np.zeros((4, 2), dtype=np.float32)
    for c in range(4):
        b = raw[:, c][fg]
        med = float(b.median())
        mad = max(float((b - med).abs().median()) * 1.4826, 1e-6)
        norm[:, c] = (raw[:, c] - med) / mad     # UNMASKED, so background is -med/mad, not 0
        stats[c] = (med, mad)
    return Volume(raw=raw, norm=norm, fg=fg, fov=fg.clone(), stats=stats,
                  orig_depth=base.shape[0], affine=np.asarray(affine, dtype=np.float32),
                  lost={}, native_hw=native_hw)


class Cfg(object):
    crop = CROP
    reg_device = "cpu"
    reg_method = "affine"
    reg_min_cover = 0.99
    reg_contrast = "T1"
    reg_lowpass = 0.25
    reg_source = "norm"
    reg_phase = True
    reg_max_shift = 0


def residual_in_brain(v_ref, v_oth, keep):
    """Mean |ref - other| where the reference has brain, over the slices `keep` selects.

    SCORED ON `raw`, NOT `norm`. These tests are about geometry, and median/MAD is not invariant
    to resampling -- smoothing drops the MAD, which shows up in `norm` as a global intensity
    mismatch no registration should be expected to fix. Measuring it here would grade the
    normaliser instead of the alignment.
    """
    k = torch.from_numpy(np.asarray(keep))
    roi = v_ref.fg[k]
    d = (v_ref.raw[k][:, CI] - v_oth.raw[k][:, CI]).abs()
    return float(d[roi].mean()), float(v_ref.raw[k][:, CI].abs()[roi].mean()), int(k.sum())


def pair(t):
    """The reference session and a second session related to it by `t`.

    A_ref = I, so a ref index IS a world point. Session 2's index q must name the same world
    point as ref index t @ q, so A_other = A_ref @ t = t.
    """
    return make_vol(sample_on_grid(np.eye(4)), np.eye(4)), make_vol(sample_on_grid(t), t)


# (name, deg about z, deg about w, (dh, dw, dz) shift in voxels)
CASES = [
    ("pure translation", 0.0, 0.0, (5.0, -4.0, 3.0)),
    ("in-plane rotation", 7.0, 0.0, (0.0, 0.0, 0.0)),
    ("tilt", 0.0, 6.5, (0.0, 0.0, 0.0)),
    ("rotation and translation", 6.0, 5.0, (4.0, -6.0, 2.0)),
]


@pytest.mark.parametrize("name,rz,rw,shift", CASES,
                         ids=[c[0].replace(" ", "_") for c in CASES])
def test_affine_registration_recovers_the_transform(name, rz, rw, shift):
    t = rigid(rz, rw, shift)
    v_ref, v_oth = pair(t)
    v_pre = make_vol(sample_on_grid(t), t)       # unregistered copy, to score "before" against

    # The sampling matrix register_patient should derive: output index p on the reference grid
    # reads session 2 at inv(A_other) @ A_ref @ p = inv(t) @ p.
    m, oblique = affine_matrix((v_ref.affine, v_ref.native_hw, D),
                               (v_oth.affine, v_oth.native_hw, D), CROP)
    assert np.allclose(m, np.linalg.inv(t), atol=1e-4), f"{name}: header transform is wrong"

    offsets, valid = register_patient([v_ref, v_oth], Cfg(), ref=0)
    keep = valid[0] & valid[1]
    before, _, _ = residual_in_brain(v_ref, v_pre, keep)
    after, scale, n_keep = residual_in_brain(v_ref, v_oth, keep)

    assert n_keep >= D // 3, f"{name}: only {n_keep}/{D} slices survived"
    assert before > 0.05 * scale, (
        f"{name}: the fixture is barely misaligned to begin with ({before:.1f} vs scale "
        f"{scale:.1f}); the case proves nothing")
    # All that is left after a correct transform is trilinear interpolation of a smooth field.
    assert after < 0.02 * scale, (
        f"{name}: residual {after:.1f} is {after / scale:.1%} of the reference's own scale "
        f"(was {before:.1f} before) -- more than interpolation can account for")
    assert after < 0.15 * before, (
        f"{name}: residual {after:.1f} vs {before:.1f} before -- registration did not help")

    # the record written into the file has to describe what actually happened
    assert v_oth.reg["method"] == "affine"
    assert np.allclose(v_oth.reg["matrix"], m, atol=1e-4)
    assert v_oth.reg["oblique"] == pytest.approx(oblique)
    assert np.allclose(v_oth.reg["grid_affine"], v_ref.affine, atol=1e-6)
    assert v_oth.reg["grid_native_hw"] == tuple(v_ref.native_hw)
    assert offsets[0] == (0, 0, 0)


def test_oblique_is_zero_for_a_pure_translation_and_tracks_the_angle():
    """`oblique` is the number the deferral decision rested on; pin what it means."""
    _, ob = affine_matrix((np.eye(4), (CROP, CROP), D),
                          (rigid(0.0, 0.0, (5.0, -4.0, 3.0)), (CROP, CROP), D), CROP)
    assert ob == pytest.approx(0.0, abs=1e-6), "a translation is not oblique"
    for deg in (2.0, 6.5, 13.0):
        _, ob = affine_matrix((np.eye(4), (CROP, CROP), D),
                              (rigid(0.0, deg, (0.0, 0.0, 0.0)), (CROP, CROP), D), CROP)
        assert ob == pytest.approx(math.sin(math.radians(deg)), abs=1e-4), (
            f"oblique should be sin(angle); {deg} deg gave {ob}")


def test_affine_beats_xcorr_on_a_pure_tilt():
    """The whole point of the change: a translation cannot undo a rotation."""
    t = rigid(0.0, 6.5, (0.0, 0.0, 0.0))
    runs = {}
    for method in ("xcorr", "affine"):
        cfg = Cfg()
        cfg.reg_method = method
        v_ref, v_oth = pair(t)
        _, valid = register_patient([v_ref, v_oth], cfg, ref=0)
        runs[method] = (v_ref, v_oth, valid[0] & valid[1])

    # ONE slice set for both, or the numbers are not comparable: the methods disagree about which
    # slices survive, and a method that drops the hard ones would look better for it.
    keep = np.logical_and.reduce([r[2] for r in runs.values()])
    resid = {m: residual_in_brain(vr, vo, keep)[0] for m, (vr, vo, _) in runs.items()}
    assert resid["affine"] < 0.25 * resid["xcorr"], (
        f"affine {resid['affine']:.1f} did not beat xcorr {resid['xcorr']:.1f} on a pure tilt")


def test_reference_is_untouched():
    v_ref, v_oth = pair(rigid(5.0, 4.0, (3.0, 2.0, -2.0)))
    keep = v_ref.raw.clone()
    register_patient([v_ref, v_oth], Cfg(), ref=0)
    assert torch.equal(v_ref.raw, keep), "the reference volume was modified"


def test_background_is_not_zeroed_at_the_edge():
    """`norm` background is -median/MAD, so zero-padding would paint a mid-tissue rim.

    That rim is the artefact the whole no-masking rework was about. Pin that resampling does not
    reintroduce it.
    """
    v_ref, v_oth = pair(rigid(8.0, 0.0, (0.0, 0.0, 0.0)))   # empties the frame corners
    bg = float(v_oth.norm[:, CI][~v_oth.fg].median())
    assert bg < -0.5, f"fixture is wrong: unmasked median/MAD background should be negative ({bg})"
    register_patient([v_ref, v_oth], Cfg(), ref=0)
    corners = v_oth.norm[D // 2, CI][[0, 0, -1, -1], [0, -1, 0, -1]]
    assert torch.all(corners < bg + 1.0), (
        f"corner values {corners.tolist()} are not background-like (background {bg:.2f}) -- the "
        f"resampler zero-padded instead of replicating the border")


def test_slices_without_source_coverage_are_dropped():
    """A big tilt has no source data for the outermost slices; they must not be kept."""
    v_ref, v_oth = pair(rigid(0.0, 22.0, (0.0, 0.0, 0.0)))
    _, valid = register_patient([v_ref, v_oth], Cfg(), ref=0)
    assert not valid[1].all(), "every slice was marked valid despite a 22 deg tilt"
    assert valid[1].sum() > 0, "no slice survived a 22 deg tilt"


def test_stored_to_world_index_order_is_hwz():
    """A padded session's crop offset lands on h and w, never on z."""
    world = stored_to_world(np.eye(4), (40, 50), 64)
    # stored (0,0,0) is native ((40-64)//2, (50-64)//2, 0) = (-12, -7, 0)
    assert np.allclose(world @ np.array([0.0, 0.0, 0.0, 1.0]), [-12.0, -7.0, 0.0, 1.0])
    # a step along the THIRD index component is a step in world z, unshifted
    assert np.allclose(world @ np.array([0.0, 0.0, 1.0, 1.0]), [-12.0, -7.0, 1.0, 1.0])
