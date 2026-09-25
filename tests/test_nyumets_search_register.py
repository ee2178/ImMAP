# -*- coding: utf-8 -*-
"""Does `--reg-method search` recover a known dz, in-plane shift, and in-plane drift along z?

Same discipline as test_nyumets_register.py: the expectations come from the CONSTRUCTION, never
from the code under test, and the fixture is ANALYTIC -- one phantom defined as a function of the
world point, sampled by each study on its own offset grid. Nothing is resampled to build it, so the
only error left afterwards is the registration's own.

Sign convention, from the construction: a study sampled at world z + oz holds, at index i, what the
reference holds at index i + oz. `translate(a, t)` sets out[i] = a[i + t], so the shift that puts
the study back on the reference is t = -o on every axis.
"""
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.modules.setdefault("nibabel", types.ModuleType("nibabel"))

from preprocessing.nyumets_h5 import Volume, masked_ncc, register_patient  # noqa: E402

H = W = 64
D_REF, D_OTHER = 48, 44
CI = 1                                       # T1 is stored channel 1
_rng = np.random.default_rng(0)
_BLOBS = [(_rng.uniform(10, 38), _rng.uniform(16, 48), _rng.uniform(16, 48),
           _rng.uniform(4, 12), _rng.uniform(-1, 1, size=4)) for _ in range(40)]


def phantom(z, y, x):
    """(4, ...) channels + head mask at WORLD (z, y, x). Smooth, non-periodic, contrast-specific."""
    head = (((y - 32) / 26.0) ** 2 + ((x - 32) / 22.0) ** 2 + ((z - 24) / 22.0) ** 2) < 1
    ch = np.zeros((4,) + np.shape(z))
    for cz, cy, cx, s2, w in _BLOBS:
        g = np.exp(-((z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2) / (2 * s2))
        ch += w[:, None, None, None] * g
    return ch * head + head, head


def study(depth, o=(0, 0, 0), drift=0.0):
    """A Volume sampled at world (z + oz, y + oy, x + ox + drift * (z_world - 24))."""
    z, y, x = np.mgrid[0:depth, 0:H, 0:W].astype(np.float64)
    zw = z + o[0]
    ch, head = phantom(zw, y + o[1], x + o[2] + drift * (zw - 24))
    img = torch.from_numpy(np.ascontiguousarray(ch.transpose(1, 0, 2, 3)).astype(np.float32))
    fg = torch.from_numpy(head)
    return Volume(raw=img.clone() * 1000 + 500, norm=img.clone(), fg=fg.clone(), fov=fg.clone(),
                  stats=None, orig_depth=depth, affine=np.eye(4), lost={}, native_hw=(H, W))


def cfg(**over):
    c = SimpleNamespace(reg_method="search", reg_contrast="T1", reg_device="cpu", reg_stride=2,
                        reg_max_dz=12, reg_search=8, reg_slicewise=True, reg_slice_radius=6,
                        reg_slice_min_px=100, reg_smooth=5)
    for k, v in over.items():
        setattr(c, k, v)
    return c


def run(o, drift=0.0, **over):
    vols = [study(D_REF), study(D_OTHER, o, drift)]
    offsets, valid = register_patient(vols, cfg(**over), ref=0)
    return vols, offsets, valid


@pytest.mark.parametrize("o", [(9, 0, 0), (-7, 0, 0), (5, -6, 4), (-4, 7, -5)])
def test_recovers_whole_volume_shift_exactly(o):
    _, offsets, _ = run(o, reg_slicewise=False)
    assert offsets[0] == (0, 0, 0)
    assert offsets[1] == tuple(-c for c in o)


def test_registered_study_matches_the_reference():
    vols, _, _ = run((5, -6, 4))
    a, b = vols[0].norm[:, CI], vols[1].norm[:, CI]
    ncc = masked_ncc(a, b, vols[0].fg, vols[1].fg)
    assert ncc > 0.99
    assert vols[1].reg["ncc"] == pytest.approx(ncc, abs=0.02)     # recorded value is strided


def test_per_slice_shift_follows_an_in_plane_drift():
    drift, ox = 0.3, 2                          # dw drifts 0.3 px per slice about world z = 24
    vols, offsets, valid = run((4, 0, ox), drift=drift)
    dz = offsets[1][0]
    assert dz == -4
    sh = vols[1].reg["slice_shift"]
    # construction: output slice z holds world z, whose in-plane offset is ox + drift * (z - 24)
    z = np.flatnonzero(valid[1] & vols[0].fg.reshape(D_REF, -1).any(1).numpy())
    z = z[(z > 6) & (z < 42)]                   # brain wide enough for a slice-wise estimate
    want = -(ox + drift * (z - 24))
    assert np.abs(sh[z, 1] - want).max() <= 1.0
    assert np.abs(sh[z, 0]).max() <= 1            # no drift in dh


def test_per_slice_beats_one_shift_on_a_drift():
    one, _, _ = run((0, 0, 0), drift=0.3, reg_slicewise=False)
    per, _, _ = run((0, 0, 0), drift=0.3)
    score = lambda vols: masked_ncc(vols[0].norm[:, CI], vols[1].norm[:, CI],
                                    vols[0].fg, vols[1].fg)
    assert score(per) > score(one) + 0.02


def test_every_array_moves_together():
    vols, _, _ = run((5, -6, 4))
    v = vols[1]
    # raw was built as norm * 1000 + 500, so the shift must keep that relation wherever both
    # hold data; masks move with the pixels
    both = v.fg
    assert torch.allclose(v.raw[:, CI][both], v.norm[:, CI][both] * 1000 + 500, atol=1e-2)
    assert torch.equal(v.fg, v.fov)
    assert torch.equal(v.fg.bool(), vols[0].fg.bool() & v.fg.bool()) or \
        (v.fg.bool() ^ vols[0].fg.bool()).float().mean() < 0.02


def test_reference_is_untouched():
    ref0 = study(D_REF)
    vols, _, _ = run((5, -6, 4))
    assert torch.equal(vols[0].norm, ref0.norm)
    assert vols[0].reg["method"] == "search"
    assert np.all(vols[0].reg["slice_shift"] == 0)


def test_no_slicewise_means_one_shift_everywhere():
    vols, offsets, _ = run((0, 0, 0), drift=0.3, reg_slicewise=False)
    sh = vols[1].reg["slice_shift"]
    assert np.all(sh == np.asarray(offsets[1][1:]))


def test_valid_slices_follow_dz():
    _, offsets, valid = run((9, 0, 0), reg_slicewise=False)
    # the other study (depth 44, padded to 48) acquired its indices 0..43; after dz = -9 those land
    # on reference indices 9..52, of which 9..47 exist
    assert offsets[1][0] == -9
    assert np.array_equal(np.flatnonzero(valid[1]), np.arange(9, D_REF))


# ---- --reg-method centroid: dz by correlation, (dh, dw) from the brain-mask centroid ------------

@pytest.mark.parametrize("o", [(9, 0, 0), (5, -6, 4), (-4, 7, -5)])
def test_centroid_recovers_whole_volume_shift(o):
    # the phantom's head mask moves with the anatomy, so the centroid offset IS the shift
    vols, offsets, _ = run(o, reg_method="centroid")
    assert offsets[1] == tuple(-c for c in o)
    assert vols[1].reg["method"] == "centroid"
    assert vols[1].reg["ncc"] > 0.99


def test_centroid_shift_is_uniform_across_slices():
    vols, offsets, _ = run((5, -6, 4), reg_method="centroid")
    assert np.all(vols[1].reg["slice_shift"] == np.asarray(offsets[1][1:]))
    assert np.all(vols[0].reg["slice_shift"] == 0)


def test_centroid_moves_every_array_together():
    vols, _, _ = run((5, -6, 4), reg_method="centroid")
    v = vols[1]
    assert torch.allclose(v.raw[:, CI][v.fg], v.norm[:, CI][v.fg] * 1000 + 500, atol=1e-2)
    assert torch.equal(v.fg, v.fov)
