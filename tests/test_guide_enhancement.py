# -*- coding: utf-8 -*-
"""`guide_enhancement`: the guide is the prior study's enhancement map (CT1 - T1)_+, not its CT1.

Checked on a phantom with known values: every guide plane must be exactly the scaled, masked
(CT1 - T1)_+ of a slice of the OTHER study, one plane per window slice whatever `guide_idx` says,
in both loaders; and `guide_center=False` must leave such a guide uncentred in the net.
"""
import os
import sys
import types

import h5py
import numpy as np
import pytest
import torch

sys.modules.setdefault("nibabel", types.ModuleType("nibabel"))

import datasets  # noqa: E402,F401  (registers the loaders)
from datasets.NYUMets.synth_register import n_input_channels  # noqa: E402
from datasets.registry import build_loader  # noqa: E402
from models.sb_guided_groupcdl import SBGuidedGroupCDL  # noqa: E402

H, NZ = 16, 8
SCALES = [2.0, 4.0, 5.0, 10.0]


def volume(seed):
    rng = np.random.default_rng(seed)
    img = rng.uniform(1.0, 9.0, (NZ, H, H, 4)).astype(np.float32)
    img[:, :, :3, 1] = 0                       # T1 support: the mask must zero the guide there
    return img


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    d = tmp_path_factory.mktemp("nyu")
    for s in (0, 1):
        case = "PT_s%02d" % s
        os.makedirs(d / case)
        with h5py.File(d / case / (case + "_img.h5"), "w") as f:
            f.create_dataset("img_raw", data=volume(s))
            f.create_dataset("img", data=volume(s))
            f.create_dataset("mask", data=np.ones((NZ, H, H, 1), np.uint8))
            f.create_dataset("slice_index", data=np.arange(NZ, dtype=np.int32))
            f.attrs["patient"], f.attrs["session"] = "PT", "s%02d" % s
    return str(d)


def block(root, name, window, **kw):
    d = dict(name=name, root=root, x0_idx=2, x1_idx=1, cond_idx=[0, 1, 3], image_key="img_raw",
             scales=SCALES, guide_mode="other_study", guide_idx=[2], guide_window=window,
             guide_slice="index", deterministic=True, mask_source="t1ct1", mask_apply=True,
             mask_guides="target", batch_size=NZ, num_workers=0)
    d.update(kw)
    return d


def expected(study, z, mask):
    """(CT1 - T1)_+ of one stored slice, scaled, under the TARGET session's mask."""
    v = volume(study)[min(max(z, 0), NZ - 1)] / np.asarray(SCALES, np.float32)
    return np.clip(v[..., 2] - v[..., 1], 0.0, None) * mask


@pytest.mark.parametrize("window", [0, 1])
@pytest.mark.parametrize("guide_idx", [[2], [1, 2]])           # ignored either way
def test_bridge_loader_returns_the_prior_enhancement_map(root, window, guide_idx):
    ld = build_loader(block(root, "nyumets_guided", window, guide_idx=guide_idx,
                            guide_enhancement=True), shuffle=False, drop_last=False)
    assert ld.dataset.n_guide_planes == 2 * window + 1
    batch = next(iter(ld))
    guide = batch["guide"].numpy()[:, :, 0]                 # (B, G, 1, H, W) -> (B, G, H, W)
    assert guide.shape[1] == 2 * window + 1 and float(guide.min()) >= 0.0
    for i in range(NZ):                                        # study 0's slices: the guide is study 1
        tgt = volume(0)[i]
        mask = ((tgt[..., 1] != 0) & (tgt[..., 2] != 0)).astype(np.float32)
        for j, dz in enumerate(range(-window, window + 1)):
            np.testing.assert_allclose(guide[i, j], expected(1, i + dz, mask), rtol=1e-5, atol=1e-6)
        assert float(np.abs(guide[i][:, :, :3]).max()) == 0.0  # outside the target's T1 support
    assert float(guide.max()) > 0.1                            # and not an all-zero plane


def test_off_is_the_stored_contrast_as_before(root):
    ld = build_loader(block(root, "nyumets_guided", 0), shuffle=False, drop_last=False)
    guide = next(iter(ld))["guide"].numpy()[:, :, 0]
    tgt = volume(0)[0]
    mask = ((tgt[..., 1] != 0) & (tgt[..., 2] != 0)).astype(np.float32)
    np.testing.assert_allclose(guide[0, 0], volume(1)[0][..., 2] / SCALES[2] * mask, rtol=1e-5)


def test_synthesis_loader_appends_the_same_planes(root):
    kw = block(root, "nyumets_synth", 1, guide_idx=[1, 2], guide_enhancement=True)
    X = next(iter(build_loader(kw, shuffle=False, drop_last=False)))[0].numpy()
    assert X.shape[1] == n_input_channels(**kw) == 3 + 3
    tgt = volume(0)[2]
    mask = ((tgt[..., 1] != 0) & (tgt[..., 2] != 0)).astype(np.float32)
    for j, dz in enumerate((-1, 0, 1)):
        np.testing.assert_allclose(X[2, 3 + j], expected(1, 2 + dz, mask), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("center", [True, False])
def test_guide_center_switch(center):
    torch.manual_seed(0)
    net = SBGuidedGroupCDL(K=1, M=4, P=3, s=2, Mh=2, C=4, prior_idx=1, guide_window=3,
                           attn_backend="gather", spectral_init=False, s_mode="free",
                           guide_center=center, kind="brownian", tau=0.1, n_points=50)
    g = torch.rand(2, 1, 16, 16) + 0.5
    out = net._prep_guides(g, (0, 0, 0, 0), g)[0]
    want = g - g.mean(dim=(1, 2, 3), keepdim=True) if center else g
    assert torch.allclose(out, want)
