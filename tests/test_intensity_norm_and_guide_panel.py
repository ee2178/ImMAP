# -*- coding: utf-8 -*-
"""Two loader-side additions for the longitudinal runs.

1. intensity_norm="volume_median": every contrast of every volume divided by ITS OWN in-brain
   median (the h5's norm_stats centre) instead of one fixed `scales` per contrast. On a phantom
   whose second study is the first at 1.7x the scanner gain, the prior guide must come back on the
   current study's scale -- and with the fixed scales it must not.
2. visualization.guide_panel: the labels match the loader's own plane order and count for every
   way guides travel (own `guide` entry, tail of cond, tail of the synthesis X), and sample 0's
   planes are pulled out of the batch exactly.
"""
import os
import sys
import types

import h5py
import numpy as np
import pytest
import torch

sys.modules.setdefault("nibabel", types.ModuleType("nibabel"))

import datasets  # noqa: E402,F401
from datasets.registry import build_loader  # noqa: E402
from visualization.guide_panel import guide_layout, guide_panel, sample_guides  # noqa: E402

H, NZ = 16, 8
GAIN = 1.7


def volume(seed, gain):
    rng = np.random.default_rng(seed)
    img = rng.uniform(800.0, 1200.0, (NZ, H, H, 4)).astype(np.float32)
    img[:, :, :3, :] = 0.0                            # outside the head
    return img * gain


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    d = tmp_path_factory.mktemp("nyu")
    base = volume(0, 1.0)
    for s, gain in ((0, 1.0), (1, GAIN)):             # SAME anatomy, different scanner gain
        img = base * gain
        case = "PT_s%02d" % s
        os.makedirs(d / case)
        with h5py.File(d / case / (case + "_img.h5"), "w") as f:
            f.create_dataset("img_raw", data=img)
            f.create_dataset("img", data=img)
            f.create_dataset("mask", data=(img[..., 1:2] > 0).astype(np.uint8))
            f.create_dataset("slice_index", data=np.arange(NZ, dtype=np.int32))
            med = np.array([np.median(img[..., c][img[..., c] > 0]) for c in range(4)])
            f.attrs["norm_stats"] = np.stack([med, np.ones(4)], axis=1).astype(np.float32)
            f.attrs["patient"], f.attrs["session"] = "PT", "s%02d" % s
    return str(d)


def block(root, name="nyumets_guided", **kw):
    d = dict(name=name, root=root, x0_idx=2, x1_idx=1, cond_idx=[0, 1, 3], image_key="img_raw",
             scales=[1000.0, 1000.0, 1000.0, 1000.0], guide_mode="other_study", guide_idx=[1, 2],
             guide_window=1, guide_slice="index", deterministic=True, mask_source="t1ct1",
             mask_apply=True, mask_guides="target", batch_size=NZ, num_workers=0)
    d.update(kw)
    return d


def ratio_prior_to_current(root, norm):
    b = next(iter(build_loader(block(root, intensity_norm=norm), shuffle=False, drop_last=False)))
    ct1 = b["x0"][:, 0]
    prior_ct1 = b["guide"][:, 3, 0]          # plane order: offset -1..+1 outer, [T1, CT1] inner
    m = ct1 > 0
    return float(prior_ct1[m].median() / ct1[m].median()), float(ct1[m].median())


def test_fixed_scales_keep_the_gain_and_volume_median_removes_it(root):
    r_fixed, _ = ratio_prior_to_current(root, "none")
    r_norm, med = ratio_prior_to_current(root, "volume_median")
    # the loader serves study 0 and study 1 in turn, so the ratio is GAIN or 1/GAIN
    assert min(abs(r_fixed - GAIN), abs(r_fixed - 1 / GAIN)) < 0.02, r_fixed
    assert abs(r_norm - 1.0) < 0.02, r_norm
    assert abs(med - 1.0) < 0.05, med                  # tissue lands near 1


def test_volume_median_needs_the_attr(root, tmp_path):
    import shutil
    d = tmp_path / "copy"
    shutil.copytree(root, d)
    for f in d.glob("*/*_img.h5"):
        with h5py.File(f, "a") as h:
            del h.attrs["norm_stats"]
    with pytest.raises(KeyError, match="norm_stats"):
        next(iter(build_loader(block(str(d), intensity_norm="volume_median"), shuffle=False,
                               drop_last=False)))


def test_unknown_intensity_norm_is_refused(root):
    with pytest.raises(ValueError, match="intensity_norm"):
        build_loader(block(root, intensity_norm="zscore"), shuffle=False, drop_last=False)


# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("how", ["guide", "cond", "synth"])
def test_guide_layout_matches_the_loader(root, how):
    kw = dict(guide_as_cond=True) if how == "cond" else {}
    name = "nyumets_synth" if how == "synth" else "nyumets_guided"
    ld = build_loader(block(root, name=name, **kw), shuffle=False, drop_last=False)
    lay = guide_layout(ld)
    assert lay["labels"] == ["prior T1 z-1", "prior CT1 z-1", "prior T1 z+0", "prior CT1 z+0",
                             "prior T1 z+1", "prior CT1 z+1"]
    assert lay["as_cond"] is (how != "guide") and lay["n_cond"] == 3 and lay["t1_pos"] == 1
    b = next(iter(ld))
    if how == "synth":
        X = b[0]
        g = sample_guides(lay, X=X)
        assert torch.equal(g, X[0, 3:])
    elif how == "cond":
        g = sample_guides(lay, cond=b[2])
        assert torch.equal(g, b[2][0, 3:])
    else:
        g = sample_guides(lay, guide=b["guide"])
        assert torch.equal(g, b["guide"][0].flatten(0, 1))
    tiles, cap = guide_panel([("T1", g[0]), ("CT1", g[1])], g, lay, (0.0, 2.0), g[0] > 0)
    assert tiles.shape == (2 + 6, 1, H, H) and float(tiles.min()) >= 0 and float(tiles.max()) <= 1
    assert cap.split(" | ")[2] == "prior T1 z-1"


def test_enhancement_guides_are_labelled_and_windowed(root):
    ld = build_loader(block(root, guide_idx=[2], guide_enhancement=True, guide_window=0),
                      shuffle=False, drop_last=False)
    lay = guide_layout(ld)
    assert lay["labels"] == ["prior (CT1-T1)+"] and lay["enh"] == [True]
    g = torch.full((1, H, H), 0.5)
    tiles, cap = guide_panel([], g, lay, (0.0, 2.0))
    assert torch.allclose(tiles, torch.full_like(tiles, 0.5)) and "[0, 1]" in cap


def test_no_guides_no_layout(root):
    ld = build_loader(block(root, guide_mode="none", guide_window=0), shuffle=False,
                      drop_last=False)
    assert guide_layout(ld) is None
