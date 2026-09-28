# -*- coding: utf-8 -*-
"""`nyumets_synth` must hand train_synthesis the same conditioning its i2sb twin sees.

The end-to-end arm of the longitudinal comparison only means something if the two arms differ in
ONE thing -- the objective. That rests entirely on this adapter putting the right planes in the
right order, so plane identity is decoded from the pixels rather than inferred from shapes.

X = [cond contrasts of this session] ++ [prior study's guide planes, offset outermost]
y = this session's x0_idx contrast
"""
import sys
import types

import h5py
import numpy as np
import pytest
import torch

sys.modules.setdefault("nibabel", types.ModuleType("nibabel"))

from datasets.NYUMets.synth_register import (          # noqa: E402
    build_nyumets_synth_loader, n_input_channels,
)

H = 8
NZ = 12
NC_STORED = 4                      # FLAIR, T1, T1ce, T2
COND = [0, 1, 3]
GUIDE = [1, 2]


def encode(study, z, c):
    """A pixel value naming its own (study, slice, contrast), exactly representable."""
    return 10000.0 * study + 100.0 * z + c


def decode(plane):
    v = plane.reshape(-1)
    assert torch.allclose(v, v[0].expand_as(v)), "plane is not constant; fixture is broken"
    study, rem = divmod(round(float(v[0])), 10000)
    z, c = divmod(rem, 100)
    return int(study), int(z), int(c)


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    d = tmp_path_factory.mktemp("nyu_synth")
    for study in (0, 1, 2):
        case = "PT_s%02d" % study
        sub = d / case
        sub.mkdir()
        img = np.empty((NZ, H, H, NC_STORED), dtype=np.float32)
        for z in range(NZ):
            for c in range(NC_STORED):
                img[z, :, :, c] = encode(study, z, c)
        with h5py.File(str(sub / (case + "_img.h5")), "w") as f:
            f.create_dataset("img", data=img)
            f.create_dataset("img_raw", data=img)
            f.create_dataset("mask", data=np.ones((NZ, H, H, 1), np.uint8))
            f.create_dataset("slice_index", data=np.arange(NZ, dtype=np.int32))
            f.attrs["patient"] = "PT"
            f.attrs["session"] = "s%02d" % study
    return str(d)


def mid_sample(root, **over):
    """One sample from the MIDDLE of a volume, unbatched.

    Not batch 0 of the DataLoader: that is slice 0, where a guide window correctly clamps to the
    edge of the prior study (see tests/test_longitudinal_guides.py) and the offsets are no longer
    distinct. Plane-order checks need a slice with room either side.
    """
    dl = loader(root, **over)
    ds = dl.dataset
    return ds[len(ds) // 2]


def loader(root, **over):
    kw = dict(root=root, image_key="img_raw", x0_idx=2, x1_idx=1, cond_idx=COND,
              guide_mode="other_study", guide_idx=GUIDE, guide_slice="index",
              deterministic=True, batch_size=2, num_workers=0)
    kw.update(over)
    return build_nyumets_synth_loader(**kw)


def test_batch_is_the_four_tuple_train_synthesis_expects(root):
    X, y, mask, et = next(iter(loader(root, guide_window=1)))
    assert X.dim() == 4 and y.dim() == 4
    assert y.shape[1] == 1, "the target must be one channel"
    assert mask.shape == y.shape
    assert et.shape == y.shape
    assert float(et.abs().sum()) == 0.0, "NYUMets has no ET mask; it must be zeros"


@pytest.mark.parametrize("w", [0, 1, 2])
def test_channel_order_is_cond_then_guides(root, w):
    X, y, _, _ = mid_sample(root, guide_window=w)
    nc = len(GUIDE)
    assert X.shape[0] == len(COND) + nc * (2 * w + 1)
    assert X.shape[0] == n_input_channels(cond_idx=COND, guide_mode="other_study",
                                          guide_idx=GUIDE, guide_window=w), \
        "n_input_channels disagrees with the loader, so a config's in_chans would be wrong"

    tgt_study, tgt_z, tgt_c = decode(y[0])
    assert tgt_c == 2, "y should be the x0_idx contrast (T1ce), decoded %d" % tgt_c

    # the first len(COND) channels are THIS session's cond contrasts, in cond_idx order
    for k, c in enumerate(COND):
        st, z, cc = decode(X[k])
        assert (st, z, cc) == (tgt_study, tgt_z, c), (
            "X channel %d should be this session's contrast %d at the target slice, got "
            "study %d slice %d contrast %d" % (k, c, st, z, cc))

    # then the guides: offset outermost (-w..+w), contrast innermost, all from ONE other study
    gids = [decode(X[len(COND) + j]) for j in range(nc * (2 * w + 1))]
    sibs = {st for st, _, _ in gids}
    assert len(sibs) == 1 and sibs != {tgt_study}, (
        "guides must come from exactly one OTHER study, got %s (target %d)" % (sibs, tgt_study))
    z0 = gids[len(gids) // 2][1]
    for j, (_, z, c) in enumerate(gids):
        off, ci = divmod(j, nc)
        assert z == z0 + (off - w), (
            "guide %d should be offset %+d (slice %d), decoded %d" % (j, off - w, z0 + off - w, z))
        assert c == GUIDE[ci], (
            "guide %d should be contrast %d, decoded %d" % (j, GUIDE[ci], c))


def test_guide_slice_index_matches_the_target_level(root):
    """guide_slice='index' pairs by ORIGINAL slice index -- what registration makes meaningful."""
    X, y, _, _ = mid_sample(root, guide_window=0)
    _, tgt_z, _ = decode(y[0])
    _, g_z, _ = decode(X[len(COND)])
    assert g_z == tgt_z, "prior plane at slice %d but the target is at %d" % (g_z, tgt_z)


def test_no_guide_gives_cond_only(root):
    X, _, _, _ = mid_sample(root, guide_mode="none", guide_idx=None)
    assert X.shape[0] == len(COND)
    assert X.shape[0] == n_input_channels(cond_idx=COND, guide_mode="none")


def test_residual_anchor_channel_is_t1(root):
    """The BraTS synthesis configs use residual_src_idx=1 with cond_idx [0,1,3]; same here."""
    X, _, _, _ = mid_sample(root, guide_window=0)
    _, _, c = decode(X[1])
    assert c == 1, "channel 1 should be T1 (the residual anchor), decoded contrast %d" % c


def test_single_study_patients_are_dropped(root, tmp_path):
    """other_study cannot serve a patient with one session, so the split must exclude it."""
    import shutil
    d = tmp_path / "mixed"
    shutil.copytree(root, str(d))
    solo = d / "SOLO_s00"
    solo.mkdir()
    img = np.zeros((NZ, H, H, NC_STORED), np.float32)
    with h5py.File(str(solo / "SOLO_s00_img.h5"), "w") as f:
        f.create_dataset("img", data=img)
        f.create_dataset("img_raw", data=img)
        f.create_dataset("mask", data=np.ones((NZ, H, H, 1), np.uint8))
        f.create_dataset("slice_index", data=np.arange(NZ, dtype=np.int32))
        f.attrs["patient"] = "SOLO"
        f.attrs["session"] = "s00"
    dl = loader(str(d), guide_window=0)
    pats = {p for p in dl.dataset.ds.patients}
    assert pats == {"PT"}, "SOLO should have been dropped, got %s" % sorted(pats)
