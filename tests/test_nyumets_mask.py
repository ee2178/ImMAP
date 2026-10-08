# -*- coding: utf-8 -*-
"""The T1 & CT1 support mask in the NYUMets loaders (mask_source / mask_apply / mask_guides).

Each contrast in the fixture has its OWN support, planted by hand, so every assertion is about
which pixels survive:

    T1     zero in the left 6 columns            CT1    zero in the top 4 rows
    FLAIR  zero in the right 10 columns          T2     nonzero everywhere
    h5 `mask`   a small centred box (the "cuts into signal" mask)

The second study of patient PT is the first shifted 3 pixels down, so its own T1 & CT1 support
differs from the first study's.
"""
import os
import sys
import types

import h5py
import numpy as np
import pytest
import torch
import torchvision

sys.modules.setdefault("nibabel", types.ModuleType("nibabel"))

import datasets  # noqa: E402,F401  (registers the loaders)
from datasets.registry import build_loader  # noqa: E402

H, NZ, SHIFT = 24, 6, 3
SCALES = [2.0, 4.0, 5.0, 10.0]


def _volume(shift):
    rng = np.random.default_rng(shift)
    img = rng.uniform(1.0, 2.0, (NZ, H, H, 4)).astype(np.float32)
    img[:, :, 14:, 0] = 0          # FLAIR: no right columns
    img[:, :, :6, 1] = 0           # T1: no left columns
    img[:, :4, :, 2] = 0           # CT1: no top rows
    img = np.roll(img, shift, axis=1)                    # the other study sits lower
    box = np.zeros((NZ, H, H, 1), np.uint8)
    box[:, 8:16, 8:16] = 1
    return img, box


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    d = tmp_path_factory.mktemp("nyu")
    for pid, sess in (("PT", (0, 1)), ("SOLO", (0,))):
        for s in sess:
            case = "%s_s%02d" % (pid, s)
            os.makedirs(d / case)
            img, box = _volume(SHIFT * s)
            with h5py.File(d / case / (case + "_img.h5"), "w") as f:
                f.create_dataset("img_raw", data=img)
                f.create_dataset("img", data=img)
                f.create_dataset("mask", data=box)
                f.create_dataset("slice_index", data=np.arange(NZ, dtype=np.int32))
                f.attrs["patient"], f.attrs["session"] = pid, "s%02d" % s
    return str(d)


def ds(root, name="nyumets_guided", **kw):
    cfg = dict(name=name, root=root, x0_idx=2, x1_idx=1, cond_idx=[0, 3], image_key="img_raw",
               scales=SCALES, guide_mode="none", deterministic=True, random_flips=False,
               batch_size=2, num_workers=0)
    cfg.update(kw)
    return build_loader(cfg, shuffle=False, drop_last=False).dataset


def support(shift=0):
    """T1 & CT1 support of a study shifted `shift` rows: (1, H, H) bool."""
    m = np.ones((H, H), bool)
    m[:, :6] = False               # T1
    m[:4, :] = False               # CT1
    return torch.from_numpy(np.roll(m, shift, axis=0))[None]


def test_default_is_unchanged(root):
    x0, x1, cond, mask = ds(root)[0]
    assert float(mask.sum()) == 64.0                      # the stored box
    assert float((x0 != 0).float().mean()) > 0.8          # nothing was multiplied
    assert float((cond[1] != 0).float().mean()) == 1.0    # T2 is everywhere


def test_t1ct1_mask_is_the_shared_support_of_those_two_only(root):
    x0, x1, cond, mask = ds(root, mask_source="t1ct1")[0]
    assert torch.equal(mask.bool(), support())
    assert mask.dtype == torch.float32 and mask.shape == (1, H, H)
    # FLAIR's smaller field of view does NOT shrink it
    assert float(mask[0, 10, 20]) == 1.0 and float(cond[0, 10, 20]) == 0.0
    # mask_apply off: the images are untouched even though the mask changed
    assert float((cond[1] != 0).float().mean()) == 1.0


def test_apply_masks_every_image_of_the_session(root):
    d = ds(root, mask_source="t1ct1", mask_apply=True)
    x0, x1, cond, mask = d[0]
    raw = ds(root)[0]
    m = support()
    for got, want in ((x0, raw[0]), (x1, raw[1]), (cond[:1], raw[2][:1]), (cond[1:], raw[2][1:])):
        assert torch.equal(got, want * m)
        assert float(got[~m.expand_as(got)].abs().max()) == 0.0
    assert float(cond[1][m[0]].abs().min()) > 0           # T2 survives everywhere inside


def test_apply_with_the_h5_mask_uses_the_stored_box(root):
    x0, x1, cond, mask = ds(root, mask_apply=True)[0]
    assert float(mask.sum()) == 64.0
    assert float((cond[1] != 0).sum()) == 64.0            # T2 cut down to the box: the old behaviour


@pytest.mark.parametrize("mode", ["target", "own", "none"])
def test_guides_follow_mask_guides(root, mode):
    kw = dict(guide_mode="other_study", guide_idx=[1, 2, 3], guide_slice="index",
              mask_source="t1ct1", mask_apply=True, mask_guides=mode)
    s = ds(root, **kw)[0]
    raw = ds(root, guide_mode="other_study", guide_idx=[1, 2, 3], guide_slice="index")[0]
    g, g_raw = s["guide"][:, 0], raw["guide"][:, 0]       # (3, H, H): prior T1, CT1, T2
    assert torch.equal(s["mask"].bool(), support())       # the returned mask is ALWAYS the target's
    if mode == "none":
        assert torch.equal(g, g_raw)
        return
    m = support(0 if mode == "target" else SHIFT)         # this file is study 0; its sibling is 1
    assert torch.equal(g, g_raw * m)
    # prior T2 is nonzero everywhere, so its support after masking IS the mask that was used
    assert torch.equal(g[2] != 0, m[0])
    assert not torch.equal(support(0), support(SHIFT))    # the two rules really differ here


def test_same_session_guides_get_the_target_mask_under_own(root):
    s = ds(root, guide_mode="same_session", guide_contrasts=[3], mask_source="t1ct1",
           mask_apply=True, mask_guides="own")[0]
    assert torch.equal(s["guide"][0, 0] != 0, support()[0])


def test_other_study_bridge_start_is_masked_like_a_guide(root):
    kw = dict(x1_source="other_study", guide_slice="index", mask_source="t1ct1", mask_apply=True)
    raw = ds(root, x1_source="other_study", guide_slice="index")[0]
    tgt = ds(root, mask_guides="target", **kw)[0]
    own = ds(root, mask_guides="own", **kw)[0]
    assert torch.equal(tgt["x1"], raw["x1"] * support(0))
    assert torch.equal(own["x1"], raw["x1"] * support(SHIFT))
    assert torch.equal(tgt["y"], raw["y"] * support(0))   # this session's T1: always the target mask


def test_mask_and_images_move_together_through_the_crop(root):
    torch.manual_seed(0)
    d = ds(root, mask_source="t1ct1", mask_apply=True, crop_size=12, random_flips=True,
           guide_mode="other_study", guide_idx=[3], guide_slice="index")
    for i in range(6):
        s = d[i]
        m = s["mask"].bool()
        if hasattr(torchvision, "__version__"):           # a stubbed torchvision does not crop
            assert s["x0"].shape == (1, 12, 12)
        assert float(s["cond"][1][~m[0]].abs().max() if (~m).any() else 0.0) == 0.0
        assert torch.equal(s["cond"][1] != 0, m[0])       # T2: support == the cropped mask
        assert torch.equal(s["guide"][0, 0] != 0, m[0])


def test_eps_and_mask_idx(root):
    x0, x1, cond, mask = ds(root, mask_source="t1ct1", mask_eps=10.0)[0]
    assert float(mask.sum()) == 0.0                       # nothing exceeds 10 after scaling
    _, _, _, m_t1 = ds(root, mask_source="t1ct1", mask_idx=[1])[0]
    assert float(m_t1[0, 0, 10]) == 1.0 and float(m_t1[0, 10, 0]) == 0.0    # T1's support alone


def test_synthesis_loader_passes_the_options_through(root):
    """The registrar pass-through trap: a key missing from the signature is silently dropped."""
    X, y, mask, et = ds(root, name="nyumets_synth", cond_idx=[0, 1, 3], mask_source="t1ct1",
                        mask_apply=True)[0]
    m = support()
    assert torch.equal(mask.bool(), m)
    assert torch.equal(X[2] != 0, m[0]) and float(y[~m].abs().max()) == 0.0
    Xg, _, mg, _ = ds(root, name="nyumets_synth", cond_idx=[3], guide_mode="other_study",
                      guide_idx=[3], guide_slice="index", mask_source="t1ct1", mask_apply=True,
                      mask_guides="own")[0]
    assert torch.equal(Xg[0] != 0, support(0)[0]) and torch.equal(Xg[1] != 0, support(SHIFT)[0])


def test_bad_values(root):
    with pytest.raises(ValueError, match="mask_source"):
        ds(root, mask_source="brain")
    with pytest.raises(ValueError, match="mask_guides"):
        ds(root, mask_guides="theirs")
