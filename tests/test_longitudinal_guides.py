# -*- coding: utf-8 -*-
"""The guide-plane ORDER that NYUMetsGuidedDataset emits, pinned by decoding the planes.

notebooks/immap_sb_longitudinal.ipynb runs one loader at the largest `guide_window` any arm needs
and carves each arm's sub-stack out of that single stack, so all arms see the same slices AND the
same sibling study. That slicing is only correct while the plane order stays:

    slice offset OUTERMOST (-w .. +w), contrast innermost

If it ever changes, the notebook silently hands nets channels they were not trained on -- a
train/eval mismatch with no error and no obvious artefact. So this decodes plane identity from the
pixels rather than asserting a shape, and pins the notebook's slice arithmetic against it.
"""
import os
import sys
import types

import h5py
import numpy as np
import pytest
import torch

sys.modules.setdefault("nibabel", types.ModuleType("nibabel"))

from datasets.NYUMets.longitudinal_dataset import NYUMetsGuidedDataset      # noqa: E402

H = W_PIX = 8
NZ = 12
NC_STORED = 4                       # FLAIR, T1, T1ce, T2


def encode(study, z, c):
    """A pixel value that names its own (study, slice, contrast), exactly representable."""
    return 10000.0 * study + 100.0 * z + c


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    d = tmp_path_factory.mktemp("nyu_guided")
    for study in (0, 1, 2):
        case = "PT_s%02d" % study
        sub = d / case
        sub.mkdir()
        img = np.empty((NZ, H, W_PIX, NC_STORED), dtype=np.float32)
        for z in range(NZ):
            for c in range(NC_STORED):
                img[z, :, :, c] = encode(study, z, c)
        with h5py.File(str(sub / (case + "_img.h5")), "w") as f:
            f.create_dataset("img", data=img)
            f.create_dataset("img_raw", data=img)
            f.create_dataset("mask", data=np.ones((NZ, H, W_PIX, 1), np.uint8))
            f.create_dataset("slice_index", data=np.arange(NZ, dtype=np.int32))
            f.attrs["patient"] = "PT"
            f.attrs["session"] = "s%02d" % study
    return str(d)


def make_ds(root, guide_window, guide_idx=(1, 2), guide_as_cond=False):
    cfg = types.SimpleNamespace(
        root=root, x0_idx=2, x1_idx=1, cond_idx=[0, 1, 3],
        image_key="img_raw", scales=None,
        guide_mode="other_study", guide_idx=list(guide_idx), guide_contrasts=None,
        n_guides=1, min_slice_gap=5, guide_slice="index", guide_window=guide_window,
        deterministic=True, guide_as_cond=guide_as_cond,
        center_crop=None, crop_size=None, random_flips=False, slice_range=None,
        x1_source="contrast", x1_other_idx=None, y_idx=None,
    )
    return NYUMetsGuidedDataset(cfg)


def decode(plane):
    """A guide plane -> (study, z, contrast). Every pixel must agree."""
    v = plane.reshape(-1)
    assert torch.allclose(v, v[0].expand_as(v)), "plane is not constant; fixture is broken"
    q = float(v[0])
    study, rem = divmod(round(q), 10000)
    z, c = divmod(rem, 100)
    return int(study), int(z), int(c)


@pytest.mark.parametrize("w", [0, 1, 2, 3])
def test_guide_plane_order_is_offset_outermost_contrast_innermost(root, w):
    ds = make_ds(root, w)
    gi = [1, 2]
    sample = ds[len(ds) // 2]                       # a slice away from either end
    guide = sample["guide"]
    assert guide.shape[0] == len(gi) * (2 * w + 1), (
        "expected %d guide planes, got %d" % (len(gi) * (2 * w + 1), guide.shape[0]))

    ids = [decode(guide[k]) for k in range(guide.shape[0])]
    sib_study = {s for s, _, _ in ids}
    assert len(sib_study) == 1, "one sibling study per sample, got %s" % sorted(sib_study)

    z0 = ids[len(ids) // 2][1]                      # the centre plane's slice
    for k, (_, z, c) in enumerate(ids):
        off, ci = divmod(k, len(gi))
        assert z == z0 + (off - w), (
            "plane %d should be offset %+d (slice %d), decoded slice %d -- the ORDER changed"
            % (k, off - w, z0 + off - w, z))
        assert c == gi[ci], (
            "plane %d should be contrast %d, decoded %d -- contrast is no longer innermost"
            % (k, gi[ci], c))


def test_notebook_substack_slicing_reproduces_a_smaller_window(root):
    """The notebook's `guide_for`: block (W-w)*NC : (W+w+1)*NC of the shared W stack.

    Checked against what the dataset ACTUALLY returns for that smaller window, on the same sample,
    so this is the real train/eval equivalence the notebook relies on -- not a restatement of the
    formula.
    """
    big_w, gi = 2, [1, 2]
    nc = len(gi)
    big = make_ds(root, big_w)
    i = len(big) // 2
    shared = big[i]["guide"]

    for w in (0, 1, 2):
        small = make_ds(root, w)
        assert len(small) == len(big), "the two datasets index different samples"
        want = small[i]["guide"]
        got = shared[(big_w - w) * nc:(big_w + w + 1) * nc]
        assert got.shape == want.shape, (
            "w=%d: sliced %s but the loader gives %s" % (w, tuple(got.shape), tuple(want.shape)))
        assert torch.equal(got, want), (
            "w=%d: the sub-stack is not what a w=%d loader returns; planes %s vs %s"
            % (w, w, [decode(g) for g in got], [decode(g) for g in want]))


def test_prior_ct1_channel_index(root):
    """OCT1_CH = W*NC + guide_idx.index(x0_idx) is the prior study's CT1 at offset 0."""
    for big_w in (0, 1, 2):
        gi = [1, 2]
        nc = len(gi)
        ds = make_ds(root, big_w)
        i = len(ds) // 2
        s = ds[i]
        ch = big_w * nc + gi.index(2)               # x0_idx = 2 = T1ce
        study, z, c = decode(s["guide"][ch])
        assert c == 2, "OCT1_CH decoded contrast %d, expected 2 (CT1)" % c
        # offset 0 means the matched slice: same ORIGINAL index as the target, since
        # guide_slice="index" and every study here shares one slice_index
        tgt_z = decode(s["x0"][0] if s["x0"].shape[0] == 1 else s["x0"])[1]
        assert z == tgt_z, (
            "prior CT1 is at slice %d but the target is at %d; guide_slice='index' broken"
            % (z, tgt_z))
        assert study != decode(s["x0"])[0], "the 'prior' plane came from the target's own study"


def test_guide_as_cond_appends_in_the_same_order(root):
    """With guide_as_cond the planes ride on cond; the notebook mimics that by concatenating.

    Folding the guide away also drops the sample back to the TUPLE layout (x0, x1, cond, mask),
    because nothing is left that needs a key -- see training.i2sb._batch_parts. The notebook asks
    for guide_as_cond=False precisely so it gets the dict and can slice the stack per arm.
    """
    w = 1
    i = 5
    plain = make_ds(root, w, guide_as_cond=False)[i]
    folded = make_ds(root, w, guide_as_cond=True)[i]
    assert isinstance(plain, dict) and "guide" in plain, "expected the dict layout with a guide"
    assert not isinstance(folded, dict), "guide_as_cond should fold the guide away entirely"

    # The dict layout is (G, 1, H, W) deliberately -- models.guided_prox reads a 4-D guide as ONE
    # plane of G channels, so the stacked form must be 5-D once batched. The notebook's
    # `guide_planes` flattens that back, and this is the equivalence it relies on.
    g = plain["guide"]
    assert g.dim() == 4 and g.shape[1] == 1, (
        "expected (G, 1, H, W); got %s. The notebook's guide_planes() flatten is wrong now."
        % (tuple(g.shape),))
    flat = g.flatten(0, 1)                               # unbatched analogue of flatten(1, 2)

    cond_folded = folded[2]
    n_cond = plain["cond"].shape[0]
    assert cond_folded.shape[0] == n_cond + flat.shape[0]
    assert torch.equal(cond_folded[:n_cond], plain["cond"]), "cond changed"
    assert torch.equal(cond_folded[n_cond:], flat), (
        "guides are appended in a different order than the notebook's cat([cond, guide])")


def test_window_repeats_the_edge_slice(root):
    """Past either end of the prior study the window repeats the edge, it does not wrap."""
    w = 2
    ds = make_ds(root, w)
    gi = [1, 2]
    nc = len(gi)
    for i in (0, len(ds) - 1):                      # first and last slice of some study
        ids = [decode(p) for p in ds[i]["guide"]]
        zs = [z for _, z, _ in ids][::nc]           # one per offset
        assert min(zs) >= 0 and max(zs) <= NZ - 1, "window ran off the volume: %s" % zs
        assert zs == sorted(zs), "offsets are not monotonic in z: %s" % zs
