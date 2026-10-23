# -*- coding: utf-8 -*-
"""Inference-time registration (datasets/NYUMets/inference_registration.py, loader `register`).

A phantom patient: the prior study is the current one under a KNOWN 3-D transform (rotation,
scale, shift) applied to all four contrasts. Registering from FLAIR / T1 / T2 alone must

  1. bring the prior's CT1 -- never seen by the fit -- onto the current CT1 (prior_to_current),
     and the current CT1 onto the prior's (current_to_prior);
  2. be blind to the current CT1: replacing it with noise changes neither the transform nor the
     guides;
  3. run once per pair, shared by two loaders at different scalings, and not at all on a rerun
     with a cache_dir;
  4. refuse the target contrast among the registration contrasts, and a non-index slice match.
"""
import math
import os
import sys
import types

import h5py
import numpy as np
import pytest
import torch

sys.modules.setdefault("nibabel", types.ModuleType("nibabel"))

import datasets  # noqa: E402,F401
import datasets.NYUMets.inference_registration as IR  # noqa: E402
from datasets.registry import build_loader  # noqa: E402

Z, H = 40, 48
LO, HI = 12, 28                                       # slice_range


def anatomy(seed=0):
    """A head: an ellipsoid with internal blobs, smooth enough to register, asymmetric enough to
    pin rotation."""
    rng = np.random.default_rng(seed)
    z, y, x = np.mgrid[:Z, :H, :H].astype(np.float32)
    head = (((z - 20) / 15) ** 2 + ((y - 24) / 17) ** 2 + ((x - 23) / 14) ** 2) < 1
    a = np.zeros((Z, H, H), np.float32)
    for _ in range(9):
        c = rng.uniform([10, 12, 12], [30, 36, 34])
        s = rng.uniform(2.5, 5.0)
        a += rng.uniform(0.4, 1.0) * np.exp(-(((z - c[0]) ** 2 + (y - c[1]) ** 2
                                               + (x - c[2]) ** 2) / (2 * s * s)))
    return head * (0.4 + a)


def contrasts(a, ct1_noise=None):
    """(Z, H, W) anatomy -> (Z, H, W, 4) FLAIR, T1, CT1, T2 with different contrast mappings."""
    head = (a > 0).astype(np.float32)
    flair = 800 * a ** 2 * head
    t1 = 1000 * a
    ct1 = 1200 * a + 600 * np.clip(a - 0.9, 0, None) if ct1_noise is None else ct1_noise
    t2 = 900 * (1.6 - a) * head
    return np.stack([flair, t1, ct1, t2], -1).astype(np.float32)


def true_M():
    """current index (h, w, z) -> prior index: 5 deg about z, 3% scale, a 2-voxel shift."""
    t = math.radians(5.0)
    A = np.array([[math.cos(t), -math.sin(t), 0], [math.sin(t), math.cos(t), 0], [0, 0, 1]]) * 1.03
    c = np.array([24.0, 23.0, 20.0])
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = A, c - A @ c + np.array([2.0, -1.5, 1.0])
    return M


def write(root, case, img):
    os.makedirs(os.path.join(root, case))
    with h5py.File(os.path.join(root, case, case + "_img.h5"), "w") as f:
        f.create_dataset("img_raw", data=img)
        f.create_dataset("img", data=img)
        f.create_dataset("mask", data=(img[..., 1:2] > 0).astype(np.uint8))
        f.create_dataset("slice_index", data=np.arange(Z, dtype=np.int32))
        med = [np.median(img[..., c][img[..., c] > 0]) for c in range(4)]
        f.attrs["norm_stats"] = np.stack([med, np.ones(4)], 1).astype(np.float32)


def make_root(base, ct1_noise=False):
    """PA_s00 = current; PA_s01 = the prior: PA_s00's contrasts read through the true transform's
    INVERSE, so registering the prior onto the current must recover true_M."""
    cur = contrasts(anatomy(), np.random.default_rng(5).uniform(0, 2000, (Z, H, H)).astype(
        np.float32) if ct1_noise else None)
    # prior(y) = current_clean(Minv y)  <=>  prior(M x) = current_clean(x)
    clean = contrasts(anatomy())
    prior = IR.warp(clean, np.linalg.inv(true_M()), (Z, H, H))
    write(base, "PA_s00", cur)
    write(base, "PA_s01", prior)
    return str(base)


def loader(root, **kw):
    d = dict(name="nyumets_guided", root=root, x0_idx=2, x1_idx=1, cond_idx=[0, 1, 3],
             image_key="img_raw", intensity_norm="volume_scale", guide_mode="other_study",
             guide_idx=[2], guide_window=1, guide_slice="index", deterministic=True,
             mask_source="t1ct1", mask_apply=True, mask_guides="target",
             slice_range=[LO, HI], batch_size=4, num_workers=0)
    d.update(kw)
    return build_loader(d, shuffle=False, drop_last=False)


def items_of(ds, case):
    return [i for i in range(len(ds))
            if os.path.basename(ds.img_paths[int(ds.file_id[i])]).startswith(case)]


def corr(a, b, m):
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    return float((a * b).sum() / (a.norm() * b.norm()))


@pytest.fixture(autouse=True)
def fresh_cache():
    IR._MATRICES.clear()
    yield
    IR._MATRICES.clear()


def guide_vs_target(ds, case):
    """Mean corr of the prior CT1 at offset 0 (plane 1 of 3) with the current CT1, over slices."""
    out = []
    for i in items_of(ds, case):
        s = ds[i]
        m = s["mask"][0] > 0
        out.append(corr(s["guide"][1, 0], s["x0"][0], m))
    return float(np.mean(out))


def test_prior_to_current_aligns_the_unseen_ct1(tmp_path):
    root = make_root(tmp_path / "a")
    before = guide_vs_target(loader(root).dataset, "PA_s00")
    ds = loader(root, register="prior_to_current").dataset
    after = guide_vs_target(ds, "PA_s00")
    info = ds.registration_info(0, 1)
    assert info["fit_ok"] and info["ncc"] > info["ncc_init"]
    assert after > 0.97 and after > before + 0.1, (before, after)


def test_current_to_prior_moves_the_target(tmp_path):
    root = make_root(tmp_path / "b")
    ds = loader(root, register="current_to_prior").dataset
    after = guide_vs_target(ds, "PA_s00")
    assert after > 0.97, after
    # the target itself moved: it is no longer the stored current CT1
    stored = loader(root).dataset
    i = items_of(ds, "PA_s00")[3]
    assert not torch.allclose(ds[i]["x0"], stored[i]["x0"])


def test_the_current_ct1_is_never_used(tmp_path):
    clean, noisy = make_root(tmp_path / "c"), make_root(tmp_path / "d", ct1_noise=True)
    a = loader(clean, register="prior_to_current").dataset
    b = loader(noisy, register="prior_to_current").dataset
    ga, gb = a[items_of(a, "PA_s00")[4]], b[items_of(b, "PA_s00")[4]]
    assert len(IR._MATRICES) == 2                        # one per root: same settings, two keys
    Ms = [v[0] for v in IR._MATRICES.values()]
    assert np.allclose(Ms[0], Ms[1]), "the transform depends on the current CT1"
    # the guide planes are the warped PRIOR, identical whatever the current CT1 holds
    # (the mask differs: t1ct1 is a property of the target, applied after)
    assert torch.allclose(ga["guide"] * ga["mask"] * gb["mask"], gb["guide"] * ga["mask"] * gb["mask"],
                          atol=1e-5)


def test_one_fit_per_pair_shared_and_cached(tmp_path, monkeypatch):
    import preprocessing.nyumets_h5 as NH
    calls = []
    real = NH.fit_affine

    def counted(*a, **k):
        calls.append(1)
        return real(*a, **k)
    monkeypatch.setattr(NH, "fit_affine", counted)
    root = make_root(tmp_path / "e")
    reg = dict(direction="prior_to_current", cache_dir=str(tmp_path / "cache"))
    a = loader(root, register=reg).dataset
    b = loader(root, register=reg, intensity_norm="none", scales=[1000.0] * 4).dataset
    for ds in (a, b):
        for i in items_of(ds, "PA_s00"):
            ds[i]
    assert len(calls) == 1, len(calls)                   # both scalings, every slice: one fit
    assert len(os.listdir(tmp_path / "cache")) == 1
    IR._MATRICES.clear()                                 # a new process
    c = loader(root, register=reg).dataset
    c[items_of(c, "PA_s00")[0]]
    assert len(calls) == 1, "the disk cache was not used"


def test_refusals(tmp_path):
    root = make_root(tmp_path / "f")
    with pytest.raises(ValueError, match="target contrast"):
        loader(root, register={"direction": "prior_to_current", "contrasts": [1, 2]})
    with pytest.raises(ValueError, match="guide_slice 'index'"):
        loader(root, register="prior_to_current", guide_slice="matched")
    with pytest.raises(ValueError, match="direction"):
        loader(root, register="sideways")


# ---- masks inference can make: the current T1, and the registered prior CT1 ----------------
def masks_of(ds, case):
    return [ds[i]["mask"][0] for i in items_of(ds, case)]


def test_t1_mask_ignores_the_current_ct1(tmp_path):
    clean, noisy = make_root(tmp_path / "g"), make_root(tmp_path / "h", ct1_noise=True)
    a = masks_of(loader(clean, mask_source="t1").dataset, "PA_s00")
    b = masks_of(loader(noisy, mask_source="t1").dataset, "PA_s00")
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    # and it IS the T1 support
    ds = loader(clean, mask_source="t1", mask_apply=False).dataset
    s = ds[items_of(ds, "PA_s00")[5]]
    assert torch.equal(s["mask"][0], (s["cond"][1] > 0).float())    # cond = FLAIR, T1, T2


def test_t1_prior_ct1_mask_uses_the_registered_prior(tmp_path):
    clean, noisy = make_root(tmp_path / "i"), make_root(tmp_path / "j", ct1_noise=True)
    kw = dict(mask_source="t1_prior_ct1", register="prior_to_current")
    a = masks_of(loader(clean, **kw).dataset, "PA_s00")
    b = masks_of(loader(noisy, **kw).dataset, "PA_s00")
    assert all(torch.equal(x, y) for x, y in zip(a, b)), "the mask saw the current CT1"
    ds = loader(clean, mask_apply=False, **kw).dataset
    t1 = loader(clean, mask_source="t1", mask_apply=False).dataset
    for i in items_of(ds, "PA_s00")[::4]:
        s = ds[i]
        prior_ct1 = s["guide"][1, 0]                   # the registered prior CT1, offset 0
        want = ((t1[i]["mask"][0] > 0) & (prior_ct1 > 0)).float()
        assert torch.equal(s["mask"][0], want)
    # registered, the prior's head covers the current T1 support; UNregistered, the misaligned
    # prior CT1 cuts it -- why this mask is meant for after registration
    area = lambda ms: sum(float(m.sum()) for m in ms)                     # noqa: E731
    t1_area = area(masks_of(loader(clean, mask_source="t1").dataset, "PA_s00"))
    stored = area(masks_of(loader(clean, mask_source="t1_prior_ct1").dataset, "PA_s00"))
    assert area(a) >= 0.99 * t1_area and stored < 0.97 * t1_area, (area(a), stored, t1_area)


def test_t1_prior_ct1_needs_a_prior(tmp_path):
    root = make_root(tmp_path / "k")
    with pytest.raises(ValueError, match="t1_prior_ct1 needs guide_mode other_study"):
        loader(root, mask_source="t1_prior_ct1", guide_mode="none", guide_window=0)
