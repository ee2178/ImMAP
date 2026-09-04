"""
Sanity tests for the eval dump and the figure tooling it feeds
(`datasets/fastmri/loader.py` indexing, `evaluation/tasks.py`,
`scripts/dump_eval.py`, `figures/common.py`).

Run with:  python -m tests.test_dump_eval

Everything is synthetic and self-contained: a pair of fastMRI-shaped HDF5 trees
in a temp directory, and a stub network that returns the adjoint. That is
enough, because what is under test is the BOOKKEEPING -- which slice ends up
where, whether identity survives the loader, whether the file the viewer opens
holds what the viewer expects -- and none of it depends on the reconstruction
being any good. `tests/fastmri_fixture.py` covers the real-data questions.

The one thing not covered here is `build_model` / `load_state_dict`, which need
an actual checkpoint.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile

import h5py
import numpy as np
import torch

from datasets.fastmri.loader import FastMRIDataset
from datasets.registry import build_loader
from evaluation.metrics import build_metrics

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import dump_eval as de                                          # noqa: E402

from figures import common as fc                                # noqa: E402

PASS, FAIL = [], []
NC, H, W = 4, 32, 32
SLICES = {"volA.h5": 7, "volB.h5": 5, "volC.h5": 9}


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'ok ' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def build_trees(root):
    """A pair of fastMRI-shaped trees. `image` is (nslice, H, W) -- the loader
    squeezes kspace and smaps but NOT the image, so a spurious channel axis here
    would only show up much later as a 5-D tensor inside a metric."""
    ks, sm = os.path.join(root, "k"), os.path.join(root, "s")
    os.makedirs(ks); os.makedirs(sm)
    rng = np.random.default_rng(0)
    for fn, n in SLICES.items():
        with h5py.File(os.path.join(ks, fn), "w") as f:
            f.attrs["acquisition"] = "AXT2"
            f["kspace"] = np.zeros((n, NC, H, W), np.complex64)
        with h5py.File(os.path.join(sm, fn), "w") as f:
            f["smaps"] = np.ones((n, NC, H, W), np.complex64) / np.sqrt(NC)
            img = rng.normal(0, 1, (n, H, W)).astype(np.float32)
            img[:, :4, :] = img[:, -4:, :] = 0.0       # a support to mask on
            f["image"] = (img + 0j).astype(np.complex64)
            # Slice s is scalable by s, so "did we get the slice we asked for"
            # is answerable from the data itself.
            f["image"][...] = f["image"][:] * (1 + np.arange(n))[:, None, None]
    return ks, sm


class StubNet(torch.nn.Module):
    """A 'reconstruction' that is the adjoint times a learned scalar."""
    pad_stride = 2

    def __init__(self):
        super().__init__()
        self.g = torch.nn.Parameter(torch.tensor(1.05))

    def forward(self, y, E=None, sigma=None):
        return E.adjoint(y) * self.g, None


def base_cfg(ks, sm):
    return {
        "task": "recon",
        "experiment": {"name": "stub"},
        "model": {"type": "StubNet"},
        "data": {"val": dict(name="fastmri", task="recon", anatomy="brain",
                             kspace_root=ks, smap_root=sm, scale_fac=1.0,
                             pad_multiple=2, batch_size=1, start_slice=0,
                             end_slice=1, crop_size=None, center_crop=None,
                             random_flips=False)},
        "training": {"use_organ_mask": True, "noise_dist": "uniform",
                     "val_noise_std": 0.05},
        "mri": {"R": 4, "acs_lines": 8, "mask_dist": "uniform", "mask_offset": 0,
                "kspace_type": "simulated", "whiten_kspace": False},
    }


# ---------------------------------------------------------------------------
def test_indexing(ks, sm):
    common = dict(task="recon", anatomy="brain", kspace_root=ks, smap_root=sm,
                  scale_fac=1.0, pad_multiple=2)

    # The default path must be exactly what training has always done.
    d = FastMRIDataset(**common, start_slice=0, end_slice=1)
    b = d[0]
    check("sampling path: one item per volume", len(d) == len(SLICES), f"{len(d)}")
    check("sampling path: 5-tuple, (NC,H,W) kspace",
          len(b) == 5 and tuple(b[0].shape) == (NC, H, W),
          str([tuple(x.shape) for x in b[:3]]))
    check("sampling path: image keeps its (1,H,W) shape",
          tuple(b[2].shape) == (1, H, W), str(tuple(b[2].shape)))
    try:
        d.item_id(0)
        check("item_id refuses the sampling path", False)
    except RuntimeError:
        check("item_id refuses the sampling path", True)

    d = FastMRIDataset(**common, start_slice=0, end_slice=None,
                       enumerate_slices=True)
    ids = [d.item_id(i) for i in range(len(d))]
    check("enumerate: every slice of every volume",
          len(d) == sum(SLICES.values()), f"{len(d)}")
    check("enumerate: volume-major, ascending",
          ids[0] == ("volA", 0) and ids[7] == ("volB", 0), str(ids[:2] + ids[7:8]))

    d = FastMRIDataset(**common, start_slice=2, end_slice=6, enumerate_slices=True)
    per = {}
    for i in range(len(d)):
        v, s = d.item_id(i)
        per.setdefault(v, []).append(s)
    check("enumerate: range clipped to each volume's own length",
          per == {"volA": [2, 3, 4, 5], "volB": [2, 3, 4], "volC": [2, 3, 4, 5]},
          str(per))

    d = FastMRIDataset(**common, enumerate_slices=True, end_slice=2,
                       volumes=["volC", "volA"])
    check("volumes= honours the caller's order",
          [d.item_id(i) for i in range(len(d))] ==
          [("volC", 0), ("volC", 1), ("volA", 0), ("volA", 1)])
    try:
        FastMRIDataset(**common, volumes=["volZ"])
        check("a missing volume is named, not silently dropped", False)
    except ValueError as e:
        check("a missing volume is named, not silently dropped", "volZ" in str(e))

    # The item must BE the slice item_id claims it is.
    d = FastMRIDataset(**common, enumerate_slices=True, start_slice=3, end_slice=5)
    with h5py.File(os.path.join(sm, "volA.h5")) as f:
        want = np.abs(f["image"][3]).max()
    check("the item is the slice item_id names",
          d.item_id(0) == ("volA", 3)
          and abs(float(d[0][2].abs().max()) - float(want)) < 1e-4)

    # The registry drops any key the loader function does not name.
    ld = build_loader(dict(name="fastmri", **common, enumerate_slices=True,
                           start_slice=0, end_slice=2, volumes=["volB"],
                           batch_size=1, num_workers=0),
                      shuffle=False, drop_last=False)
    check("build_loader passes the new keys through",
          len(ld.dataset) == 2 and ld.dataset.item_id(1) == ("volB", 1))
    check("shuffle=False keeps loader order == index order",
          [ld.dataset.item_id(i) for i, _ in enumerate(ld)] ==
          [("volB", 0), ("volB", 1)])


def test_helpers(ks, sm):
    check("parse_slices", de.parse_slices("4:20") == (4, 20)
          and de.parse_slices("8") == (0, 8)
          and de.parse_slices("") == (0, None)
          and de.parse_slices("6:") == (6, None))
    check("out_dir: default is beside the checkpoint",
          de.out_dir(os.path.join("a", "b"), "a", None).replace("\\", "/")
          == "a/b/eval_dump")
    check("out_dir: --out mirrors the run tree",
          de.out_dir(os.path.join("a", "b"), "a", "o").replace("\\", "/") == "o/b")
    check("same_source ignores batch-shaped keys",
          de.same_source({"name": "fastmri", "anatomy": "brain"},
                         {"name": "fastmri", "anatomy": "brain", "batch_size": 4}))
    check("same_source catches a different anatomy",
          not de.same_source({"anatomy": "brain"}, {"anatomy": "knee"}))

    cfg = base_cfg(ks, sm)
    check("resolve_volumes sorts, so every run gets the same list",
          de.resolve_volumes(cfg["data"]["val"], None, 2) == ["volA", "volB"])
    check("resolve_volumes passes an explicit list through",
          de.resolve_volumes(cfg["data"]["val"], ["volC"], 2) == ["volC"])

    check("_np squeezes batch and channel",
          de._np(torch.ones(1, 1, 5, 5, dtype=torch.complex64)).shape == (5, 5)
          and de._np(torch.ones(1, 3, 5, 5)).shape == (5, 5))


def test_dump_run(ks, sm, tmp):
    cfg = base_cfg(ks, sm)
    metrics = build_metrics(["psnr", "ssim", "nrmse"])
    args = argparse.Namespace(slices="2:6", workers=0, seed=1234, limit=None)
    dest = os.path.join(tmp, "dumped", "stub", "eval_dump")

    n = de.dump_run("trained_nets/stub", cfg, StubNet(), 1, 10, args,
                    ["volA", "volC"], metrics, 0.05, dest)
    check("one file per volume, all slices in range", n == 8
          and sorted(os.listdir(dest)) == ["volA.h5", "volC.h5"], f"n={n}")

    with h5py.File(os.path.join(dest, "volA.h5")) as f:
        want = ("reference", "recon", "zero_filled", "organ_mask",
                "sampling_mask", "slice_index",
                "psnr_slice", "ssim_slice", "nrmse_slice")
        check("every documented dataset is present",
              all(k in f for k in want), str([k for k in want if k not in f]))
        check("stacks are (nslice, H, W) with no transpose needed",
              f["reference"].shape == (4, H, W), str(f["reference"].shape))
        check("images are complex", np.iscomplexobj(f["reference"][:]))
        check("slice_index records the SOURCE slice numbers",
              list(f["slice_index"][:]) == [2, 3, 4, 5])
        check("masks are 0/1", set(np.unique(f["organ_mask"][:])) <= {0, 1})
        check("metrics are finite, one per slice",
              np.isfinite(f["psnr_slice"][:]).all()
              and f["psnr_slice"].shape == (4,))
        check("provenance is in the attrs",
              f.attrs["sigma"] == 0.05 and f.attrs["R"] == 4
              and f.attrs["volume"] == "volA" and f.attrs["seed"] == 1234
              and bool(f.attrs["use_organ_mask"])
              and f.attrs["dump_version"] == de.DUMP_VERSION)

    # Same seed, same noise: this is the property that lets two runs be
    # compared pixel for pixel rather than just on average.
    dest2 = os.path.join(tmp, "dumped", "stub2", "eval_dump")
    de.dump_run("trained_nets/stub", cfg, StubNet(), 1, 10, args,
                ["volA", "volC"], metrics, 0.05, dest2)
    with h5py.File(os.path.join(dest, "volA.h5")) as a, \
            h5py.File(os.path.join(dest2, "volA.h5")) as b:
        check("same seed -> identical measurement and output",
              np.array_equal(a["recon"][:], b["recon"][:])
              and np.array_equal(a["zero_filled"][:], b["zero_filled"][:]))

    args3 = argparse.Namespace(slices="2:6", workers=0, seed=1234, limit=3)
    dest3 = os.path.join(tmp, "dumped", "stub3", "eval_dump")
    n3 = de.dump_run("trained_nets/stub", cfg, StubNet(), 1, 10, args3,
                     ["volA", "volC"], metrics, 0.05, dest3)
    with h5py.File(os.path.join(dest3, "volA.h5")) as f:
        short = f["recon"].shape[0] == 3 and list(f["slice_index"][:]) == [2, 3, 4]
    check("--limit flushes the partial volume rather than losing it",
          n3 == 3 and os.listdir(dest3) == ["volA.h5"] and short)

    return os.path.join(tmp, "dumped")


def test_figures(dump_root, tmp):
    fc.DUMP_ROOT = dump_root
    fc.COLUMNS = [("Zero-filled", "@zero_filled"), ("Stub", "stub"),
                  ("Stub2", "stub2"), ("Ground truth", None)]
    fc.NAME = fc.VARIANT = "roundtrip"
    fc.HERE = tmp                     # keep picks out of the repo

    check("volumes() is the intersection across columns",
          fc.volumes() == ["volA", "volC"], str(fc.volumes()))

    v = fc.load("stub", "volA")
    check("Volume exposes the stack, the metrics and the attrs",
          len(v) == 4 and sorted(v.metrics) == ["nrmse", "psnr", "ssim"]
          and v.attrs["run"] == "stub")
    check("at() maps a source slice number to a stack position",
          v.at(4) == 2 and v.at(99) is None)
    check("a shared dataset resolves to the first named column",
          fc.load("@zero_filled", "volA").img.shape == v.img.shape)
    check("check_comparable is quiet when the columns agree",
          fc.check_comparable("volA") == [], str(fc.check_comparable("volA")))

    ref, organ = v.ref[1], v.organ[1]
    vmax = fc.window(ref, organ)
    c0, r0, size = fc.fov_box(ref)
    check("window is positive and finite", np.isfinite(vmax) and vmax > 0,
          f"{vmax:.3f}")
    check("fov_box stays inside the frame",
          0 <= c0 and 0 <= r0 and c0 + size <= ref.shape[1]
          and r0 + size <= ref.shape[0], f"({c0},{r0},{size})")

    cols = {lab: fc.load(spec, "volA").img[1]
            for lab, spec in fc.COLUMNS if fc.col_spec(spec)[0]}
    dis = fc.disagreement(ref, cols)
    check("disagreement is a non-negative spread",
          dis.shape == ref.shape and (dis >= 0).all())
    z = fc.auto_zoom(ref, dis, size)
    check("auto_zoom returns an in-frame corner",
          0 <= z[0] <= size and 0 <= z[1] <= size, str(z))

    u8 = fc.to8(ref, vmax)
    check("to8 is uint8 and saturates at the window",
          u8.dtype == np.uint8 and u8.max() <= 255)
    check("apply_cmap gives RGB on both matplotlib generations",
          fc.apply_cmap(dis / (dis.max() + 1e-12), "inferno").shape
          == ref.shape + (3,))

    # Two columns at different sigmas must not silently sit side by side.
    p = os.path.join(dump_root, "stub2", "eval_dump", "volA.h5")
    with h5py.File(p, "r+") as f:
        f.attrs["sigma"] = 0.09
    fc._CACHE.clear()
    w = fc.check_comparable("volA")
    check("check_comparable catches a sigma mismatch",
          any("sigma" in x for x in w), str(w))

    rows = [dict(volume="volA", slice=3)]
    fc.save_rows(rows)
    fc.save_zooms({fc.row_tag(rows[0]): (5, 7)})
    check("rows and zooms round-trip through their own files",
          fc.active_rows() == rows and fc.load_zooms() == {"volA_s3": (5, 7)})


if __name__ == "__main__":
    tmp = tempfile.mkdtemp(prefix="immap_dump_test_")
    try:
        ks, sm = build_trees(tmp)
        for name, fn, a in (("indexing", test_indexing, (ks, sm)),
                            ("helpers", test_helpers, (ks, sm))):
            print(f"\n--- {name} ---")
            fn(*a)
        print("\n--- dump_run ---")
        dump_root = test_dump_run(ks, sm, tmp)
        print("\n--- figures ---")
        test_figures(dump_root, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("failed:", FAIL)
    sys.exit(1 if FAIL else 0)
