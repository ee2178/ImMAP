"""
The size census (scripts/fastmri_sizes.py) and the multi-size timing it feeds
(scripts/time_net.py --sizes / --sizes-from).

Run with `python -m tests.test_fastmri_sizes`.

1. the census, on fastMRI-shaped trees built here: volumes are grouped by
   (H, W, coils); the acquisition filter and the config's slice range are the
   loader's; splits are kept apart and summed; the commonest size comes first;
2. `--sizes` parsing, and the census file read back as weights;
3. the timing itself, on a toy net: every size is timed on its own AND mixed,
   the net grid is the embedded one where the size needs it, and the same
   parameters are still there afterwards (the sweep must not rebuild the net).

CPU only, a few seconds.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import h5py
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import fastmri_sizes as fs                                      # noqa: E402
import time_net as tn                                           # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(f"[{'ok ' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def volume(ks, sm, name, shape, coils, slices, acq="AXT2"):
    H, W = shape
    with h5py.File(os.path.join(ks, name + ".h5"), "w") as f:
        f.attrs["acquisition"] = acq
        f["kspace"] = np.zeros((slices, coils, H, W), np.complex64)
    with h5py.File(os.path.join(sm, name + ".h5"), "w") as f:
        f["smaps"] = np.zeros((slices, coils, H, W), np.complex64)
        f["image"] = np.zeros((slices, H, W), np.complex64)


def trees(root):
    """-> a config whose two splits hold a known mix of sizes."""
    data = {}
    spec = dict(
        train=[("a", (24, 16), 4, 5), ("b", (24, 16), 4, 5), ("c", (24, 16), 4, 2),
               ("d", (32, 20), 3, 4), ("e", (24, 16), 6, 5)],
        val=[("f", (32, 20), 3, 6), ("g", (24, 16), 4, 5)])
    for split, vols in spec.items():
        ks, sm = os.path.join(root, split, "k"), os.path.join(root, split, "s")
        os.makedirs(ks)
        os.makedirs(sm)
        for name, shape, coils, slices in vols:
            volume(ks, sm, name, shape, coils, slices)
        volume(ks, sm, "t1", (40, 40), 8, 5, acq="AXT1")       # filtered out for brain
        data[split] = dict(name="fastmri", task="recon", anatomy="brain", batch_size=1,
                           kspace_root=ks, smap_root=sm, scale_fac=1.0,
                           start_slice=1, end_slice=4)
    return {"data": data}


def test_census(tmp):
    cfg = trees(tmp)
    sizes = fs.census(cfg, ["train", "val"])
    got = {(s["H"], s["W"], s["coils"]): s for s in sizes}
    check("volumes are grouped by (H, W, coils); the other acquisition is not counted",
          set(got) == {(24, 16, 4), (32, 20, 3), (24, 16, 6)}
          and got[(24, 16, 4)]["volumes"] == 4 and got[(32, 20, 3)]["volumes"] == 2
          and got[(24, 16, 6)]["volumes"] == 1,
          ", ".join(f"{k}: {v['volumes']}" for k, v in got.items()))
    # [1, 4) of 5 slices is 3, of 2 slices is 1, of 4 is 3, of 6 is 3
    check("slices are counted inside the config's [start_slice, end_slice)",
          got[(24, 16, 4)]["slices"] == 3 + 3 + 1 + 3 and got[(32, 20, 3)]["slices"] == 3 + 3,
          f"{got[(24, 16, 4)]['slices']}, {got[(32, 20, 3)]['slices']}")
    check("splits are kept apart as well as summed",
          got[(24, 16, 4)]["by_split"]["train"]["volumes"] == 3
          and got[(24, 16, 4)]["by_split"]["val"]["volumes"] == 1
          and "val" not in got[(24, 16, 6)]["by_split"])
    check("the commonest size comes first",
          [(s["H"], s["W"], s["coils"]) for s in sizes]
          == [(24, 16, 4), (32, 20, 3), (24, 16, 6)])
    one = dict(cfg["data"]["train"], end_slice=None)
    alone = fs.scan_split({"data": {"train": one}}, "train")
    check("end_slice=None is the loader's rule: one slice per volume",
          alone[(24, 16, 4)] == dict(volumes=3, slices=3))

    # the script end to end, and the file read back by the timing tool
    path, out = os.path.join(tmp, "cfg.json"), os.path.join(tmp, "sizes.json")
    with open(path, "w") as f:
        json.dump(cfg, f)
    argv = sys.argv
    sys.argv = ["fastmri_sizes.py", "--config", path, "--out", out]
    try:
        fs.main()
    finally:
        sys.argv = argv
    loaded = tn.load_sizes(out)
    check("the census file reads back as sizes weighted by volumes, commonest first",
          [(s["H"], s["W"], s["coils"], s["weight"]) for s in loaded]
          == [(24, 16, 4, 4.0), (32, 20, 3, 2.0), (24, 16, 6, 1.0)]
          and len(tn.load_sizes(out, top=2)) == 2)
    return out


def test_parse():
    got = tn.parse_sizes(["640x320x20", "768X396", "640,372,15"], 16)
    check("--sizes: HxW[xCOILS], coils default to --coils",
          [(s["H"], s["W"], s["coils"], s["weight"]) for s in got]
          == [(640, 320, 20, None), (768, 396, 16, None), (640, 372, 15, None)])
    bad = 0
    for spec in ("640", "640x320x20x1", "640xabc", "0x320"):
        try:
            tn.parse_sizes([spec], 16)
        except SystemExit:
            bad += 1
    check("a malformed size is refused", bad == 4, f"{bad}/4")


def test_timing(tmp, census_path):
    tn._import_repo()
    cfg = {"model": {"type": "MGLPDSNet",
                     "params": dict(M=6, C=1, P=3, s=2, K=[1, [2, 2, 2]], lam0=1e-3, tau0=0.5,
                                    theta0=0.0, alpha0=1.0, is_complex=True,
                                    preproc="kspace", resize_noise=True,
                                    coarse_op="rediscretize")},
           "training": {"complex_conv": "planar"}, "mri": {"R": 4, "acs_lines": 8}}
    path = os.path.join(tmp, "toy_R4.json")
    with open(path, "w") as f:
        json.dump(cfg, f)
    sizes = tn.load_sizes(census_path, top=2) + tn.parse_sizes(["30x16x4"], 4)
    for s in sizes:
        s["weight"] = s["weight"] or 1.0
    row = tn.time_config(path, (24, 16), 4, None, 2, 1, torch.device("cpu"), True, None,
                         sizes=sizes)
    ss = row["sizes"]["sizes"]
    check("every size is timed on its own and mixed, all three columns",
          len(ss) == 3 and all(set(s["own"]) >= {"infer", "forward", "backward"}
                               and set(s["mixed"]) >= {"infer", "forward", "backward"}
                               and s["own"]["forward"]["median"] > 0 for s in ss))
    check("the net grid is the embedded one where the size needs it (stride 8)",
          [tuple(s["grid"]) for s in ss] == [(24, 16), (32, 24), (32, 16)]
          and [s["embedded"] for s in ss] == [False, True, True],
          str([tuple(s["grid"]) for s in ss]))
    check("the main row is still the --size measurement",
          row["forward"]["median"] > 0 and row["embedded"] is False)
    tn.print_sizes([row], True)

    one = tn.time_config(path, (24, 16), 4, None, 1, 1, torch.device("cpu"), False, None,
                         sizes=sizes[:1])
    check("a single size has no mixed arm", one["sizes"]["sizes"][0]["mixed"] is None)
    tn.print_sizes([one], False)
    check("no --sizes, no sweep",
          "sizes" not in tn.time_config(path, (24, 16), 4, None, 1, 1, torch.device("cpu"),
                                        False, None))


def main():
    with tempfile.TemporaryDirectory() as tmp:
        print("\n--- test_census")
        census_path = test_census(tmp)
        print("\n--- test_parse")
        test_parse()
        print("\n--- test_timing")
        test_timing(tmp, census_path)
    print(f"\n{'FAILED: ' + ', '.join(FAIL) if FAIL else 'all checks passed'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
