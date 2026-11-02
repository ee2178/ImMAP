"""
scripts/alias_patches.py: other nets on the patches an earlier scan flagged.

Run with `python -m tests.test_alias_patches`. CPU, a few seconds; no torch.

1. Only flagged slices get an image, in the earlier scan's location.
2. Each method's slice is looked up in its own dump: a core net dumped whole and a new
   net dumped on the flagged slice alone, from different trees, share one image.
3. A net with the ghost scores a higher rho there than a net without it.
4. A slice some net has no dump for is skipped and reported, not fatal.
5. Each dump's training step and k-space type are reported, and a mix is warned about.
6. A method found in two trees is read from the first.
"""

from __future__ import annotations

import csv
import os
import sys
import tempfile

import h5py
import numpy as np
from scipy import ndimage as ndi

from scripts import alias_patches as ap

FAIL = []
N = 192
RNG = np.random.default_rng(0)
REF = 1.0 + 0.3 * ndi.gaussian_filter(RNG.standard_normal((N, N)), 12) * 12
GHOST = np.zeros((N, N))
GHOST[60:130, 120:122] = 0.5
ZF = ndi.gaussian_filter(REF, (0, 3)) + GHOST + 0.02 * RNG.standard_normal((N, N))
NOISE = 0.01 * np.random.default_rng(2).standard_normal((N, N))


def check(name, ok, detail=""):
    print(f"[{'ok ' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def dump(root, method, vol, rec, step, slices, kspace="measurement_awgn"):
    d = os.path.join(root, f"{method}_R8", "eval_dump")
    os.makedirs(d, exist_ok=True)
    n = len(slices)
    with h5py.File(os.path.join(d, vol + ".h5"), "w") as h:
        h["recon"] = np.stack([rec] * n).astype(np.complex64)
        h["reference"] = np.stack([REF] * n).astype(np.complex64)
        h["zero_filled"] = np.stack([ZF] * n).astype(np.complex64)
        h["organ_mask"] = np.ones((n, N, N), np.uint8)
        h["slice_index"] = np.array(slices, np.int32)
        h.attrs["step"] = step
        h.attrs["kspace_type"] = kspace


def write_flags(path):
    rows = [dict(volume="volA", slice=4, rank=0, rho=0.9, best_method="old", row=95, col=120, flagged=1),
            dict(volume="volB", slice=3, rank=1, rho=0.88, best_method="old", row=95, col=120, flagged=1),
            dict(volume="volC", slice=3, rank=2, rho=0.2, best_method="old", row=95, col=120, flagged=0)]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    with tempfile.TemporaryDirectory() as root:
        flags = os.path.join(root, "slices_R8.csv")
        write_flags(flags)
        core, new = os.path.join(root, "core"), os.path.join(root, "new")
        dump(core, "corenet", "volA", REF + 0.4 * GHOST + NOISE, 150000, slices=(2, 3, 4))
        dump(core, "corenet", "volB", REF + NOISE, 150000, slices=(2, 3, 4))
        dump(new, "newnet", "volA", REF + NOISE, 90000, slices=(4,))     # flagged slice only, no volB
        dump(new, "corenet", "volA", REF + NOISE, 1, slices=(4,))        # must lose to the core tree
        out = os.path.join(root, "out")
        ap.main(["--flags", flags, "--runs", core, new, "--R", "8", "--methods", "corenet", "newnet", "--out", out])
        pngs = sorted(n for n in os.listdir(out) if n.endswith(".png"))
        check("one image, for the flagged slice every net has", pngs == ["patch_R8_000_volA_s4.png"], str(pngs))
        with open(os.path.join(out, "patches_R8.csv")) as f:
            rows = list(csv.DictReader(f))
        check("one row, at the first scan's location",
              len(rows) == 1 and rows[0]["row"] == "95" and rows[0]["col"] == "120", str(rows))
        check("different positions in different dumps, and the first tree wins: the ghost is the core net's",
              float(rows[0]["rho_corenet"]) > 0.5 > float(rows[0]["rho_newnet"]), str(rows[0]))
        txt = open(os.path.join(out, "patches_R8.txt")).read()
        check("the missing slice is reported", "volB" in txt and "no dump for ['newnet']" in txt, txt)
        check("each dump's step is reported", "150000" in txt and "90000" in txt, txt)
        check("one k-space type: no warning", "WARNING" not in txt)

        dump(new, "newnet", "volA", REF + NOISE, 90000, slices=(4,), kspace="simulated")
        ap.main(["--flags", flags, "--runs", core, new, "--R", "8", "--methods", "corenet", "newnet", "--out", out])
        check("mixed k-space types are warned about", "WARNING" in open(os.path.join(out, "patches_R8.txt")).read())
    print(f"\n{'FAILED: ' + ', '.join(FAIL) if FAIL else 'all checks passed'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
