"""
scripts/alias_check.py: residual aliasing, with zero-filled as the template.

Run with `python -m tests.test_alias_check`. CPU, a few seconds; no torch.

1. hp_pe removes what is constant along phase-encode only.
2. local_corr: 1 for e == t; an unrelated pair stays near 0.
3. null_offsets lie halfway between the aliasing shifts.
4. check_slice: a ghost in both zero-filled and the residual is found where it is;
   an under-reconstructed edge and a clean recon are not; a line the zero-filled
   image does not have is not aliasing.
"""

from __future__ import annotations

import csv
import os
import sys
import tempfile
from argparse import Namespace

import h5py
import numpy as np
from scipy import ndimage as ndi

from scripts import alias_check as ac

FAIL = []
A = Namespace(win_rows=25, win_cols=9, rho=0.8, margin=0.2, z=3.0, min_px=8,
              min_length=10.0, min_aspect=3.0)
N = 192


def check(name, ok, detail=""):
    print(f"[{'ok ' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def vline(c, amp, r0=60, r1=130):
    x = np.zeros((N, N))
    x[r0:r1, c:c + 2] = amp
    return x


RNG = np.random.default_rng(0)
REF = 1.0 + 0.3 * ndi.gaussian_filter(RNG.standard_normal((N, N)), 12) * 12 + vline(50, 0.8, 30, 160)
REF = REF + 0.02 * RNG.standard_normal((N, N))
ZF = ndi.gaussian_filter(REF, (0, 3)) + 0.02 * RNG.standard_normal((N, N))
INNER = ndi.binary_erosion(np.ones((N, N), bool), iterations=4)
MASK = np.zeros(N, np.uint8)
MASK[::11] = 1
MASK[N // 2 - 4:N // 2 + 4] = 1


def blobs(zf, rec):
    return ac.check_slice(REF, zf, rec, INNER, ac.null_offsets(MASK), A)[3]


def hit(bl, r, c, d=12):
    return any(abs(b[2] - r) <= d and abs(b[3] - c) <= d for b in bl)


def test_helpers():
    x = np.tile(np.arange(N, dtype=float) % 2, (N, 1))
    check("hp_pe keeps variation along PE", np.abs(ac.hp_pe(x)).max() > 0.1)
    check("hp_pe removes what is flat along PE",
          np.abs(ac.hp_pe(np.tile(np.arange(N, dtype=float)[:, None], (1, N)))).max() < 1e-9)
    r = np.random.default_rng(1)
    a, b = (ndi.gaussian_filter(r.standard_normal((N, N)), 1) for _ in range(2))
    win = (25, 9)
    check("corr(e, e) = 1", np.median(ac.local_corr(a, a, win)) > 0.99)
    check("unrelated pair near 0", abs(np.median(ac.local_corr(a, b, win))) < 0.1)
    off = ac.null_offsets(MASK)
    _, step = ac.ghost_period(MASK)
    check("null offsets are halfway between aliasing shifts",
          all(abs(abs(s) / step % 1 - 0.5) < 0.05 for s in off), str(off))


def test_check_slice():
    ghost = vline(120, 0.5)
    check("a ghost in zero-filled and the residual is found",
          hit(blobs(ZF + ghost, REF + 0.4 * ghost), 95, 120))
    check("a ghost the net removed is not", not hit(blobs(ZF + ghost, REF), 95, 120))
    check("a line zero-filled does not have is not aliasing",
          not hit(blobs(ZF, REF + 0.4 * ghost), 95, 120))
    weak = ndi.gaussian_filter(REF, (0, 1.5))
    check("an under-reconstructed edge is not aliasing", not hit(blobs(ZF, weak), 95, 50))


def test_flag_rho_by_r():
    check("parse_by_r", ac.parse_by_r("8:0.85,12:0.80") == {8: 0.85, 12: 0.8} and ac.parse_by_r("") == {})


def test_patches():
    ghost = vline(120, 0.5)
    rho, _, z, _ = ac.check_slice(REF, ZF + ghost, REF + 0.4 * ghost, INNER, ac.null_offsets(MASK), A)
    top = ac.top_patches(rho, z, INNER, A, n=3)
    check("the top patch is on the ghost", abs(top[0][3] - 120) <= 8 and 60 <= top[0][2] <= 130, str(top[0]))
    check("rho comes out sorted", all(a[0] >= b[0] for a, b in zip(top, top[1:])))
    check("one patch per neighbourhood", abs(top[0][2] - top[1][2]) > 25 or abs(top[0][3] - top[1][3]) > 25)
    half = max(A.win_rows, A.win_cols) // 2
    check("patches stay half a window inside the mask",
          all(4 + half <= r < N - 4 - half and 4 + half <= c < N - 4 - half for _, _, r, c in top))


def test_top_slices():
    """Each slice x method counts once, and a volume cannot fill more than its slices."""
    ghost = vline(120, 0.5)
    with tempfile.TemporaryDirectory() as root:
        # the other two nets: only a faint residual, which must not rank however it correlates
        faint = 0.01 * np.random.default_rng(2).standard_normal((N, N))
        for name, rec, zf in (("lpdsnet", REF + 0.4 * ghost, ZF + ghost), ("varnetmaps", REF + faint, ZF + ghost),
                              ("mg81v6", REF + faint, ZF + ghost)):
            d = os.path.join(root, f"{name}_R8", "eval_dump")
            os.makedirs(d)
            with h5py.File(os.path.join(d, "volA.h5"), "w") as h:
                h["recon"] = np.stack([rec, REF + faint]).astype(np.complex64)
                h["reference"] = np.stack([REF, REF]).astype(np.complex64)
                h["zero_filled"] = np.stack([zf, ZF]).astype(np.complex64)
                h["organ_mask"] = np.ones((2, N, N), np.uint8)
                h["sampling_mask"] = np.stack([np.tile(MASK, (N, 1))] * 2)
                h["slice_index"] = np.array([3, 4], np.int32)
                h.attrs["kspace_type"] = "measurement_awgn"
        out = os.path.join(root, "out")
        ac.main(["--runs", root, "--out", out, "--top-slices", "1"])
        with open(os.path.join(out, "slices_R8.csv")) as f:
            rows = list(csv.DictReader(f))
        check("one row per slice x method", len(rows) == 6, str(len(rows)))
        lp = [r for r in rows if r["method"] == "lpdsnet"]
        check("the ranking is by rho, the ghost slice first",
              [r["slice"] for r in lp] == ["3", "4"] and float(lp[0]["rho"]) > 0.8, str(lp[0]))
        crops = [n for n in os.listdir(out) if n.endswith(".png")]
        check("only the ghost slice is flagged",
              [(r["method"], r["slice"]) for r in rows if r["flagged"] == "1"] == [("lpdsnet", "3")])
        check("flags_R8.txt counts it", "lpdsnet" in open(os.path.join(out, "flags_R8.txt")).read())
        check("only the strong residual gets an image", len(crops) == 1 and "lpdsnet" in crops[0], str(crops))
        faint_rows = [r for r in rows if r["method"] != "lpdsnet"]
        check("a slice with no strong place has rho -1", all(r["rho"] == "-1.0" for r in faint_rows),
              str([r["rho"] for r in faint_rows]))


def main():
    for t in (test_helpers, test_check_slice, test_flag_rho_by_r, test_patches, test_top_slices):
        print(f"\n== {t.__name__}")
        t()
    print(f"\n{'FAILED: ' + ', '.join(FAIL) if FAIL else 'all checks passed'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
