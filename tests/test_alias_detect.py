"""
scripts/alias_detect.py: new lines in a recon that the reference does not have.

Run with `python -m tests.test_alias_detect`. CPU, a few seconds; no torch.

1. Nothing new -> nothing found: the same image, a blurred one, a reference line
   moved within the tolerance.
2. A new thin line is found where it was added, bright or dark, oblique or not;
   a dark line over a bright reference line counts too.
3. Not lines -> not found: a round blob, a short line.
4. scan_volume on real HDF5 dumps: one row per slice x method, the right method
   flagged, kspace_type recorded, and main() warns when the types are mixed.
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

from scripts import alias_detect as ad

FAIL = []
A = Namespace(scales=[1.0, 1.5, 2.5], tol=5, k=3.0, min_px=8, min_length=10.0, min_aspect=3.0)
N = 192


def check(name, ok, detail=""):
    print(f"[{'ok ' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def add_line(u, r0, c0, r1, c1, amp):
    """A line 2 px wide from (r0, c0) to (r1, c1), added to a copy of u."""
    out = u.copy()
    n = 2 * max(abs(r1 - r0), abs(c1 - c0)) + 1
    for r, c in zip(np.linspace(r0, r1, n).astype(int), np.linspace(c0, c1, n).astype(int)):
        out[r:r + 2, c:c + 2] += amp
    return out


def phantom(seed=0):
    """Smooth tissue, one real bright line at column 40, faint noise."""
    rng = np.random.default_rng(seed)
    u = 1.0 + 0.2 * ndi.gaussian_filter(rng.standard_normal((N, N)), 12) * 12
    u = add_line(u, 30, 40, 160, 40, 0.8)
    return u + 0.02 * rng.standard_normal((N, N))


REF = phantom()
INNER = ndi.binary_erosion(np.ones((N, N), bool), iterations=4)
LINES = ad.reference_lines(REF, INNER, A)


def found(rec):
    return ad.find_blobs(ad.new_line_z(rec, LINES, INNER), A)


def hit(blobs, r, c, d=12):
    return any(abs(b[2] - r) <= d and abs(b[3] - c) <= d for b in blobs)


def test_nothing_new():
    check("the reference itself: nothing", not found(REF))
    check("a blurred reference: nothing", not found(ndi.gaussian_filter(REF, 1.5)))
    # the line itself, not np.roll of the image: a roll's wrap-around seam is a real new line
    moved = add_line(add_line(REF, 30, 40, 160, 40, -0.8), 30, 41, 160, 41, 0.8)
    check("a reference line moved by 1 px (within the tolerance): nothing", not found(moved))


def test_new_lines():
    rec = add_line(REF, 60, 120, 120, 120, 0.5)
    b = found(rec)
    check("a new bright line is found, where it was added", hit(b, 90, 120), str(b[:2]))
    b = found(add_line(REF, 60, 120, 120, 120, -0.5))
    check("a new dark line is found", hit(b, 90, 120), str(b[:2]))
    b = found(add_line(REF, 60, 100, 120, 130, 0.5))
    check("an oblique line is found", hit(b, 90, 115), str(b[:2]))
    b = found(add_line(REF, 50, 40, 130, 40, -1.6))
    check("a dark line over a bright reference line is found", hit(b, 90, 40), str(b[:2]))


def test_not_lines():
    yy, xx = np.mgrid[:N, :N]
    disc = ((yy - 90) ** 2 + (xx - 120) ** 2 <= 8 ** 2).astype(float)
    check("a round blob is not a line", not hit(found(REF + 0.5 * disc), 90, 120))
    check("a 6 px line is too short", not hit(found(add_line(REF, 90, 120, 95, 120, 0.5)), 92, 120))


def write_dump(root, method, vol, recon, kspace_type):
    d = os.path.join(root, f"{method}_R8", "eval_dump")
    os.makedirs(d, exist_ok=True)
    with h5py.File(os.path.join(d, vol + ".h5"), "w") as h:
        h["recon"] = recon[None].astype(np.complex64)
        h["reference"] = REF[None].astype(np.complex64)
        h["zero_filled"] = REF[None].astype(np.complex64)
        h["organ_mask"] = np.ones((1, N, N), np.uint8)
        h["slice_index"] = np.array([7], np.int32)
        h.attrs["kspace_type"] = kspace_type
        h.attrs["anatomy"] = "knee"


def test_end_to_end():
    with tempfile.TemporaryDirectory() as root:
        write_dump(root, "lpdsnet", "volA", add_line(REF, 60, 120, 120, 120, 0.5), "measurement_awgn")
        write_dump(root, "varnetmaps", "volA", REF, "measurement_awgn")
        write_dump(root, "mg81v6", "volA", REF, "simulated")
        dumps = {m: os.path.join(root, f"{m}_R8", "eval_dump")
                 for m in ("lpdsnet", "varnetmaps", "mg81v6")}
        rows, _ = ad.scan_volume(dumps, "volA", list(dumps), A)
        by = {r["method"]: r for r in rows}
        check("one row per method", len(rows) == 3 and rows[0]["slice"] == 7)
        check("only the net with the line scores",
              by["lpdsnet"]["score"] > 0 and by["varnetmaps"]["score"] == 0 == by["mg81v6"]["score"])
        check("kspace_type is recorded per method",
              by["lpdsnet"]["kspace_type"] == "measurement_awgn"
              and by["mg81v6"]["kspace_type"] == "simulated")

        out = os.path.join(root, "out")
        ad.main(["--runs", root, "--out", out, "--png", "1"])
        with open(os.path.join(out, "detect_R8.txt")) as f:
            txt = f.read()
        check("main warns about mixed k-space types", "WARNING: this scan mixes" in txt)
        with open(os.path.join(out, "detect_R8.csv")) as f:
            check("the CSV has kspace_type", "kspace_type" in next(csv.reader(f)))
        check("main renders a crop", any(n.endswith(".png") for n in os.listdir(out)))


def main():
    for t in (test_nothing_new, test_new_lines, test_not_lines, test_end_to_end):
        print(f"\n== {t.__name__}")
        t()
    print(f"\n{'FAILED: ' + ', '.join(FAIL) if FAIL else 'all checks passed'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
