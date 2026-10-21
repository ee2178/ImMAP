"""
Masked display in the figure tools (figures/common.py::slice_view, MASK_DISPLAY).

Run with `python -m tests.test_display_mask`. numpy + h5py + Pillow; no torch.

Dumps written by `scripts/dump_eval.py --mask organ` (the default since
2026-10-20) score their metrics inside the organ mask and record
`use_organ_mask`. The viewers and make_figure draw from `slice_view`, so:

1. such dumps are DRAWN masked by default: every column and the reference are
   zero outside the dump's organ_mask, unchanged inside;
2. so are the residual and the disagreement overlay derived from them;
3. the HDF5 data stays raw -- masking is display only (the aliasing scan's
   background measures need the full image);
4. dumps whose metrics were NOT masked are drawn unmasked by default;
5. MASK_DISPLAY = True / False forces it either way, and slice_view(mask=...)
   overrides per call (the viewers' `k` toggle);
6. the Tk-free core and the web viewer follow it: the `k` toggle flips it, and
   the web viewer's panel PNGs are black outside the mask.
"""

from __future__ import annotations

import io
import os
import sys
import tempfile

import h5py
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from figures import common as fc                                   # noqa: E402

FAIL = []
H = W = 64


def check(name, ok, detail=""):
    print(f"[{'ok ' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def write_dumps(root, masked):
    """Two runs + the reference, one volume, 3 slices; organ = a centred disc."""
    yy, xx = np.mgrid[:H, :W]
    organ = ((yy - H / 2) ** 2 + (xx - W / 2) ** 2) < (H / 3) ** 2
    rng = np.random.default_rng(0)
    ref = np.stack([organ * (1.0 + 0.2 * k) for k in range(3)]).astype(np.complex64)
    for run, noise in (("netA_R8", 0.05), ("netB_R8", 0.10)):
        d = os.path.join(root, run, "eval_dump")
        os.makedirs(d, exist_ok=True)
        # background noise OUTSIDE the organ too -- what masking must hide
        rec = (ref + noise * rng.standard_normal(ref.shape)).astype(np.complex64)
        zf = (ref + 0.3 * rng.standard_normal(ref.shape)).astype(np.complex64)
        with h5py.File(os.path.join(d, "vol1.h5"), "w") as f:
            f["recon"], f["reference"], f["zero_filled"] = rec, ref, zf
            f["organ_mask"] = np.stack([organ] * 3).astype(np.uint8)
            f["slice_index"] = np.arange(3, dtype=np.int32)
            f["psnr_slice"] = np.full(3, 30.0, np.float32)
            f.attrs.update(sigma=0.015, R=8, seed=1234, anatomy="brain",
                           use_organ_mask=masked)
    return organ


def config(root, path):
    with open(path, "w") as f:
        f.write(f'DUMP_ROOT = r"{root}"\n'
                'COLUMNS = [("Zero-filled", "@zero_filled"), ("A", "netA_R8"), '
                '("B", "netB_R8"), ("Ground truth", None)]\n'
                'NAME = "masktest"\nCROP_FOV = False\n')


def reset():
    fc._CACHE.clear()
    fc.MASK_DISPLAY = "auto"


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root_m, root_u = os.path.join(tmp, "masked"), os.path.join(tmp, "unmasked")
        organ = write_dumps(root_m, masked=True)
        write_dumps(root_u, masked=False)
        out = organ == 0

        # 1-3 ---------------------------------------------------------------
        cfg = os.path.join(tmp, "cfg_m.py")
        config(root_m, cfg)
        reset()
        fc.apply_config(cfg)
        fc.HERE = tmp                                # picks files go to the temp dir
        v = fc.slice_view("vol1", 1, crop=False)
        cols_out = max(float(np.abs(a[out]).max()) for a in v["cols"].values())
        raw = fc.load("netA_R8", "vol1").img[1]
        check("masked-metric dumps are DRAWN masked by default",
              v["display_masked"] and cols_out == 0.0, f"max |image| outside = {cols_out}")
        check("...unchanged inside the mask",
              np.array_equal(v["cols"]["A"][organ], raw[organ]))
        check("...the reference too", float(np.abs(v["ref"][out]).max()) == 0.0)
        dis = fc.disagreement(v["ref"], v["methods"])
        res = np.abs(v["ref"] - v["cols"]["B"])
        check("residual and disagreement are zero outside the mask",
              float(res[out].max()) == 0.0 and float(dis[out].max()) == 0.0)
        check("the dump data stays raw (display-only masking)",
              float(np.abs(raw[out]).max()) > 0.0)

        # 5: per-call override and MASK_DISPLAY -----------------------------
        vu = fc.slice_view("vol1", 1, crop=False, mask=False)
        check("slice_view(mask=False) shows the full field of view",
              not vu["display_masked"] and float(np.abs(vu["cols"]["A"][out]).max()) > 0)
        fc.MASK_DISPLAY = False
        check("MASK_DISPLAY=False turns it off for masked dumps",
              not fc.slice_view("vol1", 1, crop=False)["display_masked"])

        # 6: the core's `k` toggle and the web viewer -----------------------
        fc.MASK_DISPLAY = "auto"
        from figures.webviewer import WebPicker
        from PIL import Image
        wp = WebPicker(3)
        c = wp.current()
        check("viewer core follows the default", c["display_masked"])
        png = wp.panels(c, wp._image_key(c))[wp.shown_labels().index("A")]
        im = np.asarray(Image.open(io.BytesIO(png)).convert("L"))
        check("web viewer panel PNG is black outside the mask",
              int(im[out].max()) == 0 and int(im[organ].max()) > 0,
              f"max outside {int(im[out].max())}, inside {int(im[organ].max())}")
        k0 = wp._image_key(c)
        wp.do("toggle", {"name": "mask"})
        c2 = wp.current()
        check("the `k` toggle unmasks, and re-keys the cached images",
              not c2["display_masked"] and wp._image_key(c2) != k0)
        wp.do("toggle", {"name": "mask"})
        check("...and masks again", wp.current()["display_masked"])

        # 4 ----------------------------------------------------------------
        cfg_u = os.path.join(tmp, "cfg_u.py")
        config(root_u, cfg_u)
        reset()
        fc.apply_config(cfg_u)
        vu = fc.slice_view("vol1", 1, crop=False)
        check("dumps with UNmasked metrics are drawn unmasked by default",
              not vu["display_masked"] and float(np.abs(vu["cols"]["A"][out]).max()) > 0)
        fc.MASK_DISPLAY = True
        check("MASK_DISPLAY=True masks them anyway",
              fc.slice_view("vol1", 1, crop=False)["display_masked"])
        reset()

    print(f"\n{'FAILED: ' + ', '.join(FAIL) if FAIL else 'all checks passed'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
