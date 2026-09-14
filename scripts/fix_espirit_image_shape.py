#!/usr/bin/env python3
"""
Rewrite `image` from (S, 1, H, W) to (S, H, W) in already-written ESPIRiT files.

`scripts/make_espirit_smaps.py` used to store the ground truth with a singleton
channel axis. Every reader of these files -- the dataset loader, the eval
helpers in datasets/fastmri/common.py, the RAM scripts -- indexes `image` as
(S, H, W), and the extra axis ends up as a stray dimension in simulated k-space
(see datasets/fastmri/loader.py::_as_image_slice). The writer is fixed; this
converts the files it wrote before that.

Only `image` is touched; `smaps` are copied nowhere and not read. Files already
(S, H, W) are skipped, so it is safe to rerun, and safe to run while an ESPIRiT
job is still writing (unfinished `.partial` files are ignored; anything written
after the fix is already in the new layout).

Crash-safe per file: the new dataset is written as `image_fixed`, then `image`
is deleted and `image_fixed` renamed. A file interrupted in between is detected
and completed on the next run.

Usage
-----
    python scripts/fix_espirit_image_shape.py DIR [DIR ...]            # dry run
    python scripts/fix_espirit_image_shape.py DIR [DIR ...] --apply
"""

import argparse
import glob
import os
import sys

import h5py


def fix_file(path, apply):
    """Return a one-word status for `path`."""
    with h5py.File(path, "r+" if apply else "r") as f:
        if "image_fixed" in f:                      # interrupted earlier run
            if not apply:
                return "resume"
            if "image" in f:
                del f["image"]
            f.move("image_fixed", "image")
            return "resumed"
        if "image" not in f:
            return "no-image"
        shape = f["image"].shape
        if len(shape) == 3:
            return "ok"
        if len(shape) != 4 or shape[1] != 1:
            return f"unexpected{tuple(shape)}"
        if not apply:
            return "would-fix"
        f.create_dataset("image_fixed", data=f["image"][:, 0])
        del f["image"]
        f.move("image_fixed", "image")
        return "fixed"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dirs", nargs="+", help="directories of ESPIRiT .h5 files")
    p.add_argument("--apply", action="store_true",
                   help="rewrite the files (default: report only)")
    args = p.parse_args()

    counts, odd = {}, []
    for d in args.dirs:
        files = sorted(glob.glob(os.path.join(d, "*.h5")))
        if not files:
            print(f"  (no .h5 in {d})")
        for path in files:
            try:
                status = fix_file(path, args.apply)
            except OSError as e:                    # e.g. locked by a writer
                status = f"error:{e}"
            counts[status] = counts.get(status, 0) + 1
            if status not in ("ok", "fixed", "would-fix"):
                odd.append((path, status))

    for path, status in odd:
        print(f"  {status:14s} {path}")
    print("  " + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    if not args.apply and counts.get("would-fix", 0) + counts.get("resume", 0):
        print("  dry run -- rerun with --apply to rewrite")
    return 1 if any(s.startswith(("unexpected", "error")) for _, s in odd) else 0


if __name__ == "__main__":
    sys.exit(main())
