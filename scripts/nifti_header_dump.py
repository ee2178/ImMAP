#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dump EVERY NIfTI header field, and sweep a cohort for anything that could carry a timestamp.

Motivation: NYUMets study directories are random 10-digit `image_id`s with no time in them, so
before accepting that the imaging alone cannot be ordered chronologically, rule out the header.

WHERE A DATE COULD POSSIBLY HIDE IN NIfTI-1. The 348-byte header has NO calendar-date field --
ANALYZE 7.5 had `exp_date`/`exp_time`, but NIfTI-1 reused those bytes. What remains is:

  * four free-text fields: descrip (80 chars), aux_file (24), db_name (18), intent_name (16).
    dcm2niix often writes "TE=..;Time=HHMMSS.sss" into `descrip`, which is a time OF DAY (no
    date) and is commonly stripped by de-identification.
  * header EXTENSIONS -- an arbitrary blob after the 348 bytes, keyed by code. Code 2 is a
    DICOM fragment and 6 is plain text/JSON, either of which CAN carry AcquisitionDateTime.
    This is the only place a full timestamp realistically survives, and it is the part a
    `hdr['descrip']` spot check misses entirely.
  * a JSON sidecar next to the file (dcm2niix -b y), which is not in the NIfTI at all.

`toffset` and `slice_duration` are NOT dates: they describe the time axis of a 4D series.

    python scripts/nifti_header_dump.py --root ../datasets/NYUMets/data/imaging/patientId
    python scripts/nifti_header_dump.py --path <one file.nii>          # single-file dump only
"""

import os
import glob
import argparse

import numpy as np
import nibabel as nib

TEXT_FIELDS = ("descrip", "aux_file", "db_name", "intent_name")


def _txt(hdr, key):
    """A NIfTI char field as a clean str ('' when blank)."""
    try:
        raw = bytes(hdr[key])
    except Exception:
        return ""
    return raw.decode("latin-1").replace("\x00", "").strip()


def dump_one(path):
    img = nib.load(path)
    hdr = img.header
    print(f"=== {path} ===")
    print(f"class: {type(img).__name__}   header: {type(hdr).__name__}\n")

    print("--- every header field ---")
    for key, val in hdr.items():
        if key in TEXT_FIELDS:
            shown = repr(_txt(hdr, key))
            extra = "   <-- free text, could hold a timestamp"
        else:
            v = np.asarray(val)
            shown = np.array2string(v, precision=4, max_line_width=90) if v.size > 1 else str(v)
            extra = ""
        print(f"  {key:<16} {shown}{extra}")

    print("\n--- time-ish fields (NOT calendar dates) ---")
    for key in ("toffset", "slice_duration", "slice_start", "slice_end", "slice_code",
                "xyzt_units"):
        try:
            print(f"  {key:<16} {np.asarray(hdr[key])}")
        except Exception:
            pass

    print("\n--- header extensions (the one place a full timestamp could survive) ---")
    exts = getattr(hdr, "extensions", [])
    if not exts:
        print("  NONE -- no extension blocks at all")
    for i, e in enumerate(exts):
        try:
            content = e.get_content()
        except Exception as err:
            content = f"<unreadable: {err}>"
        if isinstance(content, (bytes, bytearray)):
            content = content.decode("latin-1", "replace")
        print(f"  [{i}] code={e.get_code()}  len={len(str(content))}")
        print(f"      {str(content)[:600]}")

    sidecar = path
    for suf in (".nii.gz", ".nii"):
        if sidecar.endswith(suf):
            sidecar = sidecar[: -len(suf)]
            break
    sidecar += ".json"
    print(f"\n--- JSON sidecar ---\n  {sidecar}: "
          f"{'EXISTS' if os.path.exists(sidecar) else 'absent'}")


def sweep(root, limit):
    """One file is not proof. Check across studies for ANY non-empty text or extension."""
    files = sorted(glob.glob(os.path.join(root, "*", "*", "*", "*.nii*")))
    if not files:
        files = sorted(glob.glob(os.path.join(root, "**", "*.nii*"), recursive=True))
    if not files:
        raise SystemExit(f"no .nii under {root}")
    rng = np.random.default_rng(0)
    sel = [files[i] for i in sorted(rng.permutation(len(files))[:limit])]

    seen = {k: {} for k in TEXT_FIELDS}
    n_ext, n_sidecar, bad = 0, 0, 0
    for p in sel:
        try:
            hdr = nib.load(p).header
        except Exception:
            bad += 1
            continue
        for k in TEXT_FIELDS:
            v = _txt(hdr, k)
            seen[k][v] = seen[k].get(v, 0) + 1
        if getattr(hdr, "extensions", []):
            n_ext += 1
        base = p[:-7] if p.endswith(".nii.gz") else p[:-4]
        if os.path.exists(base + ".json"):
            n_sidecar += 1

    print(f"\n\n=== swept {len(sel)} files of {len(files)} under {root} "
          f"({bad} unreadable) ===")
    for k in TEXT_FIELDS:
        vals = sorted(seen[k].items(), key=lambda kv: -kv[1])
        nonempty = sum(n for v, n in vals if v)
        print(f"\n  {k}: {nonempty}/{len(sel)} non-empty")
        for v, n in vals[:6]:
            print(f"      {n:>5}  {v!r}")
    print(f"\n  files with a header extension : {n_ext}/{len(sel)}")
    print(f"  files with a JSON sidecar     : {n_sidecar}/{len(sel)}")
    if nonempty == 0 and n_ext == 0 and n_sidecar == 0:
        print("\n  => the imaging carries NO acquisition timestamp anywhere. Ordering studies"
              "\n     requires the release's CSV tables joined on image_id (time_from_gk_days).")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None, help="patient-ID level; sweeps the cohort")
    ap.add_argument("--path", default=None, help="dump this one file")
    ap.add_argument("--limit", type=int, default=200, help="files to sweep")
    a = ap.parse_args()
    if not a.root and not a.path:
        ap.error("need --root or --path")

    if a.path:
        dump_one(a.path)
    if a.root:
        first = next(iter(sorted(glob.glob(os.path.join(a.root, "**", "*.nii*"),
                                           recursive=True))), None)
        if first and not a.path:
            dump_one(first)
        sweep(a.root, a.limit)


if __name__ == "__main__":
    main()
