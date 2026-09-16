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
import re
import glob
import argparse

import numpy as np
import nibabel as nib

TEXT_FIELDS = ("descrip", "aux_file", "db_name", "intent_name")

# REDACTION. NYUMets carries an AFNI (code 4) extension whose HISTORY_NOTE preserves the
# PRE-de-identification file paths -- source-side numeric identifiers, a username and a
# hostname -- because the extension was written before the de-id step and passed through it
# untouched. Printing it verbatim would copy those identifiers into terminals and SLURM logs,
# so extension text is redacted before display. Timestamps are kept: they are processing
# times, and they are what this script exists to evaluate.
_REDACT = (
    (re.compile(r"(?:/[\w.\-]+){2,}"), "<path>"),
    (re.compile(r"\b[\w.\-]+@[\w.\-]+"), "<user@host>"),
    (re.compile(r"\b\d{6,}\b"), "<id>"),
)
_AFNI_ATR = re.compile(r'atr_name="([^"]+)"')


def redact(text):
    for rx, sub in _REDACT:
        text = rx.sub(sub, text)
    return text


def _txt(hdr, key):
    """A NIfTI char field as a clean str ('' when blank)."""
    try:
        raw = bytes(hdr[key])
    except Exception:
        return ""
    return raw.decode("latin-1").replace("\x00", "").strip()


def _ext_text(e):
    try:
        content = e.get_content()
    except Exception as err:
        return f"<unreadable: {err}>"
    if isinstance(content, (bytes, bytearray)):
        content = content.decode("latin-1", "replace")
    return str(content)


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
        content = _ext_text(e)
        code = e.get_code()
        print(f"  [{i}] code={code}{'  (AFNI XML attributes)' if code == 4 else ''}"
              f"  len={len(content)}")
        names = _AFNI_ATR.findall(content)
        if names:
            print(f"      AFNI attributes: {', '.join(names)}")
        for m in re.finditer(r'atr_name="(HISTORY_NOTE|IDCODE_DATE)"\s*>\s*(.*?)</AFNI_atr>',
                             content, re.S):
            print(f"      {m.group(1)} (redacted): {redact(m.group(2).strip())[:400]}")
        if not names:
            print(f"      (redacted) {redact(content)[:400]}")

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
    codes, atr_names = {}, {}
    date_like = re.compile(r"(acqui|study.?date|series.?date|content.?date|datetime|scan.?date)",
                           re.I)
    date_hits = {}
    for p in sel:
        try:
            hdr = nib.load(p).header
        except Exception:
            bad += 1
            continue
        for k in TEXT_FIELDS:
            v = _txt(hdr, k)
            seen[k][v] = seen[k].get(v, 0) + 1
        exts = getattr(hdr, "extensions", [])
        if exts:
            n_ext += 1
        for e in exts:
            c = e.get_code()
            codes[c] = codes.get(c, 0) + 1
            text = _ext_text(e)
            for name in set(_AFNI_ATR.findall(text)):
                atr_names[name] = atr_names.get(name, 0) + 1
            # any DICOM-style acquisition/study date key, in ANY extension type
            for m in set(date_like.findall(text)):
                date_hits[m.lower()] = date_hits.get(m.lower(), 0) + 1
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

    print("\n  extension codes (2=DICOM, 4=AFNI, 6=comment/text, 44=JSON-ish):")
    for c, n in sorted(codes.items()):
        print(f"      code {c:<4} x{n}")
    if atr_names:
        print("\n  AFNI attribute names, across the sweep:")
        for name, n in sorted(atr_names.items(), key=lambda kv: (-kv[1], kv[0])):
            flag = "   <-- processing timestamp, NOT acquisition" \
                if name in ("HISTORY_NOTE", "IDCODE_DATE") else ""
            print(f"      {n:>5}  {name}{flag}")

    print("\n  DICOM-style acquisition/study-date keys found in any extension:")
    if date_hits:
        for k, n in sorted(date_hits.items()):
            print(f"      {n:>5}  {k!r}   <-- INSPECT: this could be a real acquisition date")
    else:
        print("      none")

    any_text = any(v for k in TEXT_FIELDS for v in seen[k] if v)
    if not any_text and not date_hits and n_sidecar == 0:
        print("\n  => no ACQUISITION timestamp anywhere: text fields blank, no sidecars, and the"
              "\n     extensions carry only processing dates. Ordering studies still requires the"
              "\n     release's CSV tables joined on image_id (time_from_gk_days).")


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
