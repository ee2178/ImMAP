#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Register a patient's NYUMets studies to one another, on h5s that are ALREADY BUILT.

    # measure only -- writes nothing, prints the offsets                (default)
    python -m scripts.register_nyumets --root ../datasets/NYUMets_h5

    # same, then write the registered set somewhere new
    python -m scripts.register_nyumets --root ../datasets/NYUMets_h5 \
                                       --out  ../datasets/NYUMets_h5_reg --apply

preprocessing/nyumets_h5.py already registers while it builds, and that is the better path:
it sees the FULL volumes before any slice filtering. This script exists for the two cases that
one does not cover -- checking how much misregistration an existing set has without paying for
a rebuild from NIfTI, and re-registering with different `--reg-*` settings without redoing the
normalisation. Every piece of the algorithm is imported from the builder, so the two cannot
drift apart.

WHAT IT DOES, per patient:
  1. Rebuilds each study's volume on its ORIGINAL z axis, placing stored slices back at their
     `slice_index` and marking everything else unacquired. A built h5 has already been cut to
     its own brain-bearing slices, so this is the closest thing to the full volume that
     survives -- see the accuracy note below.
  2. Aligns every study onto the deepest one by 3-D FFT cross-correlation of the low-passed,
     median/MAD-normalised `--reg-contrast` (`preprocessing.nyumets_h5.register_patient`).
  3. Intersects the acquired ranges and each study's own kept slices into ONE slice set for the
     patient, so every study of a patient ends up with the IDENTICAL `slice_index`.
  4. Rewrites every dataset (img_raw, img_median_mad, mask, support) shifted and truncated, and
     records `reg_offset` / `reg_reference` / `reg_applied` / `reg_rule`.

ACCURACY NOTE. Registering built h5s is strictly weaker than registering at build time: the
slices each study dropped are gone, so the through-plane correlation runs on the surviving
extent. When the studies kept very different ranges, prefer a rebuild. `--min-overlap` refuses
a patient whose studies share too few slices to give a trustworthy dz.

Patients with one study are left alone (nothing to register), and are copied unchanged under
`--apply --out`.
"""

import argparse
import glob
import os
import shutil

import h5py
import numpy as np
import torch

from preprocessing.nyumets_h5 import (
    CONTRASTS, MODE, register_patient, translate,
)
from preprocessing.cmap import norm_key

DATASETS = ("img_raw", norm_key(MODE), "mask", "support")


class Study:
    """One built h5, reconstituted on its original z axis."""

    __slots__ = ("path", "patient", "session", "depth", "index", "acquired", "arrays",
                 "attrs", "hw")

    def __init__(self, path):
        self.path = path
        with h5py.File(path, "r") as f:
            self.attrs = dict(f.attrs)
            self.index = np.asarray(f["slice_index"]).astype(np.int64)
            self.depth = int(f.attrs.get("orig_depth", self.index.max() + 1))
            if self.index.max() >= self.depth:               # older files, or a bad attr
                self.depth = int(self.index.max()) + 1
            self.arrays = {}
            for name in DATASETS:
                if name not in f:
                    continue
                a = np.asarray(f[name])                       # (n, H, W, C)
                full = np.zeros((self.depth,) + a.shape[1:], dtype=a.dtype)
                full[self.index] = a
                self.arrays[name] = full
            self.hw = self.arrays[norm_key(MODE)].shape[1:3]
        self.patient = str(self.attrs.get("patient", os.path.basename(path).split("_")[0]))
        self.session = str(self.attrs.get("session", ""))
        self.acquired = np.zeros(self.depth, dtype=bool)
        self.acquired[self.index] = True

    @property
    def case(self):
        return f"{self.patient}_{self.session}" if self.session else self.patient


def _volume(st):
    """A `Volume`-shaped view of a Study, for `register_patient` (which shifts .raw/.norm/.fg/.fov)."""
    from preprocessing.nyumets_h5 import Volume

    to_dchw = lambda a: torch.from_numpy(np.ascontiguousarray(a.transpose(0, 3, 1, 2)))
    return Volume(raw=to_dchw(st.arrays["img_raw"]),
                  norm=to_dchw(st.arrays[norm_key(MODE)]),
                  fg=torch.from_numpy(st.arrays["mask"][..., 0].astype(np.uint8)),
                  fov=torch.from_numpy(st.arrays.get(
                      "support", st.arrays["mask"])[..., 0].astype(np.uint8)),
                  stats=st.attrs.get("norm_stats"), orig_depth=st.depth,
                  affine=st.attrs.get("affine"), lost={}, native_hw=st.hw)


def index_patients(root):
    """-> {patient: [Study, ...]}, every */*_img.h5 under `root`."""
    paths = sorted(glob.glob(os.path.join(root, "*", "*_img.h5")))
    if not paths:
        raise RuntimeError(f"no */*_img.h5 under {root}")
    out = {}
    for p in paths:
        st = Study(p)
        out.setdefault(st.patient, []).append(st)
    return out


def write_registered(st, vol, idx, offset, reference, out_path, cfg):
    """Copy the source h5, replacing the four arrays with their shifted, truncated selves."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.abspath(out_path) != os.path.abspath(st.path):
        shutil.copyfile(st.path, out_path)
    from_dchw = lambda t: np.ascontiguousarray(t.numpy()[idx].transpose(0, 2, 3, 1))
    new = {"img_raw": from_dchw(vol.raw), norm_key(MODE): from_dchw(vol.norm),
           "mask": vol.fg.numpy()[idx][..., None], "support": vol.fov.numpy()[idx][..., None]}
    with h5py.File(out_path, "a") as f:
        for name, a in new.items():
            if name not in st.arrays:
                continue
            was = f[name]
            comp = was.compression
            del f[name]
            f.create_dataset(name, data=a.astype(st.arrays[name].dtype),
                             chunks=(1,) + a.shape[1:], compression=comp)
        del f["slice_index"]
        f.create_dataset("slice_index", data=idx.astype(np.int32))
        # `img` is a soft link to the normalised array; deleting the target breaks it
        if "img" in f:
            del f["img"]
        f["img"] = h5py.SoftLink("/" + norm_key(MODE))
        f.attrs["reg_applied"] = True
        f.attrs["reg_offset"] = np.asarray(offset, dtype=np.int32)
        f.attrs["reg_reference"] = str(reference)
        f.attrs["reg_rule"] = ("3-D FFT cross-correlation of the low-passed normalised "
                               f"{cfg.reg_contrast} (lowpass={cfg.reg_lowpass}), integer voxel "
                               "translation only, applied POST-BUILD by "
                               "scripts/register_nyumets.py")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="a built NYUMets h5 directory")
    ap.add_argument("--out", default=None,
                    help="write the registered set here (default: alongside --root with a "
                         "_reg suffix). Ignored without --apply")
    ap.add_argument("--apply", action="store_true",
                    help="actually write. Without it this only measures and prints")
    ap.add_argument("--in-place", action="store_true", dest="in_place",
                    help="rewrite --root itself. Destructive; --apply is still required")
    ap.add_argument("--reg-contrast", default="T1", choices=CONTRASTS, dest="reg_contrast")
    ap.add_argument("--reg-lowpass", type=float, default=0.25, dest="reg_lowpass")
    ap.add_argument("--reg-max-shift", type=int, default=40, dest="reg_max_shift")
    ap.add_argument("--min-overlap", type=int, default=8, dest="min_overlap",
                    help="refuse a patient whose studies share fewer acquired slices than "
                         "this -- too little to trust the through-plane offset")
    ap.add_argument("--resid-warn", type=float, default=0.15, dest="resid_warn",
                    help="flag a patient whose studies still disagree by more than this "
                         "(mean |other - ref| / mean |ref| on the common slices) AFTER the "
                         "shift -- usually too little overlap to register a built set")
    ap.add_argument("--limit", type=int, default=None, help="first N patients only")
    cfg = ap.parse_args()
    cfg.register = True

    out_root = cfg.root if cfg.in_place else (cfg.out or cfg.root.rstrip("/\\") + "_reg")
    if cfg.apply and os.path.abspath(out_root) == os.path.abspath(cfg.root) and not cfg.in_place:
        raise SystemExit("--out is --root; pass --in-place if that is really the intent")

    patients = index_patients(cfg.root)
    if cfg.limit:
        patients = dict(list(patients.items())[:cfg.limit])
    multi = sum(1 for v in patients.values() if len(v) > 1)
    print(f"{len(patients)} patient(s), {sum(len(v) for v in patients.values())} studies; "
          f"{multi} patient(s) have more than one study")
    print(f"mode: {'APPLY -> ' + out_root if cfg.apply else 'REPORT ONLY (no writes)'}")
    print(f"offset on the low-passed normalised {cfg.reg_contrast} "
          f"(lowpass={cfg.reg_lowpass}, max shift {cfg.reg_max_shift})\n")

    rows, skipped, resid_rows = [], [], []
    for pid, studies in patients.items():
        studies.sort(key=lambda s: s.case)
        hw = {tuple(s.hw) for s in studies}
        if len(hw) != 1:
            skipped.append((pid, f"studies on different grids {sorted(hw)}"))
            print(f"  !! {pid}: {skipped[-1][1]}")
            continue

        depth = max(s.depth for s in studies)
        acq = [np.pad(s.acquired, (0, depth - s.depth)) for s in studies]
        overlap = int(np.logical_and.reduce(acq).sum()) if len(studies) > 1 else int(acq[0].sum())
        if len(studies) > 1 and overlap < cfg.min_overlap:
            skipped.append((pid, f"only {overlap} shared acquired slices"))
            print(f"  !! {pid}: {skipped[-1][1]} -- left unregistered")
            continue

        vols = [_volume(s) for s in studies]
        ref = max(range(len(studies)), key=lambda i: (int(acq[i].sum()), -i))
        ci = CONTRASTS.index(cfg.reg_contrast)
        # register_patient shifts in place, so keep the BEFORE picture to score against
        pre = [v.norm[:, ci].clone() for v in vols]
        if len(studies) > 1:
            offsets, valid = register_patient(vols, cfg, ref=ref, real=acq)
        else:
            offsets = [(0, 0, 0)]
            valid = [acq[0]]

        common = np.logical_and.reduce(valid)
        idx = np.where(common)[0]
        if idx.size == 0:
            skipped.append((pid, "no slice survived the common range"))
            print(f"  !! {pid}: {skipped[-1][1]}")
            continue

        # RESIDUAL. How much the studies still disagree where they overlap, before vs after.
        # On a post-build set this is the number that says whether the offset can be trusted:
        # each study was already cut to its own brain-bearing slices, so the correlation sees
        # only the surviving extent, and a small overlap gives a poor dz. A residual that
        # barely moves means "rebuild instead", not "they were already aligned".
        resid = None
        if len(studies) > 1:
            def _pad(q):
                if q.shape[0] >= depth:
                    return q
                return torch.cat([q, torch.zeros((depth - q.shape[0],) + q.shape[1:])])

            def _disagree(planes):
                r = planes[ref][idx]
                den = max(float(r.abs().mean()), 1e-8)
                return max(float((planes[i][idx] - r).abs().mean()) / den
                           for i in range(len(planes)) if i != ref)
            resid = (_disagree([_pad(q) for q in pre]),
                     _disagree([v.norm[:, ci] for v in vols]))

        tag = "" if len(studies) > 1 else "   (single study: nothing to register)"
        print(f"  {pid}: {len(studies)} studies, ref {studies[ref].case}, "
              f"{idx.size} common slices {idx.min()}..{idx.max()}{tag}")
        if resid is not None:
            flag = "" if resid[1] <= cfg.resid_warn else "   <-- STILL MISALIGNED"
            print(f"      disagreement vs ref: {resid[0]:.3f} -> {resid[1]:.3f}{flag}")
            resid_rows.append((pid, resid))
        for i, st in enumerate(studies):
            mark = " <- ref" if i == ref else ""
            print(f"      {st.case:<28s} offset (dz={offsets[i][0]:+d}, dh={offsets[i][1]:+d}, "
                  f"dw={offsets[i][2]:+d}){mark}")
            rows.append((pid, st.case, offsets[i], i == ref))
            if cfg.apply:
                rel = os.path.relpath(st.path, cfg.root)
                write_registered(st, vols[i], idx, offsets[i], studies[ref].case,
                                 os.path.join(out_root, rel), cfg)
        del vols

    moved = [r for r in rows if not r[3]]
    if moved:
        import statistics
        print(f"\n{len(moved)} non-reference study(ies):")
        for ax, name in enumerate(("dz (slice)", "dh (R)", "dw (A)")):
            v = [abs(r[2][ax]) for r in moved]
            print(f"  {name:<11} |offset| median {statistics.median(v):5.1f}  max {max(v):3d}  "
                  f"nonzero on {sum(x > 0 for x in v)}/{len(v)}")
        worst = max(moved, key=lambda r: max(abs(c) for c in r[2]))
        print(f"  largest: {worst[1]} -> {worst[2]}")
    if resid_rows:
        import statistics
        worse = [r for r in resid_rows if r[1][1] > cfg.resid_warn]
        print(f"\nresidual disagreement after registration: median "
              f"{statistics.median([r[1][1] for r in resid_rows]):.3f}, "
              f"{len(worse)}/{len(resid_rows)} patient(s) above --resid-warn {cfg.resid_warn}")
        if worse:
            print("  Those studies did not align. Registering a BUILT set only sees the slices "
                  "each study kept, so a small overlap gives a poor dz -- rebuild with "
                  "preprocessing/nyumets_h5.py, which registers on the full volumes. Rotation "
                  "between visits would also look like this.")
            for pid, r in worse[:8]:
                print(f"    {pid}: {r[0]:.3f} -> {r[1]:.3f}")
    if skipped:
        print(f"\n{len(skipped)} patient(s) skipped:")
        for pid, why in skipped[:10]:
            print(f"  {pid}: {why}")
    if not cfg.apply:
        print("\nREPORT ONLY -- nothing was written. Add --apply to write, or rebuild with "
              "preprocessing/nyumets_h5.py, which registers on the full volumes.")


if __name__ == "__main__":
    main()
