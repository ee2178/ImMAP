#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Register a patient's NYUMets studies to one another, on h5s that are ALREADY BUILT.

    # survey a random sample of 40 patients; writes nothing        (the default)
    python -m scripts.register_nyumets --root ../datasets/NYUMets_h5

    # the whole cohort, still read-only
    python -m scripts.register_nyumets --root ../datasets/NYUMets_h5 --sample 0

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

COST. Report mode reads ONE contrast of `img_median_mad` per study and never opens `img_raw`,
`mask` or `support`, so a survey is roughly an eighth of the bytes a full pass would move; the
pixels for `--apply` are read per patient and dropped again. It is still far too heavy for a
login node, which kills compute -- run it in a cpu_short job (see bigpurple/nyumets_h5.sbatch
for the conda preamble) or an interactive allocation. `--sample` defaults to 40 patients so
that a survey stays a survey.
"""

import argparse
import glob
import os
import shutil

import h5py
import numpy as np
import torch

from preprocessing.nyumets_h5 import (
    CONTRASTS, MODE, lowpass_inplane, register_patient, translate, translation_offset,
)
from preprocessing.cmap import norm_key

DATASETS = ("img_raw", norm_key(MODE), "mask", "support")


class Study:
    """One built h5. Opening reads only the attrs and `slice_index`; the pixels come later.

    Loading is deferred because report mode never needs most of them: the offset and the
    residual are computed from ONE contrast of `img_median_mad`, while `img_raw`, `mask` and
    `support` are only needed to write a registered copy. Reading all four at full depth for
    every study of every patient is what gets the job killed.
    """

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
            self.hw = tuple(f[norm_key(MODE)].shape[1:3])
        self.arrays = {}
        self.patient = str(self.attrs.get("patient", os.path.basename(path).split("_")[0]))
        self.session = str(self.attrs.get("session", ""))
        self.acquired = np.zeros(self.depth, dtype=bool)
        self.acquired[self.index] = True

    @property
    def case(self):
        return f"{self.patient}_{self.session}" if self.session else self.patient

    def _on_z(self, a):
        """(n, ...) stored slices -> (orig_depth, ...) on the original z axis, gaps zero."""
        full = np.zeros((self.depth,) + a.shape[1:], dtype=a.dtype)
        full[self.index] = a
        return full

    def planes(self, ci):
        """(D, H, W) float32 of ONE normalised contrast -- all report mode reads.

        h5py slices in the file, so only that channel leaves disk: a quarter of the bytes of
        `img_median_mad`, and `img_raw` / `mask` / `support` are never touched at all.
        """
        with h5py.File(self.path, "r") as f:
            a = np.asarray(f[norm_key(MODE)][:, :, :, ci], dtype=np.float32)
        return torch.from_numpy(self._on_z(a))

    def load(self):
        """Read every dataset at full depth. Only `--apply` needs this."""
        if self.arrays:
            return self
        with h5py.File(self.path, "r") as f:
            for name in DATASETS:
                if name in f:
                    self.arrays[name] = self._on_z(np.asarray(f[name]))
        return self

    def unload(self):
        self.arrays = {}


def _volume(st):
    """A loaded Study as a `Volume`, for `register_patient` (which shifts raw/norm/fg/fov).

    `--apply` only. Call `st.load()` first; report mode never builds one of these.
    """
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
    """-> {patient: [Study, ...]}, every */*_img.h5 under `root`. Reads attrs only."""
    paths = sorted(glob.glob(os.path.join(root, "*", "*_img.h5")))
    if not paths:
        raise RuntimeError(f"no */*_img.h5 under {root}")
    out = {}
    for p in paths:
        st = Study(p)
        out.setdefault(st.patient, []).append(st)
    return out


def measure(planes, acq, cfg, ref):
    """-> offsets, from the reg-contrast planes alone. No pixels are moved.

    The measuring half of `preprocessing.nyumets_h5.register_patient`, on one contrast instead
    of four whole volumes -- same `lowpass_inplane`, same `translation_offset`, same
    --reg-max-shift rejection, so report mode and --apply cannot disagree (tests pin that they
    return the same offsets). Volumes are zero-padded at the END to the deepest study first,
    exactly as register_patient does, so index 0 stays original slice 0 for all of them.
    """
    depth = max(p.shape[0] for p in planes)
    dev = torch.device(getattr(cfg, "reg_device", None) or "cpu")

    def probe(q):
        if q.shape[0] < depth:
            q = torch.cat([q, torch.zeros((depth - q.shape[0],) + q.shape[1:])])
        return lowpass_inplane(q.to(dev), cfg.reg_lowpass)

    fixed = probe(planes[ref])
    offsets = []
    for i, q in enumerate(planes):
        if i == ref:
            offsets.append((0, 0, 0))
            continue
        t = translation_offset(fixed, probe(q))
        if cfg.reg_max_shift and max(abs(c) for c in t) > cfg.reg_max_shift:
            print(f"  ?? offset {t} exceeds --reg-max-shift {cfg.reg_max_shift}; "
                  f"leaving this study unshifted")
            t = (0, 0, 0)
        offsets.append(t)
    return offsets


def residual(planes, offsets, idx, ref):
    """Worst disagreement with the reference over `idx`, before and after the shift."""
    depth = max(p.shape[0] for p in planes)
    pad = lambda q: (q if q.shape[0] >= depth else
                     torch.cat([q, torch.zeros((depth - q.shape[0],) + q.shape[1:])]))
    pre = [pad(q) for q in planes]
    post = [translate(q, t) if any(t) else q for q, t in zip(pre, offsets)]

    def worst(ps):
        r = ps[ref][idx]
        den = max(float(r.abs().mean()), 1e-8)
        return max(float((ps[i][idx] - r).abs().mean()) / den
                   for i in range(len(ps)) if i != ref)
    return worst(pre), worst(post)


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
    ap.add_argument("--device", default=None, dest="reg_device",
                    help="where the FFTs run: cuda, cuda:0, cpu. Default: cuda when one is "
                         "visible. Only the low-passed probe moves, so GPU memory scales with "
                         "ONE contrast, not four whole volumes")
    ap.add_argument("--resid-warn", type=float, default=0.15, dest="resid_warn",
                    help="flag a patient whose studies still disagree by more than this "
                         "(mean |other - ref| / mean |ref| on the common slices) AFTER the "
                         "shift -- usually too little overlap to register a built set")
    ap.add_argument("--sample", type=int, default=40,
                    help="survey this many RANDOMLY CHOSEN patients (0 = every patient). A "
                         "random sample, not the first N: patient directories sort by ID and "
                         "the head of that list is not a random slice of the cohort. 40 pins "
                         "the offset distribution well enough to decide whether to rebuild")
    ap.add_argument("--seed", type=int, default=0,
                    help="which sample. Change it to check the survey was not a fluke")
    ap.add_argument("--multi-only", action="store_true", dest="multi_only",
                    help="sample only from patients that HAVE more than one study -- the only "
                         "ones registration can say anything about")
    ap.add_argument("--allow-partial", action="store_true", dest="allow_partial",
                    help="permit --apply on a sample. Off by default: it would write a set in "
                         "which only some patients are registered")
    cfg = ap.parse_args()
    cfg.register = True
    if cfg.reg_device is None:
        cfg.reg_device = "cuda" if torch.cuda.is_available() else "cpu"
    if cfg.reg_device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(f"--device {cfg.reg_device}, but torch sees no CUDA device")

    out_root = cfg.root if cfg.in_place else (cfg.out or cfg.root.rstrip("/\\") + "_reg")
    if cfg.apply and os.path.abspath(out_root) == os.path.abspath(cfg.root) and not cfg.in_place:
        raise SystemExit("--out is --root; pass --in-place if that is really the intent")

    if cfg.apply and cfg.sample and not cfg.allow_partial:
        raise SystemExit(
            f"--apply with --sample {cfg.sample} would write a set in which only {cfg.sample} "
            f"patients are registered and the rest are not -- one that looks complete and is "
            f"not. Use --sample 0 for every patient, or --allow-partial if that really is the "
            f"intent.")

    patients = index_patients(cfg.root)
    n_all = len(patients)
    n_multi = sum(1 for v in patients.values() if len(v) > 1)
    pool = [k for k, v in patients.items() if len(v) > 1] if cfg.multi_only else list(patients)
    sampled = bool(cfg.sample) and cfg.sample < len(pool)
    if sampled:
        rng = np.random.default_rng(cfg.seed)
        pick = sorted(rng.choice(len(pool), size=cfg.sample, replace=False).tolist())
        pool = [pool[i] for i in pick]
    patients = {k: patients[k] for k in pool}
    how = (f"random sample, seed {cfg.seed}" + (", multi-study only" if cfg.multi_only else "")
           if sampled else "all of them")
    print(f"{n_all} patient(s) under {cfg.root}, {n_multi} with more than one study")
    print(f"surveying {len(patients)} patient(s), "
          f"{sum(len(v) for v in patients.values())} studies  ({how})")
    print(f"mode: {'APPLY -> ' + out_root if cfg.apply else 'REPORT ONLY (no writes)'}")
    print(f"offset on the low-passed normalised {cfg.reg_contrast} "
          f"(lowpass={cfg.reg_lowpass}, max shift {cfg.reg_max_shift}) on {cfg.reg_device}\n")

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

        ref = max(range(len(studies)), key=lambda i: (int(acq[i].sum()), -i))
        ci = CONTRASTS.index(cfg.reg_contrast)

        # MEASURE from one contrast. Report mode stops here and never touches img_raw / mask /
        # support, which is what makes a whole-cohort survey affordable.
        planes = [st.planes(ci) for st in studies]
        offsets = measure(planes, acq, cfg, ref) if len(studies) > 1 else [(0, 0, 0)]

        # The common slice set needs each study's acquired extent carried through its own shift.
        valid = [torch.as_tensor(np.pad(a, (0, depth - a.shape[0])) if a.shape[0] < depth else a)
                 for a in acq]
        valid = [translate(v, (t[0],)).numpy() if any(t) else np.asarray(v)
                 for v, t in zip(valid, offsets)]
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
        resid = residual(planes, offsets, idx, ref) if len(studies) > 1 else None
        del planes

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
            # Only now are the full volumes read, and only for a patient that is being
            # written. register_patient RE-MEASURES from them; `measure` above used the same
            # primitives on the same planes, so the offsets agree (tests/test pins it).
            vols = [_volume(st.load()) for st in studies]
            if len(studies) > 1:
                register_patient(vols, cfg, ref=ref, real=acq)
            for i, st in enumerate(studies):
                rel = os.path.relpath(st.path, cfg.root)
                write_registered(st, vols[i], idx, offsets[i], studies[ref].case,
                                 os.path.join(out_root, rel), cfg)
                st.unload()
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
        print("\nREPORT ONLY -- nothing was written."
              + (f" This was {len(patients)} of {n_multi} multi-study patients; --sample 0 "
                 f"does all of them, and a different --seed draws a different sample."
                 if sampled else "")
              + " To write, add --apply --sample 0. To do better, rebuild with "
                "preprocessing/nyumets_h5.py, which registers on the full volumes.")


if __name__ == "__main__":
    main()
