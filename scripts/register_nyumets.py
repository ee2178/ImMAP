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
import time

import h5py
import numpy as np
import torch

from preprocessing.nyumets_h5 import (
    CONTRASTS, MODE, affine_offset, lowpass_inplane, register_patient, translate,
    translation_offset,
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

    def planes(self, ci, source="norm"):
        """(D, H, W) float32 of ONE contrast -- all report mode reads.

        h5py slices in the file, so only that channel leaves disk: a quarter of the bytes of
        `img_median_mad`, and `img_raw` / `mask` / `support` are never touched at all.
        """
        key = "img_raw" if source == "raw" else norm_key(MODE)
        with h5py.File(self.path, "r") as f:
            a = np.asarray(f[key][:, :, :, ci], dtype=np.float32)
        return torch.from_numpy(self._on_z(a))

    def masks(self):
        """(D, H, W) bool brain mask on the original z axis. uint8, one channel: cheap."""
        with h5py.File(self.path, "r") as f:
            a = np.asarray(f["mask"][:, :, :, 0])
        return torch.from_numpy(self._on_z(a).astype(bool))

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


def find_h5(root):
    """Every *_img.h5 under `root`, whatever the nesting depth.

    The builder writes <root>/<patient>_<session>/<case>_img.h5, so one level is the normal
    case and the cheap glob is tried first. A split directory of symlinks, or a set arranged
    per patient as <root>/<patient>/<session>/, nests one deeper -- fall back to a recursive
    walk rather than reporting "no files" for a layout that merely differs.
    """
    root = os.path.expanduser(root)
    for pattern in (os.path.join(root, "*", "*_img.h5"),
                    os.path.join(root, "*_img.h5"),
                    os.path.join(root, "**", "*_img.h5")):
        hits = sorted(glob.glob(pattern, recursive=True))
        if hits:
            return hits
    if not os.path.isdir(root):
        raise RuntimeError(
            f"{root} is not a directory (cwd is {os.getcwd()}). In a notebook the cwd is "
            f"usually notebooks/, so a path like '../datasets/...' resolves one level off -- "
            f"give an absolute path, or one relative to the repo root.")
    listing = sorted(os.listdir(root))[:8]
    raise RuntimeError(
        f"no *_img.h5 anywhere under {root}; it contains "
        f"{len(os.listdir(root))} entries, first few: {listing}")


def index_patients(root):
    """-> {patient: [Study, ...]}, every *_img.h5 under `root`. Reads attrs only."""
    out = {}
    for p in find_h5(root):
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
        t = translation_offset(fixed, probe(q), phase=bool(getattr(cfg, "reg_phase", True)))
        if cfg.reg_max_shift and max(abs(c) for c in t) > cfg.reg_max_shift:
            print(f"  ?? offset {t} exceeds --reg-max-shift {cfg.reg_max_shift}; "
                  f"leaving this study unshifted")
            t = (0, 0, 0)
        offsets.append(t)
    return offsets


def residual(planes, offsets, idx, ref, masks=None):
    """Worst disagreement with the reference over `idx`, before and after the shift.

    RESTRICTED TO WHERE BOTH STUDIES HAVE BRAIN when `masks` is given, and that matters more
    than it sounds. `img_median_mad` is zero outside the brain mask, so a whole-frame mean is
    diluted by background and then dominated by the two masks DISAGREEING -- a pair that is
    perfectly aligned but masked slightly differently scores as badly as a pair that is not
    aligned at all. Scoring inside the intersection measures the thing the name claims.

    Even then the floor is not zero on real data: two scans months apart differ by noise,
    scanner state and genuine progression. Read the before -> after change, not the absolute.
    """
    depth = max(p.shape[0] for p in planes)

    def pad(q, fill=0):
        if q.shape[0] >= depth:
            return q
        z = torch.full((depth - q.shape[0],) + q.shape[1:], fill, dtype=q.dtype)
        return torch.cat([q, z])

    def shifted(qs, fill=0):
        return [translate(pad(q, fill), t) if any(t) else pad(q, fill)
                for q, t in zip(qs, offsets)]

    pre, post = [pad(q) for q in planes], shifted(planes)
    if masks is None:
        keep_pre = keep_post = None
    else:
        keep_pre = torch.stack([pad(m, False) for m in masks]).all(0)[idx]
        keep_post = torch.stack(shifted(masks, False)).all(0)[idx]

    def worst(ps, keep):
        r = ps[ref][idx]
        sel = (lambda a: a[keep]) if keep is not None else (lambda a: a)
        rv = sel(r)
        if rv.numel() == 0:
            return float("nan")
        den = max(float(rv.abs().mean()), 1e-8)
        return max(float((sel(ps[i][idx]) - rv).abs().mean()) / den
                   for i in range(len(ps)) if i != ref)
    return worst(pre, keep_pre), worst(post, keep_post)


def needs_write(st, idx, offset):
    """False when this study is already exactly what would be written -- skip the I/O.

    A reference study whose kept slices already equal the patient's common set is the common
    case, and rewriting it means reading and writing a few hundred MB to reproduce the file
    byte for byte.
    """
    return bool(any(offset)) or not np.array_equal(st.index, idx)


def write_registered(st, vol, idx, offset, reference, out_path, cfg):
    """Write the registered study as a FRESH file, then rename it over the target.

    NOT an in-file rewrite. HDF5 does not reclaim the space of a deleted dataset, so
    `del f[name]` + `create_dataset` inside the live file grows it by the size of the arrays on
    every pass -- fatal for `--in-place`, which is meant to be repeatable. A crash halfway
    through would also leave a truncated dataset where the data used to be. Building beside the
    target and `os.replace`-ing is atomic, self-compacting, and costs the same write either way.

    Datasets this does not touch are copied across, so nothing in the file is lost.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    from_dchw = lambda t: np.ascontiguousarray(t.cpu().numpy()[idx].transpose(0, 2, 3, 1))
    new = {"img_raw": from_dchw(vol.raw), norm_key(MODE): from_dchw(vol.norm),
           "mask": vol.fg.cpu().numpy()[idx][..., None],
           "support": vol.fov.cpu().numpy()[idx][..., None]}
    link_name = "/" + norm_key(MODE)
    tmp = out_path + ".tmp"
    try:
        with h5py.File(st.path, "r") as src, h5py.File(tmp, "w") as dst:
            for k, v in src.attrs.items():
                dst.attrs[k] = v
            for name in src:
                if isinstance(src.get(name, getlink=True), h5py.SoftLink):
                    continue                      # remade below, once its target exists
                if name in new:
                    a = new[name].astype(src[name].dtype)
                    dst.create_dataset(name, data=a, chunks=(1,) + a.shape[1:],
                                       compression=src[name].compression)
                elif name == "slice_index":
                    dst.create_dataset("slice_index", data=idx.astype(np.int32))
                else:
                    src.copy(name, dst, name)
            if "img" in src or link_name.lstrip("/") in dst:
                dst["img"] = h5py.SoftLink(link_name)
            dst.attrs["reg_applied"] = True
            dst.attrs["reg_offset"] = np.asarray(offset, dtype=np.int32)
            dst.attrs["reg_reference"] = str(reference)
            dst.attrs["reg_rule"] = (
                f"3-D FFT cross-correlation of the low-passed {cfg.reg_source} "
                f"{cfg.reg_contrast} (lowpass={cfg.reg_lowpass}, "
                f"{'phase' if cfg.reg_phase else 'plain'}), integer voxel translation applied to "
                f"img_raw / {norm_key(MODE)} / mask / support alike, by "
                f"scripts/register_nyumets.py")
        os.replace(tmp, out_path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def geometry(patients, cfg):
    """Is an integer-index translation even the right model? Attrs only -- no pixels read.

    A shift in INDEX space only means something if the studies sample the same physical grid.
    NYUMets is a clinical cohort, so slice thickness and in-plane spacing vary between studies
    of one patient; when they do, no integer offset can align them and a cross-correlation will
    still return a confident-looking number. This prints the geometry so that possibility is
    settled before anything is rebuilt.
    """
    import statistics

    rows, spacing_bad, depth_bad, fov_bad = [], [], [], []
    for pid, studies in patients.items():
        if len(studies) < 2:
            continue
        studies.sort(key=lambda s: s.case)
        geo = []
        for st in studies:
            aff = np.asarray(st.attrs.get("affine", np.eye(4)), dtype=np.float64)
            sp = tuple(round(float(np.linalg.norm(aff[:3, k])), 3) for k in range(3))
            geo.append((st.case, sp, st.depth, int(st.index.min()), int(st.index.max()),
                        tuple(int(v) for v in st.attrs.get("native_size", (0, 0)))))
        sp_set = {g[1] for g in geo}
        z_set = {g[1][2] for g in geo}
        d_set = {g[2] for g in geo}
        n_set = {g[5] for g in geo}
        rows.append((pid, geo, len(sp_set) > 1, len(z_set) > 1))
        if len(z_set) > 1:
            spacing_bad.append(pid)
        if len(d_set) > 1:
            depth_bad.append(pid)
        if len(n_set) > 1:
            fov_bad.append(pid)

    n = len(rows)
    if not n:
        print("no multi-study patient in this sample")
        return
    print(f"\n{n} multi-study patient(s):\n")
    print(f"  studies whose SLICE SPACING differs within the patient : "
          f"{len(spacing_bad)}/{n}  ({100 * len(spacing_bad) / n:.0f}%)")
    print(f"  studies whose full-spacing triple differs              : "
          f"{sum(1 for r in rows if r[2])}/{n}")
    print(f"  studies whose orig_depth differs                       : "
          f"{len(depth_bad)}/{n}")
    print(f"  studies whose native in-plane size differs              : "
          f"{len(fov_bad)}/{n}")
    print("\n  An integer-index shift can only align studies that share a physical grid. Where")
    print("  the slice spacing differs, registration has to RESAMPLE to a common grid first --")
    print("  cross-correlation will still return a confident-looking offset, and it will be")
    print("  meaningless.\n")

    for pid, geo, sp_mix, z_mix in rows[:8]:
        flag = "  <-- SPACING DIFFERS" if z_mix else ("  <-- spacing triple differs" if sp_mix
                                                      else "")
        print(f"  {pid}{flag}")
        for case, sp, depth, lo, hi, native in geo:
            print(f"      {case:<30s} spacing {sp}  depth {depth:>4d}  kept {lo:>3d}..{hi:<3d}"
                  f"  native {native}")
    if len(rows) > 8:
        print(f"  ... and {len(rows) - 8} more")

    # WHAT THE HEADERS ALREADY KNOW. With one voxel size across the cohort the studies differ
    # by a translation, and the affines state it exactly -- so print it, and print how oblique
    # they are, because a rotation is the one thing no integer shift can fix.
    print("\n  offset predicted by the AFFINES (no pixels involved), vs each study's stored")
    print("  reg_offset if the set was registered:\n")
    pred, obl = [], []
    for pid, studies in list(patients.items())[:8]:
        if len(studies) < 2:
            continue
        studies.sort(key=lambda s: s.case)
        crop = max(0, int(studies[0].attrs.get("crop_size", 0) or 0))   # -1 = no crop
        ref = max(range(len(studies)), key=lambda i: (int(studies[i].acquired.sum()), -i))
        key = lambda st: (np.asarray(st.attrs.get("affine", np.eye(4))),
                          tuple(int(v) for v in st.attrs.get("native_size", (crop, crop))),
                          st.depth)
        print(f"  {pid}  (ref {studies[ref].case})")
        for i, st in enumerate(studies):
            if i == ref:
                continue
            t, ob = affine_offset(key(studies[ref]), key(st), crop)
            was = (tuple(np.asarray(st.attrs["reg_offset"]).tolist())
                   if "reg_offset" in st.attrs else None)
            pred.append(t)
            obl.append(ob)
            print(f"      {st.case:<30s} affine says (dz={t[0]:+4d}, dh={t[1]:+4d}, "
                  f"dw={t[2]:+4d})  oblique {ob:.3f}"
                  + (f"   stored {was}" if was else ""))
    if pred:
        import statistics as _st
        print("")
        for ax, name in enumerate(("dz", "dh", "dw")):
            v = [abs(t[ax]) for t in pred]
            print(f"  |{name}| from the affines: median {_st.median(v):.1f}  max {max(v)}")
        print(f"  obliqueness: median {_st.median(obl):.4f}  max {max(obl):.4f}   "
              f"(0 = pure translation; > ~0.02 and a shift cannot align them)")

    zs = [g[1][2] for _, geo, _, _ in rows for g in geo]
    print(f"\n  slice spacing across all sampled studies: median {statistics.median(zs):.2f} mm, "
          f"min {min(zs):.2f}, max {max(zs):.2f}, {len(set(zs))} distinct value(s)")


def compare(patients, cfg):
    """Every (source x correlation) variant on the same patients, scored by residual.

    The question this answers -- does it matter whether the offset is measured on raw or on
    normalised data -- has a real answer and it is not the same for every cohort, so measure
    it rather than argue it. `none` is the control: the residual with NO shift at all. A
    variant that cannot beat `none` is not registering anything.
    """
    import statistics

    variants = [("none", None, None),
                ("norm, phase", "norm", True), ("norm, plain", "norm", False),
                ("raw,  phase", "raw", True), ("raw,  plain", "raw", False)]
    scores = {v[0]: [] for v in variants}
    moved = {v[0]: [] for v in variants}
    ci = CONTRASTS.index(cfg.reg_contrast)
    n_done = 0

    for pid, studies in patients.items():
        studies.sort(key=lambda s: s.case)
        if len(studies) < 2 or len({tuple(s.hw) for s in studies}) != 1:
            continue
        depth = max(s.depth for s in studies)
        acq = [np.pad(a.acquired, (0, depth - a.depth)) for a in studies]
        if int(np.logical_and.reduce(acq).sum()) < cfg.min_overlap:
            continue
        ref = max(range(len(studies)), key=lambda i: (int(acq[i].sum()), -i))
        cache = {}
        for name, src, phase in variants:
            # the residual is always scored on the NORMALISED planes, whatever the offset was
            # measured on -- otherwise `raw` and `norm` would be graded on different scales
            scoring = cache.setdefault("norm", [st.planes(ci, "norm") for st in studies])
            if src is None:
                offs = [(0, 0, 0)] * len(studies)
            else:
                pl = cache.setdefault(src, [st.planes(ci, src) for st in studies])
                offs = measure(pl, acq, dict_cfg(cfg, reg_source=src, reg_phase=phase), ref)
            valid = [translate(torch.as_tensor(a), (t[0],)).numpy() if any(t) else a
                     for a, t in zip(acq, offs)]
            idx = np.where(np.logical_and.reduce(valid))[0]
            if idx.size == 0:
                continue
            mk = cache.setdefault("mask", [st.masks() for st in studies])
            scores[name].append(residual(scoring, offs, idx, ref, masks=mk)[1])
            moved[name].append(sum(1 for t in offs if any(t)))
        cache.clear()
        n_done += 1
        if n_done % 10 == 0:
            print(f"  {n_done} patients...")

    if not n_done:
        print("no patient had two comparable studies to compare on")
        return
    print(f"\n{n_done} patient(s) compared. Residual after the shift "
          f"(mean |other - ref| / mean |ref| INSIDE the joint brain mask, lower is better):\n")
    print(f"  {'variant':<14}{'median':>9}{'mean':>9}{'p90':>9}   studies shifted")
    base = statistics.median(scores["none"]) if scores["none"] else float("nan")
    for name, _, _ in variants:
        v = scores[name]
        if not v:
            continue
        med = statistics.median(v)
        gain = "" if name == "none" else f"   ({100 * (1 - med / base):+.0f}% vs none)"
        print(f"  {name:<14}{med:9.3f}{statistics.mean(v):9.3f}"
              f"{sorted(v)[int(0.9 * (len(v) - 1))]:9.3f}   {sum(moved[name])}{gain}")
    best = min((n for n, _, _ in variants[1:] if scores[n]),
               key=lambda n: statistics.median(scores[n]), default=None)
    print(f"\n  best: {best}" if best else "")
    print("  If NOTHING beats `none` by much, the studies are not related by a translation: "
          "rotation, or a residual that is dominated by genuine change between visits rather "
          "than by misalignment. Compare `none` against the visual check before rebuilding.")


def dict_cfg(cfg, **over):
    """A shallow copy of cfg with a few fields overridden -- variants must not mutate cfg."""
    import copy as _copy
    c = _copy.copy(cfg)
    for k, v in over.items():
        setattr(c, k, v)
    return c


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
    ap.add_argument("--reg-max-shift", type=int, default=0, dest="reg_max_shift",
                    help="reject an offset larger than this in any axis. 0 (the default) "
                         "accepts whatever the correlation finds: the 40 this used to default "
                         "to was arbitrary, and the 2026-09-23 survey caught it rejecting dz of "
                         "43-49 that looked genuine -- large through-plane, small and consistent "
                         "in-plane. --min-overlap is the principled guard")
    ap.add_argument("--min-overlap", type=int, default=8, dest="min_overlap",
                    help="refuse a patient whose studies share fewer acquired slices than "
                         "this -- too little to trust the through-plane offset")
    ap.add_argument("--reg-source", default="norm", choices=("norm", "raw"),
                    dest="reg_source",
                    help="measure the offset on img_median_mad ('norm') or img_raw ('raw'). "
                         "They are NOT equivalent: normalize_masked ends with `out = out * fg`, "
                         "so the normalised array is exactly zero outside the brain mask and "
                         "its strongest edge is a mask boundary that differs between studies")
    ap.add_argument("--no-phase", action="store_false", dest="reg_phase",
                    help="plain cross-correlation instead of phase correlation (what "
                         "generate_longibrain.jl does)")
    ap.add_argument("--geometry", action="store_true",
                    help="print each patient's per-study voxel spacing, depth and kept range "
                         "and exit. Attrs only, no pixels -- run this FIRST when registration "
                         "is not converging: an integer-index shift cannot align studies that "
                         "do not share a physical grid")
    ap.add_argument("--compare", action="store_true",
                    help="measure every (source x correlation) variant on the sample and print "
                         "which one aligns best. Read-only; settles the choice on YOUR data "
                         "instead of on an argument about DC terms")
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

    if (cfg.compare or cfg.geometry) and cfg.apply:
        raise SystemExit("--compare / --geometry are read-only measurements; drop --apply")
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

    if cfg.geometry:
        return geometry(patients, cfg)
    if cfg.compare:
        return compare(patients, cfg)

    rows, skipped, resid_rows = [], [], []
    n_write, n_skip, secs_io, t_start = [0], [0], [0.0], time.time()
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
        planes = [st.planes(ci, cfg.reg_source) for st in studies]
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
        resid = (residual(planes, offsets, idx, ref, masks=[st.masks() for st in studies])
                 if len(studies) > 1 else None)
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
            todo = [i for i, st in enumerate(studies) if needs_write(st, idx, offsets[i])]
            n_skip[0] += len(studies) - len(todo)
            if todo:
                # Only now are the full volumes read, and only for the studies that change.
                # register_patient RE-MEASURES from them; `measure` above used the same
                # primitives on the same planes, so the offsets agree (a test pins that).
                _t = time.time()
                vols = [_volume(st.load()) for st in studies]
                if len(studies) > 1:
                    register_patient(vols, cfg, ref=ref, real=acq)
                for i in todo:
                    st = studies[i]
                    rel = os.path.relpath(st.path, cfg.root)
                    write_registered(st, vols[i], idx, offsets[i], studies[ref].case,
                                     os.path.join(out_root, rel), cfg)
                    st.unload()
                del vols
                secs_io[0] += time.time() - _t
                n_write[0] += len(todo)

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
    if cfg.apply:
        print(f"\nwrote {n_write[0]} study(ies), skipped {n_skip[0]} already correct; "
              f"{secs_io[0]:.0f}s of read+shift+write out of {time.time() - t_start:.0f}s total")
        print("  This job is I/O bound -- each study is a few hundred MB of h5 in and out. The "
              "GPU shortens the correlation and the shift, not the transfer.")
    if not cfg.apply:
        print("\nREPORT ONLY -- nothing was written."
              + (f" This was {len(patients)} of {n_multi} multi-study patients; --sample 0 "
                 f"does all of them, and a different --seed draws a different sample."
                 if sampled else "")
              + " To write, add --apply --sample 0. To do better, rebuild with "
                "preprocessing/nyumets_h5.py, which registers on the full volumes.")


if __name__ == "__main__":
    main()
