#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build one HDF5 per (patient, session) from NYUMets, for sessions carrying the full
FLAIR / T1 / CT1 / T2 quartet. Sessions missing any of the four are skipped.

Layout mirrors preprocessing/cmap.py, so the existing BraTS loaders read these unchanged:

    <out>/<patient>_<session>/<patient>_<session>_img.h5
        img_raw          (n, H, W, C) float32   unnormalized intensities
        img_median_mad   (n, H, W, C) float32   per-contrast median/MAD, within brain
        img              -> soft link to img_median_mad (an alias, zero extra bytes)
        mask             (n, H, W, 1) uint8     brain mask
        support          (n, H, W, 1) uint8     common acquisition support (see below)
        slice_index      (n,)         int32     maps n back to the original RAS z

COMMON SUPPORT. Contrasts of one session do not always cover the same anatomy -- a T1 whose
FOV reaches the eyes paired with a CT1 whose FOV does not is the usual case. Every channel is
therefore masked to the INTERSECTION of the per-contrast acquisition supports, so T1 and CT1
share a domain exactly. Without it a T1 -> CT1 bridge is trained to delete an eye that
legitimately exists in its input, and the loss is dominated by a region the target never
acquired. Disable with --no-intersect-support. Per-contrast `support_lost_<C>` attrs record
how much of each contrast's own support the intersection dropped -- a large value on one
contrast IS the FOV mismatch.

Channel order is [FLAIR, T1, T1ce, T2] -- the BraTS order, so `contrast_idx` and every
existing config keep their meaning. NYUMets writes the enhanced T1 as CT1; the stored
LABEL stays T1ce.

Normalization is median/MAD (robust centre AND scale), computed per contrast over the
IN-BRAIN voxels of the whole volume via cmap.normalize_masked -- same statistics and
same background-zero convention as the BraTS h5s. Following cmap, the stats come from
the percentile-CLIPPED volume while `img_raw` stores the UNCLIPPED data, so inverting
img_median_mad -> raw is exact only away from the clipped tails.

    python -m preprocessing.nyumets_h5 --root ../datasets/NYUMets/data/imaging/patientId \
                                       --out ~/scratch/datasets/NYUMets_h5 --crop 224
"""

import os
import re
import csv
import glob
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import h5py
import nibabel as nib

from preprocessing.cmap import (
    normalize_masked, channelwise_percentile_clip, center_crop_spatial, norm_key,
)

CONTRASTS = ("FLAIR", "T1", "T1ce", "T2")          # stored channel order (BraTS order)
MODE = "median-mad"

# First match wins. FLAIR before T2 ("T2_FLAIR" is FLAIR); CT1 before T1 (t1 is a substring
# of ct1). The CT1 guard is a lookbehind, not \bct1\b -- '_' is a word character, so \b never
# fires between the '_' and the 'c' in 'NYU0001_CT1.nii.gz'.
RULES = (
    ("FLAIR", r"flair"),
    ("T1ce",  r"(?<![a-z0-9])c[\W_]?t1(?!\d)"),
    ("T1",    r"(?<![a-z0-9])t1(?!\d)"),
    ("T2",    r"(?<![a-z0-9])t2(?!\d)"),
)
RULES = tuple((lab, re.compile(p, re.I)) for lab, p in RULES)
SKIP_RE = re.compile(r"(seg|label|lesion|mask|roi|contour)", re.I)
DATE_RE = re.compile(r"((?:19|20)\d{2})[-_.]?(\d{2})[-_.]?(\d{2})")


def find_sessions(root):
    """-> {(patient, session): {contrast: path}}, keeping only complete quartets.

    LAYOUT: patientId/<PID>/studyId/<STUDY_ID>/{FLAIR,T1,CT1,T2}.nii -- `studyId` is a fixed
    literal directory and the session is one level BELOW it.

    The session key is therefore the IMMEDIATE PARENT DIRECTORY of the file: contrasts
    acquired together live together, which is the only rule that survives this layout. It
    replaces a date-regex-then-first-path-component fallback that returned the literal string
    "studyId" for every file, collapsing all of a patient's studies into one pseudo-session
    whose contrasts were then taken from DIFFERENT dates -- which is what produced both the
    mass "shapes differ" failures and the apparent in-plane disagreements between contrasts.

    Matching is on the BASENAME, not the whole relative path, so a directory name can never
    be mistaken for a contrast tag. RTSTRUCT_segmentation.nii is dropped by SKIP_RE;
    RTSTRUCT_MRI.nii simply matches no contrast rule.
    """
    out = {}
    for pid in sorted(e.name for e in os.scandir(root) if e.is_dir()):
        for path in glob.glob(os.path.join(root, pid, "**", "*.nii*"), recursive=True):
            fname = os.path.basename(path)
            if SKIP_RE.search(fname):
                continue
            lab = next((l for l, rx in RULES if rx.search(fname)), None)
            if lab is None:
                continue
            ses = os.path.basename(os.path.dirname(path)) or "single"
            out.setdefault((pid, ses), {}).setdefault(lab, path)
    return {k: v for k, v in sorted(out.items()) if all(c in v for c in CONTRASTS)}


def load_ras(path):
    """(H, W, D) float32 in canonical RAS, plus the affine. No resampling.

    REFUSES complex input rather than casting it. `np.asarray(z, dtype=np.float32)` on a
    complex array discards the imaginary part behind a ComplexWarning -- it would silently
    throw away the phase and leave the real part (NOT the magnitude) in its place, which is
    worse than either intended behaviour. If NYUMets ever turns out to carry complex volumes,
    decide explicitly here whether to store magnitude+phase as separate channels.
    """
    img = nib.as_closest_canonical(nib.load(path))
    v = np.asanyarray(img.dataobj)
    while v.ndim > 3:
        v = v[..., 0]
    if np.iscomplexobj(v):
        raise NotImplementedError(
            f"{path} holds complex data ({v.dtype}); this builder stores real channels only. "
            f"Decide how to carry phase (e.g. magnitude and phase as separate channels) "
            f"before proceeding -- casting here would drop the imaginary part silently.")
    return np.asarray(v, dtype=np.float32), img.affine


def center_crop_or_pad(a, size, h_axis, w_axis):
    """Center-crop to `size` on the two spatial axes, ZERO-PADDING any axis shorter than it.

    cmap.center_crop_spatial only crops. Given H < size it computes a NEGATIVE start index,
    which Python reads as an offset from the end, so it silently returns an array of length
    (size - H) // 2 instead of failing -- a 168-row volume with --crop 224 came out 28 rows,
    and only surfaced much later as a RandomCrop error inside the dataloader. BraTS never hit
    it because every subject is 240x240; NYUMets matrix sizes are NOT uniform, so the pad
    branch is load-bearing here.

    Padding rather than resampling keeps the voxel spacing honest, and padded voxels are zero
    and fall outside the support mask, so nothing downstream mistakes them for anatomy.
    """
    is_bool = a.dtype == torch.bool
    if is_bool:
        a = a.to(torch.uint8)
    for axis in (h_axis, w_axis):
        n = a.shape[axis]
        if n > size:
            a = a.narrow(axis, (n - size) // 2, size)
        elif n < size:
            before = (size - n) // 2
            pad = [0] * (2 * a.dim())
            j = (a.dim() - 1 - axis) * 2          # F.pad fills from the LAST dim backwards
            pad[j], pad[j + 1] = before, size - n - before
            a = F.pad(a, pad)
    return a.to(torch.bool) if is_bool else a


def _morph(m, r, op):
    """2D per-slice dilate/erode of a bool mask with a (2r+1)^2 box."""
    x = m[:, None].float()
    if op == "dilate":
        x = F.max_pool2d(x, 2 * r + 1, 1, r)
    else:
        x = -F.max_pool2d(-x, 2 * r + 1, 1, r)
    return x[:, 0] > 0.5


def support_mask(img, frac, r):
    """(D,C,H,W) -> (D,C,H,W) bool: where each contrast actually has ACQUIRED data.

    This is a different question from "is this tissue", and needs a different rule. The
    threshold is low -- just above the air noise floor -- so that dark-but-acquired voxels
    (CSF on T1) stay inside the support; an opening then drops isolated noise speckle and a
    closing fills interior holes, so the result is a field-of-view region rather than a
    tissue segmentation.
    """
    out = torch.zeros(img.shape, dtype=torch.bool)
    for c in range(img.shape[1]):
        v = img[:, c].float()
        flat = v.reshape(-1)
        step = max(1, flat.numel() // 2_000_000)          # torch.quantile caps around 16M
        hi = torch.quantile(flat[::step], 0.995).clamp_min(1e-6)
        m = v > frac * hi
        if r > 0:
            m = _morph(_morph(m, r, "erode"), r, "dilate")     # opening: drop speckle
            m = _morph(_morph(m, r, "dilate"), r, "erode")     # closing: fill holes
        out[:, c] = m
    return out


def brain_mask(img, bg_frac):
    """(D,C,H,W) -> (D,H,W) bool.

    NYUMets is not skull-stripped, so there is no `> 0` rule to lean on. Scale each
    contrast by its own p99.5, AVERAGE across contrasts, then threshold. Averaging rather
    than intersecting keeps CSF, which is dark on T1 but bright on T2/FLAIR and would be
    cut by an all-contrasts-must-agree rule. This is a proxy -- swap in HD-BET or
    SynthStrip masks later if the tissue statistics need to be exact.
    """
    acc = torch.zeros((img.shape[0],) + tuple(img.shape[2:]), dtype=torch.float32)
    for c in range(img.shape[1]):
        v = img[:, c].float()
        flat = v.reshape(-1)
        step = max(1, flat.numel() // 2_000_000)          # torch.quantile caps around 16M
        hi = torch.quantile(flat[::step], 0.995).clamp_min(1e-6)
        acc += v / hi
    return (acc / img.shape[1]) > bg_frac


def build(files, cfg):
    """-> (raw, norm, mask, support, slice_idx, stats, orig_depth, affine, lost, native_hw)."""
    vols, affines = zip(*(load_ras(files[c]) for c in CONTRASTS))
    if len({v.shape for v in vols}) != 1:
        raise ValueError(f"shapes differ: {[v.shape for v in vols]} -- not on a common grid")
    if cfg.require_affine and not all(np.allclose(a, affines[0], atol=1e-3) for a in affines):
        raise ValueError("affines differ -- contrasts are not co-registered")

    img = torch.from_numpy(np.stack([v.transpose(2, 0, 1) for v in vols], axis=1))  # (D,C,H,W)

    # COMMON SUPPORT. Contrasts of the same session do not always cover the same anatomy --
    # a T1 whose FOV includes the eyes paired with a CT1 whose FOV does not is the usual
    # case. Left alone, the bridge is trained to DELETE an eye that legitimately exists in
    # its input, and the loss is dominated by a region the target simply never acquired.
    # So intersect the per-contrast supports and give every channel the same one.
    #
    # This cannot come from brain_mask: that AVERAGES the contrasts, so an eye voxel carrying
    # signal in one of four contrasts still lands at ~1/4 of its normalised value, well above
    # bg_frac, and survives.
    lost = {}
    if cfg.intersect_support:
        sup = support_mask(img, cfg.support_frac, cfg.support_close)
        fov = sup.all(dim=1)                                   # (D,H,W)
        for c, name in enumerate(CONTRASTS):
            own = sup[:, c].sum()
            lost[name] = float(1.0 - (fov & sup[:, c]).sum() / own.clamp_min(1))
        img = img * fov.unsqueeze(1).to(img.dtype)
    else:
        fov = torch.ones(img.shape[:1] + img.shape[2:], dtype=torch.bool)

    fg = brain_mask(img, cfg.bg_frac) & fov
    clipped = (channelwise_percentile_clip(img, fg, cfg.clip[0], cfg.clip[1])
               if cfg.clip else img)
    norm, stats = normalize_masked(clipped, fg, mode=MODE)

    raw, orig_depth = img, img.shape[0]
    native_hw = tuple(int(v) for v in img.shape[2:4])
    if cfg.crop:
        raw = center_crop_or_pad(raw, cfg.crop, h_axis=2, w_axis=3)
        norm = center_crop_or_pad(norm, cfg.crop, h_axis=2, w_axis=3)
        fg = center_crop_or_pad(fg, cfg.crop, h_axis=1, w_axis=2)
        fov = center_crop_or_pad(fov, cfg.crop, h_axis=1, w_axis=2)

    frac = fg.reshape(fg.shape[0], -1).float().mean(dim=1).numpy()
    keep = frac >= cfg.min_brain_frac
    if cfg.start is not None or cfg.end is not None:
        z = np.arange(orig_depth)
        keep &= (z >= (cfg.start or 0)) & (z < (cfg.end if cfg.end is not None else orig_depth))
    idx = np.where(keep)[0]
    if idx.size == 0:
        raise ValueError("no slice passed min_brain_frac")

    to_hwc = lambda t: np.transpose(t.numpy()[idx], (0, 2, 3, 1))
    return (to_hwc(raw), to_hwc(norm), fg.numpy()[idx][..., None],
            fov.numpy()[idx][..., None], idx, stats, orig_depth, affines[0], lost, native_hw)


def write_h5(path, raw, norm, mask, support, idx, stats, orig_depth, affine, key,
             files, cfg, lost, native_hw):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    comp = "gzip" if cfg.compress else None
    chunk = lambda a: (1,) + a.shape[1:]
    with h5py.File(path, "w") as f:
        f.create_dataset("img_raw", data=raw.astype(np.float32),
                         chunks=chunk(raw), compression=comp)
        f.create_dataset(norm_key(MODE), data=norm.astype(np.float32),
                         chunks=chunk(norm), compression=comp)
        # `img` is the name every existing loader reaches for; a soft link keeps that working
        # without a second copy of the pixels on disk.
        f["img"] = h5py.SoftLink("/" + norm_key(MODE))
        f.create_dataset("mask", data=mask.astype(np.uint8),
                         chunks=chunk(mask), compression=comp)
        # the common acquisition support every channel was cropped to; kept so the region
        # that was zeroed is inspectable rather than merely gone
        f.create_dataset("support", data=support.astype(np.uint8),
                         chunks=chunk(support), compression=comp)
        f.create_dataset("slice_index", data=idx.astype(np.int32))
        f.attrs["subject_id"] = f"{key[0]}_{key[1]}"
        f.attrs["patient"], f.attrs["session"] = key
        f.attrs["contrasts"] = ",".join(CONTRASTS)
        f.attrs["normalize"] = MODE
        f.attrs[f"norm_stats_{MODE.replace('-', '_')}"] = stats      # (C, 2) centre/scale
        f.attrs["norm_stats"] = stats
        f.attrs["clip_percentiles"] = np.asarray(cfg.clip or (-1, -1), dtype=np.float32)
        f.attrs["crop_size"] = int(cfg.crop) if cfg.crop else -1
        # native in-plane size BEFORE the crop/pad, so a padded session is identifiable
        f.attrs["native_size"] = np.asarray(native_hw, dtype=np.int32)
        f.attrs["orig_depth"] = int(orig_depth)
        f.attrs["min_brain_frac"] = float(cfg.min_brain_frac)
        f.attrs["mask_rule"] = f"mean_c(v/p99.5_c) > {cfg.bg_frac}"
        f.attrs["intersect_support"] = bool(cfg.intersect_support)
        f.attrs["support_rule"] = (f"intersect_c(open/close(v_c > {cfg.support_frac}*p99.5_c), "
                                   f"r={cfg.support_close})" if cfg.intersect_support else "none")
        # fraction of each contrast's OWN support dropped by the intersection: a large value
        # on one contrast is the FOV mismatch (eyes in T1, absent in CT1) this guards against
        for _c in CONTRASTS:
            f.attrs[f"support_lost_{_c}"] = float(lost.get(_c, 0.0))
        f.attrs["affine"] = np.asarray(affine, dtype=np.float32)
        f.attrs["source_files"] = ",".join(os.path.basename(files[c]) for c in CONTRASTS)
        f.attrs["axis_order"] = "N,H,W,C ; slice_index maps N back to the canonical-RAS z"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="patient-ID level, e.g. .../imaging/patientId")
    ap.add_argument("--out", required=True)
    ap.add_argument("--crop", type=int, default=0, help="center crop in-plane; 0 = none")
    ap.add_argument("--clip", type=float, nargs=2, default=(0.5, 99.5),
                    help="foreground percentile clip before the stats; '--clip 0 0' disables")
    ap.add_argument("--bg-frac", type=float, default=0.05, dest="bg_frac")
    ap.add_argument("--no-intersect-support", action="store_false", dest="intersect_support",
                    help="do NOT force a common support across contrasts (not recommended: "
                         "a bridge then learns to delete anatomy its target never acquired)")
    ap.add_argument("--support-frac", type=float, default=0.02, dest="support_frac",
                    help="acquisition-support threshold as a fraction of each contrast's "
                         "p99.5; low on purpose, so dark-but-acquired CSF stays inside")
    ap.add_argument("--support-close", type=int, default=2, dest="support_close",
                    help="radius of the opening/closing on the support mask; 0 disables")
    ap.add_argument("--support-warn", type=float, default=0.02, dest="support_warn",
                    help="warn when the intersection drops more than this fraction of a "
                         "contrast's own support")
    ap.add_argument("--min-brain-frac", type=float, default=0.02, dest="min_brain_frac")
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--require-affine", action="store_true", dest="require_affine",
                    help="skip sessions whose contrasts are not on one affine")
    ap.add_argument("--compress", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", dest="dry_run",
                    help="report the native in-plane size distribution from HEADERS ONLY and "
                         "exit -- use it to choose --crop before building anything")
    cfg = ap.parse_args()
    if cfg.clip and float(cfg.clip[1]) <= 0:
        cfg.clip = None

    sessions = find_sessions(cfg.root)
    if cfg.limit:
        sessions = dict(list(sessions.items())[:cfg.limit])
    print(f"{len(sessions)} complete {'/'.join(CONTRASTS)} sessions under {cfg.root}")

    if cfg.dry_run:
        # nib.load is lazy and as_closest_canonical only rewrites the affine, so .shape costs
        # no voxel reads -- this is seconds over the whole cohort.
        hist, per_contrast, bad, dtypes = {}, {c: {} for c in CONTRASTS}, [], {}
        for key, files in sessions.items():
            try:
                imgs = {c: nib.as_closest_canonical(nib.load(files[c])) for c in CONTRASTS}
                shp = {c: im.shape[:2] for c, im in imgs.items()}
            except Exception as err:
                bad.append((key, str(err)))
                continue
            # The NIfTI datatype is the ONLY authoritative answer to "is this complex".
            # NIfTI can hold complex64/128/256; if the files were magnitude-only DICOM
            # exports they will be int16/uint16/float32 and the phase is simply not there.
            for c, im in imgs.items():
                dt = str(im.header.get_data_dtype())
                dtypes[(c, dt)] = dtypes.get((c, dt), 0) + 1
            for c, hw in shp.items():
                per_contrast[c][hw] = per_contrast[c].get(hw, 0) + 1
            if len(set(shp.values())) != 1:
                bad.append((key, f"contrasts disagree in-plane: {shp}"))
            hw = shp[CONTRASTS[0]]
            hist[hw] = hist.get(hw, 0) + 1
        print("\nstored NIfTI datatype, by contrast:")
        for c in CONTRASTS:
            row = {dt: n for (cc, dt), n in dtypes.items() if cc == c}
            print(f"  {c:<6} " + "  ".join(f"{dt} x{n}" for dt, n in sorted(row.items())))
        cplx = sorted({dt for (_, dt) in dtypes if "complex" in dt.lower()})
        if cplx:
            print(f"  ** COMPLEX data present ({', '.join(cplx)}) -- phase is available and "
                  f"load_ras will refuse rather than silently drop it")
        else:
            print("  -> all real: these are MAGNITUDE images and the phase is not in the")
            print("     files. No amount of preprocessing recovers it; it would have to come")
            print("     from the source DICOM (or the scanner) if it was ever saved at all.")

        print("\nnative in-plane sizes (H, W), by session:")
        for hw, n in sorted(hist.items(), key=lambda kv: -kv[1]):
            print(f"  {str(hw):<14} {n:>5}")
        dims = [d for hw in hist for d in hw]
        if dims:
            print(f"\nsmallest single in-plane dimension seen: {min(dims)}")
            print(f"largest  single in-plane dimension seen: {max(dims)}")
            print("\n--crop N pads any session smaller than N and crops any larger, so:")
            print(f"  --crop {max(dims)}  keeps every voxel (pads the small ones)")
            print(f"  --crop {min(dims)}  crops everything, never pads (loses FOV on the big ones)")
            print("The loader's crop_size must be <= --crop, and a RandomCrop much smaller")
            print("than the padded region will sample mostly zeros on the small sessions.")
        if bad:
            print(f"\n{len(bad)} of {len(sessions)} sessions have contrasts that DISAGREE "
                  f"in-plane ({100*len(bad)/max(len(sessions),1):.1f}%) -- these raise "
                  f"'shapes differ' and are skipped by the build:")
            for k, why in bad[:10]:
                print(f"  !! {k[0]}/{k[1]}: {why}")
            if len(bad) > 10:
                print(f"  ... and {len(bad) - 10} more")
        return

    rows, shapes, skipped, support_lost = [], {}, [], []
    natives, padded = {}, []
    for i, (key, files) in enumerate(sessions.items(), 1):
        # ONE DIRECTORY PER SESSION, not per patient. I2SBDataset.index_img_from_root takes
        # imgs[0] -- the first *_img.h5 in each subdirectory -- so grouping a patient's
        # sessions under one folder would silently index only one of them.
        case = f"{key[0]}_{key[1]}"
        path = os.path.join(cfg.out, case, f"{case}_img.h5")
        if os.path.exists(path) and not cfg.overwrite:
            continue
        try:
            raw, norm, mask, support, idx, stats, depth, aff, lost, native = build(files, cfg)
        except Exception as err:
            skipped.append((key, str(err)))
            print(f"  !! {key[0]}/{key[1]}: {err}")
            continue
        write_h5(path, raw, norm, mask, support, idx, stats, depth, aff, key, files, cfg,
                 lost, native)
        natives[native] = natives.get(native, 0) + 1
        if cfg.crop and (native[0] < cfg.crop or native[1] < cfg.crop):
            padded.append((key, native))
        if lost:
            worst = max(lost, key=lost.get)
            if lost[worst] > cfg.support_warn:
                print(f"  ?? {key[0]}/{key[1]}: intersection dropped {lost[worst]:.1%} of "
                      f"{worst}'s support -- FOV mismatch between contrasts")
            support_lost.append((key, dict(lost)))
        shapes[raw.shape[1:3]] = shapes.get(raw.shape[1:3], 0) + 1
        rows.append({"patient": key[0], "session": key[1], "path": path,
                     "n_slices": len(idx), "orig_depth": depth,
                     "H": raw.shape[1], "W": raw.shape[2]})
        if i % 25 == 0:
            print(f"  {i}/{len(sessions)}  {len(rows)} written, {len(skipped)} skipped")

    os.makedirs(cfg.out, exist_ok=True)
    with open(os.path.join(cfg.out, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["patient", "session", "path", "n_slices",
                                          "orig_depth", "H", "W"])
        w.writeheader()
        w.writerows(rows)

    print(f"\n{len(rows)} written, {len(skipped)} skipped -> {cfg.out}")
    print("native in-plane sizes:", dict(natives))
    print("stored in-plane sizes:", dict(shapes))
    if support_lost:
        import statistics
        print("\ncommon-support intersection, fraction of each contrast's own support dropped:")
        for c in CONTRASTS:
            vals = [d[c] for _, d in support_lost]
            print(f"  {c:<6} median {statistics.median(vals):6.2%}   max {max(vals):6.2%}")
        worst = max(support_lost, key=lambda kv: max(kv[1].values()))
        print(f"  worst session: {worst[0][0]}/{worst[0][1]}  "
              + "  ".join(f"{k} {v:.1%}" for k, v in worst[1].items()))
    if padded:
        print(f"\n{len(padded)} session(s) were smaller than --crop {cfg.crop} and were "
              f"ZERO-PADDED up to it:")
        for k, n in padded[:8]:
            print(f"    {k[0]}/{k[1]}  native {n[0]}x{n[1]}")
        if len(padded) > 8:
            print(f"    ... and {len(padded) - 8} more")
    if len(shapes) > 1:
        # The failure this guards against is not cosmetic: I2SBDataset applies a fixed
        # RandomCrop, which throws the moment it meets a stored image smaller than crop_size,
        # and only after the loader has already yielded a few good batches.
        print("!! STORED SHAPES ARE RAGGED -- a loader with a fixed crop_size will fail on "
              "the odd one out. Set --crop, or raise it above the largest native size.")


if __name__ == "__main__":
    main()
