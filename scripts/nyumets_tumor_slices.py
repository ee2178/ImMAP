#!/usr/bin/env python3
"""
Find the dumped NYUMets slices that show a TUMOR -- in the current study, the prior, or both --
from the studies' own radiotherapy segmentations, and score both nets inside the lesion.

    python scripts/nyumets_tumor_slices.py                         # the two default runs
    python scripts/nyumets_tumor_slices.py --limit 20 --check      # geometry check on 20 studies
    python scripts/nyumets_tumor_slices.py --from-csv results/nyumets_eval/tumor_slices.csv
                                                                   # re-rank locally, CSV only

WHERE THE LESIONS COME FROM. A study directory of the raw data may hold RTSTRUCT_segmentation.nii,
the radiotherapy planning contours, next to its FLAIR / T1 / CT1 / T2 (730 of the 4596 four-contrast
studies; 442 of the 683 multi-study patients have at least one). The h5 build skipped them. This
script carries each one into its study's h5 space, exactly as the build carried the images:

    stored (h, w, z_out) --reg_matrix--> cropped source index --uncrop--> native canonical index
        --affine_source--> world mm --inverse seg affine--> segmentation voxel   (linear, > 0.5)

using only what the h5 stored (reg_matrix, crop_size, native_size_source, affine_source,
slice_index); the segmentation may sit on its own grid. Supported for reg_method fit / affine /
none -- the val split (NYUMets_h5_fit_val) is fit.

CHECK THE GEOMETRY BEFORE TRUSTING ANYTHING (--check, and always summarised): inside a correctly
mapped lesion the CT1 is ENHANCED, so `ct1_ratio` = mean CT1 in the lesion / median CT1 in the
brain should sit well above 1, and `in_brain` (the share of lesion pixels inside the stored brain
mask) near 1. Values near 1 / low mean the mapping, or the label choice, is off.

LABELS. RTSTRUCT files can carry more than tumors (organs at risk: brainstem, optic nerves, ...).
The script prints the label values it finds; `--labels 1 3` keeps only those (default: all > 0).

WHAT IS UNKNOWN IS NOT "NO TUMOR". A study without a segmentation file is `seg_cur = 0`, and its
slices are never counted as tumor-free; only slices of SEGMENTED studies enter the tumor /
no-tumor comparison.

OUTPUT (--out, default results/nyumets_eval/tumor_slices.csv), one row per dumped (volume, slice):

    volume, slice, prior_volume       the dump's current study and its pinned prior
    seg_cur, seg_prior                1 if that study has a segmentation
    lesion_cur, lesion_prior          lesion pixels at this ORIGINAL slice (the prior at the same
                                      index -- the stored alignment)
    psnr_base, psnr_longi             the dump's stored per-slice PSNR of the two runs
    rmse_les_base, rmse_les_longi     RMSE inside the CURRENT lesion (empty if none)
    d_les                             rmse_les_base - rmse_les_longi  (> 0: the prior study helped)

and prints: the geometry check, the tumor slices ranked by lesion size and by the lesion gain,
slices whose tumor is in ONE study only (new or resolved), and the PSNR gain on tumor vs
tumor-free slices of segmented studies. `--write-rows N` writes the top N tumor slices (by
lesion gain) as the viewer's marked rows for figures/configs/nyumets_longi.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

NYU = os.path.join("trained_nets", "nyumets")
BASE = "I2SB_Unet_NYUMets_CT1_from_all_bs64"
LONGI = "I2SB_Unet_NYUMets_CT1_from_all_plus_otherT1CT1_w2_bs64"
NIFTI_ROOT = "/gpfs/data/fenglab/LiFeng/NYUMets/NYUMets/data/imaging/patientId"
SEG_NAME = "RTSTRUCT_segmentation.nii"
CT1 = 2                                               # stored channel order FLAIR, T1, CT1, T2


# ---------------------------------------------------------------------------------------------
# the segmentation, carried into h5 space
# ---------------------------------------------------------------------------------------------
def load_seg(path):
    """-> (array (H, W, D) in canonical RAS, its 4x4 affine). Separate so tests can replace it."""
    import nibabel as nib
    img = nib.as_closest_canonical(nib.load(path))
    v = np.asanyarray(img.dataobj)
    while v.ndim > 3:
        v = v[..., 0]
    return np.asarray(v), np.asarray(img.affine, dtype=np.float64)


def seg_path(nifti_root, patient, session):
    return os.path.join(nifti_root, str(patient), "studyId", str(session), SEG_NAME)


def stored_to_seg(attrs, seg_affine):
    """4x4 taking a STORED (h, w, z_out, 1) index to a segmentation (i, j, k, 1) voxel."""
    method = str(attrs.get("reg_method", "none"))
    if method not in ("fit", "affine", "none"):
        raise NotImplementedError(f"reg_method {method!r}: only fit / affine / none store the "
                                  f"full transform in reg_matrix")
    M = np.asarray(attrs.get("reg_matrix", np.eye(4)), dtype=np.float64)
    crop = int(attrs.get("crop_size", -1))
    U = np.eye(4)
    if crop > 0:
        # stored = native + off, off = (crop - n) // 2 per in-plane axis (center_crop_or_pad)
        n = [int(x) for x in np.asarray(attrs["native_size_source"]).ravel()[:2]]
        U[0, 3], U[1, 3] = -((crop - n[0]) // 2), -((crop - n[1]) // 2)
    A_src = np.asarray(attrs["affine_source"], dtype=np.float64)
    return np.linalg.inv(seg_affine) @ A_src @ U @ M


def sample_seg(seg, T, z_out, H, W, labels=None):
    """(n, H, W) bool lesion masks at the stored slices `z_out` (original indices)."""
    from scipy.ndimage import map_coordinates
    lab = (seg > 0) if not labels else np.isin(seg, labels)
    lab = lab.astype(np.float32)
    h, w = np.meshgrid(np.arange(H, dtype=np.float64), np.arange(W, dtype=np.float64),
                       indexing="ij")
    out = np.zeros((len(z_out), H, W), bool)
    for j, z in enumerate(z_out):
        P = np.stack([h.ravel(), w.ravel(), np.full(h.size, float(z)), np.ones(h.size)])
        ijk = (T @ P)[:3]
        v = map_coordinates(lab, ijk, order=1, mode="constant", cval=0.0)
        out[j] = (v > 0.5).reshape(H, W)
    return out


def study_lesions(h5path, nifti_root, labels=None, seen=None):
    """-> dict(slice_index (n,), lesion (n, H, W) bool, ct1 (n, H, W), brain (n, H, W)) for one h5
    study, or None when it has no segmentation file. `seen` collects the label values found."""
    with h5py.File(h5path, "r") as f:
        attrs = {k: f.attrs[k] for k in f.attrs}
        sp = seg_path(nifti_root, _s(attrs["patient"]), _s(attrs["session"]))
        if not os.path.exists(sp):
            return None
        z_out = np.asarray(f["slice_index"][:]).astype(int)
        key = "img_raw" if "img_raw" in f else "img"
        ct1 = np.asarray(f[key][..., CT1], dtype=np.float32)
        brain = (np.asarray(f["mask"][..., 0]) > 0) if "mask" in f else ct1 > 0
    seg, A = load_seg(sp)
    if seen is not None:
        for v in np.unique(seg):
            seen[int(v)] = seen.get(int(v), 0) + 1
    T = stored_to_seg(attrs, A)
    les = sample_seg(seg, T, z_out, ct1.shape[1], ct1.shape[2], labels)
    return dict(slice_index=z_out, lesion=les, ct1=ct1, brain=brain)


def geometry_check(L):
    """ct1_ratio and in_brain for one study's mapped lesions (NaN when it has none)."""
    les, ct1, brain = L["lesion"], L["ct1"], L["brain"]
    if not les.any():
        return dict(ct1_ratio=np.nan, in_brain=np.nan, n_px=0)
    med = float(np.median(ct1[brain])) if brain.any() else np.nan
    return dict(ct1_ratio=float(ct1[les].mean() / max(med, 1e-6)),
                in_brain=float((les & brain).sum() / les.sum()), n_px=int(les.sum()))


def _s(v):
    return v.decode() if isinstance(v, bytes) else str(v)


# ---------------------------------------------------------------------------------------------
def orient(a, mode):
    """The dump's display turn (visualization.image.orient_image), on a (n, H, W) stack."""
    if not mode:
        return a
    a = np.swapaxes(a, -2, -1)[..., ::-1, :]
    return a[..., ::-1] if mode == "radiological" else a


def dump_dir(root, run):
    for d in (os.path.join(root, run, "eval_dump"), os.path.join(root, run)):
        if os.path.isdir(d) and any(f.endswith(".h5") for f in os.listdir(d)):
            return d
    raise SystemExit(f"no dump for {run} under {root}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=os.path.join(HERE, NYU), help="where the run dirs live")
    p.add_argument("--base", default=BASE)
    p.add_argument("--longi", default=LONGI)
    p.add_argument("--split-root", help="h5 root of the dumped split (default: the base run's "
                                        "val root, from its config.json)")
    p.add_argument("--nifti-root", default=NIFTI_ROOT, help="raw data: <root>/<patient>/studyId/"
                                                            "<session>/" + SEG_NAME)
    p.add_argument("--labels", nargs="*", type=int, help="segmentation values that are tumor "
                                                         "(default: every value > 0)")
    p.add_argument("--min-px", type=int, default=20, help="lesion pixels for a slice to count")
    p.add_argument("--limit", type=int, help="first N volumes only (a quick check)")
    p.add_argument("--check", action="store_true", help="print the geometry check per study")
    p.add_argument("--out", default=os.path.join(HERE, "results", "nyumets_eval",
                                                 "tumor_slices.csv"))
    p.add_argument("-k", type=int, default=12, help="rows per ranked list")
    p.add_argument("--write-rows", type=int, default=0,
                   help="write the top N tumor slices (by lesion gain) as the viewer's rows")
    p.add_argument("--force", action="store_true", help="overwrite an existing rows file")
    p.add_argument("--from-csv", metavar="TUMOR_CSV",
                   help="re-rank a CSV this script wrote (e.g. copied off the cluster): no dumps, "
                        "no NIfTIs, no geometry check")
    a = p.parse_args(argv)

    if a.from_csv:
        num = ("seg_cur", "seg_prior", "lesion_cur", "lesion_prior", "slice")
        flt = ("psnr_base", "psnr_longi", "rmse_les_base", "rmse_les_longi", "d_les")
        with open(a.from_csv, newline="") as f:
            rows = [{k: (int(v) if k in num else (float(v) if k in flt and v != "" else v))
                     for k, v in r.items()} for r in csv.DictReader(f)]
        a.out = a.from_csv
        report(rows, [], {}, a)
        return

    db, dl = dump_dir(a.root, a.base), dump_dir(a.root, a.longi)
    split = a.split_root
    if split is None:
        import yaml
        with open(os.path.join(a.root, a.base, "config.json")) as f:
            split = yaml.safe_load(f)["data"]["val"]["root"]
    vols = sorted({x[:-3] for x in os.listdir(db) if x.endswith(".h5")}
                  & {x[:-3] for x in os.listdir(dl) if x.endswith(".h5")})
    if a.limit:
        vols = vols[:a.limit]
    print(f"{len(vols)} dumped volumes | split {split} | segmentations under {a.nifti_root}")

    cache, seen, checks, rows = {}, {}, [], []

    def lesions(vol):
        if vol not in cache:
            path = os.path.join(split, vol, vol + "_img.h5")
            L = study_lesions(path, a.nifti_root, a.labels, seen) if os.path.exists(path) else None
            if L is not None:
                g = geometry_check(L)
                checks.append(dict(volume=vol, **g))
                if a.check:
                    print(f"  {vol}: {g['n_px']} lesion px, ct1_ratio {g['ct1_ratio']:.2f}, "
                          f"in_brain {g['in_brain']:.2f}")
                L = dict(slice_index=L["slice_index"], lesion=L["lesion"])   # drop the images
            cache[vol] = L
            if len(cache) > 8:                                               # keep memory flat
                cache.pop(next(k for k in cache if k != vol))
        return cache[vol]

    for vi, vol in enumerate(vols):
        with h5py.File(os.path.join(db, vol + ".h5"), "r") as fb, \
                h5py.File(os.path.join(dl, vol + ".h5"), "r") as fl:
            prior = _s(fb.attrs.get("prior_volume", ""))
            reg = _s(fb.attrs.get("registration", "stored"))
            mode = _s(fb.attrs.get("display_orient", "")) or None
            z = fb["slice_index"][:].astype(int)
            ref, rb, rl = fb["reference"][:], fb["recon"][:], fl["recon"][:]
            pb, pl = fb["psnr_slice"][:], fl["psnr_slice"][:]
        Lc, Lp = lesions(vol), (lesions(prior) if prior else None)

        def at(L, zz):
            if L is None:
                return None
            hit = np.where(L["slice_index"] == zz)[0]
            return L["lesion"][hit[0]] if len(hit) else np.zeros(L["lesion"].shape[1:], bool)

        for j, zz in enumerate(z):
            lc, lp = at(Lc, zz), at(Lp, zz)
            r = dict(volume=vol, slice=int(zz), prior_volume=prior,
                     seg_cur=int(Lc is not None), seg_prior=int(Lp is not None),
                     lesion_cur=int(lc.sum()) if lc is not None else 0,
                     lesion_prior=int(lp.sum()) if lp is not None else 0,
                     psnr_base=float(pb[j]), psnr_longi=float(pl[j]),
                     rmse_les_base="", rmse_les_longi="", d_les="")
            # the lesion region in the dump's frame; a current_to_prior dump moved the current
            # study, so its lesions no longer sit where the h5 put them
            if lc is not None and lc.sum() >= a.min_px and not reg.startswith("current_to_prior"):
                m = orient(lc, mode)
                eb = float(np.sqrt(((rb[j] - ref[j]) ** 2)[m].mean()))
                el = float(np.sqrt(((rl[j] - ref[j]) ** 2)[m].mean()))
                r.update(rmse_les_base=eb, rmse_les_longi=el, d_les=eb - el)
            rows.append(r)
        if (vi + 1) % 50 == 0:
            print(f"  {vi + 1}/{len(vols)} volumes", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    report(rows, checks, seen, a)


def report(rows, checks, seen, a):
    if seen:
        print(f"\nlabel values found (value: studies): {dict(sorted(seen.items()))}"
              + ("" if a.labels else "  -- every value > 0 counts as tumor; --labels to restrict"))
    if checks:
        cr = np.array([c["ct1_ratio"] for c in checks], float)
        ib = np.array([c["in_brain"] for c in checks], float)
        ok = np.isfinite(cr)
        print(f"geometry check over {ok.sum()} segmented studies with a lesion in range: CT1 in "
              f"lesion / brain median: median {np.nanmedian(cr):.2f} (p10 {np.nanpercentile(cr, 10):.2f}); "
              f"share inside the brain mask: median {np.nanmedian(ib):.2f}")
        if np.nanmedian(cr) < 1.2:
            print("  !! the lesions are not brighter than the brain on CT1 -- the mapping or the "
                  "label choice is probably wrong; inspect with --check before using the ranking")
    seg_rows = [r for r in rows if r["seg_cur"] or r["seg_prior"]]
    tum = [r for r in rows if r["lesion_cur"] >= a.min_px or r["lesion_prior"] >= a.min_px]
    print(f"\n{len(rows)} slices; {len(seg_rows)} in volumes with a segmented current or prior; "
          f"{len(tum)} show a lesion (>= {a.min_px} px) in the current or the prior")

    def table(title, rs):
        print(f"\n{title}")
        print(f"  {'volume':<22} {'slice':>5} {'les cur':>7} {'les pri':>7} {'PSNR UNet':>9} "
              f"{'PSNR w2':>8} {'d PSNR':>7} {'lesion RMSE UNet -> w2':>24}")
        for r in rs:
            les = ("" if r["d_les"] == "" else
                   f"{r['rmse_les_base']:.4f} -> {r['rmse_les_longi']:.4f}")
            print(f"  {r['volume']:<22} {r['slice']:>5} {r['lesion_cur']:>7} {r['lesion_prior']:>7} "
                  f"{r['psnr_base']:>9.2f} {r['psnr_longi']:>8.2f} "
                  f"{r['psnr_longi'] - r['psnr_base']:>+7.2f} {les:>24}")

    def one_per_volume(rs, key, k):
        out, used = [], set()
        for r in sorted(rs, key=key):
            if r["volume"] in used:
                continue
            out.append(r)
            used.add(r["volume"])
            if len(out) == k:
                break
        return out

    cur = [r for r in rows if r["lesion_cur"] >= a.min_px]
    table("Largest tumor in the CURRENT study (one slice per volume):",
          one_per_volume(cur, lambda r: -r["lesion_cur"], a.k))
    gain = [r for r in cur if r["d_les"] != ""]
    best = one_per_volume(gain, lambda r: -r["d_les"], a.k)
    table("Largest error reduction INSIDE the current tumor (one per volume):", best)
    table("Tumor in the current study only -- new, or grown into this slice (both segmented):",
          one_per_volume([r for r in cur if r["seg_prior"] and r["lesion_prior"] < a.min_px],
                         lambda r: -r["lesion_cur"], a.k))
    table("Tumor in the prior only -- resolved, or treated (both segmented):",
          one_per_volume([r for r in rows if r["seg_cur"] and r["lesion_cur"] < a.min_px
                          and r["lesion_prior"] >= a.min_px], lambda r: -r["lesion_prior"], a.k))

    # the application: does the prior study help on tumor slices?
    segc = [r for r in rows if r["seg_cur"]]
    t = np.array([r["psnr_longi"] - r["psnr_base"] for r in segc if r["lesion_cur"] >= a.min_px])
    n = np.array([r["psnr_longi"] - r["psnr_base"] for r in segc if r["lesion_cur"] == 0])
    if t.size and n.size:
        print(f"\nPSNR gain of the prior study, slices of SEGMENTED current studies: tumor "
              f"{np.nanmean(t):+.2f} dB (median {np.nanmedian(t):+.2f}, n={t.size}) vs tumor-free "
              f"{np.nanmean(n):+.2f} dB (median {np.nanmedian(n):+.2f}, n={n.size})")
    if gain:
        d = np.array([r["d_les"] for r in gain], float)
        print(f"lesion RMSE, UNet - w2 (> 0: the prior helped inside the tumor): mean {d.mean():+.4f}, "
              f"median {np.median(d):+.4f}, w2 better on {100 * (d > 0).mean():.0f}% of {d.size} slices")
    print(f"\nevery slice -> {a.out}")

    if a.write_rows and best:
        path = os.path.join(HERE, "figures", "rows", "nyumets_longi.json")
        if os.path.exists(path) and not a.force:
            print(f"{path} exists -- not overwritten; pass --force")
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump([dict(volume=r["volume"], slice=r["slice"])
                           for r in best[:a.write_rows]], f, indent=2)
            print(f"wrote {min(a.write_rows, len(best))} row(s) -> {path}")


if __name__ == "__main__":
    main()
