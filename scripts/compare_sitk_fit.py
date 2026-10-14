#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SimpleITK registration (several settings) against a `--reg-method fit` build, on the same pairs.

    python -m scripts.compare_sitk_fit --fit-root ~/scratch/datasets/NYUMets_h5_fit \\
        --out cache/sitk_vs_fit --shard 0 --nshards 1
    python -m scripts.compare_sitk_fit --out cache/sitk_vs_fit --summarize

READ-ONLY on the data. For each sampled (patient, other study) of the fit build:

  1. every SimpleITK variant in --variants (preprocessing/sitk_register.py: VARIANTS) registers
     the other study's T1 NIfTI onto the reference T1 NIfTI -- the reference the fit build used;
  2. its T1, CT1 and head mask are resampled onto the reference NIfTI's grid and put through the
     builder's reorientation and --crop (`to_builder_space`), so they line up index for index with
     the fit build's h5s. A GEOMETRY CHECK first proves that mapping on the reference itself;
  3. per slice in --slice-range: the masked T1 and CT1 correlation with the reference, for the fit
     build and for the variant, BOTH on the reference brain AND the fit build's mask AND the
     variant's mask -- identical pixels;
  4. on --slices of those, the residual search of scripts/diagnose_nyumets_registration.py: the
     extra in-plane rotation / scale / shift and the slice offset that would still help.

Sharded so a job array can split the patients: shard i of n takes every n-th sampled patient and
writes <out>/slices_<i>.csv, residuals_<i>.csv, pairs_<i>.csv. --summarize reads every shard's
files and prints the comparison; it can run anywhere, once all shards are done.

`fit` has a home advantage on the T1 score: it optimises masked T1 correlation, the score used
here. The CT1 correlation and the residual rows are the fairer comparison.
"""

import argparse
import csv
import glob
import math
import os
import time

import h5py
import numpy as np
import torch

DEFAULT_NIFTI = "/gpfs/data/fenglab/LiFeng/NYUMets/NYUMets/data/imaging/patientId"
T1, CT1 = 1, 2


# ---------------------------------------------------------------------------------------------
def masked_ncc(a, b, m, min_px=500):
    if int(m.sum()) < min_px:
        return float("nan")
    x, y = a[m] - a[m].mean(), b[m] - b[m].mean()
    return float((x * y).sum() / max(math.sqrt(float((x * x).sum() * (y * y).sum())), 1e-8))


def fit_index(root):
    """{patient: {session: h5 path}} of a built set."""
    out = {}
    for p in sorted(glob.glob(os.path.join(root, "*", "*_img.h5"))):
        with h5py.File(p, "r") as f:
            pid, ses = f.attrs.get("patient"), f.attrs.get("session")
        pid = pid.decode() if isinstance(pid, bytes) else str(pid)
        ses = ses.decode() if isinstance(ses, bytes) else str(ses)
        out.setdefault(pid, {})[ses] = p
    return out


def h5_load(path):
    with h5py.File(path, "r") as f:
        return dict(z=np.asarray(f["slice_index"]).astype(int),
                    t1=np.asarray(f["img_raw"][..., T1], dtype=np.float32),
                    ct1=np.asarray(f["img_raw"][..., CT1], dtype=np.float32),
                    m=np.asarray(f["mask"][..., 0]).astype(bool),
                    crop=int(f.attrs.get("crop_size", -1)),
                    ref=str(f.attrs.get("reg_reference", "")),
                    method=str(f.attrs.get("reg_method", "")),
                    fit_ok=bool(f.attrs.get("reg_fit_ok", True)))


def sample_patients(fit, n, seed):
    multi = [p for p in sorted(fit) if len(fit[p]) > 1]
    order = np.random.default_rng(seed).permutation(len(multi))
    return [multi[i] for i in order[:n]] if n else [multi[i] for i in order]


# ---------------------------------------------------------------------------------------------
def run_shard(cfg):
    import SimpleITK as sitk
    from preprocessing import sitk_register as sr
    from scripts.diagnose_nyumets_registration import similarity_search, best_dz

    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(cfg.threads)
    torch.set_num_threads(cfg.threads)
    variants = {v: sr.VARIANTS[v] for v in cfg.variants}
    fit = fit_index(cfg.fit_root)
    pids = sample_patients(fit, cfg.sample, cfg.seed)[cfg.shard::cfg.nshards]
    angles = np.arange(-cfg.max_angle, cfg.max_angle + 1e-6, 1.0)
    scales = np.round(np.arange(1 - cfg.max_scale, 1 + cfg.max_scale + 1e-6, 0.02), 4)
    os.makedirs(cfg.out, exist_ok=True)
    print(f"shard {cfg.shard}/{cfg.nshards}: {len(pids)} patient(s); variants {list(variants)}; "
          f"{cfg.threads} thread(s); fit build {cfg.fit_root}")

    files = {k: open(os.path.join(cfg.out, f"{k}_{cfg.shard}.csv"), "w", newline="")
             for k in ("slices", "residuals", "pairs")}
    W = {"slices": csv.DictWriter(files["slices"], ["patient", "ref", "other", "variant", "z",
                                                    "ncc_t1", "ncc_ct1", "fit_ncc_t1", "fit_ncc_ct1"]),
         "residuals": csv.DictWriter(files["residuals"], ["patient", "other", "method", "z", "before",
                                                          "after", "angle", "scale", "dh", "dw", "dz"]),
         "pairs": csv.DictWriter(files["pairs"], ["patient", "ref", "other", "method", "secs", "rot",
                                                  "scale_min", "scale_max", "shift_mm", "fit_ok",
                                                  "geometry_ok", "stop"])}
    for w in W.values():
        w.writeheader()

    t_all = time.time()
    for n_p, pid in enumerate(pids):
        nif = sr.sessions_of(cfg.nifti_root, pid)
        first = h5_load(next(iter(fit[pid].values())))
        ref = first["ref"][len(pid) + 1:] if first["ref"].startswith(pid + "_") else None
        if ref not in fit[pid] or ref not in nif:
            print(f"  {pid}: reference {first['ref']!r} has no NIfTI under {cfg.nifti_root} -- skipped")
            continue
        R = h5_load(fit[pid][ref])
        fixed_native = sr.read(nif[ref]["T1"], lps=False)
        fixed_lps = sitk.DICOMOrient(fixed_native, "LPS")
        fmask_lps = sr.head_mask(fixed_lps)

        # GEOMETRY CHECK: the reference NIfTI through the mapping must BE the stored reference
        ref_bs = sr.to_builder_space(sitk.GetArrayFromImage(fixed_native), nif[ref]["T1"], R["crop"])
        rows = R["z"] < ref_bs.shape[0]
        err = (np.abs(ref_bs[R["z"][rows]] - R["t1"][rows]).max()
               / max(float(np.abs(R["t1"]).max()), 1e-6)) if rows.any() else float("inf")
        geo_ok = bool(err < 1e-4)
        if not geo_ok:
            print(f"  {pid}: GEOMETRY CHECK FAILED (max rel. diff {err:.1e}) -- skipped")
            W["pairs"].writerow(dict(patient=pid, ref=ref, other="", method="", geometry_ok=False))
            continue
        rr = {int(z): k for k, z in enumerate(R["z"])}

        for oth in [s for s in sorted(fit[pid]) if s != ref and s in nif][:cfg.pairs]:
            O = h5_load(fit[pid][oth])
            ro = {int(z): k for k, z in enumerate(O["z"])}
            W["pairs"].writerow(dict(patient=pid, ref=ref, other=oth, method="fit",
                                     fit_ok=O["fit_ok"], geometry_ok=True))
            mov = {c: sr.read(nif[oth][c], lps=False) for c in ("T1", "T1ce")}
            mov_lps = sitk.DICOMOrient(mov["T1"], "LPS")
            mov_mask = sr.head_mask(mov["T1"])
            zs = [z for z in R["z"].tolist() if z in ro and z < ref_bs.shape[0]
                  and cfg.slice_range[0] <= z < cfg.slice_range[1]]
            if not zs:
                print(f"  {pid}/{oth}: no shared slice in {cfg.slice_range}")
                continue
            pick = [zs[i] for i in np.linspace(0, len(zs) - 1, min(cfg.slices, len(zs))).astype(int)]
            T = lambda x: torch.from_numpy(np.ascontiguousarray(x))

            def residuals(method, vol, vmask, row):
                for z in pick:
                    a, ma = T(R["t1"][rr[z]]), T(R["m"][rr[z]])
                    j = row(z)
                    r = similarity_search(a, ma, T(vol[j]), T(vmask[j]), angles, scales)
                    r["dz"] = best_dz(a, ma, T(vol), T(vmask), j, cfg.dz_reach)
                    W["residuals"].writerow(dict(patient=pid, other=oth, method=method, z=z,
                                                 **{k: r[k] for k in ("before", "after", "angle",
                                                                      "scale", "dh", "dw", "dz")}))

            residuals("fit", O["t1"], O["m"], lambda z: ro[z])
            SV, secs = {}, {}
            for name, kw in variants.items():
                reg = sr.register(fixed_lps, mov_lps, fmask_lps, seed=cfg.seed, **kw)
                tx = reg["tx"]
                SV[name] = dict(
                    t1=sr.to_builder_space(sr.resample_onto(mov["T1"], fixed_native, tx),
                                           nif[ref]["T1"], R["crop"]),
                    ct1=sr.to_builder_space(sr.resample_onto(mov["T1ce"], fixed_native, tx),
                                            nif[ref]["T1"], R["crop"]),
                    m=sr.to_builder_space(sr.resample_onto(mov_mask, fixed_native, tx, True),
                                          nif[ref]["T1"], R["crop"], mask=True))
                residuals(name, SV[name]["t1"], SV[name]["m"], lambda z: z)
                rot, sc, sh = sr.describe(tx)
                secs[name] = reg["secs"]
                W["pairs"].writerow(dict(patient=pid, ref=ref, other=oth, method=name,
                                         secs=round(reg["secs"], 1), rot=round(rot, 2),
                                         scale_min=round(sc[0], 4), scale_max=round(sc[-1], 4),
                                         shift_mm=round(sh, 2), geometry_ok=True,
                                         stop=reg["stop"][:80]))
            # ONE pixel set per slice for every method: the reference brain AND fit's mask AND
            # every variant's mask. fit then has a single score per slice, and every row of the
            # summary is directly comparable with every other.
            med = {"fit": []}
            for z in zs:
                joint = R["m"][rr[z]] & O["m"][ro[z]]
                for S in SV.values():
                    joint = joint & S["m"][z]
                f1 = masked_ncc(R["t1"][rr[z]], O["t1"][ro[z]], joint)
                f2 = masked_ncc(R["ct1"][rr[z]], O["ct1"][ro[z]], joint)
                med["fit"].append(f1)
                for name, S in SV.items():
                    row = dict(patient=pid, ref=ref, other=oth, variant=name, z=z,
                               ncc_t1=masked_ncc(R["t1"][rr[z]], S["t1"][z], joint),
                               ncc_ct1=masked_ncc(R["ct1"][rr[z]], S["ct1"][z], joint),
                               fit_ncc_t1=f1, fit_ncc_ct1=f2)
                    W["slices"].writerow(row)
                    med.setdefault(name, []).append(row["ncc_t1"])
            del SV
            for f in files.values():
                f.flush()
            print(f"  {pid}/{oth}: {len(zs)} slices   T1 ncc  fit {np.nanmedian(med['fit']):.3f} | "
                  + " | ".join(f"{n} {np.nanmedian(med[n]):.3f} ({secs[n]:.0f}s)" for n in variants))
        print(f"  ... {n_p + 1}/{len(pids)} patients, {time.time() - t_all:.0f}s")
    for f in files.values():
        f.close()
    print(f"shard {cfg.shard} done in {time.time() - t_all:.0f}s -> {cfg.out}")


# ---------------------------------------------------------------------------------------------
def summarize(cfg):
    def read_all(kind):
        rows = []
        for p in sorted(glob.glob(os.path.join(cfg.out, f"{kind}_*.csv"))):
            with open(p, newline="") as f:
                rows += list(csv.DictReader(f))
        return rows

    S, Rr, P = read_all("slices"), read_all("residuals"), read_all("pairs")
    if not S:
        raise SystemExit(f"no slices_*.csv under {cfg.out}")
    fl = lambda x: float(x) if x not in ("", None) else float("nan")
    pairs = {(r["patient"], r["other"]) for r in S}
    variants = sorted({r["variant"] for r in S}, key=lambda v: list(cfg.variants).index(v)
                      if v in cfg.variants else 99)
    print(f"{len(pairs)} pairs over {len({p for p, _ in pairs})} patients, "
          f"{len(glob.glob(os.path.join(cfg.out, 'slices_*.csv')))} shard file(s)\n")

    print("CORRELATION WITH THE REFERENCE, paired on identical pixels (median over slices)")
    print(f"  {'variant':<13} {'slices':>6} {'fit T1':>7} {'sitk T1':>8} {'better':>7} {'d med':>7}"
          f"   {'fit CT1':>7} {'sitk CT1':>8} {'better':>7} {'d med':>7}")
    for v in variants:
        rows = [r for r in S if r["variant"] == v]
        a = np.array([[fl(r[k]) for k in ("fit_ncc_t1", "ncc_t1", "fit_ncc_ct1", "ncc_ct1")]
                      for r in rows])
        a = a[np.isfinite(a).all(1)]
        if not len(a):
            continue
        d1, d2 = a[:, 1] - a[:, 0], a[:, 3] - a[:, 2]
        print(f"  {v:<13} {len(a):>6} {np.median(a[:, 0]):>7.3f} {np.median(a[:, 1]):>8.3f} "
              f"{100 * (d1 > 0).mean():>6.0f}% {np.median(d1):>+7.3f}   {np.median(a[:, 2]):>7.3f} "
              f"{np.median(a[:, 3]):>8.3f} {100 * (d2 > 0).mean():>6.0f}% {np.median(d2):>+7.3f}")
    print("  better = share of slices where the SimpleITK variant beats fit; d med = median difference")

    print("\nRESIDUALS LEFT BEHIND (scripts/diagnose_nyumets_registration.py search; share of slices)")
    methods = ["fit"] + variants
    checks = [("|rotation| >= 3 deg", lambda r: abs(fl(r["angle"])) >= 3),
              ("|scale - 1| >= 0.04", lambda r: abs(fl(r["scale"]) - 1) >= 0.04),
              ("shift >= 4 px", lambda r: math.hypot(fl(r["dh"]), fl(r["dw"])) >= 4),
              ("best slice not within 1", lambda r: abs(fl(r["dz"])) > 1),
              ("extra fit raises ncc >= 0.10", lambda r: fl(r["after"]) - fl(r["before"]) >= 0.10),
              ("still below 0.5 after it", lambda r: fl(r["after"]) < 0.5)]
    print("  " + " " * 30 + "".join(f"{m:>13}" for m in methods))
    for name, f in checks:
        cells = []
        for m in methods:
            rows = [r for r in Rr if r["method"] == m]
            cells.append(f"{100 * np.mean([f(r) for r in rows]):>12.0f}%" if rows else f"{'-':>13}")
        print(f"  {name:<30}" + "".join(cells))
    print("  " + f"{'(slices searched)':<30}" + "".join(
        f"{sum(r['method'] == m for r in Rr):>13}" for m in methods))

    print("\nTRANSFORMS AND COST (median over pairs)")
    print(f"  {'variant':<13} {'secs':>6} {'rot deg':>8} {'scale lo':>9} {'scale hi':>9} {'shift mm':>9}")
    for v in variants:
        rows = [r for r in P if r["method"] == v and r.get("secs")]
        if rows:
            med = lambda k: np.median([fl(r[k]) for r in rows])
            print(f"  {v:<13} {med('secs'):>6.0f} {med('rot'):>8.1f} {med('scale_min'):>9.3f} "
                  f"{med('scale_max'):>9.3f} {med('shift_mm'):>9.1f}")
    fits = [r for r in P if r["method"] == "fit"]
    if fits:
        rej = sum(str(r["fit_ok"]) == "False" for r in fits)
        print(f"  fit: {rej}/{len(fits)} of these pairs were REJECTED fits (left on the centroid shift)")
    geo = sum(str(r.get("geometry_ok")) == "False" for r in P)
    if geo:
        print(f"  !! {geo} patient(s) skipped: GEOMETRY CHECK failed")

    print("\nPAIRS WHERE THE BEST SIMPLEITK VARIANT AND fit DISAGREE MOST (median T1 ncc over slices)")
    diffs = []
    for p, o in sorted(pairs):
        rows = [r for r in S if (r["patient"], r["other"]) == (p, o)]
        fitv = np.nanmedian([fl(r["fit_ncc_t1"]) for r in rows])
        best = max(variants, key=lambda v: np.nanmedian([fl(r["ncc_t1"]) for r in rows
                                                         if r["variant"] == v] or [np.nan]))
        bv = np.nanmedian([fl(r["ncc_t1"]) for r in rows if r["variant"] == best])
        diffs.append((bv - fitv, p, o, fitv, best, bv))
    for d, p, o, fv, b, bv in sorted(diffs, key=lambda t: -abs(t[0]))[:10]:
        print(f"  {p:<10} {o:<12} fit {fv:.3f}   {b} {bv:.3f}   ({d:+.3f})")
    print("\nfit optimises masked T1 correlation -- the T1 score here -- so CT1 and the residual rows"
          "\nare the fairer comparison. Open the extreme pairs in notebooks/sitk_registration_test.ipynb.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit-root", default="~/scratch/datasets/NYUMets_h5_fit", dest="fit_root")
    ap.add_argument("--nifti-root", default=DEFAULT_NIFTI, dest="nifti_root")
    ap.add_argument("--out", default="cache/sitk_vs_fit")
    ap.add_argument("--variants", nargs="+", default=["rigid", "similarity", "affine", "affine_corr"])
    ap.add_argument("--sample", type=int, default=30, help="multi-study patients drawn (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pairs", type=int, default=2, help="other studies per patient")
    ap.add_argument("--slices", type=int, default=4, help="slices per pair for the residual search")
    ap.add_argument("--slice-range", type=int, nargs=2, default=[50, 90], dest="slice_range")
    ap.add_argument("--max-angle", type=float, default=15.0, dest="max_angle")
    ap.add_argument("--max-scale", type=float, default=0.12, dest="max_scale")
    ap.add_argument("--dz-reach", type=int, default=8, dest="dz_reach")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--summarize", action="store_true",
                    help="read every shard's CSVs under --out and print the comparison")
    cfg = ap.parse_args()
    cfg.fit_root = os.path.abspath(os.path.expanduser(cfg.fit_root))
    cfg.out = os.path.abspath(os.path.expanduser(cfg.out))
    if cfg.summarize:
        return summarize(cfg)
    from preprocessing.sitk_register import VARIANTS
    bad = [v for v in cfg.variants if v not in VARIANTS]
    if bad:
        raise SystemExit(f"unknown variant(s) {bad}; known: {sorted(VARIANTS)}")
    if not 0 <= cfg.shard < cfg.nshards:
        raise SystemExit(f"--shard {cfg.shard} is outside 0..{cfg.nshards - 1}")
    run_shard(cfg)


if __name__ == "__main__":
    main()
