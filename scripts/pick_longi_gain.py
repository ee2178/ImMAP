#!/usr/bin/env python3
"""
Rank the dumped NYUMets slices by how much the longitudinal run beats the baseline.

    python scripts/pick_longi_gain.py                              # the two default runs
    python scripts/pick_longi_gain.py --by enh -k 15               # rank by the enhancement region
    python scripts/pick_longi_gain.py --worst                      # where the prior study HURTS
    python scripts/pick_longi_gain.py --write-rows 3               # top 3 -> the viewer's marked rows

Reads only the HDF5 dumps of scripts/dump_eval_nyumets.py (numpy + h5py; no torch, no GPU), so it
runs on a login node or on the laptop against copied dumps. Both runs must have been dumped in one
sweep -- same volumes, slices, prior study, seed and units; this is checked per volume.

PER SLICE, inside the dump's organ mask (positive gain = the longitudinal run is better):

    d_psnr    PSNR(longi) - PSNR(base), the stored per-slice values
    d_ssim    SSIM(longi) - SSIM(base)
    d_enh     RMSE(base) - RMSE(longi) inside the ENHANCEMENT region: (CT1 - T1)+ > --enh-thr
              (per-volume units, so 0.15 = 15% of the volume's median brain intensity). Enhancement
              is the hard part of T1 -> CT1 and the reason to show the net the prior study.
    d_chg     RMSE(base) - RMSE(longi) inside the CHANGE region: the top (1 - --chg-q) of
              |CT1 - prior CT1|, where the two studies disagree. A gain here means the run used the
              prior study and did NOT just copy it.
    corr      corr(CT1, prior CT1): is the prior slice the same anatomy? A gain on a low-corr
              slice is suspect (misregistration happening to help).

`--by` picks the ranking (psnr, ssim, enh, chg). Slices with fewer than --min-enh enhancing pixels
are never ranked by `enh` (a 3-pixel region makes a huge, meaningless RMSE change), and slices
with less than --min-brain of the image in the mask are skipped (the top/bottom of the head).
`--per-volume` caps how many picks one volume can take, so one patient does not fill the list.

OUTPUT: a table of the top -k picks (stdout), a CSV of every slice (--csv), and the viewer command
that opens the first pick. `--write-rows N` writes the top N as the viewer config's marked rows
(figures/rows/<variant>.json); an existing file is kept unless --force.
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
NYU = os.path.join("trained_nets", "nyumets")
BASE = "I2SB_Unet_NYUMets_CT1_from_all_bs64"
LONGI = "I2SB_Unet_NYUMets_CT1_from_all_plus_otherT1CT1_w2_bs64"
MUST_AGREE = ("seed", "intensity_units", "display_orient", "prior_volume", "slice_range")
RANKS = {"psnr": "d_psnr", "ssim": "d_ssim", "enh": "d_enh", "chg": "d_chg"}


def dump_dir(root, run):
    for d in (os.path.join(root, run, "eval_dump"), os.path.join(root, run)):
        if os.path.isdir(d) and any(f.endswith(".h5") for f in os.listdir(d)):
            return d
    raise SystemExit(f"no dump for {run} under {root} -- run scripts/dump_eval_nyumets.py first")


def read(path):
    with h5py.File(path, "r") as f:
        out = {k: f[k][:] for k in f}
        out["attrs"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                        for k, v in f.attrs.items()}
    return out


def rmse(a, b, w):
    n = w.sum()
    return float(np.sqrt(((a - b) ** 2)[w].mean())) if n else np.nan


def corr(a, b, w):
    if w.sum() < 50:
        return np.nan
    a, b = a[w] - a[w].mean(), b[w] - b[w].mean()
    return float((a * b).sum() / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def score_volume(vol, B, L, args):
    for k in MUST_AGREE:
        if B["attrs"].get(k) != L["attrs"].get(k):
            raise SystemExit(f"{vol}: the two dumps disagree on {k} ({B['attrs'].get(k)} vs "
                             f"{L['attrs'].get(k)}); re-dump both runs in one sweep")
    if not np.array_equal(B["slice_index"], L["slice_index"]) or \
            not np.array_equal(B["reference"], L["reference"]):
        raise SystemExit(f"{vol}: the two dumps hold different slices or targets")
    rows = []
    for i, z in enumerate(B["slice_index"]):
        m = B["organ_mask"][i].astype(bool)
        if m.mean() < args.min_brain:
            continue
        ct1, t1, pri = B["reference"][i], B["zero_filled"][i], B["prior_ct1"][i]
        xb, xl = B["recon"][i], L["recon"][i]
        enh = m & ((ct1 - t1) > args.enh_thr)
        d = np.abs(ct1 - pri)
        chg = m & (d >= np.quantile(d[m], args.chg_q)) if m.any() else m
        r = dict(volume=vol, slice=int(z), prior=B["attrs"].get("prior_volume", ""),
                 psnr_base=float(B["psnr_slice"][i]), psnr_longi=float(L["psnr_slice"][i]),
                 d_psnr=float(L["psnr_slice"][i] - B["psnr_slice"][i]),
                 d_ssim=float(L["ssim_slice"][i] - B["ssim_slice"][i]),
                 n_enh=int(enh.sum()),
                 d_enh=(rmse(xb, ct1, enh) - rmse(xl, ct1, enh)) if enh.sum() >= args.min_enh
                 else np.nan,
                 d_chg=rmse(xb, ct1, chg) - rmse(xl, ct1, chg),
                 corr=corr(ct1, pri, m), brain=float(m.mean()))
        rows.append(r)
    return rows


def pick(rows, key, k, per_volume, worst):
    ok = [r for r in rows if np.isfinite(r[key])]
    ok.sort(key=lambda r: r[key], reverse=not worst)
    out, used = [], {}
    for r in ok:
        if used.get(r["volume"], 0) >= per_volume:
            continue
        out.append(r)
        used[r["volume"]] = used.get(r["volume"], 0) + 1
        if len(out) == k:
            break
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=os.path.join(HERE, NYU), help="where the run dirs live")
    p.add_argument("--base", default=BASE, help="the no-prior run (dump dir name)")
    p.add_argument("--longi", default=LONGI, help="the longitudinal run")
    p.add_argument("--by", choices=sorted(RANKS), default="psnr", help="ranking (default psnr)")
    p.add_argument("-k", type=int, default=10, help="how many picks to list")
    p.add_argument("--per-volume", type=int, default=2, help="at most this many picks per volume")
    p.add_argument("--worst", action="store_true", help="rank where the longitudinal run LOSES")
    p.add_argument("--enh-thr", type=float, default=0.15,
                   help="(CT1 - T1)+ above this is enhancement (per-volume units)")
    p.add_argument("--min-enh", type=int, default=30, help="min enhancing pixels to rank by enh")
    p.add_argument("--chg-q", type=float, default=0.98, help="change region = top 1-q of |CT1-prior|")
    p.add_argument("--min-brain", type=float, default=0.05,
                   help="skip slices whose mask covers less than this fraction of the image")
    p.add_argument("--csv", default=os.path.join(HERE, "results", "nyumets_eval", "longi_gain.csv"),
                   help="every scored slice ('' = none)")
    p.add_argument("--write-rows", type=int, default=0,
                   help="write the top N picks as the viewer's marked rows")
    p.add_argument("--variant", default="nyumets_longi", help="viewer config the rows are for")
    p.add_argument("--force", action="store_true", help="overwrite an existing rows file")
    a = p.parse_args(argv)

    db, dl = dump_dir(a.root, a.base), dump_dir(a.root, a.longi)
    vb = {f[:-3] for f in os.listdir(db) if f.endswith(".h5")}
    vl = {f[:-3] for f in os.listdir(dl) if f.endswith(".h5")}
    vols = sorted(vb & vl)
    if not vols:
        raise SystemExit("the two runs have no dumped volume in common")
    if vb ^ vl:
        print(f"note: {len(vb ^ vl)} volume(s) dumped for only one run are ignored", file=sys.stderr)

    rows = []
    for v in vols:
        rows += score_volume(v, read(os.path.join(db, v + ".h5")),
                             read(os.path.join(dl, v + ".h5")), a)
    if not rows:
        raise SystemExit("no slice passed --min-brain")

    if a.csv:
        os.makedirs(os.path.dirname(os.path.abspath(a.csv)), exist_ok=True)
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    # ---- overview: is the gain systematic, and where does it live? ----
    def summ(key):
        v = np.array([r[key] for r in rows], float)
        v = v[np.isfinite(v)]
        return v.mean(), np.median(v), 100 * (v > 0).mean(), v.size
    print(f"{len(vols)} volumes, {len(rows)} slices | longitudinal = {a.longi}\n"
          f"                 baseline     = {a.base}\n")
    print("%-8s %10s %10s %10s %7s" % ("gain", "mean", "median", "% better", "slices"))
    for name, key in (("psnr dB", "d_psnr"), ("ssim", "d_ssim"), ("rmse enh", "d_enh"),
                      ("rmse chg", "d_chg")):
        v = np.array([r[key] for r in rows], float)
        if not np.isfinite(v).any():
            print("%-8s %10s %10s %10s %7d" % (name, "n/a", "n/a", "n/a", 0))
            continue
        mn, md, pc, n = summ(key)
        print("%-8s %+10.4f %+10.4f %9.0f%% %7d" % (name, mn, md, pc, n))

    key = RANKS[a.by]
    top = pick(rows, key, a.k, a.per_volume, a.worst)
    print(f"\n{'worst' if a.worst else 'top'} {len(top)} by {key} "
          f"(<= {a.per_volume} per volume; gains > 0 favour the longitudinal run)")
    print("%3s %-24s %5s %8s %8s %8s %8s %8s %6s %6s" % (
        "#", "volume", "slice", "psnr_b", "d_psnr", "d_ssim", "d_enh", "d_chg", "n_enh", "corr"))
    for i, r in enumerate(top):
        print("%3d %-24s %5d %8.2f %+8.2f %+8.4f %+8.4f %+8.4f %6d %6.2f" % (
            i, r["volume"][:24], r["slice"], r["psnr_base"], r["d_psnr"], r["d_ssim"],
            r["d_enh"], r["d_chg"], r["n_enh"], r["corr"]))
    low = [r for r in top if np.isfinite(r["corr"]) and r["corr"] < 0.6]
    if low:
        print(f"\n{len(low)} pick(s) have corr(CT1, prior CT1) < 0.6: the prior slice may not be "
              f"the same anatomy, so treat those gains with suspicion")
    if a.csv:
        print(f"\nevery slice -> {a.csv}")
    if top:
        print(f"\nopen the first pick:\n  python -m figures.webviewer figures/configs/{a.variant}.py "
              f"--volume {top[0]['volume']} --slice {top[0]['slice']}")

    if a.write_rows:
        path = os.path.join(HERE, "figures", "rows", f"{a.variant}.json")
        if os.path.exists(path) and not a.force:
            print(f"\n{path} exists (your own picks?) -- not overwritten; pass --force")
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump([dict(volume=r["volume"], slice=r["slice"])
                           for r in top[:a.write_rows]], f, indent=2)
            print(f"\nwrote {min(a.write_rows, len(top))} row(s) -> {path}")


if __name__ == "__main__":
    main()
