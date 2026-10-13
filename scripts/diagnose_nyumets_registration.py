#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
How well registered is a BUILT NYUMets h5 set -- and what kind of error is left?

    python -m scripts.diagnose_nyumets_registration --root ../datasets/NYUMets_h5

READ-ONLY. For a random sample of multi-study patients it answers, on the slices training
actually uses (`--slice-range`, default [50, 90]):

  1. WHAT WAS DONE. The `reg_method` / `reg_offset` / `reg_ncc` attrs the build stored.
  2. CROSS-STUDY, IN-PLANE. For each (reference study, other study) pair and each sampled slice,
     the masked T1 correlation AS STORED, and after the best in-plane SIMILARITY transform of the
     other study -- rotation, isotropic scale and shift, found by brute force. The transform that
     wins is the error the build left behind:

         rotation  no integer translation removes it (the build's methods are translation-only)
         scale     the headers' "1 mm" is not a measurement, and the build crops in INDEX space
         shift     what a translation-only method should already have removed

  3. CROSS-STUDY, THROUGH-PLANE. Whether the other study's best-matching slice is the one at the
     same `slice_index` (what the loader's guide_slice "index" reads) or a neighbour.
  4. WITHIN A STUDY. The same in-plane search between T1 and CT1 of ONE session. The build
     stacks a session's contrasts by index and does not register them; the T1 & CT1 loader mask
     and the S = CT1 - T1 target both assume they are aligned.

The search is a diagnostic, not a registration: it is 2-D, per slice, on a 2x-downsampled grid.
A pair whose slice PLANES differ (a head tilt) cannot be fixed in-plane at all; it shows up here
as a correlation that stays low whatever the transform.
"""

import argparse
import glob
import math
import os
import statistics

import h5py
import numpy as np
import torch
import torch.nn.functional as F

NORM_KEY = "img_median_mad"
T1, CT1 = 1, 2                       # stored channel order: FLAIR, T1, T1ce, T2


# ---------------------------------------------------------------------------------------------
# the in-plane similarity search
# ---------------------------------------------------------------------------------------------
def masked_ncc(a, b, m, min_px=200):
    """Correlation of two (H, W) images inside the bool mask `m`; NaN under `min_px` pixels."""
    if int(m.sum()) < min_px:
        return float("nan")
    x, y = a[m].float(), b[m].float()
    x, y = x - x.mean(), y - y.mean()
    return float((x * y).sum() / (x.norm() * y.norm()).clamp_min(1e-8))


def warp(img, angle_deg, scale, shift=(0.0, 0.0), mode="bilinear"):
    """Rotate by `angle_deg`, scale by `scale` about the image centre, then shift by
    (dh, dw) pixels. `img` is (N, 1, H, W) with one (angle, scale) per row, or broadcast."""
    N, _, H, W = img.shape
    ang = torch.as_tensor(angle_deg, dtype=torch.float32, device=img.device).reshape(-1)
    sc = torch.as_tensor(scale, dtype=torch.float32, device=img.device).reshape(-1)
    ang, sc = ang.expand(N) * math.pi / 180.0, sc.expand(N)
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.zeros(N, 2, 3, device=img.device)
    # normalised coordinates: x spans W, y spans H -- keep the rotation rigid on a non-square grid
    theta[:, 0, 0], theta[:, 0, 1] = cos, -sin * H / W
    theta[:, 1, 0], theta[:, 1, 1] = sin * W / H, cos
    theta[:, 0, 2] = -2.0 * float(shift[1]) / W
    theta[:, 1, 2] = -2.0 * float(shift[0]) / H
    grid = F.affine_grid(theta, (N, 1, H, W), align_corners=False)
    return F.grid_sample(img, grid, mode=mode, padding_mode="zeros", align_corners=False)


def similarity_search(a, ma, b, mb, angles, scales, down=2):
    """Best in-plane rotation / scale / shift of `b` onto `a`, both (H, W) with bool masks.

    Every (angle, scale) is applied to a downsampled, mask-weighted, zero-mean copy of b; the
    shift for each comes from one FFT cross-correlation with a; the candidate with the highest
    normalised peak wins. -> dict(angle, scale, dh, dw, before, after), `before` / `after` the
    masked correlation at full resolution inside the joint mask, without and with the transform.
    """
    def prep(x, m):
        m = m.float()
        x = (x.float() - (x.float() * m).sum() / m.sum().clamp_min(1)) * m
        return F.avg_pool2d(x[None, None], down) if down > 1 else x[None, None]

    A, B = prep(a, ma), prep(b, mb)
    combos = [(g, s) for g in angles for s in scales]
    ang = torch.tensor([c[0] for c in combos], device=a.device)
    sc = torch.tensor([c[1] for c in combos], device=a.device)
    Bk = warp(B.expand(len(combos), -1, -1, -1), ang, sc)
    FA = torch.fft.rfft2(A)
    C = torch.fft.irfft2(FA * torch.fft.rfft2(Bk).conj(), s=A.shape[-2:])
    C = C / (A.norm() * Bk.flatten(1).norm(dim=1).clamp_min(1e-8)).view(-1, 1, 1, 1)
    peak, flat = C.flatten(1).max(dim=1)
    k = int(peak.argmax())
    h, w = A.shape[-2:]
    ph, pw = divmod(int(flat[k]), w)
    dh = (ph - h if ph > h // 2 else ph) * down           # roll(b, p) lands on a
    dw = (pw - w if pw > w // 2 else pw) * down

    g, s = combos[k]
    moved = warp(b[None, None].float(), g, s, (dh, dw))[0, 0]
    mmoved = warp(mb[None, None].float(), g, s, (dh, dw))[0, 0] > 0.5
    return {"angle": float(g), "scale": float(s), "dh": int(dh), "dw": int(dw),
            "before": masked_ncc(a, b, ma & mb), "after": masked_ncc(a, moved, ma & mmoved)}


def best_dz(a, ma, vol_b, mask_b, j, reach):
    """Offset d in [-reach, reach] of the slice of `vol_b` (n, H, W) around row j that correlates
    best with `a`, on 8x-pooled images so an in-plane offset of a few pixels does not decide it."""
    pool = lambda x: F.avg_pool2d(x[None, None].float(), 8)[0, 0]
    A, MA = pool(a * ma), pool(ma.float()) > 0.5
    best, arg = -2.0, 0
    for d in range(-reach, reach + 1):
        z = j + d
        if not 0 <= z < vol_b.shape[0]:
            continue
        # pooled pixels: a whole head is a few hundred of them, so the floor is low
        s = masked_ncc(A, pool(vol_b[z] * mask_b[z]), MA & (pool(mask_b[z].float()) > 0.5),
                       min_px=16)
        if s == s and s > best:
            best, arg = s, d
    return arg


# ---------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------
class Study:
    def __init__(self, path):
        self.path = path
        with h5py.File(path, "r") as f:
            self.attrs = dict(f.attrs)
            self.index = np.asarray(f["slice_index"]).astype(np.int64)
        base = os.path.basename(os.path.dirname(path))
        self.patient = str(self.attrs.get("patient", base.split("_")[0]))
        self.session = str(self.attrs.get("session", base))

    def read(self, rows, channel, dev):
        """-> (image (len(rows), H, W), bool mask) for stored rows `rows` of one contrast."""
        with h5py.File(self.path, "r") as f:
            rows = [int(r) for r in rows]
            img = np.stack([f[NORM_KEY][r, :, :, channel] for r in rows]).astype(np.float32)
            m = np.stack([f["mask"][r, :, :, 0] for r in rows]) > 0
        return torch.from_numpy(img).to(dev), torch.from_numpy(m).to(dev)


def index_patients(root):
    hits = sorted(glob.glob(os.path.join(os.path.expanduser(root), "*", "*_img.h5")))
    if not hits:
        raise SystemExit(f"no */*_img.h5 under {root} (cwd {os.getcwd()})")
    out = {}
    for p in hits:
        st = Study(p)
        out.setdefault(st.patient, []).append(st)
    return out


def pct(v, q):
    v = sorted(v)
    return v[int(q * (len(v) - 1))]


def describe(name, v, fmt="{:.2f}"):
    v = [x for x in v if x == x]
    if not v:
        return
    print(f"  {name:<34} median {fmt.format(statistics.median(v))}   p90 {fmt.format(pct(v, 0.9))}"
          f"   max {fmt.format(max(v))}")


# ---------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="a built NYUMets h5 directory (or a split of it)")
    ap.add_argument("--sample", type=int, default=30, help="multi-study patients to draw (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pairs", type=int, default=2, help="other studies compared per patient")
    ap.add_argument("--slices", type=int, default=4, help="slices sampled per pair")
    ap.add_argument("--slice-range", type=int, nargs=2, default=[50, 90], dest="slice_range",
                    help="ORIGINAL slice indices to sample from: what the configs train on")
    ap.add_argument("--max-angle", type=float, default=15.0, dest="max_angle")
    ap.add_argument("--angle-step", type=float, default=1.0, dest="angle_step")
    ap.add_argument("--max-scale", type=float, default=0.12, dest="max_scale",
                    help="scales searched are 1 +- this, in steps of --scale-step")
    ap.add_argument("--scale-step", type=float, default=0.02, dest="scale_step")
    ap.add_argument("--dz-reach", type=int, default=8, dest="dz_reach")
    ap.add_argument("--device", default=None)
    ap.add_argument("--csv", default=None, help="write one row per (pair, slice) here")
    cfg = ap.parse_args()
    dev = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    patients = index_patients(cfg.root)
    multi = sorted(k for k, v in patients.items() if len(v) > 1)
    print(f"{len(patients)} patient(s) under {cfg.root}, {len(multi)} with more than one study")
    if not multi:
        raise SystemExit("no multi-study patient: nothing to compare")
    if cfg.sample and cfg.sample < len(multi):
        rng = np.random.default_rng(cfg.seed)
        multi = [multi[i] for i in sorted(rng.choice(len(multi), cfg.sample, replace=False))]
    print(f"sampling {len(multi)} patient(s), up to {cfg.pairs} pair(s) each, {cfg.slices} "
          f"slice(s) per pair in slice_index [{cfg.slice_range[0]}, {cfg.slice_range[1]}), on {dev}")

    # ---- 1. what the build recorded -----------------------------------------------------------
    methods, stored_ncc, no_attr = {}, [], 0
    for pid in multi:
        for st in patients[pid]:
            m = str(st.attrs.get("reg_method", "")) or "(no reg_method attr)"
            methods[m] = methods.get(m, 0) + 1
            if "reg_ncc" in st.attrs:
                v = float(st.attrs["reg_ncc"])
                if v == v and v < 0.999:                  # the reference stores exactly 1
                    stored_ncc.append(v)
            else:
                no_attr += 1
    print("\n1. WHAT THE BUILD RECORDED")
    print("  reg_method: " + ", ".join(f"{k} x{v}" for k, v in sorted(methods.items())))
    if stored_ncc:
        low = sum(v < 0.2 for v in stored_ncc)
        describe("stored reg_ncc (non-reference)", stored_ncc)
        print(f"  {low}/{len(stored_ncc)} below 0.2, the build's own failure gate. NOTHING reads "
              f"this attr at training time.")
    elif no_attr:
        print("  no reg_ncc attr on these files: built before 2026-09-25, or unregistered")

    # ---- 2-4. measure ---------------------------------------------------------------------------
    angles = np.arange(-cfg.max_angle, cfg.max_angle + 1e-6, cfg.angle_step).tolist()
    scales = np.arange(1 - cfg.max_scale, 1 + cfg.max_scale + 1e-6, cfg.scale_step).round(4).tolist()
    small = np.arange(-4.0, 4.0 + 1e-6, 1.0).tolist()            # the within-study search
    rows, within = [], []
    for n_p, pid in enumerate(multi):
        studies = sorted(patients[pid], key=lambda s: s.session)
        ref = studies[0]
        for other in studies[1:1 + cfg.pairs]:
            common = np.intersect1d(ref.index, other.index)
            common = common[(common >= cfg.slice_range[0]) & (common < cfg.slice_range[1])]
            if common.size == 0:
                print(f"  {pid}: {ref.session} / {other.session} share no slice_index in range")
                continue
            pick = common[np.linspace(0, common.size - 1, min(cfg.slices, common.size)).astype(int)]
            ra = [int(np.where(ref.index == z)[0][0]) for z in pick]
            rb = [int(np.where(other.index == z)[0][0]) for z in pick]
            a, ma = ref.read(ra, T1, dev)
            b, mb = other.read(rb, T1, dev)
            lo = max(0, min(rb) - cfg.dz_reach)
            hi = min(len(other.index), max(rb) + cfg.dz_reach + 1)
            vb, vmb = other.read(range(lo, hi), T1, dev)
            for i, z in enumerate(pick):
                r = similarity_search(a[i], ma[i], b[i], mb[i], angles, scales)
                r.update(patient=pid, ref=ref.session, other=other.session, slice_index=int(z),
                         dz=best_dz(a[i], ma[i], vb, vmb, rb[i] - lo, cfg.dz_reach))
                rows.append(r)
        # within the reference study: T1 against CT1, which the build never registers
        zs = ref.index[(ref.index >= cfg.slice_range[0]) & (ref.index < cfg.slice_range[1])]
        if zs.size:
            pick = zs[np.linspace(0, zs.size - 1, min(cfg.slices, zs.size)).astype(int)]
            rr = [int(np.where(ref.index == z)[0][0]) for z in pick]
            a, ma = ref.read(rr, T1, dev)
            c, mc = ref.read(rr, CT1, dev)
            for i in range(len(rr)):
                within.append(similarity_search(a[i], ma[i], c[i], mc[i], small, [1.0]))
        if (n_p + 1) % 10 == 0:
            print(f"  ... {n_p + 1}/{len(multi)} patients")

    ok = [r for r in rows if r["before"] == r["before"] and r["after"] == r["after"]]
    if not ok:
        raise SystemExit("no usable (pair, slice): masks too small in the sampled range")
    n = len(ok)
    frac = lambda cond: f"{sum(1 for r in ok if cond(r))}/{n} ({100 * sum(1 for r in ok if cond(r)) / n:.0f}%)"
    print(f"\n2. CROSS-STUDY, IN-PLANE  ({n} slices over {len({(r['patient'], r['other']) for r in ok})} pairs; T1)")
    describe("masked T1 correlation AS STORED", [r["before"] for r in ok])
    describe("... after the best similarity fit", [r["after"] for r in ok])
    describe("|rotation| left behind (deg)", [abs(r["angle"]) for r in ok], "{:.1f}")
    describe("|scale - 1| left behind", [abs(r["scale"] - 1) for r in ok], "{:.3f}")
    describe("|shift| left behind (px)", [math.hypot(r["dh"], r["dw"]) for r in ok], "{:.1f}")
    print(f"  slices needing |rotation| >= 3 deg : {frac(lambda r: abs(r['angle']) >= 3)}")
    print(f"  slices needing |scale - 1| >= 0.04 : {frac(lambda r: abs(r['scale'] - 1) >= 0.04)}")
    print(f"  slices needing a shift >= 4 px     : {frac(lambda r: math.hypot(r['dh'], r['dw']) >= 4)}")
    print(f"  slices whose correlation the fit raises by >= 0.10 : "
          f"{frac(lambda r: r['after'] - r['before'] >= 0.10)}")
    print(f"  slices still below 0.5 AFTER the fit: {frac(lambda r: r['after'] < 0.5)}"
          f"   <- not an in-plane problem: a different slice plane, or a failed dz")
    print("  For scale: registered scans of one head correlate ~0.9 on T1. The guided prox reads a\n"
          "  15 x 15 latent window at stride 2, i.e. about +-14 px; 5 deg at 90 px from the centre\n"
          "  is 8 px, and a 5% scale error another 4-5 px.")

    print(f"\n3. CROSS-STUDY, THROUGH-PLANE")
    dz = [abs(r["dz"]) for r in ok]
    describe("|slice offset| of the best match", dz, "{:.0f}")
    print(f"  slices whose best match is NOT within 1 of the same slice_index: "
          f"{frac(lambda r: abs(r['dz']) > 1)}")

    win = [r for r in within if r["before"] == r["before"] and r["after"] == r["after"]]
    if win:
        m = len(win)
        off = sum(1 for r in win if math.hypot(r["dh"], r["dw"]) >= 2 or abs(r["angle"]) >= 2)
        print(f"\n4. WITHIN ONE STUDY: T1 vs CT1  ({m} slices)")
        describe("masked correlation as stored", [r["before"] for r in win])
        describe("|shift| preferred (px)", [math.hypot(r["dh"], r["dw"]) for r in win], "{:.1f}")
        describe("|rotation| preferred (deg)", [abs(r["angle"]) for r in win], "{:.1f}")
        print(f"  slices preferring a shift >= 2 px or a rotation >= 2 deg: {off}/{m} "
              f"({100 * off / m:.0f}%)")
        print("  T1 and CT1 differ where there is enhancement, so a small preferred shift is noise;\n"
              "  several pixels on many slices means the session's contrasts are not co-registered.")

    worst = sorted(ok, key=lambda r: r["before"])[:8]
    print("\nlowest-correlation slices as stored (look at these first):")
    for r in worst:
        print(f"  {r['patient']}  {r['ref']} <- {r['other']}  slice_index {r['slice_index']:>3d}: "
              f"{r['before']:.2f} -> {r['after']:.2f}  with rot {r['angle']:+.0f} deg, scale "
              f"{r['scale']:.2f}, shift ({r['dh']:+d}, {r['dw']:+d}), best dz {r['dz']:+d}")

    if cfg.csv:
        import csv
        keys = ["patient", "ref", "other", "slice_index", "before", "after", "angle", "scale",
                "dh", "dw", "dz"]
        with open(cfg.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows({k: r[k] for k in keys} for r in rows)
        print(f"\nwrote {cfg.csv}")
    print("\nREAD-ONLY: nothing was written to the dataset.")


if __name__ == "__main__":
    main()
