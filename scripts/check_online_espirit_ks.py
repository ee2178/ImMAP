#!/usr/bin/env python3
"""
Online ESPIRiT kernel size on REAL slices: accuracy and cost, e.g. 2 against 4.

    python scripts/check_online_espirit_ks.py --config config/brain/mg/lpdsnet_R12.json

Loads a few validation slices through the config's own data block, builds the
config's sampling mask, and estimates the operator's coil maps from each masked
measurement at every `--ks`, exactly as training does (`physics/online_smaps.py`,
the config's `mri.online_smaps_kws` with only `kernel_size` replaced).

Per kernel size, averaged over the slices:

  ms          one estimate, GPU-synchronised (after a warm-up call)
  kept        row-space kernels kept (sets the FFT count and the memory)
  peak GB     peak GPU memory of one estimate
  coh stored  |sum_c conj(s_c) s_c^stored|, inside the stored maps' support: 1 =
              the same maps up to a per-pixel phase. The stored maps are the
              ones the SENSE ground truth was built from.
  consist.    || coil - s (s^H coil) || / || coil || over that support, on the
              FULLY sampled coil images: how well the maps explain the data
  |x| vs GT   NRMSE of |s^H coil| against the stored ground-truth |image|

and, for every pair of kernel sizes, how far apart the two map sets are
(coherence, and the NRMSE between their SENSE combinations) -- the direct answer
to "does kernel 2 give the same maps as kernel 4?".

No network, no training. Needs the dataset and a GPU (CPU works, slowly).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from operators.fourier import ifftc                             # noqa: E402
from physics.mask import (effective_accel, make_acc_mask,        # noqa: E402
                          resolve_acs_lines)
from physics.online_smaps import online_smaps                    # noqa: E402


def load_slices(cfg, split, n):
    from datasets.fastmri.loader import FastMRIDataset
    d = dict(cfg["data"][split])
    for k in ("name", "task", "batch_size"):
        d.pop(k, None)
    ds = FastMRIDataset(task="recon", **d)
    if len(ds) == 0:
        raise SystemExit(f"no volumes under {d.get('smap_root')!r}")
    step = max(1, len(ds) // n)
    out = []
    for i in range(min(n, len(ds))):
        kspace, smaps, image, _om, _pad = ds[(i * step) % len(ds)]
        out.append((kspace[None] if kspace.dim() == 3 else kspace,
                    smaps[None] if smaps.dim() == 3 else smaps,
                    image.reshape(1, 1, *image.shape[-2:])))
    return out


class KeptKernels:
    """Record how many row-space kernels ESPIRiT pushed to the image domain."""

    def __enter__(self):
        self.K, self._real = None, torch.fft.ifft2

        def spy(x, *a, **k):
            if x.dim() == 5:                       # (B, C, K, ks, ks)
                self.K = int(x.shape[2])
            return self._real(x, *a, **k)

        torch.fft.ifft2 = spy
        return self

    def __exit__(self, *exc):
        torch.fft.ifft2 = self._real


def now(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter()


def nrmse(a, b, m):
    return float((a - b)[m].norm() / b[m].norm().clamp_min(1e-30))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="any mg config (data + mri blocks)")
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--n", type=int, default=6, help="slices, spread over the split")
    ap.add_argument("--split", default="val", choices=("train", "val"))
    ap.add_argument("--R", type=int, default=None, help="default: the config's mri.R")
    ap.add_argument("--reps", type=int, default=3, help="timed estimates per slice")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    mri = cfg["mri"]
    R = args.R if args.R is not None else mri["R"]
    device = torch.device(args.device)
    base_kws = dict(mri.get("online_smaps_kws") or {})
    method = mri.get("online_smaps") or "espirit"
    print(f"{args.config}\n{method}, kws {base_kws} (kernel_size replaced by --ks {args.ks}), "
          f"R={R}, {args.n} {args.split} slices, device {device}\n")

    slices = load_slices(cfg, args.split, args.n)
    per = {ks: {k: [] for k in ("ms", "kept", "gb", "coh", "coh_p1", "cons", "xerr")}
           for ks in args.ks}
    pair = {(a, b): {"coh": [], "x": []} for i, a in enumerate(args.ks) for b in args.ks[i + 1:]}

    for si, (kspace, smaps_ref, image) in enumerate(slices):
        kspace, smaps_ref, image = (t.to(device) for t in (kspace, smaps_ref, image))
        H, W = kspace.shape[-2:]
        cf = mri.get("center_frac")
        mask = make_acc_mask((H, W), accel=R, acs_lines=None if cf is not None else mri["acs_lines"],
                             dim=1, mode=mri.get("mask_dist", "uniform"),
                             center_frac=cf, adjust_accel=bool(mri.get("adjust_accel", False)),
                             device=device)
        acs = resolve_acs_lines(W, mri.get("acs_lines"), cf)
        y = mask * kspace
        coil = ifftc(kspace)                                   # fully sampled coil images
        support = smaps_ref.abs().sum(1)[0] > 0
        x_gt = image[0, 0].abs()
        if si == 0:
            print(f"slice shape {H}x{W}, {kspace.shape[1]} coils, ACS {acs} lines, "
                  f"effective R {float(effective_accel(mask)):.2f}\n")

        got = {}
        for ks in args.ks:
            kws = dict(base_kws, kernel_size=ks, acs_lines=acs)
            with torch.no_grad():
                online_smaps(y, mask, method=method, **kws)             # warm-up
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                ts = []
                with KeptKernels() as kk:
                    for _ in range(args.reps):
                        t0 = now(device)
                        s = online_smaps(y, mask, method=method, **kws)
                        ts.append(1e3 * (now(device) - t0))
            got[ks] = s
            gb = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else float("nan")
            m = support & (s.abs().sum(1)[0] > 0)
            coh = (s.conj() * smaps_ref).sum(1)[0].abs()[m]
            x = (s.conj() * coil).sum(1, keepdim=True)
            cons = float(((coil - s * x)[0][:, m]).norm() / coil[0][:, m].norm())
            p = per[ks]
            p["ms"].append(statistics.median(ts)); p["kept"].append(kk.K or 0); p["gb"].append(gb)
            p["coh"].append(float(coh.mean())); p["coh_p1"].append(float(coh.quantile(0.01)))
            p["cons"].append(cons); p["xerr"].append(nrmse(x[0, 0].abs(), x_gt, m))
        for (a, b), acc in pair.items():
            m = support & (got[a].abs().sum(1)[0] > 0) & (got[b].abs().sum(1)[0] > 0)
            acc["coh"].append(float((got[a].conj() * got[b]).sum(1)[0].abs()[m].mean()))
            xa = (got[a].conj() * coil).sum(1)[0].abs()
            xb = (got[b].conj() * coil).sum(1)[0].abs()
            acc["x"].append(nrmse(xa, xb, m))
        del got

    mean = statistics.fmean
    print(f"  {'ks':>3} {'ms':>8} {'kept':>6} {'peak GB':>8} | {'coh stored':>10} {'(worst 1%)':>10}"
          f" {'consist.':>9} {'|x| vs GT':>10}")
    for ks in args.ks:
        p = per[ks]
        print(f"  {ks:>3} {mean(p['ms']):>8.1f} {mean(p['kept']):>6.1f} {mean(p['gb']):>8.2f} |"
              f" {mean(p['coh']):>10.4f} {mean(p['coh_p1']):>10.4f} {mean(p['cons']):>9.4f}"
              f" {mean(p['xerr']):>10.4f}")
    print(f"\n  per-slice ms: " + "   ".join(
        f"ks={ks}: " + " ".join(f"{t:.0f}" for t in per[ks]["ms"]) for ks in args.ks))
    if pair:
        print("\n  between kernel sizes (inside the stored support):")
        for (a, b), acc in pair.items():
            print(f"    ks={a} vs ks={b}: map coherence {mean(acc['coh']):.4f}, "
                  f"|x| NRMSE {mean(acc['x']):.4f}, "
                  f"time {mean(per[a]['ms']) / mean(per[b]['ms']):.2f}x")
    print("\n  The stored maps came from a 20x20 ACS of fully sampled data with their own "
          "kernel and a hard\n  support, so `coh stored` < 1 for every kernel size; compare "
          "the ROWS, and read the pairwise line.")


if __name__ == "__main__":
    main()
