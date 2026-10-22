"""
figures/plot_noise_sweep.py: the noise-level generalization summary.

Run with `python -m tests.test_noise_sweep`. numpy + matplotlib; no torch.

1. per (anatomy, R, net, sigma) the mean and SEM over volumes;
2. volumes are MATCHED across nets at each sigma (a net missing a volume drops
   it for everyone at that sigma);
3. the difference to --ref is paired and signed so that up is better -- for
   NRMSE too (lower NRMSE -> positive difference);
4. the figure and the CSV are written.
"""

from __future__ import annotations

import csv
import os
import shutil
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "figures"))
import plot_noise_sweep as ns                                   # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(f"[{'ok ' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def write(root, sigma, rows):
    d = os.path.join(root, f"s{sigma}")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "volumes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def row(net, run, vol, psnr, nrmse):
    return dict(anatomy="knee", R=8, net=net, run=run, volume=vol, n_slices=13,
                psnr=psnr, ssim=0.8, nrmse=nrmse)


def main():
    tmp = tempfile.mkdtemp()
    try:
        root = os.path.join(tmp, "sweep")
        # sigma 0.01: three volumes; LPDSNet misses vol2 -> only vol0, vol1 count
        write(root, 0.01, [row("MGLPDS-Big", "mg169v6_R8", "vol0", 30, 0.10),
                           row("MGLPDS-Big", "mg169v6_R8", "vol1", 32, 0.08),
                           row("MGLPDS-Big", "mg169v6_R8", "vol2", 50, 0.01),
                           row("LPDSNet", "lpdsnet_R8", "vol0", 29, 0.12),
                           row("LPDSNet", "lpdsnet_R8", "vol1", 30, 0.11)])
        write(root, 0.05, [row("MGLPDS-Big", "mg169v6_R8", "vol0", 25, 0.20),
                           row("MGLPDS-Big", "mg169v6_R8", "vol1", 27, 0.18),
                           row("LPDSNet", "lpdsnet_R8", "vol0", 22, 0.30),
                           row("LPDSNet", "lpdsnet_R8", "vol1", 23, 0.26)])
        out = os.path.join(tmp, "out", "ns")
        ns.main(["--root", root, "--metrics", "psnr", "nrmse", "--ref", "MGLPDS-Big",
                 "--out", out])
        rows = list(csv.DictReader(open(out + ".csv")))
        get = lambda net, s, m: next(r for r in rows if r["net"] == net
                                     and float(r["sigma"]) == s and r["metric"] == m)
        big, lp = get("MGLPDS-Big", 0.01, "psnr"), get("LPDSNet", 0.01, "psnr")
        check("volumes matched across nets (vol2 dropped)",
              int(big["n"]) == 2 and abs(float(big["mean"]) - 31.0) < 1e-6, str(big))
        check("SEM over the matched volumes",
              abs(float(big["sem"]) - np.std([30, 32], ddof=1) / np.sqrt(2)) < 1e-6)
        check("paired PSNR difference to the reference (LPDSNet - Big)",
              abs(float(lp["diff_vs_ref"]) - (-1.5)) < 1e-6, lp["diff_vs_ref"])
        lpn = get("LPDSNet", 0.05, "nrmse")
        check("NRMSE difference sign-flipped: worse NRMSE -> negative",
              abs(float(lpn["diff_vs_ref"]) - (-(0.28 - 0.19))) < 1e-6, lpn["diff_vs_ref"])
        check("figure written", all(os.path.exists(f"{out}.{e}") for e in ("png", "pdf")))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{'FAILED: ' + ', '.join(FAIL) if FAIL else 'all checks passed'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
