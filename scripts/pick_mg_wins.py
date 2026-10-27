#!/usr/bin/env python3
"""
Pick the slices where MGLPDS beats BOTH other nets by the most, as figure rows.

    python scripts/pick_mg_wins.py                      # brain + knee, R8 + R12, 3 rows each
    python scripts/pick_mg_wins.py --anatomy knee --R 12 -k 4
    python figures/make_figure.py figures/configs/mgwins_knee_R12.py   # then the figure

Ranks the DUMPED slices with scripts/rank_slices.py (--target mg81v6 --mode
wins: MGLPDS's PSNR minus the better of E2E-VarNet and LPDSNet), the numbers
the viewer shows, keeps the best slice of each of the top k VOLUMES (k rows of
one patient would make a dull figure), and writes them to
figures/rows/mgwins_<anatomy>_R<R>.json -- the rows file of the matching
figures/configs/mgwins_<anatomy>_R<R>.py. Nothing under rows/main_* is touched.
The ranking itself is saved beside it as results/mg_wins/<anatomy>_R<R>.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METHODS = ["varnetmaps", "lpdsnet", "mg81v6"]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--anatomy", nargs="*", default=["brain", "knee"])
    p.add_argument("--R", type=int, nargs="*", default=[8, 12])
    p.add_argument("-k", type=int, default=3, help="rows (distinct volumes) per figure")
    p.add_argument("--metric", default="psnr")
    p.add_argument("--dump-root", default="trained_nets/mg_recon")
    a = p.parse_args(argv)

    os.makedirs(os.path.join(ROOT, "results", "mg_wins"), exist_ok=True)
    for anat in a.anatomy:
        for R in a.R:
            ranking = os.path.join(ROOT, "results", "mg_wins", f"{anat}_R{R}.csv")
            subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "rank_slices.py"),
                            "--anatomy", anat, "--R", str(R), "--dump-root", a.dump_root,
                            "--methods", *METHODS, "--target", "mg81v6", "--mode", "wins",
                            "--metric", a.metric, "--top", "0", "--csv", ranking],
                           check=True, stdout=subprocess.DEVNULL)
            with open(ranking, newline="") as f:
                ranked = list(csv.DictReader(f))          # best margin first
            rows, seen = [], set()
            for r in ranked:
                if r["volume"] not in seen:
                    seen.add(r["volume"])
                    rows.append(r)
                if len(rows) == a.k:
                    break
            out = os.path.join(ROOT, "figures", "rows", f"mgwins_{anat}_R{R}.json")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "w") as f:
                json.dump([{"volume": r["volume"], "slice": int(r["slice"])} for r in rows],
                          f, indent=2)
            print(f"{anat} R{R} -> {os.path.relpath(out, ROOT)}")
            for r in rows:
                print(f"   {r['volume']:<30} slice {int(r['slice']):>3}   "
                      + "  ".join(f"{m} {float(r[m]):.2f}" for m in METHODS)
                      + f"   MGLPDS +{float(r['margin']):.2f} dB")


if __name__ == "__main__":
    sys.exit(main())
