#!/usr/bin/env python3
"""
List the dump files of the volumes scripts/alias_scan.py ranked worst, for tar.

    python scripts/alias_pick.py --runs trained_nets/mg_recon/brain -k 5 > worst.txt
    tar cf brain_worst.tar -T worst.txt

Reads the scan's own CSVs (`<runs>/alias_scan/scan_R<R>.csv`), so nothing is
re-scanned. A volume is scored by its WORST slice for the target method, and
the top k are taken by each ranking asked for (`--by`, default both of the
scan's aliasing tables: absolute ghost size `gerr`, and `excess` = gerr over
the median of the other methods), then merged.

Every picked volume is listed for EVERY method and EVERY R the scan covered, not
only the R it was flagged at: the viewer shows a volume only when every column
of its config has it, and the R8 and R12 configs are separate. The scan's
report, CSVs and PNGs are listed too. Paths are relative to the current
directory, one per line, as `tar -T` wants them; the picks and their scores go
to stderr.
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import os
import re
import sys

import numpy as np


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default="trained_nets/mg_recon/brain")
    p.add_argument("--scan", help="scan output dir (default <runs>/alias_scan)")
    p.add_argument("-k", type=int, default=5, help="volumes per ranking (default 5)")
    p.add_argument("--by", nargs="+", default=["gerr", "excess"],
                   choices=["gerr", "excess", "nrmse_excess"])
    p.add_argument("--target", default="lpdsnet")
    a = p.parse_args(argv)

    scan = a.scan or os.path.join(a.runs, "alias_scan")
    csvs = sorted(glob.glob(os.path.join(scan, "scan_R*.csv")))
    if not csvs:
        raise SystemExit(f"no scan_R*.csv in {scan} -- run scripts/alias_scan.py --out first")

    picked = collections.OrderedDict()            # volume -> reasons
    methods_at = {}
    for path in csvs:
        R = int(re.search(r"scan_R(\d+)\.csv$", path).group(1))
        by = collections.defaultdict(dict)
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                by[(row["volume"], row["pos"])][row["method"]] = row
        methods_at[R] = sorted({m for v in by.values() for m in v})
        others = [m for m in methods_at[R] if m != a.target]

        def score(rows, key):
            t = rows[a.target]
            if key == "gerr":
                return float(t["gerr"])
            col = "gerr" if key == "excess" else "nrmse"
            return float(t[col]) / (np.median([float(rows[m][col]) for m in others]) + 1e-12)

        for key in a.by:
            worst = collections.defaultdict(lambda: (-np.inf, None))
            for (vol, _), rows in by.items():
                if a.target not in rows or not others:
                    continue
                s = score(rows, key)
                if s > worst[vol][0]:
                    worst[vol] = (s, rows[a.target]["slice"])
            for vol, (s, sl) in sorted(worst.items(), key=lambda kv: -kv[1][0])[:a.k]:
                picked.setdefault(vol, []).append(f"R{R} {key} {s:.4g} @ slice {sl}")

    files = sorted(glob.glob(os.path.join(scan, "*")))
    missing = 0
    for vol in picked:
        for R, methods in sorted(methods_at.items()):
            for m in methods:
                f = os.path.join(a.runs, f"{m}_R{R}", "eval_dump", vol + ".h5")
                if os.path.exists(f):
                    files.append(f)
                else:
                    missing += 1
                    print(f"[missing] {f}", file=sys.stderr)

    print(f"{len(picked)} volume(s), {len(files)} files:", file=sys.stderr)
    for vol, why in picked.items():
        print(f"  {vol}: " + "; ".join(why), file=sys.stderr)
    if missing:
        print(f"  ({missing} expected dump file(s) missing -- that volume will not "
              f"show in the viewer for the affected config)", file=sys.stderr)
    size = sum(os.path.getsize(f) for f in files)
    print(f"  total {size / 2**30:.2f} GiB", file=sys.stderr)
    # Bytes, so Windows text mode cannot turn "\n" into "\r\n" -- tar -T would
    # take the "\r" as part of every file name.
    sys.stdout.flush()
    sys.stdout.buffer.write("".join(f.replace(os.sep, "/") + "\n" for f in files).encode())


if __name__ == "__main__":
    main()
