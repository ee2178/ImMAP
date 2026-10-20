#!/usr/bin/env python3
"""
Paired Wilcoxon signed-rank tests between the nets, volume by volume.

    python scripts/wilcoxon_main.py                         # results/main/volumes.csv
    python scripts/wilcoxon_main.py --ref MGLPDS-Big        # one net against the rest
    python scripts/wilcoxon_main.py --unit slices           # slice-level (see below)

Reads what `scripts/eval_per_volume.py --combine` writes. For each (anatomy, R,
metric) family it tests every pair of nets -- or, with --ref, the reference
against each other net -- on the volumes BOTH were scored on.

Why volumes, not slices
-----------------------
Slices of one volume share a patient, a coil set and a noise floor, so they are
not independent; testing on slices treats 13 knee slices of one patient as 13
patients and the p-values come out far too small. The volume is the unit that
was sampled independently, so it is the default. `--unit slices` exists to
show the difference, not to report.

Reported per pair (A vs B, difference = A - B)
---------------------------------------------
    n             volumes in common (pairs with a zero difference are dropped
                  by the test, per Wilcoxon's original method)
    mean_A/B      the two nets' means over those volumes
    median_diff   median of A - B
    A_better      fraction of volumes on which A beats B (direction per metric:
                  PSNR/SSIM higher is better, NRMSE/LPIPS lower)
    r_rb          matched-pairs rank-biserial correlation, in [-1, 1], signed
                  so that POSITIVE MEANS A IS BETTER -- an effect size, which
                  a p-value on hundreds of volumes is not
    p             two-sided Wilcoxon signed-rank
    p_holm        Holm-adjusted over the tests in its (anatomy, R, metric) family
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys

import numpy as np
from scipy.stats import rankdata, wilcoxon

HIGHER_IS_BETTER = {"psnr": True, "ssim": True, "nrmse": False, "lpips": False}
ORDER = ["E2E-VarNet", "LPDSNet", "MGLPDS-Small", "MGLPDS-Big"]

# Runs that share a display label (brain's MGLPDS-Big stopgap `mglpds` and the
# real `mg169v6`): keep the FIRST of these present, never pool the two.
PREFERRED_CELLS = ["mg169v6", "mglpds"]


def _cell(run):
    head, sep, tail = run.rpartition("_R")
    return head if sep and tail.isdigit() else run


def one_run_per_net(rows):
    """Drop rows of a less-preferred run where two runs share (anatomy, R, net)."""
    runs = {}
    for r in rows:
        runs.setdefault((r["anatomy"], r["R"], r["net"]), set()).add(r.get("run", ""))
    keep = {}
    for key, rs in runs.items():
        rank = sorted(rs, key=lambda x: (PREFERRED_CELLS.index(_cell(x))
                                         if _cell(x) in PREFERRED_CELLS else 99, x))
        keep[key] = rank[0]
        if len(rs) > 1:
            print(f"[{key[0]} R{key[1]}] {key[2]}: using {rank[0]}, ignoring "
                  f"{', '.join(rank[1:])}", file=sys.stderr)
    return [r for r in rows if r.get("run", "") == keep[(r["anatomy"], r["R"], r["net"])]]


def holm(ps):
    """Holm step-down adjusted p-values, in the input order."""
    ps = np.asarray(ps, dtype=float)
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, (m - k) * ps[i])
        adj[i] = min(1.0, running)
    return adj


def rank_biserial(d):
    """Matched-pairs rank-biserial: (sum of + ranks - sum of - ranks) / total."""
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def load(path, unit):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path} is empty")
    rows = one_run_per_net(rows)
    metrics = [m for m in HIGHER_IS_BETTER if m in rows[0]]
    key = (lambda r: r["volume"]) if unit == "volumes" else \
          (lambda r: (r["volume"], r["slice"]))
    data = {}                       # (anatomy, R) -> net -> {unit: {metric: v}}
    for r in rows:
        d = data.setdefault((r["anatomy"], int(r["R"])), {}).setdefault(r["net"], {})
        d[key(r)] = {m: float(r[m]) for m in metrics if r.get(m) not in (None, "")}
    return data, metrics


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", help="default results/main/<unit>.csv")
    p.add_argument("--unit", choices=["volumes", "slices"], default="volumes")
    p.add_argument("--ref", help="test this net against every other one only")
    p.add_argument("--metrics", nargs="*", help="default: every metric in the file")
    p.add_argument("--out", help="default results/main/wilcoxon_<unit>.csv")
    args = p.parse_args()

    inp = args.inp or os.path.join("results", "main", f"{args.unit}.csv")
    out = args.out or os.path.join("results", "main", f"wilcoxon_{args.unit}.csv")
    data, metrics = load(inp, args.unit)
    if args.metrics:
        metrics = [m for m in metrics if m in args.metrics]

    results = []
    for (anatomy, R), nets in sorted(data.items()):
        names = sorted(nets, key=lambda n: (ORDER.index(n) if n in ORDER else 99, n))
        if args.ref:
            if args.ref not in nets:
                print(f"[{anatomy} R{R}] no {args.ref!r}; nets: {', '.join(names)}",
                      file=sys.stderr)
                continue
            pairs = [(args.ref, n) for n in names if n != args.ref]
        else:
            pairs = list(itertools.combinations(names, 2))

        for m in metrics:
            sign = 1.0 if HIGHER_IS_BETTER[m] else -1.0
            fam = []
            for a, b in pairs:
                common = sorted(k for k in set(nets[a]) & set(nets[b])
                                if m in nets[a][k] and m in nets[b][k])
                if len(common) < 6:
                    print(f"[{anatomy} R{R} {m}] {a} vs {b}: only {len(common)} "
                          f"common {args.unit} -- skipped", file=sys.stderr)
                    continue
                xa = np.array([nets[a][k][m] for k in common])
                xb = np.array([nets[b][k][m] for k in common])
                d = xa - xb
                pval = (1.0 if not np.any(d != 0)
                        else float(wilcoxon(xa, xb, zero_method="wilcox",
                                            alternative="two-sided").pvalue))
                fam.append(dict(
                    anatomy=anatomy, R=R, metric=m, A=a, B=b, n=len(common),
                    mean_A=xa.mean(), mean_B=xb.mean(),
                    median_diff=float(np.median(d)),
                    A_better=float(np.mean(sign * d > 0)),
                    r_rb=sign * rank_biserial(d), p=pval))
            for row, ph in zip(fam, holm([r["p"] for r in fam]) if fam else []):
                row["p_holm"] = float(ph)
            results += fam

    if not results:
        raise SystemExit("no pair had enough common volumes to test")

    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        for r in results:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v)
                        for k, v in r.items()})

    cur = None
    for r in results:
        fam = (r["anatomy"], r["R"], r["metric"])
        if fam != cur:
            cur = fam
            print(f"\n{r['anatomy']} R={r['R']}  {r['metric']}  "
                  f"({'higher' if HIGHER_IS_BETTER[r['metric']] else 'lower'} is better)")
            print(f"  {'A':<13} {'B':<13} {'n':>4} {'mean A':>9} {'mean B':>9} "
                  f"{'med A-B':>9} {'A better':>8} {'r_rb':>6} {'p_holm':>9}")
        star = "*" if r["p_holm"] < 0.05 else " "
        print(f"  {r['A']:<13} {r['B']:<13} {r['n']:>4} {r['mean_A']:>9.4f} "
              f"{r['mean_B']:>9.4f} {r['median_diff']:>+9.4f} {r['A_better']:>8.0%} "
              f"{r['r_rb']:>+6.2f} {r['p_holm']:>9.2g}{star}")
    print(f"\nwrote {len(results)} tests to {out}")


if __name__ == "__main__":
    main()
