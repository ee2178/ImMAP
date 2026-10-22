#!/usr/bin/env python3
"""
The main comparison as a TABLE: mean +- standard error per experiment, with the
same paired significance tests as figures/plot_main_bars.py.

    python figures/table_main.py                     # results/main/volumes.csv
    python figures/table_main.py --nets E2E-VarNet LPDSNet MGLPDS-Small MGLPDS-Big \
        --rename --ref MGLPDS-Big                    # all four, Big as reference

One block per experiment (anatomy x R), one row per net, one column per
metric -- so each experiment is read on its own scale, which the shared axes
of the bar chart could not do.

    entry   mean over the volumes EVERY listed net was scored on (the same
            patients in every row of a block) +- the standard error of that
            mean, std / sqrt(n)
    bold    the best mean in its block and column (direction per metric)
    *       differs from --ref: two-sided Wilcoxon signed-rank, paired by
            volume, Holm-adjusted over that block-and-column's comparisons --
            the test and correction of scripts/wilcoxon_main.py

The standard error describes ONE net's spread over patients; it is not the
test. Two entries whose +- ranges overlap can still differ significantly,
because the test is paired.

Default nets: E2E-VarNet, LPDSNet and MGLPDS-Small shown as "MGLPDS" (the
reference), as in the paper figure. Writes, under figures/out/ (or --out):

    <out>.tex             a booktabs tabular, for \\input in the paper
    <out>_standalone.tex  compiles it on its own
    <out>.md              the same table in Markdown
    <out>.csv             every number in it, unrounded, with the p-values
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys

import numpy as np
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Loading, the test and the correction come from the bar chart (which takes
# them from scripts/wilcoxon_main.py), so the table and the figure cannot
# disagree.
_spec = importlib.util.spec_from_file_location(
    "plot_main_bars", os.path.join(HERE, "plot_main_bars.py"))
_pb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pb)
load, holm, HIGHER_IS_BETTER, ORDER = _pb.load, _pb.holm, _pb.HIGHER_IS_BETTER, _pb.ORDER

# metric -> (column label, decimals, display scale). SSIM and NRMSE are shown as
# PERCENTAGES (x100, 2 decimals): 0.8661 +- 0.0014 reads as 86.61 +- 0.14, the
# same digits without the leading zeros. The CSV keeps the raw values.
HEAD = {"psnr": ("PSNR (dB)", 2, 1), "ssim": ("SSIM (%)", 2, 100),
        "nrmse": ("NRMSE (%)", 2, 100), "lpips": ("LPIPS", 4, 1)}
DEFAULT_NETS = ["E2E-VarNet", "LPDSNet", "MGLPDS-Small"]
DEFAULT_RENAME = ["MGLPDS-Small=MGLPDS"]


def block_title(g, tex):
    """"Brain, R=8" for an (anatomy, R) experiment; the first field as written when R <= 0
    (NYUMets' slice-range groups, which have no acceleration and their own capitalisation)."""
    if int(g[1]) <= 0:
        return str(g[0])
    return f"{g[0].capitalize()}, " + (f"$R{{=}}{g[1]}$" if tex else f"R={g[1]}")


def sem(x):
    """Standard error of the mean: sample std / sqrt(n)."""
    return 0.0 if len(x) < 2 else float(np.std(x, ddof=1) / np.sqrt(len(x)))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", default=os.path.join(ROOT, "results", "main",
                                                            "volumes.csv"))
    p.add_argument("--metrics", nargs="*", default=["psnr", "ssim", "nrmse"])
    p.add_argument("--nets", nargs="*", default=DEFAULT_NETS,
                   help="nets to show (names as in the CSV)")
    p.add_argument("--rename", nargs="*", default=DEFAULT_RENAME, metavar="OLD=NEW",
                   help="relabel nets; pass --rename with nothing to keep CSV names")
    p.add_argument("--ref", default="MGLPDS",
                   help="significance is tested against this net (after renaming)")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--out", default=os.path.join(HERE, "out", "main_table"),
                   help="output path WITHOUT extension")
    a = p.parse_args()

    if not os.path.exists(a.inp):
        raise SystemExit(f"{a.inp} not found -- copy results/main/volumes.csv from the cluster")
    rename = dict(kv.split("=", 1) for kv in a.rename)
    back = {new: old for old, new in rename.items()}
    data, have = load(a.inp, a.nets, rename)
    metrics = [m for m in a.metrics if m in have]
    groups = sorted(data, key=lambda g: (g[0], g[1]))
    order = lambda n: (ORDER.index(back.get(n, n)) if back.get(n, n) in ORDER else 99, n)

    blocks, rows_csv = [], []
    for g in groups:
        nets = sorted(data[g], key=order)
        missing = [rename.get(n, n) for n in a.nets if rename.get(n, n) not in nets]
        if missing:
            print(f"[{g[0]} R{g[1]}] no results for {', '.join(missing)}", file=sys.stderr)
        cells, ns = {}, set()
        for m in metrics:
            vols = sorted(set.intersection(*[{v for v, d in data[g][n].items() if m in d}
                                             for n in nets]))
            if len(vols) < 2:
                continue
            ns.add(len(vols))
            x = {n: np.array([data[g][n][v][m] for v in vols]) for n in nets}
            sign = 1 if HIGHER_IS_BETTER[m] else -1
            best = max(nets, key=lambda n: sign * x[n].mean())
            others = [n for n in nets if n != a.ref] if a.ref in nets else []
            ps = []
            for n in others:
                d = x[a.ref] - x[n]
                ps.append(1.0 if not np.any(d != 0) else float(
                    wilcoxon(x[a.ref], x[n], zero_method="wilcox",
                             alternative="two-sided").pvalue))
            padj = dict(zip(others, holm(ps))) if others else {}
            for n in nets:
                cells[(n, m)] = dict(mean=float(x[n].mean()), se=sem(x[n]),
                                     best=(n == best), p_holm=padj.get(n))
                rows_csv.append(dict(anatomy=g[0], R=g[1], net=n, metric=m, n=len(vols),
                                     mean=x[n].mean(), se=sem(x[n]),
                                     p_holm_vs_ref=padj.get(n, ""), best=int(n == best)))
        blocks.append((g, nets, cells, ns))

    def entry(c, m, tex):
        _, dec, k = HEAD[m]
        mu, se = f"{k * c['mean']:.{dec}f}", f"{k * c['se']:.{dec}f}"
        star = c["p_holm"] is not None and c["p_holm"] < a.alpha
        if tex:
            mu = f"\\mathbf{{{mu}}}" if c["best"] else mu
            return f"${mu} \\pm {se}{'^{*}' if star else ''}$"
        mu = f"**{mu}**" if c["best"] else mu
        return f"{mu} ± {se}{' *' if star else ''}"

    arrow = {m: ("↑" if HIGHER_IS_BETTER[m] else "↓") for m in metrics}
    tarrow = {m: ("$\\uparrow$" if HIGHER_IS_BETTER[m] else "$\\downarrow$") for m in metrics}
    note = (f"Mean $\\pm$ standard error over test volumes. Bold: best in its "
            f"experiment. $^{{*}}$: differs from {a.ref} (paired Wilcoxon signed-rank "
            f"over volumes, Holm-adjusted, $p<{a.alpha:g}$).")

    # ---- LaTeX ---------------------------------------------------------------
    L = ["% Generated by figures/table_main.py. Do not edit by hand: rerun it.",
         "% Needs \\usepackage{booktabs}.",
         f"% Caption text: {note}",
         "\\begin{tabular}{l" + "c" * len(metrics) + "}", "\\toprule",
         "Method & " + " & ".join(f"{HEAD[m][0].replace('%', chr(92) + '%')} {tarrow[m]}"
                                 for m in metrics) + " \\\\"]
    for g, nets, cells, ns in blocks:
        nlab = "/".join(str(n) for n in sorted(ns))
        L += ["\\midrule",
              f"\\multicolumn{{{len(metrics) + 1}}}{{l}}{{\\textit{{{block_title(g, True)} "
              f"($n{{=}}{nlab}$ volumes)}}}} \\\\"]
        for n in nets:
            L.append(f"{n} & " + " & ".join(entry(cells[(n, m)], m, True)
                                            if (n, m) in cells else "--" for m in metrics)
                     + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(f"{a.out}.tex", "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")
    with open(f"{a.out}_standalone.tex", "w", encoding="utf-8", newline="\n") as f:
        f.write("\\documentclass[border=4pt]{standalone}\n\\usepackage{booktabs}\n"
                "\\begin{document}\n"
                f"\\input{{{os.path.basename(a.out)}.tex}}\n\\end{{document}}\n")

    # ---- Markdown --------------------------------------------------------------
    M = []
    for g, nets, cells, ns in blocks:
        nlab = "/".join(str(n) for n in sorted(ns))
        M += [f"**{block_title(g, False)}** (n={nlab} volumes)", "",
              "| Method | " + " | ".join(f"{HEAD[m][0]} {arrow[m]}" for m in metrics) + " |",
              "|---|" + "---|" * len(metrics)]
        M += [f"| {n} | " + " | ".join(entry(cells[(n, m)], m, False)
                                       if (n, m) in cells else "–" for m in metrics) + " |"
              for n in nets]
        M.append("")
    M.append(note.replace("$\\pm$", "±").replace("$^{*}$", "*").replace("$p<", "p < ")
             .replace("$", ""))
    with open(f"{a.out}.md", "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(M) + "\n")

    with open(f"{a.out}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0]))
        w.writeheader()
        w.writerows({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()}
                    for r in rows_csv)

    try:                       # a Windows console (cp1252) has no arrows
        sys.stdout.reconfigure(errors="replace")
    except AttributeError:
        pass
    print("\n".join(M))
    print(f"\nwrote {a.out}.tex, _standalone.tex, .md, .csv")


if __name__ == "__main__":
    main()
