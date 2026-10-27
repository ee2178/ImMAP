#!/usr/bin/env python3
"""
The MGLPDS ablations as a TABLE: the width sweep (subbands M at a fixed number
of V-cycles) and the depth sweep (V-cycles V at a fixed M), from the
per-volume metrics torch/exp16_ablation_eval.sbatch writes.

    python figures/table_ablation.py                 # results/ablation16_knee_R8/volumes.csv
    python figures/table_ablation.py --in results/ablation_knee_R8/volumes.csv \
        --out figures/out/ablation14_table          # exp14's (pre-exp16) runs

Each net is read off its cell name, mg<M>v<V>[suffix] (mg121v6vtxAnestU ->
M=121, V=6), so the tree must hold ONE run per (M, V) -- keep different
families (exp14 vs exp16) in different --out trees of the eval.

    entry   mean over the volumes EVERY net in the table was scored on
            +- the standard error of that mean, as in figures/table_main.py
    bold    the best mean in its sweep and column
    *       differs from the reference cell (--ref, default M=81 V=6, the
            shipped MGLPDS-Small): two-sided Wilcoxon signed-rank, paired by
            volume, Holm-adjusted over that column's comparisons
    Params  trainable parameters, from the eval's <anatomy>/<run>.meta.json

The reference row sits in both sweeps (it is the V=6 point of the width sweep
and the M=81 point of the depth sweep). Writes, under figures/out/ (or --out):

    <out>.tex             a booktabs tabular, for \\input in the paper
    <out>_standalone.tex  compiles it on its own
    <out>.md              the same table in Markdown
    <out>.csv             every number in it, unrounded, with the p-values
"""

from __future__ import annotations

import argparse
import csv
import glob
import importlib.util
import json
import os
import re
import sys

import numpy as np
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Loading, the test, the correction and the number formats come from the main
# table, so the two tables cannot disagree.
_spec = importlib.util.spec_from_file_location(
    "table_main", os.path.join(HERE, "table_main.py"))
_tm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tm)
load, holm, HIGHER_IS_BETTER = _tm.load, _tm.holm, _tm.HIGHER_IS_BETTER
HEAD, sem, block_title = _tm.HEAD, _tm.sem, _tm.block_title

CELL = re.compile(r"^mg(\d+)v(\d+)")


def mv_of(net):
    """`mg121v6vtxAnestU` -> (121, 6); None for anything that is not an mg cell."""
    m = CELL.match(net)
    return (int(m[1]), int(m[2])) if m else None


def n_params_of(root):
    """(anatomy, R, net) -> trainable parameters, from the eval's meta files."""
    out = {}
    for mp in glob.glob(os.path.join(root, "*", "*.meta.json")):
        with open(mp) as f:
            meta = json.load(f)
        out[(meta["anatomy"], int(meta["R"]), meta["label"])] = meta.get("n_params")
    return out


def fmt_params(n):
    if not n:
        return "--"
    return f"{n / 1e6:.2f}M" if n >= 1e6 else f"{n / 1e3:.0f}k"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", default=os.path.join(
        ROOT, "results", "ablation16_knee_R8", "volumes.csv"))
    p.add_argument("--metrics", nargs="*", default=["psnr", "ssim", "nrmse"])
    p.add_argument("--width-v", type=int, default=6, help="V of the width sweep")
    p.add_argument("--depth-m", type=int, default=81, help="M of the depth sweep")
    p.add_argument("--ref", type=int, nargs=2, default=[81, 6], metavar=("M", "V"),
                   help="significance is tested against this cell")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--out", default=os.path.join(HERE, "out", "ablation_table"),
                   help="output path WITHOUT extension")
    a = p.parse_args()

    if not os.path.exists(a.inp):
        raise SystemExit(f"{a.inp} not found -- run torch/exp16_ablation_eval.sbatch "
                         f"and copy its volumes.csv (and the */*.meta.json) down")
    data, have = load(a.inp)
    metrics = [m for m in a.metrics if m in have]
    params = n_params_of(os.path.dirname(os.path.abspath(a.inp)))
    ref_mv = tuple(a.ref)

    blocks, rows_csv = [], []
    for g in sorted(data, key=lambda g: (g[0], g[1])):
        nets = {}
        for n in data[g]:
            mv = mv_of(n)
            if mv is None:
                print(f"[{g[0]} R{g[1]}] {n}: not an mg<M>v<V> cell, left out",
                      file=sys.stderr)
            elif mv in nets:
                raise SystemExit(f"[{g[0]} R{g[1]}] {nets[mv]} and {n} are both "
                                 f"M={mv[0]} V={mv[1]} -- score them into separate trees")
            else:
                nets[mv] = n
        sweeps = [(f"width ($V{{=}}{a.width_v}$)", f"width (V={a.width_v})",
                   sorted(mv for mv in nets if mv[1] == a.width_v)),
                  (f"depth ($M{{=}}{a.depth_m}$)", f"depth (M={a.depth_m})",
                   sorted(mv for mv in nets if mv[0] == a.depth_m))]
        shown = sorted({mv for *_, mvs in sweeps for mv in mvs})
        for mv in sorted(set(nets) - set(shown)):
            print(f"[{g[0]} R{g[1]}] {nets[mv]}: in neither sweep, left out", file=sys.stderr)
        ref = ref_mv if ref_mv in shown else None
        if ref is None:
            print(f"[{g[0]} R{g[1]}] no M={ref_mv[0]} V={ref_mv[1]} run: no significance "
                  f"marks", file=sys.stderr)

        x, ns, padj = {}, set(), {}
        for m in metrics:
            vols = sorted(set.intersection(*[{v for v, d in data[g][nets[mv]].items()
                                              if m in d} for mv in shown]))
            if len(vols) < 2:
                continue
            ns.add(len(vols))
            for mv in shown:
                x[(mv, m)] = np.array([data[g][nets[mv]][v][m] for v in vols])
            if ref is not None:
                others = [mv for mv in shown if mv != ref]
                ps = []
                for mv in others:
                    d = x[(ref, m)] - x[(mv, m)]
                    ps.append(1.0 if not np.any(d != 0) else float(
                        wilcoxon(x[(ref, m)], x[(mv, m)], zero_method="wilcox",
                                 alternative="two-sided").pvalue))
                padj.update({(mv, m): q for mv, q in zip(others, holm(ps))})

        for tex_t, md_t, mvs in sweeps:
            cells = {}
            for m in metrics:
                if (mvs[0] if mvs else None, m) not in x:
                    continue
                sign = 1 if HIGHER_IS_BETTER[m] else -1
                best = max(mvs, key=lambda mv: sign * x[(mv, m)].mean())
                for mv in mvs:
                    c = dict(mean=float(x[(mv, m)].mean()), se=sem(x[(mv, m)]),
                             best=(mv == best), p_holm=padj.get((mv, m)))
                    cells[(mv, m)] = c
                    rows_csv.append(dict(anatomy=g[0], R=g[1], sweep=md_t, M=mv[0], V=mv[1],
                                         net=nets[mv],
                                         n_params=params.get((g[0], g[1], nets[mv]), ""),
                                         metric=m, n=len(x[(mv, m)]), mean=c["mean"],
                                         se=c["se"], p_holm_vs_ref=c["p_holm"] or "",
                                         best=int(c["best"])))
            if mvs:
                blocks.append((g, tex_t, md_t, mvs, nets, cells, ns))

    if not blocks:
        raise SystemExit(f"no mg<M>v<V> runs in either sweep in {a.inp}")

    def entry(c, m, tex):
        _, dec, k = HEAD[m]
        mu, se = f"{k * c['mean']:.{dec}f}", f"{k * c['se']:.{dec}f}"
        star = c["p_holm"] is not None and c["p_holm"] < a.alpha
        if tex:
            mu = f"\\mathbf{{{mu}}}" if c["best"] else mu
            return f"${mu} \\pm {se}{'^{*}' if star else ''}$"
        mu = f"**{mu}**" if c["best"] else mu
        return f"{mu} ± {se}{' *' if star else ''}"

    def nparams(g, net):
        return fmt_params(params.get((g[0], g[1], net)))

    arrow = {m: ("↑" if HIGHER_IS_BETTER[m] else "↓") for m in metrics}
    tarrow = {m: ("$\\uparrow$" if HIGHER_IS_BETTER[m] else "$\\downarrow$") for m in metrics}
    note = (f"Mean $\\pm$ standard error over test volumes. Bold: best in its sweep. "
            f"$^{{*}}$: differs from $M{{=}}{ref_mv[0]}$, $V{{=}}{ref_mv[1]}$ (paired "
            f"Wilcoxon signed-rank over volumes, Holm-adjusted, $p<{a.alpha:g}$).")
    ncol = 3 + len(metrics)

    # ---- LaTeX ---------------------------------------------------------------
    L = ["% Generated by figures/table_ablation.py. Do not edit by hand: rerun it.",
         "% Needs \\usepackage{booktabs}.",
         f"% Caption text: {note}",
         "\\begin{tabular}{rrr" + "c" * len(metrics) + "}", "\\toprule",
         "$M$ & $V$ & Params & " + " & ".join(
             f"{HEAD[m][0].replace('%', chr(92) + '%')} {tarrow[m]}" for m in metrics)
         + " \\\\"]
    for g, tex_t, _, mvs, nets, cells, ns in blocks:
        nlab = "/".join(str(n) for n in sorted(ns))
        L += ["\\midrule",
              f"\\multicolumn{{{ncol}}}{{l}}{{\\textit{{{block_title(g, True)}, {tex_t} "
              f"($n{{=}}{nlab}$ volumes)}}}} \\\\"]
        for mv in mvs:
            L.append(f"{mv[0]} & {mv[1]} & {nparams(g, nets[mv])} & " + " & ".join(
                entry(cells[(mv, m)], m, True) if (mv, m) in cells else "--"
                for m in metrics) + " \\\\")
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
    for g, _, md_t, mvs, nets, cells, ns in blocks:
        nlab = "/".join(str(n) for n in sorted(ns))
        M += [f"**{block_title(g, False)}, {md_t}** (n={nlab} volumes)", "",
              "| M | V | Params | " + " | ".join(f"{HEAD[m][0]} {arrow[m]}" for m in metrics)
              + " |", "|---|---|---|" + "---|" * len(metrics)]
        M += [f"| {mv[0]} | {mv[1]} | {nparams(g, nets[mv])} | " + " | ".join(
            entry(cells[(mv, m)], m, False) if (mv, m) in cells else "–" for m in metrics)
            + " |" for mv in mvs]
        M.append("")
    M.append(note.replace("$\\pm$", "±").replace("$^{*}$", "*").replace("$p<", "p < ")
             .replace("{=}", "=").replace("$", ""))
    with open(f"{a.out}.md", "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(M) + "\n")

    if rows_csv:
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
