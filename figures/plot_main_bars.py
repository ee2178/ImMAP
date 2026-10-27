#!/usr/bin/env python3
"""
Bar chart of the main comparison, with significance brackets.

    python figures/plot_main_bars.py                          # results/main/volumes.csv
    python figures/plot_main_bars.py --ref MGLPDS-Small       # brackets vs another net
    python figures/plot_main_bars.py --pairs all              # every pair (busy)
    python figures/plot_main_bars.py --metrics psnr ssim --zero
    python figures/plot_main_bars.py --nets E2E-VarNet LPDSNet MGLPDS-Small         --rename MGLPDS-Small=MGLPDS --ref MGLPDS      # three bars, one MGLPDS

Reads what `scripts/eval_per_volume.py --combine` writes (one row per run and
volume). One panel per metric; within a panel one group of bars per experiment
(anatomy x R), one bar per net, in a fixed order and colour.

    bar         mean over the volumes ALL FOUR nets were scored on, so every bar
                in a group and every test on it use the same patients
    whisker     standard error of that mean, std / sqrt(n) over volumes. It
                describes ONE net's spread; it is NOT the significance test --
                two overlapping whiskers can still differ significantly,
                because the test is paired
    bracket     two-sided Wilcoxon signed-rank, paired by volume, Holm-adjusted
                over the brackets drawn in that panel's group -- the same test
                and correction as scripts/wilcoxon_main.py, so the stars agree
                with its table. * p<0.05, ** p<0.01, *** p<0.001, n.s. otherwise

By default the brackets compare --ref (MGLPDS-Big) with each other net: the
paper's claim is about the multigrid net, and all six pairs in a group of four
bars is hard to read. `--pairs all` draws every pair anyway.

The value axis does NOT start at zero unless --zero: PSNR differences of a few
tenths of a dB are invisible on a 0-40 dB axis. The figure says so.

Writes <out>.png (300 dpi) and <out>.pdf, plus <out>_tests.csv with every
bracket's numbers. --tikz also writes <out>.tikz (a pgfplots tikzpicture, for
\input in the paper) and <out>_standalone.tex (compiles it on its own) -- drawn
from the SAME numbers and bracket layout as the PNG, so the two cannot drift.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import os
import sys

import numpy as np
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# The test, correction and conventions live in scripts/wilcoxon_main.py; load
# them from there rather than restating them (scripts/ is not a package).
_spec = importlib.util.spec_from_file_location(
    "wilcoxon_main", os.path.join(ROOT, "scripts", "wilcoxon_main.py"))
_wm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_wm)
holm, HIGHER_IS_BETTER, ORDER = _wm.holm, _wm.HIGHER_IS_BETTER, _wm.ORDER
one_run_per_net = _wm.one_run_per_net

# Fixed per net -- colour follows the method, never its position. The first
# four categorical slots of the reference palette (dataviz skill,
# references/palette.md), validated for adjacent marks in light mode.
COLORS = {"E2E-VarNet": "#2a78d6", "LPDSNet": "#eb6834",
          "MGLPDS-Small": "#1baf7a", "MGLPDS-Big": "#eda100"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#ffffff"
PALETTE = list(COLORS.values())


def color_of(n, nets):
    """A net's fixed colour, or -- for a net this palette does not name (the NYUMets runs) --
    the palette slot of its position among those unnamed nets, so they are still told apart."""
    if n in COLORS:
        return COLORS[n]
    unnamed = [x for x in nets if x not in COLORS]
    return PALETTE[unnamed.index(n) % len(PALETTE)] if n in unnamed else MUTED


def group_label(g, sep="\n"):
    """An experiment's label: "<anatomy> R=<R>", or the first field alone when R <= 0 (an
    experiment with no acceleration, e.g. NYUMets' slice-range groups)."""
    # a long R-less name ("slices 80-89, trained") breaks at its comma, so neighbours don't collide
    return str(g[0]).replace(", ", sep) if int(g[1]) <= 0 else f"{g[0]}{sep}R={g[1]}"
YLABEL = {"psnr": "PSNR (dB)", "ssim": "SSIM", "nrmse": "NRMSE", "lpips": "LPIPS"}


def load(path, nets=None, rename=None):
    """(anatomy, R) -> net -> {volume: {metric: value}}, and the metric names.
    `nets` keeps only those nets (original names); `rename` relabels them."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path} is empty")
    rows = one_run_per_net(rows)
    if nets:
        rows = [r for r in rows if r["net"] in nets]
        if not rows:
            raise SystemExit(f"none of {nets} in {path}")
    rename = rename or {}
    metrics = [m for m in HIGHER_IS_BETTER if m in rows[0]]
    data = {}
    for r in rows:
        net = rename.get(r["net"], r["net"])
        d = data.setdefault((r["anatomy"], int(r["R"])), {}).setdefault(net, {})
        d[r["volume"]] = {m: float(r[m]) for m in metrics if r.get(m) not in (None, "")}
    return data, metrics


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."


def sem(x):
    """Standard error of the mean: sample std / sqrt(n)."""
    n = len(x)
    return 0.0 if n < 2 else float(x.std(ddof=1) / np.sqrt(n))


def _f(v):
    return f"{v:.6g}"


def write_tikz(path, panels, nets, colors, groups, note, ncol, by_net=False):
    """A pgfplots groupplot of `panels` (what the matplotlib loop drew). `by_net`: one group,
    each bar labelled on the x axis with its net, and no legend."""
    name = os.path.splitext(os.path.basename(path))[0]
    legend = "".join(ch for ch in name if ch.isalnum()) + "legend"
    cname = {n: f"mbc{i}" for i, n in enumerate(nets)}
    nrow = -(-len(panels) // ncol)
    # Axis boxes sized so ncol of them, their labels and the gaps fill the width.
    axw = (0.86 - 0.13 * (ncol - 1)) / ncol
    axh = 0.24 if ncol > 2 else 0.22
    if by_net:
        xs = [x for n in nets for (b, x, _, _) in panels[0]["bars"] if b == n]
        ticks = ",".join(_f(x) for x in xs)
        ticklabels = ", ".join("{%s}" % n for n in nets)
        xlo, xhi = xs[0] - 0.6, xs[-1] + 0.6
    else:
        ticks = ",".join(str(i) for i in range(len(groups)))
        ticklabels = ", ".join(("{%s}" % group_label(g, "\\\\")) if int(g[1]) <= 0 else
                               "{%s\\\\$R{=}%d$}" % (g[0], g[1]) for g in groups)
        xlo, xhi = -0.5, len(groups) - 0.5
    L = [f"% Generated by figures/plot_main_bars.py --tikz. Do not edit by hand: rerun it.",
         f"% Needs \\usepackage{{pgfplots}} \\usepgfplotslibrary{{groupplots}}",
         f"% \\pgfplotsset{{compat=1.16}} (or newer). Width: \\mainbarswidth, default",
         f"% \\linewidth -- e.g. \\newcommand{{\\mainbarswidth}}{{\\textwidth}} in a figure*.",
         f"% It must name a LENGTH (\\textwidth, \\linewidth, a \\newlength): defined as",
         f"% literal text like 17cm, 0.2\\mainbarswidth would read as 0.217cm.",
         f"% Caption text: {note}.",
         "\\providecommand{\\mainbarswidth}{\\linewidth}"]
    L += [f"\\definecolor{{{cname[n]}}}{{HTML}}{{{colors[n].lstrip('#').upper()}}}"
          for n in nets]
    L += ["\\definecolor{mbink}{HTML}{0B0B0B}", "\\definecolor{mbmuted}{HTML}{52514E}",
          "\\definecolor{mbgrid}{HTML}{E1E0D9}",
          "\\begin{tikzpicture}",
          "\\begin{groupplot}[",
          f"  group style={{group size={ncol} by {nrow}, horizontal sep=0.13\\mainbarswidth,"
          f" vertical sep=0.11\\mainbarswidth}},",
          f"  scale only axis, width={axw:.3f}\\mainbarswidth, height={axh:.3f}\\mainbarswidth,",
          f"  xmin={_f(xlo)}, xmax={_f(xhi)}, xtick={{{ticks}}},",
          f"  xticklabels={{{ticklabels}}},",
          "  xticklabel style={align=center, font=\\footnotesize},",
          "  yticklabel style={font=\\footnotesize, /pgf/number format/fixed},",
          "  ylabel style={font=\\small}, axis x line*=bottom, axis y line*=left,",
          "  axis line style={mbmuted}, tick style={mbmuted}, major tick length=2pt,",
          "  ymajorgrids, grid style={mbgrid, line width=0.3pt},",
          "  scaled y ticks=false,",
          "]"]
    for k, pn in enumerate(panels):
        opts = [f"ymin={_f(pn['base'])}", f"ymax={_f(pn['top'])}",
                f"ylabel={{{pn['ylabel']}}}"]
        if k == 0 and not by_net:
            opts += [f"legend to name={legend}", "legend columns=-1",
                     "legend style={draw=none, font=\\small, "
                     "/tikz/every even column/.append style={column sep=10pt}}",
                     "area legend"]
        L.append(f"\\nextgroupplot[{', '.join(opts)}]")
        for n in nets:
            pts = [b for b in pn["bars"] if b[0] == n]
            if not pts:
                continue
            coords = " ".join(f"({_f(x)},{_f(mu)}) +- (0,{_f(ci)})" for _, x, mu, ci in pts)
            L.append(f"\\addplot[ybar, bar shift=0pt, bar width={_f(pn['width'])}, "
                     f"fill={cname[n]}, draw=white, line width=0.4pt, "
                     f"error bars/.cd, y dir=both, y explicit, "
                     f"error bar style={{mbink, line width=0.5pt}}, "
                     f"error mark options={{rotate=90, mbink, mark size=1.2pt, "
                     f"line width=0.5pt}}] coordinates {{{coords}}};")
            if k == 0 and not by_net:
                L.append(f"\\addlegendentry{{{n}}}")
        # The bars run down past ymin and are clipped there AFTER the axis line
        # is drawn, so they cover it everywhere but the gaps. Redraw it on top.
        L.append("\\draw[mbmuted, line width=0.4pt] (rel axis cs:0,0) -- "
                 "(rel axis cs:1,0);")
        for x1, x2, y, yb, lab, sig in pn["brackets"]:
            L.append(f"\\draw[mbink, line width=0.45pt] (axis cs:{_f(x1)},{_f(yb)}) -- "
                     f"(axis cs:{_f(x1)},{_f(y)}) -- (axis cs:{_f(x2)},{_f(y)}) -- "
                     f"(axis cs:{_f(x2)},{_f(yb)});")
            txt = lab if sig else "n.s."
            style = ("font=\\scriptsize, text=mbink" if sig
                     else "font=\\tiny, text=mbmuted")
            L.append(f"\\node[{style}, anchor=south, inner sep=0.5pt] at "
                     f"(axis cs:{_f((x1 + x2) / 2)},{_f(y)}) {{{txt}}};")
    L.append("\\end{groupplot}")
    # Legend centred over the top row: over the middle panel for an odd count,
    # between the two middle ones for an even count.
    ntop = min(ncol, len(panels))
    if by_net:
        pass
    elif ntop % 2:
        L.append(f"\\node[above=8pt] at (group c{(ntop + 1) // 2}r1.north) "
                 f"{{\\pgfplotslegendfromname{{{legend}}}}};")
    else:
        L.append(f"\\path (group c{ntop // 2}r1.north) -- (group c{ntop // 2 + 1}r1.north) "
                 f"node[midway, above=8pt] {{\\pgfplotslegendfromname{{{legend}}}}};")
    L.append("\\end{tikzpicture}")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")

    wrapper = os.path.splitext(path)[0] + "_standalone.tex"
    with open(wrapper, "w", encoding="utf-8", newline="\n") as f:
        f.write("\\documentclass[border=4pt]{standalone}\n"
                "\\usepackage{pgfplots}\n\\usepgfplotslibrary{groupplots}\n"
                "\\pgfplotsset{compat=1.16}\n"
                # A LENGTH, not \newcommand{...}{17cm}: "0.2\mainbarswidth" would
                # then read as "0.217cm", collapsing every panel.
                "\\newlength{\\mainbarswidth}\\setlength{\\mainbarswidth}{17cm}\n"
                "\\begin{document}\n"
                f"\\input{{{os.path.basename(path)}}}\n"
                "\\end{document}\n")
    return wrapper


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", default=os.path.join(ROOT, "results", "main",
                                                            "volumes.csv"))
    p.add_argument("--metrics", nargs="*", help="default: every metric in the file")
    p.add_argument("--nets", nargs="*",
                   help="only these nets, in the usual order (default: all in the file)")
    p.add_argument("--rename", nargs="*", default=[], metavar="OLD=NEW",
                   help="relabel a net (its colour stays the original's), e.g. "
                        "MGLPDS-Small=MGLPDS")
    p.add_argument("--ref", default="MGLPDS-Big",
                   help="brackets compare this net with each other one")
    p.add_argument("--pairs", choices=["ref", "all"], default="ref")
    p.add_argument("--zero", action="store_true", help="start the value axis at 0")
    p.add_argument("--out", default=os.path.join(HERE, "out", "main_bars"),
                   help="output path WITHOUT extension")
    p.add_argument("--tikz", action="store_true",
                   help="also write <out>.tikz (pgfplots) and <out>_standalone.tex")
    p.add_argument("--by-net", action="store_true",
                   help="one experiment only: label each bar with its net on the x axis instead "
                        "of a legend, and leave the experiment's name to the caption")
    a = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not os.path.exists(a.inp):
        raise SystemExit(f"{a.inp} not found -- copy results/main/volumes.csv from the "
                         f"cluster (written by scripts/eval_per_volume.py --combine)")
    rename = dict(kv.split("=", 1) for kv in a.rename)
    back = {new: old for old, new in rename.items()}
    data, metrics = load(a.inp, a.nets, rename)
    if a.metrics:
        metrics = [m for m in metrics if m in a.metrics]
    groups = sorted(data, key=lambda g: (g[0], g[1]))
    nets_all = sorted({n for g in groups for n in data[g]},
                      key=lambda n: (ORDER.index(back.get(n, n))
                                     if back.get(n, n) in ORDER else 99, n))

    # ---- statistics ----------------------------------------------------------
    stats, tests = {}, []
    for g in groups:
        nets = [n for n in nets_all if n in data[g]]
        missing = [rename.get(n, n) for n in (a.nets or ORDER) if rename.get(n, n) not in nets]
        if missing:
            print(f"[{g[0]} R{g[1]}] no results for {', '.join(missing)}", file=sys.stderr)
        for m in metrics:
            vols = sorted(set.intersection(*[
                {v for v, d in data[g][n].items() if m in d} for n in nets]))
            if len(vols) < 6:
                print(f"[{g[0]} R{g[1]} {m}] only {len(vols)} volumes common to all nets "
                      f"-- skipped", file=sys.stderr)
                continue
            x = {n: np.array([data[g][n][v][m] for v in vols]) for n in nets}
            stats[(g, m)] = dict(n=len(vols), x=x,
                                 mean={n: float(x[n].mean()) for n in nets},
                                 ci={n: sem(x[n]) for n in nets})
            if a.pairs == "all":
                pairs = list(itertools.combinations(nets, 2))
            else:
                pairs = [(a.ref, n) for n in nets if n != a.ref] if a.ref in nets else []
            fam = []
            for u, w in pairs:
                d = x[u] - x[w]
                pv = (1.0 if not np.any(d != 0) else
                      float(wilcoxon(x[u], x[w], zero_method="wilcox",
                                     alternative="two-sided").pvalue))
                fam.append(dict(anatomy=g[0], R=g[1], metric=m, A=u, B=w, n=len(vols),
                                mean_A=float(x[u].mean()), mean_B=float(x[w].mean()),
                                p=pv))
            for row, ph in zip(fam, holm([r["p"] for r in fam]) if fam else []):
                row["p_holm"] = float(ph)
                row["stars"] = stars(ph)
            stats[(g, m)]["tests"] = fam
            tests += fam
    if not stats:
        raise SystemExit("nothing to plot")

    # ---- figure ---------------------------------------------------------------
    # Up to three panels in one row; four or more as a two-column grid.
    ncol = len(metrics) if len(metrics) <= 3 else 2
    nrow = int(np.ceil(len(metrics) / ncol))
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 2.9 * nrow),
                             squeeze=False, facecolor=SURFACE)
    nb = len(nets_all)
    if a.by_net and len(groups) != 1:
        raise SystemExit(f"--by-net needs exactly one experiment, the file has {len(groups)}")
    # by net: bars one unit apart, so each has room for its name underneath
    width = 0.6 if a.by_net else 0.8 / nb
    panels = []                       # what was drawn, for --tikz

    for k, m in enumerate(metrics):
        pbars, pbrackets = [], []
        ax = axes[k // ncol][k % ncol]
        ax.set_facecolor(SURFACE)
        tops, bottoms = [], []
        labelled = set()             # legend entry on a net's FIRST bar, wherever
        for gi, g in enumerate(groups):   # that is -- not only in group 0
            s = stats.get((g, m))
            if s is None:
                continue
            xs = {}
            for bi, n in enumerate(nets_all):
                if n not in s["mean"]:
                    continue
                xc = float(bi) if a.by_net else gi + (bi - (nb - 1) / 2) * width
                xs[n] = xc
                mu, ci = s["mean"][n], s["ci"][n]
                ax.bar(xc, mu, width * 0.92,
                       color=color_of(back.get(n, n), [back.get(x, x) for x in nets_all]),
                       edgecolor=SURFACE, linewidth=0.8, zorder=2,
                       label=None if n in labelled else n)
                labelled.add(n)
                ax.errorbar(xc, mu, yerr=ci, color=INK, lw=0.8, capsize=1.6, zorder=3)
                pbars.append((n, xc, mu, ci))
                tops.append(mu + ci)
                bottoms.append(mu - ci)
            s["xs"] = xs
        lo, hi = min(bottoms), max(tops)
        span = hi - lo if hi > lo else abs(hi) or 1.0
        base = 0.0 if a.zero else lo - 0.35 * span
        if lo >= 0:                  # padding must not push a non-negative
            base = max(base, 0.0)    # metric's axis below zero
        step = 0.15 * (hi - base)

        # Brackets, stacked from the shortest span up, above the group's tallest
        # whisker.
        top_used = hi
        for gi, g in enumerate(groups):
            s = stats.get((g, m))
            if s is None or not s.get("tests"):
                continue
            ytop = max(s["mean"][n] + s["ci"][n] for n in s["mean"])
            fam = sorted(s["tests"], key=lambda r: abs(s["xs"][r["A"]] - s["xs"][r["B"]]))
            for lvl, r in enumerate(fam):
                x1, x2 = sorted((s["xs"][r["A"]], s["xs"][r["B"]]))
                y = ytop + (lvl + 0.6) * step
                ax.plot([x1, x1, x2, x2], [y - 0.25 * step, y, y, y - 0.25 * step],
                        color=INK, lw=0.7, zorder=4)
                sig = r["stars"] != "n.s."
                ax.text((x1 + x2) / 2, y + (0.02 if sig else 0.08) * step, r["stars"],
                        ha="center", va="bottom", color=INK if sig else MUTED,
                        fontsize=8 if sig else 6.5, zorder=5)
                top_used = max(top_used, y + 0.9 * step)
                pbrackets.append((x1, x2, y, y - 0.25 * step, r["stars"], sig))

        ax.set_ylim(base, top_used)
        if a.by_net:
            ax.set_xticks([x for _, x, _, _ in pbars])
            ax.set_xticklabels([n for n, _, _, _ in pbars])
            ax.set_xlim(-0.6, nb - 0.4)
        else:
            ax.set_xticks(range(len(groups)))
            ax.set_xticklabels([group_label(g) for g in groups])
        arrow = "↑" if HIGHER_IS_BETTER[m] else "↓"
        ax.set_ylabel(f"{YLABEL.get(m, m)} {arrow}")
        panels.append(dict(
            base=base, top=top_used, width=width * 0.92, bars=pbars, brackets=pbrackets,
            ylabel=f"{YLABEL.get(m, m)} $\\{'uparrow' if HIGHER_IS_BETTER[m] else 'downarrow'}$"))
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(length=2)
    for k in range(len(metrics), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")

    if not a.by_net:
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False,
                   bbox_to_anchor=(0.5, 1.0), handlelength=1.0, columnspacing=1.2)
    ns = sorted({s["n"] for s in stats.values()})
    what = (f"vs {a.ref}" if a.pairs == "ref" else "all pairs")
    note = (f"mean ± s.e.m. over {ns[0] if len(ns) == 1 else f'{ns[0]}-{ns[-1]}'} "
            f"volumes; brackets: paired Wilcoxon {what}, Holm-adjusted"
            + ("" if a.zero else "; value axes do not start at 0"))
    fig.text(0.5, 0.005, note, ha="center", va="bottom", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1.0 if a.by_net else 0.94))

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}.{ext}", dpi=300, facecolor=SURFACE)
    with open(f"{a.out}_tests.csv", "w", newline="") as f:
        if tests:
            w = csv.DictWriter(f, fieldnames=list(tests[0]))
            w.writeheader()
            w.writerows({k: (f"{v:.6g}" if isinstance(v, float) else v)
                         for k, v in r.items()} for r in tests)

    for r in tests:
        print(f"{r['anatomy']:<6} R{r['R']:<3} {r['metric']:<6} {r['A']:<13} vs "
              f"{r['B']:<13} n={r['n']:<4} {r['mean_A']:.4f} vs {r['mean_B']:.4f}  "
              f"p_holm={r['p_holm']:.2g} {r['stars']}")
    print(f"\nwrote {a.out}.png, .pdf, _tests.csv")
    if a.tikz:
        colors = {n: color_of(back.get(n, n), [back.get(x, x) for x in nets_all])
                  for n in nets_all}
        wrapper = write_tikz(f"{a.out}.tikz", panels, nets_all, colors, groups,
                             note.replace("\u00b1", "+/-"), ncol, by_net=a.by_net)
        print(f"wrote {a.out}.tikz and {wrapper}")


if __name__ == "__main__":
    main()
