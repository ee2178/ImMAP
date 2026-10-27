#!/usr/bin/env python3
"""
NYUMets dumps -> the per-volume CSV the main-comparison table, bar chart and Wilcoxon script read.

    python scripts/nyumets_volumes_csv.py                  # the two default runs
    python figures/table_main.py     --in results/nyumets_eval/volumes.csv \
        --nets UNet "UNet + prior w2" --ref "UNet + prior w2" --out figures/out/nyumets_table
    python figures/plot_main_bars.py --in results/nyumets_eval/volumes.csv \
        --ref "UNet + prior w2" --out figures/out/nyumets_bars
    python scripts/wilcoxon_main.py  --in results/nyumets_eval/volumes.csv --ref "UNet + prior w2"

(this script prints those commands with your run labels filled in).

SPLIT ACROSS MACHINES -- CSVs on the cluster, figures on the laptop:

    cluster:  python scripts/nyumets_volumes_csv.py --profile ''      # h5py + numpy only
    laptop:   scp <cluster>:<ImMAP>/results/nyumets_eval/{volumes,slices}.csv results/nyumets_eval/
              python scripts/nyumets_volumes_csv.py --from-csv results/nyumets_eval/slices.csv
              ... then the three commands above, unchanged.

Reads the per-slice metrics scripts/dump_eval_nyumets.py stored in each dump (numpy + h5py only,
so it runs on a login node or against copied dumps) and writes, in the format of
`scripts/eval_per_volume.py --combine`:

    slices.csv    anatomy, R, net, run, volume, slice, psnr, ssim, nrmse
    volumes.csv   anatomy, R, net, run, volume, n_slices, psnr, ssim, nrmse   (slice means)

The volume is the unit of the statistics, as for fastMRI: slices of one patient are not
independent samples (scripts/wilcoxon_main.py explains).

GROUPS. The fastMRI scripts group experiments by (anatomy, R); NYUMets has no R, so the `anatomy`
column carries a slice-range group and R is 0, which the table and the chart print as the group
name alone. By default ONE group, the whole dumped range. `--trained-hi Z` (the end of the
training slice range) also splits it into the part below Z and the part from Z on, and shades the
latter in the profile -- for when slices the nets never saw should be read separately. Slices
the nets never saw are a different question. `--groups 80:90 90:100` sets them by hand.

PROFILE. Also draws <--profile>.png/.pdf: every net's mean PSNR per ORIGINAL slice index (mean
+- s.e.m. over volumes) and, below it, the paired gain of each net over the first one along z,
with the unseen slices shaded. The bar chart pools slices; this shows WHERE along the head a
gain lives.
"""

from __future__ import annotations

import argparse
import csv
import os

import h5py
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NYU = os.path.join(HERE, "trained_nets", "nyumets")
DEFAULT_RUNS = ["UNet=I2SB_Unet_NYUMets_CT1_from_all_bs64",
                "UNet + prior w2=I2SB_Unet_NYUMets_CT1_from_all_plus_otherT1CT1_w2_bs64"]
METRICS = ("psnr", "ssim", "nrmse")
# the reference palette plot_main_bars.py uses for nets it does not name (first slots, in order)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"


def dump_dir(root, run):
    for d in (os.path.join(root, run, "eval_dump"), os.path.join(root, run)):
        if os.path.isdir(d) and any(f.endswith(".h5") for f in os.listdir(d)):
            return d
    raise SystemExit(f"no dump for {run} under {root} -- run scripts/dump_eval_nyumets.py first")


def prior_similarity(ref, pri, mask):
    """Per slice, how far the (registered) prior CT1 is from the current CT1: the mean squared
    difference inside the evaluation mask, in the dump's intensity units (per-volume scaling, so
    tissue ~1). 0 = the prior IS the current CT1; larger = a worse guide. (n,), NaN for slices
    with almost no mask. Unlike a correlation it also counts a global brightness mismatch."""
    n = ref.shape[0]
    mse = np.full(n, np.nan)
    for i in range(n):
        m = mask[i] > 0
        if m.sum() < 50:
            continue
        d = ref[i][m].astype(np.float64) - pri[i][m].astype(np.float64)
        mse[i] = float((d * d).mean())
    return mse


def read_run(d, similarity=False):
    """volume -> {"slice": (n,), metric: (n,)} from a run's dump, and the shared attrs. With
    `similarity`, also "prior_mse" per slice (prior_similarity)."""
    out, attrs = {}, {}
    for f in sorted(os.listdir(d)):
        if not f.endswith(".h5"):
            continue
        with h5py.File(os.path.join(d, f), "r") as h:
            out[f[:-3]] = dict(slice=h["slice_index"][:].astype(int),
                               **{m: h[f"{m}_slice"][:].astype(float) for m in METRICS})
            if similarity and "prior_ct1" in h and "organ_mask" in h:
                out[f[:-3]]["prior_mse"] = prior_similarity(
                    h["reference"][:], h["prior_ct1"][:], h["organ_mask"][:])
            if not attrs:
                attrs = {k: h.attrs[k] for k in ("seed", "intensity_units", "nfe", "data_range")
                         if k in h.attrs}
    return out, attrs


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", default=DEFAULT_RUNS, metavar="LABEL=RUN_DIR_NAME",
                   help="the nets, in table order; the label is what the table and chart print")
    p.add_argument("--root", default=NYU, help="where the run directories live")
    p.add_argument("--groups", nargs="*", metavar="LO:HI",
                   help="slice-range groups (ORIGINAL indices, hi exclusive); default: the "
                        "dumped range, plus its trained and unseen parts")
    p.add_argument("--trained-hi", type=int, default=None,
                   help="end of the slice range the nets trained on: splits the groups there and "
                        "shades the profile from it (default: no split)")
    p.add_argument("--out", default=os.path.join(HERE, "results", "nyumets_eval"),
                   help="directory for slices.csv and volumes.csv")
    p.add_argument("--profile", default=os.path.join(HERE, "figures", "out", "nyumets_profile"),
                   help="z-profile figure, path without extension ('' = none)")
    p.add_argument("--tikz", action="store_true",
                   help="also write the profile and the similarity plot as pgfplots (.tikz + _standalone.tex)")
    p.add_argument("--similarity", default=os.path.join(HERE, "figures", "out",
                                                         "nyumets_similarity"),
                   help="gain vs prior-similarity figure, path without extension ('' = none)")
    p.add_argument("--data-range", type=float, default=2.0,
                   help="the metrics' data_range, for the prior's PSNR (10 log10(range^2 / prior_mse))"
                        " in --from-csv mode; read from the dumps otherwise (abstract runs: 2.0)")
    p.add_argument("--from-csv", metavar="SLICES_CSV",
                   help="draw the z-profile from a slices.csv this script wrote (e.g. copied off "
                        "the cluster) instead of reading dumps; writes no CSV")
    a = p.parse_args(argv)

    if a.from_csv:
        data, labels = {}, []
        with open(a.from_csv, newline="") as f:
            rows = list(csv.DictReader(f))
        if a.similarity:
            if any(r.get("prior_mse") for r in rows):
                nets = list(dict.fromkeys(r["net"] for r in rows))
                similarity_plot(*gain_vs_similarity(rows, nets, a.data_range), nets, a.similarity,
                                tikz=a.tikz)
            else:
                print("this slices.csv has no prior_mse column -- rebuild it on the cluster "
                      "(scripts/nyumets_volumes_csv.py --profile '') for the similarity plot")
        for r in rows:
            if r["net"] not in data:
                data[r["net"]] = {}
                labels.append(r["net"])
            d = data[r["net"]].setdefault(r["volume"], {"slice": [], "psnr": []})
            d["slice"].append(int(r["slice"]))
            d["psnr"].append(float(r["psnr"]))
        for lab in labels:
            for d in data[lab].values():
                d["slice"], d["psnr"] = np.array(d["slice"]), np.array(d["psnr"])
        vols = sorted(set.intersection(*[set(data[lab]) for lab in labels]))
        if a.profile:
            profile(data, labels, vols, a.trained_hi, a.profile, a.tikz)
        return

    runs = [r.split("=", 1) if "=" in r else (r, r) for r in a.runs]
    data, attrs = {}, {}
    for i, (label, run) in enumerate(runs):
        # the prior and the target are the same in every run's dump: measure similarity once
        data[label], attrs[label] = read_run(dump_dir(a.root, run), similarity=(i == 0))
    labels = [lab for lab, _ in runs]
    for lab in labels[1:]:
        bad = {k for k in attrs[lab] if attrs[lab][k] != attrs[labels[0]].get(k)}
        if bad:
            raise SystemExit(f"{lab} was dumped differently from {labels[0]} ({sorted(bad)}); "
                             f"re-dump both in one sweep")
    vols = sorted(set.intersection(*[set(data[lab]) for lab in labels]))
    if not vols:
        raise SystemExit("the runs have no dumped volume in common")
    lo = min(int(data[labels[0]][v]["slice"].min()) for v in vols)
    hi = max(int(data[labels[0]][v]["slice"].max()) for v in vols) + 1

    if a.groups:
        groups = [tuple(int(x) for x in g.split(":")) for g in a.groups]
    else:
        groups = [(lo, hi)]
        if a.trained_hi is not None and lo < a.trained_hi < hi:
            groups += [(lo, a.trained_hi), (a.trained_hi, hi)]

    def gname(g):
        # ", <tag>" is where the bar chart breaks the tick label (plot_main_bars.group_label)
        th = a.trained_hi
        tag = ("" if th is None
               else ", all" if (g[0], g[1]) == (lo, hi) and len(groups) > 1
               else ", unseen" if g[0] >= th
               else ", trained" if g[1] <= th else "")
        return f"slices {g[0]}-{g[1] - 1}{tag}"

    os.makedirs(a.out, exist_ok=True)
    srows, vrows = [], []
    for g in groups:
        for (lab, run) in runs:
            for v in vols:
                d = data[lab][v]
                sel = (d["slice"] >= g[0]) & (d["slice"] < g[1])
                if not sel.any():
                    continue
                base = dict(anatomy=gname(g), R=0, net=lab, run=run, volume=v)
                if g == groups[0]:
                    s0 = data[labels[0]][v]                  # similarity lives in the first run
                    for i in np.where(sel)[0]:
                        sim = {k: (f"{s0[k][i]:.6g}" if k in s0 and np.isfinite(s0[k][i]) else "")
                               for k in ("prior_mse",)}
                        srows.append(dict(base, slice=int(d["slice"][i]),
                                          **{m: f"{d[m][i]:.6g}" for m in METRICS}, **sim))
                vrows.append(dict(base, n_slices=int(sel.sum()),
                                  **{m: f"{np.nanmean(d[m][sel]):.6g}" for m in METRICS}))
    for name, rows in (("slices.csv", srows), ("volumes.csv", vrows)):
        with open(os.path.join(a.out, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"{len(vols)} volumes x {len(labels)} nets; groups: "
          + ", ".join(gname(g) for g in groups))
    print(f"wrote {os.path.join(a.out, 'volumes.csv')} and slices.csv")

    if a.profile:
        profile(data, labels, vols, a.trained_hi, a.profile, a.tikz)
    if a.similarity:
        rng = float(attrs[labels[0]].get("data_range", a.data_range))
        similarity_plot(*gain_vs_similarity(srows, labels, rng), labels, a.similarity, tikz=a.tikz)

    q = lambda s: f'"{s}"' if " " in s else s           # noqa: E731
    vin = os.path.relpath(os.path.join(a.out, "volumes.csv"), HERE)
    ref, nets = q(labels[-1]), " ".join(q(lab) for lab in labels)
    print("\nnext (from the repo root):\n"
          f"  python figures/table_main.py --in {vin} --nets {nets} --rename --ref {ref} "
          f"--out figures/out/nyumets_table\n"
          f"  python figures/plot_main_bars.py --in {vin} --nets {nets} --ref {ref} "
          f"--out figures/out/nyumets_bars\n"
          f"  python scripts/wilcoxon_main.py --in {vin} --ref {ref} "
          f"--out {os.path.join(os.path.dirname(vin), 'wilcoxon_volumes.csv')}")


def gain_vs_similarity(rows, labels, data_range):
    """slices.csv rows -> (gain, prior PSNR) arrays over the slices both nets have: gain =
    PSNR(last net) - PSNR(first net), the prior study's paired gain; prior PSNR = 10 log10(
    data_range^2 / prior_mse), the registered prior CT1 scored against the current CT1 exactly as
    the nets are (same mask, same data_range) -- log MSE up to a sign and a shift."""
    by = {}
    for r in rows:
        by.setdefault((r["volume"], int(r["slice"])), {})[r["net"]] = r
    g, e = [], []
    for k, x in by.items():
        if labels[0] in x and labels[-1] in x and x[labels[0]].get("prior_mse") not in (None, ""):
            g.append(float(x[labels[-1]]["psnr"]) - float(x[labels[0]]["psnr"]))
            e.append(float(x[labels[0]]["prior_mse"]))
    g, e = np.asarray(g, float), np.asarray(e, float)
    ok = np.isfinite(g) & np.isfinite(e) & (e > 0)
    return g[ok], 10 * np.log10(data_range ** 2 / e[ok])


def similarity_plot(gain, ppsnr, labels, out, n_bins=10, width=2.0, min_n=100, tikz=False):
    """Per-slice gain of the prior study against how close the prior CT1 is to the current CT1
    (the prior's own PSNR against it, dB; high = similar), as a line: mean gain +- 1 SD over the
    slices in fixed `width`-dB bins spanning the data (bins with < `min_n` slices dropped; the
    s.e.m. is ~0.01 dB, too small to draw). Also prints equal-count bins (deciles, for quoting in
    text)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def stats(s, lo, hi, mid):
        x = gain[s]
        return dict(lo=lo, hi=hi, mid=mid, n=int(s.sum()), mean=float(x.mean()),
                    sd=float(x.std(ddof=1)), se=float(x.std(ddof=1) / np.sqrt(x.size)),
                    better=float((x > 0).mean()))

    def table(rows, title):
        print(f"\n{title}")
        print(f"  {'prior PSNR (dB)':>19} {'n':>6} {'mean gain':>10} {'+-sd':>6} {'+-se':>6} {'better':>7}")
        for r in rows:
            print(f"  {r['lo']:>9.2f}-{r['hi']:<9.2f} {r['n']:>6} {r['mean']:>+10.2f} {r['sd']:>6.2f} "
                  f"{r['se']:>6.2f} {100 * r['better']:>6.0f}%")

    head = (f"prior-study PSNR gain ({labels[-1]} - {labels[0]}) by prior PSNR (prior CT1 vs "
            f"current CT1 in the mask; high = similar), {gain.size} slices")
    q = np.unique(np.quantile(ppsnr, np.linspace(0, 1, n_bins + 1)))
    qi = np.clip(np.searchsorted(q, ppsnr, side="right") - 1, 0, len(q) - 2)
    table([stats(qi == b, q[b], q[b + 1], 0.0) for b in range(len(q) - 1)],
          head + ", equal-count bins:")

    e = np.arange(np.floor(ppsnr.min() / width) * width, ppsnr.max() + width, width)
    bi = np.clip(np.searchsorted(e, ppsnr, side="right") - 1, 0, len(e) - 2)
    rows = [stats(bi == b, e[b], e[b + 1], 0.5 * (e[b] + e[b + 1]))
            for b in range(len(e) - 1) if (bi == b).sum() >= min_n]
    table(rows, head + f", {width:g} dB bins with >= {min_n} slices (plotted):")

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    mid = np.array([r["mid"] for r in rows])
    ax.errorbar(mid, [r["mean"] for r in rows], yerr=[r["sd"] for r in rows], color=PALETTE[1],
                lw=1.6, marker="o", ms=4, capsize=2.5, zorder=3)
    ax.axhline(0, color=MUTED, lw=0.8, zorder=2)
    ax.set_xlabel("prior PSNR: registered prior CT1 vs current CT1 (dB)")
    ax.set_ylabel("PSNR gain (dB)")
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.text(0.5, 0.005, f"{gain.size} test slices; mean ± 1 SD in {width:g} dB bins",
             ha="center", va="bottom", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=300)
    print(f"wrote {out}.png, .pdf")
    if tikz:
        write_similarity_tikz(out, gain, rows, width)


def write_similarity_tikz(out, gain, rows, width):
    r"""The similarity plot as pgfplots: <out>.tikz (for \input; width \nyusimwidth, a LENGTH,
    default \linewidth) and <out>_standalone.tex."""
    name = os.path.basename(out)
    f = lambda v: f"{v:.5g}"                                             # noqa: E731
    xlo, xhi = rows[0]["lo"], rows[-1]["hi"]
    # explicit y limits: pgfplots' automatic ones ignore the error bars and clip them
    lo = min(r["mean"] - r["sd"] for r in rows)
    hi = max(r["mean"] + r["sd"] for r in rows)
    ylo, yhi = np.floor((lo - 0.1) * 2) / 2, np.ceil((hi + 0.1) * 2) / 2
    pts = " ".join(f"({f(r['mid'])},{f(r['mean'])}) +- (0,{f(r['sd'])})" for r in rows)
    L = [r"% Generated by scripts/nyumets_volumes_csv.py --tikz. Do not edit by hand: rerun it.",
         r"% Needs \usepackage{pgfplots} \pgfplotsset{compat=1.16}. Width: \nyusimwidth, a LENGTH",
         r"% (default \linewidth).",
         rf"% Caption text: {gain.size} test slices; mean gain +/- 1 SD over the slices in "
         rf"{width:g} dB bins of the prior PSNR (bins with < 100 slices omitted).",
         r"\providecommand{\nyusimwidth}{\linewidth}",
         rf"\definecolor{{nysim}}{{HTML}}{{{PALETTE[1].lstrip('#').upper()}}}",
         r"\definecolor{nysimmuted}{HTML}{52514E}", r"\definecolor{nysimgrid}{HTML}{E1E0D9}",
         r"\begin{tikzpicture}",
         # scale only axis: the axis box itself is 0.85 x 0.5 of the width, labels sit outside it
         r"\begin{axis}[scale only axis, width=0.85\nyusimwidth, height=0.5\nyusimwidth,",
         rf"  xmin={f(xlo)}, xmax={f(xhi)}, xtick distance={f(2 * width)},",
         rf"  ymin={f(ylo)}, ymax={f(yhi)},",
         r"  axis lines=left, axis line style={nysimmuted}, tick style={nysimmuted},",
         r"  tick align=outside, ymajorgrids, grid style={nysimgrid},",
         r"  xlabel={prior PSNR, registered prior CT1 vs.\ current CT1 (dB)},",
         r"  ylabel={PSNR gain (dB)},",
         r"  label style={font=\small}, tick label style={font=\footnotesize}]",
         rf"\addplot[nysimmuted, line width=0.5pt, forget plot] coordinates {{({f(xlo)},0) ({f(xhi)},0)}};",
         r"\addplot[nysim, line width=1pt, mark=*, mark size=1.6pt, error bars/.cd, y dir=both,",
         rf"  y explicit, error bar style={{line width=0.6pt}}] coordinates {{{pts}}};",
         r"\end{axis}", r"\end{tikzpicture}"]
    with open(f"{out}.tikz", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L) + "\n")
    with open(f"{out}_standalone.tex", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join([
            r"\documentclass[border=4pt]{standalone}", r"\usepackage{pgfplots}",
            r"\pgfplotsset{compat=1.16}",
            r"\newlength{\nyusimwidth}\setlength{\nyusimwidth}{11cm}",
            r"\begin{document}", rf"\input{{{name}.tikz}}", r"\end{document}"]) + "\n")
    print(f"wrote {out}.tikz and {out}_standalone.tex")


def profile(data, labels, vols, trained_hi, out, tikz=False):
    """Mean PSNR per original slice (+- s.e.m. over volumes), and each net's paired gain over
    the first net along z."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    zs = sorted({int(z) for v in vols for z in data[labels[0]][v]["slice"]})
    at = {lab: {z: [] for z in zs} for lab in labels}
    gain = {lab: {z: [] for z in zs} for lab in labels[1:]}
    for v in vols:
        b = dict(zip(data[labels[0]][v]["slice"], data[labels[0]][v]["psnr"]))
        for lab in labels:
            for z, x in zip(data[lab][v]["slice"], data[lab][v]["psnr"]):
                if np.isfinite(x):
                    at[lab][int(z)].append(x)
                    if lab != labels[0] and np.isfinite(b.get(z, np.nan)):
                        gain[lab][int(z)].append(x - b[z])

    def ms(lst):
        x = np.asarray(lst, float)
        return (x.mean(), x.std(ddof=1) / np.sqrt(x.size)) if x.size > 1 else (np.nan, np.nan)

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(2, 1, figsize=(5.2, 4.4), sharex=True,
                           gridspec_kw=dict(height_ratios=[3, 2]))
    zz = np.array(zs)
    top, bot = {}, {}                       # what is drawn, for --tikz
    for i, lab in enumerate(labels):
        mu, se = np.array([ms(at[lab][z]) for z in zs]).T
        top[lab] = (mu, se)
        c = PALETTE[i % len(PALETTE)]
        ax[0].plot(zz, mu, color=c, lw=1.4, label=lab)
        ax[0].fill_between(zz, mu - se, mu + se, color=c, alpha=0.2, lw=0)
        if lab != labels[0]:
            g, gs = np.array([ms(gain[lab][z]) for z in zs]).T
            bot[f"{lab} - {labels[0]}"] = (g, gs)
            ax[1].plot(zz, g, color=c, lw=1.4, label=f"{lab} - {labels[0]}")
            ax[1].fill_between(zz, g - gs, g + gs, color=c, alpha=0.2, lw=0)
    ax[1].axhline(0, color=MUTED, lw=0.8)
    for x in ax:
        if trained_hi is not None and zs[0] < trained_hi <= zs[-1]:
            x.axvspan(trained_hi - 0.5, zs[-1] + 0.5, color=GRID, alpha=0.6, lw=0, zorder=0)
        x.grid(axis="y", color=GRID, lw=0.6)
        for sp in ("top", "right"):
            x.spines[sp].set_visible(False)
    ax[1].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax[0].set_ylabel("PSNR (dB) ↑")
    ax[1].set_ylabel("paired gain (dB)")
    ax[1].set_xlabel("original slice index" + (f"  (shaded: >= {trained_hi}, unseen in training)"
                                                 if trained_hi is not None
                                                 and zs[0] < trained_hi <= zs[-1] else ""))
    ax[0].legend(frameon=False, loc="lower left")
    ax[1].legend(frameon=False, loc="lower left", fontsize=8)
    fig.text(0.5, 0.005, f"mean ± s.e.m. over {len(vols)} volumes", ha="center", va="bottom",
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=300)
    print(f"wrote {out}.png, .pdf")
    if tikz:
        write_profile_tikz(out, zz, top, bot, len(vols),
                           trained_hi if trained_hi is not None and zs[0] < trained_hi <= zs[-1]
                           else None)


def write_profile_tikz(out, zz, top, bot, n_vols, shade_from=None):
    r"""The profile as a pgfplots groupplot: <out>.tikz (for \input; width \nyuprofilewidth, a
    LENGTH, default \linewidth) and <out>_standalone.tex -- the same numbers as the PNG."""
    name = os.path.basename(out)
    f = lambda v: f"{v:.5g}"                                             # noqa: E731
    lo, hi = zz[0] - 0.5, zz[-1] + 0.5

    def band(m, s, col):
        up = " ".join(f"({f(a)},{f(b)})" for a, b in zip(zz, m + s) if np.isfinite(b))
        dn = " ".join(f"({f(a)},{f(b)})" for a, b in reversed(list(zip(zz, m - s))) if np.isfinite(b))
        return (rf"  \addplot[draw=none, fill={col}, fill opacity=0.2, forget plot] "
                rf"coordinates {{{up} {dn}}} -- cycle;")

    def line(m, col, lab):
        pts = " ".join(f"({f(a)},{f(b)})" for a, b in zip(zz, m) if np.isfinite(b))
        return (rf"  \addplot[{col}, line width=0.9pt] coordinates {{{pts}}};" "\n"
                rf"  \addlegendentry{{{lab}}}")

    def shade():
        if shade_from is None:
            return []
        return [rf"  \fill[nypgrid, opacity=0.6] (axis cs:{f(shade_from - 0.5)},"
                r"\pgfkeysvalueof{/pgfplots/ymin}) rectangle "
                rf"(axis cs:{f(hi)},\pgfkeysvalueof{{/pgfplots/ymax}});"]

    labs = list(top)
    cols = {lab: f"nyp{i}" for i, lab in enumerate(labs)}
    L = [r"% Generated by scripts/nyumets_volumes_csv.py --tikz. Do not edit by hand: rerun it.",
         r"% Needs \usepackage{pgfplots} \usepgfplotslibrary{groupplots} \pgfplotsset{compat=1.16}.",
         r"% Width: \nyuprofilewidth, a LENGTH (\textwidth, \linewidth, a \newlength); default",
         r"% \linewidth.",
         rf"% Caption text: mean +/- s.e.m. over {n_vols} volumes; bottom: paired gain over "
         rf"{labs[0]}.",
         r"\providecommand{\nyuprofilewidth}{\linewidth}"]
    L += [rf"\definecolor{{{cols[lab]}}}{{HTML}}{{{PALETTE[i % len(PALETTE)].lstrip('#').upper()}}}"
          for i, lab in enumerate(labs)]
    L += [r"\definecolor{nypmuted}{HTML}{52514E}", r"\definecolor{nypgrid}{HTML}{E1E0D9}",
          r"\begin{tikzpicture}",
          r"\begin{groupplot}[group style={group size=1 by 2, vertical sep=0.04\nyuprofilewidth,",
          r"    x descriptions at=edge bottom},",
          rf"  width=0.92\nyuprofilewidth, xmin={f(lo)}, xmax={f(hi)}, scale only axis=false,",
          r"  axis lines=left, axis line style={nypmuted}, tick style={nypmuted},",
          r"  tick align=outside, ymajorgrids, grid style={nypgrid},",
          r"  legend style={draw=none, fill=none, font=\footnotesize}, legend cell align=left,",
          r"  label style={font=\small}, tick label style={font=\footnotesize},",
          r"  xtick distance=5, legend pos=south east]",
          r"\nextgroupplot[height=0.5\nyuprofilewidth, ylabel={PSNR (dB) $\uparrow$}]"]
    L += shade()
    L += [band(m, s, cols[lab]) for lab, (m, s) in top.items()]
    L += [line(m, cols[lab], lab) for lab, (m, s) in top.items()]
    L += [r"\nextgroupplot[height=0.34\nyuprofilewidth, ylabel={paired gain (dB)},",
          r"  xlabel={original slice index}]"]
    L += shade()
    for i, (lab, (m, s)) in enumerate(bot.items()):
        L += [band(m, s, cols[labs[i + 1]]), line(m, cols[labs[i + 1]], lab)]
    L += [rf"  \addplot[nypmuted, line width=0.5pt, forget plot] coordinates {{({f(lo)},0) "
          rf"({f(hi)},0)}};",
          r"\end{groupplot}", r"\end{tikzpicture}"]
    with open(f"{out}.tikz", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L) + "\n")
    with open(f"{out}_standalone.tex", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join([
            r"\documentclass[border=4pt]{standalone}", r"\usepackage{pgfplots}",
            r"\usepgfplotslibrary{groupplots}", r"\pgfplotsset{compat=1.16}",
            r"\newlength{\nyuprofilewidth}\setlength{\nyuprofilewidth}{12cm}",
            r"\begin{document}", rf"\input{{{name}.tikz}}", r"\end{document}"]) + "\n")
    print(f"wrote {out}.tikz and {out}_standalone.tex")


if __name__ == "__main__":
    main()
