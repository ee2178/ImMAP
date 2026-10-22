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
name alone. By default: the whole dumped range, and -- when it crosses --trained-hi, the end of
the training slice range (90) -- the trained part and the unseen part separately, since slices
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


def read_run(d):
    """volume -> {"slice": (n,), metric: (n,)} from a run's dump, and the shared attrs."""
    out, attrs = {}, {}
    for f in sorted(os.listdir(d)):
        if not f.endswith(".h5"):
            continue
        with h5py.File(os.path.join(d, f), "r") as h:
            out[f[:-3]] = dict(slice=h["slice_index"][:].astype(int),
                               **{m: h[f"{m}_slice"][:].astype(float) for m in METRICS})
            if not attrs:
                attrs = {k: h.attrs[k] for k in ("seed", "intensity_units", "nfe") if k in h.attrs}
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
    p.add_argument("--trained-hi", type=int, default=90,
                   help="end of the slice range the nets trained on (default 90)")
    p.add_argument("--out", default=os.path.join(HERE, "results", "nyumets_eval"),
                   help="directory for slices.csv and volumes.csv")
    p.add_argument("--profile", default=os.path.join(HERE, "figures", "out", "nyumets_profile"),
                   help="z-profile figure, path without extension ('' = none)")
    p.add_argument("--from-csv", metavar="SLICES_CSV",
                   help="draw the z-profile from a slices.csv this script wrote (e.g. copied off "
                        "the cluster) instead of reading dumps; writes no CSV")
    a = p.parse_args(argv)

    if a.from_csv:
        data, labels = {}, []
        with open(a.from_csv, newline="") as f:
            for r in csv.DictReader(f):
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
        profile(data, labels, vols, a.trained_hi, a.profile or "nyumets_profile")
        return

    runs = [r.split("=", 1) if "=" in r else (r, r) for r in a.runs]
    data, attrs = {}, {}
    for label, run in runs:
        data[label], attrs[label] = read_run(dump_dir(a.root, run))
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
        if lo < a.trained_hi < hi:
            groups += [(lo, a.trained_hi), (a.trained_hi, hi)]

    def gname(g):
        # ", <tag>" is where the bar chart breaks the tick label (plot_main_bars.group_label)
        tag = (", all" if (g[0], g[1]) == (lo, hi) and len(groups) > 1 and lo < a.trained_hi < hi
               else ", unseen" if g[0] >= a.trained_hi
               else ", trained" if g[1] <= a.trained_hi else "")
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
                    for i in np.where(sel)[0]:
                        srows.append(dict(base, slice=int(d["slice"][i]),
                                          **{m: f"{d[m][i]:.6g}" for m in METRICS}))
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
        profile(data, labels, vols, a.trained_hi, a.profile)

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


def profile(data, labels, vols, trained_hi, out):
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
    for i, lab in enumerate(labels):
        mu, se = np.array([ms(at[lab][z]) for z in zs]).T
        c = PALETTE[i % len(PALETTE)]
        ax[0].plot(zz, mu, color=c, lw=1.4, label=lab)
        ax[0].fill_between(zz, mu - se, mu + se, color=c, alpha=0.2, lw=0)
        if lab != labels[0]:
            g, gs = np.array([ms(gain[lab][z]) for z in zs]).T
            ax[1].plot(zz, g, color=c, lw=1.4, label=f"{lab} - {labels[0]}")
            ax[1].fill_between(zz, g - gs, g + gs, color=c, alpha=0.2, lw=0)
    ax[1].axhline(0, color=MUTED, lw=0.8)
    for x in ax:
        if zs[0] < trained_hi <= zs[-1]:
            x.axvspan(trained_hi - 0.5, zs[-1] + 0.5, color=GRID, alpha=0.6, lw=0, zorder=0)
        x.grid(axis="y", color=GRID, lw=0.6)
        for sp in ("top", "right"):
            x.spines[sp].set_visible(False)
    ax[1].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax[0].set_ylabel("PSNR (dB) ↑")
    ax[1].set_ylabel("paired gain (dB)")
    ax[1].set_xlabel("original slice index" + (f"  (shaded: >= {trained_hi}, unseen in training)"
                                                 if zs[0] < trained_hi <= zs[-1] else ""))
    ax[0].legend(frameon=False, loc="lower left")
    ax[1].legend(frameon=False, loc="lower left", fontsize=8)
    fig.text(0.5, 0.005, f"mean ± s.e.m. over {len(vols)} volumes", ha="center", va="bottom",
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=300)
    print(f"wrote {out}.png, .pdf")


if __name__ == "__main__":
    main()
