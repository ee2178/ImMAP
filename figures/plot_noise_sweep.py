#!/usr/bin/env python3
"""
Noise-level generalization: each net's metric against the TEST noise level, with
the training range shaded -- the CDLNet / LPDSNet figure.

    python figures/plot_noise_sweep.py                                  # results/noise_sweep_measurement_awgn
    python figures/plot_noise_sweep.py --metrics psnr ssim --ref MGLPDS-Big
    python figures/plot_noise_sweep.py --nets E2E-VarNet LPDSNet MGLPDS-Small \\
        --rename MGLPDS-Small=MGLPDS --ref MGLPDS

Reads `<root>/s<sigma>/volumes.csv`, one tree per test sigma, as
torch/noise_sweep.sbatch writes them (scripts/eval_per_volume.py), through the
main bar chart's own loader -- same labels, same one-run-per-net rule.

Per (anatomy, R), two rows of panels per metric:
  top     mean +- SEM over volumes, per net, against sigma
  bottom  the PAIRED difference to --ref (net - ref, per volume, then mean +- SEM)
          -- the absolute curves fall steeply with sigma and hide the gaps
          between nets; the differences show who generalizes. Positive = better
          than the reference for every metric (NRMSE / LPIPS differences are
          sign-flipped so up is always better).
Volumes are matched across nets at each sigma (only those every net has).

Also writes <out>.csv: anatomy, R, net, sigma, metric, mean, sem, n, and the
paired difference to the reference with its SEM.
"""

from __future__ import annotations

import argparse
import csv
import glob
import importlib.util
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

_spec = importlib.util.spec_from_file_location(
    "plot_main_bars", os.path.join(HERE, "plot_main_bars.py"))
_pb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pb)
load, HIGHER_IS_BETTER, ORDER = _pb.load, _pb.HIGHER_IS_BETTER, _pb.ORDER
INK, MUTED, GRID, SURFACE, YLABEL = _pb.INK, _pb.MUTED, _pb.GRID, _pb.SURFACE, _pb.YLABEL

# Colour follows the net (the main figure's four), plus RAM. Markers too, so a
# net is identifiable without colour.
COLORS = dict(_pb.COLORS, RAM="#8a55c9", MGLPDS="#1baf7a")
MARKERS = {"E2E-VarNet": "s", "LPDSNet": "^", "MGLPDS-Small": "o", "MGLPDS": "o",
           "MGLPDS-Big": "D", "RAM": "v"}
ORDER_ALL = ORDER + ["MGLPDS", "RAM"]


def train_range(anatomy):
    """The training sigma range, read from the config generator (as the
    launchers do); None if it cannot be read."""
    try:
        spec = importlib.util.spec_from_file_location(
            "_mkcfg", os.path.join(ROOT, "scripts", "make_mg_recon_configs.py"))
        gen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gen)
        ns = gen.NOISE_STD
        return tuple(float(v) for v in (ns[anatomy] if isinstance(ns, dict) else ns))
    except Exception:                                              # noqa: BLE001
        return None


def sem(x):
    x = np.asarray(x, float)
    return 0.0 if len(x) < 2 else float(np.std(x, ddof=1) / np.sqrt(len(x)))


def collect(root, nets=None, rename=None):
    """{(anatomy, R): {sigma: {net: {volume: {metric: value}}}}}, metric names."""
    out, metrics = {}, None
    trees = sorted(glob.glob(os.path.join(root, "s*", "volumes.csv")))
    if not trees:
        raise SystemExit(f"no s<sigma>/volumes.csv under {root}")
    for path in trees:
        m = re.search(r"[\\/]s([0-9.eE+-]+)[\\/]volumes\.csv$", path)
        if not m:
            continue
        sigma = float(m.group(1))
        data, ms = load(path, nets=nets, rename=rename)
        metrics = metrics or ms
        for key, by_net in data.items():
            out.setdefault(key, {})[sigma] = by_net
    return out, metrics


def summarise(sweep, metrics, ref):
    """Rows of the summary CSV, and {(key, metric): {net: [(sigma, mean, sem,
    dmean, dsem)]}} for plotting."""
    rows, curves = [], {}
    for key, by_sigma in sorted(sweep.items()):
        for m in metrics:
            sign = 1.0 if HIGHER_IS_BETTER[m] else -1.0
            for sigma in sorted(by_sigma):
                by_net = by_sigma[sigma]
                nets = [n for n in by_net if any(m in v for v in by_net[n].values())]
                common = sorted(set.intersection(*[{v for v, d in by_net[n].items() if m in d}
                                                   for n in nets])) if nets else []
                if not common:
                    continue
                x = {n: np.array([by_net[n][v][m] for v in common]) for n in nets}
                for n in nets:
                    d = sign * (x[n] - x[ref]) if ref in x else None
                    rec = dict(anatomy=key[0], R=key[1], net=n, sigma=sigma, metric=m,
                               mean=float(x[n].mean()), sem=sem(x[n]), n=len(common),
                               diff_vs_ref=("" if d is None else float(d.mean())),
                               diff_sem=("" if d is None else sem(d)))
                    rows.append(rec)
                    curves.setdefault((key, m), {}).setdefault(n, []).append(
                        (sigma, rec["mean"], rec["sem"],
                         None if d is None else float(d.mean()), None if d is None else sem(d)))
    return rows, curves


def plot(curves, keys, metrics, ref, out, scale):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, NullLocator

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "font.family": "DejaVu Sans"})
    nrow, ncol = 2 * len(metrics), len(keys)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 2.5 * nrow),
                             squeeze=False, facecolor=SURFACE, sharex="col")
    handles = {}
    for j, key in enumerate(keys):
        tr = train_range(key[0])
        for i, m in enumerate(metrics):
            by_net = curves.get((key, m), {})
            nets = sorted(by_net, key=lambda n: ORDER_ALL.index(n) if n in ORDER_ALL else 99)
            sig = sorted({s for pts in by_net.values() for s, *_ in pts})
            for r, diff in ((2 * i, False), (2 * i + 1, True)):
                ax = axes[r][j]
                ax.set_facecolor(SURFACE)
                if tr:
                    ax.axvspan(tr[0], tr[1], color=GRID, alpha=0.7, lw=0, zorder=0)
                for n in nets:
                    pts = sorted(by_net[n])
                    s = np.array([p[0] for p in pts])
                    if diff:
                        if pts[0][3] is None:
                            continue
                        y = np.array([p[3] for p in pts]); e = np.array([p[4] for p in pts])
                    else:
                        y = np.array([p[1] for p in pts]); e = np.array([p[2] for p in pts])
                    c = COLORS.get(n, MUTED)
                    ax.fill_between(s, y - e, y + e, color=c, alpha=0.18, lw=0, zorder=2)
                    h, = ax.plot(s, y, color=c, lw=1.6, marker=MARKERS.get(n, "o"), ms=4.5,
                                 mec=SURFACE, mew=0.8, zorder=3, label=n)
                    handles.setdefault(n, h)
                if diff:
                    ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
                    unit = " (dB)" if m == "psnr" else ""
                    ax.set_ylabel(f"Δ {m.upper()} vs {ref}{unit}\n(up = better)", fontsize=8)
                else:
                    ax.set_ylabel(YLABEL.get(m, m))
                ax.set_xscale(scale)
                if sig:
                    ax.xaxis.set_major_locator(FixedLocator(sig))
                    ax.xaxis.set_minor_locator(NullLocator())
                    ax.set_xticklabels([f"{v:g}" for v in sig], rotation=45, fontsize=7.5)
                ax.grid(axis="y", color=GRID, lw=0.6)
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
                if r == 0:
                    ax.set_title(f"{key[0].capitalize()}, R={key[1]}", color=INK, fontsize=10)
                if r == nrow - 1:
                    ax.set_xlabel("test noise σ")
    order = [n for n in ORDER_ALL if n in handles] + [n for n in handles if n not in ORDER_ALL]
    fig.legend([handles[n] for n in order], order, loc="upper center", ncol=len(order),
               frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.5, 0.0, "Shaded: training range. Bands: ± SEM over volumes "
             "(paired across nets at each σ). Every net is told σ of the added noise.",
             ha="center", va="bottom", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=os.path.join(ROOT, "results", "noise_sweep_measurement_awgn"))
    p.add_argument("--metrics", nargs="*", default=["psnr"],
                   help="psnr ssim nrmse lpips (default psnr)")
    p.add_argument("--nets", nargs="*", help="keep only these (labels before --rename)")
    p.add_argument("--rename", nargs="*", default=[], metavar="OLD=NEW")
    p.add_argument("--ref", default="MGLPDS-Big",
                   help="the reference of the difference panels (a label after --rename)")
    p.add_argument("--x-scale", choices=["log", "linear"], default="log")
    p.add_argument("--out", default=os.path.join(HERE, "out", "noise_sweep"))
    a = p.parse_args(argv)

    rename = dict(r.split("=", 1) for r in a.rename)
    sweep, metrics = collect(a.root, nets=a.nets, rename=rename)
    metrics = [m for m in a.metrics if m in metrics]
    if not metrics:
        raise SystemExit(f"none of {a.metrics} in the sweep")
    rows, curves = summarise(sweep, metrics, a.ref)
    nets_seen = {r["net"] for r in rows}
    if a.ref not in nets_seen:
        print(f"[noise_sweep] reference {a.ref!r} not in {sorted(nets_seen)}: "
              f"difference panels left empty", file=sys.stderr)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(f"{a.out}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()}
                    for r in rows)
    keys = sorted(sweep)
    plot(curves, keys, metrics, a.ref, a.out, a.x_scale)

    for key in keys:
        for m in metrics:
            print(f"\n{key[0]} R{key[1]}  {m}   (mean over volumes; * = inside the training range)")
            by_net = curves[(key, m)]
            sig = sorted({s for pts in by_net.values() for s, *_ in pts})
            tr = train_range(key[0])
            print(f"  {'':<14}" + "".join(
                f"{(f'{s:g}' + ('*' if tr and tr[0] <= s <= tr[1] else '')):>9}" for s in sig))
            for n in sorted(by_net, key=lambda n: ORDER_ALL.index(n) if n in ORDER_ALL else 99):
                vals = {s: mu for s, mu, *_ in by_net[n]}
                print(f"  {n:<14}" + "".join(
                    f"{vals[s]:9.3f}" if s in vals else f"{'-':>9}" for s in sig))
    print(f"\nwrote {a.out}.png, .pdf, .csv")


if __name__ == "__main__":
    main()
