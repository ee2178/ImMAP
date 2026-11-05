#!/usr/bin/env python3
"""
Is a net's aliasing sensitive to the image's INTENSITY? Rescale the input, keep the SNR.

    python scripts/scale_sweep.py --runs trained_nets/mg_recon/knee \
        --only lpdsnetcf32_R8 mg81v6vtxAnestUcf32_R8 varnetmapscf32_R8 ramcf32_R8 \
        --flags results/alias_slices_cf32_all/knee/slices_R8.csv --top 40 --controls 40 \
        --out results/scale_sweep/knee

The question (2026-11-05): LPDS thresholds are w0 + w1 sigma with w0 in absolute data units,
and the data are only relatively normalized, so a volume brighter than the training average
gets a relatively weaker sparsity prior, which could let ghosts through. Two tests, no
training:

  1. INTENSITY OF THE FLAGGED SLICES. s = RAM's signal statistic mean(|Re y|, |Im y|)
     (models/cg_init.py::ram_signal_level) for each flagged slice and for a random sample
     of unflagged ones from the same alias_check.py slices_R<R>.csv.
  2. RESCALING. Each trained net is run on (c y, c sigma) for every c in --scales and its
     output divided by c. Same noise draw, same operator (online maps are estimated from
     the unscaled y), same SNR: only the absolute intensity the net sees changes. A
     scale-equivariant net gives the same image at every c. If the ghost score falls at
     some c, the net's thresholds are mis-scaled for those slices.

Scores, per slice and c, on the dump images (the recon adapter, as dump_eval.py):
  rho_at   alias_check.py's local correlation of the PE-high-passed residual with the
           PE-high-passed zero-filled image, AT the slice's flagged location from the CSV
           (the same pixel for every c, so it follows that ghost)
  rho_top  the slice's highest rho where the residual is strong enough to see, as
           alias_check.py --top-slices scores a slice (strength scale shared across c)
  psnr, ssim, nrmse from the dump, inside the organ mask

Outputs under --out: <run>/c<c>/eval_dump/*.h5 (ordinary dumps), sweep.csv (one row per
run x slice x c), summary.txt, sweep.png.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import torch
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import datasets                                                    # noqa: F401,E402
from alias_check import hp_pe, local_corr, slice_sigma, strength_map, top_patches  # noqa: E402
from dump_eval import dump_run, find_runs, load_run, same_source  # noqa: E402
from evaluation.metrics import build_metrics                       # noqa: E402
from evaluation.tasks import _call_net, default_sigma, force_organ_mask  # noqa: E402
from models.cg_init import ram_signal_level                        # noqa: E402

METRICS = ("psnr", "ssim", "nrmse")


def select_slices(path, top, controls, seed):
    """{(volume, slice): CSV row} for the `top` best-ranked flagged slices and `controls`
    unflagged ones drawn at random (seeded)."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    flagged = sorted((r for r in rows if int(r["flagged"])), key=lambda r: int(r["rank"]))
    flagged = flagged[:top] if top else flagged
    if not flagged:
        raise SystemExit(f"nothing flagged in {path}")
    rest = [r for r in rows if not int(r["flagged"])]
    n = len(flagged) if controls is None else controls
    pick = np.random.default_rng(seed).choice(len(rest), size=min(n, len(rest)), replace=False)
    chosen = flagged + [rest[i] for i in sorted(pick)]
    return {(r["volume"], int(r["slice"])): r for r in chosen}


def scaled_call(c, s_log, cur):
    """The forward on (c y, c sigma), divided back by c. Records s of the UNSCALED y."""
    def call(net, y, E, sigma):
        s_log[cur["id"]] = float(ram_signal_level(y).mean())
        sig = None if sigma is None else sigma * c
        return _call_net(net, y * c, E, sig) / c
    return call


def label(c):
    return f"c{c:g}"


def score_run(dests, scales, chosen, s_log, a):
    """One row per slice x c, from the dumps in `dests` {c: dir}."""
    win = (a.win_rows, a.win_cols)
    vols = sorted({v for v, _ in chosen})
    out = []
    for vol in vols:
        data = {}
        for c in scales:
            with h5py.File(os.path.join(dests[c], vol + ".h5"), "r") as h:
                data[c] = {k: h[k][:] for k in ("recon", "reference", "zero_filled",
                                                "organ_mask", "slice_index")
                           + tuple(f"{m}_slice" for m in METRICS) if k in h}
        base = data[scales[0]]
        for j, sl in enumerate(base["slice_index"]):
            sl = int(sl)
            row = chosen.get((vol, sl))
            if row is None:
                continue
            ref, zf = np.abs(base["reference"][j]), np.abs(base["zero_filled"][j])
            inner = ndi.binary_erosion(base["organ_mask"][j] > 0, iterations=4)
            recs = {c: np.abs(data[c]["recon"][j]) for c in scales}
            sigma = slice_sigma(ref, list(recs.values()), inner) if inner.sum() >= 100 else np.nan
            t = hp_pe(zf)
            r0, c0 = int(row["row"]), int(row["col"])
            for c in scales:
                e = hp_pe(recs[c] - ref)
                rho = local_corr(e, t, win)
                at = float(rho[r0, c0]) if r0 >= 0 and c0 >= 0 else np.nan
                top = np.nan
                if np.isfinite(sigma):
                    st = strength_map(e, sigma, win)
                    tp = top_patches(np.where(st > a.min_strength, rho, -np.inf),
                                     np.abs(e) / (sigma + 1e-12), inner, a, n=1)
                    top = tp[0][0] if tp else -1.0
                d = dict(volume=vol, slice=sl, flagged=int(row["flagged"]),
                         rank=int(row["rank"]), s=s_log.get((vol, sl), np.nan), c=c,
                         rho_at=round(at, 4), rho_top=round(float(top), 4))
                for m in METRICS:
                    k = f"{m}_slice"
                    d[m] = round(float(data[c][k][j]), 5) if k in data[c] else np.nan
                out.append(d)
    return out


def q(x, p):
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    return np.percentile(x, p) if len(x) else np.nan


def summarize(rows, runs, scales):
    L = []
    first = [r for r in rows if r["run"] == runs[0] and r["c"] == scales[0]]
    for name, fl in (("flagged", 1), ("control", 0)):
        s = [r["s"] for r in first if r["flagged"] == fl]
        L.append(f"s  {name:<8} n={len(s):<4} median {q(s, 50):.4g}   IQR {q(s, 25):.4g} - {q(s, 75):.4g}")
    sf = [r["s"] for r in first if r["flagged"]]
    sc = [r["s"] for r in first if not r["flagged"]]
    if sf and sc:
        frac = np.mean([np.mean(np.asarray(sc) < v) for v in sf])
        L.append(f"   a flagged slice is brighter than {100 * frac:.0f}% of the controls on average "
                 f"(50% = no shift)")
    L.append("")
    for run in runs:
        rr = [r for r in rows if r["run"] == run]
        L.append(run)
        L.append(f"  {'c':>6}  {'rho_at fl':>10}{'rho_top fl':>11}{'nrmse fl':>10}"
                 f"{'rho_at ctl':>11}{'rho_top ctl':>12}{'nrmse ctl':>10}")
        for c in scales:
            f = [r for r in rr if r["c"] == c and r["flagged"]]
            k = [r for r in rr if r["c"] == c and not r["flagged"]]
            L.append(f"  {c:>6g}  {q([r['rho_at'] for r in f], 50):>10.3f}"
                     f"{q([r['rho_top'] for r in f], 50):>11.3f}{q([r['nrmse'] for r in f], 50):>10.4f}"
                     f"{q([r['rho_at'] for r in k], 50):>11.3f}{q([r['rho_top'] for r in k], 50):>12.3f}"
                     f"{q([r['nrmse'] for r in k], 50):>10.4f}")
        # Which c each flagged slice likes best, against its brightness.
        best = {}
        for r in rr:
            if r["flagged"] and np.isfinite(r["rho_at"]):
                key = (r["volume"], r["slice"])
                if key not in best or r["rho_at"] < best[key][1]:
                    best[key] = (r["c"], r["rho_at"], r["s"])
        if best:
            s_all = np.array([b[2] for b in best.values()])
            cuts = np.percentile(s_all, [100 / 3, 200 / 3])
            for lo, hi, nm in ((-np.inf, cuts[0], "dimmest third"), (cuts[0], cuts[1], "middle third"),
                               (cuts[1], np.inf, "brightest third")):
                cs = [b[0] for b in best.values() if lo < b[2] <= hi]
                L.append(f"  flagged, {nm:<16} lowest rho_at at c = median {q(cs, 50):g} (n={len(cs)})")
        L.append("")
    return "\n".join(L)


def plot(rows, runs, scales, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(runs) + 1
    fig, ax = plt.subplots(1, n, figsize=(3.6 * n, 3.3), squeeze=False)
    ax = ax[0]
    ink, fl_col, ct_col = "#222222", "#c4452f", "#2f6db5"
    for i, run in enumerate(runs):
        for fl, col, ls, nm in ((1, fl_col, "-", "flagged"), (0, ct_col, "--", "control")):
            med = [q([r["rho_at"] for r in rows if r["run"] == run and r["c"] == c
                      and r["flagged"] == fl], 50) for c in scales]
            lo = [q([r["rho_at"] for r in rows if r["run"] == run and r["c"] == c
                     and r["flagged"] == fl], 25) for c in scales]
            hi = [q([r["rho_at"] for r in rows if r["run"] == run and r["c"] == c
                     and r["flagged"] == fl], 75) for c in scales]
            ax[i].plot(scales, med, ls, color=col, lw=2, marker="o", ms=4, label=nm)
            ax[i].fill_between(scales, lo, hi, color=col, alpha=0.12, lw=0)
        ax[i].set_xscale("log", base=2)
        ax[i].set_xticks(scales, [f"{c:g}" for c in scales])
        ax[i].axvline(1.0, color="#999999", lw=0.8, zorder=0)
        ax[i].set_title(run, fontsize=8, color=ink)
        ax[i].set_xlabel("input scale c (y and sigma)", fontsize=8)
        if i == 0:
            ax[i].set_ylabel("rho at the flagged ghost\n(median, IQR band)", fontsize=8)
            ax[i].legend(fontsize=7, frameon=False)
        ax[i].grid(alpha=0.25)
        ax[i].spines[["top", "right"]].set_visible(False)
    first = [r for r in rows if r["run"] == runs[0] and r["c"] == scales[0]]
    sf = [r["s"] for r in first if r["flagged"]]
    sc = [r["s"] for r in first if not r["flagged"]]
    bins = np.histogram_bin_edges(sf + sc, bins=20)
    ax[-1].hist(sc, bins=bins, color=ct_col, alpha=0.55, label="control")
    ax[-1].hist(sf, bins=bins, color=fl_col, alpha=0.55, label="flagged")
    ax[-1].set_title("signal level s = mean(|Re y|, |Im y|)", fontsize=8, color=ink)
    ax[-1].set_xlabel("s", fontsize=8)
    ax[-1].legend(fontsize=7, frameon=False)
    ax[-1].spines[["top", "right"]].set_visible(False)
    for A in ax:
        A.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", required=True, help="anatomy dir of trained runs")
    p.add_argument("--only", nargs="+", required=True, help="run directory NAMES, e.g. lpdsnetcf32_R8")
    p.add_argument("--flags", required=True, help="an alias_check.py slices_R<R>.csv")
    p.add_argument("--top", type=int, default=40, help="best-ranked flagged slices (0: all)")
    p.add_argument("--controls", type=int, help="unflagged slices drawn at random "
                                                "(default: as many as flagged)")
    p.add_argument("--scales", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0, 4.0])
    p.add_argument("--val-kspace-type", choices=["simulated", "measurement_awgn"],
                   default="measurement_awgn", help="must match the CSV's kspace_type")
    p.add_argument("--sigma", type=float, help="default: each run's val_noise_std")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--win-rows", type=int, default=25)
    p.add_argument("--win-cols", type=int, default=9)
    p.add_argument("--min-strength", type=float, default=2.5)
    p.add_argument("--device")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--rescore", action="store_true",
                   help="skip the forwards; score the dumps already under --out (s is then "
                        "read back from sweep.csv)")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    scales = sorted(set(a.scales))
    if 1.0 not in scales:
        raise SystemExit("--scales must include 1 (the net as trained)")
    scales = [1.0] + [c for c in scales if c != 1.0]           # c = 1 first: it is the base
    os.makedirs(a.out, exist_ok=True)

    chosen = select_slices(a.flags, a.top, a.controls, a.seed)
    keep = {}
    for v, s in chosen:
        keep.setdefault(v, set()).add(s)
    nfl = sum(int(r["flagged"]) for r in chosen.values())
    print(f"{nfl} flagged + {len(chosen) - nfl} control slices in {len(keep)} volumes | "
          f"scales {scales}")

    runs = find_runs(a.runs, a.only, exact=True)
    names = [os.path.basename(os.path.normpath(r)) for r in runs]
    missing = sorted(set(a.only) - set(names))
    if missing:
        print(f"[sweep] no trained run named: {' '.join(missing)}  (skipped)")
    if not runs:
        raise SystemExit("no runs")
    order = {n: i for i, n in enumerate(a.only)}
    runs = sorted(runs, key=lambda r: order[os.path.basename(os.path.normpath(r))])
    names = [os.path.basename(os.path.normpath(r)) for r in runs]

    s_prev = {}
    if a.rescore:
        with open(os.path.join(a.out, "sweep.csv"), newline="") as f:
            for r in csv.DictReader(f):
                s_prev[(r["volume"], int(r["slice"]))] = float(r["s"])

    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    metrics = build_metrics(list(METRICS))
    sl_all = [s for v in keep.values() for s in v]
    rows, first_val = [], None
    for run_dir, name in zip(runs, names):
        dests = {c: os.path.join(a.out, name, label(c), "eval_dump") for c in scales}
        s_log = dict(s_prev)
        if not a.rescore:
            cfg, model, n_params, step = load_run(run_dir, device)
            if first_val is None:
                first_val = cfg["data"]["val"]
            elif not same_source(cfg["data"]["val"], first_val):
                print(f"[skip] {name}: reads a different val set")
                continue
            force_organ_mask(cfg)
            cfg.setdefault("training", {})["val_kspace_type"] = a.val_kspace_type
            sigma = a.sigma if a.sigma is not None else default_sigma(cfg)
            for c in scales:
                cur = {"id": None}
                args = SimpleNamespace(slices=f"{min(sl_all)}:{max(sl_all) + 1}",
                                       workers=a.workers, seed=a.seed, limit=None,
                                       keep_slices=keep, call=scaled_call(c, s_log, cur),
                                       on_item=lambda v, s, cur=cur: cur.update(id=(v, s)))
                n = dump_run(run_dir, cfg, model, n_params, step, args, sorted(keep),
                             metrics, sigma, dests[c])
                print(f"  {name:<34} c={c:<5g} sigma={sigma:g}  {n} slices  (step {step})")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if not s_prev:
            s_prev = dict(s_log)              # s depends on the measurement only, not the net
        for r in score_run(dests, scales, chosen, s_log, a):
            rows.append(dict(run=name, **r))

    if not rows:
        raise SystemExit("nothing scored")
    rows.sort(key=lambda r: (names.index(r["run"]), -r["flagged"], r["rank"], r["c"]))
    with open(os.path.join(a.out, "sweep.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    plot_scales = sorted(scales)
    txt = summarize(rows, names, plot_scales)
    print("\n" + txt)
    with open(os.path.join(a.out, "summary.txt"), "w") as f:
        f.write(f"flags {a.flags} | {nfl} flagged + {len(chosen) - nfl} controls | "
                f"scales {plot_scales}\n\n" + txt + "\n")
    plot(rows, names, plot_scales, os.path.join(a.out, "sweep.png"))
    print(f"wrote sweep.csv, summary.txt, sweep.png to {a.out}")


if __name__ == "__main__":
    main()
