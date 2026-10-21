#!/usr/bin/env python3
"""
Rank dumped slices by how well or badly one method does -- what to open in the
viewer.

    python scripts/rank_slices.py --anatomy brain --R 8                       # mg169v6's biggest wins
    python scripts/rank_slices.py --anatomy brain --R 8 --target lpdsnet --mode loses
    python scripts/rank_slices.py --anatomy knee --R 12 --target lpdsnet --mode worst
    python scripts/rank_slices.py --anatomy brain --R 8 --vs lpdsnet          # wins over lpdsnet alone
    python scripts/rank_slices.py --anatomy brain --R 8 --by volume           # whole volumes

Reads only the per-slice metrics scripts/dump_eval.py stored
(`<dump-root>/<anatomy>/<cell>_R<R>/eval_dump/<volume>.h5`, `<metric>_slice`),
so it runs on a login node in seconds: no torch, no GPU, no images loaded. Those
are the numbers the viewer shows, scored inside the organ mask.

Off the cluster, rank from a CSV instead of the HDF5 files:

    python scripts/rank_slices.py --export dump_slices.csv            # on the cluster
    python scripts/rank_slices.py --from-csv dump_slices.csv --anatomy brain --R 8

`--from-csv results/main/slices.csv` also works and covers EVERY val volume, but
those numbers come from eval_per_volume's per-slice noise seeds, not the dumps'
draw, so they differ slightly from what the viewer shows, and a volume it ranks
may not have been dumped.

Modes, for `--target` against `--vs` (default: every other method):

    wins    target beats the BEST of the others by the most
    loses   target trails the best of the others by the most
    worst   target's own value is worst (the others shown for context)
    best    target's own value is best

`margin` is signed so that positive always means the target is better, whatever
the metric's direction (PSNR/SSIM up, NRMSE/LPIPS down). Only volumes dumped for
EVERY method are ranked -- the viewer shows a volume only when every column has
it.

The last column is what to type into the web viewer's filter box (or pass as
`--volume ... --slice ...` to figures/viewer.py) to land on that slice.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

HIGHER_IS_BETTER = {"psnr": True, "ssim": True, "nrmse": False, "lpips": False}
METHODS = ["varnetmaps", "lpdsnet", "mg81v6", "mg169v6"]


CSV_KEYS = ["anatomy", "R", "net", "run", "volume", "slice"]


def _dump_files(dump_root, anatomy, R, m):
    d = os.path.join(dump_root, anatomy, f"{m}_R{R}", "eval_dump")
    files = sorted(glob.glob(os.path.join(d, "*.h5")))
    if not files:
        raise SystemExit(f"no dumps in {d} -- check --dump-root / --methods")
    return files


def load(dump_root, anatomy, R, methods, metric):
    """{method: {volume: (slice_index, values)}}, and each method's kspace_type."""
    import h5py

    out, kinds = {}, {}
    for m in methods:
        out[m] = {}
        for fp in _dump_files(dump_root, anatomy, R, m):
            with h5py.File(fp, "r") as f:
                key = f"{metric}_slice"
                if key not in f:
                    raise SystemExit(f"{fp} has no {key} (stored: "
                                     f"{sorted(k[:-6] for k in f if k.endswith('_slice'))})")
                kinds.setdefault(m, set()).add(str(f.attrs.get("kspace_type", "?")))
                out[m][os.path.splitext(os.path.basename(fp))[0]] = (
                    f["slice_index"][:].astype(int), f[key][:].astype(float))
    return out, kinds


def load_csv(path, anatomy, R, methods, metric):
    """`load`, from a CSV in the layout of eval_per_volume's slices.csv (which
    `--export` also writes): anatomy, R, net, run, volume, slice, <metrics>.
    Blank values (a metric left out of the merge) are skipped. eval_per_volume's
    volumes.csv (no `slice` column, one mean per volume) loads as one row per
    volume with slice -1, which only a --by volume ranking can use."""
    import csv

    acc = {m: {} for m in methods}
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        if metric not in (rd.fieldnames or []):
            raise SystemExit(f"{path} has no '{metric}' column ({rd.fieldnames})")
        for r in rd:
            if r["anatomy"] != anatomy or int(r["R"]) != R or r[metric] in ("", None):
                continue
            m = r["run"].rpartition("_R")[0] or r["run"]
            if m in acc:
                sl = int(r["slice"]) if "slice" in r else -1
                acc[m].setdefault(r["volume"], []).append((sl, float(r[metric])))
    missing = [m for m in methods if not acc[m]]
    if missing:
        raise SystemExit(f"{path}: no {anatomy} R{R} {metric} rows for {missing}")
    out = {m: {v: (np.array([s for s, _ in ps]), np.array([x for _, x in ps]))
               for v, ps in vols.items()} for m, vols in acc.items()}
    return out, {}


def export(dump_root, methods, path):
    """Every dumped per-slice metric, all anatomies and R, into one small CSV --
    copy that off the cluster instead of the HDF5 files."""
    import csv
    import re

    import h5py

    rows, metrics = [], []
    for d in sorted(glob.glob(os.path.join(dump_root, "*", "*_R*", "eval_dump"))):
        run = os.path.basename(os.path.dirname(d))
        anatomy = os.path.basename(os.path.dirname(os.path.dirname(d)))
        mt = re.fullmatch(r"(.+)_R(\d+)", run)
        if not mt or mt.group(1) not in methods:
            continue
        for fp in sorted(glob.glob(os.path.join(d, "*.h5"))):
            with h5py.File(fp, "r") as f:
                ms = sorted(k[:-6] for k in f if k.endswith("_slice"))
                metrics += [m for m in ms if m not in metrics]
                vals = {m: f[f"{m}_slice"][:] for m in ms}
                vol = os.path.splitext(os.path.basename(fp))[0]
                for i, sl in enumerate(f["slice_index"][:]):
                    rows.append(dict(anatomy=anatomy, R=mt.group(2), net=mt.group(1),
                                     run=run, volume=vol, slice=int(sl),
                                     **{m: f"{float(v[i]):.6g}" for m, v in vals.items()}))
    if not rows:
        raise SystemExit(f"no <anatomy>/<cell>_R<R>/eval_dump under {dump_root} for {methods}")
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=CSV_KEYS + metrics, restval="")
        wr.writeheader()
        wr.writerows(rows)
    runs = sorted({(r["anatomy"], r["run"]) for r in rows})
    print(f"wrote {path}: {len(rows)} slices from {len(runs)} runs "
          f"({', '.join(a + '/' + r for a, r in runs)})")


def rows_for(data, methods):
    """One row per (volume, slice) present for every method."""
    vols = sorted(set.intersection(*(set(data[m]) for m in methods)))
    rows = []
    for v in vols:
        idx = {m: dict(zip(*data[m][v])) for m in methods}
        for sl in sorted(set.intersection(*(set(idx[m]) for m in methods))):
            rows.append(dict(volume=v, slice=sl, **{m: idx[m][sl] for m in methods}))
    return rows


def by_volume(rows, methods):
    acc = {}
    for r in rows:
        acc.setdefault(r["volume"], []).append(r)
    return [dict(volume=v, slice=None, n=len(rs),
                 **{m: float(np.mean([r[m] for r in rs])) for m in methods})
            for v, rs in acc.items()]


def score(rows, target, others, metric, mode):
    up = HIGHER_IS_BETTER.get(metric, True)
    sign = 1.0 if up else -1.0
    for r in rows:
        best_other = (max if up else min)(r[o] for o in others)
        r["best_other"] = best_other
        r["margin"] = sign * (r[target] - best_other)
        r["own"] = sign * r[target]
    key = {"wins": lambda r: -r["margin"], "loses": lambda r: r["margin"],
           "worst": lambda r: r["own"], "best": lambda r: -r["own"]}[mode]
    return sorted(rows, key=key)


# --pick: which rankings choose the volumes to dump. (name, target, mode); target
# "MG" is mg169v6 where the CSV has it for that anatomy and R, else mg81v6.
PICKS = [
    ("MG wins", "MG", "wins"),             # MG beats the best other by the most
    ("LPDSNet loses", "lpdsnet", "loses"),  # LPDSNet trails the best other by the most
    ("MG loses", "MG", "loses"),            # where another net beats MG -- outliers
    ("MG worst", "MG", "worst"),            # hardest slices for MG outright
]


def pick(csv_path, out_dir, k, metric, methods):
    """Run every PICKS ranking for every (anatomy, R) in the CSV; per anatomy,
    write the union of the top `k` DISTINCT volumes of each to <anatomy>.txt (one
    name per line, what torch/mg_recon_dump.sbatch PICKS_DIR= reads) and the
    slices to open to <anatomy>.md. Both R go in one list: the dump job dumps
    every R of an anatomy in one task."""
    import csv

    with open(csv_path, newline="") as f:
        cells = {}
        for r in csv.DictReader(f):
            cells.setdefault((r["anatomy"], int(r["R"])), set()).add(
                r["run"].rpartition("_R")[0] or r["run"])
    os.makedirs(out_dir, exist_ok=True)
    fmt = "{:.2f}" if metric == "psnr" else "{:.4f}"
    for anatomy in sorted({a for a, _ in cells}):
        chosen, report = [], [f"# {anatomy}: volumes picked from {csv_path} ({metric})\n"]
        for R in sorted(R for a, R in cells if a == anatomy):
            have = [m for m in methods if m in cells[(anatomy, R)]]
            mg = "mg169v6" if "mg169v6" in have else "mg81v6"
            data, _ = load_csv(csv_path, anatomy, R, have, metric)
            rows = rows_for(data, have)
            report.append(f"\n## R{R}  (MG = {mg}; nets: {', '.join(have)})\n")
            for name, target, mode in PICKS:
                target = mg if target == "MG" else target
                if target not in have:
                    continue
                others = [m for m in have if m != target]
                ranked = score([dict(r) for r in rows], target, others, metric, mode)
                report.append(f"\n**{name}** ({target} vs best of {', '.join(others)}; "
                              f"margin > 0 = {target} better)\n\n"
                              f"| volume | slice | " + " | ".join(have)
                              + " | margin | viewer filter |\n|"
                              + "---|" * (len(have) + 4) + "\n")
                seen = []
                for r in ranked:
                    if r["volume"] in seen:
                        continue
                    seen.append(r["volume"])
                    report.append(
                        f"| {short(r['volume'])} | {r['slice']} | "
                        + " | ".join(fmt.format(r[m]) for m in have)
                        + f" | {fmt.format(r['margin'])} | "
                          f"`{r['volume'].split('_')[-1]} {r['slice']}` |\n")
                    if len(seen) == k:
                        break
                chosen += [v for v in seen if v not in chosen]
        txt = os.path.join(out_dir, f"{anatomy}.txt")
        with open(txt, "w", newline="\n") as f:
            f.write("\n".join(chosen) + "\n")
        with open(os.path.join(out_dir, f"{anatomy}.md"), "w") as f:
            f.write("".join(report))
        print(f"{anatomy}: {len(chosen)} volumes -> {txt} (where to look: {anatomy}.md)")


def short(volume):
    """`file_brain_AXT2_201_2010294` -> `brain_AXT2_201_2010294` (viewer's pretty)."""
    return volume[5:] if volume.startswith("file_") else volume


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--anatomy", choices=["brain", "knee"])
    p.add_argument("--R", type=int)
    p.add_argument("--dump-root", default="trained_nets/mg_recon")
    p.add_argument("--from-csv", metavar="CSV",
                   help="rank from this CSV instead of the dumps: one written by "
                        "--export (= the viewer's numbers), or eval_per_volume's "
                        "slices.csv (every val volume, but its own noise draw)")
    p.add_argument("--export", metavar="CSV",
                   help="write every dumped per-slice metric to CSV and stop")
    p.add_argument("--methods", nargs="+", default=METHODS,
                   help="cells to load (default: the main comparison's four)")
    p.add_argument("--target", default="mg169v6")
    p.add_argument("--vs", nargs="+", help="compare against these (default: every other method)")
    p.add_argument("--mode", default="wins", choices=["wins", "loses", "worst", "best"])
    p.add_argument("--metric", default="psnr", choices=sorted(HIGHER_IS_BETTER))
    p.add_argument("--by", default="slice", choices=["slice", "volume"],
                   help="rank single slices, or volumes by their mean over slices")
    p.add_argument("--first-n", type=int, metavar="N",
                   help="only the first N volumes in sorted order -- the ones "
                        "dump_eval dumps by default (mg_recon_dump: brain 60, "
                        "knee 40), so a CSV of every val volume ranks only what "
                        "the viewer can open")
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--csv", help="also write the full ranking here")
    p.add_argument("--pick", metavar="DIR",
                   help="write the volumes worth dumping to DIR/<anatomy>.txt "
                        "(+ DIR/<anatomy>.md, the slices to open) from --from-csv, "
                        "every anatomy and R in it, and stop")
    p.add_argument("-k", type=int, default=3,
                   help="--pick: distinct volumes per ranking per R (default 3)")
    a = p.parse_args(argv)
    if a.export:
        return export(a.dump_root, a.methods, a.export)
    if a.pick:
        if not a.from_csv:
            p.error("--pick needs --from-csv (e.g. results/main/slices.csv)")
        return pick(a.from_csv, a.pick, a.k, a.metric, a.methods)
    if a.anatomy is None or a.R is None:
        p.error("--anatomy and --R are required (except with --export / --pick)")

    if a.target not in a.methods:
        a.methods = a.methods + [a.target]
    others = a.vs or [m for m in a.methods if m != a.target]
    bad = [m for m in others if m not in a.methods]
    if bad:
        raise SystemExit(f"--vs {bad} not among --methods {a.methods}")

    if a.from_csv:
        data, kinds = load_csv(a.from_csv, a.anatomy, a.R, a.methods, a.metric)
        if all((s == -1).all() for vols in data.values() for s, _ in vols.values()):
            if a.by != "volume":
                print("(per-volume CSV: ranking whole volumes, --by volume)", file=sys.stderr)
            a.by = "volume"
    else:
        data, kinds = load(a.dump_root, a.anatomy, a.R, a.methods, a.metric)
    if len({k for ks in kinds.values() for k in ks}) > 1:
        print(f"WARNING: dumps were measured differently ({kinds}) -- not comparable",
              file=sys.stderr)
    if a.first_n:
        keep = set(sorted(set.union(*(set(d) for d in data.values())))[:a.first_n])
        data = {m: {v: x for v, x in d.items() if v in keep} for m, d in data.items()}
    rows = rows_for(data, a.methods)
    if not rows:
        raise SystemExit("no volume/slice is present for every method")
    if a.by == "volume":
        rows = by_volume(rows, a.methods)
    ranked = score(rows, a.target, others, a.metric, a.mode)

    n_vol = len({r["volume"] for r in rows})
    print(f"{a.anatomy} R{a.R}  {a.metric}  {a.mode}: {a.target} vs "
          f"{'best of ' if len(others) > 1 else ''}{', '.join(others)}  "
          f"({f'{len(rows)} slices, ' if a.by == 'slice' else ''}{n_vol} volumes; "
          f"margin > 0 = {a.target} better)\n")
    w = max(len(m) for m in a.methods)
    head = (f"{'#':>3}  {'volume':<28} {'sl':>3}  "
            + "  ".join(f"{m:>{max(w, 8)}}" for m in a.methods)
            + f"  {'margin':>8}  viewer filter")
    print(head)
    print("-" * len(head))
    fmt = "{:.2f}" if a.metric == "psnr" else "{:.4f}"
    for i, r in enumerate(ranked[:a.top], 1):
        sl = "" if r["slice"] is None else str(r["slice"])
        print(f"{i:>3}  {short(r['volume']):<28} {sl:>3}  "
              + "  ".join(f"{fmt.format(r[m]):>{max(w, 8)}}" for m in a.methods)
              + f"  {fmt.format(r['margin']):>8}  {r['volume'].split('_')[-1]} {sl}".rstrip())

    if a.csv:
        import csv
        cols = ["volume", "slice"] + a.methods + ["best_other", "margin"]
        with open(a.csv, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            wr.writeheader()
            wr.writerows(ranked)
        print(f"\nwrote {a.csv}")


if __name__ == "__main__":
    main()
