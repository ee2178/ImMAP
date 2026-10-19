"""
Bookkeeping tests for scripts/eval_per_volume.py and scripts/wilcoxon_main.py.

Run with:  python -m tests.test_eval_per_volume

Synthetic fastMRI trees and the adjoint stub from tests/test_dump_eval.py; the
network itself is never built (`load_run` is replaced), because what is under
test is which slices get scored, under what noise, and how they are merged and
paired -- not the reconstruction.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch

from tests.test_dump_eval import SLICES, StubNet, base_cfg, build_trees, check, PASS, FAIL

SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "scripts")
sys.path.insert(0, SCRIPTS)
import dump_eval as de                                          # noqa: E402
import eval_per_volume as epv                                   # noqa: E402
import wilcoxon_main as wm                                      # noqa: E402

METRICS = ["psnr", "ssim", "nrmse"]


def make_run(root, anatomy, run, cfg):
    d = os.path.join(root, anatomy, run)
    os.makedirs(d)
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(cfg, f)
    open(os.path.join(d, "net.ckpt"), "w").close()
    return d


def args_for(runs_root, out, **kw):
    a = dict(runs_root=runs_root, out=out, metrics=METRICS, slices=None,
             volumes=None, n_volumes=None, sigma=None, seed=1234, workers=0,
             limit=None, overwrite=False)
    a.update(kw)
    return argparse.Namespace(**a)


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_eval(tmp, ks, sm):
    cfg = base_cfg(ks, sm)
    cfg["data"]["train"] = dict(cfg["data"]["val"], start_slice=1, end_slice=4)
    runs_root, out = os.path.join(tmp, "runs"), os.path.join(tmp, "res")
    for run in ("lpdsnet_R4", "mg81v6_R4"):
        make_run(runs_root, "knee", run, cfg)
    de.load_run = lambda run_dir, device: (json.load(open(
        os.path.join(run_dir, "config.json"))), StubNet(), 1, 10)

    dev = torch.device("cpu")
    p = epv.eval_run(os.path.join(runs_root, "knee", "lpdsnet_R4"), "knee",
                     args_for(runs_root, out), METRICS, dev)
    rows = read(p)
    want = {(v[:-3], s) for v in SLICES for s in range(1, 4)}
    check("default slices = the run's TRAINING range, every val volume",
          {(r["volume"], int(r["slice"])) for r in rows} == want and len(rows) == len(want),
          f"{len(rows)} rows")
    full = {(r["volume"], r["slice"]): r for r in rows}

    # An interrupted job: only part of the set, then resume.
    out2 = os.path.join(tmp, "res2")
    epv.eval_run(os.path.join(runs_root, "knee", "lpdsnet_R4"), "knee",
                 args_for(runs_root, out2, limit=4), METRICS, dev)
    part = read(os.path.join(out2, "knee", "lpdsnet_R4.slices.csv"))
    epv.eval_run(os.path.join(runs_root, "knee", "lpdsnet_R4"), "knee",
                 args_for(runs_root, out2), METRICS, dev)
    resumed = read(os.path.join(out2, "knee", "lpdsnet_R4.slices.csv"))
    check("interrupted at 4 slices", len(part) == 4)
    check("resume completes every slice exactly once",
          sorted((r["volume"], r["slice"]) for r in resumed) == sorted(full))
    check("resumed values == uninterrupted values (noise is seeded per slice)",
          all(resumed_r[m] == full[(resumed_r["volume"], resumed_r["slice"])][m]
              for resumed_r in resumed for m in METRICS))

    # A settings change must not mix rows: a new sigma starts over.
    epv.eval_run(os.path.join(runs_root, "knee", "lpdsnet_R4"), "knee",
                 args_for(runs_root, out2, sigma=0.2), METRICS, dev)
    redo = read(os.path.join(out2, "knee", "lpdsnet_R4.slices.csv"))
    check("changed sigma recomputes instead of resuming",
          len(redo) == len(want)
          and any(r["psnr"] != full[(r["volume"], r["slice"])]["psnr"] for r in redo))

    # Different volume subsets: per-slice seeding gives the same noise anyway.
    epv.eval_run(os.path.join(runs_root, "knee", "mg81v6_R4"), "knee",
                 args_for(runs_root, out, volumes=["volC"]), METRICS, dev)
    sub = read(os.path.join(out, "knee", "mg81v6_R4.slices.csv"))
    check("same slice, same noise, whatever else the run was given",
          all(r["psnr"] == full[(r["volume"], r["slice"])]["psnr"] for r in sub)
          and {r["volume"] for r in sub} == {"volC"})

    epv.combine(out)
    vols = read(os.path.join(out, "volumes.csv"))
    a = [r for r in vols if r["run"] == "lpdsnet_R4" and r["volume"] == "volA"][0]
    mean = np.mean([float(r["psnr"]) for r in rows if r["volume"] == "volA"])
    check("volume value = mean of its slices",
          abs(float(a["psnr"]) - mean) < 1e-4 and a["n_slices"] == "3"
          and a["net"] == "LPDSNet" and a["R"] == "4", f"{a['psnr']} vs {mean}")
    check("labels: mg81v6 -> MGLPDS-Small, brain mglpds -> MGLPDS-Big",
          {r["net"] for r in vols} == {"LPDSNet", "MGLPDS-Small"}
          and epv.label_of("mglpds_R12") == "MGLPDS-Big"
          and epv.label_of("mg169v6_R8") == "MGLPDS-Big")


def test_wilcoxon(tmp):
    rng = np.random.default_rng(1)
    n = 30
    path = os.path.join(tmp, "volumes.csv")
    base = rng.normal(30, 2, n)
    nets = {"E2E-VarNet": base, "LPDSNet": base + rng.normal(0, .3, n),
            "MGLPDS-Big": base + 0.5 + rng.normal(0, .1, n)}
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["anatomy", "R", "net", "run", "volume",
                                          "n_slices", "psnr", "nrmse"])
        w.writeheader()
        for net, v in nets.items():
            for i in range(n):
                w.writerow(dict(anatomy="knee", R=8, net=net, run=net, volume=f"v{i}",
                                n_slices=13, psnr=v[i], nrmse=1 / v[i]))
        # A volume only one net has: must not enter any pair.
        w.writerow(dict(anatomy="knee", R=8, net="MGLPDS-Big", run="x", volume="extra",
                        n_slices=13, psnr=99, nrmse=0.01))
    outp = os.path.join(tmp, "w.csv")
    r = subprocess.run([sys.executable, os.path.join(SCRIPTS, "wilcoxon_main.py"),
                        "--in", path, "--out", outp, "--ref", "MGLPDS-Big"],
                       capture_output=True, text=True)
    check("wilcoxon_main runs", r.returncode == 0, r.stderr[-400:])
    res = {(x["metric"], x["B"]): x for x in read(outp)}
    big_vs_var = res[("psnr", "E2E-VarNet")]
    check("pairs only common volumes", big_vs_var["n"] == str(n))
    check("A better on every volume -> A_better 1, r_rb +1, tiny p",
          float(big_vs_var["A_better"]) == 1.0 and float(big_vs_var["r_rb"]) == 1.0
          and float(big_vs_var["p"]) < 1e-5)
    nr = res[("nrmse", "E2E-VarNet")]
    check("lower-is-better metric: sign flips so + still means A better",
          float(nr["A_better"]) == 1.0 and float(nr["r_rb"]) == 1.0
          and float(nr["median_diff"]) < 0)
    check("holm: adjusted >= raw, family of 2",
          all(float(x["p_holm"]) >= float(x["p"]) for x in res.values()))
    check("holm matches the textbook step-down",
          np.allclose(wm.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06]))


def main():
    tmp = tempfile.mkdtemp()
    try:
        ks, sm = build_trees(tmp)
        test_eval(tmp, ks, sm)
        test_wilcoxon(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
