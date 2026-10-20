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
             limit=None, overwrite=False, val_kspace_type=None, mask="organ")
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


def test_recompute(tmp, ks, sm):
    """--recompute nrmse: one column re-scored, the rest kept, never mixed."""
    cfg = base_cfg(ks, sm)
    cfg["data"]["train"] = dict(cfg["data"]["val"], start_slice=1, end_slice=4)
    runs_root, out = os.path.join(tmp, "runs_rc"), os.path.join(tmp, "res_rc")
    make_run(runs_root, "knee", "lpdsnet_R4", cfg)
    run_dir = os.path.join(runs_root, "knee", "lpdsnet_R4")
    dev = torch.device("cpu")
    p = epv.eval_run(run_dir, "knee", args_for(runs_root, out), METRICS, dev)
    fresh = {(r["volume"], r["slice"]): r for r in read(p)}
    meta_p = os.path.join(out, "knee", "lpdsnet_R4.meta.json")
    check("a new run records each metric's definition",
          json.load(open(meta_p))["metric_versions"]["nrmse"] == "relative-l2")

    # Pretend these rows predate the NRMSE change: old version, wrong values,
    # and a PSNR sentinel that recompute must leave alone.
    meta = json.load(open(meta_p))
    meta["metric_versions"] = {"psnr": "v1", "ssim": "v1"}       # nrmse absent = v1
    json.dump(meta, open(meta_p, "w"))
    rows = read(p)
    for r in rows:
        r["nrmse"], r["psnr"] = "999", "123"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    try:
        epv.eval_run(run_dir, "knee", args_for(runs_root, out), METRICS, dev)
        check("a normal pass refuses to extend an old-definition file", False)
    except SystemExit as e:
        check("a normal pass refuses to extend an old-definition file",
              "--recompute nrmse" in str(e))
    epv.combine(out)
    v = read(os.path.join(out, "volumes.csv"))
    check("combine leaves old-definition values out (blank), keeps the rest",
          all(r["nrmse"] == "" and r["psnr"] == "123" for r in v))

    try:
        epv.recompute_run(run_dir, "knee", args_for(runs_root, out, sigma=0.2), ["nrmse"], dev)
        check("recompute refuses when the reconstructions would differ", False)
    except SystemExit as e:
        check("recompute refuses when the reconstructions would differ", "sigma" in str(e))

    epv.recompute_run(run_dir, "knee", args_for(runs_root, out, limit=4), ["nrmse"], dev)
    check("interrupted recompute leaves the run's file untouched",
          all(r["nrmse"] == "999" for r in read(p))
          and os.path.exists(os.path.join(out, "knee", "lpdsnet_R4.recompute-nrmse.csv")))
    epv.recompute_run(run_dir, "knee", args_for(runs_root, out), ["nrmse"], dev)
    done = read(p)
    check("recompute restores exactly the fresh values, every row once",
          len(done) == len(fresh)
          and all(r["nrmse"] == fresh[(r["volume"], r["slice"])]["nrmse"] for r in done))
    check("recompute keeps every other column as it was",
          all(r["psnr"] == "123" for r in done))
    check("recompute records the new definition and cleans up",
          json.load(open(meta_p))["metric_versions"]["nrmse"] == "relative-l2"
          and not os.path.exists(os.path.join(out, "knee", "lpdsnet_R4.recompute-nrmse.csv")))
    epv.eval_run(run_dir, "knee", args_for(runs_root, out), METRICS, dev)
    check("after recompute a normal pass resumes without complaint", len(read(p)) == len(fresh))


def test_val_kspace(tmp, ks, sm):
    """Validation measurement: simulated k-space, chosen per run, never mixed."""
    import inspect
    from evaluation import tasks
    from training.recon import train_recon

    check("train_recon takes val_kspace_type (config key reaches it)",
          "val_kspace_type" in inspect.signature(train_recon).parameters)

    seen = []
    real = tasks.prepare_measurement
    def spy(*a, **k):
        seen.append(k["kspace_type"])
        return real(*a, **k)
    tasks.prepare_measurement = spy
    try:
        cfg = base_cfg(ks, sm)
        batch = next(iter(epv.build_loader(dict(cfg["data"]["val"], num_workers=0),
                                           shuffle=False, drop_last=False)))
        tasks.recon(StubNet(), batch, cfg, torch.device("cpu"), 0.05)
        cfg2 = base_cfg(ks, sm)
        cfg2["mri"]["kspace_type"] = "measurement_awgn"
        cfg2["training"]["val_kspace_type"] = "simulated"
        tasks.recon(StubNet(), batch, cfg2, torch.device("cpu"), 0.05)
    finally:
        tasks.prepare_measurement = real
    check("no val key: the eval measurement is the training kspace_type",
          seen[0] == "simulated")
    check("training.val_kspace_type wins over the training kspace_type",
          seen[1] == "simulated" and tasks.val_kspace_type(cfg2) == "simulated")
    cfg3 = base_cfg(ks, sm)
    cfg3["mri"]["kspace_type"] = "measurement_awgn"
    check("an old measured run with no val key evaluates on measured k-space",
          tasks.val_kspace_type(cfg3) == "measurement_awgn")

    # eval_per_volume: rows from before the key existed count as the run's
    # training kspace_type, so they resume when that matches and start over when not.
    cfg = base_cfg(ks, sm)
    cfg["data"]["train"] = dict(cfg["data"]["val"], start_slice=1, end_slice=4)
    runs_root, out = os.path.join(tmp, "runs_vk"), os.path.join(tmp, "res_vk")
    run_dir = make_run(runs_root, "knee", "lpdsnet_R4", cfg)
    dev = torch.device("cpu")
    p = epv.eval_run(run_dir, "knee", args_for(runs_root, out, val_kspace_type="simulated"),
                     METRICS, dev)
    meta_p = os.path.join(out, "knee", "lpdsnet_R4.meta.json")
    check("the eval records how the measurement was made",
          json.load(open(meta_p))["val_kspace_type"] == "simulated")
    meta = json.load(open(meta_p))
    del meta["val_kspace_type"]                     # an older meta.json
    json.dump(meta, open(meta_p, "w"))
    rows = read(p)
    for r in rows:
        r["psnr"] = "123"                          # a sentinel a recompute would erase
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    epv.eval_run(run_dir, "knee", args_for(runs_root, out, val_kspace_type="simulated"),
                 METRICS, dev)
    check("old rows made on the same k-space resume (kept, not redone)",
          all(r["psnr"] == "123" for r in read(p)))
    meta = json.load(open(meta_p))
    del meta["val_kspace_type"]
    json.dump(meta, open(meta_p, "w"))
    cfg["mri"]["kspace_type"] = "measurement_awgn"  # they were MEASURED rows
    json.dump(cfg, open(os.path.join(run_dir, "config.json"), "w"))
    epv.eval_run(run_dir, "knee", args_for(runs_root, out, val_kspace_type="simulated"),
                 METRICS, dev)
    check("old rows made on measured k-space are redone for a simulated eval",
          all(r["psnr"] != "123" for r in read(p))
          and json.load(open(meta_p))["val_kspace_type"] == "simulated")


def test_force_mask(tmp, ks, sm):
    """Every run is scored inside the stored-map organ mask unless --mask run."""
    cfg = base_cfg(ks, sm)
    cfg["data"]["train"] = dict(cfg["data"]["val"], start_slice=1, end_slice=3)
    cfg["training"]["use_organ_mask"] = False            # trained unmasked
    cfg["data"]["val"]["organ_mask_source"] = "none"
    runs_root, dev = os.path.join(tmp, "runs_fm"), torch.device("cpu")
    run_dir = make_run(runs_root, "knee", "lpdsnet_R4", cfg)

    out = os.path.join(tmp, "res_fm_organ")
    epv.eval_run(run_dir, "knee", args_for(runs_root, out), METRICS, dev)
    meta = json.load(open(os.path.join(out, "knee", "lpdsnet_R4.meta.json")))
    check("default: an unmasked run is scored inside the smaps organ mask",
          meta["use_organ_mask"] is True and meta["mask_source"] == "smaps", str(meta))

    out2 = os.path.join(tmp, "res_fm_run")
    epv.eval_run(run_dir, "knee", args_for(runs_root, out2, mask="run"), METRICS, dev)
    meta2 = json.load(open(os.path.join(out2, "knee", "lpdsnet_R4.meta.json")))
    check("--mask run keeps the run's own setting",
          meta2["use_organ_mask"] is False and meta2["mask_source"] == "none")
    try:
        epv.recompute_run(run_dir, "knee", args_for(runs_root, out2), ["nrmse"], dev)
        check("recompute refuses rows scored on another mask", False)
    except SystemExit as e:
        check("recompute refuses rows scored on another mask",
              "use_organ_mask" in str(e) or "mask_source" in str(e), str(e)[:90])


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
        test_recompute(tmp, ks, sm)
        test_val_kspace(tmp, ks, sm)
        test_force_mask(tmp, ks, sm)
        test_wilcoxon(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
