#!/usr/bin/env python3
"""
Configs for GUIDED reconstruction on fastMRI knee: PDFS main contrast, PD guide (TGVN's setup).

Set 1 of notes/guided_recon_tgvn.md, trained under the SAME settings as the one-shot recon grid --
not TGVN's own recipe -- so the comparison is like for like at the grid's compute. A cell is
(main R, model, guide R); guide R = 1 is a fully-sampled guide, and unguided models get only that
cell. The default cells are the first five experiments:

    tgvn_R8_g1     TGVN, fully-sampled guide                 task recon
    tgvn_R8_g2     TGVN, 2x guide                            task recon
    sbtgvn_R8_g1   i2sb-TGVN (BridgeTGVN), fully-sampled     task i2sb_recon
    sbtgvn_R8_g2   i2sb-TGVN, 2x guide                       task i2sb_recon
    lpdsnet_R8     LPDSNet, no guide (on the same PDFS data) task recon

Optional models: varnet_img (TGVN's matched no-guide E2E-VarNet), lggs (LGGSNet), and the TGVN
depth sweep tgvn_c10 / tgvn_c8 / tgvn_c6 (E2E-VarNet width, 10 / 8 / 6 cascades):

    python scripts/make_guided_recon_configs.py --models tgvn_c10 tgvn_c8 tgvn_c6 lggs         --guide-accels 1

What is inherited, from what scripts/make_mg_recon_configs.py writes TODAY for knee at that R:
    mri          the measured protocol: measured k-space + AWGN, online ESPIRiT, center_frac,
                 adjust_accel, uniform mask (taken from the lpdsnet cell)
    training     steps, epochs, val cadence, noise range, val noise / seed, loss, clip
    optimizer    Adam at the grid LR (LGGS: the grid's group-attention LR, from mggrouplpds)
    scheduler    cosine over the grid's step count
    data         scale_fac, slice ranges, organ_mask_source
    lpdsnet      its whole model block, from the grid's lpdsnet cell

What is set here:
    data.name    fastmri_paired: TGVN's patient-level PD/PDFS splits
                 (config/knee/guided/splits/), raw k-space from fastMRI train AND val
    target       the OFFLINE SENSE image of the PDFS volume (ESPIRiT, ACS 32, from the full
                 k-space; PDFS_MAPS, written by torch/espirit_knee_pairs.sbatch) -- not TGVN's RSS.
                 Every model is scored on |.| against it; i2sb-TGVN still bridges to the
                 combination under the operator's own (online) maps, x0_source "operator",
                 since the offline maps' phase reference is not the operator's.
    guide        mri.guide_R: the PD scan undersampled by equispaced lines with no centre
                 (TGVN's side mask), redrawn per training step; 1 = fully sampled
    TGVN         upstream defaults (12 cascades, 18 chans) + a noise-level plane in each
                 regularizer (cond_sigma; upstream TGVN takes no noise level), on the
                 OPERATOR'S online ESPIRiT maps (use_smaps) instead of its own SensitivityModel:
                 the same maps i2sb-TGVN and LPDSNet get, so the comparison differs in the
                 method, not in who estimates the coil maps (decided 2026-11-05; at the reduced
                 budget, not learning the maps also makes TGVN's job easier)
    i2sb-TGVN    BridgeTGVN: the same TGVN, on the operator's (online ESPIRiT) maps, started at
                 x_t, conditioned on the bridge position and sigma_n; complex-MSE loss (the
                 bridge state is complex; the grid's magnitude NL1+NL2 cannot train it). Its
                 tau is TAU below, intensity-relative per slice (TAU_SCALE / TAU_REF);
                 calibrate with scripts/calibrate_recon_bridge.py.

    python scripts/make_guided_recon_configs.py                  # the five default cells
    python scripts/make_guided_recon_configs.py --guide-accels 1 2 4 --models tgvn
    python scripts/make_guided_recon_configs.py --list-cells
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

ANATOMY = "knee"
ACCELS = (8,)
GUIDE_ACCELS = (1, 2)
DEFAULT_MODELS = ("tgvn", "sbtgvn", "lpdsnet")
SPLITS = "config/knee/guided/splits"
KSPACE_ROOTS = ["../datasets/fastmri/knee/multicoil_train",
                "../datasets/fastmri/knee/multicoil_val"]
# Offline ESPIRiT maps + SENSE images of the PDFS (target) volumes: torch/espirit_knee_pairs.sbatch
# at thresh_eig 0.9 (the 0.95 maps in .../pdfs_espirit_acs32 cut into the anatomy).
PDFS_MAPS = "../datasets/fastmri_preprocessed/knee_coil_combined/pdfs_espirit_acs32_eig0.9"
TARGET = "sense"

# Brownian-bridge temperature for i2sb-TGVN, tau = RMS(x0 - x1)/2 inside the object at the main
# R (written to BOTH i2sb and model.params). Measured 2026-11-05 with
# scripts/calibrate_recon_bridge.py on sbtgvn_R8_g1 (knee PDFS, R=8, measured protocol, 60 train
# slices): RMS 0.0469 -> tau 0.0234; per-slice median 0.043, range [0.027, 0.125]; top-10% tail
# 0.121 -> tau 0.060 (ceiling). Re-measure for another main R. Set to 0.03, a little above the
# gap rule, on the user's call: the measurement noise is real even where coil combination hides
# it in the image, and a slightly wider ladder is the safer side.
TAU = 0.03
# sampling-time CG data consistency (I2SB-Recon, arXiv 2411.14269; sb/i2sb_recon.py): the
# paper's defaults. Validation only -- val/* is WITH it, val/nodc_* the plain sampler on the same
# noise draws (val cost x2). dc_iters 0 turns it off.
I2SB_DC = {"dc_iters": 7, "dc_lam": 7e-5}
# INTENSITY-RELATIVE bridge: slice i runs at tau_i = TAU * s_i / TAU_REF, s_i the RMS of its
# zero-filled image over its Otsu foreground (sb/i2sb_recon.py::bridge_noise_scale) -- slices of
# one volume differ several-fold in intensity, and one absolute tau is near-silent on the bright
# ones. TAU_REF is the MEDIAN s over training slices, so TAU stays the tau of a typical slice.
# Measured 2026-11-05 (calibrate_recon_bridge.py, sbtgvn_R8_g1, 60 train slices): median s =
# 0.1481, range [0.082, 0.306]; the gap rule in these units gives tau 0.0212 (absolute: 0.0234).
# The per-slice GAP spread barely moves under the scaling (std/mean 0.45 -> 0.44): intensity sets
# how visible the noise is, but not how large the aliasing gap is. Re-measure for another main R.
TAU_SCALE = "x1_rms"
TAU_REF = 0.1481
SCHEDULE = dict(kind="brownian", tau=TAU, n_points=1000, beta_max=0.3)

TGVN_SIZE = dict(num_cascades=12, sens_chans=8, sens_pools=4, chans=18, pools=4)

# The published LGGS point, models/guided_lpds.py::lggs_defaults(), with the grid's attention
# backend. Written out rather than imported so this script needs no torch model stack;
# tests/test_guided_recon.py pins it to lggs_defaults().
LGGS_PARAMS = dict(K=30, M=169, C=1, P=7, s=2, Mh=64, window=1, guide_window=15,
                   joint_softmax=True, sim_fun="pidistance", lam0=1e-3, tau0=0.5, theta0=0.0,
                   dK=5, nheads=1, init_strategy="semi_orthogonal", is_complex=True,
                   preproc="kspace", attn_backend="flex")

MODELS = {
    "tgvn": dict(task="recon", guided=True, grid_base="varnet", pad_multiple=1,
                 model=dict(type="TGVN", params=dict(guided=True, cond_sigma=True,
                                                     use_smaps=True, **TGVN_SIZE))),
    "sbtgvn": dict(task="i2sb_recon", guided=True, grid_base="varnet", pad_multiple=1,
                   model=dict(type="BridgeTGVN", params=dict(guided=True, cond_sigma=True,
                                                             **TGVN_SIZE, **SCHEDULE))),
    "lpdsnet": dict(task="recon", guided=False, grid_base="lpdsnet", pad_multiple=None,
                    model=None),                       # the grid's lpdsnet model block
    "varnet_img": dict(task="recon", guided=False, grid_base="varnet", pad_multiple=1,
                       model=dict(type="TGVN", params=dict(guided=False, cond_sigma=True,
                                                           use_smaps=True, **TGVN_SIZE))),
    "lggs": dict(task="recon", guided=True, grid_base="mggrouplpds", pad_multiple=2,
                 model=dict(type="LGGS", params=LGGS_PARAMS)),
}

# Depth sweep for TGVN under the reduced budget: width held at E2E-VarNet's (chans 18,
# sens_chans 8 -- the grid's varnet cell), cascades cut. Per forward at knee R8 with a
# fully-sampled guide, one cascade costs ~64 GFLOP (two 18-chan U-Nets + the CG trust projector),
# so 10 / 8 / 6 cascades are ~640 / 511 / 383 GFLOP against LGGS's ~677 (12: ~768).
TGVN_CASCADE_SWEEP = (10, 8, 6)
for _n in TGVN_CASCADE_SWEEP:
    MODELS[f"tgvn_c{_n}"] = dict(
        task="recon", guided=True, grid_base="varnet", pad_multiple=1,
        model=dict(type="TGVN", params=dict(guided=True, cond_sigma=True, use_smaps=True,
                                            **dict(TGVN_SIZE, num_cascades=_n))))

# PSNR / SSIM against each slice's own peak |gt| (training/metrics.py). The paired loader only
# multiplies by a fixed scale_fac, so slice peaks sit well below 1 and vary across a volume:
# the default data_range=1 inflated PSNR by -20 log10(peak) dB, most on the dimmest slices.
METRIC_RANGE = "gt_max"


def grid_cells(r, tags):
    """{grid model tag: config} for the mg grid's knee cells used as templates, at R."""
    tags = sorted(set(tags) | {"lpdsnet"})
    with tempfile.TemporaryDirectory() as d:
        subprocess.run([sys.executable, os.path.join(REPO, "scripts", "make_mg_recon_configs.py"),
                        "--out", d, "--anatomy", ANATOMY, "--accels", str(r), "--only", *tags],
                       check=True, cwd=REPO, stdout=subprocess.DEVNULL)
        out = {}
        for t in tags:
            with open(os.path.join(d, ANATOMY, "mg", f"{t}_R{r}.json")) as f:
                out[t] = json.load(f)
        return out


def _data(split, base_split, pad_multiple):
    keep = ("batch_size", "start_slice", "end_slice", "scale_fac", "organ_mask_source")
    return {"name": "fastmri_paired",
            "split_csv": f"{SPLITS}/{split}.csv",
            "kspace_roots": list(KSPACE_ROOTS),
            **{k: base_split[k] for k in keep if k in base_split},
            "target": TARGET,
            "smap_root": PDFS_MAPS,
            "pad_multiple": pad_multiple,
            "num_workers": 8}


def cell_name(r, model, g):
    return f"{model}_R{r}" + (f"_g{g}" if MODELS[model]["guided"] else "")


def make_config(r, model, g, grid):
    spec = MODELS[model]
    base = grid[spec["grid_base"]]
    model_block = spec["model"] if spec["model"] is not None else base["model"]
    pad = (spec["pad_multiple"] if spec["pad_multiple"] is not None
           else base["data"]["train"]["pad_multiple"])
    training = {k: v for k, v in base["training"].items() if k != "log_levels"}
    training["metric_range"] = METRIC_RANGE
    mri = dict(grid["lpdsnet"]["mri"], guide_R=int(g), guide_acs_lines=0,
               guide_mask_dist="uniform")
    name = cell_name(r, model, g)
    cfg = {
        "task": spec["task"],
        "experiment": {"name": f"guided_{ANATOMY}_PDFS_from_PD_{name}"},
        "model": json.loads(json.dumps(model_block)),
        "paths": {"save_dir": f"trained_nets/guided_recon/{ANATOMY}/{name}", "ckpt": None},
        "data": {"train": _data("train", base["data"]["train"], pad),
                 "val": _data("val", base["data"]["val"], pad)},
        "training": training,
        "mri": mri,
        "optimizer": base["optimizer"],
        "scheduler": base["scheduler"],
        "wandb": {"project": "guided_recon", "id": None},
    }
    if spec["task"] == "i2sb_recon":
        cfg["training"]["loss_type"] = "complex-mse"
        cfg["i2sb"] = {**SCHEDULE, "deterministic": False, "posterior": "ddpm",
                       "x1_source": "adjoint", "x0_source": "operator",
                       "tau_scale": TAU_SCALE, "tau_ref": TAU_REF,
                       "val_mode": "full_recon", "val_nfe": 20, **I2SB_DC}
    return cfg


def cells(accels=ACCELS, models=DEFAULT_MODELS, guide_accels=GUIDE_ACCELS):
    out = []
    for r in accels:
        for m in models:
            for g in (guide_accels if MODELS[m]["guided"] else (1,)):
                if (r, m, g) not in out:
                    out.append((r, m, g))
    return out


def config_path(root, r, model, g):
    return os.path.join(root, ANATOMY, "guided", f"{cell_name(r, model, g)}.json")


def _write_config():
    """training/config_io.py::write_config, loaded BY PATH (as make_mg_recon_configs.py does):
    it spells floats so yaml.safe_load -- how train.py reads configs -- gets numbers back
    (json.dump's `1e-06` loads as the STRING '1e-06'), and refuses a config that would not
    round-trip. By path, because `import training...` would run training/__init__.py
    (torchvision)."""
    import importlib.util
    p = os.path.join(REPO, "training", "config_io.py")
    spec = importlib.util.spec_from_file_location("_immap_config_io", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.write_config


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="config")
    p.add_argument("--accels", nargs="*", type=int, default=list(ACCELS))
    p.add_argument("--models", nargs="*", choices=list(MODELS), default=list(DEFAULT_MODELS))
    p.add_argument("--guide-accels", nargs="*", type=int, default=list(GUIDE_ACCELS),
                   help="guide (PD) accelerations for the guided models; 1 = fully sampled")
    p.add_argument("--list-cells", action="store_true",
                   help="print 'index<TAB>R<TAB>model<TAB>guide R<TAB>name' and exit")
    args = p.parse_args()
    write_config = _write_config()

    todo = cells(args.accels, args.models, args.guide_accels)
    if args.list_cells:
        for i, (r, m, g) in enumerate(todo):
            print(f"{i}\tR{r}\t{m}\tg{g}\t{cell_name(r, m, g)}")
        return
    for r in sorted({c[0] for c in todo}):
        grid = grid_cells(r, [MODELS[m]["grid_base"] for _, m, _ in todo])
        for rr, m, g in todo:
            if rr != r:
                continue
            cfg = make_config(r, m, g, grid)
            path = config_path(args.out, r, m, g)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            write_config(cfg, path)
            print(f"{path}: {cfg['task']} {cfg['model']['type']} R={r} guide_R={g} "
                  f"lr={cfg['optimizer']['params']['lr']}")


if __name__ == "__main__":
    main()
