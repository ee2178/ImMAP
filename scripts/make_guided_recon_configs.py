#!/usr/bin/env python3
"""
Configs for GUIDED reconstruction on fastMRI knee: PDFS main contrast, fully-sampled PD guide.

Set 1 of the guided experiments (notes/guided_recon_tgvn.md): TGVN and LGGS, plus TGVN's matched
no-side-information baseline, trained under the SAME settings as the one-shot recon grid -- not
TGVN's own recipe -- so the comparison is like for like at the grid's compute.

What is inherited, from what scripts/make_mg_recon_configs.py writes TODAY for knee at that R:
    mri          the measured protocol: measured k-space + AWGN, online ESPIRiT, center_frac,
                 adjust_accel, uniform mask (taken from the lpdsnet cell)
    training     steps, epochs, val cadence, noise range, val noise / seed, loss, clip
    optimizer    Adam at the grid LR; LGGS (group attention) gets the grid's group LR, from the
    scheduler    mggrouplpds cell; the others the plain one, from the varnet cell
    data         scale_fac, slice ranges, organ_mask_source

What is set here:
    data.name    fastmri_paired: TGVN's patient-level PD/PDFS splits
                 (config/knee/guided/splits/), raw k-space from fastMRI train AND val
    target       RSS of the PDFS k-space (no offline maps exist for PDFS; TGVN's target too)
    guide        the FULLY-SAMPLED PD scan. Side undersampling is a later study.
    models       varnet_img  TGVN(guided=False): upstream image-domain E2E-VarNet, same size
                 tgvn        TGVN_1S at upstream defaults (12 cascades, 18 chans)
                 lggs        LGGSNet at the published LGGS point (models/guided_lpds.py)

    python scripts/make_guided_recon_configs.py                     # 3 models at R=8
    python scripts/make_guided_recon_configs.py --accels 4 --models tgvn
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
SPLITS = "config/knee/guided/splits"
KSPACE_ROOTS = ["../datasets/fastmri/knee/multicoil_train",
                "../datasets/fastmri/knee/multicoil_val"]

TGVN_SIZE = dict(num_cascades=12, sens_chans=8, sens_pools=4, chans=18, pools=4)
MODELS = {
    "varnet_img": dict(type="TGVN", params=dict(guided=False, **TGVN_SIZE),
                       grid_base="varnet", pad_multiple=1),
    "tgvn": dict(type="TGVN", params=dict(guided=True, **TGVN_SIZE),
                 grid_base="varnet", pad_multiple=1),
    "lggs": dict(type="LGGS", params=None, grid_base="mggrouplpds", pad_multiple=2),
}


# The published LGGS point, models/guided_lpds.py::lggs_defaults(), with the grid's attention
# backend. Written out rather than imported so this script needs no torch model stack;
# tests/test_guided_recon.py pins it to lggs_defaults().
LGGS_PARAMS = dict(K=30, M=169, C=1, P=7, s=2, Mh=64, window=1, guide_window=15,
                   joint_softmax=True, sim_fun="pidistance", lam0=1e-3, tau0=0.5, theta0=0.0,
                   dK=5, nheads=1, init_strategy="semi_orthogonal", is_complex=True,
                   preproc="kspace", attn_backend="flex")


def _lggs_params():
    return dict(LGGS_PARAMS)


def grid_cells(r):
    """{grid model tag: config} for the mg grid's knee cells used as templates, at R."""
    tags = sorted({m["grid_base"] for m in MODELS.values()} | {"lpdsnet"})
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
            "pad_multiple": pad_multiple,
            "num_workers": 8}


def make_config(r, model, grid):
    spec = MODELS[model]
    base = grid[spec["grid_base"]]
    params = dict(spec["params"]) if spec["params"] is not None else _lggs_params()
    training = {k: v for k, v in base["training"].items() if k != "log_levels"}
    name = f"{model}_R{r}"
    return {
        "task": "recon",
        "experiment": {"name": f"guided_{ANATOMY}_PDFS_from_PD_{name}"},
        "model": {"type": spec["type"], "params": params},
        "paths": {"save_dir": f"trained_nets/guided_recon/{ANATOMY}/{name}", "ckpt": None},
        "data": {"train": _data("train", base["data"]["train"], spec["pad_multiple"]),
                 "val": _data("val", base["data"]["val"], spec["pad_multiple"])},
        "training": training,
        "mri": grid["lpdsnet"]["mri"],
        "optimizer": base["optimizer"],
        "scheduler": base["scheduler"],
        "wandb": {"project": "guided_recon", "id": None},
    }


def cells(accels=ACCELS, models=tuple(MODELS)):
    return [(r, m) for r in accels for m in models]


def config_path(root, r, model):
    return os.path.join(root, ANATOMY, "guided", f"{model}_R{r}.json")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="config")
    p.add_argument("--accels", nargs="*", type=int, default=list(ACCELS))
    p.add_argument("--models", nargs="*", choices=list(MODELS), default=list(MODELS))
    p.add_argument("--list-cells", action="store_true",
                   help="print 'index<TAB>R<TAB>model' and exit")
    args = p.parse_args()

    todo = cells(args.accels, args.models)
    if args.list_cells:
        for i, (r, m) in enumerate(todo):
            print(f"{i}\tR{r}\t{m}")
        return
    for r in sorted({r for r, _ in todo}):
        grid = grid_cells(r)
        for rr, m in todo:
            if rr != r:
                continue
            cfg = make_config(r, m, grid)
            path = config_path(args.out, r, m)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", newline="\n") as f:
                json.dump(cfg, f, indent=4)
                f.write("\n")
            print(f"{path}: {cfg['model']['type']} {cfg['mri']['kspace_type']} R={r} "
                  f"lr={cfg['optimizer']['params']['lr']}")


if __name__ == "__main__":
    main()
