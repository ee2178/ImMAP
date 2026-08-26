"""Source for notebooks/compare_mg_recon.ipynb -- run this file to (re)build it.

Kept as a .py so the notebook's code is reviewable and diffable; the .ipynb is
the artefact. Cells are separated by the CELL marker.
"""
import json
import os

MD = "markdown"
CODE = "code"

CELLS = [
(MD, """# Compare the multigrid recon nets on one slice

Loads every trained run under `RUNS_GLOB`, pushes **the same** measurement
through each, and plots outputs over residuals.

Everything except the network is shared: one slice, one sampling mask, one
noise realisation. That is the whole point -- any difference you see is the
network.

Edit the config cell, then Run All."""),

(CODE, '''import glob, json, os

import matplotlib.pyplot as plt
import numpy as np
import torch
%matplotlib inline

# os.chdir('/scratch/ee2178/ImMAP')   # <-- EDIT to your repo root if needed
                                      #     (ckpt paths in the configs are relative to it)

import datasets                                   # registers the loaders
from datasets.registry import build_loader
from models import build_model
from operators import FFT2D, Mask, Sense
from physics.mask import get_mask_cached as get_mask
from training.common import prepare_measurement
from evaluation.metrics import REGISTRY as METRICS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)'''),

(MD, "## Config"),

(CODE, '''# One acceleration at a time: R lives in each run's config and changes the mask,
# so mixing R=4 and R=8 here would put incomparable images side by side. The
# cell below checks for that rather than trusting the glob.
RUNS_GLOB = "trained_nets/mg_recon/knee/*_R8"

SLICE  = 0        # which validation batch to show
SIGMA  = None     # None -> each run's own val_noise_std (what training validated at)
GAIN   = 4.0      # residual amplification; the panels say what it is
SEED   = 1234     # fixes the noise realisation'''),

(CODE, '''runs = sorted(d for d in glob.glob(RUNS_GLOB)
              if os.path.exists(os.path.join(d, "config.json"))
              and os.path.exists(os.path.join(d, "net.ckpt")))
assert runs, f"no runs with config.json + net.ckpt matching {RUNS_GLOB}"

cfgs = {d: json.load(open(os.path.join(d, "config.json"))) for d in runs}

Rs = {c["mri"]["R"] for c in cfgs.values()}
assert len(Rs) == 1, (
    f"runs span accelerations {sorted(Rs)} -- they would get different masks and "
    f"the comparison would be meaningless. Narrow RUNS_GLOB to one R.")

for d, c in cfgs.items():
    print(f"{os.path.basename(d):<16} {c['experiment']['name']:<34} R={c['mri']['R']}")'''),

(MD, """## Build one measurement, shared by every net

`prepare_measurement` and the encoding operator are built exactly as
`training/recon.py` builds them, from the first run's config -- so the input the
nets see here is the input they saw at validation."""),

(CODE, '''ref = cfgs[runs[0]]                      # the mri/data block is shared (R is asserted equal)
sigma_val = SIGMA if SIGMA is not None else ref["training"].get("val_noise_std", 0.0)

loader = build_loader(ref["data"]["val"], shuffle=False, drop_last=False)
batch = next(b for i, b in enumerate(loader) if i == SLICE)
kspace, smaps, image, _, _pad_hw = (b.to(device) for b in batch)

mri = ref["mri"]
mask = get_mask(image, R=mri["R"], acs_lines=mri["acs_lines"],
                mode=mri.get("mask_dist", "uniform"), offset=mri.get("mask_offset", 0))

gen = torch.Generator(device=device).manual_seed(SEED)
y, sigma_n, extra = prepare_measurement(
    image=image, kspace=kspace, mask=mask, smaps=smaps,
    kspace_type=mri.get("kspace_type", "simulated"),
    noise_std=sigma_val, noise_dist=ref["training"].get("noise_dist", "uniform"),
    whiten_kspace=mri.get("whiten_kspace", False), generator=gen)

E = Mask(mask) @ FFT2D() @ Sense(extra["smaps"])
zero_filled = E.adjoint(y)               # the artefact BEFORE any network

print(f"slice {SLICE}  {tuple(image.shape[-2:])}  R={mri['R']}  sigma={float(sigma_val):.4g}")'''),

(MD, "## Run every net on it"),

(CODE, '''@torch.no_grad()
def run(run_dir):
    cfg = cfgs[run_dir]
    net = build_model(cfg).to(device)
    ckpt = torch.load(os.path.join(run_dir, "net.ckpt"), map_location=device,
                      weights_only=False)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()
    out, _ = net(y, E=E, sigma=sigma_n)
    if mri.get("whiten_kspace", False) and "Zinv" in extra:
        out = extra["Zinv"] * out
    del net
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


# Ground truth and zero-filled first, then one column per net.
panels = {"ground truth": image, f"zero-filled (R={mri['R']})": zero_filled}
for d in runs:
    panels[cfgs[d]["experiment"]["name"].replace("_synth", "")] = run(d)

mag = {k: v[0, 0].abs().float().cpu().numpy() for k, v in panels.items()}
gt = mag["ground truth"]
vmax = gt.max()                          # ONE intensity scale for every panel

def score(x):
    a = torch.as_tensor(x)[None, None]
    b = torch.as_tensor(gt)[None, None]
    return (float(METRICS["psnr"]["fn"](b, a)[0]),
            float(METRICS["ssim"]["fn"](b, a)[0]))

for k in panels:
    p, s = score(mag[k])
    print(f"{k:<34} PSNR {p:6.2f} dB   SSIM {s:.4f}")'''),

(MD, """## Outputs over residuals

One `subplots` grid: row 0 is what each method produced, row 1 is
`|output - ground truth|` at the same amplification. The image row shares one
colour scale (the ground truth's max), so brightness is comparable across
columns; the residual row shares its own."""),

(CODE, '''keys = list(panels)
n = len(keys)
fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 5.6), constrained_layout=True)
axes = np.atleast_2d(axes)

for j, k in enumerate(keys):
    res = np.abs(mag[k] - gt)
    p, s = score(mag[k])

    axes[0, j].imshow(mag[k], cmap="gray", vmin=0, vmax=vmax)
    title = k if k == "ground truth" else f"{k}\\n{p:.2f} dB / {s:.3f}"
    axes[0, j].set_title(title, fontsize=8)

    axes[1, j].imshow(res, cmap="inferno", vmin=0, vmax=vmax / GAIN)
    axes[1, j].set_title(f"residual  rms {100 * res.std() / vmax:.2f}%", fontsize=8)

    for i in (0, 1):
        axes[i, j].set_xticks([]); axes[i, j].set_yticks([])

axes[0, 0].set_ylabel("output", fontsize=9)
axes[1, 0].set_ylabel(f"|residual| x{GAIN:g}", fontsize=9)
fig.suptitle(f"slice {SLICE}   R={mri['R']}   sigma={float(sigma_val):.4g}", fontsize=10)
plt.show()'''),

(MD, """## Residuals alone

Bigger, and with the amplification under your control -- turn `GAIN` up until
the structure is legible. Regular stripes are unrecovered undersampling;
grid-aligned blocking at a coarse level's scale is a multigrid transfer
artefact (see `docs/multigrid_port.md`)."""),

(CODE, '''RES_GAIN = 8.0                           # independent of the GAIN above

keys = [k for k in panels if k != "ground truth"]
n = len(keys)
fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.6), constrained_layout=True)
axes = np.atleast_1d(axes)

for j, k in enumerate(keys):
    res = np.abs(mag[k] - gt)
    im = axes[j].imshow(res, cmap="inferno", vmin=0, vmax=vmax / RES_GAIN)
    axes[j].set_title(f"{k}\\nrms {100 * res.std() / vmax:.2f}%  peak "
                      f"{100 * res.max() / vmax:.1f}%", fontsize=8)
    axes[j].set_xticks([]); axes[j].set_yticks([])

fig.colorbar(im, ax=axes.tolist(), shrink=0.8, label=f"|error|, clipped at {100/RES_GAIN:.0f}% of peak")
fig.suptitle(f"residuals x{RES_GAIN:g}", fontsize=10)
plt.show()'''),
]


def build():
    cells = []
    for kind, src in CELLS:
        lines = src.split("\n")
        source = [l + "\n" for l in lines[:-1]] + [lines[-1]]
        cell = {"cell_type": kind, "metadata": {}, "source": source}
        if kind == CODE:
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "compare_mg_recon.ipynb")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
        f.write("\n")
    return out


if __name__ == "__main__":
    print("wrote", build())
