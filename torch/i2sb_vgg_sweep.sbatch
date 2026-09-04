#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=torch_pr_89_tandon_advanced
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=48:00:00
#SBATCH --job-name=SBCDLNet-vgg-sweep
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --array=0-5          # MUST equal len(VGG_WEIGHTS) - 1; checked at runtime
#
# 1-D sweep over the perceptual-loss weight for SBCDLNet: loss = MSE + vgg_weight * VGG(relu4_3).
# Replaces reading the startup balance line and guessing -- log-spaced over five decades, which is
# the range the balance actually spans.
#
# ---------------------------------------------------------------------------------------------
# HOW TO READ THIS SWEEP -- the part that decides whether it was worth the GPU hours
# ---------------------------------------------------------------------------------------------
#
# DO NOT RANK ON val/psnr. Adding a perceptual term trades distortion for perceptual quality; that
# is the entire point of it. PSNR measures distortion, so it is MONOTONE DECREASING in vgg_weight
# by construction and would pick vgg_weight=0 every single time, no matter how good the perceptual
# runs look. A sweep ranked on PSNR is a sweep that cannot answer the question it was run for.
#
# So every cell here sets `val_lpips: true`, and the criterion is the FRONTIER:
#
#     x-axis  val/psnr   (higher better)  -- what you give up
#     y-axis  val/lpips  (LOWER better)   -- what you buy
#
# Plot the six cells in that plane. vgg_weight=0 anchors the top-left. What you are looking for is
# a cell where LPIPS drops substantially for a PSNR cost you are willing to pay -- typically a knee
# somewhere in the middle. If LPIPS never improves, the perceptual term is not helping this task
# and the honest conclusion is to keep the MSE-only model.
#
# Also worth eyeballing: val/example in wandb. Perceptual losses fix things (texture, edge
# sharpness) that neither PSNR nor LPIPS fully captures, and they introduce artifacts that both can
# miss. Look at the ET region in particular.
#
# LPIPS CAVEATS. It is computed on jointly min-max normalised magnitude images and is reported
# UNMASKED (see evaluation/metrics.py:lpips -- a deep-feature distance cannot honestly be
# restricted to a region), whereas psnr/ssim here are masked. Do not quote them as if they covered
# the same pixels. If LPIPS weights are not cached on the compute node the run warns ONCE and
# continues without the metric rather than dying 40 hours in -- so warm the cache on a login node
# first (`python -c "import lpips; lpips.LPIPS(net='alex')"`), or you will get a sweep you cannot
# rank.
#
# ---------------------------------------------------------------------------------------------
# CELL 0 IS THE CONTROL, AND IT IS CHEAP ON PURPOSE
# ---------------------------------------------------------------------------------------------
# vgg_weight=0 switches loss_type to "complex-mse" rather than running "mse-vgg" with a zero
# coefficient. The objective is bit-identical -- mse_vgg's pixel term IS complex_mse -- but it
# skips the VGG16 forward on every step, so the control costs what the baseline costs. It is
# therefore also a re-run of config/BraTS/i2sb_sbcdlnet_mednorm.json under this sweep's naming,
# which is what you want: the control must share the seed, schedule and epoch count of the cells
# it anchors, not be borrowed from an earlier run with different settings.
#
# COST: 6 full trainings, 5 of them paying a VGG16 forward per step. PROBE FIRST -- set
# SWEEP_EPOCHS to something short, plot the frontier, then full-train the one or two cells worth it.
#
# PREREQUISITE: the h5 files need 'img_median_mad' --
#   python preprocessing/cmap.py --config config/BraTS/cmap.yaml
#
# Submit:  sbatch torch/i2sb_vgg_sweep.sbatch
# One cell: sbatch --array=3 torch/i2sb_vgg_sweep.sbatch

# ---- sweep knobs (edit these) --------------------------------------------------------------
BASE_CONFIG="config/BraTS/i2sb_sbcdlnet_vgg.json"
SWEEP_NAME="I2SB_SBCDLNet_vggsweep"
VGG_WEIGHTS=(0 1e-3 1e-2 1e-1 1e0 1e1)   # 0 = MSE-only control (see above)
SIZE_NORMALIZE="false"                   # true reproduces the TF reference's 1/(H*W*CH), which
                                         # divides the perceptual term by ~3e5 -- if you turn it
                                         # on, shift every weight up by ~5 decades to compensate
IMAGENET_NORM="false"                    # true = per-sample min-max to [0,1] + ImageNet mean/std
                                         # before VGG (better conditioned, deviates from the ref)
SWEEP_EPOCHS=""                          # empty = the config's num_epochs; set e.g. 300 to probe
# ---------------------------------------------------------------------------------------------

source ~/.bashrc
conda activate gcdl
cd ~/scratch/ee2178/ImMAP

set -eo pipefail
: "${SLURM_ARRAY_TASK_ID:?must be run as an array job (sbatch sets SLURM_ARRAY_TASK_ID)}"

N_W=${#VGG_WEIGHTS[@]}
if [ "${SLURM_ARRAY_TASK_ID}" -ge "${N_W}" ]; then
    echo "[vggsweep] task ${SLURM_ARRAY_TASK_ID} >= ${N_W} cells -- fix --array"
    echo "           (should be 0-$(( N_W - 1 ))) or add weights"
    exit 1
fi

VW="${VGG_WEIGHTS[${SLURM_ARRAY_TASK_ID}]}"
TAG="vgg_${VW}"
RUN_DIR="trained_nets/brats/${SWEEP_NAME}/${TAG}"
GEN_CONFIG="${RUN_DIR}/config.gen.json"
mkdir -p "${RUN_DIR}" logs

echo "[vggsweep] task ${SLURM_ARRAY_TASK_ID}/$(( N_W - 1 )): vgg_weight=${VW} -> ${RUN_DIR}"

python - "${BASE_CONFIG}" "${GEN_CONFIG}" "${VW}" "${RUN_DIR}" \
         "${SWEEP_NAME}_${TAG}" "${SIZE_NORMALIZE}" "${IMAGENET_NORM}" "${SWEEP_EPOCHS}" <<'PY'
import json, sys
from training.common import write_config

args = sys.argv[1:]
base, out, vgg_weight, save_dir, name, size_norm, imnet_norm = args[:7]
epochs = args[7] if len(args) > 7 else ""      # optional; robust if the empty arg is dropped

with open(base) as f:
    cfg = json.load(f)

if cfg.get("task") != "i2sb":
    raise SystemExit(f"[vggsweep] BASE_CONFIG task is {cfg.get('task')!r}, expected 'i2sb'.")

vgg_weight = float(vgg_weight)
size_norm = size_norm.strip().lower() == "true"
imnet_norm = imnet_norm.strip().lower() == "true"
tr = cfg["training"]

# ---- 1. loss_params only reaches the registry entry on the UNWEIGHTED path (see
#         training/i2sb.py:_bridge_loss). train_i2sb refuses the combination at startup; catch it
#         here so a bad base config fails in one second rather than after the data loads.
if tr.get("loss_weight", "uniform") != "uniform" or float(tr.get("et_weight", 1)) != 1:
    raise SystemExit(
        f"[vggsweep] base config has loss_weight={tr.get('loss_weight')!r} / "
        f"et_weight={tr.get('et_weight')}, which route the objective through the POINTWISE loss "
        f"registry -- that path takes no loss_params, so vgg_weight would be silently dropped. "
        f"Set loss_weight='uniform' and et_weight=1 in {base}.")

# ---- 2. the loss for this cell ----
if vgg_weight == 0.0:
    # Bit-identical to mse_vgg with vgg_weight=0 (its pixel term IS complex_mse), minus the VGG
    # forward. loss_params must go too: complex-mse takes only (x, y, sigma).
    tr["loss_type"] = "complex-mse"
    tr.pop("loss_params", None)
else:
    tr["loss_type"] = "mse-vgg"
    tr["loss_params"] = {"vgg_weight": vgg_weight, "mse_weight": 1.0,
                         "size_normalize": size_norm, "imagenet_norm": imnet_norm}

# ---- 3. the metric that makes the sweep rankable ----
# Without this every cell reports only psnr/ssim/nrmse, all of which are monotone in vgg_weight
# by construction, and the sweep cannot be read. See the header.
tr["val_lpips"] = True

# ---- 4. schedule sanity: the SB* nets duplicate cfg["i2sb"] in model.params, and a drift means
#         the wrong bridge coefficients at every step. Nothing here touches the schedule, so this
#         only guards a base config that was already inconsistent.
mparams = cfg["model"]["params"]
for key in ("kind", "n_points", "beta_max", "tau"):
    if key in mparams and key in cfg["i2sb"] and mparams[key] != cfg["i2sb"][key]:
        raise SystemExit(
            f"[vggsweep] model.params.{key}={mparams[key]!r} != i2sb.{key}="
            f"{cfg['i2sb'][key]!r} in {base}. The net recovers its bridge coefficients by "
            f"inverting this schedule -- fix the base config.")

# ---- 5. run bookkeeping ----
paths = cfg.setdefault("paths", {})
paths["save_dir"] = save_dir
paths["ckpt"] = None                            # fresh run; train.py rewrites this to net.ckpt
cfg.setdefault("experiment", {})["name"] = name
cfg.setdefault("wandb", {})["id"] = None        # a new wandb run per cell
if epochs:
    tr["num_epochs"] = int(epochs)

# write_config, NOT json.dump: train.py reads configs with yaml.safe_load, where json's `1e-06`
# for eta_min comes back as the STRING '1e-06' and CosineAnnealingLR dies on it. write_config
# writes plain decimal and re-reads with the same loader, so a bad config fails here.
write_config(cfg, out)

print(f"[vggsweep] vgg_weight={vgg_weight:g} loss_type={tr['loss_type']} "
      f"loss_params={tr.get('loss_params')}\n"
      f"           model={cfg['model']['type']} val_lpips=True "
      f"num_epochs={tr.get('num_epochs')} -> {out}")
PY

python train.py "${GEN_CONFIG}"
