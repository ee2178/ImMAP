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
#SBATCH --job-name=LatentI2SB-imgwsweep
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --array=0-6          # ONE TASK PER WEIGHT below -- keep this 0-(len(IMAGE_WEIGHTS)-1) in sync
#
# Latent-I2SB mixed-loss sweep over the IMAGE-domain loss weight. In loss_mode="mixed" the loss is
#     latent_weight * (z0 latent term)  +  image_weight * (decoded-image term)
# latent_weight is held FIXED at the base config's value; each array task overrides
# training.image_weight (+ savenames) and trains it, so every weight gets its own folder + wandb run.
# image_weight=0 is the pure-latent baseline; too large a weight risks the latent-collapse the latent
# term is there to prevent -- the sweep is to find the sweet spot.
#
# Submit:  sbatch torch/latent_i2sb_imgweight_sweep.sbatch
# Probe first: set SWEEP_EPOCHS to something small (e.g. 500), compare val, then full-train the winner.

# ---- sweep knobs (edit these) ------------------------------------------------------------------
BASE_CONFIG="config/BraTS/latent_inf_latent_bridge.json"   # latent-I2SB config (task "latent_i2sb");
                                                           # swap for latent_inf_img_bridge.json
SWEEP_NAME="LatentI2SB_imgw_sweep"        # subfolder under trained_nets/brats/ + wandb name prefix
IMAGE_WEIGHTS=(0.0 0.01 0.05 0.1 0.3 1.0 3.0)   # image-domain loss weights; LEN MUST MATCH --array
SWEEP_EPOCHS=""                           # empty = config's num_epochs; set e.g. 500 to probe cheaply
# ------------------------------------------------------------------------------------------------

# Environment (mirrors torch/latent_i2sb.sbatch)
source ~/.bashrc
conda activate gcdl
cd ~/scratch/ee2178/ImMAP

# flex_attention JIT-compiles Triton kernels on the FIRST iteration (looks like a "freeze" at step 0
# -- it is compiling, not hung). Point the Inductor/Triton caches at PERSISTENT scratch so the
# compile is paid ONCE, not every job (the default lands in job-ephemeral state and is wiped).
export TORCHINDUCTOR_CACHE_DIR=/scratch/ee2178/.cache/torchinductor
export TRITON_CACHE_DIR=/scratch/ee2178/.cache/triton

set -eo pipefail
: "${SLURM_ARRAY_TASK_ID:?must be run as an array job (sbatch sets SLURM_ARRAY_TASK_ID)}"

if [ "${SLURM_ARRAY_TASK_ID}" -ge "${#IMAGE_WEIGHTS[@]}" ]; then
    echo "[sweep] task ${SLURM_ARRAY_TASK_ID} >= ${#IMAGE_WEIGHTS[@]} weights -- fix --array or the list"
    exit 1
fi
IW="${IMAGE_WEIGHTS[${SLURM_ARRAY_TASK_ID}]}"
TAG="imgw_${IW}"
RUN_DIR="trained_nets/brats/${SWEEP_NAME}/${TAG}"
GEN_CONFIG="${RUN_DIR}/config.gen.json"

mkdir -p "${RUN_DIR}" logs

# Write this task's config: force loss_mode="mixed", override image_weight + savenames, fresh wandb run.
python - "${BASE_CONFIG}" "${GEN_CONFIG}" "${IW}" "${RUN_DIR}" "${SWEEP_NAME}_${TAG}" "${SWEEP_EPOCHS}" <<'PY'
import json, sys
from training.common import write_config      # yaml-safe writer; see note below
args = sys.argv[1:]
base, out, iw, save_dir, name = args[:5]
epochs = args[5] if len(args) > 5 else ""      # optional; robust if the empty arg is dropped
with open(base) as f:
    cfg = json.load(f)
if cfg.get("task") != "latent_i2sb":
    raise SystemExit(f"[sweep] BASE_CONFIG task is {cfg.get('task')!r}, expected 'latent_i2sb'.")
tr = cfg.setdefault("training", {})
tr["loss_mode"]    = "mixed"                   # image_weight only bites the mixed loss
tr["image_weight"] = float(iw)                 # latent_weight is left as-is (sweep image weight only)
paths = cfg.setdefault("paths", {}); paths["save_dir"] = save_dir; paths["ckpt"] = None
cfg.setdefault("experiment", {})["name"] = name
cfg.setdefault("wandb", {})["id"] = None       # new run id per weight
if epochs:
    tr["num_epochs"] = int(epochs)
# write_config, NOT json.dump: train.py reads configs with yaml.safe_load, where json's
# `1e-06` for eta_min comes back as the STRING '1e-06' and CosineAnnealingLR then dies on
# it. write_config writes plain decimal instead and verifies the result with the same
# loader train.py uses, so a bad config fails here rather than mid-training.
write_config(cfg, out)
print(f"[sweep] image_weight={iw}  (latent_weight={tr.get('latent_weight')}, loss_mode={tr['loss_mode']}, "
      f"latent_loss={tr.get('latent_loss')}) -> {out}  (save_dir={save_dir}, num_epochs={tr.get('num_epochs')})")
PY

echo "[sweep] array task ${SLURM_ARRAY_TASK_ID}: image_weight=${IW}  ->  ${RUN_DIR}"
python train.py "${GEN_CONFIG}"
