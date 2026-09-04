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
#SBATCH --job-name=I2SB-betamaxsweep
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --array=0-4          # ONE TASK PER BETA_MAX BELOW -- keep this 0-(len(BETA_MAXES)-1) in sync with BETA_MAXES
#
# Image-domain I2SB beta_max sweep. Each array task copies an image-domain i2sb config, overrides
# i2sb.beta_max (+ the save_dir / experiment name), and trains it -- so every beta_max lands in its
# own trained_nets/.../beta_max_<v> folder and its own wandb run. I2SB's paper default is
# beta_max=0.3, tuned for the paper's easy restoration tasks; contrast synthesis is a much harder
# map, so a LARGER peak schedule noise (more stochasticity) may be needed -- hence the sweep over a
# range that brackets 0.3 and extends above it.
#
# NOTE: beta_max parameterizes I2SB's default (symmetric/triangular) noise schedule -- NOT the
# Brownian schedule that `tau` drives. Unlike the tau sweep, this script does NOT force
# i2sb.kind="brownian"; it leaves i2sb.kind as the base config sets it. Make sure BASE_CONFIG's
# i2sb.kind is the schedule where beta_max is active (i.e. NOT "brownian"), or beta_max is a no-op.
#
# Submit:   sbatch torch/i2sb_beta_max_sweep.sbatch
# Probe first, then full-train the winner: set SWEEP_EPOCHS below to something small (e.g. 50).
# ---- sweep knobs (edit these) ------------------------------------------------------------------
BASE_CONFIG="config/BraTS/i2sb_paper.json"   # image-domain I2SB config (task must be "i2sb"); swap for
                                       # i2sb_vgg.json / i2sb_synth.json to sweep the conditional runs
SWEEP_NAME="I2SB_beta_max_sweep"       # subfolder under trained_nets/brats/ + wandb name prefix
BETA_MAXES=(0.10 0.20 0.30 0.50 0.70)   # peak schedule beta; LEN MUST MATCH --array
SWEEP_EPOCHS=""                        # empty = use the config's num_epochs; set e.g. 50 to probe cheaply
# ------------------------------------------------------------------------------------------------
# Environment (mirrors torch/i2sb.sbatch)
source ~/.bashrc
conda activate gcdl
cd ~/scratch/ee2178/ImMAP
set -eo pipefail
: "${SLURM_ARRAY_TASK_ID:?must be run as an array job (sbatch sets SLURM_ARRAY_TASK_ID)}"
if [ "${SLURM_ARRAY_TASK_ID}" -ge "${#BETA_MAXES[@]}" ]; then
    echo "[sweep] task ${SLURM_ARRAY_TASK_ID} >= ${#BETA_MAXES[@]} beta_max values -- shrink --array or add values"
    exit 1
fi
BETA_MAX="${BETA_MAXES[${SLURM_ARRAY_TASK_ID}]}"
TAG="beta_max_${BETA_MAX}"
RUN_DIR="trained_nets/brats/${SWEEP_NAME}/${TAG}"
GEN_CONFIG="${RUN_DIR}/config.gen.json"
mkdir -p "${RUN_DIR}" logs
# Write this task's config: override beta_max + savenames, fresh wandb run, optional epoch cap.
python - "${BASE_CONFIG}" "${GEN_CONFIG}" "${BETA_MAX}" "${RUN_DIR}" "${SWEEP_NAME}_${TAG}" "${SWEEP_EPOCHS}" <<'PY'
import json, sys
from training.common import write_config      # yaml-safe writer; see note below
args = sys.argv[1:]
base, out, beta_max, save_dir, name = args[:5]
epochs = args[5] if len(args) > 5 else ""     # optional; robust if the empty arg is dropped
with open(base) as f:
    cfg = json.load(f)
if cfg.get("task") != "i2sb":
    raise SystemExit(f"[sweep] BASE_CONFIG task is {cfg.get('task')!r}, expected 'i2sb' (image-domain "
                     f"bridge). Point BASE_CONFIG at an i2sb config, or relax this check for latent.")
cfg["i2sb"]["beta_max"] = float(beta_max)
# NOTE: no i2sb.kind override here. The tau sweep forced kind="brownian" because tau only affects
# the Brownian schedule; beta_max instead drives I2SB's default (symmetric/triangular) schedule.
# Forcing "brownian" would make beta_max a no-op, so we leave i2sb.kind as the base config sets it.
paths = cfg.setdefault("paths", {}); paths["save_dir"] = save_dir; paths["ckpt"] = None
cfg.setdefault("experiment", {})["name"] = name
cfg.setdefault("wandb", {})["id"] = None          # new run id per beta_max
if epochs:
    cfg.setdefault("training", {})["num_epochs"] = int(epochs)
# write_config, NOT json.dump: train.py reads configs with yaml.safe_load, where json's
# `1e-06` for eta_min comes back as the STRING '1e-06' and CosineAnnealingLR then dies on
# it. write_config writes plain decimal instead and verifies the result with the same
# loader train.py uses, so a bad config fails here rather than mid-training.
write_config(cfg, out)
print(f"[sweep] beta_max={beta_max} -> {out}  (save_dir={save_dir}, "
      f"num_epochs={cfg.get('training', {}).get('num_epochs')})")
PY
echo "[sweep] array task ${SLURM_ARRAY_TASK_ID}: beta_max=${BETA_MAX}  ->  ${RUN_DIR}"
python train.py "${GEN_CONFIG}"

