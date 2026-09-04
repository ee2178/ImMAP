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
#SBATCH --job-name=DTCDLNet-lamET-sweep
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --array=0-4          # ONE TASK PER WEIGHT BELOW -- keep 0-(len(LAMS)-1) in sync with LAMS
#
# et_weight sweep over the STANDARD DT-CDLNet-with-dS task (config/BraTS/dt_synthesis.json:
# task=dt_synthesis, use_dS=true, direct T1ce prediction -- NOT residual mode). Each array
# task copies the base config, overrides training.et_weight plus save_dir / experiment name,
# and trains it, so every weight gets its own trained_nets folder and its own wandb run.
#
# Five values, log-spaced:  ~4x, ~12x, ~34x, ~107x, ~335x.
#
# These are the SAME five arms this sweep always had, restated in the units the loss now
# takes. et_weight IS the per-ET-pixel multiplier, which the old area-normalized lam_et set
# only indirectly as 1 + lam_et*(N_all/N_ET) ~ 1 + 334*lam_et; the old grid
# (0.01, 0.0316, 0.1, 0.3162, 1.0) maps onto exactly the values below. Log spacing for the
# same reason as before: linear would put four of five points in the aggressive regime.
#
# Submit:   sbatch torch/synth_lam_sweep.sbatch
# Probe:    set SWEEP_EPOCHS below (e.g. 600) to rank the values cheaply before committing
#           48h x 5. T_max is rescaled to match so the cosine schedule still completes.

# ---- sweep knobs (edit these) ------------------------------------------------------------------
BASE_CONFIG="config/BraTS/dt_synthesis.json"       # must be task=dt_synthesis with use_dS=true
SWEEP_NAME="DT_CDLNet_dS_et_weight_sweep"          # subfolder under trained_nets/brats/ + wandb prefix
LAMS=(4.3 11.6 34.4 107 335)                       # per-ET-pixel weights; LEN MUST MATCH --array
SWEEP_EPOCHS=""                                    # empty = use the config's num_epochs
# ------------------------------------------------------------------------------------------------

# Environment (mirrors torch/dt_synthesis_et.sbatch)
source ~/.bashrc
conda activate gcdl
cd ~/scratch/ee2178/ImMAP

set -eo pipefail
: "${SLURM_ARRAY_TASK_ID:?must be run as an array job (sbatch sets SLURM_ARRAY_TASK_ID)}"

if [ "${SLURM_ARRAY_TASK_ID}" -ge "${#LAMS[@]}" ]; then
    echo "[sweep] task ${SLURM_ARRAY_TASK_ID} >= ${#LAMS[@]} weights -- shrink --array or add values"
    exit 1
fi
LAM="${LAMS[${SLURM_ARRAY_TASK_ID}]}"
TAG="etw_${LAM}"
RUN_DIR="trained_nets/brats/${SWEEP_NAME}/${TAG}"
GEN_CONFIG="${RUN_DIR}/config.gen.json"

mkdir -p "${RUN_DIR}" logs

# Write this task's config: override et_weight + savenames, fresh wandb run, optional epoch cap.
python - "${BASE_CONFIG}" "${GEN_CONFIG}" "${LAM}" "${RUN_DIR}" "${SWEEP_NAME}_${TAG}" "${SWEEP_EPOCHS}" <<'PY'
import json, sys
from training.common import write_config      # yaml-safe writer; see note below

args = sys.argv[1:]
base, out, lam, save_dir, name = args[:5]
epochs = args[5] if len(args) > 5 else ""     # optional; robust if the empty arg is dropped

with open(base) as f:
    cfg = json.load(f)

if cfg.get("task") != "dt_synthesis":
    raise SystemExit(f"[sweep] BASE_CONFIG task is {cfg.get('task')!r}, expected 'dt_synthesis'.")
if not cfg["model"]["params"].get("use_dS", False):
    raise SystemExit("[sweep] BASE_CONFIG has use_dS=false; this sweep is for the WITH-dS task.")

tr = cfg.setdefault("training", {})
tr["et_weight"] = float(lam)

# et_weight is a no-op unless the loader actually returns an ET mask -- fail here rather than
# after the job has queued, waited, and started.
for split in ("train", "val"):
    if not cfg["data"][split].get("et_mask", False):
        raise SystemExit(f"[sweep] data.{split}.et_mask is not true; the ET term would be a no-op.")

paths = cfg.setdefault("paths", {}); paths["save_dir"] = save_dir; paths["ckpt"] = None
cfg.setdefault("experiment", {})["name"] = name
cfg.setdefault("wandb", {})["id"] = None      # fresh run id per weight

if epochs:
    tr["num_epochs"] = int(epochs)
    # Rescale the cosine horizon with the epoch cap. Left alone, a shortened probe would
    # traverse only part of the curve and every arm would end at a high LR -- which is not
    # what the full run does, so the ranking would not transfer.
    sched = cfg.get("scheduler", {})
    if sched.get("type") == "CosineAnnealingLR":
        sched["params"]["T_max"] = int(tr["num_epochs"]) * int(tr["steps_per_epoch"])

# write_config, NOT json.dump: train.py reads configs with yaml.safe_load, where a float
# needs both a decimal point and a signed exponent. json.dump writes eta_min as `1e-06`,
# which yaml returns as the STRING '1e-06' and CosineAnnealingLR then dies on. write_config
# canonicalizes and verifies with the same loader train.py uses.
write_config(cfg, out)
print(f"[sweep] et_weight={lam} -> {out}  (save_dir={save_dir}, "
      f"num_epochs={tr.get('num_epochs')}, T_max={cfg.get('scheduler',{}).get('params',{}).get('T_max')})")
PY

echo "[sweep] array task ${SLURM_ARRAY_TASK_ID}: et_weight=${LAM}  ->  ${RUN_DIR}"
python train.py "${GEN_CONFIG}"

