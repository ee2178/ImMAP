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
#SBATCH --array=0-5          # ONE TASK PER LAMBDA BELOW -- keep 0-(len(LAMS)-1) in sync with LAMS
#
# lam_et sweep over the STANDARD DT-CDLNet-with-dS task (config/BraTS/dt_synthesis.json:
# task=dt_synthesis, use_dS=true, direct T1ce prediction -- NOT residual mode). Each array
# task copies the base config, overrides training.lam_et plus save_dir / experiment name,
# and trains it, so every lambda gets its own trained_nets folder and its own wandb run.
#
# Six arms: lam_et = 0 (CONTROL) plus five log-spaced over [0.001, 0.1].
#
# The control is not optional. lam_et=0 short-circuits the ET term entirely, so arm 0 is the
# plain un-weighted loss trained with the SAME code, config, data scaling and seed as the
# rest. Without it, "detail outside the ET mask looks worse" has nothing to be worse THAN --
# any earlier run differs in scales and in several code changes besides.
#
# Why this range: the ET term is area-normalized, so one ET pixel carries weight
# 1 + lam_et*(N_all/N_ET). At ~0.3% ET area that is
#
#     lam_et:   0      0.001   0.00316   0.01    0.0316   0.1
#     ET px:    1x     1.3x    2.1x      4.3x    11.5x    34x
#     share:    0%     0.2%    0.7%      2.2%    6.6%     18%
#
# i.e. this grid spans "no weighting" to "moderate", after the previous [0.01, 1.0] grid
# turned out to start too high. Log spacing because the multiplier moves by decades.
#
# Submit:   sbatch slurm/dt_synthesis_lam_et_sweep.sbatch
# Probe:    set SWEEP_EPOCHS below (e.g. 600) to rank the values cheaply before committing
#           48h x 6. T_max is rescaled to match so the cosine schedule still completes.
#
# Before spending 6 jobs: if the degradation is a stalled LR rather than lam_et (the ET term
# raises loss variance, backtracking fires, LR decays, training stops), no value here helps.
# notebooks/lam_et_sweep.ipynb sections 1-3 separate those in a few minutes on one GPU.

# ---- sweep knobs (edit these) ------------------------------------------------------------------
BASE_CONFIG="config/BraTS/dt_synthesis.json"       # must be task=dt_synthesis with use_dS=true
SWEEP_NAME="DT_CDLNet_dS_lam_et_sweep"             # subfolder under trained_nets/brats/ + wandb prefix
LAMS=(0 0.001 0.00316 0.01 0.0316 0.1)             # 0 = control, then log-spaced 1e-3 .. 1e-1
                                                   # LEN MUST MATCH --array
SWEEP_EPOCHS=""                                    # empty = use the config's num_epochs
# ------------------------------------------------------------------------------------------------

# Environment (mirrors slurm/dt_synthesis_et.sbatch)
source ~/.bashrc
conda activate gcdl
cd ~/scratch/ee2178/ImMAP

set -eo pipefail
: "${SLURM_ARRAY_TASK_ID:?must be run as an array job (sbatch sets SLURM_ARRAY_TASK_ID)}"

if [ "${SLURM_ARRAY_TASK_ID}" -ge "${#LAMS[@]}" ]; then
    echo "[sweep] task ${SLURM_ARRAY_TASK_ID} >= ${#LAMS[@]} lambdas -- shrink --array or add values"
    exit 1
fi
LAM="${LAMS[${SLURM_ARRAY_TASK_ID}]}"
TAG="lam_et_${LAM}"
RUN_DIR="trained_nets/brats/${SWEEP_NAME}/${TAG}"
GEN_CONFIG="${RUN_DIR}/config.gen.json"

mkdir -p "${RUN_DIR}" logs

# Write this task's config: override lam_et + savenames, fresh wandb run, optional epoch cap.
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
tr["lam_et"] = float(lam)

# lam_et is a no-op unless the loader actually returns an ET mask -- fail here rather than
# after the job has queued, waited, and started.
for split in ("train", "val"):
    if not cfg["data"][split].get("et_mask", False):
        raise SystemExit(f"[sweep] data.{split}.et_mask is not true; the ET term would be a no-op.")

paths = cfg.setdefault("paths", {}); paths["save_dir"] = save_dir; paths["ckpt"] = None
cfg.setdefault("experiment", {})["name"] = name
cfg.setdefault("wandb", {})["id"] = None      # fresh run id per lambda

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
print(f"[sweep] lam_et={lam} -> {out}  (save_dir={save_dir}, "
      f"num_epochs={tr.get('num_epochs')}, T_max={cfg.get('scheduler',{}).get('params',{}).get('T_max')})")
PY

echo "[sweep] array task ${SLURM_ARRAY_TASK_ID}: lam_et=${LAM}  ->  ${RUN_DIR}"
python train.py "${GEN_CONFIG}"
