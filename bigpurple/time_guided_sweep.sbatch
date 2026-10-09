#!/bin/bash
#
# Guided nets timed across attention BACKENDS x NUMBER OF GUIDES x BATCH SIZE --
# scripts/time_guided_sweep.py on the GPU the runs train on. Synthetic inputs; no dataset is read.
#
#   mkdir -p bigpurple/logs && sbatch bigpurple/time_guided_sweep.sbatch
#
# For every (backend, guide count, batch size): a full training step, an inference forward and
# the peak GPU memory. A cell that runs out of memory is recorded and the larger batches of that
# pair are skipped. The LAST table is the one to read: per backend and guide count, the largest
# batch that fit and the best throughput. Also written to
#
#   bigpurple/logs/guided_sweep_<jobid>.csv
#
# Backends:
#   gather_loop   gather with its original one-offset-at-a-time similarity (the old baseline)
#   gather        gather, vectorised similarity
#   flex          FlexAttention, fused, compiled as train.py compiles it
#   triton        the hand-written fused kernel (switches sim_fun to pidistance if it must)
#
# Knobs (environment variables; defaults in brackets):
#
#   CONFIG    [config/NYUMets/i2sb_lggs_session.json]   any guided bridge config
#   BACKENDS  ["gather_loop gather flex triton"]
#   GUIDES    ["1 3 5"]                                 number of guide planes
#   BATCHES   ["1 2 4 8 16 32"]                         ascending (stops at the first OOM)
#   SIZE      [128]                                     frame side (the training crop)
#   REPS      [5]   WARMUP [2]
#   OVERRIDE  [unset]                                   merged into model.params, e.g.
#                                                       OVERRIDE='{guide_window: 7, dK: 10}'
#
#   CONFIG=config/NYUMets/i2sb_sbguidedcdl_otherstudy_sfree.json sbatch bigpurple/time_guided_sweep.sbatch
#   BACKENDS="gather flex" GUIDES="1 5" sbatch bigpurple/time_guided_sweep.sbatch
#
# The first use of a fused backend under the joint softmax checks itself against gather on a
# small problem and stops with an error if they disagree (models/guided_prox.py).
#
# radiology is the only GPU partition these jobs run on. Typed GRES (gpu:a100:1) so the numbers
# are comparable with the A100 training runs.

#SBATCH --partition=radiology
#SBATCH --job-name=guided-sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=EE2178@nyulangone.org
#SBATCH --chdir=/gpfs/scratch/ee2178/ImMAP
#SBATCH --output=/gpfs/scratch/ee2178/ImMAP/bigpurple/logs/%x_%j.out
#SBATCH --error=/gpfs/scratch/ee2178/ImMAP/bigpurple/logs/%x_%j.err

# No `set -euo pipefail` -- it breaks conda activate (unset $PS1 / $CONDA_SHLVL under -u).

# module load anaconda3/gpu/2023.09          # uncomment if conda is not already on PATH
CONDA_BASE="$(conda info --base 2>/dev/null)"
if [ -z "$CONDA_BASE" ]; then
    echo "conda not on PATH -- uncomment the module load line above" >&2
    exit 1
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate env || { echo "conda activate env failed" >&2; exit 1; }

export PYTHONNOUSERSITE=1

IMMAP_ROOT="${IMMAP_ROOT:-$HOME/scratch/ImMAP}"
cd "$IMMAP_ROOT" || { echo "no such dir: $IMMAP_ROOT" >&2; exit 1; }
mkdir -p bigpurple/logs

# --- GPU preflight (nvidia-smi answers over NVML even when the device cannot be opened) ------
PREFLIGHT_TRIES="${PREFLIGHT_TRIES:-6}"
PREFLIGHT_WAIT="${PREFLIGHT_WAIT:-30}"
gpu_ok=0
for try in $(seq 1 "$PREFLIGHT_TRIES"); do
    if python - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    print("no CUDA device visible -- was --gres honoured?", file=sys.stderr)
    sys.exit(1)
try:
    torch.zeros(1, device="cuda")
except Exception as err:
    print(f"context creation failed: {err}", file=sys.stderr)
    sys.exit(1)
print(f"gpu    : {torch.cuda.get_device_name(0)}  (context OK)")
PY
    then
        gpu_ok=1
        break
    fi
    echo "PREFLIGHT: attempt $try/$PREFLIGHT_TRIES failed; retrying in ${PREFLIGHT_WAIT}s" >&2
    sleep "$PREFLIGHT_WAIT"
done
if [ "$gpu_ok" -ne 1 ]; then
    echo "PREFLIGHT FAILED after $PREFLIGHT_TRIES attempts on $(hostname) (job $SLURM_JOB_ID)" >&2
    nvidia-smi --query-gpu=index,name,compute_mode,memory.used --format=csv 2>&1 >&2
    exit 1
fi

# --- knobs --------------------------------------------------------------------------------
CONFIG="${CONFIG:-config/NYUMets/i2sb_lggs_session.json}"
BACKENDS="${BACKENDS:-gather_loop gather flex triton}"
GUIDES="${GUIDES:-1 3 5}"
BATCHES="${BATCHES:-1 2 4 8 16 32}"
SIZE="${SIZE:-128}"
REPS="${REPS:-5}"
WARMUP="${WARMUP:-2}"
OUT="bigpurple/logs/guided_sweep_${SLURM_JOB_ID:-local}.csv"

ARGS=(--backends $BACKENDS --guides $GUIDES --batches $BATCHES --size "$SIZE"
      --reps "$REPS" --warmup "$WARMUP" --csv "$OUT")
[ -n "${OVERRIDE:-}" ] && ARGS+=(--model-override "$OVERRIDE")

echo "host   : $(hostname)"
echo "python : $(which python)"
echo "config : $CONFIG"
echo "sweep  : backends [$BACKENDS]  guides [$GUIDES]  batches [$BATCHES]  frame $SIZE"

python -m scripts.time_guided_sweep "$CONFIG" "${ARGS[@]}"
