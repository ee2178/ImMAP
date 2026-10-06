#!/bin/bash
#
# How large a batch fits each net, and what an effective batch of 64 / 256 costs --
# scripts/time_batch_scaling.py on the GPU the runs train on. Synthetic batches with the shapes
# each config's loader produces; no dataset is read.
#
#   mkdir -p bigpurple/logs && sbatch bigpurple/time_batch_scaling.sbatch
#
# Per config: time per micro-batch, samples/s and peak memory for each batch size up to the first
# OOM, then the cheapest (micro-batch x accumulation) reaching each target effective batch and
# the wall clock of a full run. Written to
#
#   bigpurple/logs/batch_scaling_<jobid>.csv   and   ..._plan.csv
#
# Knobs (environment variables; defaults in brackets):
#
#   CONFIGS  [the S-map nets + references]  configs to time, space-separated
#   BATCHES  ["4 8 16 32 64 128 256"]       micro-batch sizes, ascending (stops at the first OOM)
#   TARGETS  ["8 64 256"]                   effective batch sizes to plan for
#   SIZE     [128]                          frame side (the training crop)
#   STEPS    [300000]                       optimizer steps of a full run
#   REPS     [8]   WARMUP [2]
#   OVERRIDE [unset]                        JSON merged into model.params, e.g.
#                                           OVERRIDE='{"attn_backend": "flex", "joint_softmax": false}'
#
#   CONFIGS="config/NYUMets/i2sb_sbcdlnet_sgate.json" BATCHES="32 64 128" sbatch bigpurple/time_batch_scaling.sbatch
#
# radiology is the only GPU partition these jobs run on. Typed GRES (gpu:a100:1) so the numbers
# are comparable with the A100 training runs.

#SBATCH --partition=radiology
#SBATCH --job-name=batch-scaling
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
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
N=config/NYUMets
CONFIGS="${CONFIGS:-$N/i2sb_unet_all_bs256.json $N/i2sb_sbcdlnet_all.json $N/i2sb_sbcdlnet_sfree.json $N/i2sb_sbcdlnet_sgate.json $N/synth_sbcdlnet_sgate.json $N/i2sb_sbguidedcdl_otherstudy_sgate.json $N/synth_sbguidedcdl_otherstudy_sgate.json}"
BATCHES="${BATCHES:-4 8 16 32 64 128 256}"
TARGETS="${TARGETS:-8 64 256}"
SIZE="${SIZE:-128}"
STEPS="${STEPS:-300000}"
REPS="${REPS:-8}"
WARMUP="${WARMUP:-2}"
OUT="bigpurple/logs/batch_scaling_${SLURM_JOB_ID:-local}.csv"

ARGS=(--batches $BATCHES --targets $TARGETS --size "$SIZE" --steps "$STEPS"
      --reps "$REPS" --warmup "$WARMUP" --csv "$OUT")
[ -n "${OVERRIDE:-}" ] && ARGS+=(--model-override "$OVERRIDE")

echo "host   : $(hostname)"
echo "python : $(which python)"
echo "batches: $BATCHES   frame $SIZE   reps $REPS (+$WARMUP warmup)"

python -m scripts.time_batch_scaling $CONFIGS "${ARGS[@]}"
