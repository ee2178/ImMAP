#!/bin/bash
#
# Time LGGS with learned data consistency across UNet sizes for E -- scripts/time_lggs_learned_dc.py
# on the GPU the runs actually train on. CPU timings of this are noise; only these numbers count.
#
#   mkdir -p bigpurple/logs && sbatch bigpurple/time_lggs_dc.sbatch
#
# Runs one sweep per frame size in SIZES (default: 128 = the training crop, 224 = a full stored
# slice), each with the training step timed, and writes one CSV per size next to the log:
#
#   bigpurple/logs/lggs_dc_timing_<jobid>_<size>.csv
#
# Knobs (environment variables; defaults in brackets):
#
#   SIZES   ["128 224"]       frame sizes to sweep
#   BATCH   [4]               the LGGS training batch
#   UNETS   [script default]  E sizes as width:levels:convs, e.g. "4:1:1 8:1:1 16:3:2"
#   K       [config's]        number of LGGS sweeps
#   REPS    [10]  WARMUP [3]
#   NFE     [20]              sampling steps, for the sample/slice column
#   PLAIN   [unset]           1 = bare MagnitudeOperator, no bridge DC wrapper
#   BACKEND [config's]        guided prox backend (gather | flex | triton ...)
#   EXTRA   [unset]           anything else, passed through verbatim
#
#   SIZES=224 UNETS="4:1:1 16:3:2" sbatch bigpurple/time_lggs_dc.sbatch
#   BACKEND=flex sbatch bigpurple/time_lggs_dc.sbatch
#
# radiology is the only GPU partition these jobs run on. Typed GRES (gpu:a100:1) because an
# untyped request can land on a card the conda env's torch was not built for -- and because the
# point is timings comparable with the A100 training runs.

#SBATCH --partition=radiology
#SBATCH --job-name=lggs-dc-time
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
SIZES="${SIZES:-128 224}"
BATCH="${BATCH:-4}"
REPS="${REPS:-10}"
WARMUP="${WARMUP:-3}"
NFE="${NFE:-20}"

ARGS=(--batch "$BATCH" --reps "$REPS" --warmup "$WARMUP" --nfe "$NFE" --backward)
[ -n "${UNETS:-}" ]   && ARGS+=(--unets $UNETS)
[ -n "${K:-}" ]       && ARGS+=(--K "$K")
[ -n "${BACKEND:-}" ] && ARGS+=(--attn-backend "$BACKEND")
[ -n "${PLAIN:-}" ]   && ARGS+=(--plain)
[ -n "${EXTRA:-}" ]   && ARGS+=($EXTRA)

echo "host   : $(hostname)"
echo "python : $(which python)"
echo "sizes  : $SIZES   batch $BATCH   reps $REPS (+$WARMUP warmup)"

# One size failing (e.g. OOM outside the per-arm guard) must not cost the others.
status=0
for SIZE in $SIZES; do
    OUT="bigpurple/logs/lggs_dc_timing_${SLURM_JOB_ID:-local}_${SIZE}.csv"
    echo
    echo "================================ size $SIZE ================================"
    python -m scripts.time_lggs_learned_dc --size "$SIZE" "${ARGS[@]}" --csv "$OUT" \
        || { echo "!! size $SIZE failed" >&2; status=1; }
done
exit $status
