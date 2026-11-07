#!/bin/bash
#
# Benchmark LGGS's triton guide branch, implicit vs materialised softmax weights --
# scripts/bench_lggs_materialize.py -- on the GPU the runs train on.
#
#   mkdir -p bigpurple/logs && sbatch bigpurple/bench_lggs_materialize.sbatch
#
# Writes bigpurple/logs/lggs_materialize_<jobid>.json next to the log. The script checks every
# new kernel against the torch reference first and stops without timing if one disagrees.
#
# Knobs (environment variables; defaults in brackets):
#
#   STAGES  ["kernels branch net"]
#   BATCH   [1]    GUIDES [1]    K [config's]    ITERS [10]    WARMUP [3]
#   EXTRA   [unset]   passed through verbatim, e.g. "--H 320 --W 160"
#
#   BATCH=2 GUIDES=3 sbatch bigpurple/bench_lggs_materialize.sbatch
#

#SBATCH --partition=radiology
#SBATCH --job-name=lggs-materialize-bench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
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

STAGES="${STAGES:-kernels branch net}"
ARGS=(--stages $STAGES --batch "${BATCH:-1}" --guides "${GUIDES:-1}"
      --iters "${ITERS:-10}" --warmup "${WARMUP:-3}"
      --json "bigpurple/logs/lggs_materialize_${SLURM_JOB_ID:-local}.json")
[ -n "${K:-}" ]     && ARGS+=(--K "$K")
[ -n "${EXTRA:-}" ] && ARGS+=($EXTRA)

echo "host   : $(hostname)"
echo "python : $(which python)"
python scripts/bench_lggs_materialize.py "${ARGS[@]}"
