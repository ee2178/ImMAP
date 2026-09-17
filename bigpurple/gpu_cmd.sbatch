#!/bin/bash
#
# Run ONE command on a GPU node -- the batch stand-in for an interactive session when the queue
# will not give you one. The command comes from $CMD (or the arguments after the script):
#
#   CMD="python -m scripts.time_cdlnet_dc --config config/NYUMets/i2sb_cdlnet_learned_dc_mini.json \
#        --size 128 --batch 8 --backward" sbatch bigpurple/gpu_cmd.sbatch
#
#   sbatch bigpurple/gpu_cmd.sbatch python -m scripts.show_nyumets_masks --n 6 --erode 3
#
# Partition / GPU type / walltime are overridden on the command line, which beats the #SBATCH
# lines below:
#
#   sbatch --partition=gl40s_short --gres=gpu:l40s:1 bigpurple/gpu_cmd.sbatch ...
#   sbatch --partition=a100_short --exclusive --time=00:30:00 bigpurple/gpu_cmd.sbatch ...
#
# Output lands in bigpurple/logs/<job-name>_<jobid>.out; pass --job-name to make it findable.
# For TRAINING use the task's own script (i2sb_unet_nyumets.sbatch with CFG=...), which also
# handles the split and the data-root checks.

#SBATCH --partition=a100_short
#SBATCH --job-name=gpu-cmd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=EE2178@nyulangone.org
#SBATCH --chdir=/gpfs/scratch/ee2178/ImMAP
#SBATCH --output=/gpfs/scratch/ee2178/ImMAP/bigpurple/logs/%x_%j.out
#SBATCH --error=/gpfs/scratch/ee2178/ImMAP/bigpurple/logs/%x_%j.err

# No `set -euo pipefail` -- it breaks conda activate (unset $PS1 / $CONDA_SHLVL under -u).

if [ -z "$CMD" ] && [ "$#" -eq 0 ]; then
    echo "nothing to run: set CMD=... or pass the command as arguments" >&2
    exit 1
fi

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
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>&1 >&2
    echo "a100_short hands out cards held in Exclusive_Process mode; resubmit, add --exclusive," >&2
    echo "or use --partition=radiology / gl40s_short." >&2
    exit 1
fi

echo "host   : $(hostname)"
echo "cwd    : $(pwd)"

if [ -n "$CMD" ]; then
    echo "cmd    : $CMD"
    # eval, so quoting and pipes inside CMD behave as typed
    eval "$CMD"
else
    echo "cmd    : $*"
    "$@"
fi
