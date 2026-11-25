#!/bin/bash
#
# I2SB-Recon CG-weight sweep on a few random validation slices (scripts/i2sb_lambda_sweep.py), in
# an INTERACTIVE GPU job: run it from a login node and it asks Slurm for a GPU (srun --pty), runs
# the sweep there with the output streaming to your terminal, and gives the GPU back when done.
# Inside an existing allocation (salloc, an interactive shell) it just runs.
#
#   bash bigpurple/i2sb_lambda_sweep.sh                                  # defaults: both sbunet_small
#                                                                        # runs, 8 slices, 7 lams
#   bash bigpurple/i2sb_lambda_sweep.sh --n-samples 12 --lams 1e-3 1e-2 1e-1 1
#   bash bigpurple/i2sb_lambda_sweep.sh --sample-seed 417                # the same slices again
#   TIME=00:30:00 bash bigpurple/i2sb_lambda_sweep.sh --no-figures
#
# Every argument goes to the python script (python scripts/i2sb_lambda_sweep.py --help).
# Results: bigpurple/logs/lambda_sweep_<time>/ (summary.txt, per_slice.csv, <run>.png).
# Run it inside tmux if your VPN drops: losing the SSH session kills an srun --pty job.

PARTITION="${PARTITION:-radiology}"
GPU="${GPU:-a100:1}"
TIME="${TIME:-01:00:00}"
MEM="${MEM:-32G}"
CPUS="${CPUS:-4}"
IMMAP_ROOT="${IMMAP_ROOT:-/gpfs/scratch/ee2178/ImMAP}"

if [ -z "$SLURM_JOB_ID" ]; then
    echo "requesting a GPU: partition $PARTITION, gpu $GPU, time $TIME (waits in the queue)"
    exec srun --partition="$PARTITION" --gres="gpu:$GPU" --cpus-per-task="$CPUS" \
        --mem="$MEM" --time="$TIME" --job-name=lambda-sweep --pty \
        bash "$(readlink -f "$0")" "$@"
fi

# No `set -u`: it breaks conda activate.
CONDA_BASE="$(conda info --base 2>/dev/null)"
if [ -z "$CONDA_BASE" ]; then
    module load anaconda3/gpu/2023.09 2>/dev/null
    CONDA_BASE="$(conda info --base 2>/dev/null)"
fi
[ -n "$CONDA_BASE" ] || { echo "conda not on PATH (module load anaconda3/...?)" >&2; exit 1; }
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate env || { echo "conda activate env failed" >&2; exit 1; }
export PYTHONNOUSERSITE=1
export IMMAP_CLUSTER=bigpurple

cd "$IMMAP_ROOT" || { echo "no such dir: $IMMAP_ROOT" >&2; exit 1; }
mkdir -p bigpurple/logs
echo "host: $(hostname)   job: $SLURM_JOB_ID   $(date '+%F %T')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -u scripts/i2sb_lambda_sweep.py "$@"
