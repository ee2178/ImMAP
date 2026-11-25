#!/bin/bash
#
# Run ONE command on a bigpurple GPU in an INTERACTIVE job -- the bigpurple twin of
# torch/run_gpu.sh. From a login node it asks Slurm for a GPU (srun --pty), runs the command in
# the repo with the output streaming to your terminal, and gives the GPU back when done. Inside
# an existing allocation (salloc, an interactive shell) it just runs.
#
#   bash bigpurple/run_gpu.sh python scripts/smaps_check.py --n 30 --json bigpurple/logs/smaps_check.json
#   bash bigpurple/run_gpu.sh python scripts/inspect_tgvn_trust.py \
#       trained_nets/guided_recon/knee/s4_noise-vlow/tgvn_c6_R8_g1 --n 20 --figs 4 \
#       --json bigpurple/logs/tgvn_trust.json
#   TIME=00:20:00 GPU=a100:1 bash bigpurple/run_gpu.sh python ...
#
# smaps_check's default config was generated on torch; with IMMAP_CLUSTER=bigpurple (set below)
# it swaps in bigpurple's k-space roots itself. The TGVN run dir is the ported one
# (scripts/port_runs.py), already on bigpurple's roots.
#
# ANY GPU on radiology: an untyped gpu:1. Only radiology by default (2026-11-25): jobs on
# a100_short / gl40s_short died at once (a100_short also hands out GPUs held in
# Exclusive_Process mode, bigpurple/i2sb_unet_nyumets.sbatch). Others opt in:
# PARTITION=radiology,a100_short ... A V100 (compute
# capability 7.0) has no kernels in the installed torch build, so the card is checked first and
# a V100 stops the job -- rerun, or pin a type: GPU=a100:1.
# Run it inside tmux if your VPN drops: losing the SSH session kills an srun --pty job.

PARTITION="${PARTITION:-radiology}"
GPU="${GPU:-1}"
TIME="${TIME:-01:00:00}"
MEM="${MEM:-32G}"
CPUS="${CPUS:-4}"
IMMAP_ROOT="${IMMAP_ROOT:-/gpfs/scratch/ee2178/ImMAP}"

[ "$#" -gt 0 ] || { echo "usage: bash bigpurple/run_gpu.sh <command> [args...]" >&2; exit 1; }

if [ -z "$SLURM_JOB_ID" ]; then
    echo "requesting a GPU: partition $PARTITION, gpu $GPU, time $TIME (waits in the queue)"
    exec srun --partition="$PARTITION" --gres="gpu:$GPU" --cpus-per-task="$CPUS" \
        --mem="$MEM" --time="$TIME" --job-name=gpu-cmd --pty \
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
python - <<'PY' || exit 1
import sys, torch
if not torch.cuda.is_available():
    sys.exit("no CUDA device visible")
cc = torch.cuda.get_device_capability(0)
if cc < (7, 5):
    sys.exit(f"{torch.cuda.get_device_name(0)} is compute capability {cc[0]}.{cc[1]}: no kernels "
             f"for it in this torch build. Rerun, or pin a type: GPU=a100:1")
PY
"$@"
