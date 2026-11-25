#!/bin/bash
#
# Run ONE command on a torch GPU in an INTERACTIVE job: from a login node it asks Slurm for a GPU
# (srun --pty), runs the command in the repo with the output streaming to your terminal, and
# gives the GPU back when done. Inside an existing allocation it just runs.
#
#   bash torch/run_gpu.sh python scripts/smaps_check.py --n 30
#   bash torch/run_gpu.sh python scripts/inspect_tgvn_trust.py \
#       trained_nets/guided_recon/knee/s4_noise-vlow/tgvn_c6_R8_g1 --n 20 --figs 4
#   TIME=00:20:00 CONSTRAINT=l40s bash torch/run_gpu.sh python ...
#
# Any GPU by default (gpu:1, no constraint); cards below compute capability 7.5 are refused
# before the command runs (no kernels in the installed torch build).
# Run it inside tmux if your VPN drops: losing the SSH session kills an srun --pty job.

ACCOUNT="${ACCOUNT:-torch_pr_89_tandon_advanced}"
GPU="${GPU:-1}"
CONSTRAINT="${CONSTRAINT:-}"
TIME="${TIME:-01:00:00}"
MEM="${MEM:-32G}"
CPUS="${CPUS:-4}"
IMMAP_ROOT="${IMMAP_ROOT:-$HOME/scratch/ee2178/ImMAP}"

[ "$#" -gt 0 ] || { echo "usage: bash torch/run_gpu.sh <command> [args...]" >&2; exit 1; }

if [ -z "$SLURM_JOB_ID" ]; then
    echo "requesting a GPU: account $ACCOUNT, gpu $GPU${CONSTRAINT:+ ($CONSTRAINT)}, time $TIME (waits in the queue)"
    exec srun --account="$ACCOUNT" --gres="gpu:$GPU" ${CONSTRAINT:+--constraint="$CONSTRAINT"} \
        --cpus-per-task="$CPUS" --mem="$MEM" --time="$TIME" --job-name=gpu-cmd --pty \
        bash "$(readlink -f "$0")" "$@"
fi

# `source ~/.bashrc` and `conda activate` trip over `set -u`; activate first (as torch/*.sbatch).
source ~/.bashrc
conda activate gcdl || { echo "conda activate gcdl failed" >&2; exit 1; }
export PYTHONNOUSERSITE=1
export IMMAP_CLUSTER=torch
cd "$IMMAP_ROOT" || { echo "no such dir: $IMMAP_ROOT" >&2; exit 1; }
mkdir -p torch/logs

echo "host: $(hostname)   job: $SLURM_JOB_ID   $(date '+%F %T')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python - <<'PY' || exit 1
import sys, torch
if not torch.cuda.is_available():
    sys.exit("no CUDA device visible")
cc = torch.cuda.get_device_capability(0)
if cc < (7, 5):
    sys.exit(f"{torch.cuda.get_device_name(0)} is compute capability {cc[0]}.{cc[1]}: no kernels "
             f"for it in this torch build. Rerun, or pin a card: CONSTRAINT=l40s")
PY
"$@"
