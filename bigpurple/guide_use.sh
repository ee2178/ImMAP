#!/bin/bash
#
# Guide-influence test (scripts/diagnose_guide_use.py) in an INTERACTIVE GPU job: run it from a
# login node and it asks Slurm for a GPU (srun --pty), runs the test with the output streaming to
# your terminal, and gives the GPU back when done. Inside an existing allocation (salloc, an
# interactive shell) it just runs.
#
#   bash bigpurple/guide_use.sh                                   # LGGS, LGGS guide-W, TGVN c6,
#                                                                 # LPDSNet (reference), 30 slices
#   bash bigpurple/guide_use.sh --n 10                            # quicker
#   RUNS="lggs_R8_g1 tgvn_c6_R8_g1" bash bigpurple/guide_use.sh   # other runs (names under ROOT)
#   TIME=00:30:00 bash bigpurple/guide_use.sh
#
# Per run: PSNR / SSIM / NRMSE with the true, rotated (wrong anatomy), zero and scaled-down
# guides, the drops against the true guide, and LGGS's attention mass on the guide per layer.
# Arguments after the script go to the python script (python scripts/diagnose_guide_use.py -h);
# --n defaults to 30 here. Results also in bigpurple/logs/guide_use_<time>.json.
# Run it inside tmux if your VPN drops: losing the SSH session kills an srun --pty job.
#
# ANY GPU on radiology: an untyped gpu:1. Only radiology by default (2026-11-25): jobs on
# a100_short / gl40s_short died at once (a100_short also hands out GPUs held in
# Exclusive_Process mode, bigpurple/i2sb_unet_nyumets.sbatch). Others opt in:
# PARTITION=radiology,a100_short ... The one card that
# cannot work is a V100 (compute capability 7.0: the installed torch build ships no kernels for
# it, bigpurple/jupyter.sbatch), so the job checks the card first and stops on one -- then rerun,
# or pin a type: GPU=a100:1 bash bigpurple/guide_use.sh

PARTITION="${PARTITION:-radiology}"
GPU="${GPU:-1}"
TIME="${TIME:-01:00:00}"
MEM="${MEM:-32G}"
CPUS="${CPUS:-4}"
IMMAP_ROOT="${IMMAP_ROOT:-/gpfs/scratch/ee2178/ImMAP}"
ROOT="${ROOT:-trained_nets/guided_recon/knee/s4_noise-vlow}"
RUNS="${RUNS:-lggs_R8_g1 lggs_guide_w_R8_g1 tgvn_c6_R8_g1 lpdsnet_R8}"

if [ -z "$SLURM_JOB_ID" ]; then
    echo "requesting a GPU: partition $PARTITION, gpu $GPU, time $TIME (waits in the queue)"
    exec srun --partition="$PARTITION" --gres="gpu:$GPU" --cpus-per-task="$CPUS" \
        --mem="$MEM" --time="$TIME" --job-name=guide-use --pty \
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

# the runs that are actually here (a run never copied over is skipped, not fatal)
DIRS=()
for r in $RUNS; do
    if [ -f "$ROOT/$r/config.json" ] && [ -f "$ROOT/$r/net_bestval.ckpt" ]; then
        DIRS+=("$ROOT/$r")
    else
        echo "[skip] $r: no config.json or net_bestval.ckpt in $ROOT/$r" >&2
    fi
done
[ "${#DIRS[@]}" -gt 0 ] || { echo "none of RUNS exist under $ROOT" >&2; exit 1; }

# --n 30 unless given
case " $* " in *" --n "*) N_ARG=() ;; *) N_ARG=(--n 30) ;; esac
JSON="bigpurple/logs/guide_use_$(date +%Y%m%d_%H%M%S).json"
echo "host: $(hostname)   job: $SLURM_JOB_ID   $(date '+%F %T')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python - <<'PY' || exit 1
import sys, torch
if not torch.cuda.is_available():
    sys.exit("no CUDA device visible")
cc = torch.cuda.get_device_capability(0)
if cc < (7, 5):
    sys.exit(f"{torch.cuda.get_device_name(0)} is compute capability {cc[0]}.{cc[1]}: this torch "
             f"build has no kernels for it. Rerun, or pin a type: GPU=a100:1 bash bigpurple/guide_use.sh")
PY
python -u scripts/diagnose_guide_use.py "${DIRS[@]}" "${N_ARG[@]}" --json "$JSON" "$@"
