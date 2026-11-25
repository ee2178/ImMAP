#!/bin/bash
#
# Guide-influence test (scripts/diagnose_guide_use.py) in an INTERACTIVE GPU job on TORCH -- the
# torch twin of bigpurple/guide_use.sh. Run it from a login node and it asks Slurm for a GPU
# (srun --pty), runs the test with the output streaming to your terminal, and gives the GPU back
# when done. Inside an existing allocation (salloc, an interactive shell) it just runs.
#
#   bash torch/guide_use.sh                       # image-loss vs SSDU pairs + LPDSNet references
#   bash torch/guide_use.sh --n 10                # quicker
#   RUNS="lggs_guide_w_R8_g1 tgvn_c6_R8_g1" bash torch/guide_use.sh
#   TIME=00:30:00 bash torch/guide_use.sh
#
# Per run: PSNR / SSIM / NRMSE with the true, rotated (wrong anatomy), zero and scaled-down
# guides, the drops against the true guide, and LGGS's attention mass on the guide per layer.
# Each run is scored with its net_bestval.ckpt. LPDSNet takes no guide: a reference row.
# Arguments after the script go to the python script (python scripts/diagnose_guide_use.py -h);
# --n defaults to 30 here. Results also in torch/logs/guide_use_<time>.json.
# Run it inside tmux if your VPN drops: losing the SSH session kills an srun --pty job.
#
# ANY GPU: gpu:1 with no constraint (set CONSTRAINT=l40s etc. to pin one). The card is checked
# first: anything below compute capability 7.5 has no kernels in the installed torch build.

ACCOUNT="${ACCOUNT:-torch_pr_89_tandon_advanced}"
GPU="${GPU:-1}"
CONSTRAINT="${CONSTRAINT:-}"
TIME="${TIME:-01:00:00}"
MEM="${MEM:-32G}"
CPUS="${CPUS:-4}"
IMMAP_ROOT="${IMMAP_ROOT:-$HOME/scratch/ee2178/ImMAP}"
ROOT="${ROOT:-trained_nets/guided_recon/knee/s4_noise-vlow}"
RUNS="${RUNS:-lggs_guide_w_R8_g1 lggs_guide_w_ssdu_R8_g1 tgvn_c6_R8_g1 tgvn_c6_ssdu_R8_g1 lpdsnet_R8 lpdsnet_ssdu_R8}"

if [ -z "$SLURM_JOB_ID" ]; then
    echo "requesting a GPU: account $ACCOUNT, gpu $GPU${CONSTRAINT:+ ($CONSTRAINT)}, time $TIME (waits in the queue)"
    exec srun --account="$ACCOUNT" --gres="gpu:$GPU" ${CONSTRAINT:+--constraint="$CONSTRAINT"} \
        --cpus-per-task="$CPUS" --mem="$MEM" --time="$TIME" --job-name=guide-use --pty \
        bash "$(readlink -f "$0")" "$@"
fi

# `source ~/.bashrc` and `conda activate` trip over `set -u`; activate first (as torch/*.sbatch).
source ~/.bashrc
conda activate gcdl || { echo "conda activate gcdl failed" >&2; exit 1; }
export PYTHONNOUSERSITE=1
export IMMAP_CLUSTER=torch

cd "$IMMAP_ROOT" || { echo "no such dir: $IMMAP_ROOT" >&2; exit 1; }
mkdir -p torch/logs

# the runs that are actually here (a run not trained yet is skipped, not fatal)
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
JSON="torch/logs/guide_use_$(date +%Y%m%d_%H%M%S).json"
echo "host: $(hostname)   job: $SLURM_JOB_ID   $(date '+%F %T')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python - <<'PY' || exit 1
import sys, torch
if not torch.cuda.is_available():
    sys.exit("no CUDA device visible")
cc = torch.cuda.get_device_capability(0)
if cc < (7, 5):
    sys.exit(f"{torch.cuda.get_device_name(0)} is compute capability {cc[0]}.{cc[1]}: this torch "
             f"build has no kernels for it. Rerun, or pin a card: CONSTRAINT=l40s bash torch/guide_use.sh")
PY
python -u scripts/diagnose_guide_use.py "${DIRS[@]}" "${N_ARG[@]}" --json "$JSON" "$@"
