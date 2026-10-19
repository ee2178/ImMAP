#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=torch_pr_89_tandon_advanced
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=08:00:00
#SBATCH --job-name=MainEval
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%A_%a.out
#SBATCH --error=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%A_%a.err
#SBATCH --array=0-15
#
# PER-VOLUME METRICS FOR THE MAIN COMPARISON (scripts/eval_per_volume.py):
# PSNR, SSIM, NRMSE, LPIPS on every val volume, for the significance tests in
# scripts/wilcoxon_main.py. One array task per RUN:
#
#    0- 3  brain R8   varnetmaps lpdsnet mg81v6 mglpds
#    4- 7  brain R12  varnetmaps lpdsnet mg81v6 mglpds
#    8-11  knee  R8   varnetmaps lpdsnet mg81v6 mg169v6
#   12-15  knee  R12  varnetmaps lpdsnet mg81v6 mg169v6
#
# Brain's MGLPDS-Big is `mglpds` (no full-length brain mg169v6 exists); knee's
# is `mg169v6`. Slices: each run's training range (brain 0:8, knee 12:25).
#
# BEFORE THE FIRST SUBMIT, on a LOGIN node (compute nodes have no route out, and
# LPIPS needs pretrained AlexNet weights in ~/.cache/torch):
#   conda activate gcdl
#   python -c "from evaluation.metrics import warm_up; warm_up(['lpips'], 'cpu')"
#
# Submit from the repo root:
#   sbatch torch/main_eval.sbatch                  # all 16
#   sbatch --array=8-15 torch/main_eval.sbatch     # knee only
#
# RESUMABLE: rows are written as they go to results/main/<anatomy>/<run>.slices.csv;
# resubmitting the same task finishes the volumes that are missing. A newer
# checkpoint or a different sigma/seed/slices starts that run over.
#
# Then (login node, or the laptop after copying results/main/ down -- only CSVs):
#   python scripts/eval_per_volume.py --combine
#   python scripts/wilcoxon_main.py
#   python scripts/wilcoxon_main.py --ref MGLPDS-Big

RUNS=(
    brain/varnetmaps_R8  brain/lpdsnet_R8  brain/mg81v6_R8  brain/mglpds_R8
    brain/varnetmaps_R12 brain/lpdsnet_R12 brain/mg81v6_R12 brain/mglpds_R12
    knee/varnetmaps_R8   knee/lpdsnet_R8   knee/mg81v6_R8   knee/mg169v6_R8
    knee/varnetmaps_R12  knee/lpdsnet_R12  knee/mg81v6_R12  knee/mg169v6_R12
)
# Rediscretized multigrid runs LAUNCHED BEFORE 2026-10-07 trained with the
# Galerkin coarse operator on slice sizes that do not divide by the stride
# (operators/coarse.py); they must be evaluated the same way. Of the runs above
# that is brain mg81v6_R12 (exp9, 2026-10-02). `mglpds` is Galerkin throughout;
# the knee runs and brain mg81v6_R8 were launched after the change.
EMBEDDED_OFF="${EMBEDDED_OFF:-brain/mg81v6_R12}"

METRICS="${METRICS:-psnr ssim nrmse lpips}"
SEED="${SEED:-1234}"
SIGMA="${SIGMA:-}"                     # empty = each run's own val_noise_std
OUT="${OUT:-results/main}"
WORKERS="${WORKERS:-6}"
EXTRA="${EXTRA:-}"                     # e.g. EXTRA="--n-volumes 3" for a probe

for _c in "${IMMAP_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" "$HOME/scratch/ee2178/ImMAP"; do
    if [ -n "${_c}" ] && [ -f "${_c}/scripts/eval_per_volume.py" ]; then
        IMMAP_ROOT="${_c}"
        break
    fi
done
if [ ! -f "${IMMAP_ROOT:-}/scripts/eval_per_volume.py" ]; then
    echo "[eval] cannot locate scripts/eval_per_volume.py; submit from the repo root" \
         "or set IMMAP_ROOT" >&2
    exit 1
fi

# Activate first, then the guards (see the no-set-euo note in the other launchers).
source ~/.bashrc
conda activate gcdl
cd "${IMMAP_ROOT}"

set -eo pipefail
: "${SLURM_ARRAY_TASK_ID:?must be run as an array job}"

if [ "${SLURM_ARRAY_TASK_ID}" -ge "${#RUNS[@]}" ]; then
    echo "[eval] task ${SLURM_ARRAY_TASK_ID} >= ${#RUNS[@]} runs -- fix --array" >&2
    exit 1
fi
ENTRY="${RUNS[${SLURM_ARRAY_TASK_ID}]}"
ANATOMY="${ENTRY%%/*}"
RUN="${ENTRY#*/}"

for e in ${EMBEDDED_OFF}; do
    if [ "${e}" = "${ENTRY}" ]; then
        export IMMAP_REDISCRETIZE_EMBEDDED=0
    fi
done

echo "[eval] ${ENTRY}  metrics='${METRICS}' seed=${SEED}" \
     "IMMAP_REDISCRETIZE_EMBEDDED=${IMMAP_REDISCRETIZE_EMBEDDED:-1}"
echo "[eval] $(date '+%F %T')  node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python scripts/eval_per_volume.py \
    --anatomy "${ANATOMY}" --run "${RUN}" \
    --metrics ${METRICS} \
    --seed "${SEED}" \
    --workers "${WORKERS}" \
    --out "${OUT}" \
    ${SIGMA:+--sigma "${SIGMA}"} \
    ${EXTRA}

echo "[eval] ${ENTRY} done $(date '+%F %T')"
