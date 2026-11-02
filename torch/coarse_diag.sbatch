#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=torch_pr_89_tandon_advanced
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=06:00:00
#SBATCH --job-name=CoarseDiag
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%A_%a.out
#SBATCH --error=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%A_%a.err
#SBATCH --array=0
#
# CORRECTION SPECTRA AND ALPHA KNOCKOUTS OF A TRAINED MULTIGRID NET (scripts/coarse_diag.py):
# how much of each V-cycle's correction is above the coarse grid's Nyquist, x path against
# z path, and what PSNR and rho do with each path zeroed at inference.
#
#   sbatch torch/coarse_diag.sbatch                 # exp18's knee R8 net, now
#
# After exp19, its three nets against that one (the array covers the four; a net with
# no z path reports none):
#   RUNS="knee/mg81v6vtxAnestUcf32_R8 knee/mg81v6vtxAnestUgalcf32_R8 knee/mg81v6vtxAnestUfoldcf32_R8 knee/mg81v6vtxAnestUaz0cf32_R8" \
#       sbatch --array=0-3 torch/coarse_diag.sbatch
#
# The net as trained against another coarse operator, same weights, its own rows:
#   COARSE_OP=galerkin sbatch torch/coarse_diag.sbatch
#
# OUT/knockouts.csv, spectra.csv and summary.txt are merged after every task, under a
# lock, so the last task leaves the complete set. FLAGS is the alias scan's ranked slices;
# when it is not there the random slices still run. About 80 slices at 8 forwards each;
# time not measured on the cluster.
#
# SUBMIT FROM THE REPO ROOT.

_RUNS_GIVEN="${RUNS:-}"
RUNS=(knee/mg81v6vtxAnestUcf32_R8)
if [ -n "${_RUNS_GIVEN}" ]; then
    read -r -a RUNS <<< "${_RUNS_GIVEN}"
fi
OUT="${OUT:-results/coarse_diag}"
FLAGS="${FLAGS:-results/alias_slices_cf32_all/knee/slices_R8.csv}"
COARSE_OP="${COARSE_OP:-}"
EXTRA="${EXTRA:-}"                     # e.g. EXTRA="--sample-volumes 2" for a probe

for _c in "${IMMAP_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" "$HOME/scratch/ee2178/ImMAP"; do
    if [ -n "${_c}" ] && [ -f "${_c}/scripts/coarse_diag.py" ]; then
        IMMAP_ROOT="${_c}"
        break
    fi
done
if [ ! -f "${IMMAP_ROOT:-}/scripts/coarse_diag.py" ]; then
    echo "[diag] cannot locate scripts/coarse_diag.py; submit from the repo root or set IMMAP_ROOT" >&2
    exit 1
fi

# Activate first, then the guards (conda activate trips set -u / set -e).
source ~/.bashrc
conda activate gcdl
cd "${IMMAP_ROOT}"
set -eo pipefail

: "${SLURM_ARRAY_TASK_ID:?must be run as an array job (sbatch sets SLURM_ARRAY_TASK_ID)}"
ENTRY="${RUNS[${SLURM_ARRAY_TASK_ID}]:-}"
if [ -z "${ENTRY}" ]; then
    echo "[diag] task ${SLURM_ARRAY_TASK_ID}: --array must be within 0-$(( ${#RUNS[@]} - 1 ))" >&2
    exit 1
fi
RUN_DIR="trained_nets/mg_recon/${ENTRY}"
if [ ! -f "${RUN_DIR}/net.ckpt" ]; then
    echo "[diag] ${RUN_DIR} has no net.ckpt" >&2
    exit 1
fi
if [ ! -f "${FLAGS}" ]; then
    echo "[diag] no ${FLAGS}: random slices only" >&2
fi

mkdir -p "${OUT}"
echo "[diag] ${ENTRY} ${COARSE_OP:+coarse_op=${COARSE_OP}} out=${OUT} $(date '+%F %T')  node=$(hostname)"
python scripts/coarse_diag.py --run "${RUN_DIR}" --out "${OUT}" \
    $( [ -f "${FLAGS}" ] && echo --flags "${FLAGS}" ) \
    ${COARSE_OP:+--coarse-op "${COARSE_OP}"} ${EXTRA}
flock "${OUT}/.combine.lock" python scripts/coarse_diag.py --out "${OUT}" --combine
echo "[diag] done $(date '+%F %T') -- read ${OUT}/summary.txt"
