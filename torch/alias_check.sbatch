#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=torch_pr_89_tandon_advanced
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --job-name=AliasCheck
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%A_%a.out
#SBATCH --error=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%A_%a.err
#SBATCH --array=0-1
#
# Flag dumped slices with residual aliasing (scripts/alias_check.py --top-slices):
# a slice is flagged when E = HP(|recon| - |ref|) follows T = HP(|zero-filled|)
# (rho > its cutoff) somewhere the residual is strong (strength > MIN_STRENGTH). CPU
# only; it reads the HDF5 of scripts/dump_eval.py.
#
# rho runs higher on knee, so the cutoff is per anatomy: brain FLAG_RHO, knee
# KNEE_FLAG_RHO per R. Both were chosen by eye on a few local volumes (2026-10-30),
# so recheck them on the measured dumps.
#
#   0  brain
#   1  knee
#
#   sbatch torch/alias_check.sbatch                                     # measured dumps
#   DUMPS=trained_nets/mg_recon OUT=results/alias_slices_simulated sbatch torch/alias_check.sbatch
#   METHODS="varnetmaps lpdsnet mg81v6 mg169v6 ram" sbatch torch/alias_check.sbatch
#
# DUMPS holds <anatomy>/<method>_R<R>/eval_dump/. The default is the measured tree of
# torch/dump_full_val.sbatch; trained_nets/mg_recon mixes simulated and measured dumps
# (see the kspace_type column). A volume is scanned only if every method in METHODS
# has it, and a method with no dump at an R skips that R.
#
# TIME was about a minute per volume on a laptop for three methods at both R, so the
# full brain val set may need most of the limit; not measured on the cluster.
#
# OUT/<anatomy>/ gets flags_R<R>.txt (counts and top flagged slices),
# slices_R<R>.csv (every slice, ranked) and an image per flagged slice, up to
# TOP_SLICES per method. Pull them down with
#   scp -r torch:scratch/ee2178/ImMAP/results/alias_slices results/alias_slices_torch
#
# SUBMIT FROM THE REPO ROOT.

ANATOMIES=(brain knee)
DUMPS="${DUMPS:-trained_nets/mg_recon_measured}"
OUT="${OUT:-results/alias_slices}"
METHODS="${METHODS:-varnetmaps lpdsnet mg81v6}"
TOP_SLICES="${TOP_SLICES:-100}"
MIN_STRENGTH="${MIN_STRENGTH:-2.5}"
FLAG_RHO="${FLAG_RHO:-0.75}"
KNEE_FLAG_RHO="${KNEE_FLAG_RHO:-8:0.85,12:0.80}"

: "${SLURM_ARRAY_TASK_ID:?must be run as an array job (sbatch sets SLURM_ARRAY_TASK_ID)}"
ANATOMY="${ANATOMIES[${SLURM_ARRAY_TASK_ID}]:-}"
if [ -z "${ANATOMY}" ]; then
    echo "[alias] task ${SLURM_ARRAY_TASK_ID}: --array must be within 0-1" >&2
    exit 1
fi

for _c in "${IMMAP_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" "$HOME/scratch/ee2178/ImMAP"; do
    if [ -n "${_c}" ] && [ -f "${_c}/scripts/alias_check.py" ]; then
        IMMAP_ROOT="${_c}"
        break
    fi
done
if [ ! -f "${IMMAP_ROOT:-}/scripts/alias_check.py" ]; then
    echo "[alias] cannot locate scripts/alias_check.py; submit from the repo root or set IMMAP_ROOT" >&2
    exit 1
fi

# Activate first, then the guards (conda activate trips set -u / set -e).
source ~/.bashrc
conda activate gcdl
cd "${IMMAP_ROOT}"
set -eo pipefail

BY_R=""
[ "${ANATOMY}" = "knee" ] && BY_R="${KNEE_FLAG_RHO}"

if [ ! -d "${DUMPS}/${ANATOMY}" ]; then
    echo "[alias] no ${DUMPS}/${ANATOMY} -- dump first (torch/dump_full_val.sbatch), or set DUMPS" >&2
    exit 1
fi

echo "[alias] ${ANATOMY}: dumps=${DUMPS} methods='${METHODS}' $(date '+%F %T')  node=$(hostname)"
python scripts/alias_check.py \
    --runs "${DUMPS}/${ANATOMY}" \
    --methods ${METHODS} \
    --top-slices "${TOP_SLICES}" \
    --min-strength "${MIN_STRENGTH}" \
    --flag-rho "${FLAG_RHO}" \
    ${BY_R:+--flag-rho-by-r "${BY_R}"} \
    --out "${OUT}/${ANATOMY}"
echo "[alias] ${ANATOMY} done $(date '+%F %T')"
