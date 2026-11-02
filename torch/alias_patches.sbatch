#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=torch_pr_89_tandon_advanced
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --job-name=AliasPatches
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%j.out
#SBATCH --error=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%j.err
#
# NEW NETS ON THE PATCHES AN EARLIER ALIAS SCAN FLAGGED. Runs NEW_NETS on the flagged
# slices of FLAGS only (scripts/dump_eval.py --flags), then draws each flagged patch with
# CORE_NETS from their existing dumps beside them (scripts/alias_patches.py). The core
# nets are not run again. Safe while the new nets train: it only reads their net.ckpt and
# writes under DUMP_OUT, though a checkpoint caught mid-save is skipped, so read the log.
#
#   sbatch torch/alias_patches.sbatch                    # exp19's nets, knee R=8
#   TOP=20 sbatch torch/alias_patches.sbatch             # only the top 20 flagged slices
#
# Options, with the defaults below:
#   ANATOMY R FLAGS       which scan to follow
#   NEW_NETS              cells to run, without _R<R>
#   CORE_NETS             cells already dumped under CORE_DUMPS. The first must be a complete
#                         dump: the reference and zero-filled image come from it
#   CORE_DUMPS            the tree the scan read (alias_check.sbatch's DUMPS)
#   VAL_KSPACE            must match the kspace_type column of FLAGS
#
# Outputs: OUT/patch_R<R>_<rank>_<volume>_s<slice>.png, patches_R<R>.csv, patches_R<R>.txt.
# The .txt lists each dump's training step and warns if the k-space types differ.
#
# SUBMIT FROM THE REPO ROOT.

ANATOMY="${ANATOMY:-knee}"
R="${R:-8}"
FLAGS="${FLAGS:-results/alias_slices_cf32_all/${ANATOMY}/slices_R${R}.csv}"
NEW_NETS="${NEW_NETS:-mg81v6vtxAnestUgalcf32 mg81v6vtxAnestUfoldcf32 mg81v6vtxAnestUaz0cf32}"
CORE_NETS="${CORE_NETS:-lpdsnetcf32 ramcf32 mg81v6vtxAnestUcf32}"
CORE_DUMPS="${CORE_DUMPS:-trained_nets/mg_recon_measured}"
DUMP_OUT="${DUMP_OUT:-trained_nets/mg_flagged}"
OUT="${OUT:-results/alias_patches_exp19/${ANATOMY}}"
VAL_KSPACE="${VAL_KSPACE:-measurement_awgn}"
TOP="${TOP:-0}"

for _c in "${IMMAP_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" "$HOME/scratch/ee2178/ImMAP"; do
    if [ -n "${_c}" ] && [ -f "${_c}/scripts/alias_patches.py" ]; then
        IMMAP_ROOT="${_c}"
        break
    fi
done
if [ ! -f "${IMMAP_ROOT:-}/scripts/alias_patches.py" ]; then
    echo "[patches] cannot locate scripts/alias_patches.py; submit from the repo root or set IMMAP_ROOT" >&2
    exit 1
fi

# Activate first, then the guards (conda activate trips set -u / set -e).
source ~/.bashrc
conda activate gcdl
cd "${IMMAP_ROOT}"
set -eo pipefail

if [ ! -f "${FLAGS}" ]; then
    echo "[patches] no ${FLAGS} -- run torch/alias_check.sbatch first, or set FLAGS" >&2
    exit 1
fi

_only=()
for _n in ${NEW_NETS}; do _only+=("${_n}_R${R}"); done

echo "[patches] ${ANATOMY} R=${R}: new='${NEW_NETS}' core='${CORE_NETS}' flags=${FLAGS} $(date '+%F %T')  node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python scripts/dump_eval.py \
    --runs "trained_nets/mg_recon/${ANATOMY}" \
    --exact --only "${_only[@]}" \
    --flags "${FLAGS}" --flags-top "${TOP}" \
    --val-kspace-type "${VAL_KSPACE}" \
    --metrics psnr ssim nrmse \
    --out "${DUMP_OUT}/${ANATOMY}"

python scripts/alias_patches.py \
    --flags "${FLAGS}" --top "${TOP}" \
    --runs "${CORE_DUMPS}/${ANATOMY}" "${DUMP_OUT}/${ANATOMY}" \
    --R "${R}" \
    --methods ${CORE_NETS} ${NEW_NETS} \
    --out "${OUT}"

echo "[patches] done $(date '+%F %T') -> ${OUT}"
