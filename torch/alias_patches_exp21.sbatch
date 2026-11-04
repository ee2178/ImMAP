#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=torch_pr_89_tandon_advanced
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=03:00:00
#SBATCH --job-name=AliasPatches21
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%j.out
#SBATCH --error=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%j.err
#
# EXP21 NETS ON THE PATCHES THE cf32 ALIAS SCAN FLAGGED (as torch/alias_patches.sbatch did
# for exp19). Each exp21 cell sits beside its exp18 twin, with RAM as the reference.
#
# Unlike alias_patches.sbatch, EVERY net is dumped again here with --flags, the twins and RAM
# included. A dump's noise draw depends on its volume list, so a net dumped on the flagged
# slices sees different noise than one dumped whole; dumping all of them the same way puts
# the cg cells and their twins on the same noise. 7 nets x the flagged slices, minutes each.
#
# Safe while exp21 trains: it only reads net.ckpt, and a checkpoint caught mid-save is
# skipped, so read the log. patches_R<R>.txt records each dump's training step: the exp21
# cells are younger than their twins until they finish, so rerun this then.
#
#   sbatch torch/alias_patches_exp21.sbatch                 # knee R=8, every flagged slice
#   TOP=20 sbatch torch/alias_patches_exp21.sbatch          # only the top 20 flagged slices
#
# Options, with the defaults below:
#   ANATOMY R FLAGS       which scan to follow (exp19 used the same FLAGS)
#   NETS                  cells to dump and draw, in panel order, without _R<R>. The first
#                         supplies the reference and zero-filled image
#   DUMP_OUT              where the flagged-slice dumps go (overwritten on every run)
#   VAL_KSPACE            must match the kspace_type column of FLAGS
#
# Outputs: OUT/patch_R<R>_<rank>_<volume>_s<slice>.png, patches_R<R>.csv, patches_R<R>.txt.
#
# SUBMIT FROM THE REPO ROOT.

ANATOMY="${ANATOMY:-knee}"
R="${R:-8}"
FLAGS="${FLAGS:-results/alias_slices_cf32_all/${ANATOMY}/slices_R${R}.csv}"
NETS="${NETS:-lpdsnetcf32 lpdsnetcgcf32 varnetmapscf32 varnetmapscgcf32 mg81v6vtxAnestUcf32 mg81v6vtxAnestUcgcf32 ramcf32}"
DUMP_OUT="${DUMP_OUT:-trained_nets/mg_flagged_exp21}"
OUT="${OUT:-results/alias_patches_exp21/${ANATOMY}}"
VAL_KSPACE="${VAL_KSPACE:-measurement_awgn}"
TOP="${TOP:-0}"

for _c in "${IMMAP_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" "$HOME/scratch/ee2178/ImMAP"; do
    if [ -n "${_c}" ] && [ -f "${_c}/scripts/alias_patches.py" ]; then
        IMMAP_ROOT="${_c}"
        break
    fi
done
if [ ! -f "${IMMAP_ROOT:-}/scripts/alias_patches.py" ]; then
    echo "[patches21] cannot locate scripts/alias_patches.py; submit from the repo root or set IMMAP_ROOT" >&2
    exit 1
fi

# Activate first, then the guards (conda activate trips set -u / set -e).
source ~/.bashrc
conda activate gcdl
cd "${IMMAP_ROOT}"
set -eo pipefail

if [ ! -f "${FLAGS}" ]; then
    echo "[patches21] no ${FLAGS} -- run torch/alias_check.sbatch first, or set FLAGS" >&2
    exit 1
fi

_only=()
for _n in ${NETS}; do _only+=("${_n}_R${R}"); done

echo "[patches21] ${ANATOMY} R=${R}: nets='${NETS}' flags=${FLAGS} $(date '+%F %T')  node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python scripts/dump_eval.py \
    --runs "trained_nets/mg_recon/${ANATOMY}" \
    --exact --only "${_only[@]}" \
    --flags "${FLAGS}" --flags-top "${TOP}" \
    --val-kspace-type "${VAL_KSPACE}" \
    --metrics psnr ssim nrmse \
    --out "${DUMP_OUT}/${ANATOMY}"

# Only the fresh tree: every method then comes from a dump with the same noise.
python scripts/alias_patches.py \
    --flags "${FLAGS}" --top "${TOP}" \
    --runs "${DUMP_OUT}/${ANATOMY}" \
    --R "${R}" \
    --methods ${NETS} \
    --out "${OUT}"

echo "[patches21] done $(date '+%F %T') -> ${OUT}"
