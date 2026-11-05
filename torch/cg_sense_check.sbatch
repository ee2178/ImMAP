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
#SBATCH --job-name=CGSenseCheck
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%j.out
#SBATCH --error=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%j.err
#
# WHERE DO THE FLAGGED GHOSTS COME FROM? (2026-11-05, scripts/cg_sense_check.py). Every net
# dumped on the cf32 alias scan's flagged slices with the same noise, and beside them two
# Tikhonov CG-SENSE solves of the same measurement: through the operator's online maps
# (what the nets are given) and through the STORED maps (what the reference is made with).
# Each flagged slice is classed maps / data / prior by which solves leave its ghost.
#
#   sbatch torch/cg_sense_check.sbatch                    # knee R8, every flagged slice
#   LAM=1e-2 OUT=results/cg_sense_check/knee_R8_lam1e-2 sbatch torch/cg_sense_check.sbatch
#
# Options, with the defaults below:
#   ANATOMY R FLAGS     which scan to follow (exp19 / exp21 / the scale sweep used the same)
#   NETS                cells to dump and compare, without _R<R>
#   LAM                 the Tikhonov weight (1e-3, as mg_recon_dump.sbatch's CG_SENSE_LAM)
#   DUMP_OUT            where the dumps go (overwritten on every run)
#   TOP                 only the top N flagged slices (0 = all)
#
# Safe while exp21 / exp22 train: it only reads net.ckpt. SUBMIT FROM THE REPO ROOT.

ANATOMY="${ANATOMY:-knee}"
R="${R:-8}"
FLAGS="${FLAGS:-results/alias_slices_cf32_all/${ANATOMY}/slices_R${R}.csv}"
NETS="${NETS:-lpdsnetcf32 mg81v6vtxAnestUcf32 varnetmapscf32 ramcf32}"
LAM="${LAM:-1e-3}"
DUMP_OUT="${DUMP_OUT:-trained_nets/mg_flagged_cgsense}"
OUT="${OUT:-results/cg_sense_check/${ANATOMY}_R${R}}"
VAL_KSPACE="${VAL_KSPACE:-measurement_awgn}"
TOP="${TOP:-0}"

for _c in "${IMMAP_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" "$HOME/scratch/ee2178/ImMAP"; do
    if [ -n "${_c}" ] && [ -f "${_c}/scripts/cg_sense_check.py" ]; then
        IMMAP_ROOT="${_c}"
        break
    fi
done
if [ ! -f "${IMMAP_ROOT:-}/scripts/cg_sense_check.py" ]; then
    echo "[cgcheck] cannot locate scripts/cg_sense_check.py; submit from the repo root or set IMMAP_ROOT" >&2
    exit 1
fi

# Activate first, then the guards (conda activate trips set -u / set -e).
source ~/.bashrc
conda activate gcdl
cd "${IMMAP_ROOT}"
set -eo pipefail

if [ ! -f "${FLAGS}" ]; then
    echo "[cgcheck] no ${FLAGS} -- run torch/alias_check.sbatch first, or set FLAGS" >&2
    exit 1
fi

_only=()
for _n in ${NETS}; do _only+=("${_n}_R${R}"); done

echo "[cgcheck] ${ANATOMY} R=${R}: nets='${NETS}' lam=${LAM} flags=${FLAGS} $(date '+%F %T') node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python scripts/dump_eval.py     --runs "trained_nets/mg_recon/${ANATOMY}"     --exact --only "${_only[@]}"     --flags "${FLAGS}" --flags-top "${TOP}"     --val-kspace-type "${VAL_KSPACE}"     --metrics psnr ssim nrmse     --cg-sense-lam "${LAM}" --cg-sense-stored     --out "${DUMP_OUT}/${ANATOMY}"

python scripts/cg_sense_check.py     --runs "${DUMP_OUT}/${ANATOMY}"     --flags "${FLAGS}" --R "${R}"     --methods ${NETS}     --out "${OUT}"

echo "[cgcheck] done $(date '+%F %T') -> ${OUT}"
