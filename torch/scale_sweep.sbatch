#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=torch_pr_89_tandon_advanced
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --job-name=ScaleSweep
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%j.out
#SBATCH --error=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%j.err
#
# IS ALIASING SENSITIVE TO INTENSITY? (2026-11-05, scripts/scale_sweep.py). Each trained net
# on (c y, c sigma), output / c, on the cf32 alias scan's flagged slices and as many random
# unflagged ones: same noise, operator and SNR, only the absolute intensity changes. Also
# compares RAM's signal level s between flagged and control slices. No training; reads
# net.ckpt only, so it is safe while exp21 trains.
#
# VarNet normalizes inside its U-Nets and RAM is close to scale-equivariant, so their curves
# should be near flat: they are the controls for the LPDS / MGLPDS curves.
#
#   sbatch torch/scale_sweep.sbatch                       # knee R8, top 40 flagged + 40 controls
#   TOP=0 sbatch torch/scale_sweep.sbatch                 # every flagged slice
#   SCALES="0.125 0.25 0.5 1 2 4 8" sbatch torch/scale_sweep.sbatch
#
# Outputs: OUT/summary.txt, OUT/sweep.csv, OUT/sweep.png, and the dumps under OUT/<run>/c<c>/.
# To re-plot without the GPU work: python scripts/scale_sweep.py ... --rescore
#
# SUBMIT FROM THE REPO ROOT.

ANATOMY="${ANATOMY:-knee}"
R="${R:-8}"
FLAGS="${FLAGS:-results/alias_slices_cf32_all/${ANATOMY}/slices_R${R}.csv}"
NETS="${NETS:-lpdsnetcf32 mg81v6vtxAnestUcf32 varnetmapscf32 ramcf32}"
SCALES="${SCALES:-0.25 0.5 1 2 4}"
TOP="${TOP:-40}"
CONTROLS="${CONTROLS:-}"                 # empty = as many as flagged
OUT="${OUT:-results/scale_sweep/${ANATOMY}_R${R}}"
VAL_KSPACE="${VAL_KSPACE:-measurement_awgn}"

for _c in "${IMMAP_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" "$HOME/scratch/ee2178/ImMAP"; do
    if [ -n "${_c}" ] && [ -f "${_c}/scripts/scale_sweep.py" ]; then
        IMMAP_ROOT="${_c}"
        break
    fi
done
if [ ! -f "${IMMAP_ROOT:-}/scripts/scale_sweep.py" ]; then
    echo "[sweep] cannot locate scripts/scale_sweep.py; submit from the repo root or set IMMAP_ROOT" >&2
    exit 1
fi

# Activate first, then the guards (conda activate trips set -u / set -e).
source ~/.bashrc
conda activate gcdl
cd "${IMMAP_ROOT}"
set -eo pipefail

if [ ! -f "${FLAGS}" ]; then
    echo "[sweep] no ${FLAGS} -- run torch/alias_check.sbatch first, or set FLAGS" >&2
    exit 1
fi

_only=()
for _n in ${NETS}; do _only+=("${_n}_R${R}"); done
_ctl=()
[ -n "${CONTROLS}" ] && _ctl=(--controls "${CONTROLS}")

echo "[sweep] ${ANATOMY} R=${R}: nets='${NETS}' scales='${SCALES}' $(date '+%F %T') node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python scripts/scale_sweep.py \
    --runs "trained_nets/mg_recon/${ANATOMY}" \
    --only "${_only[@]}" \
    --flags "${FLAGS}" --top "${TOP}" "${_ctl[@]}" \
    --scales ${SCALES} \
    --val-kspace-type "${VAL_KSPACE}" \
    --out "${OUT}"

echo "[sweep] done $(date '+%F %T') -> ${OUT}"
