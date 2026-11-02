#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=torch_pr_89_tandon_advanced
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=01:00:00
#SBATCH --job-name=GhostForce
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%j.out
#SBATCH --error=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%j.err
#
# DOES A TRAINED PRIOR TREAT A KNOWN GHOST DIFFERENTLY FROM ANATOMY? (scripts/ghost_force.py)
# Each net in NETS: its fine-level filters and thresholds from net.ckpt, applied to the ghost
# in one dumped slice, at the trained thresholds and multiples of them. The ghost always comes
# from DUMP, so every net is scored on the same artifact.
#
#   sbatch torch/ghost_force.sbatch       # LPDSNet's streak on brain R8, slice 0
#
# Another slice, with `row` and `col` from an alias_check.py slices_R<R>.csv:
#   ANATOMY=knee DUMP=trained_nets/mg_recon_measured/knee/lpdsnetcf32_R8/eval_dump/<volume>.h5 \
#       SLICE=<slice> BOX="<row-15> <row+15> <col-6> <col+6>" \
#       NETS="lpdsnetcf32_R8 mg81v6vtxAnestUcf32_R8" TAG=<name> sbatch torch/ghost_force.sbatch
#
# Outputs: OUT/<TAG>_<net>.txt, also printed to this job's log.
#
# SUBMIT FROM THE REPO ROOT.

ANATOMY="${ANATOMY:-brain}"
DUMP="${DUMP:-}"
SLICE="${SLICE:-0}"
BOX="${BOX:-412 442 128 140}"
NETS="${NETS:-lpdsnet_R8 mg81v6_R8}"
TAG="${TAG:-brain_R8_s0}"
OUT="${OUT:-results/ghost_force}"

for _c in "${IMMAP_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" "$HOME/scratch/ee2178/ImMAP"; do
    if [ -n "${_c}" ] && [ -f "${_c}/scripts/ghost_force.py" ]; then
        IMMAP_ROOT="${_c}"
        break
    fi
done
if [ ! -f "${IMMAP_ROOT:-}/scripts/ghost_force.py" ]; then
    echo "[ghost] cannot locate scripts/ghost_force.py; submit from the repo root or set IMMAP_ROOT" >&2
    exit 1
fi

# Activate first, then the guards (conda activate trips set -u / set -e).
source ~/.bashrc
conda activate gcdl
cd "${IMMAP_ROOT}"
set -eo pipefail

# Dumps live beside the checkpoints or under mg_recon_measured, depending on the job that wrote them.
if [ -z "${DUMP}" ]; then
    for _root in trained_nets/mg_recon_measured trained_nets/mg_recon; do
        _d="${_root}/brain/lpdsnet_R8/eval_dump/file_brain_AXT2_200_2000182.h5"
        if [ -f "${_d}" ]; then
            DUMP="${_d}"
            break
        fi
    done
fi
if [ ! -f "${DUMP}" ]; then
    echo "[ghost] no dump '${DUMP}' -- set DUMP" >&2
    exit 1
fi

echo "[ghost] dump=${DUMP} slice=${SLICE} box=${BOX} nets='${NETS}' $(date '+%F %T')  node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

for _net in ${NETS}; do
    _ckpt="trained_nets/mg_recon/${ANATOMY}/${_net}/net.ckpt"
    if [ ! -f "${_ckpt}" ]; then
        echo "[ghost] no ${_ckpt} -- set ANATOMY or NETS" >&2
        exit 1
    fi
    echo; echo "================ ${_net} ================"
    python scripts/ghost_force.py --dump "${DUMP}" --slice "${SLICE}" --box ${BOX} \
        --ckpt "${_ckpt}" --out "${OUT}/${TAG}_${_net}"
done
echo "[ghost] done $(date '+%F %T') -> ${OUT}"
