#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=torch_pr_89_tandon_advanced
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --job-name=ESPIRiT
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --array=0-8
#
# Replace the WALSH sensitivity maps with ESPIRiT.
#
#   0      brain  val     (one task -- the val split is small)
#   1-8    brain  train   (8 shards of the file list)
#
# Hyperparameters are the ones chosen in notebooks/espirit_brain.ipynb:
# acs=(20,20), kernel_size=8, thresh_eig=0.95.
#
# WHAT THIS REWRITES, AND WHY IT IS BOTH DATASETS
# -----------------------------------------------
# `smaps` AND `image`. `image` is the coil-combined ground truth
# `sum_c conj(s_c) c_c` -- a function of the maps. Keeping the old `image`
# alongside new maps would give a file whose ground truth is not what its own
# operator produces, and since these cells train on `kspace_type: "simulated"`
# (image -> Sense -> Fourier -> mask), every measurement would be built from an
# inconsistent pair. Nothing about that fails a shape check; it just trains on
# the wrong problem. See scripts/make_espirit_smaps.py.
#
# WHY ESPIRiT AT ALL
# ------------------
#   * Walsh has no exact zeros outside the anatomy, so the organ mask
#     (`smaps.abs().sum(0) > 0` in datasets/fastmri/loader.py, a strict float
#     test) is true almost everywhere and `use_organ_mask` silently does
#     nothing. ESPIRiT's eigenvalue threshold gives a hard support.
#   * `operators/noise.py::mri_awgn` ASSUMES unit-RSS maps -- that is what makes
#     sigma the noise std of the coil-combined adjoint, which is the quantity
#     the noise-adaptive thresholds are calibrated against. ESPIRiT's
#     power-method eigenvector is unit-norm per pixel; Walsh is not.
#
# NON-DESTRUCTIVE BY DEFAULT
# --------------------------
# Writes to `<smap_root>_espirit`, leaving the Walsh maps untouched -- they are
# the input to every run already trained, and deleting them would make those
# runs unreproducible. After this finishes, point the configs at the new root
# (scripts/make_mg_recon_configs.py, ANATOMIES[...]["smap_root"]) and
# regenerate. Set OVERWRITE=1 below only if you have decided the old maps are
# genuinely dead.
#
# Each task skips volumes already present in the output, so a timed-out or
# failed shard can simply be resubmitted. Files are written to `.partial` and
# renamed, so an interrupted write is never mistaken for a finished one.
#
# BEFORE SUBMITTING
#     python scripts/make_espirit_smaps.py --anatomy brain --split val --dry-run
#
# SUBMIT FROM THE REPO ROOT -- `#SBATCH --output=logs/...` resolves against the
# SUBMIT directory, so from torch/ the job dies before the script runs.

ANATOMY=brain
ACS=20
KERNEL=8
THRESH_EIG=0.95
OVERWRITE=0                          # 1 = write over the Walsh maps in place
CHUNK=4                              # slices per ESPIRiT call; lower if OOM

N_TRAIN_SHARDS=8

case "${SLURM_ARRAY_TASK_ID:-0}" in
    0) SPLIT=val;   SHARD=0; NSHARD=1 ;;
    *) SPLIT=train; SHARD=$(( SLURM_ARRAY_TASK_ID - 1 )); NSHARD=${N_TRAIN_SHARDS} ;;
esac

if [ "${SLURM_ARRAY_TASK_ID:-0}" -gt "${N_TRAIN_SHARDS}" ]; then
    echo "[espirit] task ${SLURM_ARRAY_TASK_ID} is outside 0-${N_TRAIN_SHARDS}." >&2
    exit 1
fi

# sbatch runs a COPY of this file from the node's spool dir, so a
# $BASH_SOURCE-relative path looks for the repo next to that copy.
for _c in "${IMMAP_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" "$HOME/scratch/ee2178/ImMAP"; do
    if [ -n "${_c}" ] && [ -f "${_c}/scripts/make_espirit_smaps.py" ]; then
        IMMAP_ROOT="${_c}"
        break
    fi
done
if [ ! -f "${IMMAP_ROOT:-}/scripts/make_espirit_smaps.py" ]; then
    echo "[espirit] cannot locate the ImMAP checkout (tried IMMAP_ROOT," \
         "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset}, ~/scratch/ee2178/ImMAP)." >&2
    echo "[espirit] submit from the repo root, or: IMMAP_ROOT=/path sbatch $0" >&2
    exit 1
fi

# NOT `set -euo`: -u breaks conda activate's use of unset variables.
set -eo pipefail

source ~/.bashrc
conda activate gcdl
cd "${IMMAP_ROOT}"
mkdir -p logs

echo "[espirit] ${ANATOMY} ${SPLIT} shard ${SHARD}/${NSHARD}"

python scripts/make_espirit_smaps.py \
    --anatomy "${ANATOMY}" \
    --split "${SPLIT}" \
    --acs "${ACS}" \
    --kernel-size "${KERNEL}" \
    --thresh-eig "${THRESH_EIG}" \
    --chunk "${CHUNK}" \
    --shard "${SHARD}" \
    --num-shards "${NSHARD}" \
    $( [ "${OVERWRITE}" = "1" ] && echo --overwrite )
