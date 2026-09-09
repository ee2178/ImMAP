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
#SBATCH --job-name=MGDump
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --array=0-1            # 0 = brain, 1 = knee
#
# Dump per-volume reconstructions for the figure viewer (scripts/dump_eval.py).
#
#   sbatch torch/mg_recon_dump.sbatch                 # brain and knee
#   sbatch --array=0 torch/mg_recon_dump.sbatch       # brain only
#
# One array task per ANATOMY; each dumps every run matching ONLY, so a single
# task covers that anatomy's R=4 and R=8 cells (R comes from each run's own
# config, not from here). Output lands in
#
#     trained_nets/mg_recon/<anatomy>/<cell>/eval_dump/<volume>.h5
#
# beside the checkpoint that produced it. Pull those down and open them with
# figures/viewer.py -- see docs/plotting.md.
#
# This is INFERENCE over a handful of slices, so it is minutes, not hours. The
# 2 h walltime is slack for the model builds, and the GPU is worth asking for
# because the unrolled nets are slow on CPU. Nothing here writes a config or a
# checkpoint: a dump is derived data and rerunning it is always safe.
#
# NOTE: this is a NEW script, not an edit to the frozen Greene launchers. It
# does not source _mg_recon_body.sh -- that body regenerates configs and calls
# train.py, neither of which a dump should ever do.

ANATOMIES=(brain knee)

# Substring filters on the run path, passed to --only. The trailing underscore
# keeps `lpdsnet_` from also selecting a future cell whose name contains it.
ONLY="${ONLY:-lpdsnet_ mglpds_}"

# Slice range and volume count PER ANATOMY, index-matched to ANATOMIES above.
# These are the ranges each cell actually trained on -- brain 0-7 (train
# start_slice 0, end_slice 8) and knee 12-24 (12, 25). Widening them is a
# legitimate thing to do, but know that you are then looking at slices the
# network never saw.
SLICES=("0:8" "12:25")
# Knee is the expensive one: a wider matrix and more slices, so fewer volumes
# for a comparable transfer. Raise once you know the range is the one you want.
NVOLS=(6 4)

METRICS="${METRICS:-psnr ssim nrmse}"   # lpips is deliberately absent: it
                                        # downloads weights on first use, and a
                                        # compute node has no route out
SEED="${SEED:-1234}"
SIGMA="${SIGMA:-}"                      # empty = each run's own val_noise_std
OUT="${OUT:-}"                          # empty = <run>/eval_dump/
WORKERS="${WORKERS:-4}"

# sbatch does NOT execute this file in place: it copies the script to the node's
# spool dir and runs that copy, so $BASH_SOURCE points at the spool and cannot
# locate the repo. Resolve it explicitly, the same way the training launchers
# do. Override with IMMAP_ROOT=/path/to/ImMAP sbatch ...
for _c in "${IMMAP_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" "$HOME/scratch/ee2178/ImMAP"; do
    if [ -n "${_c}" ] && [ -f "${_c}/scripts/dump_eval.py" ]; then
        IMMAP_ROOT="${_c}"
        break
    fi
done
if [ ! -f "${IMMAP_ROOT:-}/scripts/dump_eval.py" ]; then
    echo "[dump] cannot locate scripts/dump_eval.py (tried IMMAP_ROOT," \
         "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset}, ~/scratch/ee2178/ImMAP)." >&2
    echo "[dump] submit from the repo root, or: IMMAP_ROOT=/path/to/ImMAP sbatch $0" >&2
    exit 1
fi
export IMMAP_ROOT

# `source ~/.bashrc` and `conda activate` both trip over `set -u`, and
# `conda activate` returns non-zero often enough that `set -e` before it kills
# the job. Activate first, then turn the guards on.
source ~/.bashrc
conda activate gcdl
cd "${IMMAP_ROOT}"

set -eo pipefail
: "${SLURM_ARRAY_TASK_ID:?must be run as an array job (sbatch sets SLURM_ARRAY_TASK_ID)}"

mkdir -p logs

N_TOTAL="${#ANATOMIES[@]}"
if [ "${SLURM_ARRAY_TASK_ID}" -ge "${N_TOTAL}" ]; then
    echo "[dump] task ${SLURM_ARRAY_TASK_ID} >= ${N_TOTAL} anatomies -- fix --array"
    echo "       (should be 0-$(( N_TOTAL - 1 )): ${ANATOMIES[*]})"
    exit 1
fi

ANATOMY="${ANATOMIES[${SLURM_ARRAY_TASK_ID}]}"
SLICE_RANGE="${SLICES[${SLURM_ARRAY_TASK_ID}]}"
N_VOLUMES="${NVOLS[${SLURM_ARRAY_TASK_ID}]}"
RUNS_ROOT="trained_nets/mg_recon/${ANATOMY}"

if [ ! -d "${RUNS_ROOT}" ]; then
    echo "[dump] ${RUNS_ROOT} does not exist -- nothing has been trained for" \
         "${ANATOMY} yet, or the repo root is wrong (IMMAP_ROOT=${IMMAP_ROOT})." >&2
    exit 1
fi

echo "[dump] ${ANATOMY}: runs=${RUNS_ROOT} only='${ONLY}' slices=${SLICE_RANGE}" \
     "volumes=${N_VOLUMES} metrics='${METRICS}' seed=${SEED}"
echo "[dump] $(date '+%F %T')  node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python scripts/dump_eval.py \
    --runs "${RUNS_ROOT}" \
    --only ${ONLY} \
    --slices "${SLICE_RANGE}" \
    --n-volumes "${N_VOLUMES}" \
    --metrics ${METRICS} \
    --seed "${SEED}" \
    --workers "${WORKERS}" \
    ${SIGMA:+--sigma "${SIGMA}"} \
    ${OUT:+--out "${OUT}/${ANATOMY}"}

echo "[dump] ${ANATOMY} done $(date '+%F %T')"
du -sh "${RUNS_ROOT}"/*/eval_dump 2>/dev/null || true
