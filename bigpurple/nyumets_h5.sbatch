#!/bin/bash
# Build the NYUMets FLAIR/T1/CT1/T2 h5 set. CPU by default; only the cross-study registration
# can use a GPU (REG_DEVICE=cuda, see RUNTIME below) -- everything else is NIfTI I/O.
#
#   mkdir -p logs && sbatch bigpurple/nyumets_h5.sbatch
#
# REBUILD. Set OVERWRITE=1 or existing h5s are silently kept:
#
#   OVERWRITE=1 sbatch bigpurple/nyumets_h5.sbatch
#
# What the current defaults do, none of which is passed below:
#
#   --support store      the common acquisition support is RECORDED, not multiplied into the
#                        pixels (2026-09-22). Zeroing them baked a ragged FOV boundary into the
#                        data, which is where the boundary artefacts came from.
#   --reg-method centroid  a patient's sessions are put onto the index axis of the deepest by
#                        a dz found by masked T1 correlation, then ONE in-plane (dh, dw) that
#                        centres each study's brain mask on the reference's (2026-09-25). It
#                        replaced --reg-method affine, whose header transform re-posed studies
#                        instead of aligning them (~0 correlation after resampling), and the FFT
#                        cross-correlation (--reg-method xcorr), whose peak sometimes locked onto
#                        the wrong lag. --reg-method search finds (dh, dw) by a correlation search
#                        instead (optionally per slice): marginally better, ~5x slower.
#                        Translation only: rotation is NOT undone. Each h5 records reg_offset,
#                        reg_slice_shift and reg_ncc (the quality gate; studies below
#                        --reg-min-ncc are flagged in the log).
#
# RUNTIME. centroid registration costs ~1.5 s per non-reference study on 8-10 CPU threads
# (benchmarked 2026-09-25 at 224x224, depth ~190): ~1.5 h for the cohort on top of the NIfTI
# loading, which cpu_short's 12 h should hold. --reg-method search is ~8 s per study (~7 h); for
# that, give registration a GPU or the longer tier:
#
#   OVERWRITE=1 REG_DEVICE=cuda EXTRA="--reg-method search" OUT=... \
#       sbatch --partition=radiology --gres=gpu:a100:1 bigpurple/nyumets_h5.sbatch
#   OVERWRITE=1 EXTRA="--reg-method search" OUT=... \
#       sbatch --partition=cpu_medium --time=48:00:00 bigpurple/nyumets_h5.sbatch
#
# Build into a FRESH OUT and re-split from it, so no file from an earlier build survives:
#
#   OVERWRITE=1 OUT=~/scratch/datasets/NYUMets_h5_centroid sbatch bigpurple/nyumets_h5.sbatch
#
# EXTRA="..." appends builder flags, e.g. EXTRA="--limit 200" for a shakeout run.
#
# Stored axes stay canonical RAS; ANTERIOR-UP is applied at DISPLAY time
# (visualization.image.set_display_orient, or the configs' display_orient), never to the pixels.
#
# Every NYUMets checkpoint trained before this rebuild saw support-masked pixels and
# translation-only alignment, so none of them transfer -- they have to be retrained.
#
# cpu_short is 12h / 50 nodes on BigPurple -- the shortest tier this fits, so the shortest
# queue. Use cpu_dev (4h, 4 nodes) with --limit for a shakeout run; cpu_medium (5d) only if
# the full set really needs more than 12h.
# NOTE: the default partition `prod` has ZERO nodes, so --partition is mandatory here.

#SBATCH --partition=cpu_short
#SBATCH --job-name=nyumets-h5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=EE2178@nyulangone.org
#SBATCH --chdir=/gpfs/scratch/ee2178/ImMAP
#SBATCH --output=/gpfs/scratch/ee2178/ImMAP/logs/%x_%j.out
#SBATCH --error=/gpfs/scratch/ee2178/ImMAP/logs/%x_%j.err

# --- conda ------------------------------------------------------------------------------
# No `set -euo pipefail` here: conda's activate functions read unset variables ($PS1,
# $_CE_CONDA, $CONDA_SHLVL) before init, so `set -u` aborts the job during activation.
# Guard the two steps that can actually fail instead.
#
# module load anaconda3/cpu/2023.09          # uncomment if conda is not already on PATH
CONDA_BASE="$(conda info --base 2>/dev/null)"
if [ -z "$CONDA_BASE" ]; then
    echo "conda not on PATH -- uncomment the module load line above" >&2
    exit 1
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate env || { echo "conda activate env failed" >&2; exit 1; }

# ~/.local/lib/python3.x/site-packages shadows the env and outranks it; a stray
# `pip install --user` in this account would silently override the env's own packages.
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# --- paths ------------------------------------------------------------------------------
IMMAP_ROOT="${IMMAP_ROOT:-$HOME/scratch/ImMAP}"
NYUMETS="${NYUMETS:-$IMMAP_ROOT/../datasets/NYUMets}"   # symlink -> /gpfs/data/fenglab/...
OUT="${OUT:-$HOME/scratch/datasets/NYUMets_h5}"

cd "$IMMAP_ROOT"
echo "host   : $(hostname)"
echo "python : $(which python)"
echo "root   : $NYUMETS"
echo "out    : $OUT"

# A rebuild MUST pass --overwrite: main() skips any session whose h5 already exists, so
# without it a re-run after a preprocessing change silently keeps the old data.
OVERWRITE_FLAG=""
[ -n "${OVERWRITE:-}" ] && OVERWRITE_FLAG="--overwrite"

python -m preprocessing.nyumets_h5 \
    --root "$NYUMETS/data/imaging/patientId" \
    --out  "$OUT" \
    --crop 224 \
    --min-brain-frac 0.02 \
    --reg-device "${REG_DEVICE:-cpu}" \
    --compress $OVERWRITE_FLAG ${EXTRA:-}
