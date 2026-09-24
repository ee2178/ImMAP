#!/bin/bash
# Survey (or apply) cross-study registration on an already-built NYUMets h5 set.
#
#   mkdir -p bigpurple/logs && sbatch bigpurple/register_nyumets.sbatch
#
# Defaults to a READ-ONLY survey of 40 randomly chosen patients -- enough to see the offset
# distribution and decide whether the rebuild is worth it. Nothing is written unless APPLY=1.
#
#   SAMPLE=0 sbatch bigpurple/register_nyumets.sbatch            # every patient, still read-only
#   SEED=7   sbatch bigpurple/register_nyumets.sbatch            # a different draw
#   APPLY=1 SAMPLE=0 OUT=~/scratch/datasets/NYUMets_h5_reg \
#           sbatch bigpurple/register_nyumets.sbatch             # write a registered copy
#
# GEOMETRY SURVEY -- run this FIRST, and before trusting any offset:
#
#   GEOMETRY=1 sbatch --gres=none --time=00:30:00 bigpurple/register_nyumets.sbatch
#
# It reads ATTRS ONLY: no pixels, no FFTs, no GPU, and it defaults to every patient because
# it is cheap enough to do the whole cohort. What it answers is whether an integer-index
# shift is even the right model -- the per-study voxel spacing, depth, kept range, and the
# STORED PHYSICAL FOV (crop * in-plane spacing). `--crop N` crops in INDEX space, so two
# studies at different in-plane resolution come out of the build holding different physical
# FOVs at different anatomical scales; cross-correlation then reports a SMALL in-plane offset
# (there is no good one) while the images stay visibly off-centre. Read the
# `in-plane spacing across all sampled studies` line: more than one distinct value means the
# build has to resample to a common in-plane spacing before registration means anything.
#
# `--gres=none` because no GPU is used; the partition stays radiology, which is the only one
# these jobs run on. GEOMETRY=1 forces DEVICE=cpu so the script's CUDA check cannot fail the
# job over a device it was never going to touch.
#
# APPLY=1 with a SAMPLE is refused by the script itself: it would write a set in which only
# some patients are registered. Pass SAMPLE=0 (or ALLOW_PARTIAL=1 if you really mean it).
#
# WHY A GPU. The work is 3-D FFT cross-correlation per patient; only the low-passed probe goes
# to the device, so GPU memory scales with ONE contrast rather than four whole volumes, and a
# survey is a few minutes instead of an hour. It runs perfectly well on CPU -- DEVICE=cpu, and
# then a cpu_short allocation is the cheaper queue. The script auto-detects either way.
#
# radiology is the only GPU partition these jobs run on. Typed GRES (gpu:a100:1) because an
# untyped request can land on a card the conda env's torch was not built for.

#SBATCH --partition=radiology
#SBATCH --job-name=nyumets-register
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=EE2178@nyulangone.org
#SBATCH --chdir=/gpfs/scratch/ee2178/ImMAP
#SBATCH --output=/gpfs/scratch/ee2178/ImMAP/bigpurple/logs/%x_%j.out
#SBATCH --error=/gpfs/scratch/ee2178/ImMAP/bigpurple/logs/%x_%j.err

# --- conda ------------------------------------------------------------------------------
# No `set -euo pipefail` here: conda's activate functions read unset variables ($PS1,
# $_CE_CONDA, $CONDA_SHLVL) before init, so `set -u` aborts the job during activation.
# Guard the steps that can actually fail instead.
#
# module load anaconda3/gpu/2023.09          # uncomment if conda is not already on PATH
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

# --- paths and knobs --------------------------------------------------------------------
IMMAP_ROOT="${IMMAP_ROOT:-$HOME/scratch/ImMAP}"
ROOT="${ROOT:-$HOME/scratch/datasets/NYUMets_h5}"
OUT="${OUT:-${ROOT}_reg}"
SEED="${SEED:-0}"
if [ -n "${GEOMETRY:-}" ]; then
    # Attrs only, so the whole cohort is the sensible default and the device is irrelevant --
    # but the script hard-exits on --device cuda with no CUDA visible, and this is meant to be
    # submitted with --gres=none, so pin it to cpu unless the caller insists otherwise.
    SAMPLE="${SAMPLE:-0}"
    DEVICE="${DEVICE:-cpu}"
    if [ -n "${APPLY:-}" ]; then
        echo "GEOMETRY=1 is a read-only measurement; drop APPLY=1" >&2
        exit 1
    fi
else
    SAMPLE="${SAMPLE:-40}"
    DEVICE="${DEVICE:-cuda}"
fi

cd "$IMMAP_ROOT" || { echo "no such directory: $IMMAP_ROOT" >&2; exit 1; }
if [ ! -d "$ROOT" ]; then
    echo "ROOT does not exist: $ROOT  (build it with bigpurple/nyumets_h5.sbatch first)" >&2
    exit 1
fi

echo "host   : $(hostname)"
echo "python : $(which python)"
echo "root   : $ROOT"
echo "sample : $SAMPLE (seed $SEED)   device: $DEVICE"

# A GPU was requested, so say plainly whether one arrived -- a silent fall back to CPU on a
# radiology node is a slow job that looks like a fast one. Skipped for GEOMETRY=1, which is
# submitted with --gres=none and would only report a GPU it does not want.
if [ -z "${GEOMETRY:-}" ]; then
python - <<'PY'
import torch
if torch.cuda.is_available():
    print(f"gpu    : {torch.cuda.get_device_name(0)}")
else:
    print("gpu    : NONE VISIBLE -- torch.cuda.is_available() is False; the run will use the "
          "CPU. Check the --gres line and the conda env's torch build.")
PY
fi

ARGS=(--root "$ROOT" --sample "$SAMPLE" --seed "$SEED" --device "$DEVICE")
[ -n "${MULTI_ONLY:-}" ]    && ARGS+=(--multi-only)
[ -n "${ALLOW_PARTIAL:-}" ] && ARGS+=(--allow-partial)
if [ -n "${GEOMETRY:-}" ]; then
    ARGS+=(--geometry)
    echo "mode   : GEOMETRY SURVEY (attrs only, no pixels, no writes)"
elif [ -n "${APPLY:-}" ]; then
    ARGS+=(--apply --out "$OUT")
    echo "mode   : APPLY -> $OUT"
else
    echo "mode   : REPORT ONLY (set APPLY=1 to write)"
fi
[ -n "${EXTRA:-}" ] && ARGS+=($EXTRA)      # e.g. EXTRA="--reg-lowpass 0.4 --min-overlap 12"

echo
python -m scripts.register_nyumets "${ARGS[@]}"
