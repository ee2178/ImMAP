#!/bin/bash
# Build the NYUMets FLAIR/T1/CT1/T2 h5 set.  CPU-only: preprocessing/nyumets_h5.py runs
# entirely on CPU tensors, so this must NOT sit on a gpu partition.
#
#   mkdir -p logs && sbatch bigpurple/nyumets_h5.sbatch
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
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# --- conda ------------------------------------------------------------------------------
# module load anaconda3/cpu/2023.09          # uncomment if conda is not already on PATH
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate env

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

python -m preprocessing.nyumets_h5 \
    --root "$NYUMETS/data/imaging/patientId" \
    --out  "$OUT" \
    --crop 224 \
    --min-brain-frac 0.02 \
    --compress
