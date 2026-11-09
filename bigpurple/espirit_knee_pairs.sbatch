#!/bin/bash
#
# OFFLINE ESPIRiT maps + SENSE images of the PDFS (target) volumes, on BIGPURPLE -- the same job
# as torch/espirit_knee_pairs.sbatch (ACS 32, kernel 8, thresh_eig 0.9; see there), reading the
# paired knee copy through ../datasets/fastmri_multiknee (bash bigpurple/setup_knee_data.sh
# first). The maps are written to the SAME relative path as on torch,
# ../datasets/fastmri_preprocessed/knee_coil_combined/pdfs_espirit_acs32_eig0.9, which is what
# the guided configs' smap_root names. Volumes already written are skipped; a timed-out shard is
# simply resubmitted.
#
# Submit from the repo root:   sbatch bigpurple/espirit_knee_pairs.sbatch

#SBATCH --partition=radiology
#SBATCH --job-name=espirit-knee
#SBATCH --array=0-7%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=EE2178@nyulangone.org
#SBATCH --chdir=/gpfs/scratch/ee2178/ImMAP
#SBATCH --output=/gpfs/scratch/ee2178/ImMAP/bigpurple/logs/%x_%A_%a.out
#SBATCH --error=/gpfs/scratch/ee2178/ImMAP/bigpurple/logs/%x_%A_%a.err
# No `set -euo pipefail` -- it breaks conda activate.

CONTRAST="${CONTRAST:-pdfs}"
ACS="${ACS:-32}"
KERNEL="${KERNEL:-8}"
THRESH_EIG="${THRESH_EIG:-0.9}"
CHUNK="${CHUNK:-4}"
N_SHARDS=8
OUT="../datasets/fastmri_preprocessed/knee_coil_combined/${CONTRAST}_espirit_acs${ACS}_eig${THRESH_EIG}"
ROOTS=(../datasets/fastmri_multiknee/train ../datasets/fastmri_multiknee/val
       ../datasets/fastmri_multiknee/test)

CONDA_BASE="$(conda info --base 2>/dev/null)"
[ -n "$CONDA_BASE" ] || { echo "conda not on PATH" >&2; exit 1; }
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate env || { echo "conda activate env failed" >&2; exit 1; }
export PYTHONNOUSERSITE=1
IMMAP_ROOT="${IMMAP_ROOT:-$HOME/scratch/ImMAP}"
cd "$IMMAP_ROOT" || { echo "no such dir: $IMMAP_ROOT" >&2; exit 1; }
mkdir -p bigpurple/logs
[ -d ../datasets/fastmri_multiknee/train ] || {
    echo "no ../datasets/fastmri_multiknee: run bash bigpurple/setup_knee_data.sh" >&2; exit 1; }

SHARD="${SLURM_ARRAY_TASK_ID:-0}"
echo "[espirit] knee ${CONTRAST} acs=${ACS} eig=${THRESH_EIG} shard ${SHARD}/${N_SHARDS} -> ${OUT}"
python scripts/make_espirit_smaps.py \
    --anatomy knee \
    --pairs-csv config/knee/guided/splits/train.csv config/knee/guided/splits/val.csv \
                config/knee/guided/splits/test.csv \
    --contrast "${CONTRAST}" \
    --kspace-roots "${ROOTS[@]}" \
    --out "${OUT}" \
    --acs "${ACS}" \
    --kernel-size "${KERNEL}" \
    --thresh-eig "${THRESH_EIG}" \
    --chunk "${CHUNK}" \
    --shard "${SHARD}" \
    --num-shards "${N_SHARDS}"
