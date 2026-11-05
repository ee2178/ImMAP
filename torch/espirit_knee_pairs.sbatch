#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=torch_pr_89_tandon_advanced
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --job-name=ESPIRiT-knee-pairs
#SBATCH --output=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%A_%a.out
#SBATCH --error=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%A_%a.err
#SBATCH --array=0-7
#
# OFFLINE ESPIRiT maps + coil-combined image for the GUIDED knee volumes: by default every PDFS
# volume (the target contrast) in TGVN's patient-level splits, train + val + test
# (config/knee/guided/splits/), so the guided runs can score against -- and bridge to -- a
# coil combination computed from the FULLY SAMPLED k-space with a generous calibration region,
# instead of the RSS or the online maps the measurement's own ACS allows.
#
#   CONTRAST=pdfs (default) | pd       which column of the pair CSVs
#   ACS=32                             square calibration region, from the FULL k-space -- larger
#                                      than the brain run's 20 (and far larger than the 13-line
#                                      ACS the online maps get at R=8)
#   KERNEL=8, THRESH_EIG=0.95          as the brain maps (notebooks/espirit_brain.ipynb); 0.95
#                                      gives the maps a hard support, zero outside the anatomy
#
# Output: ../datasets/fastmri_preprocessed/knee_coil_combined/${CONTRAST}_espirit_acs${ACS}/,
# one h5 per volume with `smaps` (S, C, H, W) and `image` (S, H, W) = sum_c conj(s_c) c_c --
# the layout datasets/fastmri/loader.py reads. The splits draw from fastMRI multicoil train AND
# val, so each file is looked up in both raw roots.
#
# 8 shards of the file list, one A100 each (ESPIRiT's kernel images are gigabytes per slice on
# the 640-line readout; CHUNK halves itself on CUDA OOM). Volumes already written are skipped and
# files land via .partial + rename, so a timed-out shard is simply resubmitted.
#
# BEFORE SUBMITTING (lists the files, computes nothing):
#     python scripts/make_espirit_smaps.py --anatomy knee \
#         --pairs-csv config/knee/guided/splits/{train,val,test}.csv --contrast pdfs \
#         --kspace-roots ../datasets/fastmri/knee/multicoil_train ../datasets/fastmri/knee/multicoil_val \
#         --out ../datasets/fastmri_preprocessed/knee_coil_combined/pdfs_espirit_acs32 --dry-run
#
# Submit from the repo root:   sbatch torch/espirit_knee_pairs.sbatch
# PD side scans as well:       CONTRAST=pd sbatch --export=ALL torch/espirit_knee_pairs.sbatch
# Fewer GPUs at once:          sbatch --array=0-7%2 torch/espirit_knee_pairs.sbatch

CONTRAST="${CONTRAST:-pdfs}"
ACS="${ACS:-32}"
KERNEL="${KERNEL:-8}"
THRESH_EIG="${THRESH_EIG:-0.95}"
CHUNK="${CHUNK:-4}"
N_SHARDS=8
OUT="../datasets/fastmri_preprocessed/knee_coil_combined/${CONTRAST}_espirit_acs${ACS}"

for _c in "${IMMAP_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" "$HOME/scratch/ee2178/ImMAP"; do
    if [ -n "${_c}" ] && [ -f "${_c}/scripts/make_espirit_smaps.py" ]; then
        IMMAP_ROOT="${_c}"
        break
    fi
done
if [ ! -f "${IMMAP_ROOT:-}/scripts/make_espirit_smaps.py" ]; then
    echo "[espirit] cannot locate the ImMAP checkout; submit from the repo root" >&2
    exit 1
fi
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
if [ "${SHARD}" -ge "${N_SHARDS}" ]; then
    echo "[espirit] task ${SHARD} is outside 0-$((N_SHARDS - 1))." >&2
    exit 1
fi

set -eo pipefail
source ~/.bashrc
conda activate gcdl
cd "${IMMAP_ROOT}"
mkdir -p torch/logs

echo "[espirit] knee ${CONTRAST} acs=${ACS} shard ${SHARD}/${N_SHARDS} -> ${OUT}"
python scripts/make_espirit_smaps.py \
    --anatomy knee \
    --pairs-csv config/knee/guided/splits/train.csv config/knee/guided/splits/val.csv \
                config/knee/guided/splits/test.csv \
    --contrast "${CONTRAST}" \
    --kspace-roots ../datasets/fastmri/knee/multicoil_train ../datasets/fastmri/knee/multicoil_val \
    --out "${OUT}" \
    --acs "${ACS}" \
    --kernel-size "${KERNEL}" \
    --thresh-eig "${THRESH_EIG}" \
    --chunk "${CHUNK}" \
    --shard "${SHARD}" \
    --num-shards "${N_SHARDS}"
