#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=torch_pr_89_tandon_advanced
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --job-name=VolumeScales-knee
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%j.out
#SBATCH --error=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%j.err
#
# PER-VOLUME intensity statistics for the knee PD / PDFS pairs (train + val + test splits):
# foreground p90 / p95 / p99 of each volume's full-k-space RSS and of its ACS-only image, raw
# units -> config/knee/guided/volume_scales.json, which the guided configs' per-volume
# normalization reads (scripts/make_guided_recon_configs.py NORMALIZATION = "volume"). CPU only;
# needs only the raw k-space, not the offline maps, so it can run alongside
# torch/espirit_knee_pairs.sbatch. Commit the JSON afterwards.
#
# Submit from the repo root:   sbatch torch/volume_scales_knee.sbatch

IMMAP_ROOT="${IMMAP_ROOT:-${SLURM_SUBMIT_DIR:-${HOME}/scratch/ee2178/ImMAP}}"
source ~/.bashrc
conda activate gcdl
cd "${IMMAP_ROOT}"
set -eo pipefail
mkdir -p torch/logs

python scripts/volume_scale_stats.py --stride 2 --center-frac 0.04 \
    --out config/knee/guided/volume_scales.json
