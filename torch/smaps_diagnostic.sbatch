#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=torch_pr_89_tandon_advanced
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=02:00:00
#SBATCH --job-name=smaps-diag
#SBATCH --output=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%j.out
#SBATCH --error=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%j.err
#
# Online ESPIRiT vs the offline big-ACS maps on sample val slices: are the online maps behind the
# horizontal streaks? (scripts/compare_online_offline_smaps.py says how to read it.) Besides the
# config's online settings, three variants: a bigger kernel, a wider readout calibration band
# (the online block is clamped to 32 readout rows), and Walsh.
#
#     sbatch torch/smaps_diagnostic.sbatch
#     N_VOLUMES=12 sbatch --export=ALL torch/smaps_diagnostic.sbatch

IMMAP_ROOT="${IMMAP_ROOT:-${HOME}/scratch/ee2178/ImMAP}"
N_VOLUMES="${N_VOLUMES:-8}"
source ~/.bashrc
conda activate gcdl
cd "${IMMAP_ROOT}"
set -eo pipefail
mkdir -p torch/logs

python scripts/compare_online_offline_smaps.py \
    --config config/knee/guided/tgvn_c8_R8_g1.json --split val \
    --n-volumes "${N_VOLUMES}" --slices-per-volume 3 --panels 8 \
    --variant k6:kernel_size=6 \
    --variant clamp96:clamp=96 \
    --variant walsh:method=walsh \
    --out results/smaps_online_vs_offline
