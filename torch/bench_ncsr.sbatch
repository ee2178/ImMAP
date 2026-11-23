#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=torch_pr_89_tandon_advanced
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=01:00:00
#SBATCH --job-name=NCSR-bench
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%j.out
#SBATCH --error=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%j.err
#
# NCSR-LPDS's fused (triton) path: checked against the roll path, then one knee-size training
# step timed both ways (scripts/bench_ncsr.py), on the L40S the knee runs train on.
#
#   sbatch torch/bench_ncsr.sbatch
#   EXTRA="--no-roll" sbatch --export=ALL torch/bench_ncsr.sbatch       # fused timing only
#
# A MISMATCH in the check stops the job before timing: send the log.

IMMAP_ROOT="${IMMAP_ROOT:-${HOME}/scratch/ee2178/ImMAP}"
source ~/.bashrc
conda activate gcdl
cd "${IMMAP_ROOT}"
set -eo pipefail
mkdir -p torch/logs
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python scripts/bench_ncsr.py --json "torch/logs/ncsr_bench_${SLURM_JOB_ID:-local}.json" ${EXTRA:-}
