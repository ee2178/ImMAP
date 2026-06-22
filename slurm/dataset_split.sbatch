#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=torch_pr_89_tandon_advanced
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=48:00:00
#SBATCH --job-name=BraTS-Dataset-Split
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Activate Conda environment

source ~/.bashrc                   # Ensure conda is available
conda activate env      # Replace with your actual env name

# Navigate to the directory containing train.py

cd ~/scratch/ee2178/ImMAP       # Replace with the actual path
python datasets/BraTS/make_split_symlinks.py --source ../datasets/BraTS/BraTS2021_DataSet --val_frac 0.30 --seed 0
