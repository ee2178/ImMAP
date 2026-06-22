#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=torch_pr_89_tandon_advanced
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=48:00:00
#SBATCH --job-name=ImMAP2p5-FastMRIKnee
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Activate Conda environment

source ~/.bashrc                   # Ensure conda is available
conda activate env      # Replace with your actual env name

# Navigate to the directory containing train.py

cd ~/scratch/ee2178/ImMAP       # Replace with the actual path
python3 train.py config/knee/joint_denoising_recon.json
# Continue training from checkpoint
# python3 train.py trained_nets/knee/LPDS_JDR_R6_polynomialsteps/config.json
