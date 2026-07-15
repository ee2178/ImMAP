#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=torch_pr_89_tandon_advanced
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=48:00:00
#SBATCH --job-name=GroupCDLDict-BraTS
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Activate Conda environment

source ~/.bashrc                   # Ensure conda is available
conda activate env      # Replace with your actual env name

# Navigate to the directory containing train.py

cd ~/scratch/ee2178/ImMAP       # Replace with the actual path

# Dictionary pretraining for the latent-I2SB bridge. Both are plain `task: denoiser` runs;
# the dictionary is D = B[0] and the latent is the sparse code z (GroupCDL returns both).
# Keep M and sc identical across the two, or step 3 has no shared latent shape to bridge in.

# step 2 -- T1ce dictionary        (C=1, cond_idx: [], ch0: "x0")
python train.py config/BraTS/groupcdl_dict_t1ce.json

# step 1 -- joint dictionary       (C=4, cond_idx: [0,1,3], ch0: "bridge")
# python train.py config/BraTS/groupcdl_dict_joint.json
