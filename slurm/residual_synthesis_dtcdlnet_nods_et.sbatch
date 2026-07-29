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
#SBATCH --job-name=ResDTCDLNet-noDS-ET-BraTS
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Activate Conda environment

source ~/.bashrc                   # Ensure conda is available
conda activate gcdl      # Replace with your actual env name

# Navigate to the directory containing train.py

cd ~/scratch/ee2178/ImMAP       # Replace with the actual path

# RUN 3/3 -- DT-CDLNet with the dS branch REMOVED, residual target, ET-weighted loss.
#
#   task           synthesis (train_synthesis, not train_dt_synthesis: it unpacks the
#                  (x_hat, dS, z) tuple and calls net.project(), which is all this model
#                  needs, and it is the loop that implements residual_mode)
#   use_dS         false  -> no E^k / rho / tauS parameters exist at all; x_hat = D_x z^K.
#                  rho0 and tauS0 are left in the config but are inert -- keeping them means
#                  this file differs from residual_synthesis_dtcdlnet.json in exactly ONE
#                  key, so the pair is a clean dS ablation.
#   residual_mode  true, residual_src_idx 1 -> supervised on T1ce - T1
#   et_weight      330    -> ET pixels count 330x in the loss (per-pixel weight)
#   scales         [2,2,2,2]
#
# The premise worth testing here: dS exists to add enhancement on top of a base decode, but
# in residual mode the TARGET is already the enhancement, so the residual reparameterization
# and the dS branch are two mechanisms for the same job. This run asks whether dS is still
# earning its parameters once the target is the residual. dS diagnostics will read
# dS_mean = dS_active = 0 by construction.
#
# Requires 'et' in the *_img.h5 files. If they predate ET storage, back it out first (cheap,
# no regeneration): python preprocessing/cmap.py --config config/BraTS/cmap.yaml --add-seg-only

python train.py config/BraTS/residual_synthesis_dtcdlnet_nods.json
