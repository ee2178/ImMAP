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
#SBATCH --job-name=DTCDLNet-dS-ET-BraTS
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Activate Conda environment

source ~/.bashrc                   # Ensure conda is available
conda activate gcdl      # Replace with your actual env name

# Navigate to the directory containing train.py

cd ~/scratch/ee2178/ImMAP       # Replace with the actual path

# RUN 1/3 -- DT-CDLNet, dS branch ON, direct T1ce prediction, ET-weighted loss.
#
#   task           dt_synthesis (train_dt_synthesis: unrolled forward + net.project())
#   use_dS         true   -> x_hat = D_x z^K + dS^K, the full enhancement model
#   residual_mode  n/a    -> the net predicts T1ce itself, not T1ce - T1
#   et_weight      330    -> ET pixels count 330x in L(x_hat, T1ce); ~50% of the objective
#                            (see training.losses.weighted_loss -- a PER-PIXEL weight)
#   scales         [2,2,2,2] -> z-scored data compressed from ~[-2,2] to ~[-1,1]
#
# This is the reference point for the other two runs: dS present, no residual
# reparameterization. Writes to trained_nets/brats/DT_CDLNet_dS_T1ce_ET -- a NEW directory,
# so the pre-ET DT_CDLNet_T1ce run is left intact.
#
# Requires 'et' in the *_img.h5 files. If they predate ET storage, back it out first (cheap,
# no regeneration): python preprocessing/cmap.py --config config/BraTS/cmap.yaml --add-seg-only
# ET/dS numbers are no longer logged as scalars -- inspect val/dS and val/et visually.
# If global PSNR sags, lower et_weight.

python train.py config/BraTS/dt_synthesis.json
