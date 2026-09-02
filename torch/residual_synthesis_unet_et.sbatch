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
#SBATCH --job-name=ResUNet-ET-BraTS
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Activate Conda environment

source ~/.bashrc                   # Ensure conda is available
conda activate gcdl      # Replace with your actual env name

# Navigate to the directory containing train.py

cd ~/scratch/ee2178/ImMAP       # Replace with the actual path

# RUN 2/3 -- plain U-Net, residual target, ET-weighted loss.
#
#   task           synthesis (train_synthesis)
#   model          Unet2D, 3 -> 1
#   residual_mode  true, residual_src_idx 1 -> supervised on T1ce - X[:,1] = T1ce - T1
#                  (input_idx [0,1,3] is flair,t1,t2, so position 1 is T1)
#   et_weight      330    -> ET pixels count 330x in L(pred, T1ce-T1) (per-pixel weight)
#   scales         [2,2,2,2]
#
# The architecture-free control: whatever the residual reparameterization and ET weighting
# are worth, this run shows it without any unrolled-model machinery. psnr/ssim/nrmse are
# logged in the ORDINARY T1ce domain (anchor added back), so they compare directly against
# the non-residual synthesis runs; delta_* are the same metrics on the residual itself.
#
# Requires 'et' in the *_img.h5 files. If they predate ET storage, back it out first (cheap,
# no regeneration): python preprocessing/cmap.py --config config/BraTS/cmap.yaml --add-seg-only

python train.py config/BraTS/residual_synthesis_unet.json
