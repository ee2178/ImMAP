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
#SBATCH --job-name=LatentI2SB-BraTS-LatentInfImageBridge
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Activate Conda environment

source ~/.bashrc                   # Ensure conda is available
conda activate gcdl      # Replace with your actual env name

# Navigate to the directory containing train.py

cd ~/scratch/ee2178/ImMAP       # Replace with the actual path

# flex_attention JIT-compiles Triton kernels on the FIRST iteration (looks like a "freeze" at
# step 0 -- it is compiling, not hung). Point the Inductor/Triton caches at PERSISTENT scratch so
# that compile is paid ONCE, not every job: the default lands in job-ephemeral
# /state/partition1/job-*/torchinductor_* and is wiped when the job ends, so every job recompiles.
export TORCHINDUCTOR_CACHE_DIR=/scratch/ee2178/.cache/torchinductor
export TRITON_CACHE_DIR=/scratch/ee2178/.cache/triton

# Latent-I2SB "step 3": trains a NEW GroupCDL to bridge joint-latent -> T1ce-latent, with the
# loss in the T1ce IMAGE domain. Requires steps 1 & 2 (config/BraTS/groupcdl_dict_joint.json,
# groupcdl_dict_t1ce.json) to have finished -- their saved config.json + net.ckpt are loaded
# frozen via cfg["dicts"] in the config below.

# python train.py config/BraTS/latent_inf_image_bridge.json
python train.py config/BraTS/latent_inf_image_bridge.json
