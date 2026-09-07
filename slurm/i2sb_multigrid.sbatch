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
#SBATCH --job-name=I2SB-multigrid-BraTS
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

source ~/.bashrc
conda activate gcdl
cd ~/scratch/ee2178/ImMAP

# Multigrid arms of the T1 -> T1ce bridge. Each is cloned from the flat run it ablates, so only
# the model block differs -- K=[3,[8,8,4]] is 3 V-cycles at ~10.25 fine-equivalent iterations
# each, matched to the flat nets' K=30.
#
#   sbmgcdlnet  the two-fidelity bridge on V-cycles      vs  i2sb_sbcdlnet_mednorm.json
#   mgcdlnet    plain multigrid CDLNet, unconditional    vs  i2sb_cdlnet_t1.json
#
# Uncomment one. Both are ~2.5x the parameters of their flat baseline (the V-cycle carries untied
# smoothers per level plus dF's own copies); set model.params.M to 64 for near-exact parity with
# CDLNet if you want multigrid isolated from capacity.

python train.py config/BraTS/i2sb_sbmgcdlnet.json
# python train.py config/BraTS/i2sb_mgcdlnet_t1.json
