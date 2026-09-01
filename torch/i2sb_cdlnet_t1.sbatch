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
#SBATCH --job-name=I2SB-CDLNet-T1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#
# Strict T1 -> T1ce bridge with an ORDINARY CDLNet regressor (cond_idx: [], C = 1).
#
# WHY THIS RUN EXISTS. It is the ablation for SBCDLNet: same K/M/P/s (30/169/7/2), same schedule,
# same data, same loss. The only difference is the unrolled scaffold -- CDLNet solves a single
# denoising fidelity ||x_t - D z||^2, SBCDLNet solves the two-fidelity bridge problem with the
# debiased residual r = x_t - mu_1 x_1 and a prior-contrast dictionary. Capacity is matched
# deliberately so the gap measures the scaffold and not model size (0.51M vs 2.00M is the honest
# difference: SBCDLNet carries four dictionaries over C=4 inputs where this carries two over C=1).
#
# WHAT CDLNet STILL GETS RIGHT HERE. `adaptive: true` makes its threshold affine in sigma,
# t[k,0] + sigma * t[k,1], which is what lets one net cover the whole bridge -- sigma sweeps three
# decades along the schedule and a fixed threshold would be wrong nearly everywhere. What it does
# NOT get is x_1: with cond_idx empty the prior endpoint reaches the net only through x_t itself.
# That is the point of the comparison.
#
# PREREQUISITE: the h5 files need the 'img_median_mad' dataset. Regenerate with
#   python preprocessing/cmap.py --config config/BraTS/cmap.yaml
# (cmap.yaml: normalize_extra: [median-mad]). The loader raises at startup listing the available
# keys if it is missing.
#
# beta_max = 0.3 is the [-1, 1] calibration (std_fwd[-1] = 0.3755, matched to the RMS T1ce-T1
# gap once the loader divides by scales = 3), so it needs no change. CDLNet carries no schedule
# copy either way, so for THIS config beta_max lives only in cfg["i2sb"] -- unlike the
# SBCDLNet/SBUnet configs, where cfg["model"]["params"] duplicates it.
#
# Submit:  sbatch slurm/i2sb_cdlnet_t1.sbatch

source ~/.bashrc
conda activate gcdl

cd ~/scratch/ee2178/ImMAP

python train.py config/BraTS/i2sb_cdlnet_t1.json
