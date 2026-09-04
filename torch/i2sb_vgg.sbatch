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
#SBATCH --job-name=I2SB-SBCDLNet-VGG
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#
# Perceptual-loss run: SBCDLNet with the VGG16 relu4_3 feature loss from the CCL work, alongside
# the pixel MSE. Only SBCDLNet for now -- the same `loss_type` / `loss_params` block drops into
# the CDLNet and SBUnet configs unchanged when you want the others.
#
# WHY loss_type IS "mse-vgg" AND NOT "vgg-feature".
#
# The TF reference (CCL-Synthetis/Synthesis/synthesis_losses.py, Recon_perceptualLoss) is PURELY
# perceptual: it returns `perceptualLoss_weight * style_loss` and no pixel term at all, despite
# the config comment there calling the weight an "mae vs perceptual" balance. training/losses.py's
# `vgg-feature` ports that faithfully and should stay that way.
#
# That objective does not transfer to a bridge regressor. VGG features are largely invariant to a
# global affine change of intensity, so a pure perceptual loss leaves the ABSOLUTE scale of x0_hat
# unconstrained -- and sb.base.reverse_sample mixes x0_hat with the running state x_t on an
# absolute scale at every one of NFE steps, so a drift the loss cannot see compounds. `mse-vgg`
# keeps the pixel term as the anchor and adds the perceptual term on top.
#
# SETTING vgg_weight -- DO NOT SKIP THIS.
#
# The two terms are not commensurate and the right weight spans orders of magnitude with the data
# scale. `loss_params.vgg_weight: 1.0e-2` in the config is a STARTING GUESS, not a tuned value.
# train_i2sb measures the split on the first batch and prints, in the job's stdout:
#
#   [i2sb] loss balance on batch 0: mse=...  vgg=...  vgg_weight=...
#          -> perceptual term is X% of the objective; vgg_weight=... would make it 50%
#
# Read that line in logs/slurm_<jobid>.out within the first minute. Aim for the perceptual term at
# roughly 10-20% of the objective; set vgg_weight to about 0.1-0.2 times the printed parity value
# and resubmit. It warns explicitly at either degenerate end (<0.1% = effectively pure MSE,
# >95% = effectively pure VGG).
#
# Note `size_normalize: false` in the config. The reference multiplies the feature MSE by
# 1/(H*W*CH), which at this input size is ~1/3e5 -- with it on, the perceptual term is numerically
# absent next to a pixel MSE no matter what weight you pick.
#
# COST: the VGG16 forward runs on every training step, on top of the regressor. Expect a
# noticeably slower epoch than the plain-MSE run; the first step also downloads/caches the
# ImageNet VGG16 weights, which needs the node to reach the torchvision cache or the network.
#
# PREREQUISITE: the h5 files need the 'img_median_mad' dataset --
#   python preprocessing/cmap.py --config config/BraTS/cmap.yaml
#
# beta_max = 0.3 is the [-1, 1] calibration (std_fwd[-1] = 0.3755, matched to the RMS T1ce-T1
# gap once the loader divides by scales = 3), so it needs no change here. If you ever DO
# sweep it, note that it lives in cfg["i2sb"] AND cfg["model"]["params"] -- SBCDLNet carries
# its own schedule copy to invert std_fwd, and train_i2sb.assert_schedule_matches kills the
# job at startup if the two drift apart.
#
# The MSE-only control this is measured against is config/BraTS/i2sb_sbcdlnet_mednorm.json --
# identical apart from the loss, so run that one too if you have not already.
#
# Submit:  sbatch torch/i2sb_vgg.sbatch

source ~/.bashrc
conda activate gcdl

cd ~/scratch/ee2178/ImMAP

python train.py config/BraTS/i2sb_sbcdlnet_vgg.json
