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
#SBATCH --job-name=MGRecon-brain
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --array=0-7           # MUST equal the brain cell count; checked at runtime
#
# fastMRI BRAIN (T2), synthetic k-space, sigma ~ U[0, 0.01].
#
#   4 models x 2 accelerations (R=8, R=4) = 8 cells
#
#     python scripts/make_mg_recon_configs.py --list-cells --anatomy brain
#     0  brain  R8  lpdsnet
#     1  brain  R8  mglpds
#     ...
#
# Same four models as the knee launcher; see slurm/mg_recon_knee.sbatch for what
# each pair does and does not isolate.
#
# UNVERIFIED PATHS. The brain roots follow the knee pattern --
#     ../datasets/fastmri/brain/multicoil_{train,val}
#     ../datasets/fastmri_preprocessed/brain_T2W_coil_combined/{train,val}
# -- but that tree has never been checked, and neither has the slice range
# (start_slice 0, end_slice 8, versus knee's 12-25). A wrong path fails at
# loader construction, so:
#
#     sbatch --array=0 slurm/mg_recon_brain.sbatch
#
# costs ~30 seconds and confirms the setup before the other seven go out.
#
# Submit:      sbatch slurm/mg_recon_brain.sbatch
# Probe first: SWEEP_EPOCHS=20 below, then look at the val curves.

ANATOMY=brain
CONFIG_ROOT="config"
SWEEP_EPOCHS=""                      # empty = the config's num_epochs
REGENERATE=1                         # 1 = regenerate configs before running

source "$(dirname "${BASH_SOURCE[0]}")/_mg_recon_body.sh"
