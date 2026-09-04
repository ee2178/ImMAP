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
#SBATCH --job-name=I2SB-Unet
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --array=0-1
#
# The I2SB PAPER BASELINE, both conditioning settings:
#
#     task 0   i2sb_unet_t1.json    T1 -> T1ce            cond_idx []        C = 1
#     task 1   i2sb_unet_all.json   all -> T1ce           cond_idx [0,1,3]   C = 4
#
# The pair isolates one thing: how much the side contrasts (FLAIR, T2) are worth on top of the
# bridge itself. Everything else -- architecture, schedule, optimizer, loss, data -- is identical,
# so a gap is attributable to conditioning and nothing else.
#
# THE MODEL IS THE PAPER'S, NOT "A UNET". models/sb_unet.py is a port of ADM / guided-diffusion's
# UNet, which is what I2SB uses unmodified (i2sb/network.py builds it via
# guided_diffusion.script_util.create_model). The conditioning path is the part that had to be
# faithful:
#
#     I2SB runner.py:86   noise_levels = linspace(t0, T, interval) * interval    (~ the STEP index)
#          network.py:47  t = noise_levels[steps]
#          network.py:51  diffusion_model(cat([x, cond], 1), t)
#
# so the net is conditioned on the discrete bridge POSITION through a sinusoidal embedding and
# per-ResBlock FiLM -- not on sigma, despite the variable's name. Since sb.base.predict_x0 hands
# regressors sigma, SBUnet inverts std_fwd back to the step (the same lookup SBCDLNet uses) and
# then follows I2SB exactly.
#
# CONSEQUENCE FOR CONFIGS: SBUnet carries its own copy of the schedule to do that inversion, so
# kind / n_points / beta_max appear in BOTH cfg["i2sb"] AND cfg["model"]["params"] and MUST agree.
# train_i2sb calls net.assert_schedule_matches() at startup and dies on a mismatch -- if you sweep
# beta_max, override both keys.
#
# SIZE. ~23M parameters against SBCDLNet's 2.00M and CDLNet's 0.51M. The comparison is not
# parameter-matched and is not meant to be: this is the reference architecture at a sane scale for
# 128px single-contrast MRI, not an ablation. (ADM's 256px checkpoint I2SB fine-tunes is ~550M.)
#
# PREREQUISITE: the h5 files need the 'img_median_mad' dataset --
#   python preprocessing/cmap.py --config config/BraTS/cmap.yaml
#
# beta_max = 0.3 is the [-1, 1] calibration (std_fwd[-1] = 0.3755, matched to the RMS T1ce-T1
# gap once the loader divides by scales = 3), so it needs no change here. If you ever DO
# sweep it, note that it lives in cfg["i2sb"] AND cfg["model"]["params"] -- SBUnet carries
# its own schedule copy to invert std_fwd, and train_i2sb.assert_schedule_matches kills the
# job at startup if the two drift apart.
#
# Submit:  sbatch torch/i2sb_unet.sbatch
# One only:  sbatch --array=0 torch/i2sb_unet.sbatch

source ~/.bashrc
conda activate gcdl

cd ~/scratch/ee2178/ImMAP

CONFIGS=(config/BraTS/i2sb_unet_t1.json config/BraTS/i2sb_unet_all.json)

TASK=${SLURM_ARRAY_TASK_ID:-0}
if [ "$TASK" -ge "${#CONFIGS[@]}" ]; then
    echo "array index $TASK >= ${#CONFIGS[@]} configs; fix --array" >&2
    exit 1
fi

echo "task $TASK -> ${CONFIGS[$TASK]}"
python train.py "${CONFIGS[$TASK]}"
