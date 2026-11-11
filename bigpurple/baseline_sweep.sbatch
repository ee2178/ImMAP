#!/bin/bash
#
# I2SB BASELINES, fastMRI knee, on BIGPURPLE (every i2sb cell trains here; the end-to-end recon
# nets train on torch): PDFS target, PD guide, R = 8. Cells (scripts/make_guided_recon_configs.py):
#
#     0 sbunet_R8              plain UNet bridge from the zero-filled image (x1 = E^H y), x1
#                              concatenated to the input at every step (I2SB's cond_x1); no guide
#     1 sbunet_inv_R8_g1       I2SB-INVERSION (arXiv 2411.14269) as github.com/zhyjSIAT/I2SB-Inversion
#                              runs it: starts at the PD image combined with its OWN maps; ADM UNet
#                              (256 ch, mult 1,1,2,2,4,4, ~550M params, random init) seeing ONLY x_t
#                              and the step, predicting eps (x0 = x_t - std_fwd * eps); CG (7 iters,
#                              lam 7e-5) on every prediction, no support mask
#
# The bridge itself (Brownian, tau), the data protocol and the TRAINING PROTOCOL (complex-MSE
# on x0, the grid's optimizer, LR, cosine schedule, clipping, batch, steps, 20-NFE validation) are
# the ones every i2sb baseline shares; the generator's sbunet_inv comment lists theirs vs ours line by line.
#
# I2SB-INVERSION'S FULL METHOD is an EVALUATION: training-time validation runs its first pass
# (I2SB-Recon). The inversion + second pass (~1000 extra net calls per slice) is
#     python scripts/eval_i2sb_recon_dc.py --run-dir <run> --inversion --dc 7:7e-5 --nfe 80
# (their 80 NFE; training-time validation uses the shared 20)
#
# Not in the default sweep: sbunet_guided (zero-filled start + the guide as two input channels
# at every step), which I2SB-Inversion does not do. Add it with MODELS (below).
#
# Everything else is guided_i2sb_knee.sbatch, which this runs: same generator, protocol, run dirs
# (trained_nets/guided_recon/knee/<RUN_TAG>/<cell>), resume and FORCE_RESTART, NOISE, and
# checkpoints at validation only.
#
# TAU: 0.19 for both cells by default, the value the starts sweep (sbtgvn_c8_<start>_tau0.19)
# is training with, so the run names get _tau0.19 and the cells compare directly. TAU= (empty)
# uses the per-start calibration instead (bridge_tau/<start>_R8_volume*.json, measured on first
# use).
#
# Submit:        sbatch bigpurple/baseline_sweep.sbatch
# One cell:      sbatch --array=1 bigpurple/baseline_sweep.sbatch                   (sbunet_inv)
# Other cells:   MODELS="sbunet sbunet_inv sbunet_guided" sbatch --array=0-2 --export=ALL bigpurple/baseline_sweep.sbatch
# Other tau:     TAU=0.09 sbatch --export=ALL bigpurple/baseline_sweep.sbatch     (TAU= : calibrated)

#SBATCH --partition=radiology
#SBATCH --job-name=i2sb-base
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=EE2178@nyulangone.org
#SBATCH --chdir=/gpfs/scratch/ee2178/ImMAP
#SBATCH --output=/gpfs/scratch/ee2178/ImMAP/bigpurple/logs/%x_%A_%a.out
#SBATCH --error=/gpfs/scratch/ee2178/ImMAP/bigpurple/logs/%x_%A_%a.err

export MODELS="${MODELS:-sbunet sbunet_inv}"
export TAU="${TAU-0.19}"                   # set but empty -> the per-start calibration
IMMAP_ROOT="${IMMAP_ROOT:-$HOME/scratch/ImMAP}"
cd "$IMMAP_ROOT" || { echo "no such dir: $IMMAP_ROOT" >&2; exit 1; }
echo "[i2sb-base] MODELS=${MODELS} TAU=${TAU:-calibrated}"
exec bash bigpurple/guided_i2sb_knee.sbatch
