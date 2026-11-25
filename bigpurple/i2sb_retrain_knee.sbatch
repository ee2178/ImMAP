#!/bin/bash
#
# I2SB BASELINE RETRAINING, fastMRI knee, on BIGPURPLE: the settings that replace the collapsed
# sbunet_inv run (scripts/make_guided_recon_configs.py, "RETRAINING THE I2SB BASELINE"). PDFS
# target, the PD guide as the bridge start (on its own maps), R = 8, one tau for every cell
# (0.19, sbunet_inv's) so the runs compare. One array task per cell:
#
#     0 sbunet_small       small I2SB UNet (~23M, the BraTS one's size), sees x_t only
#     1 sbunet_small_x1    the same UNet, ALSO sees the guide x1 at every step (cond_x1)
#     2 sbunet_inv_stable  the BASELINE: I2SB-Inversion's 553M ADM (eps head, x_t only), at their
#                          lr 1e-5 with EMA 0.99 and 2 crops per step -- the network that
#                          collapsed at 5e-4, trained so that it cannot repeat it
#     3 sbunet_inv_x1      the same full-size ADM and recipe, ALSO seeing the guide x1 at every
#                          step (cond_x1) -- task 2 vs 3 is the guide channel at full scale
#
# Shared by every cell: a 5k-step linear warmup then cosine, clip 1.0, 320 x 320 crops,
# train/grad_norm + train/zero_grad_frac logged, the collapse alarm (dead_grad_alarm 0.5: the run
# stops if the net is cut off from its loss again), validation at a seeded random step per slice,
# and OFFLINE, PHASE-CORRECTED training pairs: both scans on the offline PDFS maps, each rotated by
# its own centre-block phase, no online ESPIRiT per step (training/i2sb_recon.py "OFFLINE PAIRS");
# the operator's online maps are phase-corrected to match.
# The small UNets: Adam 1e-4, 4 crops per step. Other cells: sbcdl (SBCDLNet on [x_t, x1]).
#
# Everything else -- config generation, data checks, resume from the run dir, FORCE_RESTART -- is
# bigpurple/guided_i2sb_knee.sbatch, which this runs with its own MODELS / TAU. Keep --array
# below at 0..(number of MODELS - 1).
#
# Submit all:    sbatch bigpurple/i2sb_retrain_knee.sbatch
# One cell:      sbatch --array=2 bigpurple/i2sb_retrain_knee.sbatch                (sbunet_inv_stable)
#                sbatch --array=3 bigpurple/i2sb_retrain_knee.sbatch                (sbunet_inv_x1)
# Other cells:   MODELS="sbunet_small sbunet_small_x1 sbunet_inv_stable sbcdl" \
#                    sbatch --array=0-3 --export=ALL bigpurple/i2sb_retrain_knee.sbatch
# Start over:    FORCE_RESTART=1 sbatch --export=ALL bigpurple/i2sb_retrain_knee.sbatch
# The sbunet_small_x1 family (2026-11-25), support-masked (loss + metrics on the anatomy, the
# sampler's hard support at inference; runs in s4_noise-vlow_supp/):
#                SUPPORT_MASK=1 MODELS="sbunet_small_x1 sbunet_small_x1_zf sbunet_small_x1_cg sbunet_small_x1_zf_cg" \
#                    sbatch --array=0-3 --export=ALL bigpurple/i2sb_retrain_knee.sbatch
# I2SB-TGVN on the same recipe (BridgeTGVN, 6 cascades, its own DC, no CG; one full slice / step):
#                SUPPORT_MASK=1 MODELS="sbtgvn_c6_rt_guide sbtgvn_c6_rt_zf" \
#                    sbatch --array=0-1 --export=ALL bigpurple/i2sb_retrain_knee.sbatch

#SBATCH --partition=radiology
#SBATCH --job-name=i2sb-retrain
#SBATCH --array=0-3
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

export MODELS="${MODELS:-sbunet_small sbunet_small_x1 sbunet_inv_stable sbunet_inv_x1}"
export TAU="${TAU:-0.19}"
[ -n "${SLURM_ARRAY_TASK_ID}" ] || { echo "[i2sb-retrain] submit as an array job" >&2; exit 1; }
echo "[i2sb-retrain] MODELS=\"${MODELS}\" TAU=${TAU}, task ${SLURM_ARRAY_TASK_ID}"
# the shared body (a plain bash script here: its #SBATCH lines are comments), in the job's chdir
exec bash bigpurple/guided_i2sb_knee.sbatch
