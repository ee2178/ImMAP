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
#SBATCH --job-name=JointLGGS-knee
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%A_%a.out
#SBATCH --error=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%A_%a.err
#SBATCH --array=0-2           # MUST equal the cell count; checked at runtime
#
# EQUAL GUIDE / TARGET ACCELERATION, knee: PDFS target at R = 8, PD guide at R = 8 too
# (equispaced, no ACS; its line phase redrawn every training step, and at validation on
# COMPLEMENTARY lines, offset 4: midway between the target's). Cells
# (scripts/make_guided_recon_configs.py --guide-accels 8):
#
#     0 joint_lggs_R8_g8    JOINT LGGS (models/joint_lggs.py): target and guide reconstructed
#                           together -- per-contrast dictionaries, 10 independent-LPDS warm-up
#                           layers, then 20 layers of ONE group threshold on both images' codes.
#                           No second network for the guide
#     1 tgvn_c6_R8_g8       TGVN, 6 cascades, on the same accelerated guide
#     2 lggs_R8_g8          plain LGGS fed the ALIASED guide directly -- an ablation, not
#                           published LGGS (which would reconstruct the guide with another net)
#
# The floor is lpdsnet_R8 (no guide) and the ceilings lggs_R8_g1 / tgvn_c6_R8_g1 (fully-sampled
# guide), all in torch/guided_recon_knee.sbatch. Everything else is that launcher, which this
# runs: generator, protocol (vlow noise, adjust_accel off, per-volume normalization, offline
# SENSE target), run dirs trained_nets/guided_recon/knee/<RUN_TAG>_noise-<preset>/<cell>, resume,
# FORCE_RESTART. The guide is scored only through the target; JointLGGSNet also returns its guide
# reconstruction (second output), which training does not supervise yet.
#
# Submit:          sbatch torch/joint_lggs_knee.sbatch
# Some cells:      MODELS="joint_lggs tgvn_c6" sbatch --array=0-1 --export=ALL torch/joint_lggs_knee.sbatch
# Timing first:    python scripts/time_guided_nets.py --no-measure --cells joint_lggs_R8_g8 lggs_R8_g1 tgvn_c6_R8_g8

export MODELS="${MODELS:-joint_lggs tgvn_c6 lggs}"
export GUIDE_ACCELS="${GUIDE_ACCELS:-8}"
IMMAP_ROOT="${IMMAP_ROOT:-${HOME}/scratch/ee2178/ImMAP}"
cd "${IMMAP_ROOT}" || { echo "no such dir: ${IMMAP_ROOT}" >&2; exit 1; }
echo "[joint-lggs] MODELS=${MODELS} GUIDE_ACCELS=${GUIDE_ACCELS}"
exec bash torch/guided_recon_knee.sbatch
