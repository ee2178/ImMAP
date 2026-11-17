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
#SBATCH --job-name=GuideAccel-knee
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%A_%a.out
#SBATCH --error=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%A_%a.err
#SBATCH --array=0-2           # MUST equal the cell count; checked at runtime
#
# ACCELERATED GUIDE, knee: PDFS target at R = 8, PD guide at R = 4 (equispaced, no ACS; its
# line phase redrawn every training step, 0 at validation). The first step toward guides that
# are not fully sampled -- TGVN does not test deeper guide acceleration and LGGS does not plan
# for it. Cells (scripts/make_guided_recon_configs.py --guide-accels 4):
#
#     0 lggs_trust_R8_g4    LGGS with TRUST-GUIDED CODES: each layer's guide codes from
#                           x + P'_delta (w - x), TGVN's ambiguous-space projector -- the target's
#                           own estimate where its data constrain it, the (aliased) guide where
#                           they do not. One learned delta, no transfer net (models/guided_lpds.py)
#     1 lggs_R8_g4          plain LGGS fed the ALIASED guide directly -- an ablation only, NOT
#                           published LGGS, which first reconstructs an accelerated guide with a
#                           separate network and guides with that (a two-stage baseline, not built)
#     2 tgvn_c6_R8_g4       TGVN, 6 cascades (its trust guidance on the accelerated guide)
#
# The fully-sampled-guide cells (lggs_R8_g1, tgvn_c6_R8_g1) are the ceiling; they run in
# torch/guided_recon_knee.sbatch. Everything else is that launcher, which this runs: generator,
# protocol (vlow noise, adjust_accel off, per-volume normalization, offline SENSE target), run
# dirs trained_nets/guided_recon/knee/<RUN_TAG>_noise-<preset>/<cell>, resume, FORCE_RESTART.
#
# NOTE the two masks share their phase at validation (both offset 0): the target's lines every
# 8th are a subset of the guide's every 4th, so the two aliasing patterns line up. Complementary
# offsets would decorrelate them -- a later experiment.
#
# Submit:          sbatch torch/guide_accel_knee.sbatch
# Some cells:      MODELS="lggs_trust tgvn_c6" sbatch --array=0-1 --export=ALL torch/guide_accel_knee.sbatch
# Other guide R:   GUIDE_ACCELS=8 sbatch --export=ALL torch/guide_accel_knee.sbatch

export MODELS="${MODELS:-lggs_trust lggs tgvn_c6}"
export GUIDE_ACCELS="${GUIDE_ACCELS:-4}"
IMMAP_ROOT="${IMMAP_ROOT:-${HOME}/scratch/ee2178/ImMAP}"
cd "${IMMAP_ROOT}" || { echo "no such dir: ${IMMAP_ROOT}" >&2; exit 1; }
echo "[guide-accel] MODELS=${MODELS} GUIDE_ACCELS=${GUIDE_ACCELS}"
exec bash torch/guided_recon_knee.sbatch
