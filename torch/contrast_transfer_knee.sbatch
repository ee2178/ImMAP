#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=torch_pr_89_tandon_advanced
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --job-name=ContrastTransfer-knee
#SBATCH --output=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%A_%a.out
#SBATCH --error=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%A_%a.err
#SBATCH --array=0-5           # MUST equal the cell count; checked at runtime
#
# PD -> PDFS GUIDE CONTRAST TRANSFER, pretraining sweep (knee R8). Tiny nets that map the PD guide,
# exactly as a guided recon net receives it (coil-combined under the operator's online maps), onto
# the fully-sampled PDFS target -- so LGGS, which pools by similarity IN THE GUIDE, can use a
# different-contrast guide. models/contrast_transfer.py, training/contrast_transfer.py; cells from
# scripts/make_guided_recon_configs.py (TRANSFER_NETS):
#
#     0 ct_pixel_R8_g1        per-pixel MLP, 1x1 convs, 32 wide, 3 layers  1.2k   |g| -> |T(g)|
#     1 ct_patch3_R8_g1       3x3 conv, then 1x1 layers                    1.4k   |g| -> |T(g)|
#     2 ct_patch7_R8_g1       7x7 conv, then 1x1 layers                    2.7k   |g| -> |T(g)|
#     3 ct_unet8_R8_g1        2-level UNet, 8 channels                     6.5k   |g| -> |T(g)|
#     4 ct_unet16_R8_g1       2-level UNet, 16 channels                    26k    |g| -> |T(g)|
#     5 ct_unet16_cplx_R8_g1  2-level UNet, 16 channels, complex in / out  26k   g -> T(g)
#
# Magnitude nets keep the guide's phase (LGGS's similarity is phase-invariant). Every net starts at
# the identity (residual, zero-initialised last layer). 20k Adam steps (1e-3, cosine), batch 4, L1
# on |.|; validation on 200 fixed val slices: val/psnr, val/ssim of |T(guide)| vs the target, and
# val/guide_psnr -- the raw guide after its best per-slice gain, the floor to beat.
# Checkpoints: net.ckpt (latest), net_bestval.ckpt (best val PSNR; what LGGS loads).
#
# THEN LGGS with the chosen transfer frozen in front of its guide (same NOISE as here):
#     MODELS="lggs_ct_unet8" sbatch --array=0 --export=ALL torch/guided_recon_knee.sbatch
#
# Everything else is torch/guided_recon_knee.sbatch, which this runs: generator, protocol (same
# measurement, noise, maps and per-volume normalization as the recon sweep), run dirs
# trained_nets/guided_recon/knee/<RUN_TAG>/<cell>, resume, FORCE_RESTART, NOISE.
#
# Submit:          sbatch torch/contrast_transfer_knee.sbatch
# Some cells:      MODELS="ct_pixel ct_unet8" sbatch --array=0-1 --export=ALL torch/contrast_transfer_knee.sbatch

export MODELS="${MODELS:-ct_pixel ct_patch3 ct_patch7 ct_unet8 ct_unet16 ct_unet16_cplx}"
IMMAP_ROOT="${IMMAP_ROOT:-${HOME}/scratch/ee2178/ImMAP}"
cd "${IMMAP_ROOT}" || { echo "no such dir: ${IMMAP_ROOT}" >&2; exit 1; }
echo "[contrast-transfer] MODELS=${MODELS}"
exec bash torch/guided_recon_knee.sbatch
