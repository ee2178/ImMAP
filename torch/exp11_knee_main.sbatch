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
#SBATCH --job-name=Exp11-KneeMain
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%A_%a.out
#SBATCH --error=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%A_%a.err
#SBATCH --array=0-7
#
# EXPERIMENT 11 -- the MAIN COMPARISON on KNEE, R=8 and R=12, FULL LENGTH.
#
#   0   knee  R8   lpdsnet      LPDSNet            K=30, M=169        1.00M params
#   1   knee  R8   varnetmaps   E2E-VarNet         on the operator's maps   29.5M
#   2   knee  R8   mg169v6      MGLPDS-Big         K=[6,[4,4,6]], M=169     3.62M
#   3   knee  R8   mg81v6       MGLPDS-Small       K=[6,[4,4,6]], M=81      1.73M
#   4   knee  R12  lpdsnet
#   5   knee  R12  varnetmaps
#   6   knee  R12  mg169v6
#   7   knee  R12  mg81v6
#
# The knee half of the VarNet / LPDSNet / multigrid table; the brain half is
# exp1 + exp8/exp9's cells of the same names. The two multigrid cells are the
# ends of exp9's width sweep: rediscretized coarse Gram, planar complex conv,
# differing only in M. This launcher SUPERSEDES exp10 (cells 2, 3, 6, 7 are
# exp10's four, in the same run dirs).
#
# PLANAR STATE ON (PLANAR_STATE=1, the default here): lpdsnet and both
# multigrid cells train with their iterates carried as real [re; im] tensors
# and the thresholding as one fused kernel, forward and backward
# (training.planar_state; models/mg_lpds.py, models/clip_triton.py). A speed
# setting only -- same parameters, same map to fp roundoff -- measured on an
# L40S at 640x320 / 20 coils as forward + backward 253 -> 93 ms (lpdsnet),
# 377 -> 253 ms (mg169v6), 188 -> 140 ms (mg81v6), at about half the memory.
# VarNet has no such switch and is untouched. PLANAR_STATE= (empty) turns it off.
#
# PROTOCOL: exp2's (torch/exp2_knee_group.sbatch), which is brain's with the
# knee data block -- measured k-space + AWGN at sigma ~ U[0.01, 0.02] (val
# 0.015), Julia masks (center_frac 0.04, adjust_accel, so R is the EFFECTIVE
# rate), operator maps estimated online by ESPIRiT (thresh_eig 0, 4x4 kernel),
# stored SENSE combination as ground truth, organ mask on.
#
# TWO THINGS TO KNOW BEFORE READING THE NUMBERS
#   * knee volumes are 368 or 372 columns wide. 372 is not a multiple of 8 (the
#     multigrid stride), so those slices are solved on a 376-wide image
#     (E @ Truncate), and their coarse levels run rediscretized at 186 and 93
#     columns (operators/coarse.py::coarse_truncate, since 2026-10-07). A knee
#     net trained before that date saw the Galerkin Gram on those slices.
#   * the stored knee maps are the ORIGINAL files, not regenerated with the
#     fixed ESPIRiT as brain's were. They define the ground truth and the mask.
#
# RERUNNING CELLS THAT ALREADY HAVE A RUN DIR. LAUNCH-ONCE
# (torch/_mg_recon_body.sh) never restarts a launched cell on its own: an
# existing trained_nets/mg_recon/knee/<cell>_R<r> is skipped if complete and
# resumed if not. To train everything again from scratch, either
#     mv trained_nets/mg_recon/knee trained_nets/mg_recon/knee_old   # keeps the old runs
# before submitting (check nothing is still running in it), or submit with
#     FORCE_RESTART=1                                                # OVERWRITES them
#
# FULL LENGTH: 6000 epochs x 50 steps = 300k steps. One 48 h slot holds them
# only above ~1.75 it/s. When a slot ends, RESUBMIT THIS FILE: LAUNCH-ONCE
# resumes an unfinished cell from its checkpoint (and keeps the planar state)
# and skips finished or still-queued ones.
#
# Probe one cell first -- it is the first real training run on the planar
# state, and its log gives the true it/s (tagged, lands in mg169v6_R8_20ep):
#     SWEEP_EPOCHS=20 sbatch --array=2 torch/exp11_knee_main.sbatch
# One rate only (renumbers to 0-3):
#     ACCELS=12 sbatch --array=0-3 torch/exp11_knee_main.sbatch
#
# SUBMIT FROM THE REPO ROOT.
#   sbatch torch/exp11_knee_main.sbatch
#
# Check the index map any time:
#   python scripts/make_mg_recon_configs.py --list-cells --anatomy knee \
#       --accels 8 12 --only lpdsnet varnetmaps mg169v6 mg81v6

ANATOMY=knee
ONLY="${ONLY:-lpdsnet varnetmaps mg169v6 mg81v6}"
ACCELS="${ACCELS:-8 12}"
PLANAR_STATE="${PLANAR_STATE-1}"     # `-`, not `:-`: an explicitly EMPTY value turns it off
ATTN="flex"                          # matches the other knee launchers --
ORGAN_MASK=1                         # regeneration rewrites every knee config,
                                     # so a mismatch silently changes the anatomy
CONFIG_ROOT="config"
SWEEP_EPOCHS="${SWEEP_EPOCHS:-}"     # empty = the config's num_epochs (full length)
REGENERATE=1                         # 1 = regenerate configs before running

# sbatch runs a COPY of this file from the node's spool dir, so a
# $BASH_SOURCE-relative path looks for the body next to that copy -- which was
# never shipped. Resolve the checkout explicitly.
for _c in "${IMMAP_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" "$HOME/scratch/ee2178/ImMAP"; do
    if [ -n "${_c}" ] && [ -f "${_c}/torch/_mg_recon_body.sh" ]; then
        IMMAP_ROOT="${_c}"
        break
    fi
done
if [ ! -f "${IMMAP_ROOT:-}/torch/_mg_recon_body.sh" ]; then
    echo "[exp11] cannot locate the ImMAP checkout (tried IMMAP_ROOT," \
         "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset}, ~/scratch/ee2178/ImMAP)." >&2
    echo "[exp11] submit from the repo root, or: IMMAP_ROOT=/path/to/ImMAP sbatch $0" >&2
    exit 1
fi

source "${IMMAP_ROOT}/torch/_mg_recon_body.sh"
