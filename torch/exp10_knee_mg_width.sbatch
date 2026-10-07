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
#SBATCH --job-name=Exp10-KneeMGW
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%A_%a.out
#SBATCH --error=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%A_%a.err
#SBATCH --array=0-3
#
# EXPERIMENT 10 -- the MG-LPDS width sweep on KNEE, at R=8 and R=12, FULL LENGTH.
#
#   0   knee  R8   mg169v6   M=169   3.62M params   the full-width reference
#   1   knee  R8   mg81v6    M=81    1,733,424
#   2   knee  R12  mg169v6
#   3   knee  R12  mg81v6
#
# The knee counterpart of exp9 (torch/exp9_brain_mg_width.sbatch): the V-cycle
# structure is fixed at K=[6,[4,4,6]] and only the dictionary width M varies --
# the two ends of exp9's range. Both cells are exp8's: the REDISCRETIZED coarse
# Gram (coarse_op="rediscretize") and the PLANAR complex conv
# (training.complex_conv="planar"). They differ ONLY in M.
#
# PROTOCOL: exp2's (torch/exp2_knee_group.sbatch), which is brain's with the
# knee data block -- measured k-space + AWGN at sigma ~ U[0.01, 0.02] (val
# 0.015), Julia masks (center_frac 0.04, adjust_accel, so R is the EFFECTIVE
# rate), operator maps estimated online by ESPIRiT (thresh_eig 0, 4x4 kernel),
# stored SENSE combination as ground truth, organ mask on.
#
# TWO THINGS TO KNOW BEFORE READING THE NUMBERS
#   * knee volumes are 368 or 372 columns wide. 372 is not a multiple of 8 (the
#     network's stride), so those slices are solved on a 376-wide image
#     (E @ Truncate). The rediscretized Gram covers both since 2026-10-07: the
#     coarse levels of a 372-wide slice run at 186 and 93 columns
#     (operators/coarse.py::coarse_truncate; before that they fell back to
#     Galerkin). Checkpoints trained either way load into each other.
#   * the stored knee maps are the ORIGINAL files, not regenerated with the
#     fixed ESPIRiT as brain's were. They define the ground truth and the mask.
#
# NOT COMPARABLE TO exp2's mglpds: that cell uses the Galerkin Gram and gauss
# convs. mg169v6 (cells 0 and 2) is the M=169 reference for this sweep.
#
# FULL LENGTH: 6000 epochs x 50 steps = 300k steps, which will likely not fit in
# one 48 h slot. When a slot ends, RESUBMIT THIS FILE: LAUNCH-ONCE in
# torch/_mg_recon_body.sh resumes an unfinished cell from its checkpoint and
# skips finished or still-queued ones. Run dirs are untagged:
# trained_nets/mg_recon/knee/<cell>_R<r>.
#
# Probe one cell first if unsure of throughput (tagged, lands in <cell>_R8_20ep):
#     SWEEP_EPOCHS=20 sbatch --array=0 torch/exp10_knee_mg_width.sbatch
# One rate only (renumbers to 0-1):
#     ACCELS=12 sbatch --array=0-1 torch/exp10_knee_mg_width.sbatch
#
# SUBMIT FROM THE REPO ROOT.
#   sbatch torch/exp10_knee_mg_width.sbatch
#
# Check the index map any time:
#   python scripts/make_mg_recon_configs.py --list-cells --anatomy knee \
#       --accels 8 12 --only mg169v6 mg81v6

ANATOMY=knee
ONLY="${ONLY:-mg169v6 mg81v6}"
ACCELS="${ACCELS:-8 12}"
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
    echo "[exp10] cannot locate the ImMAP checkout (tried IMMAP_ROOT," \
         "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset}, ~/scratch/ee2178/ImMAP)." >&2
    echo "[exp10] submit from the repo root, or: IMMAP_ROOT=/path/to/ImMAP sbatch $0" >&2
    exit 1
fi

source "${IMMAP_ROOT}/torch/_mg_recon_body.sh"
