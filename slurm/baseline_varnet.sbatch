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
#SBATCH --job-name=VarNet-base
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --array=0-3
#
# BASELINE -- fastMRI E2E-VarNet on every cell the experiments use.
#
#   0   knee   R8      matches exp2 (slurm/exp2_knee_group.sbatch)
#   1   knee   R4
#   2   brain  R8      matches exp1 (slurm/exp1_brain_mglpds.sbatch)
#   3   brain  R4
#
# One array covers both experiments: the baseline is the same model everywhere,
# so splitting it per experiment would only create two files that must be kept
# in step. Submit a subset with --array if you only need part of it, e.g.
#
#     sbatch --array=2-3 slurm/baseline_varnet.sbatch      # brain only (exp 1)
#     sbatch --array=0   slurm/baseline_varnet.sbatch      # knee R8 only (exp 2)
#
# WHAT IS AND IS NOT HELD CONSTANT
# --------------------------------
# Held: the data, the mask (R, acs_lines, mask_dist), sigma ~ U[0, 0.01], the
# loss, the LR, the schedule and the step budget -- all inherited from the same
# generator block as every other cell, so the training budget is equal.
#
# NOT held, and these are properties of the METHOD rather than choices:
#
#   * VarNet ESTIMATES its own coil maps. The unrolled cells are handed the
#     dataset's. Joint sensitivity estimation is what E2E-VarNet IS, so feeding
#     it maps would be a different method -- but it means VarNet is solving a
#     harder problem, and the comparison is biased TOWARDS the unrolled nets.
#     Say so when reporting; do not quietly claim a clean win.
#
#   * It returns an RSS MAGNITUDE image. PSNR / SSIM / NRMSE all take .abs()
#     first so those are valid, but there is no phase to compare and any
#     phase-sensitive claim is out of scope for this arm.
#
#   * It has no noise-adaptive parameter. sigma varies across the grid and the
#     unrolled cells condition their thresholds on it; VarNet cannot. That is a
#     real handicap and part of what the comparison measures -- worth checking
#     whether the gap widens with sigma before attributing it to architecture.
#
# It also carries roughly an order of magnitude more parameters than the flat
# LPDSNet. Each generated config records `count_parameters`.
#
# ARCHITECTURE IS UPSTREAM'S, UNTUNED
# -----------------------------------
# 12 cascades, chans=18, sens_chans=8 -- the published fastMRI defaults. Not
# retuned to this grid on purpose: a baseline that has been fiddled with until
# it stops winning is not a baseline. If it underperforms, the first thing to
# check is whether it has converged (VarNet is usually trained far longer than
# these 300k steps), not whether the width is wrong.
#
# The model code is vendored under models/e2evarnet/_fastmri/ and verified
# byte-identical to the clone except for two import lines --
# `python -m tests.test_e2evarnet` re-checks that, and re-checks that ImMAP's
# and fastMRI's FFT conventions still agree.
#
# BEFORE SUBMITTING
#     python -m tests.test_e2evarnet
#     SWEEP_EPOCHS=20 sbatch --array=0 slurm/baseline_varnet.sbatch
#
# SUBMIT FROM THE REPO ROOT -- `#SBATCH --output=logs/...` resolves against the
# SUBMIT directory, so from slurm/ the job dies before the script runs.
#
# Index map:
#   python scripts/make_mg_recon_configs.py --list-cells --only varnet

ONLY="varnet"
CONFIG_ROOT="config"
SWEEP_EPOCHS=""                      # empty = the config's num_epochs
REGENERATE=1

# This array spans BOTH anatomies, which the shared body does not: it indexes
# one anatomy's cell list. Resolve the cell here and hand the body a single
# (anatomy, accel) so its own bound check still applies.
case "${SLURM_ARRAY_TASK_ID:-0}" in
    0) ANATOMY=knee;  ACCELS="8" ;;
    1) ANATOMY=knee;  ACCELS="4" ;;
    2) ANATOMY=brain; ACCELS="8" ;;
    3) ANATOMY=brain; ACCELS="4" ;;
    *) echo "[varnet] task ${SLURM_ARRAY_TASK_ID} is outside 0-3." >&2; exit 1 ;;
esac
# The body indexes the FILTERED list, which is one cell long here.
SLURM_ARRAY_TASK_ID=0
export SLURM_ARRAY_TASK_ID

# sbatch runs a COPY of this file from the node's spool dir, so a
# $BASH_SOURCE-relative path looks for the body next to that copy.
for _c in "${IMMAP_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" "$HOME/scratch/ee2178/ImMAP"; do
    if [ -n "${_c}" ] && [ -f "${_c}/slurm/_mg_recon_body.sh" ]; then
        IMMAP_ROOT="${_c}"
        break
    fi
done
if [ ! -f "${IMMAP_ROOT:-}/slurm/_mg_recon_body.sh" ]; then
    echo "[varnet] cannot locate the ImMAP checkout (tried IMMAP_ROOT," \
         "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset}, ~/scratch/ee2178/ImMAP)." >&2
    echo "[varnet] submit from the repo root, or: IMMAP_ROOT=/path/to/ImMAP sbatch $0" >&2
    exit 1
fi

echo "[varnet] baseline cell: ${ANATOMY} R${ACCELS}"
source "${IMMAP_ROOT}/slurm/_mg_recon_body.sh"
