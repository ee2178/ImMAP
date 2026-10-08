#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=torch_pr_89_tandon_advanced
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=h200
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=48:00:00
#SBATCH --job-name=Exp12-MGGroup
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%A_%a.out
#SBATCH --error=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%A_%a.err
#SBATCH --array=0-3
#
# EXPERIMENT 12 -- the MULTIGRID-GROUP net, BRAIN and KNEE, R=8 and R=12,
# FULL LENGTH, on an H200.
#
#   0   brain  R8    mggroup169v6
#   1   brain  R12   mggroup169v6
#   2   knee   R8    mggroup169v6
#   3   knee   R12   mggroup169v6
#
# mggroup169v6 is MGLPDS-Big (mg169v6) with the nonlocal GroupThreshold in the
# prox slot, and NOTHING ELSE changed: K=[6,[4,4,6]], M=169, the REDISCRETIZED
# coarse Gram and the PLANAR complex conv. So against the main comparison
# (brain: exp1 / exp8 / exp9 cells; knee: torch/exp11_knee_main.sbatch) it
# differs from mg169v6 only in the prox -- "does nonlocal self-similarity help,
# on top of the multigrid net we report".
#
# It is NOT `mggrouplpds` (torch/exp2_knee_mggroup.sbatch). That cell is the
# group twin of the older `mglpds` -- Galerkin Gram, gauss convs -- and is not
# comparable to mg169v6.
#
# THE GROUP SETTINGS are `mggrouplpds`'s: window=15 (the coarse levels supply
# the long range), Mh=64, dK=5, one head, sim_fun="distance" on the flex backend
# (flex cannot form a phase-invariant similarity on complex features), and
# LR 2e-4 rather than 5e-4 (the generator keys that on the cell carrying
# attention; group cells kept tripping the backtrack at the shared rate).
#
# PROTOCOL: the main comparison's, per anatomy -- measured k-space + AWGN at
# sigma ~ U[0.01, 0.02] (val 0.015), Julia masks (center_frac 0.04,
# adjust_accel, so R is the EFFECTIVE rate), operator maps estimated online by
# ESPIRiT (thresh_eig 0, 4x4 kernel), stored SENSE combination as ground
# truth, organ mask on. 6000 epochs x 50 steps.
#
# NO PLANAR STATE HERE. `training.planar_state` (and with it the fused
# thresholding kernel) is implemented for the local prox only, so these cells
# train on the complex state; PLANAR_STATE is ignored for them. The planar
# CONV is on, as in mg169v6.
#
# H200, AND WHY ONE FILE CAN HOLD ALL FOUR: the group prox runs windowed
# attention at three grid levels and did not fit an L40S (48 GB) as
# `mggrouplpds`. Every cell here needs the same node type, so they share one
# array; `--constraint` cannot vary per task, which is why the local cells are
# in other files. Say the arm ran on different hardware if timing is reported.
#
# THIS CELL HAS NEVER BEEN TRAINED. Probe one task first and watch memory and
# it/s (tagged; lands in trained_nets/mg_recon/knee/mggroup169v6_R8_20ep):
#     SWEEP_EPOCHS=20 sbatch --array=2 torch/exp12_mggroup.sbatch
# flex compiles on first use; expect a slow first step, not a hang.
#
# A walltime kill is fine: RESUBMIT THIS FILE and LAUNCH-ONCE
# (torch/_mg_recon_body.sh) resumes each unfinished cell from its checkpoint
# and skips finished or still-queued ones.
#
# OTHER SELECTIONS. The array runs ANATOMIES x ACCELS x ONLY, anatomy-major,
# then acceleration, then cell -- size the array to the product:
#     ONLY="mggroup169v6 mggroup81v6" sbatch --array=0-7 torch/exp12_mggroup.sbatch
#         both widths (mggroup81v6 = the group twin of MGLPDS-Small, mg81v6)
#     ANATOMIES=knee sbatch --array=0-1 torch/exp12_mggroup.sbatch
#     ACCELS=8 sbatch --array=0-1 torch/exp12_mggroup.sbatch
#
# SUBMIT FROM THE REPO ROOT.
#   sbatch torch/exp12_mggroup.sbatch
#
# Index map within one anatomy:
#   python scripts/make_mg_recon_configs.py --list-cells --anatomy knee \
#       --accels 8 12 --only mggroup169v6

ANATOMIES="${ANATOMIES:-brain knee}"
ONLY="${ONLY:-mggroup169v6}"
ACCELS="${ACCELS:-8 12}"
ATTN="flex"                          # stated, not inherited: the run dir should
                                     # record which backend produced it
ORGAN_MASK=1                         # matches every other launcher of either
                                     # anatomy -- regeneration rewrites every
                                     # config of the anatomy it touches
CONFIG_ROOT="config"
SWEEP_EPOCHS="${SWEEP_EPOCHS:-}"     # empty = the config's num_epochs (full length)
REGENERATE=1                         # 1 = regenerate configs before running

# ---- one array over several anatomies ---------------------------------------------------
# The shared body runs ONE anatomy and indexes that anatomy's own cell list
# (ACCELS-major, then ONLY, in the generator's order). So split this array's
# index into (anatomy, index within the anatomy) and hand the body the second.
: "${SLURM_ARRAY_TASK_ID:?must be run as an array job (sbatch sets SLURM_ARRAY_TASK_ID)}"
_anat=(${ANATOMIES//,/ })
_only=(${ONLY//,/ })
_acc=(${ACCELS//,/ })
_per=$(( ${#_only[@]} * ${#_acc[@]} ))
_total=$(( _per * ${#_anat[@]} ))
if [ "${_per}" -eq 0 ] || [ "${SLURM_ARRAY_TASK_ID}" -ge "${_total}" ]; then
    echo "[exp12] task ${SLURM_ARRAY_TASK_ID} is outside the ${_total} cells of" \
         "ANATOMIES=\"${ANATOMIES}\" x ACCELS=\"${ACCELS}\" x ONLY=\"${ONLY}\"" \
         "-- fix --array (should be 0-$(( _total - 1 )))" >&2
    exit 1
fi
if [ "${SLURM_ARRAY_TASK_MAX:-0}" -lt "$(( _total - 1 ))" ] \
        && [ "${SLURM_ARRAY_TASK_ID}" -eq "${SLURM_ARRAY_TASK_MAX:--1}" ]; then
    echo "[exp12] NOTE: --array tops out at ${SLURM_ARRAY_TASK_MAX} but the selection" \
         "has ${_total} cells; the rest are NOT being run." >&2
fi
ANATOMY="${_anat[$(( SLURM_ARRAY_TASK_ID / _per ))]}"
echo "[exp12] array task ${SLURM_ARRAY_TASK_ID} of ${_total} -> ${ANATOMY}," \
     "cell $(( SLURM_ARRAY_TASK_ID % _per )) of that anatomy's ${_per}"
SLURM_ARRAY_TASK_ID=$(( SLURM_ARRAY_TASK_ID % _per ))
SLURM_ARRAY_TASK_MAX=$(( _per - 1 ))

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
    echo "[exp12] cannot locate the ImMAP checkout (tried IMMAP_ROOT," \
         "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset}, ~/scratch/ee2178/ImMAP)." >&2
    echo "[exp12] submit from the repo root, or: IMMAP_ROOT=/path/to/ImMAP sbatch $0" >&2
    exit 1
fi

source "${IMMAP_ROOT}/torch/_mg_recon_body.sh"
