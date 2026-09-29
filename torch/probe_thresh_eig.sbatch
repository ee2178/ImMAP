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
#SBATCH --job-name=ProbeThreshEig
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%A_%a.out
#SBATCH --error=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%A_%a.err
#SBATCH --array=0-1
#
# PROBE -- does the online ESPIRiT support threshold hurt the unrolled net?
#
#   0   lpdsnet  brain R16  online ESPIRiT thresh_eig 0.0    (the grid's setting
#                                                            since 2026-09-29)
#   1   lpdsnet  brain R16  online ESPIRiT thresh_eig 0.95   (the old setting)
#
# Both at kernel_size 4, the grid's kernel since 2026-09-29. ONLINE_SMAPS_KWS
# REPLACES the config's kws wholesale, so the kernel is spelled out below.
#
# 5000 gradient steps each (SWEEP_EPOCHS=100 x 50 steps/epoch for brain),
# cosine-annealed over those 5000, validated ONCE at the end (val fires when
# (epoch+1) % VAL_EVERY == 0). Everything else is the lpdsnet R16 cell's config.
#
# Motivation: CG-SENSE on the testbench (notebooks/espirit_acs_check.ipynb,
# section 13) looked better with thresh_eig=0 -- a 0.95 support zeroes the
# operator maps outside it, so nothing can be put there, while the SENSE ground
# truth (stored maps) may still be nonzero.
#
# Both arms are TAGGED (RUN_TAG below), so they land in their own run dirs
# (trained_nets/mg_recon/brain/lpdsnet_R16_te0_5k, ..._te095_5k) and never touch
# the full-length lpdsnet cell. Compare the two ARMS to each other; neither is
# comparable to a full-length run.
#
# Override the rate with ACCELS, e.g.  ACCELS=8 sbatch torch/probe_thresh_eig.sbatch
# (one rate per submit: the array indexes thresholds, not cells).
#
# SUBMIT FROM THE REPO ROOT.

THRESH=(0.0 0.95)
TAGS=(te0_5k te095_5k)
TASK="${SLURM_ARRAY_TASK_ID:-0}"
if [ "${TASK}" -ge "${#THRESH[@]}" ]; then
    echo "[probe] task ${TASK} but only ${#THRESH[@]} thresholds -- use --array=0-$(( ${#THRESH[@]} - 1 ))" >&2
    exit 1
fi

ANATOMY=brain
ONLY="lpdsnet"
ACCELS="${ACCELS:-16}"
case "${ACCELS}" in
    *" "*) echo "[probe] one rate per submit (ACCELS='${ACCELS}')" >&2; exit 1 ;;
esac
ORGAN_MASK=1                         # matches every other brain launcher --
                                     # regeneration rewrites every brain config
CONFIG_ROOT="config"
REGENERATE=1

SWEEP_EPOCHS=100                     # x 50 steps/epoch (brain) = 5000 steps
VAL_EVERY=100                        # = SWEEP_EPOCHS: validate once, at the end
RUN_TAG="${TAGS[${TASK}]}"
ONLINE_SMAPS_KWS="{\"thresh_eig\": ${THRESH[${TASK}]}, \"kernel_size\": 4, \"maxit\": 10}"
export VAL_EVERY ONLINE_SMAPS_KWS

# The body picks its cell by SLURM_ARRAY_TASK_ID. This array indexes the
# THRESHOLD, and the cell list (ONLY=lpdsnet, one rate) has exactly one entry,
# so every task is cell 0.
SLURM_ARRAY_TASK_ID=0
SLURM_ARRAY_TASK_MAX=0
echo "[probe] task ${TASK}: thresh_eig=${THRESH[${TASK}]} tag=${RUN_TAG} R=${ACCELS}"

# sbatch runs a COPY of this file from the node's spool dir; resolve the checkout.
for _c in "${IMMAP_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" "$HOME/scratch/ee2178/ImMAP"; do
    if [ -n "${_c}" ] && [ -f "${_c}/torch/_mg_recon_body.sh" ]; then
        IMMAP_ROOT="${_c}"
        break
    fi
done
if [ ! -f "${IMMAP_ROOT:-}/torch/_mg_recon_body.sh" ]; then
    echo "[probe] cannot locate the ImMAP checkout (tried IMMAP_ROOT," \
         "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset}, ~/scratch/ee2178/ImMAP)." >&2
    exit 1
fi

source "${IMMAP_ROOT}/torch/_mg_recon_body.sh"
