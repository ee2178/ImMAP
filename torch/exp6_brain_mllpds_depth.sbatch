#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=torch_pr_89_tandon_advanced
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=16
#SBATCH --mem=96GB
#SBATCH --time=48:00:00
#SBATCH --job-name=Exp6-MLDepth
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --array=0-1
#
# EXPERIMENT 6 -- ML-LPDS deeper instead of wider.  BRAIN R16.
#
#                     channels      K    params   rel FLOP  DC steps
#   0  mllpds48k54   48/96/192     54    244.4M    1.01x       53
#   1  mllpds64k40   64/128/256    40    321.7M    1.33x       39
#
#   reference: mllpds64 (exp4 cell 3, exp5 cell 1)
#                    64/128/256    30    241.2M    1.00x       29
#
# P = 7 and L = 3 in every cell; cascade init, slice projection, tau0 = 0.25.
#
# WHY. exp5 said cutting K costs more than width buys: at 64/128/256, K=30 beat
# K=20, and 128/256/512 at K=20 lost to mllpds64 despite 2.7x the parameters.
# At R=16 (~6% of k-space) the number of `E^H E` steps looks like the scarce
# resource, so these cells spend the budget on K instead of channels.
#
# READ IT AGAINST mllpds64, which is not re-run here:
#
#   mllpds64 vs mllpds48k54   WIDTH vs DEPTH at a fixed budget -- same params
#                             and FLOPs, one spent on channels, one on K
#   mllpds64 vs mllpds64k40   does the winner keep improving with K
#
# NO lpdsnet CONTROL AT MATCHED K, deliberately. The claim under test is that
# the multilevel formulation packs parameters in without costing time -- judge
# it on quality against wall-clock/inference time, not against flat LPDS at
# equal iteration count.
#
# MEMORY. Activations are stored for every unrolled layer, so training memory
# grows with K even when parameters do not: cell 0 carries ~1.8x mllpds64's
# activation memory at the same parameter count. Probe it before committing
# the queue (below).
#
# EVERYTHING ELSE IS HELD FIXED by the shared generator block: data, ESPIRiT
# maps, mask, sigma ~ U[0.04, 0.06], loss, LR, schedule, organ mask, R=16.
#
# HOW TO RUN
# ----------
#   # 50k iterations per cell (steps_per_epoch=50, so 1000 epochs) -- matches
#   # the exp4/exp5 looks, so mllpds64_R16_1000ep is the direct reference.
#   SWEEP_EPOCHS=1000 sbatch torch/exp6_brain_mllpds_depth.sbatch
#
#   # a quick memory shakeout of the deepest cell first
#   SWEEP_EPOCHS=20 sbatch --array=0 torch/exp6_brain_mllpds_depth.sbatch
#
#   # full length (the config's num_epochs)
#   sbatch torch/exp6_brain_mllpds_depth.sbatch
#
# A shortened run is TAGGED (<model>_R16_1000ep) and never lands in the
# full-length cell's dir. Re-submitting the same tagged cell is skipped by
# LAUNCH-ONCE -- pass FORCE_RESTART=1, or delete the run dir, to redo one.
#
# Rough wall-clock at 50k iterations, scaling the ~3 it/s measured for
# mllpdsw2 (48/96/192, K=30) by FLOPs: cell 0 ~8.3h, cell 1 ~10.7h.
#
# SUBMIT FROM THE REPO ROOT. `logs/` is relative to the SUBMIT directory; from
# torch/ the job dies before the script runs.

ANATOMY=brain
ONLY="${ONLY:-mllpds48k54 mllpds64k40}"
ACCELS="${ACCELS:-16}"
ORGAN_MASK=1                         # matches every other brain launcher --
                                     # regeneration rewrites every brain config,
                                     # so a mismatch silently un-masks the anatomy
CONFIG_ROOT="config"
SWEEP_EPOCHS="${SWEEP_EPOCHS:-}"     # empty = the config's num_epochs
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
    echo "[exp6] cannot locate the ImMAP checkout (tried IMMAP_ROOT," \
         "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset}, ~/scratch/ee2178/ImMAP)." >&2
    echo "[exp6] submit from the repo root, or: IMMAP_ROOT=/path/to/ImMAP sbatch $0" >&2
    exit 1
fi

source "${IMMAP_ROOT}/torch/_mg_recon_body.sh"
