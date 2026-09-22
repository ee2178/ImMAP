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
#SBATCH --job-name=Exp5-MLSize
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --array=0-3
#
# EXPERIMENT 5 -- how wide, and how deep, ML-LPDS wants to be.  BRAIN R16.
#
#                     channels      K    params   rel FLOP  DC steps
#   0  mllpdsw2      48/96/192     30    135.8M    1.00x       29
#   1  mllpds64      64/128/256    30    241.2M    1.77x       29
#   2  mllpds64k20   64/128/256    20    160.8M    1.18x       19
#   3  mllpds128k20  128/256/512   20    642.8M    4.68x       19
#
# P = 7 in every cell. The 7x7 atom is the model, not a knob to trade away, so
# capacity is bought with channels and paid for with K.
#
# READ IT IN THREE ADJACENT PAIRS. Each changes exactly one thing:
#
#   0 vs 1    WIDTH, at K = 30
#   1 vs 2    K, at 64/128/256
#   2 vs 3    WIDTH, at K = 20
#
# WHY THE LADDER IS SHAPED THIS WAY
# ---------------------------------
# params and FLOPs go as K * P^2 * sum_l M_{l-1} M_l -- linear in K, quadratic
# in width -- so dropping K is how a wider net becomes affordable. But K is not
# free: layer 0 is the cold start, so K=30 takes 29 `E^H E` steps and K=20 takes
# 19, and ML_LPDS_COMMON's K=30 exists precisely so ML-LPDS and `lpdsnet` take
# the SAME number, leaving the prior as the only difference between them. At
# R=16 -- ~6% of k-space -- data consistency is the scarce resource.
#
# Cell 1 is therefore the load-bearing one: it is the only cell that widens
# WITHOUT also cutting data consistency. If width helps, it should show there
# first. Cell 2 then prices what K costs at that width, and cell 3 spends the
# saving on the widest bank.
#
# 128/256/512 at K=30 would be 964.1M parameters and 7.02x the FLOPs -- 7x a
# model that currently loses to 1.00M-parameter `lpdsnet`. K=20 is what makes
# that width affordable at all, and cell 2 is what tells you whether the K you
# traded away was worth more than the width you bought.
#
# THE INIT IS NOT CONTROLLED HERE. Every cell runs PER-FILTER normalisation
# (`init_norm="level"`, tau0=0.5, the default since 2026-09-22). The earlier
# exp5 runs used the cascade init (tau0=0.25), so init moved together with the
# protocol -- another reason those numbers are not comparable.
#
# MEMORY. Cell 3 is the largest at 643M params: ~2.6 GB of weights, ~7.7 GB of
# Adam state, on a 48GB L40S. If it OOMs the lever is K, not width -- activation
# cost scales with K, parameter cost with width^2.
#
# EVERYTHING ELSE IS HELD FIXED by the shared generator block, which since
# 2026-09-22 is `--protocol measured`: MEASURED k-space + AWGN at
# sigma ~ U[0.04, 0.06], Sljiva masks (center_frac 0.04, adjust_accel -- R=16 is
# the EFFECTIVE rate, where every earlier exp5 run sampled at ~R=8), operator
# maps estimated per step by ESPIRiT from each measurement's 13-line ACS, the
# stored SENSE combination as ground truth, NO organ mask, same loss/LR/schedule.
# None of the pre-2026-09-22 exp5 numbers are comparable to these.
#
# ONLINE ESPIRiT COSTS MEMORY AND TIME EVERY STEP, on top of the model. Its
# bound is ~10 GB per slice (physics/online_smaps.py::estimate_cost_gb) and it
# has not been measured on this data -- probe cell 3, the largest, first.
#
# HOW TO RUN
# ----------
#   # 50k iterations per cell (steps_per_epoch=50, so 1000 epochs).
#   # SWEEP_EPOCHS also rescales the cosine T_max, so the LR schedule COMPLETES
#   # over the shortened run instead of being cut off partway down.
#   SWEEP_EPOCHS=1000 sbatch torch/exp5_brain_mllpds_size.sbatch
#
#   # full length (the config's 6000 epochs = 300k steps)
#   sbatch torch/exp5_brain_mllpds_size.sbatch
#
#   # one cell, or a subset
#   SWEEP_EPOCHS=1000 sbatch --array=1 torch/exp5_brain_mllpds_size.sbatch
#   SWEEP_EPOCHS=1000 sbatch --array=0,1 torch/exp5_brain_mllpds_size.sbatch
#
#   # a quick shakeout before committing the queue
#   SWEEP_EPOCHS=20 sbatch --array=3 torch/exp5_brain_mllpds_size.sbatch
#
# A shortened run is TAGGED: it lands in
# trained_nets/mg_recon/brain/<model>_R16_1000ep, never in the full-length
# cell's dir, and logs to wandb under the matching name. So a 50k probe and a
# later full run coexist. Re-submitting the SAME tagged cell is skipped by
# LAUNCH-ONCE -- pass FORCE_RESTART=1, or delete the run dir, to redo one.
#
# Rough wall-clock at 50k iterations, scaling the ~3 it/s measured for cell 0:
# cell 0 ~4.6h, cell 1 ~8h, cell 2 ~5.5h, cell 3 ~22h. All inside the 48h limit,
# but cell 3 is the one to watch -- start it with SWEEP_EPOCHS=20 if unsure.
#
# NOTE ON SUBMIT DIRECTORY: submit from the repo root. The body is resolved
# below, but `logs/` is relative to the SUBMIT directory; from torch/ the job
# dies before the script runs.

ANATOMY=brain
ONLY="${ONLY:-mllpdsw2 mllpds64 mllpds64k20 mllpds128k20}"
ACCELS="${ACCELS:-16}"
# ORGAN MASK OFF (2026-09-22), in EVERY launcher: loss and metrics run over the
# whole image. Regeneration rewrites every config of an anatomy, so these must
# all agree -- turn it back on everywhere or nowhere.
ORGAN_MASK=""                       # matches every other brain launcher --
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
    echo "[exp5] cannot locate the ImMAP checkout (tried IMMAP_ROOT," \
         "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset}, ~/scratch/ee2178/ImMAP)." >&2
    echo "[exp5] submit from the repo root, or: IMMAP_ROOT=/path/to/ImMAP sbatch $0" >&2
    exit 1
fi

source "${IMMAP_ROOT}/torch/_mg_recon_body.sh"
