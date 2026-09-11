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
#SBATCH --job-name=Exp4-BrainML
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --array=0-4
#
# EXPERIMENT 4 -- multilevel LPDS against flat LPDS and the V-cycle, BRAIN R8.
#
#   0   brain  R8  lpdsnet    flat LPDS, K=30
#   1   brain  R8  mglpds     V-cycle, K=[6,[4,4,6]]
#   2   brain  R8  varnet     E2E-VarNet baseline
#   3   brain  R8  mllpds     ML-LPDS, K=6, L=3, widen=1, channels 70/70/70
#   4   brain  R8  mllpdsw2   ML-LPDS, K=6, L=3, widen=2, channels 40/80/160
#
# Every arm trains here, from scratch: no existing brain run uses the
# regenerated ESPIRiT maps. Data, mask, sigma ~ U[0.01, 0.02], loss, LR,
# schedule and organ mask all come from the same generator block.
#
# SHARED CELLS. Cells 0-2 are also cells of exp1 (brain R8). The body launches
# each cell once -- in whichever launcher reaches it first -- and skips it in
# the other (LAUNCH-ONCE in torch/_mg_recon_body.sh), so submitting both
# experiments trains each shared arm exactly once.
#
# WHAT THE COMPARISON ISOLATES
# ----------------------------
# The four LPDS cells run the same primal-dual iteration (LPDS / Condat-Vu).
# They differ in how the coarse grids enter:
#
#                                                          GFLOP   params
#   lpdsnet    one level, no coarse grids                   20.6    1.00M
#   mglpds     coarse grids as a SOLVER (V-cycle, FAS)      33.8    3.62M
#   mllpds     coarse grids as a PRIOR (one dual/level)     33.5   11.6M
#   mllpdsw2   the same, channels doubling per level        34.3   18.9M
#
# The ML cells are FLOP-matched to mglpds (conv + FFT per forward, 160x160,
# 16 coils) as an offline proxy for inference time, so (mglpds vs mllpds) is
# solver-vs-prior at equal compute and (mllpds vs mllpdsw2) is the widening
# question at equal compute. They carry 3-5x mglpds's parameters to get there.
# VarNet is the "is any of this competitive" check, not part of the question.
#
# FLOPs ARE NOT TIME. mglpds also does ~5.4 GB of memory-bound elementwise
# traffic per forward against ~0.4 GB here, so these cells will probably run
# faster on a GPU than the FLOP match suggests. Time a few forward passes
# before calling the comparison compute-matched.
#
# K=6: one layer is one gradient step plus one dual step, ~6 convolutions at
# L=3. Layer 0 is the cold start, so these cells take 5 data-consistency steps
# against lpdsnet's 29 and mglpds's ~two dozen on the fine grid alone. The
# arithmetic is in the OPT_IN block of scripts/make_mg_recon_configs.py.
#
# OPT-IN CELLS. The ML cells do not appear in the default --list-cells output,
# so the full grid's indices (which mg_recon_brain.sbatch uses) are unchanged.
# They are listed only when named:
#   python scripts/make_mg_recon_configs.py --list-cells --anatomy brain \
#       --accels 8 --only lpdsnet mglpds varnet mllpds mllpdsw2
#
# SHORT RUNS AND OTHER ACCELERATIONS
# ----------------------------------
# ONLY, ACCELS and SWEEP_EPOCHS below take their value from the sbatch line when
# one is set. An epoch is 50 steps, so all five arms at R=16 for 50k steps is
#
#     ACCELS=16 SWEEP_EPOCHS=1000 sbatch torch/exp4_brain_mllpds.sbatch
#
# Same budget and the same 1000-epoch cosine schedule for every arm, so it is a
# like-for-like look. Each lands in its own tagged run dir and wandb name
# (<model>_R16_1000ep), so the full-length R16 cells (exp3) are untouched.
#
# BEFORE SUBMITTING
#   * move the pre-ESPIRiT brain runs aside -- see exp1's BEFORE SUBMITTING;
#     LAUNCH-ONCE cannot tell maps rewritten in place from current ones.
#   * the ML cells have never run. Probe one first:
#       SWEEP_EPOCHS=20 sbatch --array=3 torch/exp4_brain_mllpds.sbatch
#     (tagged, so it lands in mllpds_R8_20ep; a repeat is skipped -- FORCE_RESTART=1)
#   * `python -m tests.test_ml_lpds` on the node is cheap and covers the model.
#
# SUBMIT FROM THE REPO ROOT. `#SBATCH --output=logs/...` resolves against the
# SUBMIT directory; from torch/ the job dies before the script runs.

ANATOMY=brain
ONLY="${ONLY:-lpdsnet mglpds varnet mllpds mllpdsw2}"
ACCELS="${ACCELS:-8}"
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
    echo "[exp4] cannot locate the ImMAP checkout (tried IMMAP_ROOT," \
         "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset}, ~/scratch/ee2178/ImMAP)." >&2
    echo "[exp4] submit from the repo root, or: IMMAP_ROOT=/path/to/ImMAP sbatch $0" >&2
    exit 1
fi

source "${IMMAP_ROOT}/torch/_mg_recon_body.sh"
