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
#SBATCH --array=0-5
#
# EXPERIMENT 4 -- the multilevel nets against flat LPDS and the V-cycle, BRAIN R16.
#
#   0   brain  R16 lpdsnet    flat LPDS, K=30
#   1   brain  R16 mglpds     V-cycle, K=[6,[4,4,6]]
#   2   brain  R16 varnet     E2E-VarNet baseline
#   3   brain  R16 mllpds     ML-LPDS,        K=30, L=3, widen=1, channels 96/96/96
#   4   brain  R16 mllpdsw2   ML-LPDS,        K=30, L=3, widen=2, channels 48/96/192
#   5   brain  R16 mlcdlw2    ML-CDLNet,      K=18, L=3, widen=2, channels 48/96/192
#
# Every arm trains here, from scratch: no existing brain run uses the
# regenerated ESPIRiT maps. Data, mask, sigma ~ U[0.01, 0.02], loss, LR,
# schedule and organ mask all come from the same generator block.
#
# R=16 leaves ~6% of k-space, where exp3 expects the coarse grids to matter
# most -- so it is the acceleration most likely to separate a multilevel PRIOR
# from a V-cycle and from flat LPDS. It used to be R8; the R8 cells are one
# override away (ACCELS=8, below) and land in their own run dirs.
#
# SHARED CELLS. Cells 0-2 are also exp3's cells (brain R16). The body launches
# each cell once -- in whichever launcher reaches it first -- and skips it in
# the other (LAUNCH-ONCE in torch/_mg_recon_body.sh), so submitting both
# experiments trains each shared arm exactly once.
#
# WHAT THE COMPARISON ISOLATES
# ----------------------------
# The four LPDS cells run the same primal-dual iteration (LPDS / Condat-Vu) and
# differ in how the coarse grids enter. The CDL cell takes the same
# multilevel dictionaries in SYNTHESIS form -- the codes generate the image
# instead of penalising it:
#
#                                                          GFLOP   params
#   lpdsnet    one level, no coarse grids                     21    1.00M
#   mglpds     coarse grids as a SOLVER (V-cycle, FAS)        34    3.62M
#   mllpds     coarse grids as a PRIOR (one dual/level)     ~329  109.0M
#   mllpdsw2   the same, channels doubling per level        ~260  135.8M
#   mlcdlw2    multilevel SYNTHESIS, ML-ISTA (K=18)          221   81.5M
#
# NOT COMPUTE-MATCHED, ON PURPOSE. The ML cells are sized for wall-clock time,
# not FLOPs. At K=18 ML-LPDS trained at ~5 it/s with the GPU near 100%, against
# ~2 it/s for the grid's fastest other nets, so K went to 30: expect ~3 it/s.
# The claim to test is that a much larger multilevel prior can still match or
# beat mglpds's INFERENCE TIME -- which means timing both on the same node, not
# reading this table.
#
# K=30 IS lpdsnet's K, so (lpdsnet vs mllpds) takes the same 29 data-consistency
# steps and differs only in the prior; (mllpds vs mllpdsw2) is the widening
# question at roughly equal cost. VarNet is the "is any of this competitive"
# check, not part of the question.
#
# THE SYNTHESIS NETS. mlcdlw2 stays at K=18: it was unstable there, and is kept
# at the size that was observed rather than grown with ML-LPDS -- so it matches
# mllpdsw2 in width but not in K. MLSplitCDLNet (mlsplitw2) is out of this
# experiment for now: it tripped the loss backtrack almost immediately. Its
# configs are still generated; run it by naming it,
#     ONLY=mlsplitw2 sbatch --array=0 torch/exp4_brain_mllpds.sbatch
# See the OPT_IN block of scripts/make_mg_recon_configs.py for the sizes and a
# suspect for the instability.
#
# OPT-IN CELLS. The ML cells do not appear in the default --list-cells output,
# so the full grid's indices (which mg_recon_brain.sbatch uses) are unchanged.
# They are listed only when named:
#   python scripts/make_mg_recon_configs.py --list-cells --anatomy brain \
#       --accels 16 --only lpdsnet mglpds varnet mllpds mllpdsw2 mlcdlw2
#
# SHORT RUNS AND OTHER ACCELERATIONS
# ----------------------------------
# ONLY, ACCELS and SWEEP_EPOCHS below take their value from the sbatch line when
# one is set. An epoch is 50 steps, so all six arms for 50k steps is
#
#     SWEEP_EPOCHS=1000 sbatch torch/exp4_brain_mllpds.sbatch
#
# Same budget and the same 1000-epoch cosine schedule for every arm, so it is a
# like-for-like look. Each lands in its own tagged run dir and wandb name
# (<model>_R16_1000ep), so the full-length cells are untouched. The same
# experiment at R=8 is `ACCELS=8 sbatch torch/exp4_brain_mllpds.sbatch`.
#
# BEFORE SUBMITTING
#   * a run dir left by an earlier size of the ML cells at this R -- e.g. a
#     K=18 mllpds_R16_1000ep look -- makes LAUNCH-ONCE refuse a relaunch of the
#     same tag, naming K. Move it aside, or resubmit with FORCE_RESTART=1.
#   * move the pre-ESPIRiT brain runs aside -- see exp1's BEFORE SUBMITTING;
#     LAUNCH-ONCE cannot tell maps rewritten in place from current ones.
#   * the ML-LPDS cells have never run at K=30. Probe the wider one first and
#     watch memory, which grows with K:
#       SWEEP_EPOCHS=20 sbatch --array=3 torch/exp4_brain_mllpds.sbatch
#     (tagged, so it lands in mllpds_R16_20ep; a repeat is skipped -- FORCE_RESTART=1)
#   * `python -m tests.test_ml_lpds` and `python -m tests.test_ml_cdlnet` on the
#     node are cheap and cover the models.
#
# SUBMIT FROM THE REPO ROOT. `#SBATCH --output=logs/...` resolves against the
# SUBMIT directory; from torch/ the job dies before the script runs.

ANATOMY=brain
ONLY="${ONLY:-lpdsnet mglpds varnet mllpds mllpdsw2 mlcdlw2}"
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
    echo "[exp4] cannot locate the ImMAP checkout (tried IMMAP_ROOT," \
         "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset}, ~/scratch/ee2178/ImMAP)." >&2
    echo "[exp4] submit from the repo root, or: IMMAP_ROOT=/path/to/ImMAP sbatch $0" >&2
    exit 1
fi

source "${IMMAP_ROOT}/torch/_mg_recon_body.sh"
