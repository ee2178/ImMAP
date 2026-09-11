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
#SBATCH --array=0-1
#
# EXPERIMENT 4 -- multilevel LPDS against the V-cycle, fastMRI BRAIN R8.
#
#   0   brain  R8  mllpds     ML-LPDS, L=3, widen=1, channels 17/17/17
#   1   brain  R8  mllpdsw2   ML-LPDS, L=3, widen=2, channels  8/16/32
#
# The comparison arms are NOT here: trained_nets/mg_recon/brain/lpdsnet_R8 and
# mglpds_R8 already exist on this exact setup (data, ESPIRiT maps, mask,
# sigma ~ U[0.01, 0.02], loss, LR, schedule, organ mask), and every one of
# those comes from the same generator block as these two cells.
#
# WHAT THE THREE-WAY COMPARISON ISOLATES
# --------------------------------------
# All four cells run the same primal-dual iteration (LPDS / Condat-Vu). They
# differ in how the coarse grids enter:
#
#   lpdsnet    one level, no coarse grids                        1.00M params
#   mglpds     coarse grids as a SOLVER (V-cycle, FAS)           3.62M
#   mllpds     coarse grids as a PRIOR (one dual per level)      3.50M
#   mllpdsw2   the same, channels doubling per level             3.81M
#
# The ML cells are parameter-matched to mglpds, so (mglpds vs mllpds) is
# solver-vs-prior at equal capacity, and (mllpds vs mllpdsw2) is the widening
# question at equal capacity. Parity costs them width at level 1 -- 17 and 8
# atoms against the baselines' 169 -- and FLOPs (~12 and ~8 GFLOP against
# mglpds's ~28), both of which are part of what a result here means. See the
# OPT_IN block in scripts/make_mg_recon_configs.py for the arithmetic.
#
# OPT-IN CELLS. Neither appears in the default --list-cells output, so the full
# grid's indices (which mg_recon_brain.sbatch uses) are unchanged. They are
# listed only when named:
#   python scripts/make_mg_recon_configs.py --list-cells --anatomy brain \
#       --accels 8 --only mllpds mllpdsw2
#
# BEFORE SUBMITTING
#   * these cells have never run. Probe one first:
#       SWEEP_EPOCHS=20 sbatch --array=0 torch/exp4_brain_mllpds.sbatch
#   * `python -m tests.test_ml_lpds` on the node is cheap and covers the model.
#
# SUBMIT FROM THE REPO ROOT. `#SBATCH --output=logs/...` resolves against the
# SUBMIT directory; from torch/ the job dies before the script runs.

ANATOMY=brain
ONLY="mllpds mllpdsw2"
ACCELS="8"                           # PINNED: only R8 has the comparison arms
ORGAN_MASK=1                         # matches mg_recon_brain.sbatch, exp1, exp3
                                     # and baseline_varnet cells 2,3 -- regeneration
                                     # rewrites every brain config, so a mismatch
                                     # silently un-masks the whole anatomy
CONFIG_ROOT="config"
SWEEP_EPOCHS=""                      # empty = the config's num_epochs
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
