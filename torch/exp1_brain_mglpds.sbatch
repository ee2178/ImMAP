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
#SBATCH --job-name=Exp1-BrainMG
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --array=0-1
#
# EXPERIMENT 1 -- does the V-cycle earn its parameters?  fastMRI BRAIN.
#
#   0   brain  R8  mglpds
#   1   brain  R4  mglpds
#
# The LPDSNet arm is NOT here: those runs already exist under
# trained_nets/mg_recon/brain/lpdsnet_R{4,8}. This submits only the missing
# half, and `scripts/eval_mg_recon.py` pairs them afterwards.
#
# WHAT MAKES THE PAIR A CLEAN ABLATION
# ------------------------------------
# `lpdsnet` and `mglpds` are the SAME class (MGLPDSNet) differing only in K:
# 30 flat sweeps against [6, [4, 4, 6]] -- 6 V-cycles of 4/4/6. Everything else
# -- M=169, P=7, s=2, the loss, the noise range, the LR and the schedule -- is
# held by `LPDS_COMMON` in the generator. So a difference is the V-cycle, not
# an architecture change.
#
# They are NOT parameter-matched: 1.00M flat against 3.62M multigrid, because
# the V-cycle also carries coarse levels, the dF copies and alpha. That is the
# comparison to read, not something to paper over -- the generated configs
# record `count_parameters` and the eval script reports it.
#
# Fine-grid sweeps: 30 flat vs 6 x 4 = 24 multigrid, so the V-cycle is also
# doing LESS work on the finest grid.
#
# BEFORE SUBMITTING
#   * confirm the brain data roots resolve; this grid has only ever run on knee
#   * one probe cell first:  SWEEP_EPOCHS=20 sbatch --array=0 slurm/exp1_brain_mglpds.sbatch
#
# SUBMIT FROM THE REPO ROOT. `#SBATCH --output=logs/...` resolves against the
# SUBMIT directory, not this file's location; from slurm/ the job dies before
# the script runs and leaves no log to say why.
#
# Check the index map any time:
#   python scripts/make_mg_recon_configs.py --list-cells --anatomy brain --only mglpds

ANATOMY=brain
# The LPDSNet arm was trained under sigma ~ U[0, 0.01] and is NOT a valid
# comparison for these cells any more -- the grid moved to U[0.04, 0.06]. Rerun
# it before reading this experiment:
#     sbatch --array=0,6 torch/mg_recon_brain.sbatch
# (cells 0 and 6 of the full brain list are lpdsnet R8 and R4.)
ONLY="mglpds"
ACCELS=""                            # empty = both R8 and R4
ORGAN_MASK=1                         # matches mg_recon_brain.sbatch and
                                     # baseline_varnet cells 2,3
CONFIG_ROOT="config"
SWEEP_EPOCHS=""                      # empty = the config's num_epochs
REGENERATE=1                         # 1 = regenerate configs before running

# sbatch runs a COPY of this file from the node's spool dir, so a
# $BASH_SOURCE-relative path looks for the body next to that copy -- which was
# never shipped. Resolve the checkout explicitly.
for _c in "${IMMAP_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" "$HOME/scratch/ee2178/ImMAP"; do
    if [ -n "${_c}" ] && [ -f "${_c}/slurm/_mg_recon_body.sh" ]; then
        IMMAP_ROOT="${_c}"
        break
    fi
done
if [ ! -f "${IMMAP_ROOT:-}/slurm/_mg_recon_body.sh" ]; then
    echo "[exp1] cannot locate the ImMAP checkout (tried IMMAP_ROOT," \
         "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset}, ~/scratch/ee2178/ImMAP)." >&2
    echo "[exp1] submit from the repo root, or: IMMAP_ROOT=/path/to/ImMAP sbatch $0" >&2
    exit 1
fi

source "${IMMAP_ROOT}/slurm/_mg_recon_body.sh"
