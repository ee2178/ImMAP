#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=torch_pr_89_tandon_advanced
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
# ^ One cell per GPU, on L40S: a lone cell on an H200 is what got exp16's cancelled
#   for low utilization (exp17's header).
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=48:00:00
#SBATCH --job-name=Exp19-Coarse
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%A_%a.out
#SBATCH --error=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%A_%a.err
#SBATCH --array=0-2
#
# EXPERIMENT 19 -- IS THE COARSE LEVEL SOLVING THE COARSE PROBLEM? KNEE R=8, FULL LENGTH
# (2026-11-01). Each cell is mg81v6vtxAnestUcf32 (exp18's MGLPDS-Small, already trained:
# the baseline, not retrained) with one thing changed (make_mg_recon_configs.py::EXP19):
#
#   0   mg81v6vtxAnestUgalcf32    coarse_op=galerkin    the exact coarse Gram; fine FFTs in
#                                                       all 84 Grams, plus R / P, so slowest
#   1   mg81v6vtxAnestUfoldcf32   coarse_op=folded      coarse FFTs, the folded mask
#   2   mg81v6vtxAnestUaz0cf32    alpha_z_zero=true     no dual coarse correction
#
# Read against the baseline: val PSNR at the same epoch (the LR schedule is shared, so a
# partial run compares at its epoch, not with the baseline's end), the val/levels panel,
# params/alpha_x. The questions: do the coarse images stay images, and at what cost?
#
# A cell still running elsewhere, or finished, is skipped by LAUNCH-ONCE and an unfinished
# one resumes, so after a 48 h slot ends, RESUBMIT THE SAME COMMAND.
#
# SUBMIT FROM THE REPO ROOT.
#   sbatch torch/exp19_coarse_op.sbatch
#
# Check the index map:
#   python scripts/make_mg_recon_configs.py --list-cells --anatomy knee --accels 8 \
#       --only mg81v6vtxAnestUgalcf32 mg81v6vtxAnestUfoldcf32 mg81v6vtxAnestUaz0cf32
#
# Once trained, evaluate against the baseline, and run the diagnostic on all four:
#   RUNS="knee/mg81v6vtxAnestUcf32_R8 knee/mg81v6vtxAnestUgalcf32_R8 knee/mg81v6vtxAnestUfoldcf32_R8 knee/mg81v6vtxAnestUaz0cf32_R8" \
#       OUT=results/exp19_knee_R8 RUNS_PER_TASK=4 sbatch --array=0 torch/main_eval.sbatch
#   RUNS="knee/mg81v6vtxAnestUcf32_R8 knee/mg81v6vtxAnestUgalcf32_R8 knee/mg81v6vtxAnestUfoldcf32_R8 knee/mg81v6vtxAnestUaz0cf32_R8" \
#       sbatch --array=0-3 torch/coarse_diag.sbatch

ANATOMY=knee
ONLY="${ONLY:-mg81v6vtxAnestUgalcf32 mg81v6vtxAnestUfoldcf32 mg81v6vtxAnestUaz0cf32}"
ACCELS="${ACCELS:-8}"
PLANAR_STATE="${PLANAR_STATE-1}"     # `-`, not `:-`: an explicitly EMPTY value turns it off
ATTN="flex"                          # matches the other knee launchers --
ORGAN_MASK=1                         # regeneration rewrites every knee config,
                                     # so a mismatch silently changes the anatomy
CONFIG_ROOT="config"
SWEEP_EPOCHS="${SWEEP_EPOCHS:-}"     # empty = the config's num_epochs (full length)
REGENERATE=1                         # 1 = regenerate configs before running
PACK="${PACK:-1}"

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
    echo "[exp19] cannot locate the ImMAP checkout (tried IMMAP_ROOT," \
         "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset}, ~/scratch/ee2178/ImMAP)." >&2
    echo "[exp19] submit from the repo root, or: IMMAP_ROOT=/path/to/ImMAP sbatch $0" >&2
    exit 1
fi

source "${IMMAP_ROOT}/torch/_mg_recon_body.sh"
