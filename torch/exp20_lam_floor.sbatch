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
#SBATCH --job-name=Exp20-LamFloor
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%A_%a.out
#SBATCH --error=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%A_%a.err
#SBATCH --array=0-2
#
# EXPERIMENT 20 -- THRESHOLDS MATCHED ACROSS LEVELS, AND A FLOOR ON ALPHA. KNEE R=8, FULL
# LENGTH (2026-11-02). Each cell is mg81v6vtxAnestUcf32 (exp18's MGLPDS-Small, already
# trained: the baseline, not retrained) with pure LPDS layers and FAS
# (make_mg_recon_configs.py::EXP20):
#
#   0   mg81v6vtxAnestUlamcf32        lam_level_scale=0.5   coarse thresholds start at lam0 / 2^l
#   1   mg81v6vtxAnestUfloorcf32      alpha in [0.5, none]  the correction cannot be turned off
#   2   mg81v6vtxAnestUlamfloorcf32   both
#
# Read against the baseline and exp19's cells: the flagged patches at a similar training step,
# params/alpha_x, val/levels, and val PSNR at the same epoch (the LR schedule is shared).
#
# A cell still running elsewhere, or finished, is skipped by LAUNCH-ONCE and an unfinished
# one resumes, so after a 48 h slot ends, RESUBMIT THE SAME COMMAND.
#
# SUBMIT FROM THE REPO ROOT.
#   sbatch torch/exp20_lam_floor.sbatch
#
# Check the index map:
#   python scripts/make_mg_recon_configs.py --list-cells --anatomy knee --accels 8 \
#       --only mg81v6vtxAnestUlamcf32 mg81v6vtxAnestUfloorcf32 mg81v6vtxAnestUlamfloorcf32
#
# The flagged patches, with the new nets beside the core nets:
#   NEW_NETS="mg81v6vtxAnestUlamcf32 mg81v6vtxAnestUfloorcf32 mg81v6vtxAnestUlamfloorcf32" \
#       OUT=results/alias_patches_exp20/knee sbatch torch/alias_patches.sbatch

ANATOMY=knee
ONLY="${ONLY:-mg81v6vtxAnestUlamcf32 mg81v6vtxAnestUfloorcf32 mg81v6vtxAnestUlamfloorcf32}"
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
    echo "[exp20] cannot locate the ImMAP checkout (tried IMMAP_ROOT," \
         "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset}, ~/scratch/ee2178/ImMAP)." >&2
    echo "[exp20] submit from the repo root, or: IMMAP_ROOT=/path/to/ImMAP sbatch $0" >&2
    exit 1
fi

source "${IMMAP_ROOT}/torch/_mg_recon_body.sh"
