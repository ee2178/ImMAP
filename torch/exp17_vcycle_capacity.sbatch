#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=torch_pr_89_tandon_advanced
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint="h200|b200"
# ^ PACK=2 (two cells per GPU) wants the big ones. Feature names must match the
#   cluster's exactly -- `sinfo -o "%f"` lists them -- or sbatch rejects the job.
#SBATCH --cpus-per-task=16
#SBATCH --mem=96GB
#SBATCH --time=48:00:00
#SBATCH --job-name=Exp17-VCap
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%A_%a.out
#SBATCH --error=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%A_%a.err
#SBATCH --array=0-1
#
# EXPERIMENT 17 -- V-CYCLES AT MATCHED CAPACITY, KNEE R=8, FULL LENGTH
# (2026-10-29).
#
# exp14's depth ablation (mg81v2 / mg81v4, K=[V, [4, 4, 6]]) cuts layers with
# the V-cycles. Here every cell keeps mg81v6's 84 layers (24 / 24 / 36 per
# level) and only redistributes them, iters = [4, 4, 6] * 6 / V, so parameters
# and conv FLOPs are the same in every cell and V is the only variable
# (make_mg_recon_configs.py::EXP17):
#
#   0   knee  R8   mg81v6vtxAnest     V=6  K=[6, [4, 4, 6]]   exp15's cell 3 -- REUSED
#   1   knee  R8   mg81v1eqvtxAnest   V=1  K=[1, [24, 24, 36]]
#   2   knee  R8   mg81v2eqvtxAnest   V=2  K=[2, [12, 12, 18]]
#   3   knee  R8   mg81v4eqvtxAnest   V=4  K=[4, [6, 6, 9]]
#
# All: vertex-centred transfer, FAS option A (fas_dual=fixed_point), TIED
# copies (untied copies are per-V-cycle parameters, so the cells would not
# match), the parent-pi fix, rediscretized coarse Gram, alphas FREE. Watch
# params/alpha_z per V-cycle: the question is whether fewer V-cycles make the
# net keep the coarse correction (L1 -> L0 alpha away from 0).
#
# PACK=2: two cells per GPU under CUDA MPS (torch/_mg_recon_body.sh). Task 0 is
# cells 0 + 1, task 1 is cells 2 + 3. Cell 0 is exp15's run: once it is
# finished (or while exp15 still has it), LAUNCH-ONCE skips it and V=1 trains
# ALONE on task 0 -- the situation that got exp16's lone cell cancelled for low
# H200 utilization. V=1 is the longest sequential chain of
# the four, so it may keep the GPU busier; if it is cancelled anyway, run it
# unpacked on an L40S:
#     PACK=1 ONLY=mg81v1eqvtxAnest sbatch --array=0 --constraint=l40s --mem=64GB \
#         torch/exp17_vcycle_capacity.sbatch
#
# A cell still running elsewhere, or finished, is skipped; an unfinished one
# resumes. After a 48 h slot ends, RESUBMIT THE SAME COMMAND.
#
# SUBMIT FROM THE REPO ROOT.
#   sbatch torch/exp17_vcycle_capacity.sbatch
#
# Check the index map:
#   python scripts/make_mg_recon_configs.py --list-cells --anatomy knee --accels 8 \
#       --only mg81v6vtxAnest mg81v1eqvtxAnest mg81v2eqvtxAnest mg81v4eqvtxAnest
#
# Evaluate against each other and exp15's vtxAnest with main_eval:
#   RUNS="knee/mg81v6vtxAnest_R8 knee/mg81v1eqvtxAnest_R8 knee/mg81v2eqvtxAnest_R8 knee/mg81v4eqvtxAnest_R8" \
#   OUT=results/exp17_knee_R8 VAL_KSPACE=measurement_awgn sbatch --array=0-3 torch/main_eval.sbatch

ANATOMY=knee
ONLY="${ONLY:-mg81v6vtxAnest mg81v1eqvtxAnest mg81v2eqvtxAnest mg81v4eqvtxAnest}"
ACCELS="${ACCELS:-8}"
PLANAR_STATE="${PLANAR_STATE-1}"     # `-`, not `:-`: an explicitly EMPTY value turns it off
ATTN="flex"                          # matches the other knee launchers --
ORGAN_MASK=1                         # regeneration rewrites every knee config,
                                     # so a mismatch silently changes the anatomy
CONFIG_ROOT="config"
SWEEP_EPOCHS="${SWEEP_EPOCHS:-}"     # empty = the config's num_epochs (full length)
REGENERATE=1                         # 1 = regenerate configs before running
PACK="${PACK:-2}"                    # cells per GPU; see --array above

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
    echo "[exp17] cannot locate the ImMAP checkout (tried IMMAP_ROOT," \
         "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset}, ~/scratch/ee2178/ImMAP)." >&2
    echo "[exp17] submit from the repo root, or: IMMAP_ROOT=/path/to/ImMAP sbatch $0" >&2
    exit 1
fi

source "${IMMAP_ROOT}/torch/_mg_recon_body.sh"
