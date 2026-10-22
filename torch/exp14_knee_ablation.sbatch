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
#SBATCH --job-name=Exp14-KneeAbl
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%A_%a.out
#SBATCH --error=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%A_%a.err
#SBATCH --array=0-4
#
# EXPERIMENT 14 -- MGLPDS ABLATIONS on KNEE R=8, FULL LENGTH.
#
#   0   knee  R8   mg121v6   width  M=121, K=[6,[4,4,6]]
#   1   knee  R8   mg81v2    depth  M=81,  K=[2,[4,4,6]]
#   2   knee  R8   mg81v4    depth  M=81,  K=[4,[4,4,6]]
#   3   knee  R8   mg100v6   width  M=100, K=[6,[4,4,6]]
#   4   knee  R8   mg144v6   width  M=144, K=[6,[4,4,6]]
#
# (Index order is the generator's MODELS order, not the order of ONLY.)
#
# TWO SWEEPS, EACH ANCHORED ON exp11's RUNS, which are NOT retrained here:
#   width  M = 81 | 100 | 121 | 144 | 169   at V=6   (81, 169 = exp11's mg81v6 / mg169v6)
#   depth  V = 2  | 4   | 6                 at M=81  (6 = exp11's mg81v6)
# The brain R=12 exp8 grid found V=2 and V=4 lose to V=6 at every M; this
# repeats the depth question on knee at full length, and fills in the width
# curve between Small and Big.
#
# EVERYTHING BUT M OR V IS exp11's knee R8 SETUP, checked config-for-config
# against mg81v6_R8 on 2026-10-21 (only model.params.M / .K, the name and the
# run dir differ): measured k-space + AWGN at sigma ~ U[0.01, 0.02], Julia
# masks (center_frac 0.04, adjust_accel), online ESPIRiT operator maps, stored
# SENSE ground truth, organ mask from the stored maps, rediscretized coarse
# Gram, planar convs, PLANAR_STATE on, 6000 epochs x 50 steps.
#
# ONE DIFFERENCE YOU MAY SEE IN wandb, not in the paper numbers: these cells
# were generated 2026-10-21, when configs validated on SIMULATED k-space
# (training.val_kspace_type), while exp11's runs validate on measured k-space.
# Between slots, with no job on the run, `python scripts/set_val_kspace.py`
# switches them to measured validation (the val curve steps at the switch).
# Compare the sweeps with torch/main_eval.sbatch, which scores every run on the
# SAME measurement (measured k-space + AWGN):
#     RUNS="knee/mg81v6_R8 knee/mg100v6_R8 knee/mg121v6_R8 knee/mg144v6_R8 knee/mg169v6_R8 knee/mg81v2_R8 knee/mg81v4_R8" \
#     OUT=results/ablation_knee_R8 VAL_KSPACE=measurement_awgn sbatch --array=0-6 torch/main_eval.sbatch
#
# EXISTING RUN DIRS. LAUNCH-ONCE (torch/_mg_recon_body.sh) resumes an
# unfinished trained_nets/mg_recon/knee/<cell>_R8 and skips a finished one --
# so check first that no earlier (short / probe) knee run sits under these
# exact names:
#     ls -d trained_nets/mg_recon/knee/{mg121v6,mg81v2,mg81v4,mg100v6,mg144v6}_R8
# Move any you find aside, or submit with FORCE_RESTART=1 (OVERWRITES them).
#
# FULL LENGTH: 300k steps; at the knee rates of exp11 a cell may need a second
# 48 h slot. When a slot ends, RESUBMIT THIS FILE: unfinished cells resume from
# their checkpoint, finished or still-queued ones are skipped.
#
# SUBMIT FROM THE REPO ROOT.
#   sbatch torch/exp14_knee_ablation.sbatch
#   ONLY="mg144v6" sbatch --array=0 torch/exp14_knee_ablation.sbatch   # one cell
#
# Check the index map any time:
#   python scripts/make_mg_recon_configs.py --list-cells --anatomy knee \
#       --accels 8 --only mg100v6 mg121v6 mg144v6 mg81v2 mg81v4

ANATOMY=knee
ONLY="${ONLY:-mg100v6 mg121v6 mg144v6 mg81v2 mg81v4}"
ACCELS="${ACCELS:-8}"
PLANAR_STATE="${PLANAR_STATE-1}"     # as exp11; `-`, so an explicitly EMPTY value turns it off
ATTN="flex"                          # matches the other knee launchers --
ORGAN_MASK=1                         # regeneration rewrites every knee config,
                                     # so a mismatch silently changes the anatomy
CONFIG_ROOT="config"
SWEEP_EPOCHS="${SWEEP_EPOCHS:-}"     # empty = the config's num_epochs (full length)
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
    echo "[exp14] cannot locate the ImMAP checkout (tried IMMAP_ROOT," \
         "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset}, ~/scratch/ee2178/ImMAP)." >&2
    echo "[exp14] submit from the repo root, or: IMMAP_ROOT=/path/to/ImMAP sbatch $0" >&2
    exit 1
fi

source "${IMMAP_ROOT}/torch/_mg_recon_body.sh"
