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
#SBATCH --job-name=Exp2-KneeGrp
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --array=0-2
#
# EXPERIMENT 2 -- is nonlocal self-similarity a good prior on KNEE?  R = 8.
#
#   0   knee  R8  lpdsnet       flat LPDS, K=30
#   1   knee  R8  mglpds        V-cycle, K=[6,[4,4,6]], local prox
#   2   knee  R8  mggrouplpds   V-cycle, same K, NONLOCAL prox at every level
#
# THE HYPOTHESIS, AND WHAT WOULD FALSIFY IT
# -----------------------------------------
# The claim is anatomy-specific: knee patches relate roughly ortholinearly, so
# nonlocal self-similarity should pay; brain patches relate more by rotation,
# so it should not. That predicts a DIFFERENCE BETWEEN ANATOMIES, which is a
# much stronger claim than "the group prox helps" -- and a cheaper one to
# believe, because a prior that helps everywhere is probably just capacity.
#
# So cell 2 beating cell 1 here is necessary but NOT sufficient. The test is
# finished only when the same pair is run on brain and does NOT show it. That
# run is deliberately not in this file: get the knee numbers first, and only
# spend the brain compute if there is something to explain.
#
# Cell 2 carries more parameters than cell 1 (the attention projections), so if
# it wins, "it had more capacity" is the competing explanation. The honest
# controls are (a) the brain arm above, and (b) an mglpds with M raised to
# match -- neither is in this submit.
#
# WHY window=15 AND NOT 35
# ------------------------
# The BSD432 group configs use W=35. Doing this inside a V-cycle is supposed to
# make that unnecessary: the COARSE levels supply the long range, so each level
# can stay local. 15x15 on the 160x160 level-0 latent, with the 80x80 and 40x40
# levels reaching the rest. Attention is O(window^2) per pixel per level, so 35
# would cost ~5.4x for reach the hierarchy already provides. If cell 2 loses,
# window is the first knob to question -- not the last.
#
# sim_fun="distance", NOT "pidistance". The phase-invariant similarities need
# |<q,k>|, which FlexAttention cannot form on complex features: its score is
# one bilinear form (Re<q,k>) and the modulus needs two. models/prox.py raises
# rather than attending on the wrong similarity. attn_backend="triton" would
# carry pidistance, at the cost of a hand-written kernel. See
# tests/test_group_lpds.py::test_flex_backend_matrix.
#
# Consequence worth holding onto: the adjacency is built on phase-SENSITIVE
# similarity. If the group arm underperforms, that is a live explanation
# alongside "self-similarity does not help here".
#
# ORGAN MASK -- ON FOR THIS EXPERIMENT
# -----------------------------------
# ORGAN_MASK=1 restricts the loss, PSNR/NRMSE/SSIM, and the logged panel to the
# coil-sensitivity support. Knee is why the switch exists: air is most of the
# slice, and an unmasked PSNR is dominated by how well each model reproduces a
# noise floor nobody is trying to reconstruct.
#
# What this changes about reading the results:
#
#   * These numbers are NOT comparable to the brain runs (exp1) or to any
#     earlier knee run, which were scored over the whole image. Different
#     region, different quantity. Do not put them in one table.
#   * The hypothesis is unaffected. It is a claim about cell 2 vs cell 1 WITHIN
#     this submit, and all three cells are masked identically.
#   * Masking is likely to FLATTER the nonlocal arm less than the local one, if
#     anything: self-similarity has more to exploit in a large uniform
#     background than in the organ. Worth remembering if cell 2's margin here
#     is smaller than an earlier unmasked probe suggested -- that is expected,
#     not a regression.
#   * SSIM additionally erodes the mask by the 5-pixel window radius, so it
#     scores the organ interior. See training/metrics.py.
#
# COUPLED TO slurm/baseline_varnet.sbatch. Regeneration rewrites EVERY config
# for the anatomy, not just this submit's cells, so both launchers that touch
# knee must set this the same way or whichever regenerates last silently wins.
# baseline_varnet sets ORGAN_MASK=1 on its knee cells (0 and 1) to match. If
# you turn this off, turn those off too.
#
# BEFORE SUBMITTING
#   * cell 2 is the only one that has never run. Probe it alone first:
#       SWEEP_EPOCHS=20 sbatch --array=2 slurm/exp2_knee_group.sbatch
#     and watch memory -- attention at three grids is new here.
#   * flex compiles on first use (train.py calls compile_flex when
#     attn_backend="flex"); expect a slow first step, not a hang.
#
# SUBMIT FROM THE REPO ROOT -- `#SBATCH --output=logs/...` resolves against the
# SUBMIT directory, so from slurm/ the job dies before the script runs.
#
# Check the index map any time:
#   python scripts/make_mg_recon_configs.py --list-cells --anatomy knee \
#       --accels 8 --only lpdsnet mglpds mggrouplpds

ANATOMY=knee
ONLY="lpdsnet mglpds mggrouplpds"
ACCELS="8"
ATTN="flex"                          # stated, not inherited: the run dir should
                                     # record which backend produced it
ORGAN_MASK=1                         # see ORGAN MASK below
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
    echo "[exp2] cannot locate the ImMAP checkout (tried IMMAP_ROOT," \
         "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset}, ~/scratch/ee2178/ImMAP)." >&2
    echo "[exp2] submit from the repo root, or: IMMAP_ROOT=/path/to/ImMAP sbatch $0" >&2
    exit 1
fi

source "${IMMAP_ROOT}/slurm/_mg_recon_body.sh"
