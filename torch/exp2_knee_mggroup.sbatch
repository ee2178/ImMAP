#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=torch_pr_89_tandon_advanced
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=h200
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=48:00:00
#SBATCH --job-name=Exp2-KneeGrp-H200
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --array=0
#
# EXPERIMENT 2, GROUP CELL -- knee R8 mggrouplpds, on an H200.
#
# THE OTHER CELLS ARE IN torch/exp2_knee_group.sbatch (lpdsnet, mglpds and the
# varnet baseline). This is one experiment split across two files, and both
# must be submitted -- the claim is a comparison BETWEEN the group cell here and
# the local cell there, so either half alone says nothing:
#
#     sbatch torch/exp2_knee_group.sbatch        # lpdsnet, mglpds, varnet  (L40S)
#     sbatch torch/exp2_knee_mggroup.sbatch      # mggrouplpds              (H200)
#
# WHY A SEPARATE FILE AND NOT AN ARRAY INDEX
# ------------------------------------------
# `--constraint` is a static #SBATCH directive: SLURM cannot vary the node type
# per array task. mggrouplpds does not fit an L40S (48 GB) -- it runs windowed
# attention at THREE grid levels, and the 15x15 window on the 160x160 level-0
# latent is where the memory goes -- so it needs the H200's 141 GB. A separate
# file is the only mechanism that gives one cell a different node type.
#
# EVERYTHING ELSE IS HELD IDENTICAL to the other two cells, because it comes
# from the same generator block: same data, mask, sigma ~ U[0.01, 0.02], loss,
# schedule, step budget, organ mask. The ONLY difference is the node it lands
# on. Nothing about the node reaches the result -- no mixed precision, no
# batch-size change, same seeds -- so this stays a controlled comparison, but
# say the arm ran on different hardware if timing is ever reported.
#
# THE GROUP CELL'S OWN CAVEATS (unchanged from exp2)
# -------------------------------------------------
# window=15, not the 35 the BSD432 group configs use: the point of doing this
# inside a V-cycle is that the COARSE levels supply the long range, so each
# level can stay local. Attention is O(window^2) per pixel per level, so 35
# would cost ~5.4x for reach the hierarchy already provides. If this arm loses,
# window is the first knob to question -- not the last.
#
# sim_fun="distance", NOT "pidistance": the phase-invariant similarities need
# |<q,k>|, which FlexAttention cannot form on complex features (its score is one
# bilinear form, Re<q,k>, and the modulus needs two). So the adjacency is built
# on a phase-SENSITIVE similarity, which is a live explanation if this arm
# underperforms -- alongside "self-similarity does not help here".
#
# It also carries more parameters than the local cell (the attention
# projections), so "it had more capacity" is the competing explanation for a
# win. The honest controls are a brain arm and an mglpds with M raised to match;
# neither is in this submit.
#
# LR IS 2e-4 HERE, NOT 5e-4. Group cells kept tripping the averaged-loss
# backtrack at the shared rate; `_lr_for` in the generator keys that on the cell
# carrying attention. Recorded in the config, not set here.
#
# BEFORE SUBMITTING
#   * this cell has never run. Probe it first and watch memory:
#       SWEEP_EPOCHS=20 sbatch torch/exp2_knee_mggroup.sbatch
#   * flex compiles on first use (train.py calls compile_flex when
#     attn_backend="flex"); expect a slow first step, not a hang.
#   * if h200 is not the constraint name on this cluster, it is a one-word
#     change on the #SBATCH line above -- check `sinfo -o "%20P %25N %35G"`.
#
# SUBMIT FROM THE REPO ROOT -- `#SBATCH --output=logs/...` resolves against the
# SUBMIT directory, so from torch/ the job dies before the script runs.
#
# Index map:
#   python scripts/make_mg_recon_configs.py --list-cells --anatomy knee \
#       --accels 8 --only mggrouplpds

ANATOMY=knee
ONLY="mggrouplpds"
ACCELS="8"
ATTN="flex"                          # stated, not inherited: the run dir should
                                     # record which backend produced it
ORGAN_MASK=1                         # MUST match exp2_knee_group.sbatch and
                                     # mg_recon_knee.sbatch -- regeneration
                                     # rewrites every knee config, so a mismatch
                                     # here silently un-masks the whole anatomy
CONFIG_ROOT="config"
SWEEP_EPOCHS="${SWEEP_EPOCHS:-}"     # empty = the config's num_epochs. Settable
                                     # on the sbatch line; a short run lands in
                                     # its own run dir, <model>_R8_<N>ep
REGENERATE=1                         # 1 = regenerate configs before running

# sbatch runs a COPY of this file from the node's spool dir, so a
# $BASH_SOURCE-relative path looks for the body next to that copy.
for _c in "${IMMAP_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" "$HOME/scratch/ee2178/ImMAP"; do
    if [ -n "${_c}" ] && [ -f "${_c}/torch/_mg_recon_body.sh" ]; then
        IMMAP_ROOT="${_c}"
        break
    fi
done
if [ ! -f "${IMMAP_ROOT:-}/torch/_mg_recon_body.sh" ]; then
    echo "[exp2-h200] cannot locate the ImMAP checkout (tried IMMAP_ROOT," \
         "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset}, ~/scratch/ee2178/ImMAP)." >&2
    echo "[exp2-h200] submit from the repo root, or: IMMAP_ROOT=/path sbatch $0" >&2
    exit 1
fi

source "${IMMAP_ROOT}/torch/_mg_recon_body.sh"
