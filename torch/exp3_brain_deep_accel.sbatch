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
#SBATCH --job-name=Exp3-BrainR16
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%A_%a.out
#SBATCH --error=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%A_%a.err
#SBATCH --array=0-2
#
# EXPERIMENT 3 -- does the V-cycle pay off at DEEP acceleration?  brain, R = 16.
#
#   0   brain  R16  lpdsnet     flat LPDS, K=30
#   1   brain  R16  mglpds      V-cycle, K=[6,[4,4,6]]
#   2   brain  R16  varnetmaps  E2E-VarNet baseline, on the operator's maps
#                               (online ESPIRiT under the measured protocol)
#
# THE HYPOTHESIS
# --------------
# R=16 leaves ~6% of k-space. The aliasing is no longer local, so a prior whose
# reach is one convolution stack has less to work with, while the V-cycle's
# coarse levels see the whole field of view at a quarter and a sixteenth of the
# resolution. If the multigrid arm ever earns its extra cost, this is where.
#
# The falsifier is cheap and worth stating: if mglpds beats lpdsnet by the SAME
# margin here as at R=8 (exp1), the V-cycle is buying capacity, not reach, and
# the deep-acceleration framing is wrong. Read exp1 and exp3 together -- neither
# is interpretable alone.
#
# VarNet is here as the "is any of this competitive" check, not as part of the
# hypothesis.
# The VarNet baseline is `varnetmaps` (since 2026-09-29): the E2E-VarNet
# cascades on the SAME online ESPIRiT maps the unrolled nets get, so the arms
# differ only in the prior. The published `varnet` (own SensitivityModel, RSS
# out) turned out to owe its lead to map estimation alone; it stays runnable
# by name (ONLY=varnet) but is no longer the reference.
# Like every cell it is launched once: a run dir that already holds an
# identical launch is skipped, a stale one is refused (LAUNCH-ONCE in
# torch/_mg_recon_body.sh; FORCE_RESTART=1 overrides).
#
# SHARED WITH exp4. All three cells here are also exp4's cells 0-2 (brain R16,
# the multilevel-net experiment), so submitting both trains each exactly once.
#
# WHAT IS SHARED WITH exp1 / exp2
# -------------------------------
# Everything except R: same data, same mask family, sigma ~ U[0.04, 0.06], same
# loss, LR, schedule and step budget, all from the same generator block. The
# organ mask is ON (brain now has ESPIRiT maps, so it selects the head rather
# than the whole field -- see torch/espirit_smaps.sbatch).
#
# BEFORE SUBMITTING
#   * brain must already be regenerated onto the ESPIRiT maps
#     (torch/espirit_smaps.sbatch), or this trains on the Walsh ground truth.
#   * probe one cell:  SWEEP_EPOCHS=20 sbatch --array=1 torch/exp3_brain_deep_accel.sbatch
#
# SUBMIT FROM THE REPO ROOT -- `#SBATCH --output=logs/...` resolves against the
# SUBMIT directory, so from torch/ the job dies before the script runs.
#
# Index map:
#   python scripts/make_mg_recon_configs.py --list-cells --anatomy brain \
#       --accels 16 --only lpdsnet mglpds varnetmaps

ANATOMY=brain
ONLY="lpdsnet mglpds varnetmaps"
ACCELS="16"
ATTN="flex"
# ORGAN MASK ON (2026-09-25), in EVERY launcher: loss and metrics run over the
# support of the STORED ESPIRiT maps (organ_mask_source="smaps"). Regeneration
# rewrites every config of an anatomy, so these must all agree -- turn it off
# everywhere or nowhere.
ORGAN_MASK=1                      # matches every other brain launcher
CONFIG_ROOT="config"
SWEEP_EPOCHS="${SWEEP_EPOCHS:-}"     # empty = the config's num_epochs. Settable
                                     # on the sbatch line; a short run lands in
                                     # its own run dir, <model>_R16_<N>ep
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
    echo "[exp3] cannot locate the ImMAP checkout (tried IMMAP_ROOT," \
         "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset}, ~/scratch/ee2178/ImMAP)." >&2
    echo "[exp3] submit from the repo root, or: IMMAP_ROOT=/path/to/ImMAP sbatch $0" >&2
    exit 1
fi

source "${IMMAP_ROOT}/torch/_mg_recon_body.sh"
