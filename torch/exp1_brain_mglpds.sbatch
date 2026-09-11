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
#SBATCH --array=0-5
#
# EXPERIMENT 1 -- does the V-cycle earn its parameters?  fastMRI BRAIN.
#
#   0   brain  R8  lpdsnet      flat LPDS, K=30
#   1   brain  R8  mglpds       V-cycle, K=[6,[4,4,6]]
#   2   brain  R8  varnet       E2E-VarNet baseline
#   3   brain  R4  lpdsnet
#   4   brain  R4  mglpds
#   5   brain  R4  varnet
#
# Every arm trains here, from scratch: no existing brain run uses the
# regenerated ESPIRiT maps (torch/espirit_smaps.sbatch, which also rewrites the
# ground truth), so nothing trained before them is comparable.
#
# SHARED CELLS. Brain R8 lpdsnet, mglpds and varnet are also cells of exp4.
# The body launches each cell once -- in whichever launcher reaches it first --
# and skips it in the other (LAUNCH-ONCE in torch/_mg_recon_body.sh).
#
# VarNet is the "is any of this competitive" check, not part of the hypothesis
# (its asymmetries are in the `varnet` entry of scripts/make_mg_recon_configs.py).
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
#   * move the pre-ESPIRiT brain runs aside. If the maps were rewritten under
#     the same smap_root, an old run's config still matches the generator's and
#     LAUNCH-ONCE would SKIP it as if it were current:
#         mv trained_nets/mg_recon/brain trained_nets/mg_recon/brain_pre_espirit
#   * one probe cell first:  SWEEP_EPOCHS=20 sbatch --array=1 torch/exp1_brain_mglpds.sbatch
#     (tagged, so it lands in mglpds_R8_20ep; a repeat is skipped -- FORCE_RESTART=1)
#
# SUBMIT FROM THE REPO ROOT. `#SBATCH --output=logs/...` resolves against the
# SUBMIT directory, not this file's location; from torch/ the job dies before
# the script runs and leaves no log to say why.
#
# Check the index map any time:
#   python scripts/make_mg_recon_configs.py --list-cells --anatomy brain \
#       --accels 8 4 --only lpdsnet mglpds varnet

ANATOMY=brain
ONLY="lpdsnet mglpds varnet"
ACCELS="8 4"                         # PINNED. Was "" (= all), which
                                     # silently grew to include R16 when
                                     # exp3 added it to the global list.
                                     # R16 belongs to exp3.
ORGAN_MASK=1                         # matches every other brain launcher
CONFIG_ROOT="config"
SWEEP_EPOCHS="${SWEEP_EPOCHS:-}"     # empty = the config's num_epochs. Settable
                                     # on the sbatch line; a short run lands in
                                     # its own run dir, <model>_R<r>_<N>ep
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
    echo "[exp1] cannot locate the ImMAP checkout (tried IMMAP_ROOT," \
         "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset}, ~/scratch/ee2178/ImMAP)." >&2
    echo "[exp1] submit from the repo root, or: IMMAP_ROOT=/path/to/ImMAP sbatch $0" >&2
    exit 1
fi

source "${IMMAP_ROOT}/torch/_mg_recon_body.sh"
