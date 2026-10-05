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
#SBATCH --job-name=Exp9-MGWidth
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%A_%a.out
#SBATCH --error=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%A_%a.err
#SBATCH --array=0-2
#
# EXPERIMENT 9 -- dictionary width at FULL LENGTH.  BRAIN R12.
#
#   0   brain  R12  mg121v6   M=121   2,589,264 params   ~0.82x mg169v6's cost
#   1   brain  R12  mg81v6    M=81    1,733,424 params   ~0.68x
#   2   brain  R12  mg100v6   M=100   2,139,948 params   ~0.75x
#
# The order is the GENERATOR's (MODELS order: exp8 cells, then EXP9's mg100v6),
# not the ONLY list's -- confirm with the --list-cells line at the bottom before
# indexing a single cell with --array=N.
#
# Follows exp8 (torch/exp8_brain_mg_arch.sbatch, 500 epochs): no V-cycle could
# be cut, M=81 dropped slightly, M=121 matched or slightly beat M=169. So the
# V-cycle structure is FIXED at exp8's K=[6,[4,4,6]] and only M varies; M=100
# (the EXP9 block in scripts/make_mg_recon_configs.py) bisects 81..121.
#
# Every cell is exp8's: the REDISCRETIZED coarse Gram (coarse_op="rediscretize")
# and the PLANAR complex conv (training.complex_conv="planar"), online ESPIRiT
# operator maps, SENSE ground truth, Julia masks at R=12, organ mask on. The
# cells differ ONLY in M.
#
# FULL LENGTH: the config's num_epochs = 6000 x 50 steps = 300k steps, 12x an
# exp8 cell. That will likely NOT fit in one 48 h slot. When a slot ends, just
# RESUBMIT THIS FILE: LAUNCH-ONCE in torch/_mg_recon_body.sh resumes an
# unfinished cell from its checkpoint, and skips finished or still-queued ones.
#
# Untagged run dirs (trained_nets/mg_recon/brain/<cell>_R12), so these do not
# collide with exp8's <cell>_R12_500ep runs.
#
# THE M=169 REFERENCE IS NOT IN THIS ARRAY. No full-length M=169 run exists
# under this protocol: the full-length `mglpds` runs use the GALERKIN coarse
# Gram, which is a different coarse problem (planar vs gauss is fp roundoff,
# rediscretize vs galerkin is not). To get one, add it by name:
#     ONLY="mg169v6" sbatch --array=0 torch/exp9_brain_mg_width.sbatch
#
# Probe one cell first if unsure of throughput (tagged, lands in <cell>_R12_20ep):
#     SWEEP_EPOCHS=20 sbatch --array=0 torch/exp9_brain_mg_width.sbatch
#
# SUBMIT FROM THE REPO ROOT.
#   sbatch torch/exp9_brain_mg_width.sbatch
#
# Check the index map any time:
#   python scripts/make_mg_recon_configs.py --list-cells --anatomy brain \
#       --accels 12 --only mg121v6 mg100v6 mg81v6

ANATOMY=brain
ONLY="${ONLY:-mg121v6 mg100v6 mg81v6}"
ACCELS="12"                          # PINNED: exp8 measured these at R12
ORGAN_MASK=1                         # matches every other brain launcher --
                                     # regeneration rewrites every brain config,
                                     # so a mismatch silently un-masks the anatomy
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
    echo "[exp9] cannot locate the ImMAP checkout (tried IMMAP_ROOT," \
         "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset}, ~/scratch/ee2178/ImMAP)." >&2
    echo "[exp9] submit from the repo root, or: IMMAP_ROOT=/path/to/ImMAP sbatch $0" >&2
    exit 1
fi

source "${IMMAP_ROOT}/torch/_mg_recon_body.sh"
