#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=torch_pr_89_tandon_advanced
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=16
#SBATCH --mem=96GB
#SBATCH --time=05:00:00
#SBATCH --job-name=Exp8-MGArch
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%A_%a.out
#SBATCH --error=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%A_%a.err
#SBATCH --array=0-9
#
# EXPERIMENT 8 -- multigrid architecture sweep.  BRAIN R12.
#
#   M x V-cycles grid, each V-cycle [4, 4, 6] per level:
#
#               V=2        V=4        V=6
#     M=169   0 mg169v2  1 mg169v4  2 mg169v6   (mg169v6 = mglpds's K)
#     M=121   3 mg121v2  4 mg121v4  5 mg121v6
#     M=81    6 mg81v2   7 mg81v4   8 mg81v6
#
#   plus one isolated arm:
#     9 mgunet   K=[1,[8,8,14]] -- ONE V-cycle, 30 layers like LPDSNet's K=30,
#                ~1:1:2 fine:mid:coarse ("U-Net-like": down once, up once).
#                ([1,[5,5,10]] cannot be built: non-coarsest levels split into
#                equal pre/post smoothing, so they need even counts.)
#
#   params:  M=169: 1.21 / 2.41 / 3.62M   M=121: 0.86 / 1.73 / 2.59M
#            M=81:  0.58 / 1.16 / 1.73M   mgunet: 1.14M   (lpdsnet: 1.00M)
#
# Every cell: the REDISCRETIZED coarse Gram (model.params.coarse_op =
# "rediscretize"; operators/coarse.py, tests/test_mg_coarse_op.py) and the
# PLANAR complex conv (training.complex_conv = "planar"; train.py applies it).
# Otherwise the grid's measured protocol: online ESPIRiT operator maps, SENSE
# ground truth, Julia masks at R=12 (effective ~11.9), organ mask on.
#
# 500 EPOCHS (x 50 steps = 25k steps), not the config's 6000, in a 5 h slot.
# The runs are TAGGED <model>_R12_500ep, so they never block a full-length run
# of the same cell. If a heavy cell (V=6 at M=169) does not finish in 5 h, it
# stops at its last checkpoint: resume it with torch/mg_recon_resume.sbatch, or
# raise --time below. Probe one cell's throughput first if unsure:
#     SWEEP_EPOCHS=10 RUN_TAG=probe sbatch --array=2 torch/exp8_brain_mg_arch.sbatch
#
# THE QUESTION: how far can the full-size MGLPDS be DOWNSIZED before it loses
# performance? (Full-size MGLPDS already beats lpdsnet and E2E-VarNet, so no
# baselines run here.) The reference is mg169v6 -- the full-parameter cell,
# under the same protocol and the same 500-epoch budget -- and every other cell
# is read as a drop relative to it.
#
# What to read:
#   * along a ROW (fixed M): how many V-cycles can go before PSNR drops?
#   * down a COLUMN (fixed V): how much dictionary width can go?
#   * matched parameter counts -- is width or depth the cheaper thing to cut?
#       ~2.5M: mg169v4 (2.41M) vs mg121v6 (2.59M)
#       1.73M: mg121v4 vs mg81v6
#       ~1.2M: mg169v2 (1.21M) vs mg81v4 (1.16M) vs mgunet (1.14M)
#   * mgunet vs mg169v2: one deep V-cycle vs two shallow ones at nearly the
#     same size.
#
# Submit (from the repo root):   sbatch torch/exp8_brain_mg_arch.sbatch
# One cell:                      sbatch --array=9 torch/exp8_brain_mg_arch.sbatch

ANATOMY=brain
ONLY="${ONLY:-mg169v2 mg169v4 mg169v6 mg121v2 mg121v4 mg121v6 mg81v2 mg81v4 mg81v6 mgunet}"
ACCELS="${ACCELS:-12}"
ORGAN_MASK=1                         # matches every other brain launcher --
                                     # regeneration rewrites every brain config,
                                     # so a mismatch silently un-masks the anatomy
CONFIG_ROOT="config"
SWEEP_EPOCHS="${SWEEP_EPOCHS:-500}"  # x 50 steps/epoch (brain) = 25k steps
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
    echo "[exp8] cannot locate the ImMAP checkout (tried IMMAP_ROOT," \
         "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset}, ~/scratch/ee2178/ImMAP)." >&2
    echo "[exp8] submit from the repo root, or: IMMAP_ROOT=/path/to/ImMAP sbatch $0" >&2
    exit 1
fi

source "${IMMAP_ROOT}/torch/_mg_recon_body.sh"
