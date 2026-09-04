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
#SBATCH --job-name=MGRecon-knee
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --array=0-11          # MUST equal the knee cell count; checked at runtime
#
# fastMRI KNEE (PD), synthetic k-space, sigma ~ U[0.04, 0.06].
#
#   4 models x 2 accelerations (R=8, R=4) = 8 cells
#
#     python scripts/make_mg_recon_configs.py --list-cells --anatomy knee
#     0  knee  R8  lpdsnet
#     1  knee  R8  mglpds
#     ...
#
# The models, as two paired comparisons:
#
#   lpdsnet     models/lpdsnet.py::LPDSNet, K=30        }  different
#   mglpds      MGCDLNet(dual=True), K=[1,[8,8,8]]      }  architectures
#
#   altsplit    AltSplitCDLNet, denoiser K=6            }  SAME class, the
#   mgaltsplit  AltSplitCDLNet, denoiser K=[1,[4,4,8]]  }  V-cycle is the only
#                                                       }  variable
#
# Only the LADMM pair isolates multigrid cleanly. lpdsnet vs mglpds crosses an
# architecture boundary as well -- LPDSNet is its own class, and ImMAP has no
# multigrid version of it -- so a difference there cannot be attributed to the
# V-cycle alone. Both cells now use the E^H E DC correction (preproc='kspace'),
# so at least that is not a third confound.
#
# Brain is a separate launcher: torch/mg_recon_brain.sbatch
#
# Submit:      sbatch torch/mg_recon_knee.sbatch
# One cell:    sbatch --array=0 torch/mg_recon_knee.sbatch
# Probe first: SWEEP_EPOCHS=20 below, then look at the val curves.

ANATOMY=knee
# MUST match exp2_knee_group.sbatch and baseline_varnet.sbatch cells 0,1.
# REGENERATE rewrites EVERY config for the anatomy, not just the cells this
# submit runs, so leaving this unset here would silently flip all 12 knee
# configs to use_organ_mask=false the next time this launcher runs.
ORGAN_MASK=1
CONFIG_ROOT="config"
SWEEP_EPOCHS=""                      # empty = the config's num_epochs
REGENERATE=1                         # 1 = regenerate configs before running

# sbatch does NOT execute this file in place: it copies the script's contents to
# the compute node's spool dir and runs that copy, so ${BASH_SOURCE[0]} is
# /opt/torch/.../job<id>/slurm_script and a $BASH_SOURCE-relative source looks
# for the body next to the copy -- which was never shipped. Resolve the repo
# explicitly. Override with IMMAP_ROOT=/path/to/ImMAP sbatch ...
for _c in "${IMMAP_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" "$HOME/scratch/ee2178/ImMAP"; do
    if [ -n "${_c}" ] && [ -f "${_c}/torch/_mg_recon_body.sh" ]; then
        IMMAP_ROOT="${_c}"
        break
    fi
done
if [ ! -f "${IMMAP_ROOT:-}/torch/_mg_recon_body.sh" ]; then
    echo "[grid] cannot locate torch/_mg_recon_body.sh (tried IMMAP_ROOT," \
         "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset}, ~/scratch/ee2178/ImMAP)." >&2
    echo "[grid] submit from the repo root, or: IMMAP_ROOT=/path/to/ImMAP sbatch $0" >&2
    exit 1
fi
export IMMAP_ROOT
source "${IMMAP_ROOT}/torch/_mg_recon_body.sh"
