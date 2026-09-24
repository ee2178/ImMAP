#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=torch_pr_89_tandon_advanced
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=16
#SBATCH --mem=96GB
#SBATCH --time=48:00:00
#SBATCH --job-name=Exp7-WaveletLPDS
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --array=0
#
# EXPERIMENT 7 -- DT-CWT LPDS baseline.  BRAIN R16.
#
#   0  wlpds16   LPDS with a 3-level DT-CWT analysis cascade, 16/64/256, K=30
#
#   reference: mllpds64 (exp4 cell 3)  and  lpdsnet (exp3 cell 0)
#
# WHAT IT IS (models/wavelet_lpds.py). `lpdsnet`'s single strided conv replaced
# by K = Q T3 T2 T1: stride-2 complex convs initialised as Kingsbury's 2D DT-CWT
# (LeGall 5/3 at level 1, qshift_06 at levels 2-3 with its 0.035 end taps
# truncated to fit P=7), hard-grouped by tree, every band carried to depth 3,
# and a fixed unitary Q forming the 12 oriented complex channels per band set.
# One dual, on the depth-3 code. Per-channel thresholds at lam0, except the 4
# coarse LL^3 channels, which start at 0 and stay learnable. B = A^H at init,
# ||K|| = 1, tau0 = 0.5 (inside the Condat-Vu bound).
#
# WHY. The first rung of the wavelet ladder: M=16 is ONE DT-CWT, redundancy 4
# (vs ~42 for lpdsnet at M=169), so it tests the prior itself. If it loses,
# rule out low redundancy before blaming the prior.
#
# THINGS TO READ OFF THE CHECKPOINT: do the LL^3 thresholds leave 0 (the trunk
# stopped being lowpass)? Does ||Im h|| / ||h|| grow? Do the one-hot carry
# kernels spread (movement toward wavelet packets)?
#
# EVERYTHING ELSE IS HELD FIXED by the shared generator block: data, ESPIRiT
# maps, mask, sigma ~ U[0.04, 0.06], loss, LR, schedule, organ mask, R=16.
#
# HOW TO RUN
# ----------
#   # 50k iterations (steps_per_epoch=50, so 1000 epochs) -- matches exp4-6,
#   # so mllpds64_R16_1000ep is the direct reference.
#   SWEEP_EPOCHS=1000 sbatch torch/exp7_brain_wavelet_lpds.sbatch
#
#   # a quick shakeout first
#   SWEEP_EPOCHS=20 sbatch torch/exp7_brain_wavelet_lpds.sbatch
#
# A shortened run is TAGGED (<model>_R16_1000ep) and never lands in the
# full-length cell's dir. Re-submitting the same tagged cell is skipped by
# LAUNCH-ONCE -- pass FORCE_RESTART=1, or delete the run dir, to redo one.
#
# SUBMIT FROM THE REPO ROOT. `logs/` is relative to the SUBMIT directory; from
# torch/ the job dies before the script runs.

ANATOMY=brain
ONLY="${ONLY:-wlpds16}"
ACCELS="${ACCELS:-16}"
# ORGAN MASK OFF (2026-09-22), in EVERY launcher: loss and metrics run over the
# whole image. Regeneration rewrites every config of an anatomy, so these must
# all agree -- turn it back on everywhere or nowhere.
ORGAN_MASK=""                       # matches every other brain launcher --
                                     # regeneration rewrites every brain config,
                                     # so a mismatch silently un-masks the anatomy
CONFIG_ROOT="config"
SWEEP_EPOCHS="${SWEEP_EPOCHS:-}"     # empty = the config's num_epochs
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
    echo "[exp7] cannot locate the ImMAP checkout (tried IMMAP_ROOT," \
         "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset}, ~/scratch/ee2178/ImMAP)." >&2
    echo "[exp7] submit from the repo root, or: IMMAP_ROOT=/path/to/ImMAP sbatch $0" >&2
    exit 1
fi

source "${IMMAP_ROOT}/torch/_mg_recon_body.sh"
