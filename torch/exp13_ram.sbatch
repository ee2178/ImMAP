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
#SBATCH --job-name=Exp13-RAM
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%A_%a.out
#SBATCH --error=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%A_%a.err
#SBATCH --array=0-3
#
# EXPERIMENT 13 -- RAM (Reconstruct Anything Model) trained FROM SCRATCH as a
# baseline, BRAIN and KNEE, R=8 and R=12, under the same protocol as every other
# cell of the main table (measured k-space + AWGN at sigma ~ U[0.01, 0.02], val
# 0.015, Julia masks, online ESPIRiT maps, organ-masked magnitude-nl1-nl2 loss,
# 6000 x 50 steps, Adam 5e-4).
#
#   0  brain R8    1  brain R12    2  knee R8    3  knee R12
#
# ONE SETTING DIFFERS FROM THE EARLIER CELLS, DELIBERATELY: backtracking. These
# runs get the defaults of 2026-10-19 -- restore when the epoch loss exceeds 100x
# its best, or a validation loss 2x its best (training/recon.py) -- while the
# varnetmaps / lpdsnet / mg runs recorded backtrack_ratio 500 and no validation
# check in their config.json. Neither check fires on a healthy run; they differ
# only if a run starts to diverge. Kept on purpose (decided 2026-10-20).
#
# THE MODEL. models/ram/: github.com/matthieutrs/ram's network vendored with one
# import moved, behind an adapter that wraps E as a deepinv physics and passes
# sigma / sqrt(2) (deepinv's per-real-component std). Complex head only, 33.6M
# parameters. `python -m tests.test_ram` checks the vendoring against upstream
# and the adapter's output against upstream's own forward, bit for bit.
#
# ONE-TIME SETUP: deepinv in the gcdl env. Compute nodes have no route out, so on
# a LOGIN node:
#     conda activate gcdl
#     pip install --dry-run deepinv==0.4.2     # read what it would change FIRST
#     pip install deepinv==0.4.2
# The dry run must NOT list torch. If it does (it pulls a newer torch when
# torchvision is missing or mismatched), stop and install with --no-deps plus
# the small pure-python deps instead, so the CUDA torch build stays:
#     pip install --no-deps deepinv==0.4.2
#     pip install einops natsort torchmetrics
# 0.4.2 is the version the port was tested against (2026-10-20). Verify with
#     python -m tests.test_ram
#
# COST. Every one of RAM's 28 residual blocks applies A^H A three times at FULL
# resolution (a 20-coil FFT pair each), on top of a 33.6M-parameter U-Net at
# 768x400. MEMORY is not the problem: one training step at full brain size
# (768x396, 20 coils) peaked ~7.5 GB above baseline (measured on CPU,
# 2026-10-20), well inside an L40S. SPEED is unmeasured on a GPU and
# expected to be the slowest cell per step. PROBE FIRST -- one cell,
# 20 epochs, tagged (lands in ram_R8_20ep), whose log gives the it/s and the
# peak memory:
#     SWEEP_EPOCHS=20 sbatch --array=0 torch/exp13_ram.sbatch
# 300k steps fit one 48 h slot only above ~1.75 it/s; below that, RESUBMIT this
# file when a slot ends -- LAUNCH-ONCE (torch/_mg_recon_body.sh) resumes each
# unfinished cell from its checkpoint and skips finished or queued ones.
#
# OTHER SELECTIONS, as in the other launchers (anatomy-major, then R):
#     ANATOMIES=knee sbatch --array=0-1 torch/exp13_ram.sbatch
#     ACCELS=12 sbatch --array=0-1 torch/exp13_ram.sbatch
#
# SUBMIT FROM THE REPO ROOT.
#   sbatch torch/exp13_ram.sbatch

ANATOMIES="${ANATOMIES:-brain knee}"
ONLY="${ONLY:-ram}"
ACCELS="${ACCELS:-8 12}"
# PLANAR_STATE and ATTN do nothing to RAM (the generator writes neither for it).
# They are set to what exp11/exp12 use because REGENERATE rewrites EVERY config
# of the anatomy: anything else would strip planar_state from the multigrid
# cells' configs between their launches.
PLANAR_STATE="${PLANAR_STATE-1}"
ATTN="flex"
ORGAN_MASK=1                         # matches every other launcher of either
                                     # anatomy -- regeneration rewrites every
                                     # config of the anatomy it touches
CONFIG_ROOT="config"
SWEEP_EPOCHS="${SWEEP_EPOCHS:-}"     # empty = the config's num_epochs (full length)
REGENERATE=1                         # 1 = regenerate configs before running

# ---- one array over several anatomies ---------------------------------------------------
# The shared body runs ONE anatomy and indexes that anatomy's own cell list
# (ACCELS-major, then ONLY, in the generator's order). So split this array's
# index into (anatomy, index within the anatomy) and hand the body the second.
: "${SLURM_ARRAY_TASK_ID:?must be run as an array job (sbatch sets SLURM_ARRAY_TASK_ID)}"
_anat=(${ANATOMIES//,/ })
_only=(${ONLY//,/ })
_acc=(${ACCELS//,/ })
_per=$(( ${#_only[@]} * ${#_acc[@]} ))
_total=$(( _per * ${#_anat[@]} ))
if [ "${_per}" -eq 0 ] || [ "${SLURM_ARRAY_TASK_ID}" -ge "${_total}" ]; then
    echo "[exp13] task ${SLURM_ARRAY_TASK_ID} is outside the ${_total} cells of" \
         "ANATOMIES=\"${ANATOMIES}\" x ACCELS=\"${ACCELS}\" x ONLY=\"${ONLY}\"" \
         "-- fix --array (should be 0-$(( _total - 1 )))" >&2
    exit 1
fi
if [ "${SLURM_ARRAY_TASK_MAX:-0}" -lt "$(( _total - 1 ))" ] \
        && [ "${SLURM_ARRAY_TASK_ID}" -eq "${SLURM_ARRAY_TASK_MAX:--1}" ]; then
    echo "[exp13] NOTE: --array tops out at ${SLURM_ARRAY_TASK_MAX} but the selection" \
         "has ${_total} cells; the rest are NOT being run." >&2
fi
ANATOMY="${_anat[$(( SLURM_ARRAY_TASK_ID / _per ))]}"
echo "[exp13] array task ${SLURM_ARRAY_TASK_ID} of ${_total} -> ${ANATOMY}," \
     "cell $(( SLURM_ARRAY_TASK_ID % _per )) of that anatomy's ${_per}"
SLURM_ARRAY_TASK_ID=$(( SLURM_ARRAY_TASK_ID % _per ))
SLURM_ARRAY_TASK_MAX=$(( _per - 1 ))

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
    echo "[exp13] cannot locate the ImMAP checkout (tried IMMAP_ROOT," \
         "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset}, ~/scratch/ee2178/ImMAP)." >&2
    echo "[exp13] submit from the repo root, or: IMMAP_ROOT=/path/to/ImMAP sbatch $0" >&2
    exit 1
fi

# PRE-FLIGHT on this job's own GPU: tests/test_ram.py runs RAM forward + backward
# on CUDA (its section 9) besides the CPU checks, so a device or install problem
# fails here in seconds instead of after wandb/data setup -- the first exp13
# submission died on a cuda/cpu mix in realign_input. Same activation as the body
# (activate first, guards after; see the note there).
source ~/.bashrc
conda activate gcdl
if ! ( cd "${IMMAP_ROOT}" && python -m tests.test_ram ); then
    echo "[exp13] tests/test_ram.py FAILED on $(hostname) -- not training." >&2
    exit 1
fi

source "${IMMAP_ROOT}/torch/_mg_recon_body.sh"
