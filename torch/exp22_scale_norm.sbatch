#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=torch_pr_89_tandon_advanced
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint="h200|b200"
# ^ PACK=2 (two cells per GPU) wants the big ones, as exp18. Feature names must match the
#   cluster's exactly -- `sinfo -o "%f"` lists them -- or sbatch rejects the job.
#SBATCH --cpus-per-task=16
#SBATCH --mem=96GB
#SBATCH --time=48:00:00
#SBATCH --job-name=Exp22-ScaleNorm
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%A_%a.out
#SBATCH --error=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%A_%a.err
#SBATCH --array=0
#
# EXPERIMENT 22 -- SCALE-NORMALIZED THRESHOLDS (2026-11-05), full length, exp18's mask
# (center_frac = 0.32 / R, adjust_accel off). `scale_norm="p99"` runs each net on y / s and
# sigma / s, s = the 99th percentile of |E^H y| / mean(E^H E 1) (the zero-filled image's
# bright-tissue level), and scales the output back: exactly scale-equivariant, thresholds
# in units of the slice's own brightness (make_mg_recon_configs.py::EXP22,
# models/mg_lpds.py; the statistic was chosen by scripts/scale_stat_compare.py). No cg
# start, so each cell is its exp18 twin plus this; mg81 also takes exp20's
# lam_level_scale=0.5. lam0 = 1.1e-3 in normalized units. ramcf32 is the reference and is
# not retrained.
#
# Cells: lpdsnetsncf32 mg81v6vtxAnestUsnlamcf32, knee R8: 2 cells, 1 task at PACK=2.
#
#   sbatch torch/exp22_scale_norm.sbatch                                  # knee R8
#   ACCELS="8 12" sbatch --array=0-1 torch/exp22_scale_norm.sbatch        # + R12
#
# Check the index map:
#   python scripts/make_mg_recon_configs.py --list-cells --anatomy knee --accels 8 #       --only lpdsnetsncf32 mg81v6vtxAnestUsnlamcf32
#
# A cell still running elsewhere, or finished, is skipped by LAUNCH-ONCE and an
# unfinished one resumes, so after a 48 h slot ends, RESUBMIT THE SAME COMMAND.
#
# SUBMIT FROM THE REPO ROOT.

ANATOMIES=brain sbatch torch/exp22_cg_init.sbatch                    # brain, 4 GPUs
#   ANATOMIES="knee brain" sbatch --array=0-7 torch/exp22_cg_init.sbatch # both
#   ACCELS=8 sbatch --array=0-1 torch/exp22_cg_init.sbatch               # knee R8 only
#
# Check the index map:
#   python scripts/make_mg_recon_configs.py --list-cells --anatomy knee --accels 8 12 \
#       --only varnetmapscgcf32 lpdsnetcgcf32 mg81v6vtxAnestUcgcf32 mg169v6vtxAnestUcgcf32
#
# A cell still running elsewhere, or finished, is skipped by LAUNCH-ONCE and an
# unfinished one resumes, so after a 48 h slot ends, RESUBMIT THE SAME COMMAND.
#
# SUBMIT FROM THE REPO ROOT.

ANATOMIES="${ANATOMIES:-knee}"
ACCELS="${ACCELS:-8}"
ONLY="${ONLY:-lpdsnetsncf32 mg81v6vtxAnestUsnlamcf32}"
PACK="${PACK:-2}"
PLANAR_STATE="${PLANAR_STATE-1}"     # `-`, not `:-`: an explicitly EMPTY value turns it off
ATTN="flex"                          # stated, not inherited: the run dir should
                                     # record which backend produced it
ORGAN_MASK=1                         # matches every other launcher of either
                                     # anatomy -- regeneration rewrites every
                                     # config of the anatomy it touches
CONFIG_ROOT="config"
SWEEP_EPOCHS="${SWEEP_EPOCHS:-}"     # empty = the config's num_epochs (full length)
REGENERATE=1                         # 1 = regenerate configs before running

# ---- one array over several anatomies, PACK cells per task --------------------------------
# The shared body runs ONE anatomy, and under PACK its task t means that
# anatomy's cells t*PACK .. t*PACK+PACK-1. So split this array's index into
# (anatomy, task within the anatomy) and hand the body the second.
: "${SLURM_ARRAY_TASK_ID:?must be run as an array job (sbatch sets SLURM_ARRAY_TASK_ID)}"
_anat=(${ANATOMIES//,/ })
_only=(${ONLY//,/ })
_acc=(${ACCELS//,/ })
_per=$(( ${#_only[@]} * ${#_acc[@]} ))               # cells per anatomy
_tpa=$(( (_per + PACK - 1) / PACK ))                 # tasks per anatomy
_total=$(( _tpa * ${#_anat[@]} ))
if [ "${_per}" -eq 0 ] || [ "${SLURM_ARRAY_TASK_ID}" -ge "${_total}" ]; then
    echo "[exp22] task ${SLURM_ARRAY_TASK_ID} is outside the ${_total} tasks of" \
         "ANATOMIES=\"${ANATOMIES}\" x ACCELS=\"${ACCELS}\" x ONLY=\"${ONLY}\" at PACK=${PACK}" \
         "-- fix --array (should be 0-$(( _total - 1 )))" >&2
    exit 1
fi
if [ "${SLURM_ARRAY_TASK_MAX:-0}" -lt "$(( _total - 1 ))" ] \
        && [ "${SLURM_ARRAY_TASK_ID}" -eq "${SLURM_ARRAY_TASK_MAX:--1}" ]; then
    echo "[exp22] NOTE: --array tops out at ${SLURM_ARRAY_TASK_MAX} but the selection" \
         "needs 0-$(( _total - 1 )) at PACK=${PACK}; the rest are NOT being run." >&2
fi
ANATOMY="${_anat[$(( SLURM_ARRAY_TASK_ID / _tpa ))]}"
echo "[exp22] SET=${SET} array task ${SLURM_ARRAY_TASK_ID} of ${_total} -> ${ANATOMY}," \
     "task $(( SLURM_ARRAY_TASK_ID % _tpa )) of that anatomy's ${_tpa} (PACK=${PACK})"
PACK_LOG_TASK="${SLURM_ARRAY_TASK_ID}"       # the cells' log names keep the real index
SLURM_ARRAY_TASK_ID=$(( SLURM_ARRAY_TASK_ID % _tpa ))
SLURM_ARRAY_TASK_MAX=$(( _tpa - 1 ))

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
    echo "[exp22] cannot locate the ImMAP checkout (tried IMMAP_ROOT," \
         "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset}, ~/scratch/ee2178/ImMAP)." >&2
    echo "[exp22] submit from the repo root, or: IMMAP_ROOT=/path/to/ImMAP sbatch $0" >&2
    exit 1
fi

source "${IMMAP_ROOT}/torch/_mg_recon_body.sh"
