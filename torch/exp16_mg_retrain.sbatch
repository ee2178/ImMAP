#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=torch_pr_89_tandon_advanced
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint="h200|b200"
# ^ PACK=2 (two cells per GPU) wants the big ones. Feature names must match the
#   cluster's exactly -- `sinfo -o "%f"` lists them -- or sbatch rejects the job.
#SBATCH --cpus-per-task=16
#SBATCH --mem=96GB
#SBATCH --time=48:00:00
#SBATCH --job-name=Exp16-MGRetrain
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%A_%a.out
#SBATCH --error=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%A_%a.err
#SBATCH --array=0-3
#
# EXPERIMENT 16 -- EVERY MULTIGRID CELL OF THE PAPER RETRAINED WITH EXP15'S
# BEST V-CYCLE (2026-10-25), FULL LENGTH.
#
# <base>vtxAnestU = <base> with the vertex-centred transfer (sinc_vertex) and
# FAS option A (fas_dual=fixed_point), UNTIED copies, the parent-pi fix -- what
# exp15's mg81v6vtxAnestU is to mg81v6, nothing else changed
# (make_mg_recon_configs.py::EXP16). New names, so the old run dirs are kept
# and stay the "before" of the comparison.
#
# TWO SETS (SET=...), each one array:
#
#   main (default)  knee + brain, R8 + R12, 8 cells, --array=0-3
#       mg81v6vtxAnestU       MGLPDS-Small   (knee R8 is exp15's run -- reused)
#       mg169v6vtxAnestU      MGLPDS-Big
#     sbatch torch/exp16_mg_retrain.sbatch
#     MGGroup is NOT retrained for now (user, 2026-10-25). Its cell is defined;
#     with it the main set is 12 cells, --array=0-5:
#       ONLY="mg81v6vtxAnestU mg169v6vtxAnestU mggroup169v6vtxAnestU" \
#           sbatch --array=0-5 torch/exp16_mg_retrain.sbatch
#
#   ablation        knee R8, 5 cells, --array=0-2
#       mg100v6 mg121v6 mg144v6 (width), mg81v2 mg81v4 (depth), each vtxAnestU;
#       mg81v6 / mg169v6 come from the main set
#     SET=ablation sbatch --array=0-2 torch/exp16_mg_retrain.sbatch
#     Task 2 holds mg81v4 ALONE (5 cells, PACK=2), which left its H200 idle
#     enough to be cancelled (2026-10-27). A lone cell goes on an L40S, unpacked:
#       SET=ablation ONLY=mg81v4vtxAnestU PACK=1 sbatch --array=0 \
#           --constraint=l40s --mem=64GB torch/exp16_mg_retrain.sbatch
#
# PACK=2: each task trains two cells side by side on its GPU under CUDA MPS
# (torch/_mg_recon_body.sh, "PACK"), because one batch-1 cell leaves an H200
# idle enough for the cluster's low-utilization cancel. Task t of an anatomy
# trains that anatomy's cells 2t and 2t+1, in the order
#     python scripts/make_mg_recon_configs.py --list-cells --anatomy knee \
#         --accels 8 12 --only mg81v6vtxAnestU mg169v6vtxAnestU
# prints (R8 then R12, then the generator's cell order); the array is
# anatomy-major. Main set: task 0 knee R8, 1 knee R12, 2 brain R8, 3 brain R12,
# each Small + Big together. Each cell
# logs to torch/logs/slurm_<job>_<task>_<anatomy>_<model>_<R>.{out,err}; the task's own
# .out names them. One cell per GPU instead (twice the array):
#     PACK=1 sbatch --array=0-7 torch/exp16_mg_retrain.sbatch
#
# A cell that is still running elsewhere, or finished, is skipped by
# LAUNCH-ONCE and its partner trains alone; an unfinished one resumes. So
# after a 48 h slot ends, RESUBMIT THE SAME COMMAND.
#
# The baselines (lpdsnet, varnetmaps, RAM) are not multigrid and are not
# retrained. Evaluate the new cells against them with main_eval as usual.
#
# SUBMIT FROM THE REPO ROOT.

SET="${SET:-main}"
case "${SET}" in
    main)
        ANATOMIES="${ANATOMIES:-knee brain}"
        ACCELS="${ACCELS:-8 12}"
        ONLY="${ONLY:-mg81v6vtxAnestU mg169v6vtxAnestU}" ;;
    ablation)
        ANATOMIES="${ANATOMIES:-knee}"
        ACCELS="${ACCELS:-8}"
        ONLY="${ONLY:-mg100v6vtxAnestU mg121v6vtxAnestU mg144v6vtxAnestU mg81v2vtxAnestU mg81v4vtxAnestU}" ;;
    *)
        echo "[exp16] SET must be main or ablation, got '${SET}'" >&2
        exit 1 ;;
esac
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
    echo "[exp16] task ${SLURM_ARRAY_TASK_ID} is outside the ${_total} tasks of" \
         "ANATOMIES=\"${ANATOMIES}\" x ACCELS=\"${ACCELS}\" x ONLY=\"${ONLY}\" at PACK=${PACK}" \
         "-- fix --array (should be 0-$(( _total - 1 )))" >&2
    exit 1
fi
if [ "${SLURM_ARRAY_TASK_MAX:-0}" -lt "$(( _total - 1 ))" ] \
        && [ "${SLURM_ARRAY_TASK_ID}" -eq "${SLURM_ARRAY_TASK_MAX:--1}" ]; then
    echo "[exp16] NOTE: --array tops out at ${SLURM_ARRAY_TASK_MAX} but the selection" \
         "needs 0-$(( _total - 1 )) at PACK=${PACK}; the rest are NOT being run." >&2
fi
ANATOMY="${_anat[$(( SLURM_ARRAY_TASK_ID / _tpa ))]}"
echo "[exp16] SET=${SET} array task ${SLURM_ARRAY_TASK_ID} of ${_total} -> ${ANATOMY}," \
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
    echo "[exp16] cannot locate the ImMAP checkout (tried IMMAP_ROOT," \
         "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset}, ~/scratch/ee2178/ImMAP)." >&2
    echo "[exp16] submit from the repo root, or: IMMAP_ROOT=/path/to/ImMAP sbatch $0" >&2
    exit 1
fi

source "${IMMAP_ROOT}/torch/_mg_recon_body.sh"
