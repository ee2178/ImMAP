#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=torch_pr_89_tandon_advanced
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --job-name=MainEval
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%A_%a.out
#SBATCH --error=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%A_%a.err
#SBATCH --array=0-3
#
# PER-VOLUME METRICS FOR THE MAIN COMPARISON (scripts/eval_per_volume.py):
# PSNR, SSIM, NRMSE, LPIPS on every val volume, for the significance tests in
# scripts/wilcoxon_main.py. RUNS_PER_TASK runs per array task, one after the
# other on that task's GPU (2026-10-29: was one GPU per run, 16 of them, which
# queued longer than the runs took). With the default 5:
#
#   0  brain R8   varnetmaps lpdsnet mg81v6 mg169v6 ram
#   1  brain R12  varnetmaps lpdsnet mg81v6 mg169v6 ram
#   2  knee  R8   varnetmaps lpdsnet mg81v6 mg169v6 ram
#   3  knee  R12  varnetmaps lpdsnet mg81v6 mg169v6 ram
#
# RAM is exp13's `ram` cell (torch/exp13_ram.sbatch), labelled "RAM". `ramdrunet`,
# its DRUNet-initialised twin ("RAM-DRUNet"), is not trained; RUNS= scores it if it ever is.
# MGLPDS-Big is `mg169v6` for both anatomies (brain's full-length runs finished
# 2026-10-21; until then brain used the `mglpds` run as a stopgap). A directory
# carrying RENAMED_FROM_mglpds.txt is that stopgap renamed, NOT a trained
# mg169v6, and is refused (STOPGAP_OK=1 scores it anyway). A run that cannot be
# found FAILS its task -- after the task's other runs are scored -- rather than
# exiting clean with nothing scored.
# Slices: each run's training range (brain 0:8, knee 12:25).
#
# BEFORE THE FIRST SUBMIT, on a LOGIN node (compute nodes have no route out, and
# LPIPS needs pretrained AlexNet weights in ~/.cache/torch):
#   conda activate gcdl
#   python -c "from evaluation.metrics import warm_up; warm_up(['lpips'], 'cpu')"
#
# Submit from the repo root:
#   sbatch torch/main_eval.sbatch                  # all 20 runs on 4 GPUs
#   sbatch --array=2-3 torch/main_eval.sbatch      # knee only
#   RUNS_PER_TASK=10 sbatch --array=0-1 ...        # 2 GPUs, one per anatomy
#   RUNS_PER_TASK=1 sbatch --array=5,11 ...        # single runs, as before
# Task t runs entries t*RUNS_PER_TASK .. (t+1)*RUNS_PER_TASK-1 of RUNS, so
# --array must cover ceil(#RUNS / RUNS_PER_TASK) tasks.
#
# THE TEST MEASUREMENT: measured k-space + AWGN by default (2026-10-22), as the
# nets train -> results/main. From 2026-10-20 to 2026-10-22 this defaulted to
# SIMULATED k-space (the SENSE ground truth through the stored maps, mask and
# AWGN, so the noise is exactly sigma), which now has to be asked for and goes
# to its own tree, so results/main is never mixed:
#   sbatch torch/main_eval.sbatch                         -> results/main
#   VAL_KSPACE=simulated sbatch torch/main_eval.sbatch    -> results/main_simulated
#   SIGMA=0.03 sbatch torch/main_eval.sbatch              -> results/main_s0.03
# SIGMA empty = each run's val_noise_std (0.015). Point the table at the tree:
#   python figures/table_main.py --in results/main/volumes.csv
#
# RE-SCORING ONE METRIC (NRMSE was redefined 2026-10-20 to ||gt-pred||/||gt||),
# on the measured-k-space results already in results/main:
#   VAL_KSPACE=measurement_awgn RECOMPUTE=nrmse sbatch torch/main_eval.sbatch
# reruns inference on the rows each run already has and replaces ONLY that
# column -- PSNR / SSIM / LPIPS are kept, and LPIPS is not even recomputed.
# Resumable like the main pass. Until a run is re-scored, the merge leaves its
# NRMSE blank rather than mixing the two definitions.
#
# RESUMABLE: rows are written as they go to results/main/<anatomy>/<run>.slices.csv;
# resubmitting the same task finishes the volumes that are missing. A newer
# checkpoint or a different sigma/seed/slices starts that run over.
#
# MERGED AUTOMATICALLY: every task ends by merging all runs written so far into
# results/main/slices.csv and volumes.csv, so the last task to finish leaves the
# complete set -- no separate --combine step. Then copy volumes.csv down (it is
# all the plots and tests read) and:
#   python figures/plot_main_bars.py
#   python scripts/wilcoxon_main.py --ref MGLPDS-Big
#
# NEW RUNS ONLY (e.g. the RAM cells) -- runs already scored resume as done, in
# seconds, so a whole group can be resubmitted; or one task over just them:
#   RUNS="brain/ram_R8 brain/ram_R12 knee/ram_R8 knee/ram_R12" \
#   RUNS_PER_TASK=4 sbatch --array=0 torch/main_eval.sbatch

# OTHER RUN SETS (e.g. the exp14 knee ablations): RUNS="<anatomy>/<run> ..."
# replaces the list below -- RUNS_PER_TASK entries per array task, in the order
# given -- and OUT keeps their results out of the main table's tree:
#   RUNS="knee/mg81v6_R8 knee/mg100v6_R8 knee/mg121v6_R8 knee/mg144v6_R8 knee/mg169v6_R8 knee/mg81v2_R8 knee/mg81v4_R8" \
#   OUT=results/ablation_knee_R8 RUNS_PER_TASK=4 sbatch --array=0-1 torch/main_eval.sbatch
# Runs already scored in another tree are scored again here (same seed, same
# numbers), so the ablation tree is complete on its own.
_RUNS_GIVEN="${RUNS:-}"
RUNS=(
    brain/varnetmaps_R8  brain/lpdsnet_R8  brain/mg81v6_R8  brain/mg169v6_R8  brain/ram_R8
    brain/varnetmaps_R12 brain/lpdsnet_R12 brain/mg81v6_R12 brain/mg169v6_R12 brain/ram_R12
    knee/varnetmaps_R8   knee/lpdsnet_R8   knee/mg81v6_R8   knee/mg169v6_R8   knee/ram_R8
    knee/varnetmaps_R12  knee/lpdsnet_R12  knee/mg81v6_R12  knee/mg169v6_R12  knee/ram_R12
)
if [ -n "${_RUNS_GIVEN}" ]; then
    read -r -a RUNS <<< "${_RUNS_GIVEN}"
fi
RUNS_PER_TASK="${RUNS_PER_TASK:-5}"
# Rediscretized multigrid runs LAUNCHED BEFORE 2026-10-07 trained with the
# Galerkin coarse operator on slice sizes that do not divide by the stride
# (operators/coarse.py); they must be evaluated the same way. Of the runs above
# that is brain mg81v6_R12 (exp9, 2026-10-02). The knee runs, brain mg81v6_R8
# and brain mg169v6 were launched after the change.
EMBEDDED_OFF="${EMBEDDED_OFF:-brain/mg81v6_R12}"

METRICS="${METRICS:-psnr ssim nrmse lpips}"
SEED="${SEED:-1234}"
SIGMA="${SIGMA:-}"                     # empty = each run's own val_noise_std
# Measured k-space + AWGN by default (2026-10-22), how the nets train and the
# measurement results/main has always used. VAL_KSPACE=simulated -> results/main_simulated.
VAL_KSPACE="${VAL_KSPACE:-measurement_awgn}"
if [ "${VAL_KSPACE}" = "measurement_awgn" ]; then _out="results/main"; else _out="results/main_${VAL_KSPACE}"; fi
# TRANSFER_FILTER: swap the multigrid nets' grid-transfer filter at inference
# (same weights): "none" = the sinc off, or "box", "bilinear", ... VarNet /
# LPDSNet have no transfers and are scored as they are, so the tree is complete.
# Its own tree, e.g. results/main_filtnone:
#   TRANSFER_FILTER=none VAL_KSPACE=measurement_awgn sbatch torch/main_eval.sbatch
TRANSFER_FILTER="${TRANSFER_FILTER:-}"
OUT="${OUT:-${_out}${SIGMA:+_s${SIGMA}}${TRANSFER_FILTER:+_filt${TRANSFER_FILTER}}}"
WORKERS="${WORKERS:-6}"
RECOMPUTE="${RECOMPUTE:-}"             # e.g. RECOMPUTE=nrmse: re-score that column only
EXTRA="${EXTRA:-}"                     # e.g. EXTRA="--n-volumes 3" for a probe

for _c in "${IMMAP_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" "$HOME/scratch/ee2178/ImMAP"; do
    if [ -n "${_c}" ] && [ -f "${_c}/scripts/eval_per_volume.py" ]; then
        IMMAP_ROOT="${_c}"
        break
    fi
done
if [ ! -f "${IMMAP_ROOT:-}/scripts/eval_per_volume.py" ]; then
    echo "[eval] cannot locate scripts/eval_per_volume.py; submit from the repo root" \
         "or set IMMAP_ROOT" >&2
    exit 1
fi

# Activate first, then the guards (see the no-set-euo note in the other launchers).
source ~/.bashrc
conda activate gcdl
cd "${IMMAP_ROOT}"

set -eo pipefail
: "${SLURM_ARRAY_TASK_ID:?must be run as an array job}"

FIRST=$(( SLURM_ARRAY_TASK_ID * RUNS_PER_TASK ))
if [ "${FIRST}" -ge "${#RUNS[@]}" ]; then
    echo "[eval] task ${SLURM_ARRAY_TASK_ID} starts at entry ${FIRST} of ${#RUNS[@]} runs" \
         "-- fix --array (0-$(( (${#RUNS[@]} + RUNS_PER_TASK - 1) / RUNS_PER_TASK - 1 ))" \
         "at RUNS_PER_TASK=${RUNS_PER_TASK})" >&2
    exit 1
fi
TASK_RUNS=("${RUNS[@]:${FIRST}:${RUNS_PER_TASK}}")

echo "[eval] task ${SLURM_ARRAY_TASK_ID}: ${TASK_RUNS[*]}"
echo "[eval] metrics='${METRICS}' seed=${SEED} val_kspace=${VAL_KSPACE} out=${OUT}"
echo "[eval] $(date '+%F %T')  node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

RUNS_ROOT="trained_nets/mg_recon"
mkdir -p "${OUT}"

# One run: check it, score it, merge. Returns non-zero on a failure, so the
# task goes on to its next run (set -e does not apply inside a `||` call).
eval_run() {
    local ENTRY="$1" ANATOMY RUN EMB=1 e
    ANATOMY="${ENTRY%%/*}"
    RUN="${ENTRY#*/}"

    # The renamed brain stopgap (see MGLPDS-Big above) must not pass for the real run.
    if [ -f "${RUNS_ROOT}/${ANATOMY}/${RUN}/RENAMED_FROM_mglpds.txt" ] && [ "${STOPGAP_OK:-0}" != "1" ]; then
        echo "[eval] ${RUNS_ROOT}/${ANATOMY}/${RUN} is the renamed mglpds stopgap"          "(RENAMED_FROM_mglpds.txt), not a trained mg169v6 -- move it aside, or"          "STOPGAP_OK=1 to score it anyway" >&2
        return 1
    fi
    if [ ! -f "${RUNS_ROOT}/${ANATOMY}/${RUN}/config.json" ] \
            || [ ! -f "${RUNS_ROOT}/${ANATOMY}/${RUN}/net.ckpt" ]; then
        echo "[eval] ${RUNS_ROOT}/${ANATOMY}/${RUN} has no config.json + net.ckpt --" \
             "nothing to evaluate for ${ENTRY}" >&2
        return 1
    fi

    for e in ${EMBEDDED_OFF}; do
        [ "${e}" = "${ENTRY}" ] && EMB=0
    done

    echo "[eval] ${ENTRY}  IMMAP_REDISCRETIZE_EMBEDDED=${EMB}  $(date '+%F %T')"
    IMMAP_REDISCRETIZE_EMBEDDED="${EMB}" python scripts/eval_per_volume.py \
        --anatomy "${ANATOMY}" --run "${RUN}" \
        --metrics ${METRICS} \
        --seed "${SEED}" \
        --workers "${WORKERS}" \
        --out "${OUT}" \
        ${SIGMA:+--sigma "${SIGMA}"} \
        --val-kspace-type "${VAL_KSPACE}" \
        ${TRANSFER_FILTER:+--transfer-filter "${TRANSFER_FILTER}"} \
        ${RECOMPUTE:+--recompute ${RECOMPUTE}} \
        ${EXTRA} || return 1
    echo "[eval] ${ENTRY} done $(date '+%F %T')"

    # Merge every run written so far (see MERGED AUTOMATICALLY above), after
    # EACH run, so a task that hits its time limit still leaves its finished
    # runs merged. Tasks can finish at the same moment, so the merge runs under
    # a lock; it writes through a temporary file, so a reader never sees a
    # half-written CSV.
    flock "${OUT}/.combine.lock" python scripts/eval_per_volume.py --out "${OUT}" --combine \
        || return 1
}

FAILED=()
for ENTRY in "${TASK_RUNS[@]}"; do
    eval_run "${ENTRY}" || FAILED+=("${ENTRY}")
done
if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "[eval] FAILED: ${FAILED[*]} -- see above; the task's other runs are scored" >&2
    exit 1
fi
echo "[eval] merged into ${OUT}/volumes.csv -- copy that down for the plots"
