#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=torch_pr_89_tandon_advanced
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=08:00:00
#SBATCH --job-name=MainEval
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%A_%a.out
#SBATCH --error=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%A_%a.err
#SBATCH --array=0-15
#
# PER-VOLUME METRICS FOR THE MAIN COMPARISON (scripts/eval_per_volume.py):
# PSNR, SSIM, NRMSE, LPIPS on every val volume, for the significance tests in
# scripts/wilcoxon_main.py. One array task per RUN:
#
#    0- 3  brain R8   varnetmaps lpdsnet mg81v6 mg169v6
#    4- 7  brain R12  varnetmaps lpdsnet mg81v6 mg169v6
#    8-11  knee  R8   varnetmaps lpdsnet mg81v6 mg169v6
#   12-15  knee  R12  varnetmaps lpdsnet mg81v6 mg169v6
#
# MGLPDS-Big is `mg169v6` for both anatomies (brain's full-length runs finished
# 2026-10-21; until then brain used the `mglpds` run as a stopgap). A directory
# carrying RENAMED_FROM_mglpds.txt is that stopgap renamed, NOT a trained
# mg169v6, and is refused (STOPGAP_OK=1 scores it anyway). A task whose run
# cannot be found FAILS rather than exiting clean with nothing scored.
# Slices: each run's training range (brain 0:8, knee 12:25).
#
# BEFORE THE FIRST SUBMIT, on a LOGIN node (compute nodes have no route out, and
# LPIPS needs pretrained AlexNet weights in ~/.cache/torch):
#   conda activate gcdl
#   python -c "from evaluation.metrics import warm_up; warm_up(['lpips'], 'cpu')"
#
# Submit from the repo root:
#   sbatch torch/main_eval.sbatch                  # all 16
#   sbatch --array=8-15 torch/main_eval.sbatch     # knee only
#
# THE TEST MEASUREMENT (2026-10-20): SIMULATED k-space by default -- the SENSE
# ground truth through the stored maps, mask and AWGN, so the noise is exactly
# sigma -- while the nets were TRAINED on measured k-space + AWGN. The operator
# still uses online ESPIRiT maps, as in training. Results go to their own tree,
# so the earlier measured-k-space results in results/main are left alone:
#   sbatch torch/main_eval.sbatch                         -> results/main_simulated
#   SIGMA=0.03 sbatch torch/main_eval.sbatch              -> results/main_simulated_s0.03
#   VAL_KSPACE=measurement_awgn sbatch ...                -> results/main (the old setup)
# SIGMA empty = each run's val_noise_std (0.015). Point the table at the tree:
#   python figures/table_main.py --in results/main_simulated/volumes.csv
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
# NEW RUNS ONLY (e.g. brain MGLPDS-Big, 2026-10-21) -- the others resume as done:
#   VAL_KSPACE=measurement_awgn sbatch --array=3,7 torch/main_eval.sbatch

RUNS=(
    brain/varnetmaps_R8  brain/lpdsnet_R8  brain/mg81v6_R8  brain/mg169v6_R8
    brain/varnetmaps_R12 brain/lpdsnet_R12 brain/mg81v6_R12 brain/mg169v6_R12
    knee/varnetmaps_R8   knee/lpdsnet_R8   knee/mg81v6_R8   knee/mg169v6_R8
    knee/varnetmaps_R12  knee/lpdsnet_R12  knee/mg81v6_R12  knee/mg169v6_R12
)
# Rediscretized multigrid runs LAUNCHED BEFORE 2026-10-07 trained with the
# Galerkin coarse operator on slice sizes that do not divide by the stride
# (operators/coarse.py); they must be evaluated the same way. Of the runs above
# that is brain mg81v6_R12 (exp9, 2026-10-02). The knee runs, brain mg81v6_R8
# and brain mg169v6 were launched after the change.
EMBEDDED_OFF="${EMBEDDED_OFF:-brain/mg81v6_R12}"

METRICS="${METRICS:-psnr ssim nrmse lpips}"
SEED="${SEED:-1234}"
SIGMA="${SIGMA:-}"                     # empty = each run's own val_noise_std
VAL_KSPACE="${VAL_KSPACE:-simulated}"  # or measurement_awgn (how the nets trained)
if [ "${VAL_KSPACE}" = "measurement_awgn" ]; then _out="results/main"; else _out="results/main_${VAL_KSPACE}"; fi
OUT="${OUT:-${_out}${SIGMA:+_s${SIGMA}}}"
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

if [ "${SLURM_ARRAY_TASK_ID}" -ge "${#RUNS[@]}" ]; then
    echo "[eval] task ${SLURM_ARRAY_TASK_ID} >= ${#RUNS[@]} runs -- fix --array" >&2
    exit 1
fi
ENTRY="${RUNS[${SLURM_ARRAY_TASK_ID}]}"
ANATOMY="${ENTRY%%/*}"
RUN="${ENTRY#*/}"

RUNS_ROOT="trained_nets/mg_recon"
# The renamed brain stopgap (see MGLPDS-Big above) must not pass for the real run.
if [ -f "${RUNS_ROOT}/${ANATOMY}/${RUN}/RENAMED_FROM_mglpds.txt" ] && [ "${STOPGAP_OK:-0}" != "1" ]; then
    echo "[eval] ${RUNS_ROOT}/${ANATOMY}/${RUN} is the renamed mglpds stopgap"          "(RENAMED_FROM_mglpds.txt), not a trained mg169v6 -- move it aside, or"          "STOPGAP_OK=1 to score it anyway" >&2
    exit 1
fi
if [ ! -f "${RUNS_ROOT}/${ANATOMY}/${RUN}/config.json" ] \
        || [ ! -f "${RUNS_ROOT}/${ANATOMY}/${RUN}/net.ckpt" ]; then
    echo "[eval] ${RUNS_ROOT}/${ANATOMY}/${RUN} has no config.json + net.ckpt --" \
         "nothing to evaluate for ${ENTRY}" >&2
    exit 1
fi

for e in ${EMBEDDED_OFF}; do
    if [ "${e}" = "${ENTRY}" ]; then
        export IMMAP_REDISCRETIZE_EMBEDDED=0
    fi
done

echo "[eval] ${ENTRY}  metrics='${METRICS}' seed=${SEED} val_kspace=${VAL_KSPACE} out=${OUT}" \
     "IMMAP_REDISCRETIZE_EMBEDDED=${IMMAP_REDISCRETIZE_EMBEDDED:-1}"
echo "[eval] $(date '+%F %T')  node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python scripts/eval_per_volume.py \
    --anatomy "${ANATOMY}" --run "${RUN}" \
    --metrics ${METRICS} \
    --seed "${SEED}" \
    --workers "${WORKERS}" \
    --out "${OUT}" \
    ${SIGMA:+--sigma "${SIGMA}"} \
    --val-kspace-type "${VAL_KSPACE}" \
    ${RECOMPUTE:+--recompute ${RECOMPUTE}} \
    ${EXTRA}

echo "[eval] ${ENTRY} done $(date '+%F %T')"

# Merge every run written so far (see MERGED AUTOMATICALLY above). Array tasks
# can finish at the same moment, so the merge runs under a lock; it writes
# through a temporary file, so a reader never sees a half-written CSV.
mkdir -p "${OUT}"
flock "${OUT}/.combine.lock" python scripts/eval_per_volume.py --out "${OUT}" --combine
echo "[eval] merged into ${OUT}/volumes.csv -- copy that down for the plots"
