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
#SBATCH --job-name=Exp16-AblEval
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%A_%a.out
#SBATCH --error=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%A_%a.err
#SBATCH --array=0-5
#
# PER-VOLUME METRICS FOR THE exp16 KNEE R8 ABLATIONS (torch/exp16_mg_retrain.sbatch,
# SET=ablation, plus the two main-set cells they are anchored on), and the
# ablation table built from them (figures/table_ablation.py). One array task
# per RUN, scored by torch/main_eval.sbatch exactly as the main comparison is
# (measured k-space + AWGN at each run's val sigma, seed 1234, slices 12:25,
# organ mask), into their own tree:
#
#   0  knee/mg81v6vtxAnestU_R8    width M=81  + depth V=6   (the reference)
#   1  knee/mg100v6vtxAnestU_R8   width M=100
#   2  knee/mg121v6vtxAnestU_R8   width M=121
#   3  knee/mg144v6vtxAnestU_R8   width M=144
#   4  knee/mg169v6vtxAnestU_R8   width M=169
#   5  knee/mg81v2vtxAnestU_R8    depth V=2
#
# mg81v4vtxAnestU (depth V=4) is LEFT OUT for now: its training was cancelled
# (low GPU use) on 2026-10-27 and resubmitted on an L40S. Once it finishes,
# score it into the same tree -- the others resume as done, and the table is
# rebuilt with it:
#   RUNS="knee/mg81v4vtxAnestU_R8" sbatch --array=0 torch/exp16_ablation_eval.sbatch
#
# BEFORE THE FIRST SUBMIT: the LPIPS weights, as for main_eval.sbatch (login node):
#   conda activate gcdl
#   python -c "from evaluation.metrics import warm_up; warm_up(['lpips'], 'cpu')"
#
# Submit from the repo root:
#   sbatch torch/exp16_ablation_eval.sbatch
#
# Every task merges what is scored so far into ${OUT}/volumes.csv and rebuilds
# the table at ${OUT}/ablation_table.{tex,md,csv}, so the last task to finish
# leaves both complete. Copy the table down, or rebuild it locally from the
# tree (the table reads the */*.meta.json for the parameter counts):
#   python figures/table_ablation.py --in results/ablation16_knee_R8/volumes.csv

ABL_RUNS="${RUNS:-knee/mg81v6vtxAnestU_R8 knee/mg100v6vtxAnestU_R8 knee/mg121v6vtxAnestU_R8 knee/mg144v6vtxAnestU_R8 knee/mg169v6vtxAnestU_R8 knee/mg81v2vtxAnestU_R8}"
OUT="${OUT:-results/ablation16_knee_R8}"
VAL_KSPACE="${VAL_KSPACE:-measurement_awgn}"   # how the nets train, and results/main

for _c in "${IMMAP_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" "$HOME/scratch/ee2178/ImMAP"; do
    if [ -n "${_c}" ] && [ -f "${_c}/torch/main_eval.sbatch" ]; then
        IMMAP_ROOT="${_c}"
        break
    fi
done
if [ ! -f "${IMMAP_ROOT:-}/torch/main_eval.sbatch" ]; then
    echo "[abl-eval] cannot locate torch/main_eval.sbatch; submit from the repo root" \
         "or set IMMAP_ROOT" >&2
    exit 1
fi

source ~/.bashrc
conda activate gcdl
cd "${IMMAP_ROOT}"
: "${SLURM_ARRAY_TASK_ID:?must be run as an array job}"

# ---- score it: main_eval.sbatch does the rest (RUNS / OUT / VAL_KSPACE from here) ----------
RUNS="${ABL_RUNS}"
source "${IMMAP_ROOT}/torch/main_eval.sbatch"

# ---- the table, from everything merged so far (same lock as the merge) -------------------
flock "${OUT}/.combine.lock" python figures/table_ablation.py \
    --in "${OUT}/volumes.csv" --out "${OUT}/ablation_table"
echo "[abl-eval] table: ${OUT}/ablation_table.{tex,md,csv}"
