#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=torch_pr_89_tandon_advanced
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=01:00:00
#SBATCH --job-name=LGGS-materialize-bench
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%j.out
#SBATCH --error=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%j.err
#
# LGGS's triton guide branch, implicit (TritonAdjacency: the attention re-run on every apply) vs
# materialised (WindowAdjacency: softmax weights built once per refresh) --
# scripts/bench_lggs_materialize.py, on the L40S the knee runs train on.
#
#   sbatch torch/bench_lggs_materialize.sbatch
#
# Stage 1 checks every new kernel against the torch reference and stops WITHOUT timing if one
# disagrees -- a failure there is a kernel bug, so send the log, not the timings. Results:
# torch/logs/lggs_materialize_<jobid>.json (and the log).
#
# Knobs (environment variables; defaults in brackets):
#
#   STAGES  ["kernels branch net"]
#   BATCH   [1]    GUIDES [1]    K [config's]    ITERS [10]    WARMUP [3]
#   EXTRA   [unset]   passed through verbatim, e.g. "--H 320 --W 160"
#
#   STAGES=kernels sbatch --time=00:15:00 torch/bench_lggs_materialize.sbatch   # kernels only
#   GUIDES=3 sbatch --export=ALL torch/bench_lggs_materialize.sbatch

IMMAP_ROOT="${IMMAP_ROOT:-${HOME}/scratch/ee2178/ImMAP}"
STAGES="${STAGES:-kernels branch net}"

source ~/.bashrc
conda activate gcdl
cd "${IMMAP_ROOT}"
set -eo pipefail
mkdir -p torch/logs

echo "[bench] host $(hostname), python $(which python)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "import torch, triton; print(f'[bench] torch {torch.__version__}, triton {triton.__version__}, cuda {torch.version.cuda}')"

ARGS=(--stages ${STAGES} --batch "${BATCH:-1}" --guides "${GUIDES:-1}"
      --iters "${ITERS:-10}" --warmup "${WARMUP:-3}"
      --json "torch/logs/lggs_materialize_${SLURM_JOB_ID:-local}.json")
[ -n "${K:-}" ]     && ARGS+=(--K "${K}")
[ -n "${EXTRA:-}" ] && ARGS+=(${EXTRA})

python scripts/bench_lggs_materialize.py "${ARGS[@]}"
