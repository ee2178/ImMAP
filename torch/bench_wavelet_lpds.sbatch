#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=torch_pr_89_tandon_advanced
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=00:45:00
#SBATCH --job-name=WLPDSBench
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#
# Wavelet-LPDS K / K^H: the original Gauss-trick path vs the interleaved one
# (one real conv per level), with and without torch.compile, for both carry
# modes (dense conv carries = the original init, fixed pixel-unshuffle).
#
# THE NUMBER TO READ FIRST: `conv / interleaved` vs `conv / gauss` in the
# operator-only table. Interleaved does 4 real multiplies per complex MAC to
# Gauss's 3 but launches 1 conv per level instead of 3. If it is not faster
# here, T3's dense conv is compute-bound on this GPU and the hybrid (Gauss at
# T3 only) is worth building.
#
# No training, no checkpoints written. Synthetic SENSE data -- shapes only.
#
# Submit (from the repo root):   sbatch torch/bench_wavelet_lpds.sbatch
# Options:  IMG="320 320" COILS=16 BATCH=1 K=30 REPS=20 \
#           CONFIG=config/brain/mg/wlpds16_R16.json \
#           EXTRA="--compile-mode max-autotune-no-cudagraphs" \
#           sbatch torch/bench_wavelet_lpds.sbatch

IMG="${IMG:-320 320}"
COILS="${COILS:-16}"
BATCH="${BATCH:-1}"
K="${K:-30}"
REPS="${REPS:-20}"
CONFIG="${CONFIG:-}"
EXTRA="${EXTRA:-}"

source ~/.bashrc
conda activate gcdl

for _c in "${IMMAP_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" "$HOME/scratch/ee2178/ImMAP"; do
    if [ -n "${_c}" ] && [ -f "${_c}/train.py" ]; then IMMAP_ROOT="${_c}"; break; fi
done
[ -f "${IMMAP_ROOT:-}/train.py" ] || { echo "[bench] cannot locate ImMAP." >&2; exit 1; }
cd "${IMMAP_ROOT}"
set -eo pipefail
mkdir -p logs

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# correctness first: a wrong operator makes every timing below meaningless
python -m tests.test_wavelet_lpds

python scripts/bench_wavelet_lpds.py --size ${IMG} --coils "${COILS}" \
    --batch "${BATCH}" --K "${K}" --reps "${REPS}" \
    ${CONFIG:+--config "${CONFIG}"} ${EXTRA}
