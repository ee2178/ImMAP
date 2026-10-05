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
#SBATCH --job-name=TimeNet
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%j.out
#SBATCH --error=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%j.err
#
# Two measurements with nothing of the training loop in them.
#
#   1. ONLINE ESPIRiT KERNEL SIZE on real validation slices
#      (scripts/check_online_espirit_ks.py): time, kernels kept, memory, and
#      agreement with the stored maps and between kernel sizes -- KS="2 4".
#
#   2. NETWORK SPEED (scripts/time_net.py): forward and backward only,
#      GPU-synchronised after warm-up -- Sljiva's `CUDA.@elapsed` around the net
#      call. Brain dimensions, synthetic data, each model's own config
#      (coarse_op, planar convs), plus the optimizer step and projection.
#
# No training, no checkpoints. Output: this job's log, plus time_net.json and
# the figure time_net.png under
#   trained_nets/profile/time_net_<jobid>/
# Redraw the figure later (dark theme, per-segment values, pdf) without a GPU:
#   python scripts/time_net.py --from-json <dir>/time_net.json \
#       --fig-out <dir>/time_net_dark.png --fig-theme dark --fig-values
#
# Submit (from the repo root):   sbatch torch/time_net.sbatch
# Options:
#   R=12 KS="2 4" N_SLICES=6      the ESPIRiT check
#   IMG="640 320" COILS=20 REPS=30
#   MODELS="lpdsnet varnetmaps mglpds mg169v6 mg121v4 mg81v2 mgunet"
#   SKIP_ESPIRIT=1 / SKIP_NETS=1  run only one half

R="${R:-12}"
KS="${KS:-2 4}"
N_SLICES="${N_SLICES:-6}"
IMG="${IMG:-640 320}"
COILS="${COILS:-20}"
REPS="${REPS:-30}"
MODELS="${MODELS:-lpdsnet varnetmaps mglpds mg169v2 mg169v4 mg169v6 mg121v2 mg121v4 mg121v6 mg81v2 mg81v4 mg81v6 mgunet}"

source ~/.bashrc
conda activate gcdl

for _c in "${IMMAP_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" "$HOME/scratch/ee2178/ImMAP"; do
    if [ -n "${_c}" ] && [ -f "${_c}/train.py" ]; then IMMAP_ROOT="${_c}"; break; fi
done
[ -f "${IMMAP_ROOT:-}/train.py" ] || { echo "[time] cannot locate ImMAP." >&2; exit 1; }
cd "${IMMAP_ROOT}"
set -eo pipefail

OUT="trained_nets/profile/time_net_${SLURM_JOB_ID:-local}"
mkdir -p "${OUT}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# The generator's current brain configs, written to this job's own directory so
# nothing under config/ (which the training launchers regenerate) is touched.
# --organ-mask matches the brain launchers; it only affects the data block.
python scripts/make_mg_recon_configs.py --out "${OUT}/gen" --anatomy brain --organ-mask > /dev/null
CFG="${OUT}/gen/brain/mg"

if [ -z "${SKIP_ESPIRIT:-}" ]; then
    echo; echo "================ 1. online ESPIRiT kernel size ================"
    python scripts/check_online_espirit_ks.py --config "${CFG}/lpdsnet_R${R}.json" \
        --ks ${KS} --n "${N_SLICES}" --R "${R}"
fi

if [ -z "${SKIP_NETS:-}" ]; then
    echo; echo "================ 2. network forward / backward ================"
    CFGS=()
    for m in ${MODELS}; do
        f="${CFG}/${m}_R${R}.json"
        [ -f "${f}" ] || { echo "[time] no config for '${m}' at R=${R} (${f})" >&2; exit 1; }
        CFGS+=("${f}")
    done
    python scripts/time_net.py --configs "${CFGS[@]}" --size ${IMG} --coils "${COILS}" \
        --reps "${REPS}" --step --json-out "${OUT}/time_net.json" \
        --fig-out "${OUT}/time_net.png"
fi
