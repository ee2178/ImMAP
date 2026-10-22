#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=torch_pr_89_tandon_advanced
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --job-name=NoiseSweep
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%A_%a.out
#SBATCH --error=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%A_%a.err
#SBATCH --array=0-7
#
# NOISE-LEVEL GENERALIZATION (the CDLNet / LPDSNet experiment): score each
# trained net at test noise levels inside AND outside its training range.
#
# Every net here trained at sigma ~ U[0.01, 0.02] (both anatomies). The unrolled
# nets are noise-ADAPTIVE: every threshold and step is affine in sigma
# (`degrees=1`, models/prox.py::Polynomial). E2E-VarNet never sees sigma. RAM
# is conditioned on it. Each net is told the TRUE test sigma (oracle sigma).
#
#    0- 3  brain R8   varnetmaps lpdsnet mg81v6 mg169v6
#    4- 7  knee  R8   varnetmaps lpdsnet mg81v6 mg169v6
#    RAM, or another R, through RUNS (one task per entry, in order):
#      RUNS="brain/ramdrunet_R8 knee/ramdrunet_R8" sbatch --array=0-1 torch/noise_sweep.sbatch
#
# THE MEASUREMENT is SIMULATED k-space by default: the SENSE ground truth through
# the stored maps, the mask and AWGN at exactly sigma. Measured k-space already
# carries the scanner's own noise, so with it the true noise level is unknown and
# cannot go BELOW the scanner's -- a sweep needs a known sigma. The nets trained
# on measured k-space + AWGN, so simulated test data is a (small, fixed) domain
# shift, the same at every sigma. VAL_KSPACE=measurement_awgn for the other view.
#
# NOISE IS PAIRED ACROSS SIGMA: eval_per_volume seeds each slice's noise by
# (seed, volume, slice), so every sigma scales the same draw, and every net sees
# the same noisy measurement at a given sigma.
#
# VOLUMES. Brain: 100 drawn at random from the 815 val volumes (--sample-volumes,
# fixed seed, the same 100 for every net and sigma -- the first 100 sorted would
# all be one site). Knee: all 100. Slices: each run's training range. At 8 sigmas
# that is about one full main-table evaluation per net.
#
# OUTPUT, one tree per sigma, merged as each task finishes:
#     results/noise_sweep_<VAL_KSPACE>/s<sigma>/{<anatomy>/<run>.slices.csv, volumes.csv}
# then the figure (copy the tree down first):
#     python figures/plot_noise_sweep.py --root results/noise_sweep_simulated
#
# RESUMABLE per (net, sigma): resubmitting finishes what is missing.
#
# SUBMIT FROM THE REPO ROOT.
#   sbatch torch/noise_sweep.sbatch
#   SIGMAS="0.005 0.05" sbatch --array=2 torch/noise_sweep.sbatch     # a probe

_RUNS_GIVEN="${RUNS:-}"                # RUNS="<anatomy>/<run> ..." replaces the list
RUNS=(
    brain/varnetmaps_R8 brain/lpdsnet_R8 brain/mg81v6_R8 brain/mg169v6_R8
    knee/varnetmaps_R8  knee/lpdsnet_R8  knee/mg81v6_R8  knee/mg169v6_R8
)
if [ -n "${_RUNS_GIVEN}" ]; then
    read -r -a RUNS <<< "${_RUNS_GIVEN}"
fi
# Training range is [0.01, 0.02]; 0.01 / 0.015 / 0.02 sit inside it.
SIGMAS="${SIGMAS:-0.0025 0.005 0.01 0.015 0.02 0.03 0.04 0.05}"
VAL_KSPACE="${VAL_KSPACE:-simulated}"
METRICS="${METRICS:-psnr ssim nrmse}"   # + lpips if wanted (weights cached on a login node)
SEED="${SEED:-1234}"
SAMPLE_BRAIN="${SAMPLE_BRAIN:-100}"    # volumes drawn per brain run; 0 = all 815
SAMPLE_KNEE="${SAMPLE_KNEE:-0}"        # 0 = all 100
SAMPLE_SEED="${SAMPLE_SEED:-0}"
WORKERS="${WORKERS:-6}"
OUT_ROOT="${OUT_ROOT:-results/noise_sweep_${VAL_KSPACE}}"

for _c in "${IMMAP_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" "$HOME/scratch/ee2178/ImMAP"; do
    if [ -n "${_c}" ] && [ -f "${_c}/scripts/eval_per_volume.py" ]; then
        IMMAP_ROOT="${_c}"
        break
    fi
done
if [ ! -f "${IMMAP_ROOT:-}/scripts/eval_per_volume.py" ]; then
    echo "[sweep] cannot locate scripts/eval_per_volume.py; submit from the repo root" \
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
    echo "[sweep] task ${SLURM_ARRAY_TASK_ID} >= ${#RUNS[@]} runs -- fix --array" >&2
    exit 1
fi
ENTRY="${RUNS[${SLURM_ARRAY_TASK_ID}]}"
ANATOMY="${ENTRY%%/*}"
RUN="${ENTRY#*/}"
RUNS_ROOT="trained_nets/mg_recon"
if [ ! -f "${RUNS_ROOT}/${ANATOMY}/${RUN}/config.json" ] \
        || [ ! -f "${RUNS_ROOT}/${ANATOMY}/${RUN}/net.ckpt" ]; then
    echo "[sweep] ${RUNS_ROOT}/${ANATOMY}/${RUN} has no config.json + net.ckpt" >&2
    exit 1
fi
if [ "${ANATOMY}" = "brain" ]; then _n="${SAMPLE_BRAIN}"; else _n="${SAMPLE_KNEE}"; fi

echo "[sweep] ${ENTRY}  sigmas='${SIGMAS}'  val_kspace=${VAL_KSPACE}  sample=${_n:-all}" \
     "-> ${OUT_ROOT}/s<sigma>"
echo "[sweep] $(date '+%F %T')  node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

for S in ${SIGMAS}; do
    OUT="${OUT_ROOT}/s${S}"
    echo "[sweep] ${ENTRY} sigma=${S} $(date '+%F %T')"
    python scripts/eval_per_volume.py \
        --anatomy "${ANATOMY}" --run "${RUN}" \
        --metrics ${METRICS} \
        --seed "${SEED}" \
        --sigma "${S}" \
        --val-kspace-type "${VAL_KSPACE}" \
        --workers "${WORKERS}" \
        --out "${OUT}" \
        $( [ "${_n}" != "0" ] && echo "--sample-volumes ${_n} --sample-seed ${SAMPLE_SEED}" )
    # Merge every run written to this sigma's tree so far (tasks can finish
    # together, hence the lock; the merge writes through a temporary file).
    mkdir -p "${OUT}"
    flock "${OUT}/.combine.lock" python scripts/eval_per_volume.py --out "${OUT}" --combine
done
echo "[sweep] ${ENTRY} done $(date '+%F %T')"
