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
#SBATCH --job-name=I2SB-recon-Unet
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%A_%a.out
#SBATCH --error=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%A_%a.err
#SBATCH --array=0-1           # MUST equal the cell count; checked at runtime
#
# I2SB RECONSTRUCTION BASELINE (task "i2sb_recon", training/i2sb_recon.py): the I2SB paper's ADM
# UNet as the bridge regressor, x1 = E^H y (zero-filled) -> x0 = the fully-sampled image.
#
#     python scripts/make_i2sb_recon_configs.py --list-cells     # index, anatomy, R, model
#     default: 0 brain R8 unet, 1 knee R8 unet
#
# SAME MEASUREMENT AS THE ONE-SHOT GRID. The configs are REGENERATED on every launch from what
# scripts/make_mg_recon_configs.py writes today for that anatomy and R (its lpdsnet cell): data
# roots, slice range, mask (center_frac, adjust_accel), measured k-space + AWGN, online ESPIRiT,
# noise range, val noise and seed. So these runs share a table with trained_nets/mg_recon/*.
# The UNet never sees y or E: data consistency enters only through x1. That is what makes it the
# baseline the unrolled bridge regressors are measured against.
#
# The bridge target is the fully-sampled image under the OPERATOR's (online) maps, not the
# stored offline-map image: the two differ in phase on some slices, which a complex bridge must
# not be asked to learn. Metrics are on |.| against the stored image, as the grid's are.
#
# tau: brain 0.11 (measured on one local volume, measured protocol, R=8); knee 0.1 is an
# UNMEASURED placeholder. Both live in scripts/make_i2sb_recon_configs.py::TAU.
#
# val/psnr is the FULL nfe=20 reverse sampler on a fixed 100-slice subset; val/zf_psnr is the
# zero-filled x1 on the same slices. net_bestval.ckpt holds the best val/psnr (EMA weights).
#
# RESUME: a requeued or resubmitted cell continues from its run dir's config.json + net.ckpt
# (train.py stamps both); REGENERATE never touches a run that has started. FORCE_RESTART=1 starts
# the cell over in a fresh run dir (the old one is renamed, not deleted).
#
# Submit:          sbatch torch/i2sb_recon_unet.sbatch
# One cell:        sbatch --array=0 torch/i2sb_recon_unet.sbatch
# More R:          ACCELS="4 8 16" sbatch --array=0-5 --export=ALL torch/i2sb_recon_unet.sbatch
# Path smoke test: sbatch --array=0 --time=00:30:00 torch/i2sb_recon_unet.sbatch
#                  (fails within a minute at loader construction if a data root is wrong)

IMMAP_ROOT="${IMMAP_ROOT:-${HOME}/scratch/ee2178/ImMAP}"
ANATOMIES="${ANATOMIES:-brain knee}"
ACCELS="${ACCELS:-8}"
REGENERATE="${REGENERATE:-1}"
FORCE_RESTART="${FORCE_RESTART:-}"

source ~/.bashrc
conda activate gcdl
cd "${IMMAP_ROOT}"
set -eo pipefail
: "${SLURM_ARRAY_TASK_ID:?must be run as an array job (sbatch sets SLURM_ARRAY_TASK_ID)}"
mkdir -p torch/logs

GEN=(python scripts/make_i2sb_recon_configs.py --anatomy ${ANATOMIES} --accels ${ACCELS})
CELLS="$("${GEN[@]}" --list-cells)"
N_TOTAL="$(echo "${CELLS}" | wc -l)"
if [ "${SLURM_ARRAY_TASK_ID}" -ge "${N_TOTAL}" ]; then
    echo "[i2sb] task ${SLURM_ARRAY_TASK_ID} >= ${N_TOTAL} cells -- fix --array (0-$((N_TOTAL - 1)))" >&2
    exit 1
fi
if [ -n "${SLURM_ARRAY_TASK_MAX:-}" ] && [ "${SLURM_ARRAY_TASK_MAX}" -lt "$((N_TOTAL - 1))" ] \
        && [ "${SLURM_ARRAY_TASK_ID}" -eq "${SLURM_ARRAY_TASK_MAX}" ]; then
    echo "[i2sb] NOTE: --array stops at ${SLURM_ARRAY_TASK_MAX}; cells" \
         "$((SLURM_ARRAY_TASK_MAX + 1))..$((N_TOTAL - 1)) are NOT being run." >&2
fi

CELL="$(echo "${CELLS}" | awk -v i="${SLURM_ARRAY_TASK_ID}" '$1 == i')"
ANATOMY="$(echo "${CELL}" | cut -f2)"
RTAG="$(echo "${CELL}" | cut -f3)"
MODEL="$(echo "${CELL}" | cut -f4)"
BASE_CONFIG="config/${ANATOMY}/i2sb/${MODEL}_${RTAG}.json"
echo "[i2sb] task ${SLURM_ARRAY_TASK_ID}/$((N_TOTAL - 1)): ${ANATOMY} ${RTAG} ${MODEL}"

if [ "${REGENERATE}" = "1" ]; then
    "${GEN[@]}"
fi
[ -f "${BASE_CONFIG}" ] || { echo "[i2sb] missing ${BASE_CONFIG}" >&2; exit 1; }

RUN_DIR="$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['paths']['save_dir'])" "${BASE_CONFIG}")"
if [ "${FORCE_RESTART}" = "1" ] && [ -d "${RUN_DIR}" ]; then
    OLD="${RUN_DIR}.old_$(date +%Y%m%d_%H%M%S)"
    mv "${RUN_DIR}" "${OLD}"
    echo "[i2sb] FORCE_RESTART: moved the previous run to ${OLD}"
fi

if [ -f "${RUN_DIR}/config.json" ] && [ -f "${RUN_DIR}/net.ckpt" ]; then
    CFG="${RUN_DIR}/config.json"
    echo "[i2sb] resuming from ${CFG}"
else
    CFG="${BASE_CONFIG}"
    echo "[i2sb] fresh run from ${CFG} -> ${RUN_DIR}"
fi

python train.py "${CFG}"
