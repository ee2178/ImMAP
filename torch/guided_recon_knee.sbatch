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
#SBATCH --job-name=GuidedRecon-knee
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%A_%a.out
#SBATCH --error=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%A_%a.err
#SBATCH --array=0-2           # MUST equal the cell count; checked at runtime
#
# GUIDED RECONSTRUCTION, fastMRI knee: PDFS main contrast, FULLY-SAMPLED PD guide (set 1 of
# notes/guided_recon_tgvn.md). Task "recon" with the paired loader `fastmri_paired`.
#
#     python scripts/make_guided_recon_configs.py --list-cells
#     default: R8 {varnet_img, tgvn, lggs}  -> 0-2   (ACCELS="8 16" --array=0-5 adds R16)
#
#     varnet_img  TGVN's matched image-domain E2E-VarNet, no side information
#     tgvn        TGVN (vendored upstream, models/tgvn/), 12 cascades x 18 chans
#     lggs        LGGSNet at the published LGGS point, flex attention
#
# SAME SETTINGS AS THE ONE-SHOT RECON GRID, NOT TGVN'S RECIPE: the configs are regenerated on every
# launch from scripts/make_mg_recon_configs.py's knee cells (measured k-space + AWGN, online
# ESPIRiT, center_frac 0.04 uniform mask, sigma ~ U[0.01, 0.02], magnitude NL1+NL2, Adam + cosine
# over 300k steps, clip 0.05). Splits are TGVN's patient-level PD/PDFS pairs
# (config/knee/guided/splits/), drawn from fastMRI multicoil train AND val; the target is the
# RSS of the PDFS k-space.
#
# RESUME: a requeued cell continues from its run dir's config.json + net.ckpt; REGENERATE never
# touches a started run. FORCE_RESTART=1 moves the old run aside and starts over.
#
# Submit:          sbatch torch/guided_recon_knee.sbatch
# One cell:        sbatch --array=1 torch/guided_recon_knee.sbatch          (tgvn R8)
# Only TGVN pair:  MODELS="varnet_img tgvn" sbatch --array=0-1 --export=ALL torch/guided_recon_knee.sbatch
# Smoke test:      sbatch --array=1 --time=00:30:00 torch/guided_recon_knee.sbatch
#                  (a wrong data root or a PD/PDFS file missing from both roots fails at loader
#                  construction, within a minute)

IMMAP_ROOT="${IMMAP_ROOT:-${HOME}/scratch/ee2178/ImMAP}"
ACCELS="${ACCELS:-8}"
MODELS="${MODELS:-varnet_img tgvn lggs}"
REGENERATE="${REGENERATE:-1}"
FORCE_RESTART="${FORCE_RESTART:-}"

source ~/.bashrc
conda activate gcdl
cd "${IMMAP_ROOT}"
set -eo pipefail
: "${SLURM_ARRAY_TASK_ID:?must be run as an array job (sbatch sets SLURM_ARRAY_TASK_ID)}"
mkdir -p torch/logs

GEN=(python scripts/make_guided_recon_configs.py --accels ${ACCELS} --models ${MODELS})
CELLS="$("${GEN[@]}" --list-cells)"
N_TOTAL="$(echo "${CELLS}" | wc -l)"
if [ "${SLURM_ARRAY_TASK_ID}" -ge "${N_TOTAL}" ]; then
    echo "[guided] task ${SLURM_ARRAY_TASK_ID} >= ${N_TOTAL} cells -- fix --array (0-$((N_TOTAL - 1)))" >&2
    exit 1
fi
if [ -n "${SLURM_ARRAY_TASK_MAX:-}" ] && [ "${SLURM_ARRAY_TASK_MAX}" -lt "$((N_TOTAL - 1))" ] \
        && [ "${SLURM_ARRAY_TASK_ID}" -eq "${SLURM_ARRAY_TASK_MAX}" ]; then
    echo "[guided] NOTE: --array stops at ${SLURM_ARRAY_TASK_MAX}; cells" \
         "$((SLURM_ARRAY_TASK_MAX + 1))..$((N_TOTAL - 1)) are NOT being run." >&2
fi

CELL="$(echo "${CELLS}" | awk -v i="${SLURM_ARRAY_TASK_ID}" '$1 == i')"
RTAG="$(echo "${CELL}" | cut -f2)"
MODEL="$(echo "${CELL}" | cut -f3)"
BASE_CONFIG="config/knee/guided/${MODEL}_${RTAG}.json"
echo "[guided] task ${SLURM_ARRAY_TASK_ID}/$((N_TOTAL - 1)): knee ${RTAG} ${MODEL}"

if [ "${REGENERATE}" = "1" ]; then
    "${GEN[@]}"
fi
[ -f "${BASE_CONFIG}" ] || { echo "[guided] missing ${BASE_CONFIG}" >&2; exit 1; }

RUN_DIR="$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['paths']['save_dir'])" "${BASE_CONFIG}")"
if [ "${FORCE_RESTART}" = "1" ] && [ -d "${RUN_DIR}" ]; then
    OLD="${RUN_DIR}.old_$(date +%Y%m%d_%H%M%S)"
    mv "${RUN_DIR}" "${OLD}"
    echo "[guided] FORCE_RESTART: moved the previous run to ${OLD}"
fi

if [ -f "${RUN_DIR}/config.json" ] && [ -f "${RUN_DIR}/net.ckpt" ]; then
    CFG="${RUN_DIR}/config.json"
    echo "[guided] resuming from ${CFG}"
else
    CFG="${BASE_CONFIG}"
    echo "[guided] fresh run from ${CFG} -> ${RUN_DIR}"
fi

python train.py "${CFG}"
