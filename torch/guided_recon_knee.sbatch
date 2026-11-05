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
#SBATCH --array=0-4           # MUST equal the cell count; checked at runtime
#
# GUIDED RECONSTRUCTION, fastMRI knee: PDFS main contrast, PD guide (set 1 of
# notes/guided_recon_tgvn.md). A cell is (main R, model, guide R):
#
#     python scripts/make_guided_recon_configs.py --list-cells
#     0 tgvn_R8_g1     TGVN, fully-sampled guide                 task recon
#     1 tgvn_R8_g2     TGVN, 2x guide                            task recon
#     2 sbtgvn_R8_g1   i2sb-TGVN (BridgeTGVN), fully-sampled     task i2sb_recon
#     3 sbtgvn_R8_g2   i2sb-TGVN, 2x guide                       task i2sb_recon
#     4 lpdsnet_R8     LPDSNet, no guide                         task recon
#
# SAME SETTINGS AS THE ONE-SHOT RECON GRID, NOT TGVN'S RECIPE: the configs are regenerated on every
# launch from scripts/make_mg_recon_configs.py's knee cells (measured k-space + AWGN, online
# ESPIRiT, center_frac 0.04 uniform mask, sigma ~ U[0.01, 0.02], Adam + cosine over 300k steps,
# clip 0.05; magnitude NL1+NL2, except complex-MSE for the complex bridge). Splits are TGVN's
# patient-level PD/PDFS pairs (config/knee/guided/splits/); the target is the RSS of the PDFS
# k-space. The guide is undersampled like TGVN's side scans: equispaced, no centre lines, a fresh
# line phase per training step (offset 0 at validation).
#
# BEFORE THE sbtgvn CELLS: their tau (make_guided_recon_configs.py TAU) is an unmeasured 0.1.
#     python scripts/calibrate_recon_bridge.py config/knee/guided/sbtgvn_R8_g1.json --slices 60
# (on a GPU node: online ESPIRiT per slice), then set TAU and resubmit.
#
# RESUME: a requeued cell continues from its run dir's config.json + net.ckpt; REGENERATE never
# touches a started run. FORCE_RESTART=1 moves the old run aside and starts over.
#
# Submit:          sbatch torch/guided_recon_knee.sbatch
# One cell:        sbatch --array=0 torch/guided_recon_knee.sbatch          (tgvn_R8_g1)
# Guide sweep:     GUIDE_ACCELS="1 2 4" MODELS="tgvn" sbatch --array=0-2 --export=ALL torch/guided_recon_knee.sbatch
# Smoke test:      sbatch --array=0 --time=00:30:00 torch/guided_recon_knee.sbatch
#                  (a wrong data root or a PD/PDFS file missing from both roots fails at loader
#                  construction, within a minute)

IMMAP_ROOT="${IMMAP_ROOT:-${HOME}/scratch/ee2178/ImMAP}"
ACCELS="${ACCELS:-8}"
MODELS="${MODELS:-tgvn sbtgvn lpdsnet}"
GUIDE_ACCELS="${GUIDE_ACCELS:-1 2}"
REGENERATE="${REGENERATE:-1}"
FORCE_RESTART="${FORCE_RESTART:-}"

source ~/.bashrc
conda activate gcdl
cd "${IMMAP_ROOT}"
set -eo pipefail
: "${SLURM_ARRAY_TASK_ID:?must be run as an array job (sbatch sets SLURM_ARRAY_TASK_ID)}"
mkdir -p torch/logs

GEN=(python scripts/make_guided_recon_configs.py --accels ${ACCELS} --models ${MODELS}
     --guide-accels ${GUIDE_ACCELS})
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
NAME="$(echo "${CELL}" | cut -f5)"
BASE_CONFIG="config/knee/guided/${NAME}.json"
echo "[guided] task ${SLURM_ARRAY_TASK_ID}/$((N_TOTAL - 1)): knee ${NAME}"

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
