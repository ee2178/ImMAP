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
# THE KNEE R8 SWEEP: PDFS main contrast, PD guide (fully sampled). One training budget for every
# cell. A cell is (main R, model, guide R):
#
#     python scripts/make_guided_recon_configs.py --list-cells
#     0 lpdsnet_R8        LPDSNet, NO guide (unguided baseline)          task recon
#     1 lggs_R8_g1        vanilla LGGS, Sljiva's grid point (triton)     task recon
#     2 tgvn_c6_R8_g1     TGVN, 6 cascades (E2E-VarNet width)            task recon
#
# 2026-11-13 (user's call): vanilla LGGS only (the guide-W / guide-conv / transfer variants showed
# no visual difference) and TGVN at 6 cascades only (deeper TGVN is noticeably slower than LGGS);
# tgvn_c12 / c10 / c8 still run by name. Added training noise U[0, 0.001] (NOISE=vlow, 10x below
# the earlier "low") to cut the artifacts seen across the board; its own run tag, s3_noise-vlow,
# so these are fresh runs, never resumes of the U[0, 0.01] ones in s3_noise-low.
#
# Runs land in trained_nets/guided_recon/knee/s3/<cell> (RUN_TAG "s3" in the generator): earlier
# sets' run dirs (s1 shared 5e3 scale + eig-0.95 maps; s2 fixed per-contrast scale) are NOT resumed. Bridge starting points for i2sb-TGVN
# have their own launcher, torch/bridge_start_knee.sbatch, once the TGVN depth is settled.
#
# SAME SETTINGS AS THE ONE-SHOT RECON GRID, NOT TGVN'S RECIPE: the configs are regenerated on every
# launch from scripts/make_mg_recon_configs.py's knee cells (measured k-space + AWGN, online
# ESPIRiT, center_frac 0.04 uniform mask, sigma ~ U[0.01, 0.02], Adam + cosine over 300k steps,
# clip 0.05; magnitude NL1+NL2). LR: the grid's (LGGS: its group-attention LR) -- TRAIN_RECIPE in
# the generator switches every cell to Sljiva's (4e-4 -> 2e-6, clip 1) at once. Splits are TGVN's
# patient-level PD/PDFS pairs (config/knee/guided/splits/). Target: the offline SENSE image of the
# PDFS k-space (ESPIRiT, ACS 32, thresh_eig 0.9 -- torch/espirit_knee_pairs.sbatch MUST have
# finished; a cell refuses to start without the maps). Intensity: PER-VOLUME normalization --
# each volume divided by its own foreground p95, the target by its ACS-only image's, the guide by
# its full image's (config/knee/guided/volume_scales.json from torch/volume_scales_knee.sbatch,
# which MUST have finished too). Val PSNR/SSIM against each slice's own peak.
#
# RESUME: a requeued cell continues from its run dir's config.json + net.ckpt; REGENERATE never
# touches a started run. FORCE_RESTART=1 moves the old run aside and starts over.
#
# Order:           MAPS=$(sbatch --parsable torch/espirit_knee_pairs.sbatch)
#                  VOLS=$(sbatch --parsable torch/volume_scales_knee.sbatch)
#                  sbatch --dependency=afterok:${MAPS}:${VOLS} torch/guided_recon_knee.sbatch
# One cell:        sbatch --array=1 torch/guided_recon_knee.sbatch          (lggs_R8_g1)
# Other cells:     MODELS="tgvn_c8" GUIDE_ACCELS="1 2 4" sbatch --array=0-2 --export=ALL torch/guided_recon_knee.sbatch
# Smoke test:      sbatch --array=0 --time=00:30:00 torch/guided_recon_knee.sbatch
# Noise:           default U[0, 0.001] (vlow); NOISE=low|grid|none ... sbatch --export=ALL ... for
#                  U[0, 0.01] / U[0.01, 0.02] / none;
#                  runs go to trained_nets/guided_recon/knee/s3_noise-<preset>/, grid to s3/)
# Timing:          python scripts/time_guided_nets.py        (fwd+bwd per step of these cells)
# Multicontrast LGGS, no extra network: the knee LGGS cell with the guide's own W_phi, W_alpha
#                  (lggs_guide_w) or its own analysis conv per layer (lggs_guide_conv):
#                  MODELS="lggs_guide_w lggs_guide_conv" sbatch --array=0-1 --export=ALL torch/guided_recon_knee.sbatch
# LGGS + guide transfer (after torch/contrast_transfer_knee.sbatch): the knee LGGS cell with a
#                  pretrained contrast-transfer net frozen in front of its guide, lggs_<ct cell>:
#                  MODELS="lggs_ct_unet8" sbatch --array=0 --export=ALL torch/guided_recon_knee.sbatch
# CLUSTER SPLIT:   end-to-end recon (LGGS / TGVN / LPDSNet, contrast transfer) runs HERE on torch;
#                  every i2sb cell runs on bigpurple (bigpurple/guided_i2sb_knee.sbatch,
#                  bigpurple/baseline_sweep.sbatch).

IMMAP_ROOT="${IMMAP_ROOT:-${HOME}/scratch/ee2178/ImMAP}"
ACCELS="${ACCELS:-8}"
MODELS="${MODELS:-lpdsnet lggs tgvn_c6}"
NOISE="${NOISE:-vlow}"          # added training noise U[0, 0.001]; see the header
GUIDE_ACCELS="${GUIDE_ACCELS:-1}"
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
if [ -n "${NOISE:-}" ]; then
    GEN+=(--noise "${NOISE}")   # added noise preset: vlow (default here) | low | grid | none; own run tag
fi
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

regenerate() {
    # Stage the generated config in a private dir NEXT TO its destination (same filesystem), so
    # putting it in place is an atomic rename -- never a cross-filesystem copy that can leave a
    # half-written config -- and parallel array tasks never read a config another is writing.
    # A full filesystem (bigpurple's shared scratch, 2026-11-10: ENOSPC on a 4 KB config) is
    # retried with backoff before giving up.
    local tmp try
    mkdir -p config/knee/guided
    for try in 1 2 3 4 5 6; do
        tmp="$(mktemp -d -p config/knee/guided .gen.XXXXXX)" \
            && "${GEN[@]}" --out "${tmp}" > "${tmp}/gen.log" \
            && mv -f "${tmp}/knee/guided/${NAME}.json" "${BASE_CONFIG}" \
            && { grep -F "${NAME}" "${tmp}/gen.log" || true; rm -rf "${tmp}"; return 0; }
        [ -n "${tmp}" ] && [ -f "${tmp}/gen.log" ] && tail -3 "${tmp}/gen.log" >&2
        [ -n "${tmp}" ] && rm -rf "${tmp}"
        echo "[guided] writing ${BASE_CONFIG} failed (attempt ${try}/6); retrying in $((30 * try)) s" >&2
        sleep $((30 * try))
    done
    echo "[guided] could not write ${BASE_CONFIG} after 6 attempts (filesystem full?)" >&2
    exit 1
}
if [ "${REGENERATE}" = "1" ]; then
    regenerate
fi
[ -f "${BASE_CONFIG}" ] || { echo "[guided] missing ${BASE_CONFIG}" >&2; exit 1; }

# the offline target maps must exist (torch/espirit_knee_pairs.sbatch); fail now, not at step 0
python - "${BASE_CONFIG}" <<'PY'
import json, os, sys
cfg = json.load(open(sys.argv[1]))
d = cfg["data"]["train"]
gt = (cfg["model"].get("params") or {}).get("guide_transfer")
if gt and not os.path.isfile(gt["ckpt"]):
    sys.exit(f"[guided] the frozen guide transfer is not trained yet: {gt['ckpt']}\n"
             f"         run torch/contrast_transfer_knee.sbatch first (same NOISE)")
root = d.get("smap_root")
if d.get("target") == "sense" and (not root or not os.path.isdir(root) or not os.listdir(root)):
    sys.exit(f"[guided] offline maps missing or empty: {root}\n"
             f"         run torch/espirit_knee_pairs.sbatch first (or submit with "
             f"--dependency=afterok:<its job id>)")
vs = d.get("volume_scales")
if vs and not os.path.isfile(vs):
    sys.exit(f"[guided] per-volume scales missing: {vs} -- run torch/volume_scales_knee.sbatch")
PY

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
