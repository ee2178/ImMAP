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
#SBATCH --job-name=BridgeStart-knee
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%A_%a.out
#SBATCH --error=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%A_%a.err
#SBATCH --array=0-2           # MUST equal the number of STARTS; checked at runtime
#
# BRIDGE STARTING POINTS for i2sb-TGVN on knee R8 (PDFS target, fully-sampled PD guide), at ONE
# TGVN depth -- the one the R8 sweep (torch/guided_recon_knee.sbatch) settles on. Same data,
# scaling, maps and training budget as that sweep; only the bridge's prior endpoint x1 differs:
#
#     0 sbtgvn_c<N>_guide_R8_g1     x1 = the guide (PD) image: contrast translation
#     1 sbtgvn_c<N>_adjoint_R8_g1   x1 = zero-filled E^H y (aliased)
#     2 sbtgvn_c<N>_acs_R8_g1       x1 = the mask's centre lines only (alias-free low-pass)
#
# TAU PER START. The gap x0 - x1 is a different thing for each start (contrast, aliasing, blur),
# so each start has its own calibration, config/knee/guided/bridge_tau/<start>_R8_volume.json
# (keyed by the normalization too: per-volume normalization uses an ABSOLUTE tau). If it is
# missing, this task writes it first --
#     python scripts/calibrate_recon_bridge.py <config> --slices 60 --json <file>
# (minutes on the GPU; the gap rule does not depend on the config's own tau) -- and regenerates
# the config from it: tau = TAU_MARGIN_ABS x the absolute gap rule RMS(x0 - x1)/2 (the margin is
# the user's call on the zero-filled start, 0.03 / 0.0234). Commit the files afterwards; delete
# one to recalibrate.
# Validation runs the sampler with and without CG data consistency (val/*, val/nodc_*).
#
# Submit:          CASCADES=8 sbatch --export=ALL torch/bridge_start_knee.sbatch
# One start:       CASCADES=8 sbatch --array=0 --export=ALL torch/bridge_start_knee.sbatch    (guide)
# Fewer starts:    CASCADES=8 STARTS="guide acs" sbatch --array=0-1 --export=ALL torch/bridge_start_knee.sbatch
# Noise:           default U[0, 0.001] (vlow, since 2026-11-13); NOISE=low|grid|none ... sbatch --export=ALL ...
#                  for U[0, 0.01] / U[0.01, 0.02] / none;
#                  runs go to trained_nets/guided_recon/knee/s3_noise-<preset>/, grid to s3/)
# Calibrate only:  CASCADES=8 CALIBRATE_ONLY=1 sbatch --time=01:00:00 --export=ALL torch/bridge_start_knee.sbatch
# Other cells:     any i2sb_recon cell of make_guided_recon_configs.py by name (CASCADES not
#                  needed): MODELS="..." sbatch --array=... --export=ALL torch/bridge_start_knee.sbatch
# CLUSTER SPLIT (2026-11-11): i2sb training now runs on BIGPURPLE -- the starts sweep
#                  (bigpurple/guided_i2sb_knee.sbatch) and the UNet baselines incl. I2SB-Inversion
#                  (bigpurple/baseline_sweep.sbatch). Torch keeps the end-to-end recon nets
#                  (torch/guided_recon_knee.sbatch, torch/contrast_transfer_knee.sbatch).

IMMAP_ROOT="${IMMAP_ROOT:-${HOME}/scratch/ee2178/ImMAP}"
ACCELS="${ACCELS:-8}"
STARTS="${STARTS:-guide adjoint acs}"
CAL_SLICES="${CAL_SLICES:-60}"
CALIBRATE_ONLY="${CALIBRATE_ONLY:-}"
FORCE_RESTART="${FORCE_RESTART:-}"
: "${MODELS:=}"
[ -n "${MODELS}" ] || : "${CASCADES:?set CASCADES to the settled TGVN depth (12, 10, 8 or 6), e.g. CASCADES=8 sbatch --export=ALL ...}"

source ~/.bashrc
conda activate gcdl
cd "${IMMAP_ROOT}"
set -eo pipefail
: "${SLURM_ARRAY_TASK_ID:?must be run as an array job (sbatch sets SLURM_ARRAY_TASK_ID)}"
mkdir -p torch/logs

if [ -z "${MODELS}" ]; then
    for s in ${STARTS}; do MODELS="${MODELS} sbtgvn_c${CASCADES}_${s}"; done
fi
GEN=(python scripts/make_guided_recon_configs.py --accels ${ACCELS} --models ${MODELS}
     --guide-accels 1)
if [ -n "${NOISE:-}" ]; then
    GEN+=(--noise "${NOISE}")   # added noise preset: vlow (default) | low | grid | none; own run tag
fi
CELLS="$("${GEN[@]}" --list-cells)"
N_TOTAL="$(echo "${CELLS}" | wc -l)"
if [ "${SLURM_ARRAY_TASK_ID}" -ge "${N_TOTAL}" ]; then
    echo "[bridge] task ${SLURM_ARRAY_TASK_ID} >= ${N_TOTAL} cells -- fix --array (0-$((N_TOTAL - 1)))" >&2
    exit 1
fi

CELL="$(echo "${CELLS}" | awk -v i="${SLURM_ARRAY_TASK_ID}" '$1 == i')"
NAME="$(echo "${CELL}" | cut -f5)"
TAU_FILE="$(echo "${CELL}" | cut -f6)"
BASE_CONFIG="config/knee/guided/${NAME}.json"
echo "[bridge] task ${SLURM_ARRAY_TASK_ID}/$((N_TOTAL - 1)): knee ${NAME} (tau file ${TAU_FILE})"

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
        echo "[bridge] writing ${BASE_CONFIG} failed (attempt ${try}/6); retrying in $((30 * try)) s" >&2
        sleep $((30 * try))
    done
    echo "[bridge] could not write ${BASE_CONFIG} after 6 attempts (filesystem full?)" >&2
    exit 1
}

regenerate
python - "${BASE_CONFIG}" <<'PY'
import json, os, sys
d = json.load(open(sys.argv[1]))["data"]["train"]
root = d.get("smap_root")
if d.get("target") == "sense" and (not root or not os.path.isdir(root) or not os.listdir(root)):
    sys.exit(f"[bridge] offline maps missing or empty: {root} -- run torch/espirit_knee_pairs.sbatch")
vs = d.get("volume_scales")
if vs and not os.path.isfile(vs):
    sys.exit(f"[bridge] per-volume scales missing: {vs} -- run torch/volume_scales_knee.sbatch")
PY

if [ ! -f "${TAU_FILE}" ]; then
    echo "[bridge] no calibration for this start yet: measuring it (${CAL_SLICES} train slices)"
    python scripts/calibrate_recon_bridge.py "${BASE_CONFIG}" --slices "${CAL_SLICES}" \
        --json "${TAU_FILE}"
    regenerate
fi
python -c "import json,sys; i=json.load(open(sys.argv[1]))['i2sb']; print(f\"[bridge] x1={i['x1_source']} tau={i['tau']} tau_ref={i['tau_ref']}\")" "${BASE_CONFIG}"
if [ "${CALIBRATE_ONLY}" = "1" ]; then
    echo "[bridge] CALIBRATE_ONLY: ${TAU_FILE} is in place; not training"
    exit 0
fi

RUN_DIR="$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['paths']['save_dir'])" "${BASE_CONFIG}")"
if [ "${FORCE_RESTART}" = "1" ] && [ -d "${RUN_DIR}" ]; then
    OLD="${RUN_DIR}.old_$(date +%Y%m%d_%H%M%S)"
    mv "${RUN_DIR}" "${OLD}"
    echo "[bridge] FORCE_RESTART: moved the previous run to ${OLD}"
fi

if [ -f "${RUN_DIR}/config.json" ] && [ -f "${RUN_DIR}/net.ckpt" ]; then
    CFG="${RUN_DIR}/config.json"
    echo "[bridge] resuming from ${CFG}"
else
    CFG="${BASE_CONFIG}"
    echo "[bridge] fresh run from ${CFG} -> ${RUN_DIR}"
fi

python train.py "${CFG}"
