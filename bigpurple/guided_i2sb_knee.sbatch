#!/bin/bash
#
# I2SB GUIDED RECONSTRUCTION, fastMRI knee, on BIGPURPLE (the standard guided recon runs stay on
# torch: torch/guided_recon_knee.sbatch). PDFS target, PD guide fully sampled, R = 8. Cells:
#
#     0 sbtgvn_c8_adjoint_R8_g1   i2sb-TGVN, 8 cascades (BridgeTGVN)
#     1 sblggs_adjoint_R8_g1      i2sb-LGGS, the knee LGGS cell as the bridge regressor
#                                 (models/guided_lpds.py::BridgeLGGS: the K sweeps start at x_t)
#
# Both bridge from the zero-filled start (x1 = E^H y); MODELS can name other starts
# (sbtgvn_c8_guide, sblggs_acs, ...). Everything else is the torch sweep's protocol, from
# scripts/make_guided_recon_configs.py: grid training settings (300k steps; LR 5e-4 TGVN,
# 2e-4 LGGS), offline SENSE target (thresh_eig 0.9), per-volume normalization, absolute tau,
# complex-MSE loss, validation with and without CG data consistency (val/*, val/nodc_*). Run
# dirs: trained_nets/guided_recon/knee/<RUN_TAG>/<cell>.
#
# BEFORE THE FIRST SUBMIT (once):
#     bash bigpurple/setup_knee_data.sh                      # data symlink + checks
#     git pull                                               # volume_scales.json from torch
#     MAPS=$(sbatch --parsable bigpurple/espirit_knee_pairs.sbatch)
#     sbatch --dependency=afterok:${MAPS} bigpurple/guided_i2sb_knee.sbatch
#
# TAU: per bridge start, config/knee/guided/bridge_tau/<start>_R8_volume.json. If missing, the
# task measures it first (scripts/calibrate_recon_bridge.py, 60 train slices, minutes) and
# regenerates its config; commit the file afterwards so torch and bigpurple agree.
#
# RESUME: a requeued cell continues from its run dir's config.json + net.ckpt.
# FORCE_RESTART=1 moves the old run aside and starts over.
#
# STARTING POINTS AT ONE TAU (the other contrast, the ACS lines, zero-filled; i2sb-TGVN, 8
# cascades): TAU replaces the per-start calibration with one value for every cell, and the run
# names get _tau<value>, so these never share a dir with calibrated runs:
#     TAU=0.09 MODELS="sbtgvn_c8_guide sbtgvn_c8_acs sbtgvn_c8_adjoint" \
#         sbatch --array=0-2 --export=ALL bigpurple/guided_i2sb_knee.sbatch
#
# Submit:        sbatch bigpurple/guided_i2sb_knee.sbatch
# One cell:      sbatch --array=1 bigpurple/guided_i2sb_knee.sbatch                  (sblggs)
# Other cells:   MODELS="sblggs_guide sblggs_acs" sbatch --array=0-1 --export=ALL bigpurple/guided_i2sb_knee.sbatch
# Noise:           default U[0, 0.001] (vlow, since 2026-11-13); NOISE=low|grid|none ... sbatch --export=ALL ...
#                  for U[0, 0.01] / U[0.01, 0.02] / none;
#                  runs go to trained_nets/guided_recon/knee/s3_noise-<preset>/, grid to s3/)

#SBATCH --partition=radiology
#SBATCH --job-name=i2sb-knee
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=EE2178@nyulangone.org
#SBATCH --chdir=/gpfs/scratch/ee2178/ImMAP
#SBATCH --output=/gpfs/scratch/ee2178/ImMAP/bigpurple/logs/%x_%A_%a.out
#SBATCH --error=/gpfs/scratch/ee2178/ImMAP/bigpurple/logs/%x_%A_%a.err
# No `set -euo pipefail` -- it breaks conda activate.

MODELS="${MODELS:-sbtgvn_c8_adjoint sblggs_adjoint}"
ACCELS="${ACCELS:-8}"
CAL_SLICES="${CAL_SLICES:-60}"
FORCE_RESTART="${FORCE_RESTART:-}"
export IMMAP_CLUSTER=bigpurple                       # the generator's k-space roots

CONDA_BASE="$(conda info --base 2>/dev/null)"
[ -n "$CONDA_BASE" ] || { echo "conda not on PATH" >&2; exit 1; }
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate env || { echo "conda activate env failed" >&2; exit 1; }
export PYTHONNOUSERSITE=1
IMMAP_ROOT="${IMMAP_ROOT:-$HOME/scratch/ImMAP}"
cd "$IMMAP_ROOT" || { echo "no such dir: $IMMAP_ROOT" >&2; exit 1; }
mkdir -p bigpurple/logs
[ -n "$SLURM_ARRAY_TASK_ID" ] || { echo "submit as an array job" >&2; exit 1; }
die() { echo "[i2sb-knee] $*" >&2; exit 1; }

GEN=(python scripts/make_guided_recon_configs.py --accels ${ACCELS} --models ${MODELS}
     --guide-accels 1)
if [ -n "${NOISE:-}" ]; then
    GEN+=(--noise "${NOISE}")   # added noise preset: vlow (default) | low | grid | none; own run tag
fi
if [ -n "${TAU:-}" ]; then
    GEN+=(--tau "${TAU}")       # one tau for every cell: no calibration, names get _tau<value>
fi
CELLS="$("${GEN[@]}" --list-cells)" || die "generator failed"
N_TOTAL="$(echo "${CELLS}" | wc -l)"
[ "${SLURM_ARRAY_TASK_ID}" -lt "${N_TOTAL}" ] || \
    die "task ${SLURM_ARRAY_TASK_ID} >= ${N_TOTAL} cells -- fix --array (0-$((N_TOTAL - 1)))"
CELL="$(echo "${CELLS}" | awk -v i="${SLURM_ARRAY_TASK_ID}" '$1 == i')"
NAME="$(echo "${CELL}" | cut -f5)"
TAU_FILE="$(echo "${CELL}" | cut -f6)"
BASE_CONFIG="config/knee/guided/${NAME}.json"
echo "[i2sb-knee] task ${SLURM_ARRAY_TASK_ID}/$((N_TOTAL - 1)): ${NAME} (tau file ${TAU_FILE})"

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
        echo "[i2sb-knee] writing ${BASE_CONFIG} failed (attempt ${try}/6); retrying in $((30 * try)) s" >&2
        sleep $((30 * try))
    done
    echo "[i2sb-knee] could not write ${BASE_CONFIG} after 6 attempts (filesystem full?)" >&2
    exit 1
}
regenerate

python - "${BASE_CONFIG}" <<'PY' || exit 1
import json, os, sys
d = json.load(open(sys.argv[1]))["data"]["train"]
for r in d["kspace_roots"]:
    if not os.path.isdir(r):
        sys.exit(f"[i2sb-knee] k-space root missing: {r} -- run bash bigpurple/setup_knee_data.sh")
root = d.get("smap_root")
if d.get("target") == "sense" and (not root or not os.path.isdir(root) or not os.listdir(root)):
    sys.exit(f"[i2sb-knee] offline maps missing or empty: {root} -- run "
             f"bigpurple/espirit_knee_pairs.sbatch")
vs = d.get("volume_scales")
if vs and not os.path.isfile(vs):
    sys.exit(f"[i2sb-knee] per-volume scales missing: {vs} -- commit it from torch")
PY

if [ "${TAU_FILE}" != "-" ] && [ ! -f "${TAU_FILE}" ]; then
    echo "[i2sb-knee] no tau calibration for this start yet: measuring it (${CAL_SLICES} slices)"
    python scripts/calibrate_recon_bridge.py "${BASE_CONFIG}" --slices "${CAL_SLICES}" \
        --json "${TAU_FILE}" || die "calibration failed"
    regenerate
fi
python -c "import json,sys; i=json.load(open(sys.argv[1]))['i2sb']; print(f\"[i2sb-knee] x1={i['x1_source']} tau={i['tau']} tau_scale={i['tau_scale']}\")" "${BASE_CONFIG}"

RUN_DIR="$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['paths']['save_dir'])" "${BASE_CONFIG}")"
if [ "${FORCE_RESTART}" = "1" ] && [ -d "${RUN_DIR}" ]; then
    OLD="${RUN_DIR}.old_$(date +%Y%m%d_%H%M%S)"
    mv "${RUN_DIR}" "${OLD}"
    echo "[i2sb-knee] FORCE_RESTART: moved the previous run to ${OLD}"
fi
if [ -f "${RUN_DIR}/config.json" ] && [ -f "${RUN_DIR}/net.ckpt" ]; then
    CFG="${RUN_DIR}/config.json"
    echo "[i2sb-knee] resuming from ${CFG}"
else
    CFG="${BASE_CONFIG}"
    echo "[i2sb-knee] fresh run from ${CFG} -> ${RUN_DIR}"
fi

python train.py "${CFG}"
