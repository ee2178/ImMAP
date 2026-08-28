# Shared body for slurm/mg_recon_{knee,brain}.sbatch -- NOT submittable alone.
#
# The caller sets ANATOMY (knee|brain) and sources this. Everything else is
# identical between the two, so it lives here rather than being copied: two
# launchers that drift apart is exactly how a grid ends up half-comparable.
#
# The caller must also set IMMAP_ROOT (the repo checkout) -- it cannot be derived
# here, because sbatch runs a COPY of the launcher from the node's spool dir and
# $BASH_SOURCE points there rather than at the repo.
#
# The caller may also override, before sourcing:
#   CONFIG_ROOT     default "config"
#   SWEEP_EPOCHS    "" = the config's num_epochs; e.g. 20 to probe first
#   REGENERATE      1 = regenerate configs from the generator before running
#   ONLY            "" = every model tag; e.g. "mglpds mggrouplpds" for a subset
#   ACCELS          "" = every acceleration; e.g. "8" for R=8 only
#   ATTN            "" = the generator default (flex); triton|gather to override.
#                   Set it in the launcher rather than relying on the default,
#                   so the run directory records which backend produced it.
#
# ONLY / ACCELS narrow AND RENUMBER the cell list, so an experiment that runs a
# subset gets its own dense 0..N-1 array range. The bound is checked at runtime
# against the filtered list, so a stale --array fails loudly rather than
# silently training the wrong cell.

: "${ANATOMY:?the calling sbatch must set ANATOMY=knee|brain}"
: "${IMMAP_ROOT:?the calling sbatch must set IMMAP_ROOT=/path/to/ImMAP}"
CONFIG_ROOT="${CONFIG_ROOT:-config}"
SWEEP_EPOCHS="${SWEEP_EPOCHS:-}"
REGENERATE="${REGENERATE:-1}"
ONLY="${ONLY:-}"
ACCELS="${ACCELS:-}"
ATTN="${ATTN:-}"

source ~/.bashrc
conda activate gcdl
cd "${IMMAP_ROOT}"

set -eo pipefail
: "${SLURM_ARRAY_TASK_ID:?must be run as an array job (sbatch sets SLURM_ARRAY_TASK_ID)}"

mkdir -p logs

# ---- resolve this task's cell from the generator ------------------------------------------
# --anatomy so each launcher indexes ONLY its own cells: the array bound is
# per-anatomy, and knee task 3 and brain task 3 are different runs.
CELLS="$(python scripts/make_mg_recon_configs.py --list-cells \
    --anatomy "${ANATOMY}" ${ONLY:+--only ${ONLY}} ${ACCELS:+--accels ${ACCELS}})"
N_TOTAL="$(echo "${CELLS}" | wc -l)"

if [ "${SLURM_ARRAY_TASK_ID}" -ge "${N_TOTAL}" ]; then
    echo "[grid] task ${SLURM_ARRAY_TASK_ID} >= ${N_TOTAL} ${ANATOMY} cells" \
         "${ONLY:+(ONLY=\"${ONLY}\")}${ACCELS:+ (ACCELS=\"${ACCELS}\")} -- fix --array"
    echo "       (should be 0-$(( N_TOTAL - 1 )))"
    exit 1
fi

CELL="$(echo "${CELLS}" | awk -v i="${SLURM_ARRAY_TASK_ID}" '$1 == i')"
RTAG="$(echo "${CELL}" | cut -f3)"          # e.g. R8
MODEL="$(echo "${CELL}" | cut -f4)"

BASE_CONFIG="${CONFIG_ROOT}/${ANATOMY}/mg/${MODEL}_${RTAG}.json"

echo "[grid] ${ANATOMY} task ${SLURM_ARRAY_TASK_ID}/$(( N_TOTAL - 1 )): ${RTAG} ${MODEL}"
echo "       config: ${BASE_CONFIG}"

# ---- (re)generate the configs -------------------------------------------------------------
# Cheap, and it keeps a stale hand-edited config from silently deciding a run.
if [ "${REGENERATE}" = "1" ]; then
    python scripts/make_mg_recon_configs.py --out "${CONFIG_ROOT}" \
        --anatomy "${ANATOMY}" ${ATTN:+--attn "${ATTN}"} >/dev/null
fi

if [ ! -f "${BASE_CONFIG}" ]; then
    echo "[grid] ${BASE_CONFIG} does not exist. Generate it with:"
    echo "       python scripts/make_mg_recon_configs.py --out ${CONFIG_ROOT}"
    exit 1
fi

RUN_DIR="trained_nets/mg_recon/${ANATOMY}/${MODEL}_${RTAG}"
GEN_CONFIG="${RUN_DIR}/config.gen.json"
mkdir -p "${RUN_DIR}"

# ---- per-cell bookkeeping (epoch override, fresh wandb run) --------------------------------
python - "${BASE_CONFIG}" "${GEN_CONFIG}" "${RUN_DIR}" "${SWEEP_EPOCHS}" <<'PY'
import json, sys

# write_config, NOT json.dump: train.py reads this file back with yaml.safe_load,
# where json's `1e-06` for eta_min comes back as the STRING '1e-06' and
# CosineAnnealingLR dies doing arithmetic on it. See training/common.py.
from training.common import write_config

base, out, save_dir = sys.argv[1:4]
epochs = sys.argv[4] if len(sys.argv) > 4 else ""

with open(base) as f:
    cfg = json.load(f)

if cfg.get("task") != "recon":
    raise SystemExit(f"[grid] {base} has task={cfg.get('task')!r}, expected 'recon'.")

mri = cfg["mri"]
if mri.get("kspace_type") != "simulated":
    raise SystemExit(
        f"[grid] {base} has kspace_type={mri.get('kspace_type')!r}. This grid is the "
        f"SYNTHETIC k-space experiment -- 'simulated' is the whole point.")

lo, hi = cfg["training"]["noise_std"]
if (lo, hi) != (0.0, 0.01):
    raise SystemExit(f"[grid] {base} has noise_std={[lo, hi]}, expected [0.0, 0.01].")

# preproc='image' pads y~ but leaves E's mask/maps behind, and subtracts a plain
# mean where reconstruction needs the E^H E DC correction. The models raise on
# the first of those, but catching it here names the config, not a tensor shape.
#
# Only models that HAVE the knob are subject to it. A baseline reconstructing
# straight from k-space -- E2EVarNet -- has no preprocessing stage at all, so
# `preproc` is absent from its params and there is nothing to check. Keyed on
# the key's presence rather than an exemption list, so the next baseline that
# lacks it does not have to be added here.
#
# AltSplitCDLNet stays exempt for a different reason: with smap_update its
# unroll re-forms E^H y per layer.
params = cfg["model"]["params"]
if ("preproc" in params and cfg["model"]["type"] != "AltSplitCDLNet"
        and params["preproc"] not in ("kspace", "identity")):
    raise SystemExit(
        f"[grid] {base}: model.params.preproc={params['preproc']!r}. "
        f"Reconstruction needs 'kspace' (pads the operator, removes DC through "
        f"E^H E) or 'identity' (no preprocessing at all).")

paths = cfg.setdefault("paths", {})
paths["save_dir"] = save_dir
paths["ckpt"] = None                    # fresh run; train.py rewrites this to net.ckpt
cfg.setdefault("wandb", {})["id"] = None
if epochs:
    cfg["training"]["num_epochs"] = int(epochs)
    cfg["scheduler"]["params"]["T_max"] = int(epochs) * cfg["training"]["steps_per_epoch"]

write_config(cfg, out)

print(f"[grid] {cfg['model']['type']} R={mri['R']} acs={mri['acs_lines']} "
      f"sigma~U{cfg['training']['noise_std']} epochs={cfg['training']['num_epochs']} -> {out}")
PY

python train.py "${GEN_CONFIG}"
