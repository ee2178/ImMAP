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
#SBATCH --job-name=MGRecon-grid
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --array=0-19          # MUST equal the cell count; checked at runtime
#
# Multigrid reconstruction grid on fastMRI, SYNTHETIC k-space, sigma ~ U[0, 0.05].
#
#   5 models x 2 anatomies (knee PD, brain T2) x 2 accelerations (R=8, R=4) = 20 cells
#
# The cell list lives in ONE place -- scripts/make_mg_recon_configs.py -- and this
# script asks it for the mapping rather than duplicating the axes:
#
#     python scripts/make_mg_recon_configs.py --list-cells
#     0  knee  R8  mgcdlnet
#     1  knee  R8  mggroupcdl
#     ...
#
# so adding a model or an acceleration means editing the generator and the
# --array bound, and nothing else.
#
# The models:
#   mgcdlnet / mggroupcdl  -- V-cycle, K=[1,[8,8,8]]
#   cdlnet / groupcdl      -- K=30 baselines (the reference's own, NOT matched)
#   ladmm                  -- AltSplitCDLNet, MGCDLNet prox
#
# MGLPDS / MGGroupLPDS are deliberately absent: ImMAP's MGLPDS is the Fenchel
# dual of the CDL prox, while the reference's multigrid MRI "LPDS" is
# mg_lpdsnet -- a different architecture that is not ported.
#
# GPU NOTE: the two group models default to attn_backend="flex". torch has no CPU
# backward for FlexAttention, so those cells are GPU-only (fine here) -- and
# MGCDLNet must expose `self.attn_backend` for train.py to compile the kernel.
#
# Submit:      sbatch slurm/mg_recon_grid.sbatch
# One cell:    sbatch --array=0 slurm/mg_recon_grid.sbatch
# Probe first: set SWEEP_EPOCHS=20 and look at the val curves before the full 28.

# ---- knobs ---------------------------------------------------------------------------------
CONFIG_ROOT="config"
SWEEP_EPOCHS=""                      # empty = the config's num_epochs; set e.g. 20 to probe
REGENERATE=1                         # 1 = regenerate configs from the generator first
# --------------------------------------------------------------------------------------------

source ~/.bashrc
conda activate gcdl
cd ~/scratch/ee2178/ImMAP

set -eo pipefail
: "${SLURM_ARRAY_TASK_ID:?must be run as an array job (sbatch sets SLURM_ARRAY_TASK_ID)}"

mkdir -p logs

# ---- resolve this task's cell from the generator ------------------------------------------
CELLS="$(python scripts/make_mg_recon_configs.py --list-cells)"
N_TOTAL="$(echo "${CELLS}" | wc -l)"

if [ "${SLURM_ARRAY_TASK_ID}" -ge "${N_TOTAL}" ]; then
    echo "[grid] task ${SLURM_ARRAY_TASK_ID} >= ${N_TOTAL} cells -- fix --array"
    echo "       (should be 0-$(( N_TOTAL - 1 )))"
    exit 1
fi

CELL="$(echo "${CELLS}" | awk -v i="${SLURM_ARRAY_TASK_ID}" '$1 == i')"
ANATOMY="$(echo "${CELL}" | cut -f2)"
RTAG="$(echo "${CELL}" | cut -f3)"          # e.g. R8
MODEL="$(echo "${CELL}" | cut -f4)"

BASE_CONFIG="${CONFIG_ROOT}/${ANATOMY}/mg/${MODEL}_${RTAG}.json"

echo "[grid] task ${SLURM_ARRAY_TASK_ID}/$(( N_TOTAL - 1 )): ${ANATOMY} ${RTAG} ${MODEL}"
echo "       config: ${BASE_CONFIG}"

# ---- (re)generate the configs -------------------------------------------------------------
# Cheap, and it keeps a stale hand-edited config from silently deciding a run.
if [ "${REGENERATE}" = "1" ]; then
    python scripts/make_mg_recon_configs.py --out "${CONFIG_ROOT}" >/dev/null
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
if (lo, hi) != (0.0, 0.05):
    raise SystemExit(f"[grid] {base} has noise_std={[lo, hi]}, expected [0.0, 0.05].")

# preproc='image' pads y~ but leaves E's mask/maps behind, and subtracts a plain
# mean where reconstruction needs the E^H E DC correction. MGCDLNet raises on the
# first of those, but catching it here names the config rather than a tensor shape.
# (AltSplitCDLNet is exempt: with smap_update its unroll re-forms E^H y per layer.)
params = cfg["model"]["params"]
if cfg["model"]["type"] != "AltSplitCDLNet" \
        and params.get("preproc") not in ("kspace", "identity"):
    raise SystemExit(
        f"[grid] {base}: model.params.preproc={params.get('preproc')!r}. "
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
