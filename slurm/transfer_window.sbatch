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
#SBATCH --job-name=MGNorm
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --array=0-1
#
# Is the  P = 4 R^T  scale split the reason the rewritten transfer underperforms?
#
# Two cells, one MGLPDSNet, fastMRI knee recon at R=8, both on the BOX kernel
# (the same 2x2 mean the pre-rewrite code restricted with):
#
#   0  transfer_filter=box        sum(h)=1, c=4   R(1)=1     P(1)=1
#   1  transfer_filter=box-unit   sum(h)=2, c=1   R(1)=2     P(1)=0.5
#
# Both keep  P R (1) = 1, so the round trip is faithful either way. What
# differs is how the factor^d is split between the two halves:
#
#   "box"       each operator preserves constants on its own, so a restricted
#               iterate sits at the same POINTWISE amplitude as the fine one --
#               which is what preload_with_widening assumes when it copies the
#               fine prox thresholds down to the coarse level.
#   "box-unit"  R IS P^T, with no scalar at all, and ||P||_2 ~ 1. Cleaner
#               operator algebra: galerkin(E).adjoint becomes a true adjoint
#               rather than 1/4 of one. The coarse iterate lands at 2x the fine
#               amplitude, and sigma_c/sigma goes 0.5 -> 1.0 to match.
#
# WHAT THIS CAN AND CANNOT SHOW. The coarse Gram  R E^H E P  is bit-identical
# under both (verified: rel diff 0.000e+00) -- the gains cancel in the product.
# So this is a change of VARIABLES on the coarse grid, not a change to the
# coarse problem. Any difference between these two arms therefore has to come
# from something nonlinear in the coarse iterate's scale: the prox thresholds,
# and eta/alpha. If the arms tie, the c factor is not what is hurting, and the
# suspects fall back to sigma_c calibration and the circular padding.
#
# Submit both:  sbatch slurm/transfer_window.sbatch
# Separately:   sbatch --array=0 slurm/transfer_window.sbatch   # box,  c=4
#               sbatch --array=1 slurm/transfer_window.sbatch   # box,  c=1
# Probe first:  SWEEP_EPOCHS=20 sbatch slurm/transfer_window.sbatch

ANATOMY=knee
MODEL=mglpds
RTAG=R8
CONFIG_ROOT="${CONFIG_ROOT:-config}"
SWEEP_EPOCHS="${SWEEP_EPOCHS:-}"

ARMS=(box box-unit)

source ~/.bashrc
conda activate gcdl

# sbatch runs a COPY of this file from the node spool dir, so $BASH_SOURCE does
# not point at the repo. Resolve it explicitly, the same way
# slurm/mg_recon_knee.sbatch does.
for _c in "${IMMAP_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" "$HOME/scratch/ee2178/ImMAP"; do
    if [ -n "${_c}" ] && [ -f "${_c}/train.py" ]; then
        IMMAP_ROOT="${_c}"
        break
    fi
done
if [ ! -f "${IMMAP_ROOT:-}/train.py" ]; then
    echo "[norm] cannot locate the ImMAP checkout (tried IMMAP_ROOT," \
         "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset}, ~/scratch/ee2178/ImMAP)." >&2
    echo "[norm] submit from the repo root, or: IMMAP_ROOT=/path/to/ImMAP sbatch $0" >&2
    exit 1
fi
cd "${IMMAP_ROOT}"

set -eo pipefail
: "${SLURM_ARRAY_TASK_ID:?must be run as an array job (sbatch sets SLURM_ARRAY_TASK_ID)}"
mkdir -p logs

if [ "${SLURM_ARRAY_TASK_ID}" -ge "${#ARMS[@]}" ]; then
    echo "[norm] task ${SLURM_ARRAY_TASK_ID} >= ${#ARMS[@]} arms -- use --array=0-1"
    exit 1
fi
FILTER="${ARMS[${SLURM_ARRAY_TASK_ID}]}"

BASE_CONFIG="${CONFIG_ROOT}/${ANATOMY}/mg/${MODEL}_${RTAG}.json"
RUN_DIR="trained_nets/transfer_norm/${MODEL}_${RTAG}_${FILTER}"
GEN_CONFIG="${RUN_DIR}/config.gen.json"

echo "[norm] task ${SLURM_ARRAY_TASK_ID}/$(( ${#ARMS[@]} - 1 )): ${MODEL} ${ANATOMY} ${RTAG} filter=${FILTER}"
echo "       base config: ${BASE_CONFIG}"
echo "       run dir:     ${RUN_DIR}"

if [ ! -f "${BASE_CONFIG}" ]; then
    echo "[norm] ${BASE_CONFIG} does not exist. Generate it with:"
    echo "       python scripts/make_mg_recon_configs.py --out ${CONFIG_ROOT} --anatomy ${ANATOMY}"
    exit 1
fi

mkdir -p "${RUN_DIR}"

# ---- stamp the arm into the config --------------------------------------------------------
python - "${BASE_CONFIG}" "${GEN_CONFIG}" "${RUN_DIR}" "${FILTER}" "${SWEEP_EPOCHS}" <<'PY'
import json, sys

# write_config, NOT json.dump: train.py reads this back with yaml.safe_load,
# where json's `1e-06` returns as the STRING '1e-06' and CosineAnnealingLR dies
# doing arithmetic on it. See training/common.py.
from training.common import write_config
from operators.resample import (_check_filter, adjoint_scale, filter_length,
                                noise_scale, split_normalisation)

base, out, save_dir, filt = sys.argv[1:5]
epochs = sys.argv[5] if len(sys.argv) > 5 else ""

# _check_filter, not `in TRANSFER_FILTERS`: the normalisation suffix is part of
# the name, so a membership test would reject "box-unit".
try:
    _check_filter(filt)
except ValueError as e:
    raise SystemExit(f"[norm] {e}")

with open(base) as f:
    cfg = json.load(f)

if cfg.get("task") != "recon":
    raise SystemExit(f"[norm] {base} has task={cfg.get('task')!r}, expected 'recon'.")

params = cfg["model"]["params"]
if params.get("K") is None or isinstance(params["K"], int):
    raise SystemExit(
        f"[norm] {base} has K={params.get('K')!r} -- a bare int means NO "
        f"multigrid, so the transfer filter is never used and both arms would "
        f"train the identical network. This ablation needs K=[K_outer, [i0, ...]].")

params["transfer_filter"] = filt

paths = cfg.setdefault("paths", {})
paths["save_dir"] = save_dir
paths["ckpt"] = None                    # fresh run; train.py rewrites this
cfg.setdefault("wandb", {})["id"] = None
cfg.setdefault("experiment", {})["name"] = f"MGLPDSNet_norm_{filt}"

if epochs:
    cfg["training"]["num_epochs"] = int(epochs)
    cfg["scheduler"]["params"]["T_max"] = (
        int(epochs) * cfg["training"]["steps_per_epoch"])

write_config(cfg, out)

# Log what the arm actually changes, rather than leaving it to be inferred from
# the filter name.
base_name, norm = split_normalisation(filt)
print(f"[norm] {cfg['model']['type']} R={cfg['mri']['R']} filter={filt} "
      f"(kernel={base_name}, norm={norm}, c={adjoint_scale(filt):g}) "
      f"L={filter_length(filt)} sigma_c/sigma={noise_scale(filter=filt):.4f} "
      f"epochs={cfg['training']['num_epochs']} -> {out}")
PY

python train.py "${GEN_CONFIG}"
