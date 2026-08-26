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
#SBATCH --array=0-2
#
# Where did the rewritten grid transfer lose ground against the old one?
#
# Three cells, one MGLPDSNet, fastMRI knee recon at R=8. All three restrict
# with the SAME 2x2 mean kernel the pre-rewrite code used, so the anti-aliasing
# is held fixed and only the pairing and its normalisation move:
#
#   0  legacy     R = 2x2 mean, P = bilinear    NOT a transpose pair
#   1  box        R = 2x2 mean, P = 4 R^T       sum(h)=1, c=4, R(1)=P(1)=1
#   2  box-unit   R = 2x2 mean, P = R^T         sum(h)=2, c=1, R(1)=2, P(1)=0.5
#
# Read as two independent one-step comparisons:
#
#   0 -> 1   does making the pair a genuine transpose matter?
#            `legacy` pairs a mean-pool restriction with bilinear
#            interpolation, which are not transposes -- they differ by 22%, and
#            the coarse Gram comes out 14.5% non-Hermitian, so dF_coarse is the
#            subgradient of no objective at all. `box` fixes exactly that and
#            changes nothing else: same kernel, same sigma_c/sigma = 0.5, same
#            FAS-correction strength.
#
#   1 -> 2   does the SCALE SPLIT matter?
#            Both keep P R (1) = 1, so the round trip is faithful either way;
#            they differ in how the factor^d is divided. "box" has each operator
#            preserve constants on its own, so a restricted iterate sits at the
#            same POINTWISE amplitude as the fine one -- what
#            preload_with_widening assumes when it copies the fine prox
#            thresholds down. "box-unit" makes R literally P^T with ||P||_2 ~ 1
#            (galerkin(E).adjoint becomes a true adjoint rather than 1/4 of
#            one), at the cost of putting the coarse iterate at 2x the fine
#            amplitude; sigma_c/sigma goes 0.5 -> 1.0 to match.
#
# WHAT ARM 2 CAN AND CANNOT SHOW. The coarse Gram R E^H E P is bit-identical
# between arms 1 and 2 (verified: rel diff 0.000e+00) -- the gains cancel in the
# product. So 1 -> 2 is a change of VARIABLES on the coarse grid, not a change
# to the coarse problem, and any difference has to come from something
# nonlinear in the coarse iterate's scale: the prox thresholds, eta, alpha. If
# 1 and 2 tie, c is exonerated.
#
# Arm 0 is a BIT-EXACT replica of the pre-rewrite code -- parameters, forward,
# loss AND gradients all match commit f74a799 to 0.000e+00 (tests/
# test_multigrid.py pins the operator half of that). So it is a real baseline,
# not an approximation of one.
#
# Submit all:   sbatch slurm/transfer_window.sbatch
# Separately:   sbatch --array=0 slurm/transfer_window.sbatch   # legacy
#               sbatch --array=1 slurm/transfer_window.sbatch   # box, c=4
#               sbatch --array=2 slurm/transfer_window.sbatch   # box, c=1
# Probe first:  SWEEP_EPOCHS=20 sbatch slurm/transfer_window.sbatch

ANATOMY=knee
MODEL=mglpds
RTAG=R8
CONFIG_ROOT="${CONFIG_ROOT:-config}"
SWEEP_EPOCHS="${SWEEP_EPOCHS:-}"

ARMS=(legacy box box-unit)

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
    echo "[norm] task ${SLURM_ARRAY_TASK_ID} >= ${#ARMS[@]} arms -- use --array=0-2"
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
                                is_adjoint_pair, noise_scale,
                                split_normalisation)

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
# `adjoint_scale` would happily report 4 for 'legacy', but legacy is not a
# transpose pair at any scale -- printing a c there would invent a relationship
# the run does not have.
pair = (f"norm={norm}, c={adjoint_scale(filt):g}" if is_adjoint_pair(filt)
        else "NOT an adjoint pair (P is bilinear, not a multiple of R^T)")
print(f"[norm] {cfg['model']['type']} R={cfg['mri']['R']} filter={filt} "
      f"(kernel={base_name}, {pair}) "
      f"L={filter_length(filt)} sigma_c/sigma={noise_scale(filter=filt):.4f} "
      f"epochs={cfg['training']['num_epochs']} -> {out}")
PY

python train.py "${GEN_CONFIG}"
