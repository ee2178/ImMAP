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
#SBATCH --job-name=MGWindow
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --array=0-1
#
# Does the Kaiser window on the sinc transfer filter earn its keep?
#
# Two cells, one MGLPDSNet, fastMRI knee recon at R=8:
#
#   0  transfer_filter=sinc              (Kaiser-windowed -- the default)
#   1  transfer_filter=sinc_unwindowed   (rectangular truncation)
#
# Same config, same seed, same kernel length (8 = 4 * factor), same
# exact-adjoint machinery. The ONLY difference is the window. Everything
# downstream that depends on the kernel -- sigma_c = sigma * ||h||_2, the coarse
# Gram -- is recomputed from it, so nothing is silently pinned at the other
# arm's value.
#
#   arm               passband 0.25pi   stopband 0.6pi   ||h||_2   P R gain
#   sinc                   0.957            0.267        0.4162      0.92
#   sinc_unwindowed        1.272            0.136        0.5971      1.62
#
# Not a foregone conclusion. Rectangular truncation has the SHARPER cutoff and
# actually aliases less; what it does instead is overshoot 27% in the passband,
# which squares to 62% amplification in the coarse round-trip P R because the
# kernel appears twice there. See notebooks/transfer_window_ablation.ipynb.
#
# The thing to watch: alpha (the coarse-correction step) is learned, and a
# systematic 1.6x gain is exactly what a learned scalar can absorb. If the
# unwindowed arm's alpha collapses, the ablation will come out flat for a
# reason that is not "the window does not matter".
#
# Submit:      sbatch slurm/transfer_window.sbatch
# One arm:     sbatch --array=0 slurm/transfer_window.sbatch
# Probe first: SWEEP_EPOCHS=20 sbatch slurm/transfer_window.sbatch

ANATOMY=knee
MODEL=mglpds
RTAG=R8
CONFIG_ROOT="${CONFIG_ROOT:-config}"
SWEEP_EPOCHS="${SWEEP_EPOCHS:-}"

ARMS=(sinc sinc_unwindowed)

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
    echo "[window] cannot locate the ImMAP checkout (tried IMMAP_ROOT," \
         "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset}, ~/scratch/ee2178/ImMAP)." >&2
    echo "[window] submit from the repo root, or: IMMAP_ROOT=/path/to/ImMAP sbatch $0" >&2
    exit 1
fi
cd "${IMMAP_ROOT}"

set -eo pipefail
: "${SLURM_ARRAY_TASK_ID:?must be run as an array job (sbatch sets SLURM_ARRAY_TASK_ID)}"
mkdir -p logs

if [ "${SLURM_ARRAY_TASK_ID}" -ge "${#ARMS[@]}" ]; then
    echo "[window] task ${SLURM_ARRAY_TASK_ID} >= ${#ARMS[@]} arms -- use --array=0-1"
    exit 1
fi
FILTER="${ARMS[${SLURM_ARRAY_TASK_ID}]}"

BASE_CONFIG="${CONFIG_ROOT}/${ANATOMY}/mg/${MODEL}_${RTAG}.json"
RUN_DIR="trained_nets/transfer_window/${MODEL}_${RTAG}_${FILTER}"
GEN_CONFIG="${RUN_DIR}/config.gen.json"

echo "[window] task ${SLURM_ARRAY_TASK_ID}/$(( ${#ARMS[@]} - 1 )): ${MODEL} ${ANATOMY} ${RTAG} filter=${FILTER}"
echo "         base config: ${BASE_CONFIG}"
echo "         run dir:     ${RUN_DIR}"

if [ ! -f "${BASE_CONFIG}" ]; then
    echo "[window] ${BASE_CONFIG} does not exist. Generate it with:"
    echo "         python scripts/make_mg_recon_configs.py --out ${CONFIG_ROOT} --anatomy ${ANATOMY}"
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
from operators.resample import TRANSFER_FILTERS, filter_length, noise_scale

base, out, save_dir, filt = sys.argv[1:5]
epochs = sys.argv[5] if len(sys.argv) > 5 else ""

if filt not in TRANSFER_FILTERS:
    raise SystemExit(f"[window] filter={filt!r} is not one of {TRANSFER_FILTERS}")

with open(base) as f:
    cfg = json.load(f)

if cfg.get("task") != "recon":
    raise SystemExit(f"[window] {base} has task={cfg.get('task')!r}, expected 'recon'.")

params = cfg["model"]["params"]
if params.get("K") is None or isinstance(params["K"], int):
    raise SystemExit(
        f"[window] {base} has K={params.get('K')!r} -- a bare int means NO "
        f"multigrid, so the transfer filter is never used and both arms would "
        f"train the identical network. This ablation needs K=[K_outer, [i0, ...]].")

params["transfer_filter"] = filt

paths = cfg.setdefault("paths", {})
paths["save_dir"] = save_dir
paths["ckpt"] = None                    # fresh run; train.py rewrites this
cfg.setdefault("wandb", {})["id"] = None
cfg.setdefault("experiment", {})["name"] = f"MGLPDSNet_window_{filt}"

if epochs:
    cfg["training"]["num_epochs"] = int(epochs)
    cfg["scheduler"]["params"]["T_max"] = (
        int(epochs) * cfg["training"]["steps_per_epoch"])

write_config(cfg, out)

# Log what the arm actually changes, rather than leaving it to be inferred from
# the filter name.
print(f"[window] {cfg['model']['type']} R={cfg['mri']['R']} filter={filt} "
      f"L={filter_length(filt)} sigma_c/sigma={noise_scale(filter=filt):.4f} "
      f"epochs={cfg['training']['num_epochs']} -> {out}")
PY

python train.py "${GEN_CONFIG}"
