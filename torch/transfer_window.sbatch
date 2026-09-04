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
#SBATCH --job-name=MGXfer
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --array=0-4
#
# Grid-transfer ablation for the MGLPDSNet V-cycle: five cells, one anatomy,
# one acceleration, everything else held fixed.
#
#   #  filter            kernel        L   pair      c   sigma_c/sigma  ||G_c||
#   0  legacy            2x2 mean +    2   NOT a    n/a     0.5000      0.4770
#                        bilinear          pair                        (13.7% non-Hermitian)
#   1  box               2x2 mean      2   P=4R^T    4      0.5000      0.5764
#   2  box-unit          2x2 mean      2   P=R^T     1      1.0000      0.5764
#   3  sinc              kaiser-sinc   8   P=4R^T    4      0.4162      0.5089
#   4  sinc_unwindowed   rect-sinc     8   P=4R^T    4      0.5971      1.1764
#
# (||G_c|| measured against a fine ||E^H E|| of 0.7598 at R=8.)
#
# Four independent one-step comparisons, all off cell 0 or 1:
#
#   0 -> 1  ADJOINTNESS. Same 2x2 mean kernel, same sigma_c, same FAS-correction
#           strength; the only change is that P becomes a true multiple of R^T.
#           `legacy` pairs mean-pool restriction with bilinear interpolation,
#           which differ by 22%, leaving the coarse Gram 13.7% non-Hermitian --
#           so dF_coarse is the subgradient of no objective at all.
#
#   1 -> 2  SCALE SPLIT. Both preserve constants through the round trip; they
#           differ in how factor^d is divided between R and P. The coarse Gram
#           is BIT-IDENTICAL between them (0.5764 both), so this is a change of
#           variables on the coarse grid, not a change to the coarse problem.
#           Any difference must come from something nonlinear in the coarse
#           iterate's scale: prox thresholds, eta, alpha. If 1 and 2 tie, c is
#           exonerated.
#
#   1 -> 3  ANTI-ALIASING. Same exact-adjoint machinery, better filter. The box
#           is only -3 dB at the coarse Nyquist and cannot be improved on at its
#           length (minimising |H(pi/2)|/|H(0)| over all [a,b] gives 1/sqrt2, at
#           a=b=1/2). NOT drop-in: sinc restricts a broadband dF to 0.81x and
#           drops sigma_c/sigma to 0.4162, so tau0 / eta_0 are ~1.2x off the
#           calibration they were tuned with. Expect this cell to need retuning
#           before its number means anything.
#
#   3 -> 4  THE WINDOW. Rectangular truncation has the SHARPER cutoff and
#           actually aliases less; what it does instead is overshoot 27% in the
#           passband, which squares to 62% amplification in the round trip P R
#           because the kernel appears twice there. That pushes ||G_c|| to 1.18
#           -- above 1, and above the fine Gram. The unrolled step is
#           z <- prox(z - A(E^H E B z - y)) with step size 1, which assumes
#           ||E^H E|| <= 1, so this cell's coarse level is NON-CONTRACTIVE.
#           Expect it to lose; the interest is by how much.
#
# Cell 0 is a BIT-EXACT replica of the pre-rewrite code -- parameters, forward,
# loss AND gradients all match commit f74a799 to 0.000e+00. A real baseline, not
# an approximation of one.
#
# Do not read cell 2's FAS-correction magnitude as 4x cell 1's: R(1)=2 under the
# unit normalisation, so a double restriction scales by 4 and the prolongation
# divides it straight back out. It is bookkeeping, not strength.
#
# Submit all:   sbatch torch/transfer_window.sbatch
# One cell:     sbatch --array=3 torch/transfer_window.sbatch
# A subset:     sbatch --array=0,1,3 torch/transfer_window.sbatch
# Probe first:  SWEEP_EPOCHS=20 sbatch torch/transfer_window.sbatch
# Other side:   ANATOMY=knee sbatch torch/transfer_window.sbatch

# ============================================================================
#  SET THE EXPERIMENT HERE
# ============================================================================
# knee and brain are NOT comparable to each other: scale_fac is 5000 vs 2000,
# so the loss is on a 2.5x different amplitude before any reconstruction
# quality enters, and the slice ranges differ too (PD knee 12-25, T2W brain
# 0-8). Compare cells within one anatomy only.
ANATOMY="${ANATOMY:-brain}"          # brain | knee
RTAG="${RTAG:-R8}"                   # R8 | R4
MODEL="${MODEL:-mglpds}"

CONFIG_ROOT="${CONFIG_ROOT:-config}"
SWEEP_EPOCHS="${SWEEP_EPOCHS:-}"

ARMS=(legacy box box-unit sinc sinc_unwindowed)

source ~/.bashrc
conda activate gcdl

# sbatch runs a COPY of this file from the node spool dir, so $BASH_SOURCE does
# not point at the repo. Resolve it explicitly, the same way
# torch/mg_recon_knee.sbatch does.
for _c in "${IMMAP_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" "$HOME/scratch/ee2178/ImMAP"; do
    if [ -n "${_c}" ] && [ -f "${_c}/train.py" ]; then
        IMMAP_ROOT="${_c}"
        break
    fi
done
if [ ! -f "${IMMAP_ROOT:-}/train.py" ]; then
    echo "[xfer] cannot locate the ImMAP checkout (tried IMMAP_ROOT," \
         "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset}, ~/scratch/ee2178/ImMAP)." >&2
    echo "[xfer] submit from the repo root, or: IMMAP_ROOT=/path/to/ImMAP sbatch $0" >&2
    exit 1
fi
cd "${IMMAP_ROOT}"

set -eo pipefail
: "${SLURM_ARRAY_TASK_ID:?must be run as an array job (sbatch sets SLURM_ARRAY_TASK_ID)}"
mkdir -p logs

case "${ANATOMY}" in
    brain|knee) ;;
    *) echo "[xfer] ANATOMY=${ANATOMY} -- expected 'brain' or 'knee'." >&2; exit 1 ;;
esac

if [ "${SLURM_ARRAY_TASK_ID}" -ge "${#ARMS[@]}" ]; then
    echo "[xfer] task ${SLURM_ARRAY_TASK_ID} >= ${#ARMS[@]} arms -- use --array=0-$(( ${#ARMS[@]} - 1 ))"
    exit 1
fi
FILTER="${ARMS[${SLURM_ARRAY_TASK_ID}]}"

BASE_CONFIG="${CONFIG_ROOT}/${ANATOMY}/mg/${MODEL}_${RTAG}.json"
# The anatomy is in the run dir: knee and brain cells are different experiments
# and must not land in the same directory.
RUN_DIR="trained_nets/transfer/${ANATOMY}/${MODEL}_${RTAG}_${FILTER}"
GEN_CONFIG="${RUN_DIR}/config.gen.json"

echo "[xfer] task ${SLURM_ARRAY_TASK_ID}/$(( ${#ARMS[@]} - 1 )): ${MODEL} ${ANATOMY} ${RTAG} filter=${FILTER}"
echo "       base config: ${BASE_CONFIG}"
echo "       run dir:     ${RUN_DIR}"

if [ ! -f "${BASE_CONFIG}" ]; then
    echo "[xfer] ${BASE_CONFIG} does not exist. Generate it with:"
    echo "       python scripts/make_mg_recon_configs.py --out ${CONFIG_ROOT} --anatomy ${ANATOMY}"
    exit 1
fi

mkdir -p "${RUN_DIR}"

# ---- stamp the arm into the config --------------------------------------------------------
python - "${BASE_CONFIG}" "${GEN_CONFIG}" "${RUN_DIR}" "${FILTER}" "${ANATOMY}" "${SWEEP_EPOCHS}" <<'PY'
import json, sys

# write_config, NOT json.dump: train.py reads this back with yaml.safe_load,
# where json's `1e-06` returns as the STRING '1e-06' and CosineAnnealingLR dies
# doing arithmetic on it. See training/common.py.
from training.common import write_config
from operators.resample import (_check_filter, adjoint_scale, filter_length,
                                is_adjoint_pair, noise_scale,
                                split_normalisation)

base, out, save_dir, filt, anatomy = sys.argv[1:6]
epochs = sys.argv[6] if len(sys.argv) > 6 else ""

# _check_filter, not `in TRANSFER_FILTERS`: the normalisation suffix is part of
# the name, so a membership test would reject "box-unit".
try:
    _check_filter(filt)
except ValueError as e:
    raise SystemExit(f"[xfer] {e}")

with open(base) as f:
    cfg = json.load(f)

if cfg.get("task") != "recon":
    raise SystemExit(f"[xfer] {base} has task={cfg.get('task')!r}, expected 'recon'.")

# The anatomy is baked into the config's data paths, not just its filename. If the
# two ever disagreed, every cell would silently train on whatever the file said.
cfg_anat = cfg["data"]["train"].get("anatomy")
if cfg_anat != anatomy:
    raise SystemExit(
        f"[xfer] {base} has data.train.anatomy={cfg_anat!r} but this job is running "
        f"ANATOMY={anatomy!r}. Regenerate the configs rather than trusting the path.")

params = cfg["model"]["params"]
if params.get("K") is None or isinstance(params["K"], int):
    raise SystemExit(
        f"[xfer] {base} has K={params.get('K')!r} -- a bare int means NO "
        f"multigrid, so the transfer filter is never used and every arm would "
        f"train the identical network. This ablation needs K=[K_outer, [i0, ...]].")

params["transfer_filter"] = filt

paths = cfg.setdefault("paths", {})
paths["save_dir"] = save_dir
paths["ckpt"] = None                    # fresh run; train.py rewrites this
cfg.setdefault("wandb", {})["id"] = None
cfg.setdefault("experiment", {})["name"] = (
    f"MGLPDSNet_xfer_{anatomy}_{cfg['mri']['R']}_{filt}")

if epochs:
    cfg["training"]["num_epochs"] = int(epochs)
    cfg["scheduler"]["params"]["T_max"] = (
        int(epochs) * cfg["training"]["steps_per_epoch"])

write_config(cfg, out)

# Log what the arm actually changes, rather than leaving it to be inferred from
# the filter name. `adjoint_scale` would happily report 4 for 'legacy', but
# legacy is not a transpose pair at any scale -- printing a c there would invent
# a relationship the run does not have.
kernel, norm = split_normalisation(filt)
pair = (f"norm={norm}, c={adjoint_scale(filt):g}" if is_adjoint_pair(filt)
        else "NOT an adjoint pair (P is bilinear, not a multiple of R^T)")
print(f"[xfer] {cfg['model']['type']} {anatomy} R={cfg['mri']['R']} "
      f"scale_fac={cfg['data']['train']['scale_fac']} filter={filt} "
      f"(kernel={kernel}, {pair}) L={filter_length(filt)} "
      f"sigma_c/sigma={noise_scale(filter=filt):.4f} "
      f"epochs={cfg['training']['num_epochs']} -> {out}")
PY

python train.py "${GEN_CONFIG}"
