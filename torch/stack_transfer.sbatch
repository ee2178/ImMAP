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
#SBATCH --job-name=StackXfer
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --array=0-2
#
# STAGE 2 of 2: seed all six outer V-cycles from the pretrained single one.
#
#   0  baseline   fresh random init -- the current MGLPDSNet, unchanged
#   1  naive      replicate the pretrained V-cycle into all 6 slots, touch nothing
#   2  damped     replicate, then scale the PRIMAL STEP SIZE by 0.01 in slots 1..5
#
# Cells 1 and 2 are identical apart from that one scale factor, and cell 0 is the
# thing they have to beat. All three train from step 0 with a fresh optimizer and
# the full LR schedule, so the wandb curves are directly comparable.
#
# WHY THE DAMPING. A K=1 network learns a TERMINAL map -- y~ to x_hat in one shot.
# Composing six terminal maps overshoots. Since x_new = x - tau * residual,
# driving the primal step size toward zero makes a V-cycle near-identity, so the
# tail starts as a no-op refinement of the pretrained map rather than as another
# terminal map. Measured init loss on a toy problem, same pretrained source:
#
#     naive replication           2.223e-01     75x WORSE than fresh
#     step size x0.01 in tail     1.893e-03     1.6x BETTER than fresh
#     fresh random                2.970e-03
#
# 0.01 rather than exactly 0: at 0 the primal branch's gradient vanishes and only
# 70 of 610 tail parameters get gradients, so the tail starts dead. At 0.01 it is
# 606 of 610 with the same init loss.
#
# NOT SETTLED, and the reason this is an experiment rather than a default: on a
# toy problem naive DID self-correct -- 2.22e-01 to 1.02e-03 within 150 steps --
# but stayed ~20x behind damped and never caught fresh. Whether that holds on
# real data at 100k pretrain steps is exactly what these three cells measure.
# The prox THRESHOLDS are transferred untouched in both arms; see
# models/stacking.py for why zeroing them destroys the pretrain.
#
# Submit:      sbatch slurm/stack_transfer.sbatch
# One cell:    sbatch --array=2 slurm/stack_transfer.sbatch
# Chained:     JID=$(sbatch --parsable slurm/stack_pretrain.sbatch)
#              sbatch --dependency=afterok:${JID} slurm/stack_transfer.sbatch
# Probe first: SWEEP_EPOCHS=20 sbatch slurm/stack_transfer.sbatch

# ============================================================================
#  SET THE EXPERIMENT HERE  (must match slurm/stack_pretrain.sbatch)
# ============================================================================
ANATOMY="${ANATOMY:-brain}"          # brain | knee
RTAG="${RTAG:-R8}"
MODEL="${MODEL:-mglpds}"

CONFIG_ROOT="${CONFIG_ROOT:-config}"
RUN_ROOT="${RUN_ROOT:-trained_nets/stack}"
SWEEP_EPOCHS="${SWEEP_EPOCHS:-}"

ARMS=(baseline naive damped)
STEP_SCALES=(- 1.0 0.01)             # unused for cell 0

source ~/.bashrc
conda activate gcdl

for _c in "${IMMAP_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" "$HOME/scratch/ee2178/ImMAP"; do
    if [ -n "${_c}" ] && [ -f "${_c}/train.py" ]; then IMMAP_ROOT="${_c}"; break; fi
done
if [ ! -f "${IMMAP_ROOT:-}/train.py" ]; then
    echo "[stack] cannot locate the ImMAP checkout." >&2; exit 1
fi
cd "${IMMAP_ROOT}"
set -eo pipefail
: "${SLURM_ARRAY_TASK_ID:?must be run as an array job}"
mkdir -p logs

case "${ANATOMY}" in brain|knee) ;; *)
    echo "[stack] ANATOMY=${ANATOMY} -- expected 'brain' or 'knee'." >&2; exit 1 ;; esac
if [ "${SLURM_ARRAY_TASK_ID}" -ge "${#ARMS[@]}" ]; then
    echo "[stack] task ${SLURM_ARRAY_TASK_ID} >= ${#ARMS[@]} arms -- use --array=0-2" >&2
    exit 1
fi

ARM="${ARMS[${SLURM_ARRAY_TASK_ID}]}"
STEP_SCALE="${STEP_SCALES[${SLURM_ARRAY_TASK_ID}]}"
BASE_CONFIG="${CONFIG_ROOT}/${ANATOMY}/mg/${MODEL}_${RTAG}.json"
PRE_DIR="${RUN_ROOT}/${ANATOMY}/${MODEL}_${RTAG}_pretrain_K1"
RUN_DIR="${RUN_ROOT}/${ANATOMY}/${MODEL}_${RTAG}_${ARM}"
GEN_CONFIG="${RUN_DIR}/config.gen.json"
INIT_CKPT="${RUN_DIR}/init.ckpt"

echo "[stack] STAGE 2  task ${SLURM_ARRAY_TASK_ID}: arm=${ARM}  ${ANATOMY} ${RTAG}"
echo "        base config: ${BASE_CONFIG}"
echo "        run dir:     ${RUN_DIR}"

[ -f "${BASE_CONFIG}" ] || { echo "[stack] ${BASE_CONFIG} missing." >&2; exit 1; }
mkdir -p "${RUN_DIR}"

# ---- build the transferred init (cells 1 and 2 only) ---------------------------------------
if [ "${ARM}" != "baseline" ]; then
    if [ ! -f "${PRE_DIR}/net.ckpt" ]; then
        echo "[stack] no pretrained checkpoint at ${PRE_DIR}/net.ckpt." >&2
        echo "[stack] run STAGE 1 first:  sbatch slurm/stack_pretrain.sbatch" >&2
        echo "[stack] (cell 0, the baseline, does not need it and can run alone:" >&2
        echo "[stack]  sbatch --array=0 slurm/stack_transfer.sbatch)" >&2
        exit 1
    fi
    python scripts/stack_init.py \
        --src "${PRE_DIR}/net.ckpt" \
        --config "${BASE_CONFIG}" \
        --out "${INIT_CKPT}" \
        --step-scale "${STEP_SCALE}"
fi

# ---- stamp the config -----------------------------------------------------------------------
python - "${BASE_CONFIG}" "${GEN_CONFIG}" "${RUN_DIR}" "${ARM}" "${ANATOMY}" \
         "${INIT_CKPT}" "${SWEEP_EPOCHS}" <<'PY'
import json, os, sys
from training.common import write_config

base, out, save_dir, arm, anatomy, init_ckpt = sys.argv[1:7]
epochs = sys.argv[7] if len(sys.argv) > 7 else ""

with open(base) as f:
    cfg = json.load(f)

if cfg.get("task") != "recon":
    raise SystemExit(f"[stack] {base} has task={cfg.get('task')!r}, expected 'recon'.")
if cfg["data"]["train"].get("anatomy") != anatomy:
    raise SystemExit(
        f"[stack] {base} is anatomy={cfg['data']['train'].get('anatomy')!r} but this job "
        f"is running ANATOMY={anatomy!r}.")

params = cfg["model"]["params"]
K = params.get("K")
if K is None or isinstance(K, int):
    raise SystemExit(f"[stack] K={K!r} is a flat stack -- there is no outer stack to seed.")

paths = cfg.setdefault("paths", {})
paths["save_dir"] = save_dir
if arm == "baseline":
    paths["ckpt"] = None
else:
    if not os.path.exists(init_ckpt):
        raise SystemExit(f"[stack] {init_ckpt} was not written by scripts/stack_init.py.")
    # Weights only. stack_init.py writes step=-1 and leaves optimizer/scheduler None, so
    # train.py starts at step 0 with a fresh Adam and the whole cosine schedule.
    paths["ckpt"] = init_ckpt

cfg.setdefault("wandb", {})["id"] = None
cfg.setdefault("experiment", {})["name"] = (
    f"MGLPDSNet_stack_{anatomy}_R{cfg['mri']['R']}_{arm}")

if epochs:
    cfg["training"]["num_epochs"] = int(epochs)
    cfg["scheduler"]["params"]["T_max"] = (
        int(epochs) * cfg["training"]["steps_per_epoch"])

write_config(cfg, out)
print(f"[stack] arm={arm} K={K} ckpt={paths['ckpt']} "
      f"epochs={cfg['training']['num_epochs']} -> {out}")
PY

python train.py "${GEN_CONFIG}"
