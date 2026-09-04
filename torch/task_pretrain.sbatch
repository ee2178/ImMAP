#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=torch_pr_89_tandon_advanced
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --job-name=TaskPre
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#
# STAGE 1 of 2: pretrain the FULL model on an EASIER inverse problem.
#
# Curriculum on the TASK, not on depth. torch/stack_pretrain.sbatch changes
# K_outer and needs scripts/stack_init.py to reshape the checkpoint; this
# changes nothing about the architecture, so the transfer in stage 2 is a plain
# state-dict load with no surgery at all.
#
# THE ONLY CHANGE IS `mri.R`. Default 2 instead of 8. Same anatomy, same coils,
# same preprocessing, same loss -- the operator gets easier and nothing else
# moves, which is what makes the learned dictionary transferable rather than
# merely reusable. The model itself never sees R: it lives entirely in the mask
# inside E, so every parameter tensor keeps its shape and meaning.
#
# WHY THE NOISE RANGE IS NOT ALSO MADE EASIER
# -------------------------------------------
# Tempting, and wrong. `degrees=1` makes every threshold and step size an affine
# function of sigma -- lambda(sigma) = w0 + w1*sigma -- FITTED over whatever
# range training samples. Pretrain on sigma ~ U[0, 0.002] and the deep run's
# sigma = 0.01 is pure extrapolation on a line that was never constrained there.
# The sampling gets easier; the noise model does not.
#
# STAGE 2 is torch/task_transfer.sbatch. Chain them:
#
#   JID=$(sbatch --parsable torch/task_pretrain.sbatch)
#   sbatch --dependency=afterok:${JID} torch/task_transfer.sbatch
#
# ============================================================================
#  SET THE EXPERIMENT HERE  (must match torch/task_transfer.sbatch)
# ============================================================================
ANATOMY="${ANATOMY:-brain}"          # brain | knee
RTAG="${RTAG:-R8}"                   # the HARD problem, i.e. the stage-2 target
MODEL="${MODEL:-mglpds}"
PRE_R="${PRE_R:-2}"                  # the EASY problem's acceleration
PRE_STEPS="${PRE_STEPS:-100000}"

CONFIG_ROOT="${CONFIG_ROOT:-config}"
RUN_ROOT="${RUN_ROOT:-trained_nets/task}"

source ~/.bashrc
conda activate gcdl

for _c in "${IMMAP_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" "$HOME/scratch/ee2178/ImMAP"; do
    if [ -n "${_c}" ] && [ -f "${_c}/train.py" ]; then IMMAP_ROOT="${_c}"; break; fi
done
if [ ! -f "${IMMAP_ROOT:-}/train.py" ]; then
    echo "[task] cannot locate the ImMAP checkout." >&2; exit 1
fi
cd "${IMMAP_ROOT}"
set -eo pipefail
mkdir -p logs

case "${ANATOMY}" in brain|knee) ;; *)
    echo "[task] ANATOMY=${ANATOMY} -- expected 'brain' or 'knee'." >&2; exit 1 ;; esac

BASE_CONFIG="${CONFIG_ROOT}/${ANATOMY}/mg/${MODEL}_${RTAG}.json"
RUN_DIR="${RUN_ROOT}/${ANATOMY}/${MODEL}_${RTAG}_pretrain_R${PRE_R}"
GEN_CONFIG="${RUN_DIR}/config.gen.json"

echo "[task] STAGE 1  pretrain on R=${PRE_R}  (target is ${RTAG})  ${ANATOMY}"
echo "       base config: ${BASE_CONFIG}"
echo "       run dir:     ${RUN_DIR}"

[ -f "${BASE_CONFIG}" ] || { echo "[task] ${BASE_CONFIG} missing." >&2; exit 1; }
mkdir -p "${RUN_DIR}"

python - "${BASE_CONFIG}" "${GEN_CONFIG}" "${RUN_DIR}" "${ANATOMY}" "${PRE_R}" "${PRE_STEPS}" <<'PY'
import json, sys
from training.common import write_config

base, out, save_dir, anatomy, pre_r, pre_steps = sys.argv[1:7]
pre_r, pre_steps = int(pre_r), int(pre_steps)

with open(base) as f:
    cfg = json.load(f)

if cfg.get("task") != "recon":
    raise SystemExit(f"[task] {base} has task={cfg.get('task')!r}, expected 'recon'.")
if cfg["data"]["train"].get("anatomy") != anatomy:
    raise SystemExit(
        f"[task] {base} is anatomy={cfg['data']['train'].get('anatomy')!r} but this job "
        f"is running ANATOMY={anatomy!r}.")

hard_r = int(cfg["mri"]["R"])
if pre_r >= hard_r:
    raise SystemExit(
        f"[task] PRE_R={pre_r} is not easier than the target R={hard_r}. The point of the "
        f"pretrain is a better-conditioned operator; pick PRE_R < {hard_r}.")

# The ONLY change. Everything else -- noise_std, loss, slices, scale_fac, the
# whole model block -- is inherited verbatim, which is what keeps the parameter
# tensors interchangeable and the sigma polynomials in range.
cfg["mri"]["R"] = pre_r

spe = cfg["training"]["steps_per_epoch"]
cfg["training"]["num_epochs"] = max(1, pre_steps // spe)
cfg["scheduler"]["params"]["T_max"] = cfg["training"]["num_epochs"] * spe

paths = cfg.setdefault("paths", {})
paths["save_dir"] = save_dir
paths["ckpt"] = None
paths.pop("init_ckpt", None)
cfg.setdefault("wandb", {})["id"] = None
cfg.setdefault("experiment", {})["name"] = (
    f"{cfg['experiment']['name']}_pretrainR{pre_r}")

write_config(cfg, out)
print(f"[task] R {hard_r} -> {pre_r}, noise_std={cfg['training']['noise_std']} (unchanged), "
      f"{cfg['training']['num_epochs']} x {spe} = {cfg['training']['num_epochs']*spe} steps")
PY

python train.py "${GEN_CONFIG}"

echo "[task] STAGE 1 done. Checkpoint: ${RUN_DIR}/net.ckpt"
echo "[task] now:  sbatch torch/task_transfer.sbatch"
