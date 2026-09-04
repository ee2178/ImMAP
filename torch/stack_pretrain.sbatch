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
#SBATCH --job-name=StackPre
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#
# STAGE 1 of 2: pretrain a SINGLE outer V-cycle.
#
# Same task, same data, same operator as the full model -- only K_outer differs
# (1 instead of 6). Nothing about the physics, the noise range or the
# preprocessing changes, so the learned step sizes and thresholds are calibrated
# to the operator the deep model will actually see.
#
# Depth costs super-linearly, so this is cheap: measured fwd+bwd at M=64, 64x64,
# iters=[4,4,6], K_outer=1 runs at 0.124x the cost of K_outer=6 -- BETTER than
# the naive 1/6. 100k steps here is roughly 12k K=6-equivalent steps.
#
# STAGE 2 is torch/stack_transfer.sbatch, which seeds all six outer slots from
# whatever this produces. Chain them:
#
#   JID=$(sbatch --parsable torch/stack_pretrain.sbatch)
#   sbatch --dependency=afterok:${JID} torch/stack_transfer.sbatch
#
# Or run this alone first and submit stage 2 by hand once you have looked at the
# curve -- there is no point seeding six slots from a pretrain that went nowhere.

# ============================================================================
#  SET THE EXPERIMENT HERE  (must match torch/stack_transfer.sbatch)
# ============================================================================
ANATOMY="${ANATOMY:-brain}"          # brain | knee
RTAG="${RTAG:-R8}"
MODEL="${MODEL:-mglpds}"
PRE_STEPS="${PRE_STEPS:-100000}"     # gradient steps for the single V-cycle

CONFIG_ROOT="${CONFIG_ROOT:-config}"
RUN_ROOT="${RUN_ROOT:-trained_nets/stack}"

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
mkdir -p logs

case "${ANATOMY}" in brain|knee) ;; *)
    echo "[stack] ANATOMY=${ANATOMY} -- expected 'brain' or 'knee'." >&2; exit 1 ;; esac

BASE_CONFIG="${CONFIG_ROOT}/${ANATOMY}/mg/${MODEL}_${RTAG}.json"
RUN_DIR="${RUN_ROOT}/${ANATOMY}/${MODEL}_${RTAG}_pretrain_K1"
GEN_CONFIG="${RUN_DIR}/config.gen.json"

echo "[stack] STAGE 1  pretrain K_outer=1   ${ANATOMY} ${RTAG}  ${PRE_STEPS} steps"
echo "        base config: ${BASE_CONFIG}"
echo "        run dir:     ${RUN_DIR}"

[ -f "${BASE_CONFIG}" ] || { echo "[stack] ${BASE_CONFIG} missing." >&2; exit 1; }
mkdir -p "${RUN_DIR}"

python - "${BASE_CONFIG}" "${GEN_CONFIG}" "${RUN_DIR}" "${ANATOMY}" "${PRE_STEPS}" <<'PY'
import json, sys
from training.common import write_config

base, out, save_dir, anatomy, pre_steps = sys.argv[1:6]
pre_steps = int(pre_steps)

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
    raise SystemExit(f"[stack] K={K!r} is a flat stack -- nothing to pretrain a V-cycle from.")
K_outer, iters = int(K[0]), list(K[1])

# The ONLY model change: one outer V-cycle instead of K_outer.
params["K"] = [1, iters]

# Hold steps_per_epoch so the wandb step axis matches the deep runs, and set the
# epoch count to hit PRE_STEPS. T_max follows, so the cosine schedule completes
# within the pretrain rather than being truncated.
spe = cfg["training"]["steps_per_epoch"]
cfg["training"]["num_epochs"] = max(1, pre_steps // spe)
cfg["scheduler"]["params"]["T_max"] = cfg["training"]["num_epochs"] * spe

paths = cfg.setdefault("paths", {})
paths["save_dir"] = save_dir
paths["ckpt"] = None
cfg.setdefault("wandb", {})["id"] = None
cfg.setdefault("experiment", {})["name"] = (
    f"MGLPDSNet_stack_{anatomy}_R{cfg['mri']['R']}_pretrainK1")

write_config(cfg, out)
print(f"[stack] K_outer {K_outer} -> 1, iters={iters}, "
      f"{cfg['training']['num_epochs']} x {spe} = "
      f"{cfg['training']['num_epochs']*spe} steps -> {out}")
PY

python train.py "${GEN_CONFIG}"

echo "[stack] STAGE 1 done. Checkpoint: ${RUN_DIR}/net.ckpt"
echo "[stack] now:  sbatch torch/stack_transfer.sbatch"
