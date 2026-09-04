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
#SBATCH --job-name=TaskXfer
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --array=0-1
#
# STAGE 2 of 2: train the HARD problem, with and without the easy-problem init.
#
#   0  baseline   fresh random init -- the current MGLPDSNet, unchanged
#   1  warm       weights from the R=PRE_R pretrain, fresh optimizer, step 0
#
# The two cells differ in exactly one config key (paths.init_ckpt), so the
# wandb curves are directly comparable.
#
# WEIGHTS ONLY, NOT A RESUME
# --------------------------
# paths.ckpt is a full resume: train.py restores the optimizer, the scheduler
# and start_step. That is wrong for a transfer -- the pretrain's Adam moments
# are calibrated to a different operator and its cosine schedule is already
# spent. paths.init_ckpt (added to train.py for this) loads the model state and
# nothing else: fresh Adam, whole schedule, step 0. Unlike the stacking flow
# this needs no checkpoint rewriting, because passing optimizer=None to
# load_ckpt is already what makes it skip them.
#
# LR_SCALE
# --------
# Default 1.0, i.e. both cells restart at the config's 5e-4 and the comparison
# is honest about wall-clock. Be aware of what that measures: Adam with zeroed
# moments at 5e-4 will move a pretrained weight a long way inside the first few
# hundred steps, so cell 1 answers "does this init survive a full-LR restart",
# not "is this init better". If cell 1 loses, rerun it with LR_SCALE=0.1 before
# concluding the pretrain did not help -- the gentler restart is a different
# arm, not a fix, so do not silently fold it into the default.
#
# Submit:      sbatch torch/task_transfer.sbatch
# One cell:    sbatch --array=1 torch/task_transfer.sbatch
# Chained:     JID=$(sbatch --parsable torch/task_pretrain.sbatch)
#              sbatch --dependency=afterok:${JID} torch/task_transfer.sbatch
# Probe first: SWEEP_EPOCHS=20 sbatch torch/task_transfer.sbatch

# ============================================================================
#  SET THE EXPERIMENT HERE  (must match torch/task_pretrain.sbatch)
# ============================================================================
ANATOMY="${ANATOMY:-brain}"          # brain | knee
RTAG="${RTAG:-R8}"
MODEL="${MODEL:-mglpds}"
PRE_R="${PRE_R:-2}"

CONFIG_ROOT="${CONFIG_ROOT:-config}"
RUN_ROOT="${RUN_ROOT:-trained_nets/task}"
SWEEP_EPOCHS="${SWEEP_EPOCHS:-}"
LR_SCALE="${LR_SCALE:-1.0}"

ARMS=(baseline warm)

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
: "${SLURM_ARRAY_TASK_ID:?must be run as an array job}"
mkdir -p logs

case "${ANATOMY}" in brain|knee) ;; *)
    echo "[task] ANATOMY=${ANATOMY} -- expected brain or knee." >&2; exit 1 ;; esac
if [ "${SLURM_ARRAY_TASK_ID}" -ge "${#ARMS[@]}" ]; then
    echo "[task] task ${SLURM_ARRAY_TASK_ID} >= ${#ARMS[@]} arms -- use --array=0-1" >&2
    exit 1
fi

ARM="${ARMS[${SLURM_ARRAY_TASK_ID}]}"
BASE_CONFIG="${CONFIG_ROOT}/${ANATOMY}/mg/${MODEL}_${RTAG}.json"
PRE_DIR="${RUN_ROOT}/${ANATOMY}/${MODEL}_${RTAG}_pretrain_R${PRE_R}"
RUN_DIR="${RUN_ROOT}/${ANATOMY}/${MODEL}_${RTAG}_${ARM}"
GEN_CONFIG="${RUN_DIR}/config.gen.json"

echo "[task] STAGE 2  task ${SLURM_ARRAY_TASK_ID}: arm=${ARM}  ${ANATOMY} ${RTAG}"
echo "       base config: ${BASE_CONFIG}"
echo "       run dir:     ${RUN_DIR}"

[ -f "${BASE_CONFIG}" ] || { echo "[task] ${BASE_CONFIG} missing." >&2; exit 1; }
mkdir -p "${RUN_DIR}"

if [ "${ARM}" = "warm" ] && [ ! -f "${PRE_DIR}/net.ckpt" ]; then
    echo "[task] no pretrained checkpoint at ${PRE_DIR}/net.ckpt." >&2
    echo "[task] run STAGE 1 first:  PRE_R=${PRE_R} sbatch torch/task_pretrain.sbatch" >&2
    echo "[task] (cell 0, the baseline, does not need it:" >&2
    echo "[task]  sbatch --array=0 torch/task_transfer.sbatch)" >&2
    exit 1
fi

python - "${BASE_CONFIG}" "${GEN_CONFIG}" "${RUN_DIR}" "${ARM}" "${ANATOMY}" \
         "${PRE_DIR}" "${SWEEP_EPOCHS}" "${LR_SCALE}" <<'PY'
import json, os, sys
from training.common import write_config

base, out, save_dir, arm, anatomy, pre_dir, epochs, lr_scale = sys.argv[1:9]
lr_scale = float(lr_scale)

with open(base) as f:
    cfg = json.load(f)

if cfg.get("task") != "recon":
    raise SystemExit(f"[task] {base} has task={cfg.get('task')!r}, expected 'recon'.")
if cfg["data"]["train"].get("anatomy") != anatomy:
    raise SystemExit(
        f"[task] {base} is anatomy={cfg['data']['train'].get('anatomy')!r} but this job "
        f"is running ANATOMY={anatomy!r}.")

paths = cfg.setdefault("paths", {})
paths["save_dir"] = save_dir
paths["ckpt"] = None
paths.pop("init_ckpt", None)

if arm == "warm":
    src_cfg_path = os.path.join(pre_dir, "config.json")
    if not os.path.exists(src_cfg_path):
        raise SystemExit(
            f"[task] {src_cfg_path} missing -- the pretrain never reached train.py's "
            f"save_config, so there is nothing to check the architecture against.")
    with open(src_cfg_path) as f:
        src_cfg = json.load(f)

    # The transfer is a strict state-dict load, so any model-block difference is
    # either a shape error at load time or, worse, a silent mismatch in meaning.
    # Name the differing keys here instead.
    a, b = cfg["model"], src_cfg["model"]
    if a["type"] != b["type"]:
        raise SystemExit(f"[task] model.type {b['type']} -> {a['type']}: not transferable.")
    diff = {k: (b["params"].get(k), a["params"].get(k))
            for k in set(a["params"]) | set(b["params"])
            if a["params"].get(k) != b["params"].get(k)}
    if diff:
        lines = "\n".join(f"           {k}: pretrain={v[0]!r}  target={v[1]!r}"
                          for k, v in sorted(diff.items()))
        raise SystemExit(
            f"[task] the pretrain and the target disagree on model.params:\n{lines}\n"
            f"       Only mri.R may differ between the two stages. Regenerate the "
            f"pretrain from the same base config.")

    src_r, dst_r = src_cfg["mri"]["R"], cfg["mri"]["R"]
    if src_r == dst_r:
        raise SystemExit(
            f"[task] the pretrain was run at R={src_r}, the same as the target. That is a "
            f"longer run, not a curriculum -- set PRE_R below {dst_r} in STAGE 1.")

    # Weights only; see the header. train.py ignores this once paths.ckpt exists,
    # so a requeued run resumes rather than re-warm-starting over its progress.
    paths["init_ckpt"] = os.path.join(pre_dir, "net.ckpt")
    print(f"[task] warm start R={src_r} -> R={dst_r}: {paths['init_ckpt']}")

if lr_scale != 1.0:
    cfg["optimizer"]["params"]["lr"] *= lr_scale

cfg.setdefault("wandb", {})["id"] = None
cfg.setdefault("experiment", {})["name"] = f"{cfg['experiment']['name']}_{arm}"

if epochs:
    cfg["training"]["num_epochs"] = int(epochs)
    cfg["scheduler"]["params"]["T_max"] = (
        int(epochs) * cfg["training"]["steps_per_epoch"])

write_config(cfg, out)
print(f"[task] arm={arm} R={cfg['mri']['R']} lr={cfg['optimizer']['params']['lr']} "
      f"init_ckpt={paths.get('init_ckpt')} epochs={cfg['training']['num_epochs']} -> {out}")
PY

python train.py "${GEN_CONFIG}"
