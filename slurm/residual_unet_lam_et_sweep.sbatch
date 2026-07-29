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
#SBATCH --job-name=ResUNet-lamET-sweep
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --array=0-5          # ONE TASK PER LAMBDA BELOW -- keep 0-(len(LAMS)-1) in sync with LAMS
#
# et_weight sweep over the RESIDUAL U-NET task (config/BraTS/residual_synthesis_unet.json:
# task=synthesis, Unet2D, residual_mode=true). The counterpart of
# slurm/synth_lam_sweep.sbatch, same grid, so the two architectures are directly
# comparable arm for arm.
#
# Six arms: et_weight = 1 (CONTROL, the unweighted loss) plus five log-spaced to ~34x.
#
#     et_weight:  1      1.3     2.1       4.3     11.6     34.4
#     share:      0%     0.2%    0.7%      2.2%    6.6%     18%
#
# et_weight IS the per-ET-pixel multiplier, so these are the same six arms as before,
# restated in the units the loss now takes (old lam_et * 334 + 1).
#
# ARM 0 IS ALSO THE BASELINE YOU LOST. The un-weighted residual U-Net that used to live in
# trained_nets/brats/Residual_Unet_T1ce was overwritten when the ET run was launched into
# that directory. et_weight=1 is exactly the unweighted loss, so arm 0 retrains exactly
# that control -- and under the CURRENT code and configs, which the old one was not
# (scales went 2.0 -> 1.0, plus the dataset transform fix and the ET-mask filter). It is
# what makes "detail outside the ET mask looks worse" a statement with a referent, and it
# restores the control/treatment pairing in notebooks/synthesis_testbench.ipynb section 9.
#
# What "detail outside ET" means here: in residual mode the net predicts T1ce - T1, so
# outside the tumor the target is the small edge/texture difference between the two
# contrasts. If the residual collapses toward zero the reconstruction becomes T1 verbatim --
# plausible-looking, but with exactly that detail missing. Watch val/resid_ratio (rms of the
# predicted residual over rms of the true one): ~1 is healthy, drifting to 0 is the collapse.
#
# Submit:   sbatch slurm/residual_unet_et_weight_sweep.sbatch
# Probe:    set SWEEP_EPOCHS below (e.g. 30) to rank cheaply before committing 48h x 6.
#           T_max is rescaled to match so the cosine schedule still completes.
#
# Note on this config's guards: backtrack_thresh is 0.5 against a magnitude-l1 residual loss
# of order 0.1, so backtracking is unlikely to fire here (unlike a tighter threshold, it is
# a loose guard, not a trigger-happy one). If an arm nonetheless stalls, check train/lr.

# ---- sweep knobs (edit these) ------------------------------------------------------------------
BASE_CONFIG="config/BraTS/residual_synthesis_unet.json"   # task=synthesis, residual_mode=true
SWEEP_NAME="Residual_Unet_et_weight_sweep"                   # subfolder under trained_nets/brats/
LAMS=(1 1.3 2.1 4.3 11.6 34.4)                            # 1 = control (unweighted), then log-spaced
                                                          # LEN MUST MATCH --array
SWEEP_EPOCHS=""                                           # empty = use the config's num_epochs
# ------------------------------------------------------------------------------------------------

# Environment (mirrors slurm/residual_synthesis_unet_et.sbatch)
source ~/.bashrc
conda activate gcdl
cd ~/scratch/ee2178/ImMAP

set -eo pipefail
: "${SLURM_ARRAY_TASK_ID:?must be run as an array job (sbatch sets SLURM_ARRAY_TASK_ID)}"

if [ "${SLURM_ARRAY_TASK_ID}" -ge "${#LAMS[@]}" ]; then
    echo "[sweep] task ${SLURM_ARRAY_TASK_ID} >= ${#LAMS[@]} weights -- shrink --array or add values"
    exit 1
fi
LAM="${LAMS[${SLURM_ARRAY_TASK_ID}]}"
TAG="etw_${LAM}"
RUN_DIR="trained_nets/brats/${SWEEP_NAME}/${TAG}"
GEN_CONFIG="${RUN_DIR}/config.gen.json"

mkdir -p "${RUN_DIR}" logs

# Write this task's config: override et_weight + savenames, fresh wandb run, optional epoch cap.
python - "${BASE_CONFIG}" "${GEN_CONFIG}" "${LAM}" "${RUN_DIR}" "${SWEEP_NAME}_${TAG}" "${SWEEP_EPOCHS}" <<'PY'
import json, sys
from training.common import write_config      # yaml-safe writer; see note below

args = sys.argv[1:]
base, out, lam, save_dir, name = args[:5]
epochs = args[5] if len(args) > 5 else ""     # optional; robust if the empty arg is dropped

with open(base) as f:
    cfg = json.load(f)

tr = cfg.setdefault("training", {})
if cfg.get("task") != "synthesis":
    raise SystemExit(f"[sweep] BASE_CONFIG task is {cfg.get('task')!r}, expected 'synthesis'.")
if not tr.get("residual_mode", False):
    raise SystemExit("[sweep] BASE_CONFIG has residual_mode=false; this sweep is the RESIDUAL task.")

tr["et_weight"] = float(lam)

# et_weight is a no-op unless the loader actually returns an ET mask. Required even for the
# et_weight=1 control, so that arm stays comparable with the others.
for split in ("train", "val"):
    if not cfg["data"][split].get("et_mask", False):
        raise SystemExit(f"[sweep] data.{split}.et_mask is not true; the ET term would be a no-op.")

# The residual anchor must be a real input channel; an out-of-range index would slice an
# EMPTY tensor and surface as a NaN loss hours later instead of failing here.
src_idx = int(tr.get("residual_src_idx", 0))
n_in = len(cfg["data"]["train"]["input_idx"])
if not 0 <= src_idx < n_in:
    raise SystemExit(f"[sweep] residual_src_idx={src_idx} out of range for {n_in} input channels.")

paths = cfg.setdefault("paths", {}); paths["save_dir"] = save_dir; paths["ckpt"] = None
cfg.setdefault("experiment", {})["name"] = name
cfg.setdefault("wandb", {})["id"] = None      # fresh run id per lambda

if epochs:
    tr["num_epochs"] = int(epochs)
    # Rescale the cosine horizon with the epoch cap. Left alone, a shortened probe would
    # traverse only part of the curve and every arm would end at a high LR -- which is not
    # what the full run does, so the ranking would not transfer.
    sched = cfg.get("scheduler", {})
    if sched.get("type") == "CosineAnnealingLR":
        sched["params"]["T_max"] = int(tr["num_epochs"]) * int(tr["steps_per_epoch"])

# write_config, NOT json.dump: train.py reads configs with yaml.safe_load, where a float
# needs both a decimal point and a signed exponent. json.dump writes eta_min as `1e-06`,
# which yaml returns as the STRING '1e-06' and CosineAnnealingLR then dies on. write_config
# canonicalizes and verifies with the same loader train.py uses.
write_config(cfg, out)
print(f"[sweep] et_weight={lam} -> {out}  (save_dir={save_dir}, "
      f"num_epochs={tr.get('num_epochs')}, T_max={cfg.get('scheduler',{}).get('params',{}).get('T_max')}, "
      f"residual_src_idx={src_idx})")
PY

echo "[sweep] array task ${SLURM_ARRAY_TASK_ID}: et_weight=${LAM}  ->  ${RUN_DIR}"
python train.py "${GEN_CONFIG}"

