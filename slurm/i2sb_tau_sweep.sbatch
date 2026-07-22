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
#SBATCH --job-name=I2SB-tausweep
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --array=0-7          # ONE TASK PER TAU BELOW -- keep this 0-(len(TAUS)-1) in sync with TAUS
#
# Image-domain I2SB tau sweep. Each array task copies an image-domain i2sb config, overrides
# i2sb.tau (+ the save_dir / experiment name), and trains it -- so every tau lands in its own
# trained_nets/.../tau_<v> folder and its own wandb run. I2SB's default tau (~0.19, noise-matched
# to the paper's beta_max=0.3) is tuned for the paper's easy restoration tasks; contrast synthesis
# is a much harder map, so a LARGER peak bridge noise (more stochasticity) may be needed -- hence
# the sweep over [0.1, 1.0].
#
# Submit:   sbatch slurm/i2sb_tau_sweep.sbatch
# Probe first, then full-train the winner: set SWEEP_EPOCHS below to something small (e.g. 50).

# ---- sweep knobs (edit these) ------------------------------------------------------------------
BASE_CONFIG="config/BraTS/i2sb.json"   # image-domain I2SB config (task must be "i2sb"); swap for
                                       # i2sb_vgg.json / i2sb_synth.json to sweep the conditional runs
SWEEP_NAME="I2SB_tau_sweep"            # subfolder under trained_nets/brats/ + wandb name prefix
TAUS=(0.10 0.20 0.30 0.40 0.50 0.60 0.80 1.00)   # peak bridge-noise std; LEN MUST MATCH --array
SWEEP_EPOCHS=""                        # empty = use the config's num_epochs; set e.g. 50 to probe cheaply
# ------------------------------------------------------------------------------------------------

# Environment (mirrors slurm/i2sb.sbatch)
source ~/.bashrc
conda activate gcdl
cd ~/scratch/ee2178/ImMAP

set -eo pipefail
: "${SLURM_ARRAY_TASK_ID:?must be run as an array job (sbatch sets SLURM_ARRAY_TASK_ID)}"

if [ "${SLURM_ARRAY_TASK_ID}" -ge "${#TAUS[@]}" ]; then
    echo "[sweep] task ${SLURM_ARRAY_TASK_ID} >= ${#TAUS[@]} taus -- shrink --array or add taus"
    exit 1
fi
TAU="${TAUS[${SLURM_ARRAY_TASK_ID}]}"
TAG="tau_${TAU}"
RUN_DIR="trained_nets/brats/${SWEEP_NAME}/${TAG}"
GEN_CONFIG="${RUN_DIR}/config.gen.json"

mkdir -p "${RUN_DIR}" logs

# Write this task's config: override tau + savenames, fresh wandb run, optional epoch cap.
python - "${BASE_CONFIG}" "${GEN_CONFIG}" "${TAU}" "${RUN_DIR}" "${SWEEP_NAME}_${TAG}" "${SWEEP_EPOCHS}" <<'PY'
import json, sys
args = sys.argv[1:]
base, out, tau, save_dir, name = args[:5]
epochs = args[5] if len(args) > 5 else ""     # optional; robust if the empty arg is dropped
with open(base) as f:
    cfg = json.load(f)
if cfg.get("task") != "i2sb":
    raise SystemExit(f"[sweep] BASE_CONFIG task is {cfg.get('task')!r}, expected 'i2sb' (image-domain "
                     f"bridge). Point BASE_CONFIG at an i2sb config, or relax this check for latent.")
cfg["i2sb"]["tau"]  = float(tau)
cfg["i2sb"]["kind"] = "brownian"                  # tau only bites the Brownian schedule
paths = cfg.setdefault("paths", {}); paths["save_dir"] = save_dir; paths["ckpt"] = None
cfg.setdefault("experiment", {})["name"] = name
cfg.setdefault("wandb", {})["id"] = None          # new run id per tau
if epochs:
    cfg.setdefault("training", {})["num_epochs"] = int(epochs)
with open(out, "w") as f:
    json.dump(cfg, f, indent=4)
print(f"[sweep] tau={tau} -> {out}  (save_dir={save_dir}, "
      f"num_epochs={cfg.get('training', {}).get('num_epochs')})")
PY

echo "[sweep] array task ${SLURM_ARRAY_TASK_ID}: tau=${TAU}  ->  ${RUN_DIR}"
python train.py "${GEN_CONFIG}"
