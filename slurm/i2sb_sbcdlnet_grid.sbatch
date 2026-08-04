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
#SBATCH --job-name=SBCDLNet-grid
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --array=0-15          # MUST equal len(BETAS)*len(ET_WEIGHTS) - 1; checked at runtime
#
# 2-D grid over the I2SB schedule scale (beta_max) and the enhancing-tumor loss weight
# (et_weight), for the SBCDLNet regressor. The array index is flattened row-major:
#
#     task = beta_index * len(ET_WEIGHTS) + et_index
#
# so consecutive tasks walk et_weight within one beta_max. Each cell gets its own
# trained_nets/.../beta_<b>_et_<w> folder, its own generated config, and its own wandb run.
#
# TWO THINGS THIS SCRIPT HAS TO DO THAT THE 1-D SWEEPS DID NOT:
#
#   1. SBCDLNet carries its OWN copy of the bridge schedule (it inverts std_fwd to recover
#      mu_0 / mu_1 / sigma_eff), so beta_max lives in BOTH cfg["i2sb"] and
#      cfg["model"]["params"]. Overriding only the first is a silent wrong-schedule run --
#      train_i2sb now calls net.assert_schedule_matches() and dies at startup, but this script
#      syncs both so it never gets that far.
#
#   2. et_weight != 1 needs the loader's et_mask, and et_mask DROPS subjects whose h5 has no
#      'et'. If some cells had it on and others off they would train on different subject sets,
#      which would confound the whole grid -- so et_mask is forced ON for every cell, including
#      the et_weight = 1 row (where the weighting is a no-op but the data is identical).
#
# PREREQUISITE: every subject needs an 'et' dataset in its h5, or the loader raises at startup.
# Check first:  python diagnose_et.py <train_root>   and backfill with
#               python preprocessing/cmap.py --config config/BraTS/cmap.yaml --add-seg-only
#
# Submit:  sbatch slurm/i2sb_sbcdlnet_grid.sbatch
# COST: 16 full trainings. Probe first -- set SWEEP_EPOCHS small, look at the surface, then
# full-train the interesting corner.

# ---- grid knobs (edit these) --------------------------------------------------------------
BASE_CONFIG="config/BraTS/i2sb_sbcdlnet.json"
SWEEP_NAME="I2SB_SBCDLNet_grid"
BETAS=(0.10 0.30 0.50 0.70)          # i2sb schedule scale; mirrored into model.params
ET_WEIGHTS=(1 50 150 330)            # PER-PIXEL tumor weight. ET is ~0.3% of a slice, so 50 puts
                                     # ~13% of the objective on it and 330 ~50%. 1 = off.
SWEEP_EPOCHS=""                      # empty = the config's num_epochs; set e.g. 50 to probe
# -------------------------------------------------------------------------------------------

source ~/.bashrc
conda activate gcdl
cd ~/scratch/ee2178/ImMAP

set -eo pipefail
: "${SLURM_ARRAY_TASK_ID:?must be run as an array job (sbatch sets SLURM_ARRAY_TASK_ID)}"

N_B=${#BETAS[@]}
N_E=${#ET_WEIGHTS[@]}
N_TOTAL=$(( N_B * N_E ))
if [ "${SLURM_ARRAY_TASK_ID}" -ge "${N_TOTAL}" ]; then
    echo "[grid] task ${SLURM_ARRAY_TASK_ID} >= ${N_B}x${N_E}=${N_TOTAL} cells -- fix --array "
    echo "       (should be 0-$(( N_TOTAL - 1 ))) or add grid values"
    exit 1
fi

BI=$(( SLURM_ARRAY_TASK_ID / N_E ))
EI=$(( SLURM_ARRAY_TASK_ID % N_E ))
BETA="${BETAS[${BI}]}"
ETW="${ET_WEIGHTS[${EI}]}"

TAG="beta_${BETA}_et_${ETW}"
RUN_DIR="trained_nets/brats/${SWEEP_NAME}/${TAG}"
GEN_CONFIG="${RUN_DIR}/config.gen.json"
mkdir -p "${RUN_DIR}" logs

echo "[grid] task ${SLURM_ARRAY_TASK_ID}/$(( N_TOTAL - 1 )): beta_max=${BETA} (row ${BI}) "
echo "       et_weight=${ETW} (col ${EI})  ->  ${RUN_DIR}"

python - "${BASE_CONFIG}" "${GEN_CONFIG}" "${BETA}" "${ETW}" "${RUN_DIR}" \
         "${SWEEP_NAME}_${TAG}" "${SWEEP_EPOCHS}" <<'PY'
import json, sys

args = sys.argv[1:]
base, out, beta_max, et_weight, save_dir, name = args[:6]
epochs = args[6] if len(args) > 6 else ""      # optional; robust if the empty arg is dropped

with open(base) as f:
    cfg = json.load(f)

if cfg.get("task") != "i2sb":
    raise SystemExit(f"[grid] BASE_CONFIG task is {cfg.get('task')!r}, expected 'i2sb'.")

beta_max, et_weight = float(beta_max), float(et_weight)

# ---- 1. schedule scale, in BOTH places it lives ----
cfg["i2sb"]["beta_max"] = beta_max
mparams = cfg["model"]["params"]
if "beta_max" in mparams:                      # SBCDLNet-style nets holding their own schedule
    mparams["beta_max"] = beta_max
for key in ("kind", "n_points", "tau"):        # the rest must already agree; don't guess
    if key in mparams and mparams[key] != cfg["i2sb"].get(key):
        raise SystemExit(
            f"[grid] model.params.{key}={mparams[key]!r} != i2sb.{key}="
            f"{cfg['i2sb'].get(key)!r} in {base}. Fix the base config: the net recovers the "
            f"bridge coefficients by inverting this schedule, so a mismatch trains on nonsense.")

# ---- 2. ET weighting, and the loader flag it depends on ----
cfg["training"]["et_weight"] = et_weight
for split in ("train", "val"):
    cfg["data"][split]["et_mask"] = True       # ON for every cell, so all cells share a dataset

POINTWISE = ("complex-mse", "magnitude-mse", "magnitude-l1")
lt = cfg["training"].get("loss_type", "complex-mse")
if et_weight != 1 and lt not in POINTWISE:
    raise SystemExit(
        f"[grid] et_weight={et_weight} needs a per-pixel loss, but loss_type={lt!r}. "
        f"Use one of {POINTWISE} in the base config.")

# ---- 3. SBCDLNet needs the bridge prior among the conditioning channels ----
if cfg["model"]["type"] == "SBCDLNet":
    for split in ("train", "val"):
        d = cfg["data"][split]
        if d["x1_idx"] not in d["cond_idx"]:
            raise SystemExit(
                f"[grid] data.{split}: x1_idx={d['x1_idx']} is not in cond_idx={d['cond_idx']}. "
                f"SBCDLNet reads the bridge prior out of the conditioning stack.")
        want = d["cond_idx"].index(d["x1_idx"])
        if mparams.get("prior_idx") != want:
            raise SystemExit(
                f"[grid] model.params.prior_idx={mparams.get('prior_idx')} but "
                f"cond_idx.index(x1_idx)={want} for data.{split}.")
        if mparams.get("C") != cfg["i2sb"].get("target_channels", 1) + len(d["cond_idx"]):
            raise SystemExit(
                f"[grid] model.params.C={mparams.get('C')} != 1 + len(cond_idx) for data.{split}.")

# ---- 4. run bookkeeping ----
paths = cfg.setdefault("paths", {})
paths["save_dir"] = save_dir
paths["ckpt"] = None                            # fresh run; train.py rewrites this to net.ckpt
cfg.setdefault("experiment", {})["name"] = name
cfg.setdefault("wandb", {})["id"] = None        # a new wandb run per cell
if epochs:
    cfg["training"]["num_epochs"] = int(epochs)

with open(out, "w") as f:
    json.dump(cfg, f, indent=4)

print(f"[grid] beta_max={beta_max} et_weight={et_weight} -> {out}\n"
      f"       model={cfg['model']['type']} C={mparams.get('C')} "
      f"prior_idx={mparams.get('prior_idx')} et_mask=True "
      f"num_epochs={cfg['training'].get('num_epochs')}")
PY

python train.py "${GEN_CONFIG}"
