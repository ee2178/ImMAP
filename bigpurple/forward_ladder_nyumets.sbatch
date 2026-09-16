#!/bin/bash
#
# Fit the CT1 -> T1 forward-operator ladder (scripts/fit_forward_ladder.py) and time each E's
# VJP against the bridge regressors. One A100: all rungs train together on one data stream, and
# the timing only means something on the GPU the bridge nets actually run on.
#
#   sbatch bigpurple/forward_ladder_nyumets.sbatch
#   CFG=config/NYUMets/forward_ladder_cond.json sbatch bigpurple/forward_ladder_nyumets.sbatch
#
#   forward_ladder.json       UNet size sweep, E(CT1) only
#   forward_ladder_cond.json  3 UNet sizes x {E(CT1), E(CT1, T2, FLAIR), E(T2, FLAIR)}; read
#                             x_swap_ratio: ~1 means the conditioned E ignores CT1
# The two are independent runs (own wandb run, own save_dir) and can go in parallel.
#
# PREREQUISITE: the train/val split (../datasets/NYUMets_h5_{train,val}). This script does NOT
# split -- run i2sb_unet_nyumets.sbatch or the array once first, or chain after it with
#   sbatch --dependency=after:<jobid> ...
# (`after`, not `afterok`: the split happens in the first minutes of a multi-day training job).
#
# Results: wandb (project schrodinger-bridge, run NYUMets_forward_ladder_unet_CT1_to_T1: per-rung
# curves, val panels, a `summary` table) and trained_nets/nyumets/forward_ladder_unet_CT1_to_T1/
# (E_<rung>.pt, loadable with models.forward_ops.load_forward_op, and ladder_results.json).

#SBATCH --partition=radiology
#SBATCH --job-name=forward-ladder-nyumets
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=EE2178@nyulangone.org
#SBATCH --chdir=/gpfs/scratch/ee2178/ImMAP
#SBATCH --output=/gpfs/scratch/ee2178/ImMAP/bigpurple/logs/%x_%j.out
#SBATCH --error=/gpfs/scratch/ee2178/ImMAP/bigpurple/logs/%x_%j.err

# No `set -euo pipefail` -- it breaks conda activate (unset $PS1 / $CONDA_SHLVL under -u).

# --- conda ------------------------------------------------------------------------------
# module load anaconda3/gpu/2023.09          # uncomment if conda is not already on PATH
CONDA_BASE="$(conda info --base 2>/dev/null)"
if [ -z "$CONDA_BASE" ]; then
    echo "conda not on PATH -- uncomment the module load line above" >&2
    exit 1
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate env || { echo "conda activate env failed" >&2; exit 1; }

export PYTHONNOUSERSITE=1

IMMAP_ROOT="${IMMAP_ROOT:-$HOME/scratch/ImMAP}"
cd "$IMMAP_ROOT" || { echo "no such dir: $IMMAP_ROOT" >&2; exit 1; }

CFG="${CFG:-config/NYUMets/forward_ladder.json}"
if [ ! -f "$CFG" ]; then
    echo "config not found: $CFG (cwd $(pwd))" >&2
    exit 1
fi

# --- GPU preflight (see i2sb_unet_nyumets.sbatch for why nvidia-smi is not enough) ---------
PREFLIGHT_TRIES="${PREFLIGHT_TRIES:-6}"
PREFLIGHT_WAIT="${PREFLIGHT_WAIT:-30}"

gpu_ok=0
for try in $(seq 1 "$PREFLIGHT_TRIES"); do
    if python - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    print("no CUDA device visible -- was --gres honoured?", file=sys.stderr)
    sys.exit(1)
try:
    torch.zeros(1, device="cuda")
except Exception as err:
    print(f"context creation failed: {err}", file=sys.stderr)
    sys.exit(1)
print(f"gpu    : {torch.cuda.get_device_name(0)}  (context OK)")
PY
    then
        gpu_ok=1
        break
    fi
    echo "PREFLIGHT: attempt $try/$PREFLIGHT_TRIES failed; retrying in ${PREFLIGHT_WAIT}s" >&2
    sleep "$PREFLIGHT_WAIT"
done

if [ "$gpu_ok" -ne 1 ]; then
    echo "PREFLIGHT FAILED after $PREFLIGHT_TRIES attempts on $(hostname) (job $SLURM_JOB_ID)" >&2
    nvidia-smi --query-gpu=index,name,compute_mode,memory.used --format=csv 2>&1 >&2
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>&1 >&2
    exit 1
fi

# --- data roots (relative to the repo root; resolve because of the cd above) ---------------
python - "$CFG" <<'PY' || exit 1
import json, os, sys
cfg = json.load(open(sys.argv[1]))
bad = [(s, cfg["data"][s]["root"]) for s in ("train", "val")
       if not os.path.isdir(cfg["data"][s]["root"])]
for s, r in bad:
    print(f"  !! data.{s}.root does not exist: {r}  (cwd {os.getcwd()})", file=sys.stderr)
if bad:
    print("  run the split first (i2sb_unet_nyumets.sbatch does it with SPLIT=auto)", file=sys.stderr)
    sys.exit(1)
PY

echo "host   : $(hostname)"
echo "config : $CFG"

python -m scripts.fit_forward_ladder "$CFG"
