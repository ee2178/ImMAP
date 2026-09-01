#!/bin/bash
#
# I2SB with the paper's ADM UNet on NYUMets, both conditioning settings:
#
#     task 0   i2sb_unet_t1.json    T1 -> CT1       cond_idx []        C = 1
#     task 1   i2sb_unet_all.json   all -> CT1      cond_idx [0,1,3]   C = 4
#
# The pair isolates one thing -- what FLAIR and T2 are worth on top of the bridge itself.
# Everything else (architecture, schedule, optimizer, loss, data) is identical, so a gap is
# attributable to conditioning and nothing else. Same design as torch/i2sb_unet.sbatch.
#
# PREREQUISITES, in order:
#   1. sbatch bigpurple/nyumets_h5.sbatch          # -> $HOME/scratch/datasets/NYUMets_h5
#   2. python datasets/make_split_symlinks.py --source ../datasets/NYUMets_h5 \
#          --config config/NYUMets/split.json          # 60/30/10, group=prefix, seed 0
#      --group prefix is REQUIRED for NYUMets (group sessions by patient); the default is
#      --group none, which would split per session and leak a patient across train and val.
#      Re-running RESHUFFLES (needs --force) -- do not redo it against a checkpoint you
#      intend to keep comparing, or the val set changes underneath you.
#   3. python -m scripts.calibrate_nyumets_bridge --root ../datasets/NYUMets_h5_train --scale 3.0
#      -> confirm beta_max; it is inherited from BraTS and NYUMets is a different cohort.
#
# beta_max lives in BOTH cfg["i2sb"] AND cfg["model"]["params"]: SBUnet carries its own copy
# of the schedule to invert std_fwd back to the step index, and train_i2sb calls
# assert_schedule_matches() at startup. If you change one, change both or the job dies.
#
# Submit:    sbatch bigpurple/i2sb_unet_nyumets.sbatch
# One only:  sbatch --array=0 bigpurple/i2sb_unet_nyumets.sbatch

#SBATCH --partition=a100_short
#SBATCH --job-name=i2sb-unet-nyumets
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# a100_short caps at 3 days; the BraTS budget is 1500 x 200 = 300k steps. If that does not
# fit, resume from the checkpoint or move to a100_long (28d, 19 nodes).
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

CONFIGS=(config/NYUMets/i2sb_unet_t1.json config/NYUMets/i2sb_unet_all.json)

TASK=${SLURM_ARRAY_TASK_ID:-0}
if [ "$TASK" -ge "${#CONFIGS[@]}" ]; then
    echo "array index $TASK >= ${#CONFIGS[@]} configs; fix --array" >&2
    exit 1
fi
CFG="${CONFIGS[$TASK]}"

# The config's data roots are RELATIVE TO THE REPO ROOT (../datasets/NYUMets_h5_{train,val},
# i.e. $HOME/scratch/datasets/... given IMMAP_ROOT=$HOME/scratch/ImMAP) -- the same convention
# the raw ../datasets/NYUMets symlink uses. They only resolve because of the cd above, so
# check them here rather than failing 20 minutes into a queued job.
python - "$CFG" <<'PY' || exit 1
import json, os, sys
cfg = json.load(open(sys.argv[1]))
bad = [(s, cfg["data"][s]["root"]) for s in ("train", "val")
       if not os.path.isdir(cfg["data"][s]["root"])]
for s, r in bad:
    print(f"  !! data.{s}.root does not exist: {r}  (cwd {os.getcwd()})", file=sys.stderr)
if bad:
    print("  run the h5 build then the split, or fix the 'root' fields in "
          f"{sys.argv[1]}.", file=sys.stderr)
    sys.exit(1)
PY

echo "host   : $(hostname)"
echo "gpu    : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "task $TASK -> $CFG"

python train.py "$CFG"
