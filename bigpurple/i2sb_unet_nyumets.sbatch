#!/bin/bash
#
# I2SB with the paper's ADM UNet on NYUMets: all contrasts -> CT1.
#
#     config/NYUMets/i2sb_unet_all.json     cond_idx [0,1,3] (FLAIR, T1, T2)   C = 4
#
# THE MODEL IS THE PAPER'S, NOT "A UNET". models/sb_unet.py is a port of ADM /
# guided-diffusion's UNet, which is what I2SB uses unmodified, conditioned on the discrete
# bridge POSITION through a sinusoidal embedding and per-ResBlock FiLM.
#
# PREREQUISITE: the h5 set from bigpurple/nyumets_h5.sbatch. The 60/30/10 patient-grouped
# split runs INSIDE this script (see SPLIT below), so chaining is one dependency:
#
#   sbatch --dependency=afterok:<h5 jobid> --kill-on-invalid-dep=yes \
#          bigpurple/i2sb_unet_nyumets.sbatch
#
# --kill-on-invalid-dep=yes matters: without it, a failed h5 job leaves this one queued
# forever as DependencyNeverSatisfied rather than failing.
#
# The split prints a bridge calibration to this job's log. It is INFORMATIONAL -- beta_max
# stays at the BraTS-inherited 0.3 unless you edit the config. beta_max lives in BOTH
# cfg["i2sb"] AND cfg["model"]["params"] (SBUnet carries its own schedule copy to invert
# std_fwd back to the step index) and train_i2sb calls assert_schedule_matches() at startup,
# so if you change it, change both or the job dies.
#
# Submit:  sbatch bigpurple/i2sb_unet_nyumets.sbatch

#SBATCH --partition=a100_short
#SBATCH --job-name=i2sb-unet-nyumets
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --chdir=/gpfs/scratch/ee2178/ImMAP
#SBATCH --output=/gpfs/scratch/ee2178/ImMAP/bigpurple/logs/%x_%j.out
#SBATCH --error=/gpfs/scratch/ee2178/ImMAP/bigpurple/logs/%x_%j.err

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

CFG="${CFG:-config/NYUMets/i2sb_unet_all.json}"
SOURCE="${SOURCE:-../datasets/NYUMets_h5}"
SPLIT_CONFIG="${SPLIT_CONFIG:-config/NYUMets/split.json}"

# --- GPU preflight, BEFORE any work ------------------------------------------------------
# `nvidia-smi --query-gpu=name` answers over NVML and succeeds even when the device cannot be
# opened, so it is NOT a usability check -- a node whose GPU is held in Exclusive_Process mode
# by a stray process prints its name happily and then fails at the first .to(device) with
# `cudaErrorDevicesUnavailable`. Allocating one real tensor is the check that means something.
python - <<'PY' || exit 1
import sys, torch
if not torch.cuda.is_available():
    print("PREFLIGHT: no CUDA device visible -- was --gres honoured?", file=sys.stderr)
    sys.exit(1)
try:
    torch.zeros(1, device="cuda")
except Exception as err:
    print(f"PREFLIGHT: GPU visible but a context could not be created on it: {err}",
          file=sys.stderr)
    print("  the card is allocated but held or wedged; resubmit, and if the same node "
          "recurs use --exclude=<node>", file=sys.stderr)
    sys.exit(1)
print(f"gpu    : {torch.cuda.get_device_name(0)}  (context OK)")
PY

# --- optional split, run in sequence before training ------------------------------------
#   SPLIT=auto   (default) split only if <SOURCE>_train is missing -- never touches an
#                existing split, so a resubmit cannot reshuffle val under a checkpoint
#   SPLIT=never  skip entirely
#   SPLIT=force  re-split even if it exists (RESHUFFLES; invalidates old comparisons)
do_split() {
    echo "[split] $SOURCE -> ${SOURCE}_{train,val,test}  ($SPLIT_CONFIG)"
    # A FILE, not `python -m`: datasets/__init__.py eagerly imports every registered loader
    # (torchvision, wandb) and this script needs none of them.
    python datasets/make_split_symlinks.py --source "$SOURCE" --config "$SPLIT_CONFIG" "$@" \
        || { echo "[split] FAILED" >&2; return 1; }
    echo "[split] bridge calibration (informational -- beta_max is NOT applied):"
    python -m scripts.calibrate_nyumets_bridge --root "${SOURCE}_train" --scale 3.0 \
        || echo "[split] calibration failed; not fatal"
    return 0
}

case "${SPLIT:-auto}" in
    never)
        echo "[split] SPLIT=never, skipping"
        ;;
    auto)
        if [ -d "${SOURCE}_train" ]; then
            echo "[split] ${SOURCE}_train exists -> skipping (SPLIT=force to redo)"
        else
            do_split || exit 1
        fi
        ;;
    force)
        do_split --force || exit 1
        ;;
    *)
        echo "SPLIT must be auto|never|force, got '${SPLIT}'" >&2
        exit 1
        ;;
esac

# The config's data roots are RELATIVE TO THE REPO ROOT (../datasets/NYUMets_h5_{train,val},
# i.e. $HOME/scratch/datasets/... given IMMAP_ROOT=$HOME/scratch/ImMAP) -- the same convention
# the raw ../datasets/NYUMets symlink uses. They only resolve because of the cd above.
python - "$CFG" <<'PY' || exit 1
import json, os, sys
cfg = json.load(open(sys.argv[1]))
bad = [(s, cfg["data"][s]["root"]) for s in ("train", "val")
       if not os.path.isdir(cfg["data"][s]["root"])]
for s, r in bad:
    print(f"  !! data.{s}.root does not exist: {r}  (cwd {os.getcwd()})", file=sys.stderr)
if bad:
    print(f"  the split did not produce them, or the 'root' fields in {sys.argv[1]} are wrong.",
          file=sys.stderr)
    sys.exit(1)
PY

echo "host   : $(hostname)"
echo "config : $CFG"

python train.py "$CFG"
