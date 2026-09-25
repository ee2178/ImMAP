#!/bin/bash
#
# Every NYUMets I2SB experiment as one job array: one config per task, one A100 per task.
#
#   idx  config                               model             inputs to the bridge regressor
#   ---  -----------------------------------  ----------------  ------------------------------------
#    0   i2sb_unet_t1.json                    SBUnet   C=1      none (T1 only as the bridge start)
#    1   i2sb_unet_all.json                   SBUnet   C=4      FLAIR, T1, T2
#    2   i2sb_unet_otherstudy.json            SBUnet   C=5      FLAIR, T1, T2 + other-study CT1 channel
#    3   i2sb_sbcdlnet_all.json               SBCDLNet          FLAIR, T1, T2
#    4   i2sb_sbguidedcdl_session.json        SBGuidedGroupCDL  T1; guides T1, T2, FLAIR
#    5   i2sb_sbguidedcdl_otherstudy.json     SBGuidedGroupCDL  T1; guide other-study CT1
#    6   i2sb_lggcdl_session.json             LGGCDL   C=1      guides T1, T2, FLAIR
#    7   i2sb_lggs_session.json               LGGS     C=1      guides T1, T2, FLAIR
#    8   i2sb_unet_from_otherct1.json         SBUnet   C=4      FLAIR, T1, T2; the bridge STARTS at
#                                                              the other study's CT1 (x1_source),
#                                                              this session's T1 comes back as `y`
#    9   i2sb_unet_otherstudy_pair.json       SBUnet   C=6      FLAIR, T1, T2 + other-study T1, CT1
#   10   i2sb_unet_otherstudy_pair_w1.json    SBUnet   C=10     ... + other-study T1/CT1 at z-1..z+1
#   11   i2sb_unet_otherstudy_pair_w2.json    SBUnet   C=14     ... + other-study T1/CT1 at z-2..z+2
#
# Paired comparisons: 1 vs 2 isolates the other-study CT1 channel; 3 vs 4 is conditioning
# channels vs guided prox; 4 vs 5 swaps same-session guides for the other-study CT1; 2 vs 8 is
# the prior scan as a CONDITIONING CHANNEL vs as the BRIDGE START (what ImMAP-SB needs, and what
# notebooks/immap_sb_longitudinal.ipynb compares).
# The GUIDE-WIDTH sweep is 2, 9, 10, 11: other-study CT1 alone, then the other study's T1/CT1
# pair at 1, 3 and 5 slices (guide_idx [1, 2], guide_window 0 / 1 / 2). SBUnet only -- the guided
# prox models pay per guide plane, the UNet only widens its first conv.
#   sbatch --array=2,9-11 bigpurple/i2sb_nyumets_array.sbatch
#
# The LONGITUDINAL subset -- the ones that use another study of the same patient -- is 2, 5, 8-11:
#   sbatch --array=2,5,8-11 bigpurple/i2sb_nyumets_array.sbatch
# (SB-LGGCDL with all 3 guides + other CT1 was dropped: it OOMs on one A100.)
#
# Submit all (at most 4 GPUs at once):
#   sbatch bigpurple/i2sb_nyumets_array.sbatch
# A subset (the --array on the command line overrides the #SBATCH line):
#   sbatch --array=2,4,5 bigpurple/i2sb_nyumets_array.sbatch
# After the h5 build:
#   sbatch --dependency=afterok:<h5 jobid> --kill-on-invalid-dep=yes \
#          bigpurple/i2sb_nyumets_array.sbatch
#
# Every config has its own paths.save_dir, so tasks never share a checkpoint directory.
# The rest (partition, preflight, split, root checks) is bigpurple/i2sb_unet_nyumets.sbatch;
# the only behavioural difference is that the split is serialised with flock, because array
# tasks start together and would otherwise all split the same directory at once.

#SBATCH --partition=radiology
#SBATCH --job-name=i2sb-nyumets
#SBATCH --array=0-11%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=EE2178@nyulangone.org
#SBATCH --chdir=/gpfs/scratch/ee2178/ImMAP
#SBATCH --output=/gpfs/scratch/ee2178/ImMAP/bigpurple/logs/%x_%A_%a.out
#SBATCH --error=/gpfs/scratch/ee2178/ImMAP/bigpurple/logs/%x_%A_%a.err

# --time is 7 days (radiology allows 28): the unrolled nets (3-7) are far slower per step
# than SBUnet, and the guided ones pay for 15x15 guide-window attention on top.
# Mail is for the array as a whole (BEGIN when the first task starts, END/FAIL when all are
# done). Add ARRAY_TASKS to --mail-type for one mail per task (24 mails for 8 tasks).
# No `set -euo pipefail` -- it breaks conda activate (unset $PS1 / $CONDA_SHLVL under -u).

CONFIGS=(
    config/NYUMets/i2sb_unet_t1.json
    config/NYUMets/i2sb_unet_all.json
    config/NYUMets/i2sb_unet_otherstudy.json
    config/NYUMets/i2sb_sbcdlnet_all.json
    config/NYUMets/i2sb_sbguidedcdl_session.json
    config/NYUMets/i2sb_sbguidedcdl_otherstudy.json
    config/NYUMets/i2sb_lggcdl_session.json
    config/NYUMets/i2sb_lggs_session.json
    config/NYUMets/i2sb_unet_from_otherct1.json
    config/NYUMets/i2sb_unet_otherstudy_pair.json
    config/NYUMets/i2sb_unet_otherstudy_pair_w1.json
    config/NYUMets/i2sb_unet_otherstudy_pair_w2.json
)

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "not an array task -- submit with sbatch (--array), or run one config through" >&2
    echo "CFG=<config> sbatch bigpurple/i2sb_unet_nyumets.sbatch" >&2
    exit 1
fi
CFG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
if [ -z "$CFG" ]; then
    echo "array index $SLURM_ARRAY_TASK_ID out of range (0-$(( ${#CONFIGS[@]} - 1 )))" >&2
    exit 1
fi

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

if [ ! -f "$CFG" ]; then
    echo "config not found: $CFG (cwd $(pwd))" >&2
    exit 1
fi

SOURCE="${SOURCE:-../datasets/NYUMets_h5}"
SPLIT_CONFIG="${SPLIT_CONFIG:-config/NYUMets/split.json}"

echo "task   : $SLURM_ARRAY_JOB_ID[$SLURM_ARRAY_TASK_ID]  ->  $CFG"

# --- GPU preflight, BEFORE any work ------------------------------------------------------
# See i2sb_unet_nyumets.sbatch: nvidia-smi succeeds on a card held in Exclusive_Process mode,
# so allocate a real tensor, and retry because holds from a previous job are often transient.
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
    echo "=========== PREFLIGHT FAILED after $PREFLIGHT_TRIES attempts ===========" >&2
    echo "host=$(hostname)  job=$SLURM_JOB_ID  gpus=$SLURM_JOB_GPUS  step=$SLURM_STEP_GPUS" >&2
    echo "CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES-<unset>}'" >&2
    scontrol show job "$SLURM_JOB_ID" 2>/dev/null | grep -iE 'gres|tres|nodelist|numnodes' >&2
    nvidia-smi --query-gpu=index,uuid,name,compute_mode,persistence_mode,memory.used \
               --format=csv 2>&1 >&2
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>&1 >&2
    echo "=======================================================================" >&2
    exit 1
fi

# --- optional split, serialised across array tasks ---------------------------------------
#   SPLIT=auto   (default) split only if <SOURCE>_train is missing
#   SPLIT=never  skip entirely
#   SPLIT=force  NOT allowed here: every task would reshuffle the split under the others.
#                Force-split once with i2sb_unet_nyumets.sbatch, then submit this array.
# The existence check runs INSIDE the lock: the first task to get it splits, the rest wait on
# the lock and then find <SOURCE>_train already there. The lock file sits next to the data on
# GPFS, which every node sees.
do_split() {
    echo "[split] $SOURCE -> ${SOURCE}_{train,val,test}  ($SPLIT_CONFIG)"
    python datasets/make_split_symlinks.py --source "$SOURCE" --config "$SPLIT_CONFIG" \
        || { echo "[split] FAILED" >&2; return 1; }
    local KEY SCALE
    KEY=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['data']['train']['image_key'])" "$CFG")
    SCALE=$(python -c "
import json,sys
c=json.load(open(sys.argv[1]))['data']['train']
s=c.get('scales') or [1,1,1,1]
print(s[c.get('x1_idx',1)])" "$CFG")
    echo "[split] bridge calibration (informational -- beta_max is NOT applied):"
    echo "[split]   image_key=$KEY  scale=$SCALE  (from $CFG)"
    python -m scripts.calibrate_nyumets_bridge --root "${SOURCE}_train" \
        --image-key "$KEY" --scale "$SCALE" \
        || echo "[split] calibration failed; not fatal"
    return 0
}

case "${SPLIT:-auto}" in
    never)
        echo "[split] SPLIT=never, skipping"
        ;;
    auto)
        (
            flock -w 3600 9 || { echo "[split] timed out waiting for the split lock" >&2; exit 1; }
            if [ -d "${SOURCE}_train" ]; then
                echo "[split] ${SOURCE}_train exists -> skipping"
            else
                do_split || exit 1
            fi
        ) 9>"${SOURCE}.split.lock" || exit 1
        ;;
    *)
        echo "SPLIT must be auto|never in an array job, got '${SPLIT}'" >&2
        exit 1
        ;;
esac

# Data roots are relative to the repo root; they only resolve because of the cd above.
python - "$CFG" <<'PY' || exit 1
import json, os, sys
cfg = json.load(open(sys.argv[1]))
bad = [(s, cfg["data"][s]["root"]) for s in ("train", "val")
       if not os.path.isdir(cfg["data"][s]["root"])]
for s, r in bad:
    print(f"  !! data.{s}.root does not exist: {r}  (cwd {os.getcwd()})", file=sys.stderr)
if bad:
    sys.exit(1)
PY

echo "host   : $(hostname)"
echo "config : $CFG"

python train.py "$CFG"
