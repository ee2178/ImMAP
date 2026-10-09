#!/bin/bash
#
# SBCDLNet on the plain SYNTHESIS task (one forward pass, no bridge), against a standard UNet.
# Non-longitudinal: every run maps this session's FLAIR, T1, T2 -> CT1, no prior study, all
# patients. Separate from i2sb_nyumets_array.sbatch so its indices are untouched.
#
#   mkdir -p bigpurple/logs && sbatch bigpurple/synth_sbcdlnet_array.sbatch
#
#   idx  config                         model      what it is
#    0   synth_unet_all_bs32.json       Unet2D     the standard UNet, at THIS array's recipe
#    1   synth_sbcdlnet_sfree.json      SBCDLNet   S = D_S z: learned DC through S's own dictionary
#
#    2   synth_sbguidedcdl_otherstudy_w0_sfree_flex.json   SBGuidedGroupCDL   as 1 + 1 guide:  prior CT1 at z
#    3   synth_sbguidedcdl_otherstudy_w1_sfree_flex.json   SBGuidedGroupCDL   as 1 + 3 guides: prior CT1 at z-1..z+1
#    4   synth_sbguidedcdl_otherstudy_w2_sfree_flex.json   SBGuidedGroupCDL   as 1 + 5 guides: prior CT1 at z-2..z+2
#
# 0 vs 1: a UNet against an unrolled sparse-coding net with a physics-informed T1 term.
# 1 vs 2: the guided prox with the prior-study CT1.  2 vs 3 vs 4: the number of guide slices.
#
# THE GUIDED RUNS (2-4) ARE LONGITUDINAL, unlike 0-1: the prior study enters through the guided
# prox's adjacency (attn_backend "flex"), never as an input channel. Two consequences:
#   * guide_mode "other_study" drops patients with a single study in the split, so 2-4 train
#     and validate on the multi-study population, not on every patient as 0-1 do. Their 1,000
#     validation slices are therefore NOT the same slices as 0-1's: compare 2-4 with each other
#     on the logged numbers, and with 0-1 only in a notebook on one shared set.
#   * they train at BATCH 8, not this array's 32 (set 2026-10-09 to keep them affordable).
#     Measured for the bridge version of this net on an A100 at batch 8: 0.22 / 0.36 / 0.50 s
#     per step for 1 / 3 / 5 guides, i.e. about 0.8 / 1.3 / 1.7 days for 300k steps; without the
#     bridge term these should come in somewhat under that. So 2-4 see a quarter of the samples
#     per step that 0-1 do: 1 vs 2 is NOT budget-matched.
#
# ONE RECIPE for 0-1 (2-4 differ in batch size, above), so the table is not a comparison of training budgets: an EFFECTIVE
# batch of 32, no EMA, 300,000 optimizer steps, Adam 1e-4 with cosine decay, complex-mse on the
# whole frame, and validation every 5,000 steps on the SAME 1,000 val slices (val_seed 0). The
# only training keys that differ are each model's own gradient clip (UNet 1.0, SBCDLNet 0.05) and
# s_weight (1-4).
#
# HOW THE 32 IS REACHED. 0 and 1 use a true batch of 32 (accum_steps 1). Measured for 1 on an
# A100 with cuDNN autotuning ON, as train.py runs it (2026-10-09), ms per micro-batch:
#
#     batch 8 -> 84     16 -> 159     32 -> 300          (95 / 101 / 107 samples per second)
#
# so one step of 32 is 0.31 s (~1.1 days for 300k steps, 15 GB), against 0.34 s as 8 x 4.
# An EARLIER version of this header said SBCDLNet's throughput peaks at a micro-batch of 8 and
# that 8 x 4 was faster. That was measured with autotuning OFF (8 -> 171, 16 -> 510, 32 -> 1142)
# and is not how training runs.
#
# SUBMITTING AN INDEX ALWAYS STARTS THAT RUN FROM SCRATCH, over whatever is in its save_dir (the
# configs have paths.ckpt = null). To RESUME an interrupted run instead, give train.py the config
# that run saved:   python train.py trained_nets/nyumets/<run name>/config.json
#
# The existing synth_unet_all run (config/NYUMets/synth_unet_all.json, no longer in any array)
# is the same UNet at batch 64 + EMA.
# It is NOT rerun here; it also validates on the whole val split, so its logged val numbers are
# not on the same slices as these.
#
# In SBCDLNet the code is driven by the conditioning stack alone: there is no bridge state, so
# `bridge_fidelity: false`. FLAIR and T2 are fitted through their own dictionary; the T1 channel
# goes through the CT1 dictionary and the enhancement map, T1 = CT1 - S. Run 1 logs the val/S
# image and the collapse/* scalars.
#
# No --partition override: radiology is where these run.

#SBATCH --partition=radiology
#SBATCH --job-name=synth-sbcdl
#SBATCH --array=0-4
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

# No `set -euo pipefail` -- it breaks conda activate (unset $PS1 / $CONDA_SHLVL under -u).

CONFIGS=(
    config/NYUMets/synth_unet_all_bs32.json
    config/NYUMets/synth_sbcdlnet_sfree.json
    config/NYUMets/synth_sbguidedcdl_otherstudy_w0_sfree_flex.json
    config/NYUMets/synth_sbguidedcdl_otherstudy_w1_sfree_flex.json
    config/NYUMets/synth_sbguidedcdl_otherstudy_w2_sfree_flex.json
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
