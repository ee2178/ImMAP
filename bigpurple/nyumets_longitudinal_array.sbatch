#!/bin/bash
#
# EVERY LONGITUDINAL NET (one that sees another study of the patient), retrained at the default
# recipe: effective batch 64 (data.batch_size 32 x training.accum_steps 2), EMA 0.99 with warm-up,
# 300,000 optimizer steps, the fit-registered data, the T1 & CT1 loader mask, slices [50, 90),
# and PER-VOLUME intensity SCALING (intensity_norm "volume_scale": x -> x / s_v per volume and
# contrast, s_v = that volume's in-brain median; nothing is subtracted). It replaces the one fixed
# scale per contrast, which left each scan's gain in the data, so prior and current studies were
# on different brightness scales. The guided nets take the prior study's CT1 as their guide.
#
#   idx  config                                                   model             task      what
#   ---  -------------------------------------------------------  ----------------  --------  ----
#    0  i2sb_unet_otherstudy_pair_w2_bs64                        SBUnet            bridge    prior T1/CT1 at z-2..z+2 as input channels
#    1  i2sb_sbguidedcdl_otherstudy_w0_sfree_flex_bs64           SBGuidedGroupCDL  bridge    guided prox, prior CT1 at z
#    2  i2sb_sbguidedcdl_otherstudy_w1_sfree_flex_bs64           SBGuidedGroupCDL  bridge    prior CT1 at z-1..z+1
#    3  i2sb_sbguidedcdl_otherstudy_w2_sfree_flex_bs64           SBGuidedGroupCDL  bridge    prior CT1 at z-2..z+2
#    4  synth_unet_otherstudy_pair_w2                            Unet2D            synthesis end-to-end UNet, prior T1/CT1 z-2..z+2 as channels
#    5  synth_sbguidedcdl_otherstudy_w0_sfree_flex_bs64          SBGuidedGroupCDL  synthesis guided prox, prior CT1 at z
#    6  synth_sbguidedcdl_otherstudy_w1_sfree_flex_bs64          SBGuidedGroupCDL  synthesis prior CT1 at z-1..z+1
#    7  synth_sbguidedcdl_otherstudy_w2_sfree_flex_bs64          SBGuidedGroupCDL  synthesis prior CT1 at z-2..z+2
#    8  synth_sbguidedcdl_otherstudy_w0_sfree_additive_flex_bs64 SBGuidedGroupCDL  synthesis w0, additive variant
#    9  synth_sbguidedcdl_otherstudy_w0_sfree_sw5_e500_flex_bs64 SBGuidedGroupCDL  synthesis w0, s_weight 5, 500 epochs
#
# THE SAME CONFIG FILES as bigpurple/nyumets_bs64_array.sbatch (bridge 1, 3-5; synthesis 9-11) and
# bigpurple/nyumets_followup_array.sbatch (4, 5), so each run writes the folder it always has.
# SUBMITTING AN INDEX RESTARTS THAT RUN FROM SCRATCH, OVER WHATEVER IS THERE. Copy a
# folder under trained_nets/nyumets/ first if you want to keep it. Do not submit one config from two arrays.
# To resume instead:  python train.py trained_nets/nyumets/<run name>/config.json
#
# Not listed: the enhancement cross-attention (xS) and UNet2M / UNet2M+LGGS runs (set aside for
# now); the w0 / w1 UNet pair runs, the older gather / gate guided configs, the cycle nets and
# the non-2M UNet+LGGS runs (no batch-64 version).
#
# Submit all (at most 4 GPUs at once):   sbatch bigpurple/nyumets_longitudinal_array.sbatch
# Bridge only / synthesis only:         sbatch --array=0-3 ...   /   sbatch --array=4-9 ...
#
# The rest (partition, preflight, split, root checks) is bigpurple/i2sb_nyumets_array.sbatch.

#SBATCH --partition=radiology
#SBATCH --job-name=nyumets-longi
#SBATCH --array=0-9%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=14-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=EE2178@nyulangone.org
#SBATCH --chdir=/gpfs/scratch/ee2178/ImMAP
#SBATCH --output=/gpfs/scratch/ee2178/ImMAP/bigpurple/logs/%x_%A_%a.out
#SBATCH --error=/gpfs/scratch/ee2178/ImMAP/bigpurple/logs/%x_%A_%a.err

# --time is 14 days (radiology allows 28): the 5-guide net is ~9.5 days at batch 64.
# Mail is for the array as a whole (BEGIN when the first task starts, END/FAIL when all are
# done). Add ARRAY_TASKS to --mail-type for one mail per task (24 mails for 8 tasks).
# No `set -euo pipefail` -- it breaks conda activate (unset $PS1 / $CONDA_SHLVL under -u).

CONFIGS=(
    config/NYUMets/i2sb_unet_otherstudy_pair_w2_bs64.json
    config/NYUMets/i2sb_sbguidedcdl_otherstudy_w0_sfree_flex_bs64.json
    config/NYUMets/i2sb_sbguidedcdl_otherstudy_w1_sfree_flex_bs64.json
    config/NYUMets/i2sb_sbguidedcdl_otherstudy_w2_sfree_flex_bs64.json
    config/NYUMets/synth_unet_otherstudy_pair_w2.json
    config/NYUMets/synth_sbguidedcdl_otherstudy_w0_sfree_flex_bs64.json
    config/NYUMets/synth_sbguidedcdl_otherstudy_w1_sfree_flex_bs64.json
    config/NYUMets/synth_sbguidedcdl_otherstudy_w2_sfree_flex_bs64.json
    config/NYUMets/synth_sbguidedcdl_otherstudy_w0_sfree_additive_flex_bs64.json
    config/NYUMets/synth_sbguidedcdl_otherstudy_w0_sfree_sw5_e500_flex_bs64.json
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

SOURCE="${SOURCE:-../datasets/NYUMets_h5_fit}"
SPLIT_CONFIG="${SPLIT_CONFIG:-config/NYUMets/split.json}"
# The fit build's splits keep every patient in the split the centroid build gave it (--like the
# centroid split record) and leave out every study registered worse than MIN_REG_NCC (its stored
# reg_ncc). Used only if ${SOURCE}_train does not exist yet -- normally it is made once by hand:
#   python datasets/make_split_symlinks.py --source ../datasets/NYUMets_h5_fit \
#       --like ../datasets/NYUMets_h5_split_record.json --min-reg-ncc 0.6
SPLIT_LIKE="${SPLIT_LIKE:-../datasets/NYUMets_h5_split_record.json}"
MIN_REG_NCC="${MIN_REG_NCC:-0.6}"
SPLIT_EXTRA=(--min-reg-ncc "$MIN_REG_NCC")
[ -f "$SPLIT_LIKE" ] && SPLIT_EXTRA+=(--like "$SPLIT_LIKE")

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
    python datasets/make_split_symlinks.py --source "$SOURCE" --config "$SPLIT_CONFIG" "${SPLIT_EXTRA[@]}" \
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
