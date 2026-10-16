#!/bin/bash
#
# The follow-up experiments: is the UNet's lead CAPACITY, and does the prior study help more as a
# PRINCIPLED PRIOR on the UNet's estimate than as input channels? Same recipe as
# bigpurple/nyumets_bs64_array.sbatch (effective batch 64 = 32 x 2, EMA 0.99 with warm-up, the
# fit-registered splits, the T1 & CT1 loader mask, slices [50, 90)), so every number here compares
# directly with that array's.
#
#   idx  config                                             model             task      what
#   ---  -------------------------------------------------  ----------------  --------  ----
#    0  i2sb_unet2m_otherstudy_pair_w2_bs64                SBUnet (2.26M)    bridge    bs64 #1 narrowed to ~2M
#    1  synth_unet2m_all_bs64                              Unet2D (2.23M)    synthesis bs64 #7 narrowed to ~2M
#    2  i2sb_unetlggs_k10_guided_otherstudy_w2_bs64        SBUnetLGGS        bridge    UNet(FLAIR,T1,T2) -> 10 LGGS sweeps,
#                                                                                      prior CT1 z-2..z+2 as guides
#    3  i2sb_unetlggs_k10_unguided_otherstudy_w2_bs64      SBUnetLGGS        bridge    the same sweeps, no guide (control)
#
# CAPACITY CONTROLS (0-1). The unrolled nets are ~2.0M parameters; the UNets they lose to are
# 23.0M (bridge) and 5.7M (synthesis). 0 and 1 change ONLY the width of bs64 #1 / #7 (bridge:
# model_channels 64 -> 20, num_head_channels 32 -> 20; synthesis: chans 32 -> 20, max_chans
# 256 -> 160). Compare 0 with bs64 #2-6 and 1 with bs64 #8-11: if the ~2M UNet still wins, the
# gap is not capacity.
#
# UNET + LGGS (2-3), models/sb_unet_lggs.py. The UNet sees only x_t and the current FLAIR/T1/T2
# and gives a CT1 estimate u; 10 LGGS sweeps (E = Identity) then refine it, with the prior study's
# CT1 entering ONLY the guided group threshold's adjacency:  x0_hat ~ argmin 1/2||x - u||^2 +
# lambda R(Ax; Aw). The UNet is WARM-STARTED from bs64 #0 (paths.init_ckpt) and fine-tuned with
# the LGGS stage: 150k steps at lr 1e-4, u_weight 0.1 keeping u a CT1 estimate. Snapshot bs64 #0's
# average FIRST, so both arms start from the same weights (the tasks refuse to start without it):
#   cp trained_nets/nyumets/I2SB_Unet_NYUMets_CT1_from_all_bs64/net_ema.ckpt #      trained_nets/nyumets/I2SB_Unet_NYUMets_CT1_from_all_bs64/net_ema_warmstart.ckpt
# Compare 2 with 3 (what the prior adds through the prior term), both with bs64 #0 at 300k steps
# (the same total budget: ~150k before the snapshot + 150k here), and 2 with bs64 #1 (the prior
# as a guided prior vs as input channels). Judge them on lesion regions of patients whose disease
# changed, not only whole-image PSNR: the guided threshold favours what the prior confirms.
#
# RUNTIME, NOT MEASURED: the ~2M UNets well under a day or two; 2 roughly 2.5 days (10 guided
# sweeps over 5 guides, scaled from the 30-iteration guided CDL net's ~9.5 days), 3 about 1.
#
# Submit all (at most 4 GPUs at once):  sbatch bigpurple/nyumets_followup_array.sbatch
# A subset:                            sbatch --array=0-1 bigpurple/nyumets_followup_array.sbatch
#
# SUBMITTING AN INDEX STARTS THAT RUN FROM SCRATCH and writes over its save_dir. To resume, give
# train.py the config the run saved:  python train.py trained_nets/nyumets/<run name>/config.json
# The rest (partition, preflight, split, root checks) is bigpurple/nyumets_bs64_array.sbatch.

#SBATCH --partition=radiology
#SBATCH --job-name=nyumets-followup
#SBATCH --array=0-3%4
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

# --time is 7 days: about 3x the slowest estimate above. Raise it if 2's first epochs say so.
# No `set -euo pipefail` -- it breaks conda activate (unset $PS1 / $CONDA_SHLVL under -u).

CONFIGS=(
    config/NYUMets/i2sb_unet2m_otherstudy_pair_w2_bs64.json
    config/NYUMets/synth_unet2m_all_bs64.json
    config/NYUMets/i2sb_unetlggs_k10_guided_otherstudy_w2_bs64.json
    config/NYUMets/i2sb_unetlggs_k10_unguided_otherstudy_w2_bs64.json
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

# A warm-started run (paths.init_ckpt) needs its snapshot BEFORE it starts: train.py would
# otherwise fail on the missing file only after the loaders and the GPU preflight.
INIT_CKPT="$(python -c "import json,sys; print(json.load(open(sys.argv[1])).get('paths', {}).get('init_ckpt') or '')" "$CFG")"
if [ -n "$INIT_CKPT" ] && [ ! -f "$INIT_CKPT" ]; then
    echo "warm start not found: $INIT_CKPT -- snapshot it first (see the header of this script)" >&2
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
