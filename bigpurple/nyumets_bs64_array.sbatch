#!/bin/bash
#
# EVERY NYUMets net at ONE training recipe, for a fair comparison: effective batch 64
# (data.batch_size 32 x training.accum_steps 2), EMA 0.99 with warm-up, 300,000 optimizer steps, the fit-
# registered data (NYUMets_h5_fit_*), the T1 & CT1 loader mask, slices [50, 90).
#
#   idx  config                                             model             task      what
#   ---  -------------------------------------------------  ----------------  --------  ----
#    0  i2sb_unet_all_bs64                                 SBUnet            bridge    FLAIR, T1, T2
#    1  i2sb_unet_otherstudy_pair_w2_bs64                  SBUnet            bridge    + prior T1/CT1 at z-2..z+2 as channels
#    2  i2sb_sbcdlnet_sfree_bs64                           SBCDLNet          bridge    FLAIR, T1, T2; S = D_S z
#    3  i2sb_sbguidedcdl_otherstudy_w0_sfree_flex_bs64     SBGuidedGroupCDL  bridge    1 guide: prior (CT1-T1)+ at z
#    4  i2sb_sbguidedcdl_otherstudy_w1_sfree_flex_bs64     SBGuidedGroupCDL  bridge    3 guides: z-1..z+1
#    5  i2sb_sbguidedcdl_otherstudy_w2_sfree_flex_bs64     SBGuidedGroupCDL  bridge    5 guides: z-2..z+2
#    6  i2sb_sbguidedcdl_otherstudy_sfree_xS_flex_bs64     SBGuidedGroupCDL  bridge    prior CT1 guide + S cross-attention
#    7  synth_unet_all_bs64                                Unet2D            synthesis FLAIR, T1, T2
#    8  synth_sbcdlnet_sfree_bs64                          SBCDLNet          synthesis learned DC only
#    9  synth_sbguidedcdl_otherstudy_w0_sfree_flex_bs64    SBGuidedGroupCDL  synthesis 1 guide
#   10  synth_sbguidedcdl_otherstudy_w1_sfree_flex_bs64    SBGuidedGroupCDL  synthesis 3 guides
#   11  synth_sbguidedcdl_otherstudy_w2_sfree_flex_bs64    SBGuidedGroupCDL  synthesis 5 guides
#   12  unsb_unet_all_bs64                                 SBUnet            unsb      FLAIR, T1, T2; paired D
#   13  unsb_unet_t1_bs64                                  SBUnet            unsb      T1 only; unpaired
#   14  i2sb_unet2m_otherstudy_pair_w2_bs64                SBUnet (~2M)      bridge    as 1, at ~2M parameters
#
# 14 is the CAPACITY CONTROL: index 1 narrowed, model_channels 20 (was 64) and num_head_channels
# 20 (was 32), nothing else -- 2.26M parameters against 23.0M for 1 and ~2.0M for the unrolled nets
# (2, 3-6). If it still beats them, the gap is not capacity. NOT in the default --array range
# below:  sbatch --array=14 bigpurple/nyumets_bs64_array.sbatch
#
#   15  i2sb_unetlggs_k10_guided_otherstudy_w2_bs64        SBUnetLGGS        bridge    0's UNet -> 10 LGGS sweeps, prior CT1 z-2..z+2 guides
#   16  i2sb_unetlggs_k10_unguided_otherstudy_w2_bs64      SBUnetLGGS        bridge    the same sweeps, no guide (control)
#
# 15-16: the UNet sees ONLY x_t and the current FLAIR/T1/T2 (index 0's inputs); the prior study
# reaches the estimate only through the LGGS adjacency. The UNet is WARM-STARTED from index 0
# (paths.init_ckpt) and fine-tuned together with the LGGS stage: 150k steps at lr 1e-4 (about
# where 0's cosine is at its midpoint), u_weight 0.1 on the UNet estimate. Snapshot 0's average
# FIRST, so both arms start from the same weights and a file the running job is not rewriting:
#   cp trained_nets/nyumets/I2SB_Unet_NYUMets_CT1_from_all_bs64/net_ema.ckpt \n#      trained_nets/nyumets/I2SB_Unet_NYUMets_CT1_from_all_bs64/net_ema_warmstart.ckpt
#   sbatch --array=15-16 bigpurple/nyumets_bs64_array.sbatch
# Compare 15 with 16 (what the prior adds through the prior term), both with 0 at 300k steps (the
# same total budget: ~150k before the snapshot + 150k here), and 15 with 1 (the prior as a guided
# prior vs the prior as input channels).
#
# WHAT STILL DIFFERS BETWEEN NETS, deliberately: each keeps its own learning rate and loss
# (the bridge UNets lr 2e-4 with sigma weighting, the unrolled nets 1e-4, UNSB its GAN Adam
# betas). EVERY net averages its weights (EMA 0.99, warm-up on): validation and the checkpoint to
# sample from (net_ema.ckpt / net_best.ckpt) use the average, net.ckpt keeps the live weights for
# resume. The averaged unrolled nets still satisfy their constraints: unit-ball filters and
# nonnegative thresholds are convex sets, and an average of points in one stays in it.
# 0 / 12 differ from the batch-256 runs only in batch size, under their own names (<name>_bs64).
#
# The guided _bs64 configs (3-6, 9-11) and i2sb_sbcdlnet_sfree_bs64 (2) are the same files the
# other two arrays list at their indices 7-11 -- so those indices there now train WITH EMA too, and
# submitting either array writes the same folder.
#
# Measured on an A100 (training step only; add ~15% for loading and logging), 300k steps:
#   SBCDLNet ~1.3 days; guided 1 / 3 / 5 guides ~3.5 / 6.5 / 9.5 days; bridge UNet ~1.5 days
#   (a quarter of its batch-256 run). Not measured: the xS net (one more branch than 3), the
#   synthesis UNet, and UNSB, whose accumulated step costs several UNet passes per micro-batch.
#
# Submit (at most 4 GPUs at once):   sbatch bigpurple/nyumets_bs64_array.sbatch
# A subset:                         sbatch --array=0-6 bigpurple/nyumets_bs64_array.sbatch
#
# SUBMITTING AN INDEX STARTS THAT RUN FROM SCRATCH and writes over its save_dir. To resume, give
# train.py the config the run saved:  python train.py trained_nets/nyumets/<run name>/config.json
# The rest (partition, preflight, split, root checks) is bigpurple/i2sb_nyumets_array.sbatch.

#SBATCH --partition=radiology
#SBATCH --job-name=nyumets-bs64
#SBATCH --array=0-13%4
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
    config/NYUMets/i2sb_unet_all_bs64.json
    config/NYUMets/i2sb_unet_otherstudy_pair_w2_bs64.json
    config/NYUMets/i2sb_sbcdlnet_sfree_bs64.json
    config/NYUMets/i2sb_sbguidedcdl_otherstudy_w0_sfree_flex_bs64.json
    config/NYUMets/i2sb_sbguidedcdl_otherstudy_w1_sfree_flex_bs64.json
    config/NYUMets/i2sb_sbguidedcdl_otherstudy_w2_sfree_flex_bs64.json
    config/NYUMets/i2sb_sbguidedcdl_otherstudy_sfree_xS_flex_bs64.json
    config/NYUMets/synth_unet_all_bs64.json
    config/NYUMets/synth_sbcdlnet_sfree_bs64.json
    config/NYUMets/synth_sbguidedcdl_otherstudy_w0_sfree_flex_bs64.json
    config/NYUMets/synth_sbguidedcdl_otherstudy_w1_sfree_flex_bs64.json
    config/NYUMets/synth_sbguidedcdl_otherstudy_w2_sfree_flex_bs64.json
    config/NYUMets/unsb_unet_all_bs64.json
    config/NYUMets/unsb_unet_t1_bs64.json
    config/NYUMets/i2sb_unet2m_otherstudy_pair_w2_bs64.json
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
