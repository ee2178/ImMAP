#!/bin/bash
#
# UNET + LGGS (models/sb_unet_lggs.py): a bridge UNet's CT1 estimate u refined by K guided
# primal-dual sweeps that treat u as the measurement and the prior study's CT1 (z-2..z+2) as the
# guide. Four factors, controls, and the UNet alone:
#
#   size     full = the 23M bridge UNet (model_channels 64) | 2m = model_channels 20
#   K        LGGS sweeps
#   noise    meas_noise {"max": 0.1, "p_zero": 0.25}: in TRAINING ONLY, LGGS sees u + sigma_n eps,
#            sigma_n ~ U(0, 0.1) (0 with probability 0.25), and the level sqrt(sigma_t^2 + sigma_n^2)
#   uguide   the UNet also gets the 5 prior-CT1 planes as input channels (unet_guides 0-4)
#
#   idx  group           config                                           unet K     noise  unet   LGGS
#   ---  --------------  -----------------------------------------------  ---- ----- ------ ------ ----
#    0  A factorial     i2sb_ulggs_full_k10_clean_ublind                 full K10   -      -       LGGS guided
#    1  A factorial     i2sb_ulggs_full_k10_clean_uguide                 full K10   -      uguide  LGGS guided
#    2  A factorial     i2sb_ulggs_full_k10_noise_ublind                 full K10   noise  -       LGGS guided
#    3  A factorial     i2sb_ulggs_full_k10_noise_uguide                 full K10   noise  uguide  LGGS guided
#    4  A factorial     i2sb_ulggs_2m_k10_clean_ublind                   2m   K10   -      -       LGGS guided
#    5  A factorial     i2sb_ulggs_2m_k10_clean_uguide                   2m   K10   -      uguide  LGGS guided
#    6  A factorial     i2sb_ulggs_2m_k10_noise_ublind                   2m   K10   noise  -       LGGS guided
#    7  A factorial     i2sb_ulggs_2m_k10_noise_uguide                   2m   K10   noise  uguide  LGGS guided
#    8  B depth         i2sb_ulggs_full_k5_clean_ublind                  full K5    -      -       LGGS guided
#    9  B depth         i2sb_ulggs_full_k20_clean_ublind                 full K20   -      -       LGGS guided
#   10  B depth         i2sb_ulggs_2m_k5_clean_ublind                    2m   K5    -      -       LGGS guided
#   11  B depth         i2sb_ulggs_2m_k20_clean_ublind                   2m   K20   -      -       LGGS guided
#   12  C unguided LGGS i2sb_ulggs_full_k10_clean_ublind_lggsunguided    full K10   -      -       LGGS unguided
#   13  C unguided LGGS i2sb_ulggs_2m_k10_clean_ublind_lggsunguided      2m   K10   -      -       LGGS unguided
#   14  D UNet only     i2sb_ulggs_full_unetonly_ublind                  full -     -      -       
#   15  D UNet only     i2sb_ulggs_full_unetonly_uguide                  full -     -      uguide  
#   16  D UNet only     i2sb_ulggs_2m_unetonly_ublind                    2m   -     -      -       
#   17  D UNet only     i2sb_ulggs_2m_unetonly_uguide                    2m   -     -      uguide  
#
# Reading it:  A = every combination of size x noise x uguide at K 10.  B vs A's clean/ublind rows
# = depth.  A clean/ublind vs C = the guide acting through LGGS (same sweeps, guide off).  A vs D
# = what the refinement stage adds over the UNet alone, with and without guide access.
#
# EVERY RUN: the same loader (other_study, prior CT1 at z-2..z+2, so multi-study patients only),
# intensity_norm volume_scale, the fit-registered data, effective batch 64, EMA 0.99, 300,000
# steps from scratch (no warm start: the old UNet checkpoints were trained on the fixed scales, and
# a UNet with guide channels has a different input width), u_weight 0.1 on the refined runs.
# Run folders: trained_nets/nyumets/ulggs/<name>. Submitting an index starts that run from scratch.
#
# NOT MEASURED: the step time of the full UNet + K sweeps. The 2m runs and K 5 are cheaper, K 20
# dearer. Read it/s off the first hour of one full K 10 run before submitting the rest.
#
# Submit all (at most 4 GPUs at once):   sbatch bigpurple/nyumets_ulggs_array.sbatch
# The factorial only:                   sbatch --array=0-7 bigpurple/nyumets_ulggs_array.sbatch
#
# The rest (partition, preflight, split, root checks) is bigpurple/i2sb_nyumets_array.sbatch.

#SBATCH --partition=radiology
#SBATCH --job-name=nyumets-ulggs
#SBATCH --array=0-17%4
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
    config/NYUMets/i2sb_ulggs_full_k10_clean_ublind.json
    config/NYUMets/i2sb_ulggs_full_k10_clean_uguide.json
    config/NYUMets/i2sb_ulggs_full_k10_noise_ublind.json
    config/NYUMets/i2sb_ulggs_full_k10_noise_uguide.json
    config/NYUMets/i2sb_ulggs_2m_k10_clean_ublind.json
    config/NYUMets/i2sb_ulggs_2m_k10_clean_uguide.json
    config/NYUMets/i2sb_ulggs_2m_k10_noise_ublind.json
    config/NYUMets/i2sb_ulggs_2m_k10_noise_uguide.json
    config/NYUMets/i2sb_ulggs_full_k5_clean_ublind.json
    config/NYUMets/i2sb_ulggs_full_k20_clean_ublind.json
    config/NYUMets/i2sb_ulggs_2m_k5_clean_ublind.json
    config/NYUMets/i2sb_ulggs_2m_k20_clean_ublind.json
    config/NYUMets/i2sb_ulggs_full_k10_clean_ublind_lggsunguided.json
    config/NYUMets/i2sb_ulggs_2m_k10_clean_ublind_lggsunguided.json
    config/NYUMets/i2sb_ulggs_full_unetonly_ublind.json
    config/NYUMets/i2sb_ulggs_full_unetonly_uguide.json
    config/NYUMets/i2sb_ulggs_2m_unetonly_ublind.json
    config/NYUMets/i2sb_ulggs_2m_unetonly_uguide.json
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
