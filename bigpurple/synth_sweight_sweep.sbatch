#!/bin/bash
#
# s_weight SWEEP on SynthCDL (config/NYUMets/synth_sbcdlnet_sfree_bs64.json): pick the weight of the
# enhancement-map loss  s_weight * MSE(S_hat, (CT1 - T1)_+)  for the runs that follow. Each arm is
# that config with ONLY s_weight, num_epochs (500 = 50k steps) and val_every_epochs (25) changed
# -- config/NYUMets/sweep_sweight/, written by one script, checked to differ in nothing else.
#
#   idx  s_weight   config
#   ---  --------   -----------------------------------------------------------------
#    0     0        sweep_sweight/synth_sbcdlnet_sfree_sw0_e500_bs64    no S supervision
#    1     1        sweep_sweight/synth_sbcdlnet_sfree_sw1_e500_bs64    the current setting
#    2     2        sweep_sweight/synth_sbcdlnet_sfree_sw2_e500_bs64
#    3     5        sweep_sweight/synth_sbcdlnet_sfree_sw5_e500_bs64
#    4    10        sweep_sweight/synth_sbcdlnet_sfree_sw10_e500_bs64
#    5    20        sweep_sweight/synth_sbcdlnet_sfree_sw20_e500_bs64
#
# SAME LR AS THE FULL RUN: T_max stays 300k, so 500 epochs is the near-constant-LR start of the
# full cosine and the arms differ in s_weight alone. Short runs RANK the weights; they do not
# show converged numbers.
#
# READING IT (all under the same fixed 1000 val slices, validation every 25 epochs):
#   val/psnr, val/ssim       whole-image quality: the s term is TRAIN-only, so val/loss is the
#                            x0 loss in every arm and these compare directly
#   collapse/S_rel_err       rms(S_hat - (CT1-T1)_+) / rms((CT1-T1)_+): how ACCURATE the
#                            enhancement map is (1 = no better than S = 0). The deciding number.
#   collapse/S_rms_ratio     how much enhancement ENERGY S has (1 = right size). Read it WITH
#                            S_rel_err: energy in the wrong place raises the ratio and the error
#   val/S                    the panel, to see whether the detail is real
# Pick the largest s_weight that lowers S_rel_err without costing whole-image PSNR/SSIM beyond
# what you will accept; a weight past the knee usually buys S_rms_ratio and nothing else.
#
# RUNTIME, NOT MEASURED for 50k steps: SBCDLNet is ~1.3 days per 300k at batch 64 (bridge,
# measured), so ~5 h of training per arm plus 20 validations, inside the 12 h limit.
#
# Submit (at most 6 GPUs at once):  sbatch bigpurple/synth_sweight_sweep.sbatch
#
# SUBMITTING AN INDEX STARTS THAT RUN FROM SCRATCH and writes over its save_dir
# (trained_nets/nyumets/sweep_sweight/...). The rest is bigpurple/nyumets_bs64_array.sbatch.

#SBATCH --partition=radiology
#SBATCH --job-name=synth-sweight
#SBATCH --array=0-5%6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=EE2178@nyulangone.org
#SBATCH --chdir=/gpfs/scratch/ee2178/ImMAP
#SBATCH --output=/gpfs/scratch/ee2178/ImMAP/bigpurple/logs/%x_%A_%a.out
#SBATCH --error=/gpfs/scratch/ee2178/ImMAP/bigpurple/logs/%x_%A_%a.err

# No `set -euo pipefail` -- it breaks conda activate (unset $PS1 / $CONDA_SHLVL under -u).

CONFIGS=(
    config/NYUMets/sweep_sweight/synth_sbcdlnet_sfree_sw0_e500_bs64.json
    config/NYUMets/sweep_sweight/synth_sbcdlnet_sfree_sw1_e500_bs64.json
    config/NYUMets/sweep_sweight/synth_sbcdlnet_sfree_sw2_e500_bs64.json
    config/NYUMets/sweep_sweight/synth_sbcdlnet_sfree_sw5_e500_bs64.json
    config/NYUMets/sweep_sweight/synth_sbcdlnet_sfree_sw10_e500_bs64.json
    config/NYUMets/sweep_sweight/synth_sbcdlnet_sfree_sw20_e500_bs64.json
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
