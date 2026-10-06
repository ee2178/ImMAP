#!/bin/bash
#
# Every NYUMets bridge and end-to-end experiment as one job array: one config per task, one
# A100 per task. 0-11 are task: i2sb (Schrodinger bridge); 12-15 are task: synthesis (direct
# regression). The runner is task-agnostic -- it picks a config by index and calls train.py.
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
#   12   synth_unet_all.json                  Unet2D  in=3      FLAIR, T1, T2
#   13   synth_unet_otherstudy_pair.json      Unet2D  in=5      ... + other-study T1, CT1
#   14   synth_unet_otherstudy_pair_w1.json   Unet2D  in=9      ... at z-1..z+1
#   15   synth_unet_otherstudy_pair_w2.json   Unet2D  in=13     ... at z-2..z+2
#   16   i2sb_unet_all_bs256.json             SBUnet   C=4      as 1,  LARGE BATCH + EMA
#   17   i2sb_unet_otherstudy_pair_w2_bs256   SBUnet   C=14     as 11, LARGE BATCH + EMA
#   18   synth_cycle_shallowF_w2.json         CycleSynth        as 15 + cycle, F = 1-level UNet
#   19   synth_cycle_sameF_w2.json            CycleSynth        as 15 + cycle, F = G's architecture
#   20   i2sb_sbcdlnet_sfree.json             SBCDLNet          as 3  + S = D_S z        (bridge + learned DC)
#   21   i2sb_sbcdlnet_sgate.json             SBCDLNet          as 3  + S = D (m . z)    (bridge + learned DC)
#   22   i2sb_sbguidedcdl_otherstudy_sfree.json   SBGuidedGroupCDL  FLAIR, T1, T2; guide other-study CT1; free S
#   23   i2sb_sbguidedcdl_otherstudy_sgate.json   SBGuidedGroupCDL  ... gated S
#
# ENHANCEMENT MAP (20-23): the T1 fidelity becomes a LEARNED DATA-CONSISTENCY term through an
# enhancement map S decoded from the same sparse code as CT1 (models/enhancement.py):
#     free   S = D_S z         T1 = (D - D_S) z        S has its own synthesis dictionary
#     gate   S = D (m . z)     T1 = D ((1 - m) . z)    m in [0,1] per atom: which textures enhance
# Trained with  loss(x0_hat, CT1) + s_weight * MSE(S_hat, (CT1 - T1)_+), on top of the usual
# bridge-informed target term. The x0 term uses the sigma weighting of 16-17 (loss_weight "snr",
# loss_weight_max 1000); the S term is NOT sigma-weighted. 3 and 5 predate that and use a uniform
# weight, so they differ from 20-21 / 22-23 in the weighting as well as in S.
# Each validation logs the val/S image and the collapse/* scalars:
#     S = 0                     S_rms_ratio -> 0;  gate_closed -> 1  or  DS_over_D_* -> 0
#     T1 constrains nothing     T1_resid -> 1;     gate_open -> 1    or  Dy_over_D_min -> 0
#     a fidelity switched off   step_xi_min / step_eta_min / step_nu_min -> 0
#     the code died / is dense  code_density -> 0 / -> 1
#     S is not one-sided        S_neg_frac large
# NOT LISTED: the learned-DC-ONLY versions (no bridge term, task: synthesis) --
# config/NYUMets/synth_sbcdlnet_s{free,gate}.json and synth_sbguidedcdl_otherstudy_s{free,gate}.json.
# The configs and the code path (bridge_fidelity: false) still exist; append them to CONFIGS to run.
#   sbatch --array=20-23 bigpurple/i2sb_nyumets_array.sbatch
#
# CYCLE (18-19): G (T1 & co -> CT1) trained jointly with F (CT1, FLAIR, T2 -> T1) on
#     L = loss(G(X), CT1) + cycle_weight * loss(F(G(X)), T1) + backward_weight * loss(F(CT1), T1)
# The last term supervises F directly on the TRUE CT1; without it F learns only from G's outputs
# and can learn to decode whatever G hides in them. G is config 15's Unet2D unchanged, so 15 is
# the no-cycle control; 18 vs 19 asks how big the backward path should be. F never sees T1
# (models/cycle.py refuses it), or both T1 terms would be satisfied by copying. Logged separately
# as train/cycle, train/backward, val/cycle, val/backward.
#   sbatch --array=15,18,19 bigpurple/i2sb_nyumets_array.sbatch
#
# LARGE BATCH + EMA + SIGMA WEIGHTING (16-17) moves our recipe toward the NVlabs reference,
# which trains at an effective batch of 256, samples from an EMA at decay 0.99, and regresses the
# SCORE (xt - x0)/sigma -- an MSE whose implicit weight on the x0 error is 1/sigma^2. Read against
# 1 and 9-11:
#
#                       batch   optimizer steps   samples   lr      EMA     loss weight
#   1, 9-11 (existing)      8           300,000      2.4M   1e-4    no      uniform
#   16-17 (these)         256           300,000     76.8M   2e-4    0.99    snr (1/sigma^2)
#
# 76.8M samples is 32x the compute of the existing runs. That is the point, and it is why these
# are meant to run for DAYS across several allocations -- resume with the same command, the
# checkpoint carries the optimizer, the scheduler, best_loss AND the EMA state, so nothing
# restarts. (A resume that re-seeded the EMA would make net_ema.ckpt worse after every requeue;
# training/i2sb.py reloads it explicitly and says so in the log.)
#
# The 256 is `data.batch_size 32` x `training.accum_steps 8`. If 32 does not fit, or if it leaves
# headroom, change both so the PRODUCT stays 256 -- that is the only number the optimizer sees.
#
# SIGMA WEIGHTING is `loss_weight: "snr"`, the 1/sigma^2 per-sample weight, with
# `loss_weight_max: 1000.0`. The cap is not cosmetic: on the brownian tau=0.1 schedule 1/sigma^2
# spans 25..25,000 and the weight is normalized by the BATCH mean, which is then set by whichever
# sample drew the smallest t. Measured, the normalizer's p99/median is 7.6x at micro-batch 32 --
# one step in a hundred down-weights every sample ~8x for no reason related to the objective.
# Capping at 1000 (sigma >= 0.0316) clamps the top 2.5% of steps and cuts the range to 40x.
# `loss_weight: "snr"` normalizes to batch-mean 1, which is why clip_grad and backtrack_thresh
# keep their meaning; `loss_type: "sigma-scaled-complex-mse"` is the un-normalized alternative and
# would need clip_grad retuned.
#
# ONLY TWO of the four settings are trained at this recipe, to save GPU time: the no-prior
# baseline (16) and the widest context (17, +-2 slices = 5 slices of the prior study). If the prior
# scan helps at all it should show between those two; the w=0 and w=1 midpoints are already covered
# at batch 8 by 9 and 10.
#
# NOTE this is one variable more than 12-15 change: sigma weighting has no analogue in
# `task: synthesis`, which has no timestep at all. Bridge-vs-end-to-end across 16-17 and 12-15 is
# therefore not a single-variable comparison; use 9-11 vs 12-15 for that.
#
# `lr 2e-4` is the least-defensible number here. sqrt-scaling 8 -> 256 would say 5.7e-4 and
# linear scaling 3.2e-3; upstream uses 5e-5 at batch 256 but FINE-TUNES from a pretrained ADM
# checkpoint, so its LR says little about a from-scratch run. If the loss spikes early, halve it
# or add a warmup before blaming the batch size.
#
# Each of these writes TWO checkpoints: net.ckpt (live weights, for resume) and net_ema.ckpt
# (the average -- SAMPLE FROM THIS ONE). Validation metrics are reported under the average.
#   sbatch --array=16,17 bigpurple/i2sb_nyumets_array.sbatch
#
# END-TO-END (12-15) is `task: synthesis`: one forward pass, pred = net(X), no bridge and no
# schedule. The conditioning is lifted from the matching i2sb config, so 12 vs 1, 13 vs 9,
# 14 vs 10 and 15 vs 11 differ in the OBJECTIVE -- and in the one input channel a bridge needs
# and a direct regressor does not (in_chans = C - 1). Loader: nyumets_synth.
#
# 12-15 and 18-19 train at an effective batch of 64 (one physical batch, no accumulation), LR
# 1e-4 (square-root scaled from 2e-4 at 256), EMA 0.99, 300,000 steps -- the bridge runs' recipe
# at a quarter of the batch, since 256 looked like more than this dataset's variation needs.
# The bridge runs 16-17 are still at 256: compare within a batch size, not across.
#   sbatch --array=12-15 bigpurple/i2sb_nyumets_array.sbatch
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
# The LONGITUDINAL subset -- the ones that use another study of the same patient -- is 2, 5, 8-11
# (13-15 for the end-to-end twins):
#   sbatch --array=2,5,8-11 bigpurple/i2sb_nyumets_array.sbatch
# The BRIDGE-vs-END-TO-END comparison is 1, 9-11 against 12-15 -- eight runs, four pairs:
#   sbatch --array=1,9-15 bigpurple/i2sb_nyumets_array.sbatch
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
#SBATCH --array=0-23%4
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
    config/NYUMets/synth_unet_all.json
    config/NYUMets/synth_unet_otherstudy_pair.json
    config/NYUMets/synth_unet_otherstudy_pair_w1.json
    config/NYUMets/synth_unet_otherstudy_pair_w2.json
    config/NYUMets/i2sb_unet_all_bs256.json
    config/NYUMets/i2sb_unet_otherstudy_pair_w2_bs256.json
    config/NYUMets/synth_cycle_shallowF_w2.json
    config/NYUMets/synth_cycle_sameF_w2.json
    config/NYUMets/i2sb_sbcdlnet_sfree.json
    config/NYUMets/i2sb_sbcdlnet_sgate.json
    config/NYUMets/i2sb_sbguidedcdl_otherstudy_sfree.json
    config/NYUMets/i2sb_sbguidedcdl_otherstudy_sgate.json
    # NOT LISTED, to save GPU time: i2sb_unet_otherstudy_pair_bs256.json (w=0) and
    # i2sb_unet_otherstudy_pair_w1_bs256.json (w=1). Both configs still exist; append them
    # here (and widen --array) for the full large-batch width sweep.
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
