#!/bin/bash
#
# Dump per-volume CT1 syntheses of NYUMets runs for the figure viewer, and score them
# (scripts/dump_eval_nyumets.py) -- the NYUMets counterpart of torch/mg_recon_dump.sbatch.
#
#   mkdir -p bigpurple/logs
#   sbatch bigpurple/nyumets_dump.sbatch                                  # the defaults below
#   N_VOLUMES=10 sbatch bigpurple/nyumets_dump.sbatch                     # a quick look first
#   RUNS="trained_nets/nyumets/<run A> trained_nets/nyumets/<run B>" sbatch bigpurple/nyumets_dump.sbatch
#
# WHAT IT WRITES, per run:   <run>/eval_dump/<volume>.h5   (figures/configs/nyumets_longi.py)
# and every per-slice metric to $CSV, with per-run means and the paired PSNR difference
# against the FIRST run at the end of the log.
#
# DEFAULT RUNS: the bridge UNet without the prior study, and with the prior study's T1/CT1 at
# z-2..z+2 as channels. Both on the val split's MULTI-STUDY patients, the same prior study per
# volume, on the per-volume intensity scale. The first is trained on the fixed scales, the second
# on volume_scale; the script runs each on its own pixels and converts the outputs.
#
# REGISTRATION (REGISTER): by default every (current, prior) pair is re-registered at inference
# from FLAIR, T1 and T2 only -- the build's alignment saw the current CT1 -- and the prior is
# resampled onto the current study. One fit per pair, kept under REG_CACHE so a rerun skips it.
# REGISTER=current_to_prior warps the current study onto the prior instead; REGISTER=none keeps
# the stored alignment.
#
# MASK (MASK_SOURCE): the runs trained with the current T1 & CT1 support, but inference has no
# current CT1. auto (default) masks with the current T1 support AND the registered prior CT1's
# (t1_prior_ct1), or the current T1 alone without registration (t1); config keeps the runs' own.
#
# SLICES are ORIGINAL indices: 80:100 by default. The runs trained on [50, 90), so 90-99 are
# slices they never saw.
#
# COST (estimated, not measured): a bridge sample is val_nfe (20) UNet calls per slice; 60 volumes
# x 20 slices x 2 runs is ~48k calls, minutes on an A100. Writing the gzip'd HDF5 is small: a few
# MB per volume per run.
#
# Nothing here writes a config or a checkpoint: rerunning it is always safe (it overwrites the dump).

#SBATCH --partition=radiology
#SBATCH --job-name=nyumets-dump
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=EE2178@nyulangone.org
#SBATCH --chdir=/gpfs/scratch/ee2178/ImMAP
#SBATCH --output=/gpfs/scratch/ee2178/ImMAP/bigpurple/logs/%x_%j.out
#SBATCH --error=/gpfs/scratch/ee2178/ImMAP/bigpurple/logs/%x_%j.err

# No `set -euo pipefail` -- it breaks conda activate (unset $PS1 / $CONDA_SHLVL under -u).

RUNS="${RUNS:-trained_nets/nyumets/I2SB_Unet_NYUMets_CT1_from_all_bs64 trained_nets/nyumets/I2SB_Unet_NYUMets_CT1_from_all_plus_otherT1CT1_w2_bs64}"
N_VOLUMES="${N_VOLUMES:-60}"           # first N val volumes in sorted order; 0 = every one
SLICES="${SLICES:-80:100}"             # ORIGINAL slice indices lo:hi
UNITS="${UNITS:-volume_scale}"         # volume_scale | none (the fixed scales)
WEIGHTS="${WEIGHTS:-ema}"              # ema (net_ema.ckpt when present) | live
SEED="${SEED:-1234}"
BATCH="${BATCH:-10}"
NFE="${NFE:-}"                         # empty = each run's val_nfe
CSV="${CSV:-results/nyumets_eval/per_slice_${SLURM_JOB_ID:-local}.csv}"
OUT="${OUT:-}"                         # empty = <run>/eval_dump/
REGISTER="${REGISTER:-prior_to_current}"     # prior_to_current | current_to_prior | none (stored)
REG_CACHE="${REG_CACHE:-cache/nyumets_reg}"  # per-pair transforms, reused by later jobs
MASK_SOURCE="${MASK_SOURCE:-auto}"     # auto (t1_prior_ct1 registered / t1) | t1 | t1_prior_ct1 | config
EXTRA="${EXTRA:-}"                     # anything else, e.g. EXTRA="--deterministic"

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

for r in $RUNS; do
    if [ ! -f "$r/config.json" ]; then
        echo "no config.json in $r -- not a trained run (or the path is wrong)" >&2
        exit 1
    fi
done

echo "host   : $(hostname)   $(date '+%F %T')"
echo "runs   : $RUNS"
echo "volumes: $N_VOLUMES   slices: $SLICES   units: $UNITS   weights: $WEIGHTS   seed: $SEED"
echo "registration: $REGISTER (cache $REG_CACHE)   mask: $MASK_SOURCE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python scripts/dump_eval_nyumets.py \
    --runs $RUNS \
    --n-volumes "$N_VOLUMES" \
    --slices "$SLICES" \
    --units "$UNITS" \
    --weights "$WEIGHTS" \
    --seed "$SEED" \
    --batch "$BATCH" \
    --csv "$CSV" \
    --register "$REGISTER" \
    --reg-cache "$REG_CACHE" \
    --mask-source "$MASK_SOURCE" \
    ${NFE:+--nfe "$NFE"} \
    ${OUT:+--out "$OUT"} \
    $EXTRA
status=$?

echo "done $(date '+%F %T') (exit $status)"
for r in $RUNS; do
    du -sh "${OUT:+$OUT/$(basename "$r")}${OUT:-$r/eval_dump}" 2>/dev/null || true
done
exit $status
