#!/bin/bash
#
# Build the NYUMets h5 set with --reg-method sitk: each patient's studies registered on the
# SOURCE NIfTIs with SimpleITK (centre of mass, rigid, then affine; mutual information), every
# other study resampled onto the reference's native grid, THEN the usual build (normalise, crop,
# slice filter). preprocessing/sitk_register.py does the registration.
#
# A job ARRAY over patients: task i builds every NSHARDS-th patient (whole patients -- registration
# is per patient) into the SAME fresh directory, and writes manifest_shard<i>.csv.
#
#   mkdir -p bigpurple/logs
#   sbatch bigpurple/nyumets_h5_sitk.sbatch                         # the whole cohort, 16 shards
#   sbatch --dependency=afterok:<array jobid> --array=0 --time=02:00:00 \
#          --export=ALL,FINISH=1 bigpurple/nyumets_h5_sitk.sbatch          # then merge + diagnose
#
# A small shakeout first is worth it (one shard of a 40-way split, ~1/40 of the patients):
#   sbatch --array=0 --export=ALL,NSHARDS=40,OUT=$HOME/scratch/datasets/NYUMets_h5_sitk_test \
#          bigpurple/nyumets_h5_sitk.sbatch
#
# OUT (default ~/scratch/datasets/NYUMets_h5_sitk) is a NEW directory: the centroid build
# (NYUMets_h5) and the fit build are never touched. The builder creates OUT itself. Its splits:
#   python datasets/make_split_symlinks.py --source ../datasets/NYUMets_h5_sitk \
#          --like ../datasets/NYUMets_h5_split_record.json
# put every patient in the split it already has, so results stay comparable.
#
# SPACE. Every task first runs scripts/check_build_space.sh: OUT is sized from an existing build
# made with the same settings (REF_BUILD, default the centroid build) and the task stops if the
# filesystem cannot hold the rest. Your QUOTA is printed but not judged unless QUOTA_FREE_GB is set.
# SKIP_SPACE_CHECK=1 skips it.
#
# RUNTIME. Registration is ~seconds per pair on small synthetic volumes; on the real 256x256x192
# NIfTIs budget up to a minute per pair on 8 threads -- UNTESTED at that size. With ~3,500 pairs
# over 16 shards that is ~220 pairs, i.e. up to ~4 h of registration per task on top of NIfTI
# loading, inside cpu_short's 12 h. Check the first task's log and raise NSHARDS if it is slower.
# SimpleITK has no GPU path; this is a CPU job.
#
# KNOBS (environment): OUT, ROOT (the NIfTI patientId level), VARIANT (rigid | similarity | affine
# | affine_corr | affine_s25; default affine), NSHARDS (must equal the array size), REF_BUILD,
# QUOTA_FREE_GB, OVERWRITE=1 (rebuild studies OUT already holds), EXTRA (more builder flags).
#
# NEEDS SimpleITK:   conda activate env && pip install SimpleITK

#SBATCH --partition=cpu_short
#SBATCH --job-name=nyumets-h5-sitk
#SBATCH --array=0-15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=EE2178@nyulangone.org
#SBATCH --chdir=/gpfs/scratch/ee2178/ImMAP
#SBATCH --output=/gpfs/scratch/ee2178/ImMAP/bigpurple/logs/%x_%A_%a.out
#SBATCH --error=/gpfs/scratch/ee2178/ImMAP/bigpurple/logs/%x_%A_%a.err

# No `set -euo pipefail` -- it breaks conda activate (unset $PS1 / $CONDA_SHLVL under -u).

# --- conda ------------------------------------------------------------------------------
# module load anaconda3/cpu/2023.09          # uncomment if conda is not already on PATH
CONDA_BASE="$(conda info --base 2>/dev/null)"
if [ -z "$CONDA_BASE" ]; then
    echo "conda not on PATH -- uncomment the module load line above" >&2
    exit 1
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate env || { echo "conda activate env failed" >&2; exit 1; }
export PYTHONNOUSERSITE=1
THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="$THREADS"

IMMAP_ROOT="${IMMAP_ROOT:-$HOME/scratch/ImMAP}"
cd "$IMMAP_ROOT" || { echo "no such directory: $IMMAP_ROOT" >&2; exit 1; }

NYUMETS="${NYUMETS:-$IMMAP_ROOT/../datasets/NYUMets}"
ROOT="${ROOT:-$NYUMETS/data/imaging/patientId}"
OUT="${OUT:-$HOME/scratch/datasets/NYUMets_h5_sitk}"
REF_BUILD="${REF_BUILD:-$HOME/scratch/datasets/NYUMets_h5}"
VARIANT="${VARIANT:-affine}"
NSHARDS="${NSHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"

# --- FINISH: merge the shard manifests and diagnose the result (one task, after the array) ---
if [ -n "${FINISH:-}" ]; then
    python - "$OUT" <<'PY' || exit 1
import csv, glob, os, sys
out = sys.argv[1]
parts = sorted(glob.glob(os.path.join(out, "manifest_shard*.csv")))
rows = [r for p in parts for r in csv.DictReader(open(p, newline=""))]
if not rows:
    sys.exit(f"no manifest_shard*.csv under {out}")
with open(os.path.join(out, "manifest.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
print(f"merged {len(parts)} shard manifest(s): {len(rows)} studies -> {out}/manifest.csv")
PY
    python -m scripts.diagnose_nyumets_registration --root "$OUT" --csv "$OUT/registration_diag.csv"
    exit $?
fi

# --- preflight ---------------------------------------------------------------------------
python -c "import SimpleITK as s; print('SimpleITK', s.Version_VersionString())" \
    || { echo "SimpleITK is not installed in env:  conda activate env && pip install SimpleITK" >&2; exit 1; }
[ -d "$ROOT" ] || { echo "ROOT does not exist: $ROOT" >&2; exit 1; }
if [ -z "${SKIP_SPACE_CHECK:-}" ]; then
    bash scripts/check_build_space.sh "$OUT" "$REF_BUILD" \
        || { echo "SPACE CHECK FAILED (above). Free space, change OUT, or SKIP_SPACE_CHECK=1." >&2; exit 1; }
fi

echo "host    : $(hostname)   threads $THREADS"
echo "shard   : $SHARD / $NSHARDS"
echo "root    : $ROOT"
echo "out     : $OUT"
echo "method  : sitk ($VARIANT)"

OVERWRITE_FLAG=""
[ -n "${OVERWRITE:-}" ] && OVERWRITE_FLAG="--overwrite"

python -m preprocessing.nyumets_h5 \
    --root "$ROOT" \
    --out  "$OUT" \
    --crop 224 \
    --min-brain-frac 0.02 \
    --reg-method sitk \
    --reg-sitk-variant "$VARIANT" \
    --shard "$SHARD" --nshards "$NSHARDS" \
    --compress $OVERWRITE_FLAG ${EXTRA:-}
