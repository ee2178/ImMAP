#!/bin/bash
#
# SimpleITK registration (a sweep of settings) against the --reg-method fit build, on the same
# study pairs, scored in the fit build's own space. READ-ONLY on the data.
# The work is scripts/compare_sitk_fit.py; the registration is preprocessing/sitk_register.py.
#
#   mkdir -p bigpurple/logs
#   sbatch bigpurple/compare_sitk_fit.sbatch                       # 4 shards x the default sweep
#   sbatch --dependency=afterok:<array jobid> --array=0 --time=00:20:00 \
#          --export=ALL,SUMMARIZE=1 bigpurple/compare_sitk_fit.sbatch     # the summary, after it
#
# Or run the summary by hand once every shard is done (cheap, any node):
#   python -m scripts.compare_sitk_fit --out cache/sitk_vs_fit --summarize
#
# WHAT IS SWEPT (VARIANTS in preprocessing/sitk_register.py; VARIANTS="..." to choose):
#   rigid         6 dof, Mattes mutual information
#   similarity    rigid, then + one isotropic scale (7 dof)
#   affine        rigid, then 12 dof -- per-axis scale and shear; the closest to fit's 9 dof
#   affine_corr   as affine, with plain correlation instead of mutual information (T1 onto T1, so
#                 correlation is a legitimate metric -- and it is the one fit optimises)
#   affine_s25    as affine, sampling 25% of the masked voxels instead of 10% (not run by default)
# Every variant registers the SAME pairs; each is scored against fit on identical pixels.
#
# WHAT COMES OUT, under OUT (default cache/sitk_vs_fit), per shard:
#   slices_<i>.csv      per (pair, variant, slice): T1 / CT1 correlation, SimpleITK and fit
#   residuals_<i>.csv   per (pair, method, slice): the diagnostic's leftover rotation / scale /
#                       shift / slice offset, for fit and every variant
#   pairs_<i>.csv       per (pair, method): seconds, the transform's rotation / scales / shift,
#                       fit's reg_fit_ok, the geometry check
#
# COST. Each pair is registered once per variant on the full-resolution NIfTIs (multi-threaded
# CPU), then scored; budget roughly a minute per pair per variant -- untested on the real
# volumes. 30 patients x 2 pairs x 4 variants over 4 shards is ~15 pairs x 4 per task, a few
# hours. SimpleITK has no GPU path, so this is a CPU job.
#
# KNOBS (environment): FIT_ROOT, NIFTI_ROOT, OUT, VARIANTS, SAMPLE (patients, 0 = all), PAIRS
# (per patient), SLICES (residual search per pair), SEED. NSHARDS must equal the array size.
#
# NEEDS SimpleITK in the env:   conda activate env && pip install SimpleITK
# The preflight below fails the task in seconds if it is missing.

#SBATCH --partition=cpu_short
#SBATCH --job-name=sitk-vs-fit
#SBATCH --array=0-3
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

FIT_ROOT="${FIT_ROOT:-$HOME/scratch/datasets/NYUMets_h5_fit}"
NIFTI_ROOT="${NIFTI_ROOT:-/gpfs/data/fenglab/LiFeng/NYUMets/NYUMets/data/imaging/patientId}"
OUT="${OUT:-cache/sitk_vs_fit}"
VARIANTS="${VARIANTS:-rigid similarity affine affine_corr}"
SAMPLE="${SAMPLE:-30}"
PAIRS="${PAIRS:-2}"
SLICES="${SLICES:-4}"
SEED="${SEED:-0}"
NSHARDS="${NSHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"

if [ -n "${SUMMARIZE:-}" ]; then
    echo "summary of $OUT"
    python -m scripts.compare_sitk_fit --out "$OUT" --variants $VARIANTS --summarize
    exit $?
fi

# --- preflight: the things that would otherwise fail an hour in ---------------------------
python -c "import SimpleITK as s; print('SimpleITK', s.Version_VersionString())" \
    || { echo "SimpleITK is not installed in env:  conda activate env && pip install SimpleITK" >&2; exit 1; }
[ -d "$FIT_ROOT" ]   || { echo "FIT_ROOT does not exist: $FIT_ROOT" >&2; exit 1; }
[ -d "$NIFTI_ROOT" ] || { echo "NIFTI_ROOT does not exist: $NIFTI_ROOT" >&2; exit 1; }

echo "host     : $(hostname)   threads $THREADS"
echo "shard    : $SHARD / $NSHARDS"
echo "fit build: $FIT_ROOT"
echo "NIfTIs   : $NIFTI_ROOT"
echo "variants : $VARIANTS"
echo "sample   : $SAMPLE patients (seed $SEED), $PAIRS pair(s) each, $SLICES residual slices"
echo "out      : $OUT"

python -m scripts.compare_sitk_fit --fit-root "$FIT_ROOT" --nifti-root "$NIFTI_ROOT" --out "$OUT" \
    --variants $VARIANTS --sample "$SAMPLE" --pairs "$PAIRS" --slices "$SLICES" --seed "$SEED" \
    --shard "$SHARD" --nshards "$NSHARDS" --threads "$THREADS"
