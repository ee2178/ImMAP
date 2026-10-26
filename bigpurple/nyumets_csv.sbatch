#!/bin/bash
#
# NYUMets dumps -> the per-slice / per-volume CSVs the tables, graphs and the prior-similarity
# plot are drawn from (scripts/nyumets_volumes_csv.py). CPU only: it reads the HDF5 dumps
# (per-slice metrics, and the registered prior CT1 + target for `prior_mse`) and writes text.
#
#   mkdir -p bigpurple/logs
#   sbatch bigpurple/nyumets_csv.sbatch
#   RUNS="UNet=I2SB_Unet_NYUMets_CT1_from_all_bs32 'UNet + prior w2'=I2SB_Unet_..._w2_bs32" ...
#
# WRITES, under OUT (default results/nyumets_eval):
#   slices.csv    per (net, volume, slice): psnr, ssim, nrmse, prior_mse
#   volumes.csv   per (net, volume): slice means -- what table_main / plot_main_bars read
# Copy both to the laptop and draw everything there:
#   scp <cluster>:<ImMAP>/results/nyumets_eval/{volumes,slices}.csv results/nyumets_eval/
#   python scripts/nyumets_volumes_csv.py --from-csv results/nyumets_eval/slices.csv --tikz
#
# RUNS are LABEL=RUN_DIR_NAME pairs (the label is what the table and plots print), baseline
# first, in the order the table lists them; the default is the two runs in the abstract. Both
# must have been dumped in the same sweep (scripts/dump_eval_nyumets.py) -- checked.
#
# COST: reads every dumped volume once (the prior CT1 and target of the first run only), a few
# MB each -- minutes for ~1000 volumes.

#SBATCH --partition=cpu_short
#SBATCH --job-name=nyumets-csv
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=EE2178@nyulangone.org
#SBATCH --chdir=/gpfs/scratch/ee2178/ImMAP
#SBATCH --output=/gpfs/scratch/ee2178/ImMAP/bigpurple/logs/%x_%j.out
#SBATCH --error=/gpfs/scratch/ee2178/ImMAP/bigpurple/logs/%x_%j.err

# No `set -euo pipefail` -- it breaks conda activate (unset $PS1 / $CONDA_SHLVL under -u).

OUT="${OUT:-results/nyumets_eval}"
ROOT="${ROOT:-trained_nets/nyumets}"

# module load anaconda3/cpu/2023.09          # uncomment if conda is not already on PATH
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

echo "host   : $(hostname)   $(date '+%F %T')"
echo "root   : $ROOT   out: $OUT"

if [ -n "$RUNS" ]; then
    # word-split RUNS the way the shell would read it typed, so quoted labels with spaces survive
    eval "set -- $RUNS"
    echo "runs   : $*"
    python scripts/nyumets_volumes_csv.py --root "$ROOT" --out "$OUT" --profile '' --similarity '' \
        --runs "$@"
else
    echo "runs   : the defaults (scripts/nyumets_volumes_csv.py DEFAULT_RUNS)"
    python scripts/nyumets_volumes_csv.py --root "$ROOT" --out "$OUT" --profile '' --similarity ''
fi
status=$?

echo "done $(date '+%F %T') (exit $status)"
ls -l "$OUT"/slices.csv "$OUT"/volumes.csv 2>/dev/null
exit $status
