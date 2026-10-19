#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=torch_pr_89_tandon_advanced
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4GB
#SBATCH --time=02:00:00
#SBATCH --job-name=MGDumpTar
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%j.out
#SBATCH --error=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%j.err
#
# Bundle one anatomy's eval dumps (and its aliasing scan) into a single tar to
# scp home, so the bundling runs as a job instead of in a login shell.
#
#     sbatch torch/tar_dumps.sbatch                      # brain -> brain_dumps.tar
#     ANATOMY=knee sbatch torch/tar_dumps.sbatch         # knee  -> knee_dumps.tar
#     WORST=3 sbatch torch/tar_dumps.sbatch              # only the scan's worst
#                                                        # volumes (scripts/alias_pick.py)
#
# Chain it after the dump job, so it starts the moment the dumps exist:
#
#     jid=$(sbatch --parsable --array=0 torch/mg_recon_dump.sbatch)
#     sbatch --dependency=afterok:${jid} torch/tar_dumps.sbatch
#
# The tar lands in the repo root as <anatomy>_dumps.tar (OUT=<path> overrides);
# it is built as <name>.part and renamed only when complete, so a file with the
# final name is never a half-written one. Then, from ImMAP/ at home:
#
#     scp <torch-host>:/home/ee2178/scratch/ee2178/ImMAP/brain_dumps.tar .
#     tar xf brain_dumps.tar
#
# PLAIN tar, no gzip: the HDF5 datasets are already gzip-compressed (re-gzipping
# a dump saves ~0.6%) and compressing ~19 GB would add minutes for nothing.
# No GPU: this is file copying.

ANATOMY="${ANATOMY:-brain}"
WORST="${WORST:-}"                      # empty = every dump; k = alias_pick -k k
OUT="${OUT:-}"                          # empty = <repo>/<anatomy>_dumps.tar

for _c in "${IMMAP_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" "$HOME/scratch/ee2178/ImMAP"; do
    if [ -n "${_c}" ] && [ -d "${_c}/trained_nets/mg_recon" ]; then
        IMMAP_ROOT="${_c}"
        break
    fi
done
if [ ! -d "${IMMAP_ROOT:-}/trained_nets/mg_recon" ]; then
    echo "[tar] cannot locate the ImMAP checkout (tried IMMAP_ROOT," \
         "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset}, ~/scratch/ee2178/ImMAP)." >&2
    exit 1
fi
cd "${IMMAP_ROOT}"

RUNS="trained_nets/mg_recon/${ANATOMY}"
OUT="${OUT:-${IMMAP_ROOT}/${ANATOMY}_dumps.tar}"
LIST="$(mktemp)"
trap 'rm -f "${LIST}"' EXIT

if [ -n "${WORST}" ]; then
    # alias_pick needs numpy; activate before turning the guards on, as the
    # other launchers do (`conda activate` trips `set -eu`).
    source ~/.bashrc
    conda activate gcdl
    set -eo pipefail
    python scripts/alias_pick.py --runs "${RUNS}" -k "${WORST}" > "${LIST}"
else
    set -eo pipefail
    # Every run's dumps, plus the scan's report / CSVs / PNGs if it ran.
    find "${RUNS}" -path "*/eval_dump/*.h5" -type f | sort > "${LIST}"
    if [ -d "${RUNS}/alias_scan" ]; then
        find "${RUNS}/alias_scan" -type f | sort >> "${LIST}"
    fi
fi

N="$(grep -c . "${LIST}" || true)"
if [ "${N}" -eq 0 ]; then
    echo "[tar] nothing to bundle under ${RUNS} -- has the dump job run?" >&2
    exit 1
fi
echo "[tar] ${ANATOMY}: ${N} files -> ${OUT}  $(date '+%F %T')"
grep -o '^.*/eval_dump/' "${LIST}" | sort | uniq -c || true

tar cf "${OUT}.part" -T "${LIST}"
mv -f "${OUT}.part" "${OUT}"

echo "[tar] done $(date '+%F %T'): $(du -h "${OUT}" | cut -f1) in ${OUT}"
echo "[tar] fetch with: scp <torch-host>:${OUT} ."
