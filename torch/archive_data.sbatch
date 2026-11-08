#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=torch_pr_89_tandon_advanced
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4GB
#SBATCH --time=12:00:00
#SBATCH --job-name=archive-data
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%j.out
#SBATCH --error=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%j.err
#
# Move rarely-used data from $SCRATCH to $ARCHIVE as ONE tar per directory (archive allows 20k
# files, so per-volume .h5 / .nii.gz files must not go in one by one). No GPU: file copying.
#
#   sbatch torch/archive_data.sbatch                 # archive + verify, KEEP the scratch copies
#   DELETE=1 sbatch --export=ALL torch/archive_data.sbatch   # ... and then free the scratch copies
#   TARGETS="datasets/BraTS" DELETE=1 sbatch --export=ALL torch/archive_data.sbatch
#
# Per target (a path relative to $SCRATCH_ROOT):
#   1. tar -cf $ARCHIVE_DIR/<name>.tar.part        plain tar: the maps and .nii.gz do not
#                                                   compress, and compressing costs hours
#   2. tar --compare against the original          byte for byte, symlinks included
#   3. rename .part -> .tar                         a file with the final name is always a
#                                                   complete, verified archive
#   4. only with DELETE=1, and only if 2 passed: rm -rf the scratch copy
# A target whose final .tar already exists is re-verified, not re-copied (a rerun resumes).
# Any failure stops the job before anything is deleted.
#
# Restore:  tar -xf $ARCHIVE/immap_archive/<name>.tar -C $SCRATCH_ROOT
#
# If $ARCHIVE is not mounted on compute nodes, the preflight below fails at once: run the same
# loop from a login / data-transfer node in tmux instead (bash torch/archive_data.sbatch).
# If the job is refused for want of a partition, add the cluster's CPU one (sinfo -s):
#   sbatch --partition=<cpu-partition> torch/archive_data.sbatch

SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/ee2178}"
ARCHIVE_DIR="${ARCHIVE_DIR:-${ARCHIVE:-/archive/ee2178}/immap_archive}"
TARGETS="${TARGETS:-datasets/BraTS datasets/fastmri_preprocessed/knee_coil_combined/test datasets/fastmri_preprocessed/brain_T2W_coil_combined/test}"
DELETE="${DELETE:-0}"

set -eo pipefail

echo "[archive] host $(hostname)  scratch ${SCRATCH_ROOT} -> ${ARCHIVE_DIR}  DELETE=${DELETE}"
mkdir -p "${ARCHIVE_DIR}" || { echo "[archive] cannot create ${ARCHIVE_DIR}: is \$ARCHIVE mounted here?" >&2; exit 1; }
touch "${ARCHIVE_DIR}/.write_test" && rm -f "${ARCHIVE_DIR}/.write_test" \
    || { echo "[archive] ${ARCHIVE_DIR} is not writable from $(hostname)" >&2; exit 1; }

# every target first: a typo must not leave half the list archived and the rest skipped
for t in ${TARGETS}; do
    [ -e "${SCRATCH_ROOT}/${t}" ] || [ -f "${ARCHIVE_DIR}/$(echo "${t}" | sed 's|^datasets/||; s|/|_|g').tar" ] \
        || { echo "[archive] no such target: ${SCRATCH_ROOT}/${t}" >&2; exit 1; }
done

for t in ${TARGETS}; do
    name="$(echo "${t}" | sed 's|^datasets/||; s|/|_|g')"          # e.g. BraTS, knee_coil_combined_test
    parent="${SCRATCH_ROOT}/$(dirname "${t}")"
    base="$(basename "${t}")"
    tarf="${ARCHIVE_DIR}/${name}.tar"
    echo
    echo "[archive] === ${t} -> ${tarf}"

    if [ ! -e "${parent}/${base}" ]; then
        echo "[archive] already gone from scratch; ${tarf} exists -- nothing to do"
        continue
    fi
    echo "[archive] $(du -sh "${parent}/${base}" | cut -f1), $(find "${parent}/${base}" -type f | wc -l) files"

    if [ ! -f "${tarf}" ]; then
        rm -f "${tarf}.part"
        tar -cf "${tarf}.part" -C "${parent}" "${base}"
        echo "[archive] written: $(du -sh "${tarf}.part" | cut -f1)"
        echo "[archive] verifying ..."
        tar --compare -f "${tarf}.part" -C "${parent}"           # non-zero exit on any mismatch
        mv "${tarf}.part" "${tarf}"
    else
        echo "[archive] ${tarf} exists; re-verifying ..."
        tar --compare -f "${tarf}" -C "${parent}"
    fi
    echo "[archive] verified: ${tarf}"

    if [ "${DELETE}" = "1" ]; then
        rm -rf "${parent:?}/${base:?}"
        echo "[archive] freed ${parent}/${base}"
    else
        echo "[archive] kept ${parent}/${base} (DELETE=1 frees it)"
    fi
done

echo
echo "[archive] done. Archive contents:"
ls -lh "${ARCHIVE_DIR}"
