#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=torch_pr_89_tandon_advanced
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8GB
#SBATCH --time=12:00:00
#SBATCH --job-name=archive-stage
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%j.out
#SBATCH --error=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%j.err
#
# STEP 1 of 2 of moving rarely-used data to $ARCHIVE, on a COMPUTE node: pack each target into
# one compressed tar in a staging folder ON SCRATCH, verify it against the originals, checksum
# it. Step 2 (torch/archive_move.sh) runs on a LOGIN node -- the only place $ARCHIVE is mounted
# on Torch -- and is then just a copy of a few large files plus a checksum check.
#
#   sbatch torch/archive_stage.sbatch
#   TARGETS="datasets/BraTS" sbatch --export=ALL torch/archive_stage.sbatch
#
# Per target (a path relative to $SCRATCH_ROOT), into $STAGE_DIR:
#   <name>.tar.zst  (or <name>.tar without zstd)  tar | zstd -T<cpus> -<level>; plain tar keeps
#                                                  symlinks as links
#   verify          decompress | tar --compare against the originals, byte for byte
#   <name>.sha256   checksum of the staged file -- what step 2 checks the ARCHIVE copy against
#   <name>.target   the target's path, so step 2 knows what to free
# Built as .part and renamed only once verified: a staged file with its final name is complete.
# Already staged (file + .sha256 present) -> skipped, so a rerun resumes. Nothing is deleted.
#
# Staging needs free scratch for the compressed size (at most the targets' size; the defaults
# total ~444 GB uncompressed). The job checks the space before each target.

SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/ee2178}"
STAGE_DIR="${STAGE_DIR:-${SCRATCH_ROOT}/archive_staging}"
TARGETS="${TARGETS:-datasets/BraTS datasets/fastmri_preprocessed/knee_coil_combined/test datasets/fastmri_preprocessed/brain_T2W_coil_combined/test}"
ZSTD_LEVEL="${ZSTD_LEVEL:-3}"           # 3: zstd's default, ~fast as the disk; 1 if CPU-bound
THREADS="${SLURM_CPUS_PER_TASK:-4}"

set -eo pipefail

if command -v zstd > /dev/null; then
    EXT=".tar.zst"
    pack() { zstd -q -T"${THREADS}" -"${ZSTD_LEVEL}" -c; }
    unpack() { zstd -q -dc "$1"; }
else
    echo "[stage] zstd not found: staging UNCOMPRESSED tars" >&2
    EXT=".tar"
    pack() { cat; }
    unpack() { cat "$1"; }
fi

echo "[stage] host $(hostname), ${THREADS} threads, -> ${STAGE_DIR} (${EXT})"
mkdir -p "${STAGE_DIR}"

for t in ${TARGETS}; do                 # every target first: a typo must not fail half way
    [ -e "${SCRATCH_ROOT}/${t}" ] || { echo "[stage] no such target: ${SCRATCH_ROOT}/${t}" >&2; exit 1; }
done

for t in ${TARGETS}; do
    name="$(echo "${t}" | sed 's|^datasets/||; s|/|_|g')"
    parent="${SCRATCH_ROOT}/$(dirname "${t}")"
    base="$(basename "${t}")"
    out="${STAGE_DIR}/${name}${EXT}"
    echo
    echo "[stage] === ${t} -> ${out}"
    if [ -f "${out}" ] && [ -f "${STAGE_DIR}/${name}.sha256" ]; then
        echo "[stage] already staged and checksummed -- skipping"
        continue
    fi

    need_kb="$(du -sk "${parent}/${base}" | cut -f1)"
    free_kb="$(df -Pk "${STAGE_DIR}" | awk 'NR == 2 {print $4}')"
    echo "[stage] $(du -sh "${parent}/${base}" | cut -f1), $(find "${parent}/${base}" -type f | wc -l)" \
         "files; $((free_kb / 1048576)) GB free where it is staged"
    if [ "${free_kb}" -lt "${need_kb}" ]; then
        echo "[stage] not enough free space to stage ${t} uncompressed-sized; stopping" >&2
        exit 1
    fi

    rm -f "${out}.part"
    tar -cf - -C "${parent}" "${base}" | pack > "${out}.part"
    echo "[stage] written: $(du -sh "${out}.part" | cut -f1)"
    echo "[stage] verifying against the originals ..."
    unpack "${out}.part" | tar --compare -f - -C "${parent}"     # non-zero exit on any mismatch
    (cd "${STAGE_DIR}" && sha256sum "${name}${EXT}.part" | sed "s|${name}${EXT}.part|${name}${EXT}|") \
        > "${STAGE_DIR}/${name}.sha256"
    echo "${t}" > "${STAGE_DIR}/${name}.target"
    mv "${out}.part" "${out}"
    echo "[stage] staged and verified: ${out}"
done

echo
echo "[stage] done. Next, on a LOGIN node (in tmux):"
echo "    bash torch/archive_move.sh                # copy to \$ARCHIVE, check, keep the originals"
echo "    DELETE=1 bash torch/archive_move.sh       # ... and free the originals"
ls -lh "${STAGE_DIR}"
