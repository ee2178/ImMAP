#!/bin/bash
#
# STEP 2 of 2: on a LOGIN (or data-transfer) node -- $ARCHIVE is not mounted on Torch's compute
# nodes -- copy what torch/archive_stage.sbatch staged on scratch into $ARCHIVE, check each copy
# against the checksum taken when it was staged, then drop the staged copy. Run it in tmux:
#
#   tmux new -s archive
#   bash torch/archive_move.sh 2>&1 | tee torch/logs/archive_move_$(date +%m%d_%H%M).log
#   DELETE=1 bash torch/archive_move.sh       # also free the ORIGINAL folders on scratch
#
# Per staged file:
#   1. cp to $ARCHIVE_DIR/<file>.part, rename when complete
#   2. sha256sum -c ON THE ARCHIVE COPY against the staging checksum (step 1 already checked the
#      staged file against the originals byte for byte, so a pass here means the archive copy
#      matches the originals)
#   3. delete the staged copy
#   4. only with DELETE=1, and only after 2 passed: rm -rf the original target on scratch
# Any failure stops before anything of that target is deleted; a rerun resumes (a file already
# in archive whose checksum passes is not copied again).
#
# Restore later:  zstd -dc $ARCHIVE/immap_archive/<name>.tar.zst | tar -xf - -C /scratch/ee2178/<parent>
#            or:  tar -xf $ARCHIVE/immap_archive/<name>.tar -C /scratch/ee2178/<parent>
# (<parent> = the target's folder, e.g. datasets for BraTS; see <name>.target in the archive)

SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/ee2178}"
STAGE_DIR="${STAGE_DIR:-${SCRATCH_ROOT}/archive_staging}"
DELETE="${DELETE:-0}"

set -eo pipefail

if [ -z "${ARCHIVE_DIR:-}" ]; then
    if [ -z "${ARCHIVE:-}" ]; then
        echo "[move] \$ARCHIVE is not set on $(hostname): run this on a login or data-transfer" \
             "node, or pass ARCHIVE_DIR=<path>." >&2
        exit 1
    fi
    ARCHIVE_DIR="${ARCHIVE}/immap_archive"
fi
mkdir -p "${ARCHIVE_DIR}"
echo "[move] host $(hostname)  ${STAGE_DIR} -> ${ARCHIVE_DIR}  DELETE=${DELETE}"

shopt -s nullglob
sums=("${STAGE_DIR}"/*.sha256)
[ ${#sums[@]} -gt 0 ] || { echo "[move] nothing staged in ${STAGE_DIR}" >&2; exit 1; }

for sum in "${sums[@]}"; do
    name="$(basename "${sum}" .sha256)"
    # <name>.tar.zst or <name>.tar; a leading '*' is sha256sum's binary-mode marker
    file="$(awk '{print $2}' "${sum}" | sed 's/^\*//')"
    target="$(cat "${STAGE_DIR}/${name}.target")"
    echo
    echo "[move] === ${file}  (from ${target})"

    if [ -f "${STAGE_DIR}/${file}" ]; then
        if [ -f "${ARCHIVE_DIR}/${file}" ] && \
                (cd "${ARCHIVE_DIR}" && sha256sum -c --quiet "${sum}" 2> /dev/null); then
            echo "[move] already in archive and checks out -- not copying again"
        else
            rm -f "${ARCHIVE_DIR}/${file}.part"
            echo "[move] copying $(du -sh "${STAGE_DIR}/${file}" | cut -f1) ..."
            cp "${STAGE_DIR}/${file}" "${ARCHIVE_DIR}/${file}.part"
            mv "${ARCHIVE_DIR}/${file}.part" "${ARCHIVE_DIR}/${file}"
        fi
    elif [ ! -f "${ARCHIVE_DIR}/${file}" ]; then
        # a checksum with no file behind it, staged or archived: something is wrong; stop
        echo "[move] ${file} is neither in ${STAGE_DIR} nor in ${ARCHIVE_DIR}; stopping" >&2
        exit 1
    fi

    cp "${sum}" "${STAGE_DIR}/${name}.target" "${ARCHIVE_DIR}/"
    echo "[move] checking the archive copy ..."
    (cd "${ARCHIVE_DIR}" && sha256sum -c "${name}.sha256")    # non-zero exit on a mismatch
    rm -f "${STAGE_DIR}/${file}"
    echo "[move] in archive and verified; staged copy removed"

    if [ "${DELETE}" = "1" ]; then
        if [ -e "${SCRATCH_ROOT}/${target}" ]; then
            rm -rf "${SCRATCH_ROOT:?}/${target:?}"
            echo "[move] freed ${SCRATCH_ROOT}/${target}"
        fi
        rm -f "${sum}" "${STAGE_DIR}/${name}.target"
    else
        echo "[move] kept ${SCRATCH_ROOT}/${target} (DELETE=1 frees it)"
    fi
done

echo
echo "[move] done. Archive:"
ls -lh "${ARCHIVE_DIR}"
