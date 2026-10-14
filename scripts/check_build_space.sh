#!/bin/bash
# Will a NEW NYUMets h5 build fit, next to the ones that already exist? READ-ONLY.
#
#   bash scripts/check_build_space.sh <new OUT dir> [an existing build to size it from]
#   bash scripts/check_build_space.sh ~/scratch/datasets/NYUMets_h5_sitk ~/scratch/datasets/NYUMets_h5
#
# The estimate is the size of an EXISTING build made with the same settings (--crop, --compress,
# the same cohort): a build's size depends on how many slices it stores, not on how it registered,
# so a sitk rebuild comes out close to the centroid or fit build. MARGIN (default 1.15) covers the
# difference -- registration changes which slices survive, and zero-filled areas compress
# differently. Whatever is ALREADY in the new OUT is subtracted (a resumed or sharded build).
#
# Checks, in order:
#   1. the estimate: bytes on disk AND file count of the existing build
#   2. free space on the filesystem holding OUT (df) -- the verdict and the exit code
#   3. your quota, printed for you to compare: quota tools differ between clusters and their
#      output is not parsed here. Set QUOTA_FREE_GB=<free quota in GB> to fold it into the verdict.
#
# Exit 0 = fits, 1 = does not fit (by df, or by QUOTA_FREE_GB if given), 2 = usage error.
# du over a few thousand large files is quick; run it on a compute node if the login node is slow.

OUT="${1:-}"
REF="${2:-$HOME/scratch/datasets/NYUMets_h5}"
MARGIN="${MARGIN:-1.15}"
if [ -z "$OUT" ]; then
    echo "usage: bash scripts/check_build_space.sh <new OUT dir> [existing build to size from]" >&2
    exit 2
fi
OUT="$(realpath -m "$OUT")"
if [ ! -d "$REF" ]; then
    echo "no existing build at $REF to size the new one from -- pass one as the second argument" >&2
    exit 2
fi
REF="$(realpath "$REF")"
if [ "$OUT" = "$REF" ]; then
    echo "OUT is the existing build itself ($OUT); this check is for a NEW directory" >&2
    exit 2
fi

gb() { awk -v k="$1" 'BEGIN { printf "%.1f", k / 1024 / 1024 }'; }   # KiB -> GiB

echo "existing build : $REF"
REF_KB=$(du -sk "$REF" | awk '{print $1}')
REF_FILES=$(find "$REF" -type f | wc -l)
REF_H5=$(find "$REF" -name '*_img.h5' | wc -l)
echo "  on disk      : $(gb "$REF_KB") GiB in $REF_FILES files ($REF_H5 studies)"

NEED_KB=$(awk -v k="$REF_KB" -v m="$MARGIN" 'BEGIN { printf "%d", k * m }')
HAVE_KB=0
if [ -d "$OUT" ]; then
    HAVE_KB=$(du -sk "$OUT" | awk '{print $1}')
    echo "new build      : $OUT  (exists, already holds $(gb "$HAVE_KB") GiB)"
else
    echo "new build      : $OUT  (does not exist yet -- the builder creates it)"
fi
REMAIN_KB=$(( NEED_KB > HAVE_KB ? NEED_KB - HAVE_KB : 0 ))
echo "  needs        : $(gb "$NEED_KB") GiB in total (x$MARGIN), $(gb "$REMAIN_KB") GiB still to write"
echo "  files        : ~$REF_FILES (inodes count against quota too)"

# the filesystem that will hold OUT: the nearest existing ancestor
PROBE="$OUT"
while [ ! -d "$PROBE" ]; do PROBE="$(dirname "$PROBE")"; done
AVAIL_KB=$(df -Pk "$PROBE" | awk 'NR == 2 {print $4}')
FS=$(df -Pk "$PROBE" | awk 'NR == 2 {print $6}')
echo
echo "filesystem     : $FS  ($(gb "$AVAIL_KB") GiB free)"

echo
echo "quota (compare against $(gb "$REMAIN_KB") GiB and ~$REF_FILES files):"
if command -v myquota >/dev/null 2>&1; then
    myquota 2>&1 | sed 's/^/  /'
elif command -v mmlsquota >/dev/null 2>&1; then
    mmlsquota -u "$(id -un)" 2>&1 | sed 's/^/  /'
elif command -v lfs >/dev/null 2>&1; then
    lfs quota -hu "$(id -un)" "$FS" 2>&1 | sed 's/^/  /'
elif command -v quota >/dev/null 2>&1; then
    quota -s 2>&1 | sed 's/^/  /'
else
    echo "  (no quota tool found on this machine)"
fi

echo
OK=1
if [ "$AVAIL_KB" -lt "$REMAIN_KB" ]; then
    echo "NO: $FS has $(gb "$AVAIL_KB") GiB free, the build needs $(gb "$REMAIN_KB") GiB more."
    OK=0
else
    echo "filesystem: OK ($(gb "$AVAIL_KB") GiB free >= $(gb "$REMAIN_KB") GiB needed)"
fi
if [ -n "${QUOTA_FREE_GB:-}" ]; then
    Q_KB=$(awk -v g="$QUOTA_FREE_GB" 'BEGIN { printf "%d", g * 1024 * 1024 }')
    if [ "$Q_KB" -lt "$REMAIN_KB" ]; then
        echo "NO: your quota has $QUOTA_FREE_GB GiB free, the build needs $(gb "$REMAIN_KB") GiB more."
        OK=0
    else
        echo "quota: OK ($QUOTA_FREE_GB GiB free >= $(gb "$REMAIN_KB") GiB needed)"
    fi
else
    echo "quota: NOT CHECKED automatically -- read the report above, or rerun with QUOTA_FREE_GB=<n>."
fi
[ "$OK" -eq 1 ] && exit 0 || exit 1
