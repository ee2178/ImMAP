#!/bin/bash
# What is using my disk? READ-ONLY -- deletes and changes nothing.
#
#   bash scripts/disk_usage.sh                  # home + scratch
#   bash scripts/disk_usage.sh /some/dir ...    # only these roots
#   DEPTH=4 TOP=30 bash scripts/disk_usage.sh   # deeper / longer listings
#
# Reports, in order:
#   1. quota (bytes AND file counts -- running out of inodes also reads as
#      "No space left on device")
#   2. per root: largest directories by size, and by file count
#   3. per root: largest single files
#   4. known suspects: conda/pip/torch/HF caches, wandb, checkpoints, logs,
#      leftover .partial files
#   5. the fastMRI preprocessed maps: Walsh vs ESPIRiT, per split
#
# `du` over a big dataset tree takes minutes on a network filesystem; run it
# inside `srun --pty` or a short batch job rather than on a login node if it is
# slow. Output also goes to logs/disk_usage_<timestamp>.txt when run from the
# repo root.

DEPTH="${DEPTH:-3}"
TOP="${TOP:-20}"
USER_NAME="${USER:-$(id -un)}"
DATASETS="${DATASETS:-$HOME/scratch/$USER_NAME/datasets}"

if [ "$#" -gt 0 ]; then
    ROOTS=("$@")
else
    ROOTS=("$HOME" "/scratch/$USER_NAME" "$HOME/scratch/$USER_NAME")
fi

# Resolve symlinks and drop duplicates / missing roots (scratch is often
# reachable both as /scratch/$USER and through a link under $HOME).
declare -A SEEN
REAL_ROOTS=()
for r in "${ROOTS[@]}"; do
    [ -e "$r" ] || continue
    p="$(realpath "$r")"
    [ -n "${SEEN[$p]:-}" ] && continue
    SEEN[$p]=1
    REAL_ROOTS+=("$p")
done

if [ -d logs ] && [ -z "${DISK_USAGE_NO_LOG:-}" ]; then
    LOG="logs/disk_usage_$(date +%Y%m%d_%H%M%S).txt"
    echo "writing a copy to $LOG"
    DISK_USAGE_NO_LOG=1 bash "$0" "$@" 2>&1 | tee "$LOG"
    exit "${PIPESTATUS[0]}"
fi

hr() { printf '\n==== %s ====\n' "$*"; }

hr "quota ($USER_NAME, $(hostname), $(date '+%F %T'))"
if command -v myquota >/dev/null 2>&1; then
    myquota
elif command -v lfs >/dev/null 2>&1; then
    for r in "${REAL_ROOTS[@]}"; do lfs quota -hu "$USER_NAME" "$r" 2>/dev/null; done
else
    quota -s 2>/dev/null || echo "(no quota tool found)"
fi
echo
df -h  "${REAL_ROOTS[@]}" 2>/dev/null | awk 'NR==1 || !seen[$0]++'
echo
df -hi "${REAL_ROOTS[@]}" 2>/dev/null | awk 'NR==1 || !seen[$0]++'

for r in "${REAL_ROOTS[@]}"; do
    hr "$r: largest directories by SIZE (depth $DEPTH)"
    du -h --max-depth="$DEPTH" "$r" 2>/dev/null | sort -rh | head -n "$TOP"

    hr "$r: largest directories by FILE COUNT (depth $DEPTH)"
    du --inodes --max-depth="$DEPTH" "$r" 2>/dev/null | sort -rn | head -n "$TOP" \
        | awk '{printf "%10d  %s\n", $1, $2}'

    hr "$r: largest files (>= 1 GB)"
    find "$r" -xdev -type f -size +1G -printf '%s\t%p\n' 2>/dev/null \
        | sort -u | sort -rn | head -n "$TOP" \
        | awk -F'\t' '{printf "%8.1f GB  %s\n", $1/1024^3, $2}'
done

hr "known suspects"
SUSPECTS=(
    "$HOME/.conda" "$HOME/miniconda3/pkgs" "$HOME/anaconda3/pkgs" "$HOME/miniforge3/pkgs"
    "$HOME/.cache/pip" "$HOME/.cache/torch" "$HOME/.cache/huggingface" "$HOME/.cache"
    "$HOME/.local" "$HOME/.triton" "$HOME/.nv"
)
for r in "${REAL_ROOTS[@]}"; do
    while IFS= read -r d; do SUSPECTS+=("$d"); done < <(
        find "$r" -xdev -maxdepth 4 -type d \
            \( -name wandb -o -name trained_nets -o -name logs -o -name checkpoints \
               -o -name __pycache__ -o -name .ipynb_checkpoints \) -prune -print 2>/dev/null)
done
declare -A DONE
for d in "${SUSPECTS[@]}"; do
    [ -d "$d" ] || continue
    p="$(realpath "$d")"
    [ -n "${DONE[$p]:-}" ] && continue
    DONE[$p]=1
    printf '%8s  %10s files  %s\n' "$(du -sh "$p" 2>/dev/null | cut -f1)" \
        "$(find "$p" -xdev -type f 2>/dev/null | wc -l)" "$p"
done | sort -rh

echo
echo "leftover .partial files (interrupted ESPIRiT writes):"
for r in "${REAL_ROOTS[@]}"; do
    find "$r" -xdev -type f -name '*.partial' -printf '%s\t%p\n' 2>/dev/null
done | sort -u | awk -F'\t' '{n++; s+=$1; printf "  %9.1f MB  %s\n", $1/1024^2, $2}
                   END {printf "  %d files, %.2f GB total\n", n, s/1024^3}'

hr "fastMRI preprocessed maps ($DATASETS/fastmri_preprocessed)"
PRE="$DATASETS/fastmri_preprocessed"
if [ -d "$PRE" ]; then
    printf '%-60s %8s %8s %9s\n' "directory" "size" ".h5" ".partial"
    find "$PRE" -mindepth 1 -maxdepth 3 -type d 2>/dev/null | sort | while IFS= read -r d; do
        n_h5=$(find "$d" -maxdepth 1 -type f -name '*.h5' 2>/dev/null | wc -l)
        n_pt=$(find "$d" -maxdepth 1 -type f -name '*.partial' 2>/dev/null | wc -l)
        [ "$n_h5" -eq 0 ] && [ "$n_pt" -eq 0 ] && continue
        printf '%-60s %8s %8d %9d\n' "${d#$PRE/}" "$(du -sh "$d" 2>/dev/null | cut -f1)" "$n_h5" "$n_pt"
    done
    for f in "$PRE"/*/*.files; do
        [ -f "$f" ] && printf '  saved split list: %s (%d names)\n' "${f#$PRE/}" "$(wc -l < "$f")"
    done
else
    echo "(not found -- set DATASETS=/path/to/datasets)"
fi
