#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=torch_pr_89_tandon_advanced
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=08:00:00
#SBATCH --job-name=MGDump
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%j.out
#SBATCH --error=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%j.err
#
# Dump per-volume reconstructions for the figure viewer (scripts/dump_eval.py).
#
#   sbatch torch/mg_recon_dump.sbatch                          # brain, then knee
#   ANATOMY_LIST=brain sbatch torch/mg_recon_dump.sbatch       # brain only
#
# LPIPS is computed by default: download its weights once on a login node
# first (see METRICS below).
#
# ONE GPU, the anatomies in turn (2026-10-29: was one array task per anatomy;
# two GPUs queue longer than one does for a job this short). Each anatomy dumps
# every run matching ONLY, so it covers that anatomy's R=8 and R=12 cells (R
# comes from each run's own config, not from here). `sbatch --array=0` / `=1`
# still works and dumps just brain / knee, as before. Output lands in
#
#     trained_nets/mg_recon/<anatomy>/<cell>/eval_dump/<volume>.h5
#
# beside the checkpoint that produced it. Pull those down and open them with
# figures/viewer.py -- see docs/plotting.md.
#
# This is INFERENCE, then an aliasing scan over what was dumped
# (scripts/alias_scan.py, CPU only; SCAN=0 skips it). At the default 60 brain /
# 40 knee volumes expect ~1 h of dumping per anatomy for the eight main runs --
# mostly HDF5 compression, not the nets -- ~1.5 h with the four RAM runs, and a
# few minutes of scan; 8 h for both anatomies is slack. The GPU is worth
# asking for because the unrolled nets are slow on CPU. Nothing here writes a
# config or a checkpoint: a dump is derived data and rerunning it is always safe.
#
# DISK: ~5 MB per dumped slice per run, so 60 brain volumes x 8 slices x 8 runs
# (4 methods x R8/R12) is ~19 GB, and knee about the same. That stays on the
# cluster: the scan writes its ranking and PNGs of the worst slices to
#
#     trained_nets/mg_recon/<anatomy>/alias_scan/scan_R<R>.{txt,csv}, R*_*.png
#
# which is all you need to pull down. rsync individual volumes for the viewer
# once the scan has said which ones are worth looking at.
#
# NOTE: this is a NEW script, not an edit to the frozen Greene launchers. It
# does not source _mg_recon_body.sh -- that body regenerates configs and calls
# train.py, neither of which a dump should ever do.

ANATOMIES=(brain knee)
# Which of them this job dumps, in order. Default all; an array task (old-style
# `sbatch --array=1`) takes just ANATOMIES[task].
ANATOMY_LIST="${ANATOMY_LIST:-}"

# Substring filters on the run path, passed to --only. The trailing underscore
# keeps `lpdsnet_` from also selecting a future cell whose name contains it.
# MGLPDS-Big is mg169v6 for both anatomies (brain's `mglpds` stopgap retired
# 2026-10-21); ONLY="... mglpds_" still dumps it. RAM (exp13) is both its
# cells: `ram_` (random init, "RAM-scratch") and `ramdrunet_` (DRUNet init,
# "RAM"); `ram_` does not match `ramdrunet_R8`. A run not trained yet is
# simply not found.
_ONLY_GIVEN="${ONLY:+1}"
ONLY="${ONLY:-lpdsnet_ mg169v6_ mg81v6_ varnetmaps_ ram_ ramdrunet_}"
# EXACT=1: the ONLY entries are run directory NAMES, matched exactly, so a cell
# is dumped without its tagged probes (<cell>_500ep, <cell>_prof, ...):
#   EXACT=1 ONLY="varnetmaps_R12 lpdsnet_R12 mg81v6_R12" sbatch --array=0 ...
EXACT="${EXACT:-}"

# Slice range and volume count PER ANATOMY, index-matched to ANATOMIES above.
# These are the ranges each cell actually trained on -- brain 0-7 (train
# start_slice 0, end_slice 8) and knee 12-24 (12, 25). Widening them is a
# legitimate thing to do, but know that you are then looking at slices the
# network never saw.
SLICES=("0:8" "12:25")
# HOW MANY VOLUMES. Enough to catch a failure that only some patients
# trigger: if a fraction p of volumes shows it, n volumes miss it with
# probability (1-p)^n -- 60 catches a 5%-of-volumes artifact 95% of the time,
# where the old 6 caught it 26% of the time. Knee has 13 slices to brain's 8,
# so 40 knee volumes is about the same number of slices. N_VOLUMES=<n>
# overrides both; N_VOLUMES=0 dumps EVERY volume in the val split.
NVOLS=(60 40)
# VOLUMES: explicit volume names (no .h5), dumped INSTEAD of the first N --
# e.g. the picks of scripts/rank_slices.py. Run one anatomy per submit:
#   VOLUMES="file1001506 file1001122" EXACT=1 ONLY="..." sbatch --array=1 ...
VOLUMES="${VOLUMES:-}"
# PICKS_DIR: dump the volumes listed in <PICKS_DIR>/<anatomy>.txt (one name per
# line, as scripts/rank_slices.py --pick writes them) -- the interesting ones,
# APPENDED to the existing dumps: each volume is its own <volume>.h5, so the
# first-N files already in eval_dump/ are left alone (a listed volume that is
# already there is rewritten). Unless ONLY is given, this dumps exactly the
# main comparison's runs (EXACT: the eight above and the four RAM runs), not
# the cells' tagged probes:
#   PICKS_DIR=results/picks sbatch torch/mg_recon_dump.sbatch
# The picked volumes get their own noise draw (the generator restarts per run
# over the listed volumes), the same for every run, so columns stay comparable.
PICKS_DIR="${PICKS_DIR:-}"

# LPIPS is IN by default since 2026-10-20 (organ-masked like the rest: the
# background is zeroed in both images, evaluation/metrics.py::lpips). It needs
# pretrained AlexNet weights, and a compute node has no route out, so download
# them ONCE on a LOGIN node (they cache under ~/.cache/torch, which compute
# nodes share):
#     conda activate gcdl
#     python -c "from evaluation.metrics import warm_up; warm_up(['lpips'], 'cpu')"
# Without that, dump_eval stops in its first second with the reason (it warms
# LPIPS up before the sweep). METRICS="psnr ssim nrmse" skips it.
METRICS="${METRICS:-psnr ssim nrmse lpips}"
SEED="${SEED:-1234}"
SIGMA="${SIGMA:-}"                      # empty = each run's own val_noise_std
# How the measurement is made: measured k-space + AWGN by default (2026-10-22),
# as the nets train and as torch/main_eval.sbatch scores them. VAL_KSPACE=simulated
# (the SENSE ground truth through the stored maps, mask and AWGN) is the
# alternative. The dump records it, and the viewer warns when dumps made both
# ways are put side by side.
VAL_KSPACE="${VAL_KSPACE:-measurement_awgn}"
OUT="${OUT:-}"                          # empty = <run>/eval_dump/
WORKERS="${WORKERS:-4}"
SCAN="${SCAN:-1}"                       # 0 = dump only, no aliasing scan
SCAN_TARGET="${SCAN_TARGET:-lpdsnet}"   # the method under suspicion
SCAN_PNG="${SCAN_PNG:-12}"              # render this many of its worst slices

# sbatch does NOT execute this file in place: it copies the script to the node's
# spool dir and runs that copy, so $BASH_SOURCE points at the spool and cannot
# locate the repo. Resolve it explicitly, the same way the training launchers
# do. Override with IMMAP_ROOT=/path/to/ImMAP sbatch ...
for _c in "${IMMAP_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" "$HOME/scratch/ee2178/ImMAP"; do
    if [ -n "${_c}" ] && [ -f "${_c}/scripts/dump_eval.py" ]; then
        IMMAP_ROOT="${_c}"
        break
    fi
done
if [ ! -f "${IMMAP_ROOT:-}/scripts/dump_eval.py" ]; then
    echo "[dump] cannot locate scripts/dump_eval.py (tried IMMAP_ROOT," \
         "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset}, ~/scratch/ee2178/ImMAP)." >&2
    echo "[dump] submit from the repo root, or: IMMAP_ROOT=/path/to/ImMAP sbatch $0" >&2
    exit 1
fi
export IMMAP_ROOT

# `source ~/.bashrc` and `conda activate` both trip over `set -u`, and
# `conda activate` returns non-zero often enough that `set -e` before it kills
# the job. Activate first, then turn the guards on.
source ~/.bashrc
conda activate gcdl
cd "${IMMAP_ROOT}"

set -eo pipefail

mkdir -p logs

# Which anatomies: ANATOMY_LIST, else the array task's one, else all of them.
if [ -z "${ANATOMY_LIST}" ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    if [ "${SLURM_ARRAY_TASK_ID}" -ge "${#ANATOMIES[@]}" ]; then
        echo "[dump] task ${SLURM_ARRAY_TASK_ID} >= ${#ANATOMIES[@]} anatomies -- fix --array"
        echo "       (should be 0-$(( ${#ANATOMIES[@]} - 1 )): ${ANATOMIES[*]})"
        exit 1
    fi
    ANATOMY_LIST="${ANATOMIES[${SLURM_ARRAY_TASK_ID}]}"
fi
ANATOMY_LIST="${ANATOMY_LIST:-${ANATOMIES[*]}}"
# VOLUMES names one anatomy's files; PICKS_DIR is the way to list both.
if [ -n "${VOLUMES}" ] && [ -z "${PICKS_DIR}" ] && [ "$(wc -w <<< "${ANATOMY_LIST}")" -gt 1 ]; then
    echo "[dump] VOLUMES is one anatomy's list -- add ANATOMY_LIST=<anatomy>, or put" \
         "the names in PICKS_DIR/<anatomy>.txt" >&2
    exit 1
fi
_VOLUMES_GIVEN="${VOLUMES}"

echo "[dump] $(date '+%F %T')  node=$(hostname)  anatomies: ${ANATOMY_LIST}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# One anatomy: dump, then scan. Returns non-zero on a failure, so the loop below
# can go on to the next anatomy (set -e does not apply inside a `||` call).
dump_anatomy() {
    local ANATOMY="$1" i SLICE_RANGE N_VOL RUNS_ROOT VOLS="${_VOLUMES_GIVEN}"
    local RUN_ONLY="${ONLY}" RUN_EXACT="${EXACT}" DUMPED _picks
    for i in "${!ANATOMIES[@]}"; do
        [ "${ANATOMIES[$i]}" = "${ANATOMY}" ] && break
    done
    if [ "${ANATOMIES[$i]}" != "${ANATOMY}" ]; then
        echo "[dump] unknown anatomy '${ANATOMY}' (have: ${ANATOMIES[*]})" >&2
        return 1
    fi
    SLICE_RANGE="${SLICES[$i]}"
    N_VOL="${N_VOLUMES:-${NVOLS[$i]}}"
    RUNS_ROOT="trained_nets/mg_recon/${ANATOMY}"

    if [ -n "${PICKS_DIR}" ]; then
        _picks="${PICKS_DIR}/${ANATOMY}.txt"
        if [ ! -s "${_picks}" ]; then
            echo "[dump] PICKS_DIR=${PICKS_DIR}: no ${_picks} -- run" \
                 "scripts/rank_slices.py --from-csv results/main/slices.csv --pick ${PICKS_DIR}" >&2
            return 1
        fi
        VOLS="$(grep -v '^[[:space:]]*\(#\|$\)' "${_picks}" | tr -d '\r' | tr '\n' ' ')"
        if [ -z "${_ONLY_GIVEN}" ]; then
            RUN_ONLY="varnetmaps_R8 lpdsnet_R8 mg81v6_R8 mg169v6_R8 ram_R8 ramdrunet_R8"
            RUN_ONLY+=" varnetmaps_R12 lpdsnet_R12 mg81v6_R12 mg169v6_R12 ram_R12 ramdrunet_R12"
            RUN_EXACT=1
        fi
        echo "[dump] ${ANATOMY}: $(wc -w <<< "${VOLS}") picked volumes from ${_picks}"
    fi

    if [ ! -d "${RUNS_ROOT}" ]; then
        echo "[dump] ${RUNS_ROOT} does not exist -- nothing has been trained for" \
             "${ANATOMY} yet, or the repo root is wrong (IMMAP_ROOT=${IMMAP_ROOT})." >&2
        return 1
    fi

    echo "[dump] ${ANATOMY}: runs=${RUNS_ROOT} only='${RUN_ONLY}' slices=${SLICE_RANGE}" \
         "volumes=${VOLS:-first ${N_VOL}} metrics='${METRICS}' seed=${SEED}" \
         "val_kspace=${VAL_KSPACE}  $(date '+%F %T')"

    python scripts/dump_eval.py \
        --runs "${RUNS_ROOT}" \
        --only ${RUN_ONLY} ${RUN_EXACT:+--exact} \
        --slices "${SLICE_RANGE}" \
        --n-volumes "${N_VOL}" \
        ${VOLS:+--volumes ${VOLS}} \
        --metrics ${METRICS} \
        --seed "${SEED}" \
        --workers "${WORKERS}" \
        ${SIGMA:+--sigma "${SIGMA}"} \
        --val-kspace-type "${VAL_KSPACE}" \
        ${OUT:+--out "${OUT}/${ANATOMY}"} || return 1

    echo "[dump] ${ANATOMY} done $(date '+%F %T')"
    DUMPED="${OUT:+${OUT}/${ANATOMY}}"
    DUMPED="${DUMPED:-${RUNS_ROOT}}"
    du -sh "${DUMPED}"/*/eval_dump 2>/dev/null || true

    if [ "${SCAN}" != "0" ]; then
        echo "[scan] aliasing scan of ${SCAN_TARGET} over ${DUMPED} $(date '+%F %T')"
        python scripts/alias_scan.py \
            --runs "${DUMPED}" \
            --target "${SCAN_TARGET}" \
            --png "${SCAN_PNG}" \
            --out "${DUMPED}/alias_scan" || return 1
        echo "[scan] done $(date '+%F %T') -> ${DUMPED}/alias_scan"
    fi
}

FAILED=()
for ANATOMY in ${ANATOMY_LIST}; do
    dump_anatomy "${ANATOMY}" || FAILED+=("${ANATOMY}")
done
if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "[dump] FAILED: ${FAILED[*]} -- see above; the others finished" >&2
    exit 1
fi
echo "[dump] all done $(date '+%F %T'): ${ANATOMY_LIST}"
