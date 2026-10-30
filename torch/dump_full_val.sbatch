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
#SBATCH --job-name=DumpFullVal
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%A_%a.out
#SBATCH --error=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%A_%a.err
#SBATCH --array=0-5
#
# Dump the WHOLE val split of the main comparison: 6 GPUs, 4 for brain and 2 for
# knee (torch/mg_recon_dump.sbatch does the work; N_VOLUMES=0 is every volume).
#
#   0  brain R8    varnetmaps lpdsnet mg81v6
#   1  brain R8    mg169v6 ram
#   2  brain R12   varnetmaps lpdsnet mg81v6
#   3  brain R12   mg169v6 ram
#   4  knee  R8    varnetmaps lpdsnet mg81v6 mg169v6 ram
#   5  knee  R12   varnetmaps lpdsnet mg81v6 mg169v6 ram
#
# Split by RUN, not by volume: the noise generator restarts per run over its
# volume list, so a run dumped whole draws the same noise as any other split of
# the runs, and every column of a slice sees the same noise.
#
# TIME, scaled from mg_recon_dump.sbatch's estimate (about 7.5 min per run per 60
# brain or 40 knee volumes, not measured at this size): 815 brain volumes about
# 1.7 h per run, 100 knee volumes about 20 min per run, so at most about 5 h a
# task. DISK about 325 GB brain, 65 GB knee; check the quota first.
#
# NO RESUME: dump_eval.py rewrites what it finds, so a failed task reruns from its
# first run. Resubmit just it, e.g. sbatch --array=3 torch/dump_full_val.sbatch.
#
# VAL_KSPACE and OUT as in mg_recon_dump.sbatch (default measured k-space, into
# trained_nets/mg_recon_measured/). LPIPS weights must be cached on a login node
# first; see METRICS there. The scan is off, since a task would scan a half-written
# tree; once all six are done, on a login or CPU node:
#   python scripts/alias_scan.py --runs trained_nets/mg_recon_measured/brain \
#       --target lpdsnet --png 12 --out trained_nets/mg_recon_measured/brain/alias_scan
#
# SUBMIT FROM THE REPO ROOT.

: "${SLURM_ARRAY_TASK_ID:?must be run as an array job (sbatch sets SLURM_ARRAY_TASK_ID)}"
case "${SLURM_ARRAY_TASK_ID}" in
    0) ANATOMY_LIST=brain; ONLY="varnetmaps_R8 lpdsnet_R8 mg81v6_R8" ;;
    1) ANATOMY_LIST=brain; ONLY="mg169v6_R8 ram_R8" ;;
    2) ANATOMY_LIST=brain; ONLY="varnetmaps_R12 lpdsnet_R12 mg81v6_R12" ;;
    3) ANATOMY_LIST=brain; ONLY="mg169v6_R12 ram_R12" ;;
    4) ANATOMY_LIST=knee;  ONLY="varnetmaps_R8 lpdsnet_R8 mg81v6_R8 mg169v6_R8 ram_R8" ;;
    5) ANATOMY_LIST=knee;  ONLY="varnetmaps_R12 lpdsnet_R12 mg81v6_R12 mg169v6_R12 ram_R12" ;;
    *) echo "[dump] task ${SLURM_ARRAY_TASK_ID}: --array must be within 0-5" >&2
       exit 1 ;;
esac
EXACT=1
N_VOLUMES=0
SCAN=0

# sbatch runs a COPY of this file from the node's spool dir, so a
# $BASH_SOURCE-relative path looks for the body next to that copy -- which was
# never shipped. Resolve the checkout explicitly.
for _c in "${IMMAP_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" "$HOME/scratch/ee2178/ImMAP"; do
    if [ -n "${_c}" ] && [ -f "${_c}/torch/mg_recon_dump.sbatch" ]; then
        IMMAP_ROOT="${_c}"
        break
    fi
done
if [ ! -f "${IMMAP_ROOT:-}/torch/mg_recon_dump.sbatch" ]; then
    echo "[dump] cannot locate torch/mg_recon_dump.sbatch (tried IMMAP_ROOT," \
         "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset}, ~/scratch/ee2178/ImMAP)." >&2
    echo "[dump] submit from the repo root, or: IMMAP_ROOT=/path/to/ImMAP sbatch $0" >&2
    exit 1
fi

source "${IMMAP_ROOT}/torch/mg_recon_dump.sbatch"
