#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=torch_pr_89_tandon_advanced
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=04:00:00
#SBATCH --job-name=WebViewer
#SBATCH --output=/home/ee2178/scratch/ee2178/ImMAP/view_logs/webviewer_%j.out
#SBATCH --error=/home/ee2178/scratch/ee2178/ImMAP/view_logs/webviewer_%j.out
#
# The figure picker (figures/webviewer.py) on a COMPUTE node, reached with
# standard port forwarding -- the way a Jupyter server is. Torch's login nodes
# cannot reach each other's ports, but any login node can reach a compute node,
# so the tunnel no longer has to ride the session the viewer was started from.
#
#   sbatch torch/webviewer.sbatch
#   sbatch --time=08:00:00 torch/webviewer.sbatch          # a longer session
#   CONFIG=figures/configs/main_knee_R8.py sbatch torch/webviewer.sbatch
#
# Once it is running (squeue -u $USER), the log holds the two lines you need:
#   grep -A2 "standard port forwarding" view_logs/webviewer_<jobid>.out
# i.e., in a NEW terminal on the laptop,
#   ssh -N -L <port>:<node>:<port> ee2178@login.torch.hpc.nyu.edu
# and in the browser the printed http://127.0.0.1:<port>/?token=... URL.
#
# CPU only: the viewer reads HDF5 and draws PNGs. Every config under
# figures/configs/ is on the page's Dataset menu. Picks are saved as you go;
# 'Done' in the page (or scancel) ends the job. The server listens on the
# node's network, so it is guarded by a random token, as Jupyter's is.

CONFIG="${CONFIG:-}"                   # where it opens; empty = the first with dumps
# One port per job, so two of these never collide on a node.
PORT="${PORT:-$(( 8800 + ${SLURM_JOB_ID:-0} % 1000 ))}"

for _c in "${IMMAP_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" "$HOME/scratch/ee2178/ImMAP"; do
    if [ -n "${_c}" ] && [ -f "${_c}/figures/webviewer.py" ]; then
        IMMAP_ROOT="${_c}"
        break
    fi
done
if [ ! -f "${IMMAP_ROOT:-}/figures/webviewer.py" ]; then
    echo "[webviewer] cannot locate figures/webviewer.py; submit from the repo root" \
         "or set IMMAP_ROOT" >&2
    exit 1
fi

# Activate first, then the guards (conda activate trips set -u / set -e).
source ~/.bashrc
conda activate gcdl
cd "${IMMAP_ROOT}"
set -eo pipefail

echo "[webviewer] job ${SLURM_JOB_ID} on $(hostname) $(date '+%F %T')"
python -u -m figures.webviewer ${CONFIG:+"${CONFIG}"} \
    --port "${PORT}" --host 0.0.0.0 --login-host login.torch.hpc.nyu.edu
