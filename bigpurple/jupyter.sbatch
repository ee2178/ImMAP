#!/bin/bash
#
# JupyterLab on a GPU node.
#
#   sbatch bigpurple/jupyter.sbatch
#   tail -f bigpurple/logs/jupyter_<jobid>.out     # prints the ssh tunnel + the token URL
#
# Override the card without editing this file:
#   PARTITION=gl40s_short GPU_TYPE=l40s sbatch bigpurple/jupyter.sbatch
#   (note: --partition/--gres come from #SBATCH below, so use `sbatch --partition=... --gres=...`
#    for a one-off; the env vars only affect what this script REPORTS, not what SLURM allocates)
#
# THE CARD MATTERS. torch 2.13.0+cu130 ships kernels for CC 7.5 / 8.0 / 8.6 / 9.0 / 10.0 / 12.0.
# A V100 is CC 7.0 and is NOT in that list: allocation succeeds, then the first kernel launch
# dies with `cudaErrorNoKernelImageForDevice`. A100 (8.0) and L40S (8.9, covered by the 8.6
# build) are fine. So this asks for a typed GRES rather than a bare `gpu:1` -- an untyped
# request on a mixed partition can and will hand you a V100.
#
# Check what each queue actually offers:  sinfo -o "%20P %25N %35G"

#SBATCH --partition=radiology
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=jupyter
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=EE2178@nyulangone.org
#SBATCH --chdir=/gpfs/scratch/ee2178/ImMAP
#SBATCH --output=/gpfs/scratch/ee2178/ImMAP/bigpurple/logs/%x_%j.out
#SBATCH --error=/gpfs/scratch/ee2178/ImMAP/bigpurple/logs/%x_%j.err

# No `set -euo pipefail` -- it breaks conda activate (unset $PS1 / $CONDA_SHLVL under -u).

# --- conda ------------------------------------------------------------------------------
CONDA_BASE="$(conda info --base 2>/dev/null)"
if [ -z "$CONDA_BASE" ]; then
    echo "conda not on PATH" >&2
    exit 1
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate env || { echo "conda activate env failed" >&2; exit 1; }
export PYTHONNOUSERSITE=1

# --- the card is usable? -----------------------------------------------------------------
# Not just "is a GPU visible" -- LAUNCH A KERNEL. An unsupported card allocates memory
# happily and only fails when a real op runs, which is how a V100 gets you 20 minutes into a
# notebook before anything complains.
python - <<'PY' || exit 1
import sys, torch
if not torch.cuda.is_available():
    print("no CUDA device visible -- was --gres honoured?", file=sys.stderr)
    sys.exit(1)
name = torch.cuda.get_device_name(0)
cc = torch.cuda.get_device_capability(0)
archs = torch.cuda.get_arch_list()
try:
    _ = (torch.zeros(8, device="cuda") > 0.5).sum().item()      # forces a kernel launch
except Exception as err:
    print(f"UNUSABLE GPU: {name} is compute capability {cc[0]}.{cc[1]}", file=sys.stderr)
    print(f"  this torch ({torch.__version__}) was built for: {archs}", file=sys.stderr)
    print(f"  {err}", file=sys.stderr)
    print("  request a typed GRES on a partition that has it, e.g."
          " --partition=radiology --gres=gpu:a100:1", file=sys.stderr)
    sys.exit(1)
print(f"gpu    : {name}  (CC {cc[0]}.{cc[1]}, kernel launch OK)")
print(f"torch  : {torch.__version__}  built for {archs}")
PY

# --- jupyter ------------------------------------------------------------------------------
# A fixed port collides when two of these run at once; derive one from the job id instead.
PORT=$(( 8888 + SLURM_JOB_ID % 1000 ))
NODE=$(hostname -s)

# The host you ssh into from your laptop -- exactly what you type after `ssh user@`. The login
# nodes moved from BigPurple to UltraViolet, so `bigpurple.nyumc.org` no longer resolves. Set it
# once here, or per job with `LOGIN_HOST=... sbatch bigpurple/jupyter.sbatch`.
LOGIN_HOST="${LOGIN_HOST:-ultraviolet.nyumc.org}"

cat <<EOF

=======================================================================
  On your laptop, open a tunnel:

      ssh -N -L ${PORT}:${NODE}:${PORT} ${USER}@${LOGIN_HOST}

  (submitted from login node: ${SLURM_SUBMIT_HOST:-unknown}. If ${LOGIN_HOST} is not what
   you ssh into, use your usual login address -- the -L part stays the same.)

  then browse to the 127.0.0.1 URL printed below.
=======================================================================

EOF

# -u so the token URL is flushed to the log immediately rather than sitting in a buffer
python -u -m jupyterlab --no-browser --ip=0.0.0.0 --port="${PORT}" --port-retries=0 \
    2>&1 || python -u -m jupyter lab --no-browser --ip=0.0.0.0 --port="${PORT}" --port-retries=0
