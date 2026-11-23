#!/bin/bash
#
# Dump per-volume reconstructions of the GUIDED knee end-to-end nets for the figure viewer
# (scripts/dump_eval.py; figures/configs/guided_knee_R8.py). The bigpurple counterpart of
# torch/mg_recon_dump.sbatch, for the runs trained on torch and copied over:
#
#     lpdsnet_R8  tgvn_c6_R8_g1  lggs_R8_g1  lggs_guide_w_R8_g1  lggs_guide_conv_R8_g1
#
# under trained_nets/guided_recon/knee/s4_noise-vlow/. Port them first (once):
#     python scripts/port_runs.py trained_nets/guided_recon/knee/s4_noise-vlow/{lpdsnet_R8,tgvn_c6_R8_g1,lggs_R8_g1,lggs_guide_w_R8_g1,lggs_guide_conv_R8_g1}
#
#   mkdir -p bigpurple/logs
#   sbatch bigpurple/guided_recon_dump.sbatch                        # the defaults below
#   N_VOLUMES=4 sbatch bigpurple/guided_recon_dump.sbatch            # a quick look first
#   CKPT=net.ckpt sbatch bigpurple/guided_recon_dump.sbatch          # lowest-train-loss weights
#
# WHAT IT WRITES, per run: <run>/eval_dump/<volume>.h5 with reference, recon, zero_filled, guide
# (the PD scan, fully sampled, coil-combined with the operator's maps), organ_mask,
# sampling_mask and per-slice metrics. Every run sees the same volumes, the same noise draw and
# the same measurement: measured PDFS k-space, online ESPIRiT maps, the run's own val sigma (0:
# the measured k-space already carries the scanner's noise).
#
# CHECKPOINT: net_bestval.ckpt by default (best validation PSNR); CKPT=net.ckpt for the lowest
# training loss. The choice is stored in each file's `ckpt` attribute.
#
# SLICES 12:25 are the ones the nets trained on; widen them and you are looking at slices the
# nets never saw.
#
# LPIPS needs AlexNet weights and compute nodes have no route out: download them once on a
# LOGIN node,
#     conda activate env
#     python -c "from evaluation.metrics import warm_up; warm_up(['lpips'], 'cpu')"
# or skip it with METRICS="psnr ssim nrmse".
#
# Nothing here writes a config or a checkpoint: rerunning it is always safe (it overwrites the
# dump).

#SBATCH --partition=radiology
#SBATCH --job-name=guided-dump
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=EE2178@nyulangone.org
#SBATCH --chdir=/gpfs/scratch/ee2178/ImMAP
#SBATCH --output=/gpfs/scratch/ee2178/ImMAP/bigpurple/logs/%x_%j.out
#SBATCH --error=/gpfs/scratch/ee2178/ImMAP/bigpurple/logs/%x_%j.err

# No `set -euo pipefail` -- it breaks conda activate (unset $PS1 / $CONDA_SHLVL under -u).

RUNS_ROOT="${RUNS_ROOT:-trained_nets/guided_recon/knee/s4_noise-vlow}"
ONLY="${ONLY:-lpdsnet_R8 tgvn_c6_R8_g1 lggs_R8_g1 lggs_guide_w_R8_g1 lggs_guide_conv_R8_g1}"
CKPT="${CKPT:-net_bestval.ckpt}"       # net_bestval.ckpt | net.ckpt
N_VOLUMES="${N_VOLUMES:-20}"           # first N val volumes in sorted order; 0 = every one
VOLUMES="${VOLUMES:-}"                 # explicit PDFS volume names instead (no .h5)
SLICES="${SLICES:-12:25}"
METRICS="${METRICS:-psnr ssim nrmse lpips}"
SEED="${SEED:-1234}"
SIGMA="${SIGMA:-}"                     # empty = each run's val_noise_std (0)
WORKERS="${WORKERS:-4}"
EXTRA="${EXTRA:-}"                     # anything else, e.g. EXTRA="--limit 20"

# module load anaconda3/gpu/2023.09          # uncomment if conda is not already on PATH
CONDA_BASE="$(conda info --base 2>/dev/null)"
[ -n "$CONDA_BASE" ] || { echo "conda not on PATH -- uncomment the module load line" >&2; exit 1; }
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate env || { echo "conda activate env failed" >&2; exit 1; }
export PYTHONNOUSERSITE=1
export IMMAP_CLUSTER=bigpurple

IMMAP_ROOT="${IMMAP_ROOT:-$HOME/scratch/ImMAP}"
cd "$IMMAP_ROOT" || { echo "no such dir: $IMMAP_ROOT" >&2; exit 1; }
mkdir -p bigpurple/logs

# A run not copied over (no config.json / checkpoint) is SKIPPED with a warning; the rest are
# dumped. A run still pointing at torch's k-space roots would die in the loader; say why up front.
FOUND=""
for r in $ONLY; do
    d="$RUNS_ROOT/$r"
    if [ ! -f "$d/config.json" ] || [ ! -f "$d/$CKPT" ]; then
        echo "[skip] $r: no config.json or $CKPT in $d" >&2
        continue
    fi
    [ -f "$d/config.torch.json" ] || echo "[warn] $d has no config.torch.json -- run scripts/port_runs.py on it if it came from torch" >&2
    FOUND="$FOUND $r"
done
[ -n "$FOUND" ] || { echo "none of the runs in ONLY exist under $RUNS_ROOT" >&2; exit 1; }
ONLY="${FOUND# }"

echo "host   : $(hostname)   $(date '+%F %T')"
echo "runs   : $ONLY  (under $RUNS_ROOT, $CKPT)"
echo "volumes: ${VOLUMES:-first $N_VOLUMES}   slices: $SLICES   seed: $SEED   metrics: $METRICS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python scripts/dump_eval.py \
    --runs "$RUNS_ROOT" \
    --only $ONLY --exact \
    --ckpt "$CKPT" \
    --slices "$SLICES" \
    --n-volumes "$N_VOLUMES" \
    ${VOLUMES:+--volumes $VOLUMES} \
    --metrics $METRICS \
    --seed "$SEED" \
    --workers "$WORKERS" \
    ${SIGMA:+--sigma "$SIGMA"} \
    $EXTRA
status=$?

echo "done $(date '+%F %T') (exit $status)"
for r in $ONLY; do du -sh "$RUNS_ROOT/$r/eval_dump" 2>/dev/null || true; done
exit $status
