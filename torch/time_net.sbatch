#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=torch_pr_89_tandon_advanced
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=01:00:00
#SBATCH --job-name=TimeNet
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%j.out
#SBATCH --error=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%j.err
#
# Three measurements with nothing of the training loop in them.
#
#   1. ONLINE ESPIRiT KERNEL SIZE on real validation slices
#      (scripts/check_online_espirit_ks.py): time, kernels kept, memory, and
#      agreement with the stored maps and between kernel sizes -- KS="2 4".
#
#   2. NETWORK SPEED (scripts/time_net.py): forward and backward only,
#      GPU-synchronised after warm-up -- Sljiva's `CUDA.@elapsed` around the net
#      call. Brain dimensions, synthetic data, each model's own config
#      (coarse_op, planar convs), plus the optimizer step and projection.
#
#   3. WHERE THE FORWARD TIME GOES (scripts/profile_mg.py --by-level --counts):
#      per network, the inference forward split into conv / FFT / grid transfer
#      / elementwise + overhead, keyed by grid size so the multigrid levels
#      separate. Compare PROPORTIONS -- the instrumented total is inflated. A
#      family that stays the same size between two widths is M-independent.
#
# No training, no checkpoints. Output: this job's log, plus time_net.json, the
# figure time_net.png and breakdown.txt (step 3's tables) under
#   trained_nets/profile/time_net_<jobid>/
# The figure is one column per network: forward, with backward stacked on top.
# Redraw it later (dark theme, pdf) without a GPU:
#   python scripts/time_net.py --from-json <dir>/time_net.json \
#       --fig-out <dir>/time_net_dark.png --fig-theme dark
#
# Submit (from the repo root):   sbatch torch/time_net.sbatch
# Options:
#   R=12 KS="2 4" N_SLICES=6      the ESPIRiT check
#   IMG="640 320" COILS=20 REPS=30
#   MODELS="lpdsnet varnetmaps mglpds mg121v6"
#   PLANAR_STATE=1                step 2 also times each unrolled net with its
#                                 iterates carried as real [re; im] tensors
#                                 (MGLPDSNet.PLANAR_STATE), paired with the normal
#                                 timing; step 3 then profiles THAT representation
#   COMPILE=1                     step 2 also times every net under torch.compile,
#                                 paired with its eager timing (COMPILE=reduce-overhead
#                                 picks a mode). Compiling can take minutes per net:
#                                 narrow MODELS, e.g. "lpdsnet varnetmaps mg81v6"
#   BREAKDOWN="mg169v6 mg81v6"    nets for step 3 (default: MODELS minus VarNet)
#   OPS=1                         step 3 also splits "elementwise / overhead" by
#                                 individual op, with each op's achieved
#                                 bandwidth (launch-bound vs memory-bound)
#   HOST=1                        step 3 also profiles the HOST side of the
#                                 forward (cProfile): which torch calls and
#                                 Python functions the time between kernels goes to
#   NO_WEIGHT_CACHE=1             step 3 with the planar conv weight rebuilt on
#                                 every call, as before 2026-10-07 (the baseline
#                                 for what that cache saves at inference)
#   SKIP_ESPIRIT=1 / SKIP_NETS=1 / SKIP_BREAKDOWN=1   leave a step out

R="${R:-12}"
KS="${KS:-2 4}"
N_SLICES="${N_SLICES:-6}"
IMG="${IMG:-640 320}"
COILS="${COILS:-20}"
REPS="${REPS:-30}"
# V=6 only: exp8 showed V=2/4 lose too much to be worth timing, and mgunet
# needs more work. Any generated cell can still be named in MODELS.
MODELS="${MODELS:-lpdsnet varnetmaps mglpds mg169v6 mg121v6 mg100v6 mg81v6}"

source ~/.bashrc
conda activate gcdl

for _c in "${IMMAP_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" "$HOME/scratch/ee2178/ImMAP"; do
    if [ -n "${_c}" ] && [ -f "${_c}/train.py" ]; then IMMAP_ROOT="${_c}"; break; fi
done
[ -f "${IMMAP_ROOT:-}/train.py" ] || { echo "[time] cannot locate ImMAP." >&2; exit 1; }
cd "${IMMAP_ROOT}"
set -eo pipefail

OUT="trained_nets/profile/time_net_${SLURM_JOB_ID:-local}"
mkdir -p "${OUT}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# The generator's current brain configs, written to this job's own directory so
# nothing under config/ (which the training launchers regenerate) is touched.
# --organ-mask matches the brain launchers; it only affects the data block.
python scripts/make_mg_recon_configs.py --out "${OUT}/gen" --anatomy brain --organ-mask > /dev/null
CFG="${OUT}/gen/brain/mg"

if [ -z "${SKIP_ESPIRIT:-}" ]; then
    echo; echo "================ 1. online ESPIRiT kernel size ================"
    python scripts/check_online_espirit_ks.py --config "${CFG}/lpdsnet_R${R}.json" \
        --ks ${KS} --n "${N_SLICES}" --R "${R}"
fi

if [ -z "${SKIP_NETS:-}" ]; then
    echo; echo "================ 2. network forward / backward ================"
    CFGS=()
    for m in ${MODELS}; do
        f="${CFG}/${m}_R${R}.json"
        [ -f "${f}" ] || { echo "[time] no config for '${m}' at R=${R} (${f})" >&2; exit 1; }
        CFGS+=("${f}")
    done
    COMPILE_FLAG=""
    case "${COMPILE:-}" in
        "") ;;
        1) COMPILE_FLAG="--compile" ;;
        *) COMPILE_FLAG="--compile ${COMPILE}" ;;
    esac
    python scripts/time_net.py --configs "${CFGS[@]}" --size ${IMG} --coils "${COILS}" \
        --reps "${REPS}" --step ${COMPILE_FLAG} ${PLANAR_STATE:+--planar-state} \
        --json-out "${OUT}/time_net.json" \
        --fig-out "${OUT}/time_net.png"
fi

if [ -z "${SKIP_BREAKDOWN:-}" ]; then
    echo; echo "================ 3. where the forward time goes ================"
    # Default: every unrolled net in MODELS. VarNet is left out -- the breakdown
    # instruments this repo's conv / FFT / resample calls, not fastMRI's U-Net.
    if [ -z "${BREAKDOWN:-}" ]; then
        for m in ${MODELS}; do
            case "${m}" in varnet*) ;; *) BREAKDOWN="${BREAKDOWN:+${BREAKDOWN} }${m}" ;; esac
        done
    fi
    : > "${OUT}/breakdown.txt"
    for m in ${BREAKDOWN}; do
        f="${CFG}/${m}_R${R}.json"
        [ -f "${f}" ] || { echo "[time] no config for '${m}' at R=${R} (${f})" >&2; exit 1; }
        # profile_mg.py takes the conv mode as a flag for the whole run, not
        # from the config, so each net runs alone with its own setting.
        PLANAR="$(python -c "import json, sys; print('--planar' if (json.load(open(sys.argv[1])).get('training') or {}).get('complex_conv') == 'planar' else '')" "${f}")"
        python scripts/profile_mg.py --configs "${f}" --size ${IMG} --coils "${COILS}" \
            --R "${R}" --reps "${REPS}" --cudnn-benchmark ${PLANAR} --by-level --counts \
            ${OPS:+--ops} ${HOST:+--host} ${NO_WEIGHT_CACHE:+--no-weight-cache} \
            ${PLANAR_STATE:+--planar-state} \
            | tee -a "${OUT}/breakdown.txt"
    done
fi
