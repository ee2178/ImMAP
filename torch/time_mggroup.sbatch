#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=torch_pr_89_tandon_advanced
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=h200
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=02:00:00
#SBATCH --job-name=TimeMGGroup
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%j.out
#SBATCH --error=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%j.err
#
# HOW FAST CAN THE GROUP NET'S ATTENTION RUN?  mggroup169v6, one training step,
# on the node type exp12 trains on.
#
# The windowed attention is nearly all of this net's runtime, and FlexAttention
# is block-sparse: it computes a full B x B score tile for every (query block,
# key block) pair with one allowed pair in it. The window is 2-D and the
# sequence row-major, so at the default B=128 a query on knee's 320x184 latent
# has ~3000 scores computed for the 225 its window uses (7%). Counted exactly,
# against torch's own BlockMask, in tests/test_flex_block_size.py:
#
#     scores per forward     B=128    B=64    B=32    B=16   needed
#     brain 640x320          5.52G   5.14G   3.29G   1.73G    0.54G
#     knee  640x368          7.41G   5.21G   2.97G   1.71G    0.63G
#     knee  640x372          7.68G   5.62G   3.22G   1.90G    0.64G
#
# (108 attention applies per forward -- 36 per grid level, the coarse-correction
# steps included -- of which 86% of the work is on the FINE grid.)
#
# Fewer scores is NOT automatically less time (small tiles cost launches and
# occupancy), which is what this job measures. The arms:
#
#   mg169v6          the local-prox net, for scale
#   mggroup_fb128    exp12's cell as it trains today
#   mggroup_fb64     \
#   mggroup_fb32      } the SAME net and the SAME map -- only `flex_block_size`
#   mggroup_fb16     /  differs (models/circulant_flex.py::_kernel_options hands
#                       the compiled kernel tiles that divide the block)
#   mggroup_triton   attn_backend="triton": the offset-loop kernel
#                    (models/circulant_triton.py), which computes only the 225
#                    needed scores. A DIFFERENT MODEL -- that kernel implements
#                    the phase-invariant `pidistance`, not flex's `distance` --
#                    so this arm prices a change of similarity, not a free swap.
#
# Each arm runs in its own process: an arm that fails to compile (a block size
# this torch refuses, say) is reported and the rest still run.
#
# Read: `fwd+bwd` is the training step's cost; `first s` is compile + warm-up,
# paid once per run and per matrix size; `peak MB` decides which node type the
# cell fits. If a smaller block wins, make it the default where the generator
# sets it (scripts/make_mg_recon_configs.py: `params["flex_block_size"] = 128`)
# BEFORE launching exp12 -- the value is recorded in each run's config, and
# LAUNCH-ONCE treats a changed config as a different run.
#
# Submit (from the repo root):   sbatch torch/time_mggroup.sbatch
# Options:
#   ANATOMY=brain                 default knee; sets the sizes below
#   SIZES="640x368x15 640x372x15" measurement sizes (HxWxCOILS); knee's two widths
#                                 by default, and they are timed MIXED too, as an
#                                 epoch meets them
#   M=81                          the narrow pair, mg81v6 / mggroup81v6 (default 169)
#   R=8  REPS=10  BLOCKS="128 64 32 16"  SKIP_TRITON=1
#   BWD=1                         also profile ONE TRAINING STEP per op and per GPU
#                                 kernel, forward and backward separately
#                                 (scripts/profile_mg.py --ops-step), for mg<M>v6
#                                 and the group cell at block BWD_BLOCK (default
#                                 16). The backward is over half the real step and
#                                 no other tool here splits it. Output is appended
#                                 to <out>/step_profile.txt
#   SKIP_TIMING=1                 with BWD=1: skip the timing arms, profile only
#   An L40S instead:              sbatch --constraint=l40s torch/time_mggroup.sbatch
#                                 The B=128 and B=64 arms may FAIL there, and not
#                                 for GPU memory: the flex kernel's tiles need more
#                                 per-block shared memory than an L40S has ("out of
#                                 resource ... Required: 114688 Hardware limit:
#                                 101376"). B=16 compiles and runs at ~15.7 GB.

ANATOMY="${ANATOMY:-knee}"
M="${M:-169}"                        # 169 or 81: times mg<M>v6 and mggroup<M>v6
R="${R:-8}"
REPS="${REPS:-10}"
BLOCKS="${BLOCKS:-128 64 32 16}"
BWD_BLOCK="${BWD_BLOCK:-16}"
case "${ANATOMY}" in
    knee)  SIZES="${SIZES:-640x368x15 640x372x15}" ;;
    brain) SIZES="${SIZES:-640x320x20}" ;;
    *) echo "[time-mggroup] ANATOMY must be knee or brain, got '${ANATOMY}'" >&2; exit 1 ;;
esac
_first="${SIZES%% *}"; _hw="${_first%x*}"
IMG="${_hw%x*} ${_hw#*x}"; COILS="${_first##*x}"

source ~/.bashrc
conda activate gcdl

for _c in "${IMMAP_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" "$HOME/scratch/ee2178/ImMAP"; do
    if [ -n "${_c}" ] && [ -f "${_c}/train.py" ]; then IMMAP_ROOT="${_c}"; break; fi
done
[ -f "${IMMAP_ROOT:-}/train.py" ] || { echo "[time-mggroup] cannot locate ImMAP." >&2; exit 1; }
cd "${IMMAP_ROOT}"
set -eo pipefail

OUT="trained_nets/profile/time_mggroup_${SLURM_JOB_ID:-local}"
mkdir -p "${OUT}/cfg"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "[time-mggroup] ${ANATOMY} R=${R} sizes=\"${SIZES}\" reps=${REPS} -> ${OUT}"

# the block-size arms are only comparable if the knob is exact
python -m tests.test_flex_block_size

# ---- configs: exp12's own, generated as its launcher generates them ----------
# (--planar-state so each net is timed as the training run would run it; nothing
# under config/ is touched)
python scripts/make_mg_recon_configs.py --out "${OUT}/gen" --anatomy "${ANATOMY}" \
    --attn flex --organ-mask --planar-state > /dev/null
SRC="${OUT}/gen/${ANATOMY}/mg"
for _f in "mg${M}v6" "mggroup${M}v6"; do
    [ -f "${SRC}/${_f}_R${R}.json" ] || { echo "[time-mggroup] no ${_f} cell at R=${R}" \
        "(M=${M}); the generator has mggroup169v6 and mggroup81v6." >&2; exit 1; }
done
cp "${SRC}/mg${M}v6_R${R}.json" "${OUT}/cfg/mg${M}v6.json"
python - "${SRC}/mggroup${M}v6_R${R}.json" "${OUT}/cfg" "${SKIP_TRITON:-}" ${BLOCKS} <<'PY'
import json, sys
src, out, skip_triton, blocks = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
base = json.load(open(src))
assert base["model"]["params"]["attn_backend"] == "flex", base["model"]["params"]
for b in blocks:
    c = json.loads(json.dumps(base))
    c["model"]["params"]["flex_block_size"] = int(b)
    json.dump(c, open(f"{out}/mggroup_fb{b}.json", "w"), indent=1)
if not skip_triton:
    c = json.loads(json.dumps(base))
    p = c["model"]["params"]
    p["attn_backend"], p["sim_fun"] = "triton", "pidistance"
    p.pop("flex_block_size", None)
    json.dump(c, open(f"{out}/mggroup_triton.json", "w"), indent=1)
PY

# the step profile needs its own block size written out, whatever BLOCKS holds
if [ -n "${BWD:-}" ] && [ ! -f "${OUT}/cfg/mggroup_fb${BWD_BLOCK}.json" ]; then
    python - "${SRC}/mggroup${M}v6_R${R}.json" "${OUT}/cfg" "${BWD_BLOCK}" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
c["model"]["params"]["flex_block_size"] = int(sys.argv[3])
json.dump(c, open(f"{sys.argv[2]}/mggroup_fb{sys.argv[3]}.json", "w"), indent=1)
PY
fi

ARMS="mg${M}v6"
for b in ${BLOCKS}; do ARMS="${ARMS} mggroup_fb${b}"; done
[ -n "${SKIP_TRITON:-}" ] || ARMS="${ARMS} mggroup_triton"

set +e
for arm in $([ -n "${SKIP_TIMING:-}" ] || echo "${ARMS}"); do
    echo; echo "================ ${arm} ================"
    python scripts/time_net.py --configs "${OUT}/cfg/${arm}.json" --size ${IMG} \
        --coils "${COILS}" --reps "${REPS}" --step --sizes ${SIZES} \
        --json-out "${OUT}/${arm}.json" 2>&1 | tee "${OUT}/${arm}.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] || echo "[time-mggroup] ${arm} FAILED -- see ${OUT}/${arm}.log"
done
set -e

# ---- one table ---------------------------------------------------------------
if [ -z "${SKIP_TIMING:-}" ]; then
python - "${OUT}" "${M}" ${ARMS} <<'PY'
import json, os, sys
out, M, arms = sys.argv[1], sys.argv[2], sys.argv[3:]
rows = {}
for a in arms:
    f = os.path.join(out, a + ".json")
    if not os.path.exists(f):
        continue
    r = json.load(open(f))["rows"][0]
    fwd, bwd = r["forward"]["median"], r["backward"]["median"]
    rows[a] = (fwd, bwd, fwd + bwd, r.get("peak_mb"), r.get("first_s"), r.get("state"))
print(f"\n================ mggroup{M}v6: one training step ================")
print(f"  {'arm':<16}{'state':>8}{'fwd ms':>9}{'bwd ms':>9}{'fwd+bwd':>10}"
      f"{'vs fb128':>10}{'vs local':>10}{'peak MB':>9}{'first s':>9}")
ref, loc = rows.get("mggroup_fb128"), rows.get(f"mg{M}v6")
for a in arms:
    if a not in rows:
        print(f"  {a:<16}   FAILED / not run")
        continue
    fwd, bwd, tot, peak, first, state = rows[a]
    s = lambda v: f"{v:>9.1f}"                      # time_net.py stores ms
    print(f"  {a:<16}{str(state):>8}{s(fwd)}{s(bwd)}{tot:>10.1f}"
          f"{(f'{ref[2] / tot:.2f}x' if ref else ''):>10}"
          f"{(f'{tot / loc[2]:.2f}x' if loc else ''):>10}"
          f"{(peak or 0):>9.0f}{(first or 0):>9.1f}")
flex = [(v[2], a) for a, v in rows.items() if a.startswith("mggroup_fb")]
if flex and not ref:
    steps, best = 6000 * 50, min(flex)
    print(f"\n  full length = {steps:,} steps: best flex arm ({best[1]}) "
          f"{best[0] * steps / 3.6e6:.0f} h of fwd+bwd alone (fb128 did not run)")
if ref:
    steps = 6000 * 50
    best = min(flex)
    print(f"\n  full length = {steps:,} steps: fb128 {ref[2] * steps / 3.6e6:.0f} h"
          f" of fwd+bwd alone;  best flex arm ({best[1]}) {best[0] * steps / 3.6e6:.0f} h")
    print("  (optimizer, projection, data and the online ESPIRiT maps come on top)")
PY
fi
# ---- where one TRAINING step goes: forward and backward, per op and per kernel ----
# One net per process: profile_mg.py takes the conv mode as a flag for the run.
# Read the family table first (forward next to backward), then the kernel
# tables -- the compiled attention kernels appear there under their own names
# (triton_tem_fused_flex_attention / ..._backward), and in the family table as
# "(outside aten)" or an autograd-node row.
if [ -n "${BWD:-}" ]; then
    : > "${OUT}/step_profile.txt"
    set +e
    for arm in "mg${M}v6" "mggroup_fb${BWD_BLOCK}"; do
        echo; echo "================ training-step profile: ${arm} ================"
        python scripts/profile_mg.py --configs "${OUT}/cfg/${arm}.json" --size ${IMG} \
            --coils "${COILS}" --R "${R}" --reps 5 --cudnn-benchmark --planar \
            --planar-state --ops-step 2>&1 | tee -a "${OUT}/step_profile.txt"
        [ "${PIPESTATUS[0]}" -eq 0 ] || echo "[time-mggroup] step profile of ${arm} FAILED"
    done
    set -e
fi
echo "[time-mggroup] per-arm logs and JSON under ${OUT}"
