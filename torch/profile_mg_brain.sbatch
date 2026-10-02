#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=torch_pr_89_tandon_advanced
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=02:00:00
#SBATCH --job-name=ProfileMGBrain
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%j.out
#SBATCH --error=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%j.err
#
# INFERENCE COST AT BRAIN DIMENSIONS -- the budget for compute-matched multigrid.
#
# Times (forward, no grad, batch 1, cudnn.benchmark ON as train.py runs it) on a
# synthetic 640x320, 20-coil SENSE problem with the grid's Julia mask:
#
#   baselines   lpdsnet (K=30), varnetmaps (the VarNet baseline), and the
#               current mglpds (K=[6,[4,4,6]])
#   V-cycle     mglpds with the K below -- the (V-cycles) x (iters per level)
#   variants    plane, spread so a per-level cost model can be fitted, each
#               timed with BOTH coarse operators (--ab-coarse): Galerkin (the
#               trained models) and rediscretized (coarse SENSE on the coarse
#               grid, MGLPDSNet(coarse_op="rediscretize")):
#                 ms = c0 + c_cycle * K + sum_l c_l * (K * iters[l])
#               i.e. ms per layer at each level, and per-V-cycle overhead
#               (restriction/prolongation and the residual Grams).
#
# Then, for mglpds only: op counts and a per-level time breakdown (how much of a
# coarse layer is its full-resolution Galerkin Gram), and the per-slice online
# ESPIRiT estimate every operator-maps model pays before the network runs.
#
# MEMORY-TRAFFIC ARMS. Nothing in this model is compute-bound: the fitted fine
# cost of 2.515 ms/layer is ~1900 MB of traffic at ~750 GB/s, 86% of an L40S's
# peak, and only ~17% of that traffic is the convolutions and FFTs themselves.
# Two changes cut the rest, and both are timed here:
#
#   planar      every complex conv as ONE real conv on a [re; im] channel
#               stacking, instead of Gauss's three convs wrapped in five
#               elementwise passes (de-interleave twice, xr + xi, two
#               combining subtractions, re-interleave). More FLOPs, far fewer
#               bytes.  models/components.py::_GaussConvNd.COMPLEX_MODE
#   fused prox  the Fenchel clip z*min(1, t/|z|) as one Triton kernel instead
#               of four eager ones.  models/clip_triton.py
#
# Step 2b A/Bs them on mglpds (with an exactness check, since planar
# reassociates every complex multiply). Step 4 re-fits the whole per-level cost
# model with both on, which is where to read the saving: these change the
# per-layer COEFFICIENTS, where K and iters only change the layer count.
#
# Output: this job's log, plus JSON + the fitted model under
#   trained_nets/profile/mg_brain_<jobid>/
# No training, no checkpoints. Shapes only: timings do not depend on data.
#
# Submit (from the repo root):   sbatch torch/profile_mg_brain.sbatch
# Options:  R=16 IMG="640 320" COILS=20 REPS=50 WARMUP=10 sbatch torch/profile_mg_brain.sbatch
#           (R changes only the mask density, which barely moves timings; it is
#            there so the online-map timing sees the grid's ACS.)

R="${R:-16}"
IMG="${IMG:-640 320}"
COILS="${COILS:-20}"
REPS="${REPS:-50}"
WARMUP="${WARMUP:-10}"

# (K_outer, [iters per level]) -- non-coarsest iters must be even (pre/post split).
#   lpdsnet-matched under Galerkin (~30 full-res Grams): 1_8_8_12, 2_4_4_4, 1_20_4_4
#   pixel-weighted ~28 (the rediscretized-operator budget): 1_24_12_24,
#     2_12_6_12, 3_8_4_8, 6_4_2_4 (cycles vs sweeps at fixed work), 3_4_12_24
#     (coarse-heavy)
VARIANTS="1:8,8,12 2:4,4,4 1:20,4,4 1:24,12,24 2:12,6,12 3:8,4,8 6:4,2,4 3:4,12,24"

source ~/.bashrc
conda activate gcdl

for _c in "${IMMAP_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" "$HOME/scratch/ee2178/ImMAP"; do
    if [ -n "${_c}" ] && [ -f "${_c}/train.py" ]; then IMMAP_ROOT="${_c}"; break; fi
done
[ -f "${IMMAP_ROOT:-}/train.py" ] || { echo "[profile] cannot locate ImMAP." >&2; exit 1; }
cd "${IMMAP_ROOT}"
set -eo pipefail

OUT="trained_nets/profile/mg_brain_${SLURM_JOB_ID:-local}"
mkdir -p "${OUT}/configs"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "[profile] R=${R} img=${IMG} coils=${COILS} reps=${REPS} warmup=${WARMUP} -> ${OUT}"

# correctness first: a wrong coarse operator makes the rediscretized timings
# meaningless (set -e stops the job if either fails)
python -m tests.test_coarse_operator
python -m tests.test_mg_coarse_op
python -m tests.test_planar_conv

# ---- configs: the generator's current brain grid, then the V-cycle variants ----
python scripts/make_mg_recon_configs.py --out "${OUT}/gen" --anatomy brain > /dev/null
BASE="${OUT}/gen/brain/mg"
for m in lpdsnet mglpds varnetmaps; do
    cp "${BASE}/${m}_R${R}.json" "${OUT}/configs/${m}.json"
done
python - "${BASE}/mglpds_R${R}.json" "${OUT}/configs" ${VARIANTS} <<'PY'
import json, sys
base, out, specs = sys.argv[1], sys.argv[2], sys.argv[3:]
cfg = json.load(open(base))
for spec in specs:
    k, iters = spec.split(":")
    iters = [int(i) for i in iters.split(",")]
    bad = [i for i in iters[:-1] if i % 2]
    if bad:
        raise SystemExit(f"[profile] {spec}: non-coarsest iters must be even, got {iters}")
    c = json.loads(json.dumps(cfg))
    c["model"]["params"]["K"] = [int(k), iters]
    name = f"mg_{k}_{'_'.join(map(str, iters))}"
    json.dump(c, open(f"{out}/{name}.json", "w"), indent=1)
    print(f"[profile] variant {name}: K={c['model']['params']['K']}")
PY

# ---- 1. wall clock, every config, one process (same GPU state) ----
# --ab-coarse: every V-cycle config is timed twice, from identical weights --
# "optimised" (Galerkin coarse levels: every coarse Gram runs the fine FFTs)
# and "rediscretize" (coarse SENSE on the coarse grid; operators/coarse.py).
CFGS=("${OUT}/configs/mglpds.json" "${OUT}/configs/lpdsnet.json" "${OUT}/configs/varnetmaps.json")
for f in "${OUT}"/configs/mg_*.json; do CFGS+=("${f}"); done
# --no-fused-prox: SoftThreshold.FUSED defaults to TRUE, so without it this
# "baseline" would already carry one of the two traffic cuts, and step 5 would
# understate what the pair is worth together.
python scripts/profile_mg.py --configs "${CFGS[@]}" --size ${IMG} --coils "${COILS}" \
    --R "${R}" --reps "${REPS}" --warmup "${WARMUP}" --cudnn-benchmark --ab-coarse \
    --no-fused-prox --json-out "${OUT}/timings.json"

# ---- 2. where mglpds's time goes, per level, under BOTH coarse operators; ----
#      and the online-map cost
python scripts/profile_mg.py --configs "${OUT}/configs/mglpds.json" "${OUT}/configs/lpdsnet.json" \
    --size ${IMG} --coils "${COILS}" --R "${R}" --reps 20 --warmup "${WARMUP}" \
    --cudnn-benchmark --ab-coarse --no-fused-prox --counts --by-level --time-online-smaps

# ---- 2b. planar conv / fused prox, A/B against the current formulation ----
# Same weights, same seed; the equivalence line at the end is the check that
# these are the same map (planar is fp-roundoff equal, not bit-equal).
# --by-level is now taken INSIDE each arm's context, so the conv / convT call
# counts fall to a third where planar applies -- which is the check that the
# arm is really active. The grid transfers do NOT fall: `_apply_complex` in
# operators/resample.py splits re/im on its own path that COMPLEX_MODE does not
# reach, so `pad` and the transfer convs are unchanged, and are the next target.
#
# Read the CLEAN median, not the breakdown's percentages. The instrumented run
# is launch-bound at a roughly fixed wall time, so a real saving shows up as
# GROWTH in "wrapper overhead" rather than as a smaller elementwise bucket.
python scripts/profile_mg.py --configs "${OUT}/configs/mglpds.json" "${OUT}/configs/lpdsnet.json" \
    --size ${IMG} --coils "${COILS}" --R "${R}" --reps "${REPS}" --warmup "${WARMUP}" \
    --cudnn-benchmark --ab-planar --ab-fused-prox --counts --by-level

# ---- 3. fit the per-level cost model per coarse operator; list the budgets ----
python scripts/fit_mg_cost_model.py "${OUT}/timings.json" "${OUT}/cost_model.json"

# ---- 4. the same sweep and cost model with planar + fused prox ON ----------
# Compare the per-layer coefficients against step 3's: the layer counts are
# identical, so every difference is traffic removed. Both coarse operators are
# timed again, because the two cuts are independent of the coarse Gram.
python scripts/profile_mg.py --configs "${CFGS[@]}" --size ${IMG} --coils "${COILS}" \
    --R "${R}" --reps "${REPS}" --warmup "${WARMUP}" --cudnn-benchmark --ab-coarse \
    --planar --fused-prox --json-out "${OUT}/timings_planar.json"
python scripts/fit_mg_cost_model.py "${OUT}/timings_planar.json" \
    "${OUT}/cost_model_planar.json"

# ---- 5. side by side: what the traffic cuts bought, per config -------------
python - "${OUT}/timings.json" "${OUT}/timings_planar.json" <<'PY'
import json, sys

def load(path):
    d = json.load(open(path))
    return {(r["name"], r["mode"]): r for r in d["records"]}

a, b = load(sys.argv[1]), load(sys.argv[2])
for mode, tag in (("optimised", "galerkin"), ("rediscretize", "rediscretize")):
    keys = [n for (n, m) in a if m == mode and (n, mode) in b]
    if not keys:
        continue
    print(f"\n======== gauss+eager -> planar+fused, {tag} coarse levels ========")
    print(f"  {'config':<16}{'K':<18}{'before':>9}{'after':>9}{'speedup':>9}")
    for n in sorted(keys, key=lambda n: a[(n, mode)]["median"]):
        x, y = a[(n, mode)]["median"], b[(n, mode)]["median"]
        print(f"  {n:<16}{str(a[(n, mode)]['K']):<18}{x:>9.1f}{y:>9.1f}"
              f"{x / y:>8.2f}x")
PY
