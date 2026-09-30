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
python scripts/profile_mg.py --configs "${CFGS[@]}" --size ${IMG} --coils "${COILS}" \
    --R "${R}" --reps "${REPS}" --warmup "${WARMUP}" --cudnn-benchmark --ab-coarse \
    --json-out "${OUT}/timings.json"

# ---- 2. where mglpds's time goes, per level, under BOTH coarse operators; ----
#      and the online-map cost
python scripts/profile_mg.py --configs "${OUT}/configs/mglpds.json" "${OUT}/configs/lpdsnet.json" \
    --size ${IMG} --coils "${COILS}" --R "${R}" --reps 20 --warmup "${WARMUP}" \
    --cudnn-benchmark --ab-coarse --counts --by-level --time-online-smaps

# ---- 3. fit the per-level cost model per coarse operator; list the budgets ----
python - "${OUT}/timings.json" "${OUT}/cost_model.json" <<'PY'
import json, sys
import numpy as np

d = json.load(open(sys.argv[1]))
base = {r["name"]: r for r in d["records"] if r["mode"] == "optimised"}
t_lpds = base["lpdsnet"]["median"]
t_var = base["varnetmaps"]["median"]
out = dict(lpdsnet_ms=t_lpds, varnetmaps_ms=t_var, size=d["size"],
           coils=d["coils"], models={})

for mode, tag in (("optimised", "galerkin"), ("rediscretize", "rediscretize")):
    recs = [r for r in d["records"] if r["mode"] == mode]
    rows, y, names = [], [], []
    for r in recs:
        K = r["K"]
        if r["type"] != "MGLPDSNet" or not isinstance(K, list) or len(K[1]) != 3:
            continue
        k, it = K[0], K[1]
        rows.append([1.0, k, k * it[0], k * it[1], k * it[2]])
        y.append(r["median"]); names.append(r["name"])
    if len(rows) < 5:
        print(f"\n[{tag}] only {len(rows)} V-cycle configs -- too few to fit 5 terms")
        continue
    A, y = np.array(rows), np.array(y)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    c0, cc, l0, l1, l2 = coef

    print(f"\n================ cost model: {tag} coarse levels (ms) ================")
    print(f"  ms = {c0:.2f} + {cc:.2f}*K  + {l0:.3f}*n0 + {l1:.3f}*n1 + {l2:.3f}*n2")
    print(f"       (n_l = layers at level l = K * iters[l])")
    print(f"  per layer: fine {l0:.3f}  level-1 {l1:.3f} ({l1 / l0:.2f}x fine)  "
          f"level-2 {l2:.3f} ({l2 / l0:.2f}x fine)")
    print(f"  lpdsnet: {t_lpds / 30:.3f} ms per layer (K=30 plain stack)")
    print(f"  fit residual: max {np.abs(pred - y).max():.2f} ms "
          f"({100 * np.abs(pred - y).max() / y.mean():.1f}% of mean)\n")
    print(f"  budgets: lpdsnet {t_lpds:.1f} ms   varnetmaps {t_var:.1f} ms\n")
    print(f"  {'config':<16}{'K':<18}{'params':>11}{'ms':>9}{'fit':>9}"
          f"{'/lpds':>8}{'/varnet':>9}")
    show = recs + ([base["lpdsnet"], base["varnetmaps"]] if mode != "optimised" else [])
    for r in sorted(show, key=lambda r: r["median"]):
        fit = f"{pred[names.index(r['name'])]:.1f}" if r["name"] in names else ""
        print(f"  {r['name']:<16}{str(r['K']):<18}{r['params']:>11,}{r['median']:>9.1f}"
              f"{fit:>9}{r['median'] / t_lpds:>8.2f}{r['median'] / t_var:>9.2f}")
    out["models"][tag] = dict(c0=c0, per_cycle=cc, per_layer=[l0, l1, l2])

# side by side: what rediscretizing buys each V-cycle config
red = {r["name"]: r for r in d["records"] if r["mode"] == "rediscretize"}
if red:
    print(f"\n================ galerkin -> rediscretize, per config ================")
    print(f"  {'config':<16}{'K':<18}{'galerkin':>10}{'rediscr.':>10}{'speedup':>9}"
          f"{'/lpds':>8}{'/varnet':>9}")
    for n in sorted(red, key=lambda n: red[n]["median"]):
        g, r = base[n]["median"], red[n]["median"]
        print(f"  {n:<16}{str(red[n]['K']):<18}{g:>10.1f}{r:>10.1f}{g / r:>8.2f}x"
              f"{r / t_lpds:>8.2f}{r / t_var:>9.2f}")

json.dump(out, open(sys.argv[2], "w"), indent=1)
print(f"\n[profile] cost models -> {sys.argv[2]}")
PY
