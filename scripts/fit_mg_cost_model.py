"""
Fit the per-level cost model to a `scripts/profile_mg.py --json-out` file.

    ms = c0 + c_cycle * K + sum_l c_l * (K * iters[l])

i.e. a per-V-cycle overhead (grid transfers and the FAS residual Grams) plus a
per-layer cost at each grid level.  Fitted separately for each coarse operator
present in the file ("optimised" = Galerkin, "rediscretize"), and reported
against the lpdsnet / varnetmaps budgets in the same run.

    python scripts/fit_mg_cost_model.py timings.json cost_model.json

Was inline in torch/profile_mg_brain.sbatch; extracted so the same fit can be
run on a second timings file -- the one taken with `--planar --fused-prox`,
which changes the per-layer coefficients rather than the layer count.
"""

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
