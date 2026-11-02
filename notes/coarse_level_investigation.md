# Is the coarse level solving the coarse problem? (started 2026-11-01)

## The question

MGLPDS coarse levels look like detail images, not solves of the coarse problem, unless
alpha = 1 is fixed, when they are near perfect with slightly worse metrics. So is the net
repurposing the coarse correction for detail, and through which path?

Two explanations, tested apart:

- **Operator.** The cropped coarse Gram differs from Galerkin. The user saw slight ringing
  in Galerkin vs rediscretized Gram checks that could look like aliasing, which could explain
  coarse images drifting during training, and iterates inside the net may be unstructured,
  where the mismatch is worst.
- **Dual path.** `z + alpha_z P(dz)` reaches the image through `-tau B_f` in the next primal
  step. P low-passes dx but not B_f, so a smooth envelope under fine atoms can carry detail.
  Untied FAS copies are a third candidate: they inject `(B_C1 - B_dF) z_c` into the first
  coarse step. Measured at 71% of that step on the old `mglpds_R8`; not tested yet.

## Measured so far (synthetic data only)

`||G - G_gal|| / ||G_gal||`, repo operators, `sinc_vertex`, R=8, 8 smooth coils
(`tests/test_exp19.py::test_folded`; embedded = 92 wide in 96, 160 high):

| Input | L1 crop | L1 folded | L2 crop | L2 folded |
|---|---:|---:|---:|---:|
| 160 grid, noise | .378 | .033 | .420 | .053 |
| 160 grid, structure | .018 | .001 | .032 | .002 |
| embedded, noise | .409 | .074 | .435 | .079 |
| embedded, structure | .021 | .005 | .039 | .005 |

The folded mask is exact without coils (5e-16). Not yet checked on real knee maps.
The FAS argument still holds: `y_c` cancels and a zero fine residual stays put, so a Gram
mismatch changes the correction's gain and shape but cannot create one.

Online ESPIRiT maps have no threshold, so no hard support (user, 2026-11-01). The renorm
halo in `coarse_smaps` does not apply.

## What was built

- `coarse_op="folded"`: `operators/coarse.py::fold_mask`, coarse FFTs with `sqrt(D_c)` as the mask.
- `alpha_z_zero=True`: `models/mg_lpds.py`, no dual correction at any level. alpha_x stays free.
- `scripts/coarse_diag.py`, `torch/coarse_diag.sbatch`: correction spectra and knockouts.
- `torch/exp19_coarse_op.sbatch`: knee R8 cf32, one change each from `mg81v6vtxAnestUcf32`.
  - `mg81v6vtxAnestUgalcf32`: full Galerkin
  - `mg81v6vtxAnestUfoldcf32`: folded mask
  - `mg81v6vtxAnestUaz0cf32`: alpha_z = 0
- Tests: `python -m tests.test_exp19`.

## Submitted?

Set up 2026-11-01. Fill in job ids here. No results yet at the time of writing.

```bash
sbatch torch/exp19_coarse_op.sbatch
sbatch torch/coarse_diag.sbatch
```

## How to read the results

**Diagnostic** (`results/coarse_diag/summary.txt`, baseline net):

- Spectra: "z hf" and "z share hf" say whether the dual path carries the energy above the
  coarse Nyquist. "z" is the first post-smoothing step only, so it is a lower bound.
- Knockouts: dPSNR and drho for `az0`, `ax0`, and per level. Trained nets expect their paths,
  so knockouts overstate a path's value; compare them with each other and with the az0 cell.

**exp19 against the baseline** (val PSNR at the same epoch, `val/levels`, `params/alpha_x`):

| Result | Reading |
|---|---|
| az0 coarse images turn into solves, PSNR holds | Dual path was carrying the detail: drop it or constrain it |
| gal / fold coarse images turn into solves | Operator matters: prefer folded if it matches gal (gal has 1.21x the FLOPs, 2.5x the FFT FLOPs) |
| gal and fold match the baseline | Operator ruled out, as a retrain rather than a swap |
| Nothing changes the images | Look at untied copies, or at a loss that scores the coarse levels |

## Next, if needed

1. Measure the untied-copy forcing: `(B_C1 - B_dF) z_c` against `R r_fine`, per V-cycle.
2. Tied copies with alpha_z = 0.
3. Level loss on the winning cell (`MG_LEVEL_LOSS`, only `vtxBnestsup` has used it so far).
4. Check folded against Galerkin on real knee maps, e.g. a swap with `--coarse-op`.
5. Unrelated, still open: knee R8 alias scan does not support the abstract's "removes LPDSNet's
   artifacts" (over cutoff: VarNet 30, Small 28, LPDSNet 25, Big 21, RAM 13). Flagged images
   in `results/alias_slices_cf32_all` have not been viewed.
