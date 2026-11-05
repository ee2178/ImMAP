# ImMAP-SB roadmap (parked 2026-11-03)

Parked while the I2SB recon baselines run. Derivation page: "ImMAP-SB Bridge Annealing"
(claude.ai artifact KGFQqdGMEykiTUoRsH1ATn). Sampler: `sb/immap_sb_ascent.py`.

## Decided

- **Sampler.** ImMAP-SB is the Kadkhodaie-Simoncelli / ImMAP ascent in the debiased state
  `u = (x_t - mu1 x1) / mu0 = x0 + sigma eps`, re-placed on the bridge at the new time each
  step, with the bridge-derived injection `gamma^2 = std_sb(t')^2 (1 - rho^2)`,
  `rho = (1-h)/(1-beta h)`. I2SB's ddpm posterior is the special case `h = 1 - r^2`,
  `beta = 1/(1+r)` (pinned in `tests/test_immap_sb_ascent.py`).
- **Prox weight stays hand-tuned** (`ImMAPProx.c`), as the analogue of ImMAP2's lambda. It only
  exists when the bridge endpoint x1 does NOT already contain the measurement (for linear-Gaussian
  data `E^H y` is sufficient, so x1 = E^H y needs no prox).

## Option 2 -- the end goal: an ImMAP2.5-type bridge regressor

Absorb lambda (and the step sizes) into an unrolled net with the REAL physics operator, as
functions of the bridge time. Per layer, on the two-fidelity objective

    J_t(x) = ||r - mu0 x||^2 / (2 std_sb^2) + ||y - E x||^2 / (2 sigma_y^2) + R(x),   r = x_t - mu1 x1

    x <- x - eta_k(t) [ mu0 (mu0 x - r)/std_sb^2 + lambda_k(t) E^H(E x - y)/sigma_y^2 + B_k z ]
    z <- clip(z + A_k xbar, l_k(t, sigma_y))

- `lambda_k(t) = exp(poly(s_log))`, coefficients initialized at 0 -> lambda = 1 = the Bayes weight
  at init; the learned lambda(t) curve is readable against the hand-tuned ImMAP2 value.
- `eta_k(t)` initialized from the curvature `mu0^2/std_sb^2 + ||E||^2/sigma_y^2` (spans decades).
- Thresholds: the `deg_tau` polynomial in the normalized sigma_eff plus LPDSNet's adaptive sigma_y
  term. `BridgeScheduleMixin` already provides `s_log` / `s_hat`; this is SBCDLNet's `s_mode`
  with the learned operator replaced by the real one.
- The debiased residual makes the two fidelities statistically independent (bridge noise vs
  measurement noise), even with x1 = E^H y -- no double counting of y.
- Limits: t -> 0 identity; t -> 1 the bridge term vanishes and the net IS a one-shot recon net,
  so the trained `trained_nets/mg_recon/*_R8` checkpoints warm-start it via `paths.init_ckpt`.
- With DC inside the net the sampler runs with `prox=None`; only beta / h_0 (pace) remain.

### Staged plan
1. Bridge-aware LPDS regressor in `task: i2sb_recon` (x1 = E^H y), lambda = 1 init + warm start,
   against the UNet baseline.
2. MGLPDS version: lift `training/i2sb_recon.py::_measure`'s refusal of the image-domain
   embedding (zero-pad x_t and x1 onto the embedded grid).
3. Guided: x1 = a prior study (or another contrast), y = this session's k-space. Needs a NYUMets
   loader with simulated k-space; the guide can also drive LGGS group adjacency in R.

### Open
- Two noise levels (sigma_y, sigma(t)): 2-D polynomial, or adaptive sigma_y term as above.
- Under a mask the fused estimate's noise is not white; a scalar threshold is a first cut.
- Option 1 (prox weight from the net's Jacobian, `c = sigma^2 tr(grad_u D)/N`, one Hutchinson
  JVP) was considered and set aside in favour of the hand-tuned weight.

## Also parked
- tau for i2sb_recon (measured protocol, R=8, x0 in the operator's phase convention): brain 0.11
  from one local volume (RMS(x0 - x1) = 0.217 inside the head). Knee still the 0.1 placeholder.
  A sweep around the measured value was proposed. (0.085 was the legacy synthetic protocol.)
