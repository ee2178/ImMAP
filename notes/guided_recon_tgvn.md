# Guided recon: TGVN baseline, then I2SB (started 2026-11-04)

Pivot: unguided I2SB recon is well explored; the target is now GUIDED recon, and the first goal is
to beat TGVN (Atalik, Chopra, Sodickson, IEEE TMI 2025, arXiv 2501.03021) with an I2SB approach.
TGVN source: `C:\Users\erice\Desktop\TGVN` (github.com/sodicksonlab/TGVN, commit 21f7cfc).

## TGVN in one paragraph

E2E-VarNet cascade plus an "ambiguous space consistency" (ASC) term:

    x^{t+1} = x^t - eta_t A^H(A x^t - k) - mu_t P'_delta(x^t - H(s; gamma_t)) - Phi(x^t; theta_t)
    P'_delta = (I + A^H A / delta^2)^{-1}       soft projector onto small singular values of A

H is a U-Net mapping the side image (side k-space reduced with the MAIN coil maps) to the main
contrast; P'_delta is applied by 10 CG iterations (code: C = S^H F^H (1 + M/delta^2) F S,
diagonal-preconditioned start). delta is ONE learned scalar (init 0.1), eta/mu one scalar per
cascade (`|dc_weight|`, `|asc_weight|`). Coil maps from fastMRI's SensitivityModel on the main
k-space. NormUnet with Cholesky-whitened complex normalization (not fastMRI's per-channel norm).

## Knee experiments to replicate (fastMRI knee, PDFS main <- PD side)

| Exp | Main (PDFS)            | Side (PD)            | Paper TGVN SSIM / PSNR | E2E-VarNet   |
|-----|------------------------|----------------------|------------------------|--------------|
| K1  | random 20x, cf 0.03    | equispaced 2x, cf 0  | 84.92 / 30.92          | 81.33 / 29.30|
| K2  | random 14x, cf 0.03    | equispaced 3x, cf 0  | 85.52 / 31.31          | 83.40 / 30.37|
| K3  | random 6x,  cf 0.05    | equispaced 3x, cf 0  | 88.02 / 32.89          | 87.42 / 32.59|

Protocol (from scripts/main_fastmri.py + appendix): raw measured k-space, NO added noise; masks
from fastMRI `RandomMaskFunc` / `EquiSpacedMaskFunc` with `apply_mask(padding=acq limits)`;
fresh masks every train step (use_seed=False), per-volume seeded at val; target =
`reconstruction_rss`, output center-cropped to it (320x320); loss MS-SSIM-L1 (alpha 0.84,
compensation 1, data_range = volume `max` attr); Adam 3e-4, ExponentialLR 0.98 per epoch, 100
epochs, batch 1/GPU on 4xA100 (10 days); T=14 cascades, 21 chans (~94M params); E2E-VarNet
baseline 30 chans, same T. Eval: SSIM 7x7 uniform, K=(0.01,0.03), data_range = volume max;
Wilcoxon signed-rank vs best baseline. Splits: `data_splits/fastmri/{train,val,test}.csv`
(368/30/30 PD,PDFS volume pairs, patient-level, over fastMRI train+val combined); slices are
paired by index within a pair (assumes equal slice counts).

Code naming trap: in main_fastmri.py "p" = PD (SIDE), "s" = PDFS (MAIN/target).

## Decided 2026-11-04: train under the grid's settings, not TGVN's recipe

Both sides run the one-shot recon grid's protocol and budget (measured k-space + AWGN, online
ESPIRiT, uniform center_frac mask, magnitude NL1+NL2, Adam + cosine over 300k batch-1 steps) --
a like-for-like comparison at reduced compute; TGVN's own recipe can be scaled back up later.
The guide is FULLY SAMPLED for now (how far the side scan can be accelerated is a later study).

- Set 1 (built 2026-11-05): multi-contrast knee, PDFS <- PD, R=8. `torch/guided_recon_knee.sbatch`
  cells from `scripts/make_guided_recon_configs.py` (main R, model, guide R):
  tgvn_R8_g1, tgvn_R8_g2, sbtgvn_R8_g1, sbtgvn_R8_g2, lpdsnet_R8 (the grid's LPDSNet, no guide).
  Optional: varnet_img, lggs. Loader `fastmri_paired`; target = RSS of PDFS.
  - Guide acceleration knob `mri.guide_R` (1 = fully sampled): equispaced, `guide_acs_lines` 0
    (TGVN's side masks have no centre), line phase redrawn per training step, offset 0 at val.
  - TGVN gets a noise-level plane in each regularizer Phi (`cond_sigma`; upstream TGVN has no
    noise input). The plane is sigma / sqrt(tr Cov) of the NormUnet input, so it is in the
    U-Net's normalized units (scale-equivariant; pinned in tests).
  - i2sb-TGVN = `BridgeTGVN`: TGVN cascades on the operator's online-ESPIRiT maps (known-maps
    mode; upstream estimates its own, which a complex bridge cannot use), started at x_t, Phi
    conditioned on the bridge position (s_log) and sigma_n; complex-MSE loss; x0 in the operator's
    phase (x0_source "operator"); x1 = E^H y. No bridge-fidelity term yet (roadmap option 2).
    tau = 0.0234 (calibrate_recon_bridge.py, R=8: RMS 0.0469, tail ceiling 0.060).
  - Known asymmetry: plain TGVN learns its maps, BridgeTGVN uses online ESPIRiT. A
    TGVN(use_smaps=True) control exists if the comparison needs it.
- Set 2 (later): longitudinal -- TGVN and SB-LGGS (the bridge data fidelity).

## Idea (2026-11-05, not built): dual bridge for jointly accelerated contrasts

Scan time with 20x main + 2x guide is dominated by the guide; accelerate BOTH contrasts at a
moderate common R instead. Two bridges, zf contrast A -> full A and zf B -> full B, with each
regressor conditioned on the other bridge's current state, interleaved during sampling. The
motivation: accelerating the guide pushes TGVN to undo the guide's aliasing in one shot, and a
two-stage "reconstruct the guide, then condition on it" pipeline can lose detail; the bridges
refine both together. Notes from discussion:
- Principled form: a JOINT bridge on (x_A, x_B) with independent per-contrast bridge noise; the
  regressors learn E[x0_A | x_t^A, x_t^B, y_A, y_B] (and B's). Training draws both states at the
  same t from the joint pairs, so the reverse process is consistent; interleaving (update A, then
  B on A's newer state) is a Gauss-Seidel sampler variant of it.
- A TGVN-style regressor fits: DC on its own k-space, trust guidance with the OTHER bridge's
  current state as the side image.
- Optional asymmetric times (t_A != t_B) let a bridge condition on a cleaner partner.

## Integration plan (ImMAP side)

1. Paired fastMRI knee loader (new): TGVN CSVs, both raw roots, fastMRI mask functions
   (vendor `fastmri.data.subsample` -- the repo vendors only model code under
   models/e2evarnet/_fastmri), dict batch {main/side k-space + masks, rss target, max, ids}.
2. Port TGVN model (models/tgvn/): TGVN_1S + VarNetImage, their NormUnet; adapter to ImMAP's
   `forward(y, guide=..., E=...)` convention (y = main masked k-space, guide = side masked
   k-space).
3. Training task with TGVN's loss / schedule (MS-SSIM-L1 port, ExponentialLR) and fastMRI-style
   eval (cropped RSS, volume-max SSIM); reuse scripts/wilcoxon_main.py.
4. Replicate K1 (E2E-VarNet + TGVN) before anything else.

Compute is the main risk: their knee run is ~1000 A100-hours (4 GPUs x 10 days).

## Where I2SB fits (to decide)

- TGVN-based bridge regressor: TGVN cascades initialized at x_t (not A^H k), plus a bridge
  term -- DC (measurement) + ASC (side info) + bridge consistency mu0(mu0 x - r)/std_sb^2 + Phi.
  This is notes/immap_sb_roadmap.md "option 2" with TGVN's trust guidance as the side channel.
  TGVN is then the t = 1 limit (bridge term vanishes), so its checkpoint warm-starts the
  regressor and NFE = 1 should reproduce it.
- Endpoint: x1 = E^H k (main zero-filled) vs x1 = side image (PD -> PDFS contrast bridge, k as
  separate DC) vs x1 = TGVN output (refinement bridge).
- Target phase: TGVN is scored on RSS magnitude and learns its own maps; a complex bridge needs a
  fixed complex x0 whose phase matches the operator (see the 2026-11-03 phase-convention finding
  in training/i2sb_recon.py). Magnitude bridge vs fixed online-ESPIRiT maps is open.
- Inference cost: TGVN 468 ms/slice; I2SB at NFE 20 is ~20x that.
