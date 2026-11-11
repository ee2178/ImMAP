#!/usr/bin/env python3
"""
Configs for GUIDED reconstruction on fastMRI knee: PDFS main contrast, PD guide (TGVN's setup).

Trained under the SAME settings as the one-shot recon grid -- not TGVN's own recipe -- so the
comparison is like for like at the grid's compute (one training budget for every cell). A cell
is (main R, model, guide R); guide R = 1 is a fully-sampled guide, and unguided models get only
that cell. Runs land in trained_nets/guided_recon/knee/<RUN_TAG>/<cell>.

THE R8 SWEEP (the default cells; torch/guided_recon_knee.sbatch):

    lpdsnet_R8          LPDSNet, NO guide (the unguided baseline, on the same PDFS data)
    lggs_R8_g1          LGGS at Sljiva's grid point, phase-invariant similarity (triton)
    tgvn_c12_R8_g1      TGVN, E2E-VarNet width, 12 / 10 / 8 / 6 cascades,
    tgvn_c10_R8_g1        fully-sampled guide
    tgvn_c8_R8_g1
    tgvn_c6_R8_g1

BRIDGE STARTS (after the TGVN depth is settled; torch/bridge_start_knee.sbatch): i2sb-TGVN at
one depth N from each starting point,

    sbtgvn_cN_guide_R8_g1     x1 = the guide (PD) image: contrast translation
    sbtgvn_cN_adjoint_R8_g1   x1 = zero-filled E^H y
    sbtgvn_cN_acs_R8_g1       x1 = the mask's centre lines only (alias-free low-pass)

each with its own tau from config/knee/guided/bridge_tau/<start>_R<R>_<NORMALIZATION>.json
(scripts/calibrate_recon_bridge.py --json; the launcher writes it on first use).

Also available: tgvn / sbtgvn (12 cascades, the first set's names), varnet_img (TGVN's matched
no-guide E2E-VarNet).

What is inherited, from what scripts/make_mg_recon_configs.py writes TODAY for knee at that R:
    mri          the measured protocol: measured k-space + AWGN, online ESPIRiT, center_frac,
                 adjust_accel, uniform mask (taken from the lpdsnet cell)
    training     steps, epochs, val cadence, noise range, val noise / seed, loss, clip
    optimizer    Adam at the grid LR (LGGS: the grid's group-attention LR, from mggrouplpds)
    scheduler    cosine over the grid's step count
    (LR, eta_min and clip are replaced by Sljiva's for EVERY cell when TRAIN_RECIPE="sljiva")
    data         slice ranges, organ_mask_source (NOT scale_fac: per contrast, set below)
    lpdsnet      its whole model block, from the grid's lpdsnet cell

What is set here:
    data.name    fastmri_paired: TGVN's patient-level PD/PDFS splits
                 (config/knee/guided/splits/), raw k-space from fastMRI train AND val
    target       the OFFLINE SENSE image of the PDFS volume (ESPIRiT, ACS 32, from the full
                 k-space; PDFS_MAPS, written by torch/espirit_knee_pairs.sbatch) -- not TGVN's RSS.
                 Every model is scored on |.| against it; i2sb-TGVN still bridges to the
                 combination under the operator's own (online) maps, x0_source "operator",
                 since the offline maps' phase reference is not the operator's.
    guide        mri.guide_R: the PD scan undersampled by equispaced lines with no centre
                 (TGVN's side mask), redrawn per training step; 1 = fully sampled
    TGVN         upstream defaults (12 cascades, 18 chans) + a noise-level plane in each
                 regularizer (cond_sigma; upstream TGVN takes no noise level), on the
                 OPERATOR'S online ESPIRiT maps (use_smaps) instead of its own SensitivityModel:
                 the same maps i2sb-TGVN and LPDSNet get, so the comparison differs in the
                 method, not in who estimates the coil maps (decided 2026-11-05; at the reduced
                 budget, not learning the maps also makes TGVN's job easier)
    i2sb-TGVN    BridgeTGVN: the same TGVN, on the operator's (online ESPIRiT) maps, started at
                 x_t, conditioned on the bridge position and sigma_n; complex-MSE loss (the
                 bridge state is complex; the grid's magnitude NL1+NL2 cannot train it). Its
                 tau is TAU below, intensity-relative per slice (TAU_SCALE / TAU_REF);
                 calibrate with scripts/calibrate_recon_bridge.py.

    python scripts/make_guided_recon_configs.py                  # the R8 sweep
    python scripts/make_guided_recon_configs.py --models sbtgvn_c8_guide sbtgvn_c8_acs
    python scripts/make_guided_recon_configs.py --guide-accels 1 2 4 --models tgvn_c8
    python scripts/make_guided_recon_configs.py --list-cells
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

ANATOMY = "knee"
ACCELS = (8,)
GUIDE_ACCELS = (1,)
DEFAULT_MODELS = ("lpdsnet", "lggs", "tgvn_c12", "tgvn_c10", "tgvn_c8", "tgvn_c6")
# Run namespace. "s3" = PER-VOLUME normalization + eig-0.9 offline maps (2026-11-06; "s2" was the
# per-contrast fixed scale_fac). A new tag means new run directories: the launchers RESUME any run
# dir that has a checkpoint, so reusing an earlier set's would silently continue those runs.
RUN_TAG = "s3"

# ADDED MEASUREMENT NOISE (the protocol is measured k-space + AWGN): training sigma range and the
# fixed validation sigma, per preset. "grid" is the mg grid's (what every run so far used); the
# noise is there to keep the nets from memorising the scans, so "low" keeps a little of it and
# "none" adds nothing (the scanner's own thermal noise stays in the k-space either way). Picked
# with --noise / the launchers' NOISE; a preset other than "grid" appends "_noise-<preset>" to the
# run tag, so its runs never share (or resume) a dir with runs at another noise level.
NOISE_PRESETS = {
    "grid": None,                                     # inherit the grid's training U[0.01, 0.02]
    "low": dict(noise_std=[0.0, 0.01]),
    "none": dict(noise_std=[0.0, 0.0]),
}
NOISE = "low"                       # the default since 2026-11-10: U[0, 0.01], user's call
# VALIDATION ADDS NO NOISE, whatever the training preset (2026-11-10): the measurement as scanned,
# its own thermal noise only -- the realistic case, and the one the i2sb samplers are judged on
# (the added noise was dragging their val metrics down as training went on, through the sampler
# and the CG step). Nets trained on the grid's U[0.01, 0.02] then see sigma_n = 0 at validation,
# just outside their training range; "low" and "none" include it.
VAL_NOISE_STD = 0.0


def run_tag():
    return RUN_TAG if NOISE == "grid" else f"{RUN_TAG}_noise-{NOISE}"


def _noise_suffix():
    """The noise preset in file names (the bridge tau calibration: the gap includes the noise)."""
    return "" if NOISE == "grid" else f"_noise-{NOISE}"
SPLITS = "config/knee/guided/splits"
# Raw k-space roots per CLUSTER (the CSV pairs are looked up in every root, so the layout does
# not matter). torch: fastMRI's multicoil train + val. bigpurple: the paired multi-contrast knee
# copy at /gpfs/scratch/rl5285/fastmrimutiknee/<split>/multicoil_<split>, reached through the symlink
# ../datasets/fastmri_multiknee (bigpurple/setup_knee_data.sh). IMMAP_CLUSTER picks one; the
# launchers set it. Every other data path (maps, scales, splits) is the same on both clusters.
KNEE_ROOTS = {
    "torch": ["../datasets/fastmri/knee/multicoil_train",
              "../datasets/fastmri/knee/multicoil_val"],
    "bigpurple": [f"../datasets/fastmri_multiknee/{s}/multicoil_{s}"
                  for s in ("train", "val", "test")],
}
CLUSTER = os.environ.get("IMMAP_CLUSTER", "torch")
if CLUSTER not in KNEE_ROOTS:
    raise SystemExit(f"IMMAP_CLUSTER={CLUSTER!r}: expected one of {sorted(KNEE_ROOTS)}")
KSPACE_ROOTS = KNEE_ROOTS[CLUSTER]
# Offline ESPIRiT maps + SENSE images of the PDFS (target) volumes: torch/espirit_knee_pairs.sbatch
# at thresh_eig 0.9 (the 0.95 maps in .../pdfs_espirit_acs32 cut into the anatomy).
PDFS_MAPS = "../datasets/fastmri_preprocessed/knee_coil_combined/pdfs_espirit_acs32_eig0.9"
TARGET = "sense"

# Brownian-bridge temperature for i2sb-TGVN, tau = RMS(x0 - x1)/2 inside the object at the main
# R (written to BOTH i2sb and model.params). Measured 2026-11-05 with
# scripts/calibrate_recon_bridge.py on sbtgvn_R8_g1 (knee PDFS, R=8, measured protocol, 60 train
# slices): RMS 0.0469 -> tau 0.0234; per-slice median 0.043, range [0.027, 0.125]; top-10% tail
# 0.121 -> tau 0.060 (ceiling). Re-measure for another main R. Set to 0.03, a little above the
# gap rule, on the user's call: the measurement noise is real even where coil combination hides
# it in the image, and a slightly wider ladder is the safer side.
# PER-CONTRAST INTENSITY SCALE (2026-11-06): each contrast's raw k-space times the factor that puts
# its pooled FOREGROUND 95th percentile at ~1 (scripts/contrast_histograms.py, notebooks/
# knee_scale_and_espirit.ipynb; train split: PDFS RSS fg p95 6.61e-5 -> 1.51e4, PD 3.42e-4 ->
# 2.92e3). Before, both shared the grid's 5e3, leaving the PDFS target at fg p95 ~0.33 and the PD
# guide at ~1.7. The measurement noise range is NOT rescaled: it is calibrated for [0, 1] images.
# Per-slice fg p95 still spreads ~2x (PDFS) / ~3x (PD) p10->p90, mostly between volumes.
PDFS_SCALE_FAC = 1.5e4
PD_SCALE_FAC = 2.9e3
_OLD_SCALE_FAC = 5e3                # what TAU / TAU_REF below were measured at

# NORMALIZATION (2026-11-06, the current setting): "volume" divides each VOLUME by its own
# foreground p95, per contrast -- the between-volume part is most of the slice-to-slice level
# spread (76% PDFS / 86% PD of the log-variance; within a volume ~15% sd), so after it one noise
# range and one ABSOLUTE bridge tau mean the same on every volume. Statistics from
# scripts/volume_scale_stats.py (VOLUME_SCALES): the TARGET by its ACS-only image's p95 (the
# measurement contains those lines, so a test-time pipeline can compute it), the fully-sampled
# GUIDE by its own full p95. "fixed" = PDFS_SCALE_FAC / PD_SCALE_FAC above (set s2).
NORMALIZATION = "volume"
VOLUME_SCALES = "config/knee/guided/volume_scales.json"
VOLUME_STAT, SIDE_VOLUME_STAT = "acs_p95", "full_p95"

TAU = 0.03 * PDFS_SCALE_FAC / _OLD_SCALE_FAC
# sampling-time CG data consistency (I2SB-Recon, arXiv 2411.14269; sb/i2sb_recon.py): the
# paper's defaults. Validation only -- val/* is WITH it, val/nodc_* the plain sampler on the same
# noise draws (val cost x2). dc_iters 0 turns it off.
I2SB_DC = {"dc_iters": 7, "dc_lam": 7e-5}
# Hard support at validation sampling: every x0 prediction is multiplied by the support of the
# OPERATOR's own online ESPIRiT maps at eigenvalue 0.9 -- the offline target's threshold, so the
# recon's background is 0 where the target's is. Test-time computable (the measurement alone).
# support_state also masks the bridge state after every step (off: the net then sees states
# like its training ones).
I2SB_SUPPORT = {"support_thresh": 0.9, "support_state": False}
# Bridge start (sb/i2sb_recon.py): "adjoint" = zero-filled E^H y (aliased); "acs" = only the
# mask's contiguous centre block, an alias-free low-pass image -- the bridge then deblurs, with
# the guide supplying structure. Its width is the mask's centre (center_frac). Uniform
# masks always sample whole lines, so "acs" keeps the full readout (a centre box cut to
# center_frac * H rows was tried and dropped: ~4% of the samples, far blurrier). Starting points
# are explored after the architecture is settled; tau / tau_ref must be re-calibrated per start.
X1_SOURCE = "adjoint"
# INTENSITY-RELATIVE bridge: slice i runs at tau_i = TAU * s_i / TAU_REF, s_i the RMS of its
# zero-filled image over its Otsu foreground (sb/i2sb_recon.py::bridge_noise_scale) -- slices of
# one volume differ several-fold in intensity, and one absolute tau is near-silent on the bright
# ones. TAU_REF is the MEDIAN s over training slices, so TAU stays the tau of a typical slice.
# Measured 2026-11-05 (calibrate_recon_bridge.py, sbtgvn_R8_g1, 60 train slices): median s =
# 0.1481, range [0.082, 0.306]; the gap rule in these units gives tau 0.0212 (absolute: 0.0234).
# The per-slice GAP spread barely moves under the scaling (std/mean 0.45 -> 0.44): intensity sets
# how visible the noise is, but not how large the aliasing gap is. Re-measure for another main R.
TAU_SCALE = "x1_rms" if NORMALIZATION == "fixed" else "none"   # volume norm: ABSOLUTE tau
TAU_REF = 0.1481 * PDFS_SCALE_FAC / _OLD_SCALE_FAC
# Both scaled by the same factor, so the bridge is UNCHANGED relative to each slice (tau_i =
# TAU * s_i / TAU_REF, s_i scales with the data) and TAU keeps its meaning, the tau of a typical
# slice: 0.03 and 0.1481 at scale_fac 5e3 are 0.09 and 0.4443 here. Re-measure both with
# calibrate_recon_bridge.py once the eig-0.9 SENSE targets exist.
SCHEDULE = dict(kind="brownian", tau=TAU, n_points=1000, beta_max=0.3)

TGVN_SIZE = dict(num_cascades=12, sens_chans=8, sens_pools=4, chans=18, pools=4)

# Sljiva's LGGS grid point (scripts/makeconfigs_longi.jl), models/guided_lpds.py::lggs_defaults(),
# with the PHASE-INVARIANT similarity pidistance, |<q,k>| - 1/2||k||^2. On complex features that
# needs attn_backend "triton" (models/circulant_triton.py): flex scores Re<q,k> only, so flex +
# pidistance is wrong on complex data (the fused-vs-gather self-check refuses it at the first
# forward). Written out rather than imported so this script needs no torch model stack;
# tests/test_guided_recon.py pins it to lggs_defaults(attn_backend="triton").
LGGS_PARAMS = dict(K=30, M=169, C=1, P=7, s=2, Mh=64, window=1, guide_window=15,
                   joint_softmax=True, sim_fun="pidistance", lam0=1e-3, tau0=0.5, theta0=0.0,
                   degrees=0, prox_degrees=1,       # noise-adaptive threshold, constant steps
                   dK=5, nheads=1, init_strategy="semi_orthogonal", is_complex=True,
                   preproc="kspace", attn_backend="triton")

# How LGGS RUNS, not what it is: each changes no parameter and, to fp roundoff, no output or
# gradient (checked against the original triton path beside an fp32 noise floor:
# scripts/bench_lggs_materialize.py). Kept apart from LGGS_PARAMS, which stays Sljiva's grid
# point. Measured on an L40S, knee R8, one training step at K = 30: 739 -> 237 ms, 19.5 -> 12.4
# GB peak (2026-11).
#   triton_materialize  hold each guide window's softmax weights per refresh (circulant_window)
#   planar_state        carry (x, z) as real [re; im] tensors (models/mg_lpds.py)
#   complex_conv        the planar complex conv (models/components.py)
#   guided_compile      torch.compile the glue around the attention (models/compile_glue.py)
LGGS_SPEED_PARAMS = dict(triton_materialize=True)
LGGS_SPEED_TRAINING = dict(planar_state=True, complex_conv="planar", guided_compile=True)

# LGGS AS PUBLISHED (arXiv 2610.22643; notes/lggs_paper_alignment.md): the port's LGGS_PARAMS with
# every paper-alignment switch on, the paper's Table S1 windows, its training recipe, and the
# same speed settings. Not in DEFAULT_MODELS -- adding it there would renumber the launcher's
# array tasks -- so launch it by name: MODELS="lggs_paper" sbatch --array=0 ...
LGGS_PAPER_PARAMS = dict(
    LGGS_PARAMS,
    guide_window=3,                 # Table S1: W_g = 3 (the port: 15)
    dK=1,                           # Gamma formed afresh in every layer (Sec. II-C, III-C)
    adj_blend=False,                # ... and not blended with the previous one
    self_sim_fun="distance",        # S_zz plain distance (Eqs. 7, 15); guide stays pidistance
    similarity_scale="none",        # no learned temperature rho on q, k
    z_init="zero",                  # z(0) = 0, K full sweeps (Eq. 13)
    guide_mean="joint",             # mu_w over all N_g guide slices (Appendix B)
)
LGGS_PAPER_DATA = dict(guide_slices=3)          # N_g = 3 slices centred on the target (III-E)

# THE OPTIMISATION RECIPE -- ONE SWITCH, THE SAME FOR EVERY CELL, so the comparison stays fair:
#   "grid"    the mg grid's values, inherited per model (Adam 5e-4, LGGS 2e-4; cosine to 1e-6;
#             clip 0.05). IN FORCE (2026-11-09): the running sweep was launched with it.
#   "sljiva"  Sljiva's config/train.yaml, the recipe of its LGGS runs and the paper: cosine
#             4e-4 -> 2e-6 over 3e5 steps, warmup 0, global-norm clip 1 (torch's
#             clip_grad_norm_ is the same rule).
# TODO(next sweep): switch to "sljiva" -- for ALL cells at once, never for some.
TRAIN_RECIPE = "grid"
SLJIVA_RECIPE = dict(lr=4e-4, eta_min=2e-6, clip_grad=1.0)

MODELS = {
    "tgvn": dict(task="recon", guided=True, grid_base="varnet", pad_multiple=1,
                 model=dict(type="TGVN", params=dict(guided=True, cond_sigma=True,
                                                     use_smaps=True, **TGVN_SIZE))),
    "sbtgvn": dict(task="i2sb_recon", guided=True, grid_base="varnet", pad_multiple=1,
                   model=dict(type="BridgeTGVN", params=dict(guided=True, cond_sigma=True,
                                                             **TGVN_SIZE, **SCHEDULE))),
    "lpdsnet": dict(task="recon", guided=False, grid_base="lpdsnet", pad_multiple=None,
                    model=None),                       # the grid's lpdsnet model block
    "varnet_img": dict(task="recon", guided=False, grid_base="varnet", pad_multiple=1,
                       model=dict(type="TGVN", params=dict(guided=False, cond_sigma=True,
                                                           use_smaps=True, **TGVN_SIZE))),
    "lggs": dict(task="recon", guided=True, grid_base="mggrouplpds", pad_multiple=2,
                 model=dict(type="LGGS", params=LGGS_PARAMS),
                 speed=dict(params=LGGS_SPEED_PARAMS, training=LGGS_SPEED_TRAINING)),
    "lggs_paper": dict(task="recon", guided=True, grid_base="mggrouplpds", pad_multiple=2,
                       model=dict(type="LGGS", params=LGGS_PAPER_PARAMS),
                       speed=dict(params=LGGS_SPEED_PARAMS, training=LGGS_SPEED_TRAINING),
                       data=LGGS_PAPER_DATA),
    # MULTICONTRAST LGGS without an extra network: the knee LGGS cell, with the PD guide given its
    # own feature space (models/guided_lpds.py; each starts equal to plain LGGS)
    #   lggs_guide_w     its own W_phi^g, W_alpha^g in the prox (guide keys and energy)
    #   lggs_guide_conv  its own analysis conv A_g in every layer (v = A_g w)
    "lggs_guide_w": dict(task="recon", guided=True, grid_base="mggrouplpds", pad_multiple=2,
                         model=dict(type="LGGS",
                                    params=dict(LGGS_PARAMS, separate_guide_w=True)),
                         speed=dict(params=LGGS_SPEED_PARAMS, training=LGGS_SPEED_TRAINING)),
    "lggs_guide_conv": dict(task="recon", guided=True, grid_base="mggrouplpds", pad_multiple=2,
                            model=dict(type="LGGS",
                                       params=dict(LGGS_PARAMS, separate_guide_conv=True)),
                            speed=dict(params=LGGS_SPEED_PARAMS,
                                       training=LGGS_SPEED_TRAINING)),
}

# Depth sweep for TGVN under the reduced budget: width held at E2E-VarNet's (chans 18,
# sens_chans 8 -- the grid's varnet cell), cascades cut. Per forward at knee R8 with a
# fully-sampled guide, one cascade costs ~64 GFLOP (two 18-chan U-Nets + the CG trust projector),
# so 10 / 8 / 6 cascades are ~640 / 511 / 383 GFLOP against LGGS's ~677 (12: ~768).
TGVN_CASCADE_SWEEP = (12, 10, 8, 6)
for _n in TGVN_CASCADE_SWEEP:
    MODELS[f"tgvn_c{_n}"] = dict(
        task="recon", guided=True, grid_base="varnet", pad_multiple=1,
        model=dict(type="TGVN", params=dict(guided=True, cond_sigma=True, use_smaps=True,
                                            **dict(TGVN_SIZE, num_cascades=_n))))

# Bridge starting points for i2sb-TGVN (sb/i2sb_recon.py::bridge_start), at every sweep depth so
# the launcher can pick the settled one: sbtgvn_c<N>_<start>.
BRIDGE_STARTS = ("guide", "adjoint", "acs")
for _n in TGVN_CASCADE_SWEEP:
    for _src in BRIDGE_STARTS:
        MODELS[f"sbtgvn_c{_n}_{_src}"] = dict(
            task="i2sb_recon", guided=True, grid_base="varnet", pad_multiple=1, x1_source=_src,
            model=dict(type="BridgeTGVN", params=dict(guided=True, cond_sigma=True,
                                                      **dict(TGVN_SIZE, num_cascades=_n),
                                                      **SCHEDULE)))
# ... and LGGS as the bridge regressor (models/guided_lpds.py::BridgeLGGS): the knee LGGS cell's
# architecture and speed settings, started at the bridge state: sblggs_<start>.
for _src in BRIDGE_STARTS:
    MODELS[f"sblggs_{_src}"] = dict(
        task="i2sb_recon", guided=True, grid_base="mggrouplpds", pad_multiple=2, x1_source=_src,
        model=dict(type="BridgeLGGS", params=dict(LGGS_PARAMS, **SCHEDULE)),
        speed=dict(params=LGGS_SPEED_PARAMS, training=LGGS_SPEED_TRAINING))

# ... and the PLAIN UNet bridge baselines (models/sb_recon_unet.py::SBReconUnet), the I2SB-Recon
# setup of arXiv 2411.14269: an ADM UNet that never sees (y, E) -- the measurement enters only
# through x1 = E^H y and the sampler's CG step. The knee i2sb UNet's size (config/knee/i2sb/
# unet_R8.json). "sbunet" is unguided; "sbunet_guided" also takes the guide (PD) image as two
# extra input channels and nothing else. Both start at the zero-filled image, so they share the
# "adjoint" bridge calibration with sbtgvn_c*_adjoint.
SBUNET_PARAMS = dict(cond_x1=True, model_channels=64, num_res_blocks=2, channel_mult=[1, 2, 3, 4],
                     attention_resolutions=[40], image_size=320, num_heads=4,
                     num_head_channels=32, dropout=0.0, use_scale_shift_norm=True,
                     resblock_updown=False, conv_resample=True, t0=1e-4, T=1.0)
for _name, _guided in (("sbunet", False), ("sbunet_guided", True)):
    MODELS[_name] = dict(
        task="i2sb_recon", guided=_guided, grid_base="varnet", pad_multiple=1,
        x1_source="adjoint",
        model=dict(type="SBReconUnet", params=dict(SBUNET_PARAMS, guided=_guided, **SCHEDULE)))

# "sbunet_inv": I2SB-INVERSION (Wang et al., arXiv 2411.14269) as its code runs it
# (github.com/zhyjSIAT/I2SB-Inversion, i2sb-inversion/: train.py, i2sb/runner.py, i2sb/network.py,
# i2sb/ckpt_util.py::config_cartesian, sample.py). What is THEIRS:
#   start      x1 = the guide (PD) image, fully sampled, coil-combined with its OWN maps (online
#              ESPIRiT on its own centre; theirs: T1 with T1's maps)        i2sb.guide_maps "own"
#   net        ADM, 256 channels, 2 res blocks, mult (1,1,2,2,4,4), attention at downsampling
#              8/16/32 (their "32,16,8" at 256 px -> 40/20/10 px at our 320), 64-ch heads,
#              resblock_updown, scale-shift norm, no dropout, RANDOM init (their pretrained-ADM
#              load is commented out in network.py), 2 channels in/out: ONLY x_t and the step
#              (--cond-x1 off), no guide channels; it predicts eps, x0 = x_t - std_fwd * eps
#   sampling   80 NFE, stochastic posterior (eta 1), CG (7 iters, lam 7e-5) on EVERY x0
#              prediction, no final hard DC, no support mask; then the INVERSION -- a
#              deterministic walk back up the bridge from that recon (their code's update and
#              sign; sb/i2sb_recon.py::bridge_inversion) -- and a second 80-NFE CG pass from it
#              (i2sb_inversion_sample; scripts/eval_i2sb_recon_dc.py --inversion --nfe 80).
# What is OURS, unified across the i2sb baselines (the user's call, 2026-11-11): the bridge
# (Brownian, tau; theirs the I2SB beta schedule), the data protocol (per-volume p95 normalization
# -- theirs divides each slice by its max --, PDFS target under the operator's maps, measured
# k-space + AWGN, online maps for E), the 320 px grid, and the WHOLE TRAINING PROTOCOL: the grid's
# optimizer, LR, cosine schedule, clipping, batch, step budget and validation (20 NFE, every val
# slice, first pass only) -- not their AdamW 1e-5 / StepLR(1000, 0.99) / EMA 0.99 / no clipping /
# batch 16 -- and the shared OBJECTIVE, complex-MSE on x0_hat, not their eps-matching MSE (set
# training.loss_target "eps" for theirs). The net still outputs eps, x0_hat = x_t - std_fwd * eps.
I2SBINV_UNET = dict(cond_x1=False, guided=False, predict="eps", model_channels=256,
                    num_res_blocks=2, channel_mult=[1, 1, 2, 2, 4, 4],
                    attention_resolutions=[40, 20, 10], image_size=320, num_heads=4,
                    num_head_channels=64, dropout=0.0, use_scale_shift_norm=True,
                    resblock_updown=True, conv_resample=True, t0=1e-4, T=1.0)
MODELS["sbunet_inv"] = dict(
    task="i2sb_recon", guided=True, grid_base="varnet", pad_multiple=1, x1_source="guide",
    model=dict(type="SBReconUnet", params=dict(I2SBINV_UNET, **SCHEDULE)),
    i2sb=dict(guide_maps="own", support_thresh=None, support_state=False))

# GUIDE CONTRAST TRANSFER (models/contrast_transfer.py; task "contrast_transfer",
# training/contrast_transfer.py): tiny nets mapping the PD guide -- as a guided net receives it,
# coil-combined under the operator's maps -- onto the PDFS target, pretrained on their own and
# then FROZEN inside LGGS (the lggs_<ct> cells below), which then pools by similarity in a
# PDFS-like guide. Torch: torch/contrast_transfer_knee.sbatch. Run dirs:
# trained_nets/guided_recon/knee/<RUN_TAG>/ct_<...>_R8_g1; LGGS loads its net_bestval.ckpt.
TRANSFER_NETS = {
    "ct_pixel":       dict(arch="pixel", mode="mag", width=32, depth=3),        # 1.2k params
    "ct_patch3":      dict(arch="patch", mode="mag", width=32, depth=3, kernel=3),  # 1.4k
    "ct_patch7":      dict(arch="patch", mode="mag", width=32, depth=3, kernel=7),  # 2.7k
    "ct_unet8":       dict(arch="unet", mode="mag", width=8, levels=2),         # 6.5k
    "ct_unet16":      dict(arch="unet", mode="mag", width=16, levels=2),        # 26k
    "ct_unet16_cplx": dict(arch="unet", mode="complex", width=16, levels=2),    # 26k
}
TRANSFER_DEFAULT = tuple(TRANSFER_NETS)
# short and cheap: the nets are tiny, the cost is the measurement (online ESPIRiT per slice).
# The loader serves one slice (volumes differ in size); batch_size slices per step are
# centre-cropped to `crop` and stacked (training/contrast_transfer.py).
TRANSFER_TRAINING = dict(num_epochs=200, steps_per_epoch=100, val_every_epochs=10,
                         val_slices=200, loss_type="l1", loss_on="mag", clip_grad=None,
                         batch_size=4, crop=[320, 320])
TRANSFER_OPTIM = dict(type="Adam", params=dict(lr=1e-3))
TRANSFER_SCHED = dict(type="CosineAnnealingLR", params=dict(T_max=200 * 100, eta_min=1e-5))
for _name, _p in TRANSFER_NETS.items():
    MODELS[_name] = dict(
        task="contrast_transfer", guided=True, grid_base="varnet", pad_multiple=1,
        model=dict(type="ContrastTransfer", params=dict(_p)),
        training=dict(TRANSFER_TRAINING,
                      **({"loss_on": "complex"} if _p["mode"] == "complex" else {})),
        optimizer=TRANSFER_OPTIM, scheduler=TRANSFER_SCHED)
    # ... and LGGS with that transfer frozen in front of its guide: lggs_<ct name>, the knee LGGS
    # cell otherwise unchanged (needs the transfer's pretraining to have finished)
    MODELS[f"lggs_{_name}"] = dict(
        task="recon", guided=True, grid_base="mggrouplpds", pad_multiple=2,
        model=dict(type="LGGS", params=LGGS_PARAMS),
        speed=dict(params=LGGS_SPEED_PARAMS, training=LGGS_SPEED_TRAINING),
        transfer=_name)

# Per-start bridge temperature. The gap x0 - x1 differs completely between starts (aliasing for
# "adjoint", blur for "acs", CONTRAST for "guide"), so each gets its own calibration, written by
# `scripts/calibrate_recon_bridge.py <config> --json BRIDGE_TAU_DIR/<start>_R<R>.json` (the bridge
# launcher does it on first use). tau = TAU_MARGIN x the intensity-relative gap rule: the margin is
# the user's 2026-11-05 call on the zero-filled start (0.03 against a relative gap rule of 0.0212,
# "a slightly wider ladder is the safer side"), kept for every start. Without a file a start falls
# back to TAU / TAU_REF above (the zero-filled calibration).
BRIDGE_TAU_DIR = "config/knee/guided/bridge_tau"
TAU_MARGIN = 0.03 / 0.0212          # on the intensity-relative gap rule (tau_scale "x1_rms")
TAU_MARGIN_ABS = 0.03 / 0.0234      # on the absolute gap rule (tau_scale "none")
# The calibration files are keyed by NORMALIZATION too: a gap measured under one intensity
# convention is not the gap under another.


def bridge_tau(start, r, root=REPO, normalization=None):
    """(tau, tau_ref, calibrated) for bridge start `start` at main R `r`. tau_ref is None for an
    absolute tau (tau_scale "none"). No file -> the zero-filled fallback TAU / TAU_REF."""
    norm = normalization or NORMALIZATION
    absolute = norm != "fixed"
    p = os.path.join(root, BRIDGE_TAU_DIR, f"{start}_R{r}_{norm}{_noise_suffix()}.json")
    if not os.path.exists(p):
        return TAU, (None if absolute else TAU_REF), False
    with open(p) as f:
        cal = json.load(f)
    if absolute:
        return float(f"{TAU_MARGIN_ABS * cal['tau_abs']:.4g}"), None, True
    return (float(f"{TAU_MARGIN * cal['tau_rel']:.4g}"), float(f"{cal['tau_ref']:.4g}"), True)

# PSNR / SSIM against each slice's own peak |gt| (training/metrics.py). The paired loader only
# multiplies by a fixed scale_fac, so slice peaks sit well below 1 and vary across a volume:
# the default data_range=1 inflated PSNR by -20 log10(peak) dB, most on the dimmest slices.
METRIC_RANGE = "gt_max"


def grid_cells(r, tags):
    """{grid model tag: config} for the mg grid's knee cells used as templates, at R."""
    tags = sorted(set(tags) | {"lpdsnet"})
    with tempfile.TemporaryDirectory() as d:
        subprocess.run([sys.executable, os.path.join(REPO, "scripts", "make_mg_recon_configs.py"),
                        "--out", d, "--anatomy", ANATOMY, "--accels", str(r), "--only", *tags],
                       check=True, cwd=REPO, stdout=subprocess.DEVNULL)
        out = {}
        for t in tags:
            with open(os.path.join(d, ANATOMY, "mg", f"{t}_R{r}.json")) as f:
                out[t] = json.load(f)
        return out


def _data(split, base_split, pad_multiple):
    keep = ("batch_size", "start_slice", "end_slice", "organ_mask_source")
    return {"name": "fastmri_paired",
            "split_csv": f"{SPLITS}/{split}.csv",
            "kspace_roots": list(KSPACE_ROOTS),
            **{k: base_split[k] for k in keep if k in base_split},
            **({"scale_fac": PDFS_SCALE_FAC, "side_scale_fac": PD_SCALE_FAC}
               if NORMALIZATION == "fixed" else
               {"volume_scales": VOLUME_SCALES, "volume_stat": VOLUME_STAT,
                "side_volume_stat": SIDE_VOLUME_STAT, "volume_target": 1.0}),
            "target": TARGET,
            "smap_root": PDFS_MAPS,
            "pad_multiple": pad_multiple,
            "num_workers": 8}


# ONE tau for every bridge cell (--tau; the launchers' TAU), e.g. to compare starting points at
# the SAME bridge temperature rather than each at its own calibrated one. Replaces the per-start
# calibration, and tags the cell names (and so the run dirs) with "_tau<value>" so these runs never
# share a directory with calibrated ones. None = per-start calibration (BRIDGE_TAU_DIR).
TAU_OVERRIDE = None


def bridge_tau_path(r, model):
    """Where a bridge cell's start-point calibration lives, else None (also under --tau)."""
    spec = MODELS[model]
    if spec["task"] != "i2sb_recon" or TAU_OVERRIDE is not None:
        return None
    return f"{BRIDGE_TAU_DIR}/{spec.get('x1_source', X1_SOURCE)}_R{r}_{NORMALIZATION}{_noise_suffix()}.json"


def cell_name(r, model, g):
    name = f"{model}_R{r}" + (f"_g{g}" if MODELS[model]["guided"] else "")
    if TAU_OVERRIDE is not None and MODELS[model]["task"] == "i2sb_recon":
        name += f"_tau{TAU_OVERRIDE:g}"
    return name


def make_config(r, model, g, grid):
    spec = MODELS[model]
    base = grid[spec["grid_base"]]
    model_block = spec["model"] if spec["model"] is not None else base["model"]
    pad = (spec["pad_multiple"] if spec["pad_multiple"] is not None
           else base["data"]["train"]["pad_multiple"])
    training = {k: v for k, v in base["training"].items() if k != "log_levels"}
    training["metric_range"] = METRIC_RANGE
    if NOISE_PRESETS[NOISE] is not None:
        training.update(NOISE_PRESETS[NOISE], noise_dist="uniform")
    training["val_noise_std"] = VAL_NOISE_STD
    if TRAIN_RECIPE == "sljiva":
        training["clip_grad"] = SLJIVA_RECIPE["clip_grad"]
    speed = spec.get("speed") or {}
    training.update(speed.get("training", {}))
    training.update(spec.get("training") or {})
    mri = dict(grid["lpdsnet"]["mri"], guide_R=int(g), guide_acs_lines=0,
               guide_mask_dist="uniform")
    name = cell_name(r, model, g)
    run = f"{ANATOMY}_PDFS_from_PD_{name}" if spec["guided"] else f"{ANATOMY}_PDFS_{name}"
    cfg = {
        "task": spec["task"],
        "experiment": {"name": f"{run}_{run_tag()}"},
        "model": json.loads(json.dumps(model_block)),
        "paths": {"save_dir": f"trained_nets/guided_recon/{ANATOMY}/{run_tag()}/{name}",
                  "ckpt": None},
        "data": {"train": dict(_data("train", base["data"]["train"], pad), **(spec.get("data") or {})),
                 "val": dict(_data("val", base["data"]["val"], pad), **(spec.get("data") or {}))},
        "training": training,
        "mri": mri,
        "optimizer": base["optimizer"],
        "scheduler": base["scheduler"],
        "wandb": {"project": "guided_recon", "id": None},
    }
    if speed.get("params"):
        cfg["model"]["params"].update(speed["params"])
    if spec.get("transfer"):
        # the frozen guide transfer: the pretrained cell's best-validation weights
        ct = spec["transfer"]
        cfg["model"]["params"]["guide_transfer"] = {
            "params": dict(MODELS[ct]["model"]["params"]),
            "ckpt": f"trained_nets/guided_recon/{ANATOMY}/{run_tag()}/{cell_name(r, ct, g)}/"
                    f"net_bestval.ckpt"}
    if TRAIN_RECIPE == "sljiva":
        cfg["optimizer"] = dict(cfg["optimizer"], params=dict(
            cfg["optimizer"]["params"], lr=SLJIVA_RECIPE["lr"]))
        cfg["scheduler"] = dict(cfg["scheduler"], params=dict(
            cfg["scheduler"]["params"], eta_min=SLJIVA_RECIPE["eta_min"]))
    elif TRAIN_RECIPE != "grid":
        raise ValueError(f"TRAIN_RECIPE must be 'grid' or 'sljiva', got {TRAIN_RECIPE!r}")
    # a cell's OWN recipe (sbunet_inv: I2SB-Inversion's; the transfer nets) beats the shared one
    if spec.get("optimizer"):
        cfg["optimizer"] = json.loads(json.dumps(spec["optimizer"]))
    if spec.get("scheduler"):
        cfg["scheduler"] = json.loads(json.dumps(spec["scheduler"]))
    if spec["task"] == "contrast_transfer":
        keep = ("noise_std", "noise_dist", "val_noise_std", "val_seed", "metric_range")
        cfg["training"] = dict({k: training[k] for k in keep if k in training},
                               **spec["training"])
    if spec["task"] == "i2sb_recon":
        start = spec.get("x1_source", X1_SOURCE)
        tau, tau_ref, _ = bridge_tau(start, r)
        if TAU_OVERRIDE is not None:
            tau = float(TAU_OVERRIDE)                   # tau_ref only matters for x1_rms
        cfg["model"]["params"]["tau"] = tau             # BridgeTGVN checks it matches i2sb
        cfg["training"]["loss_type"] = "complex-mse"
        cfg["i2sb"] = {**SCHEDULE, "tau": tau, "deterministic": False, "posterior": "ddpm",
                       "x1_source": start, "x0_source": "operator",
                       "tau_scale": TAU_SCALE, "tau_ref": tau_ref,
                       "val_mode": "full_recon", "val_nfe": 20, **I2SB_DC, **I2SB_SUPPORT}
        cfg["i2sb"].update(spec.get("i2sb") or {})
    return cfg


def cells(accels=ACCELS, models=DEFAULT_MODELS, guide_accels=GUIDE_ACCELS):
    out = []
    for r in accels:
        for m in models:
            for g in (guide_accels if MODELS[m]["guided"] else (1,)):
                if (r, m, g) not in out:
                    out.append((r, m, g))
    return out


def config_path(root, r, model, g):
    return os.path.join(root, ANATOMY, "guided", f"{cell_name(r, model, g)}.json")


def _write_config():
    """training/config_io.py::write_config, loaded BY PATH (as make_mg_recon_configs.py does):
    it spells floats so yaml.safe_load -- how train.py reads configs -- gets numbers back
    (json.dump's `1e-06` loads as the STRING '1e-06'), and refuses a config that would not
    round-trip. By path, because `import training...` would run training/__init__.py
    (torchvision)."""
    import importlib.util
    p = os.path.join(REPO, "training", "config_io.py")
    spec = importlib.util.spec_from_file_location("_immap_config_io", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.write_config


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="config")
    p.add_argument("--accels", nargs="*", type=int, default=list(ACCELS))
    p.add_argument("--models", nargs="*", choices=list(MODELS), default=list(DEFAULT_MODELS))
    p.add_argument("--guide-accels", nargs="*", type=int, default=list(GUIDE_ACCELS),
                   help="guide (PD) accelerations for the guided models; 1 = fully sampled")
    p.add_argument("--list-cells", action="store_true",
                   help="print 'index<TAB>R<TAB>model<TAB>guide R<TAB>name<TAB>tau file' "
                        "and exit (tau file: a bridge cell's calibration path, else '-')")
    p.add_argument("--noise", choices=list(NOISE_PRESETS), default="low",
                   help="added TRAINING noise: grid U[0.01, 0.02], low U[0, 0.01], none; "
                        "validation adds none (VAL_NOISE_STD); non-grid presets get their own "
                        "run tag")
    p.add_argument("--tau", type=float, default=None,
                   help="one bridge tau for every i2sb cell (no per-start calibration); the cell "
                        "names get a _tau<value> suffix")
    args = p.parse_args()
    global TAU_OVERRIDE, NOISE
    TAU_OVERRIDE = args.tau
    NOISE = args.noise
    write_config = _write_config()

    todo = cells(args.accels, args.models, args.guide_accels)
    if args.list_cells:
        for i, (r, m, g) in enumerate(todo):
            print(f"{i}\tR{r}\t{m}\tg{g}\t{cell_name(r, m, g)}\t"
                  f"{bridge_tau_path(r, m) or '-'}")
        return
    for r in sorted({c[0] for c in todo}):
        grid = grid_cells(r, [MODELS[m]["grid_base"] for _, m, _ in todo])
        for rr, m, g in todo:
            if rr != r:
                continue
            cfg = make_config(r, m, g, grid)
            path = config_path(args.out, r, m, g)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            write_config(cfg, path)
            extra = ""
            if cfg["task"] == "i2sb_recon":
                cal = TAU_OVERRIDE is not None or bridge_tau(cfg["i2sb"]["x1_source"], r)[2]
                extra = (f" x1={cfg['i2sb']['x1_source']} tau={cfg['i2sb']['tau']:g} "
                         f"tau_scale={cfg['i2sb']['tau_scale']} tau_ref={cfg['i2sb']['tau_ref']}"
                         + ("" if cal else " (UNCALIBRATED: zero-filled fallback)"))
            print(f"{path}: {cfg['task']} {cfg['model']['type']} R={r} guide_R={g} "
                  f"lr={cfg['optimizer']['params']['lr']}{extra}")


if __name__ == "__main__":
    main()
