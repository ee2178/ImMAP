#!/usr/bin/env python3
"""
Generate the multigrid fastMRI reconstruction grid.

PyTorch counterpart of Sljiva's `scripts/makeconfigs_multigrid.jl`: writes one
config per (anatomy, acceleration, model) cell to

    config/<anatomy>/mg/<model>_R<r>.json

6 models x 3 accelerations = 18 cells PER ANATOMY. knee and brain have separate
sbatch files (`torch/mg_recon_{knee,brain}.sbatch`); each asks this script for
its own cell list via `--list-cells --anatomy <a>`, so the two stay in sync.

Every cell trains on SYNTHETIC k-space (`kspace_type: "simulated"`): the clean
coil-combined image is pushed through Sense -> Fourier -> mask with complex AWGN
at sigma ~ U[0.01, 0.02] added in the coil-image domain. See
`operators/noise.py::mri_awgn`, the port of `genobs(clo::SyntheticMRIReco, ...)`
in `Sljiva/src/closures/mrireco.jl`.

Usage
-----
    python scripts/make_mg_recon_configs.py                  # write config/
    python scripts/make_mg_recon_configs.py --dry-run        # print, don't write
    python scripts/make_mg_recon_configs.py --list-cells --anatomy knee

Writing needs `training.common.write_config` (and therefore torch); `--dry-run`
and `--list-cells` are pure stdlib and run anywhere.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ===========================================================================
#  Grid axes
# ===========================================================================

# ---------------------------------------------------------------------------
#  Network hyperparameters: matched to the Julia reference
#
#  Sources, so a deviation is traceable:
#    Sljiva/config/groupcdl.yaml               -- M, p, s, d, tau0, gamma0,
#                                                 Mh, similarity, init_strategy,
#                                                 nheads, baseline K=30
#    Sljiva/scripts/makeconfigs_multigrid.jl   -- the multigrid sweep proper:
#         the "K888s2" cell (K=[1,[8,8,8]], s=2, widen=1, alpha0=0.1,
#         alpha_conv=false, dK=2) crossed with dual in {false,true} and
#         windowsize in {1,9}. Those four corners ARE the four model types
#         below -- windowsize=1 drops the group prox, dual=true is the LPDS
#         (Fenchel/clipping) read-out.
#    Sljiva/config/altsplit.yaml               -- the LADMM cell
# ---------------------------------------------------------------------------

# preproc="kspace" is the reconstruction mode (Sljiva's lpdsnet.yaml /
# mglpds.yaml use it too): it pads the OPERATOR alongside y~, and removes DC
# through E^H E instead of subtracting a plain mean. See preprocessing/kspace.py.
#
# The CDL-family constants (MG_COMMON / GROUP / VCYCLE_K / BASELINE_K) and the
# attention-backend discussion that used to live here were dropped when the grid
# lost its MGCDLNet and group cells. `docs/multigrid_port.md` still carries both,
# and `models/circulant_{flex,triton}.py` are unchanged -- adding a group cell
# back means reinstating a GROUP dict, not rebuilding anything.

# NOTE ON `mglpds`: this is now the real port of Sljiva's mg_lpds.jl
# (models/mg_lpds.py::MGLPDSNet) -- a primal-dual splitting network propagating
# a pair (x, z) with over-relaxation and a two-field FAS correction.
#
# It is NOT `MGLPDS` / `MGCDLNet(dual=True)`, which this grid used to run: that
# is a LISTA layer with a clipping prox -- one iterate, no extrapolation, one
# coarse correction. Both names still exist in build_model; they are different
# networks and their numbers are not interchangeable.

# AltSplitCDLNet, from altsplit.yaml. `denoiser_kws.K` is the ONLY difference
# between the two LADMM cells: altsplit.yaml ships `K: [1, [4,4,8]]` with
# `# K: 6` commented directly above it, so both come from the reference.
# smap_update is OFF. Profiling put the coil-map CG solve at 72.9% of the
# forward pass -- and the learned highpass inside it at 57.4% of the whole
# forward, 11 applies per solve -- for a 3.46x increase in step time. The maps
# come from the dataset and are already good; re-estimating them is not where
# this grid's accuracy is coming from. `smap_kws` is left in place so flipping
# the flag back is a one-key change.
#
# preproc stays "identity": the LADMM x-solve is (E^H E + rho I) x = E^H y,
# which is well posed on the raw adjoint -- there is no dictionary here whose
# atoms would be spent representing DC.
#
# The prox is an MGLPDSNet, matching the standalone `lpdsnet` / `mglpds` cells,
# so all four models in the grid are built from the same primal-dual smoother
# and only the OUTER algorithm (unrolled LPDS vs linearized ADMM) and the
# V-cycle differ. `denoiser_kws` is `LPDS_COMMON` verbatim apart from `preproc`,
# which `build_denoiser` sets to "image": inside the prox `E = Identity`, so
# `E^H E 1 = 1` and the kspace DC correction degenerates to a plain mean anyway.
#
# `reuse_latent=True` genuinely works here now -- `MGLPDSNet` returns its
# primal-dual pair as the latent and `denoise()` threads it back in as `state`.
# Before that it silently no-op'd, because `denoise` only looked for `z0`.
def _altsplit(denoiser_K):
    return dict(
        type="AltSplitCDLNet",
        params=dict(
            admm_iters=6, reuse_latent=True, smap_update=False, rho0=1.0,
            cg_maxit=10, cg_tol=1.0e-4, implicit_cg=True, preproc="identity",
            denoiser_type="mglpdsnet",
            denoiser_kws=dict(LPDS_DENOISER, K=denoiser_K),
            smap_kws=dict(sigma_g=1.5, sigma_max=3.0, sigma_min=0.3,
                          mu0=1.0, gamma0=0.01),
        ),
    )


# The LPDS family. M and K are THIS repo's long-standing defaults (config/knee/
# recon.json, models/lpdsnet.py), not `Sljiva/config/lpdsnet.yaml`'s M=225 / K=40
# -- an earlier revision of this file used the Julia values and it was wrong to
# switch them. The step-size parameters need no such choice: ImMAP's LPDSNet
# defaults (l0=1e-3, eta_0=0.5, theta_0=0.0) already equal lpdsnet.yaml's
# lambda0 / tau0 / theta0. `alpha0=1.0` is the coarse-correction init from
# mglpds.yaml, which has no LPDSNet counterpart.
#
# windowsize stays 1: no group models in this grid, so no Mh / attention.
LPDS_COMMON = dict(
    M=169, C=1, P=7, s=2, widen=1, degrees=1,
    lam0=1.0e-3, tau0=5.0e-1, theta0=0.0, alpha0=1.0,
    is_complex=True, preproc="kspace", resize_noise=True,
)

# "6 V-cycles with [4, 4, 6] LISTA iterations: 4 at depth 0, 4 at depth 1, 6 at
# depth 2" translates directly, because `iters[l]` IS the per-depth count --
# the V-cycle halves it into pre/post smoothing itself:
#
#   depth 0   iters[0] = 4   ->  lpdsA 2 + lpdsB 2
#   depth 1   iters[1] = 4   ->  lpdsA 2 + lpdsB 2
#   depth 2   iters[2] = 6   ->  one stack of 6 (coarsest: no split)
#
# `PDVCycle.iters_per_level` reports [4, 4, 6] back, and `tests/test_mg_lpds.py`
# pins that round-trip. 6 x (4+4+6) = 84 layers, 6 x 4 = 24 on the fine grid.
#
# NOT a checked-in Sljiva configuration: every V-cycle config there uses
# K_outer = 1 with large per-level iters (mglpds.yaml `[16,16,16]`,
# makeconfigs_mglpds.jl `[12,12,12]`). Many-outer-cycles-few-sweeps matches
# `MGLPDSNet`'s constructor default (`K=8, iters=[4,4,4]`) instead.
LPDS_VCYCLE_K = [6, [4, 4, 6]]
LPDS_BASELINE_K = 30                       # this repo's LPDSNet default

# What the LADMM prox slot gets: the same smoother as the standalone cells,
# minus `preproc` (build_denoiser pins it to "image" for the E = Identity prox).
LPDS_DENOISER = {k: v for k, v in LPDS_COMMON.items() if k != "preproc"}

MODELS = {
    # SAME CLASS, differing only in K -- so this pair is a clean multigrid
    # ablation. It did not used to be: `lpdsnet` was models/lpdsnet.py::LPDSNet
    # and `mglpds` was MGCDLNet(dual=True), which crossed an architecture
    # boundary as well as the V-cycle. models/mg_lpds.py is the real port, so
    # both cells are now MGLPDSNet.
    "lpdsnet": dict(type="MGLPDSNet",
                    params=dict(LPDS_COMMON, K=LPDS_BASELINE_K)),
    "mglpds":  dict(type="MGLPDSNet",
                    params=dict(LPDS_COMMON, K=LPDS_VCYCLE_K)),
    # The nonlocal arm. Identical to `mglpds` except for what sits in the prox
    # slot, so the pair isolates the prior rather than the architecture:
    # SoftThreshold -> GroupThreshold, everything else held.
    #
    # WINDOW = 15, not the 35 the BSD432 group configs use. The point of doing
    # this inside a V-cycle is that the COARSE levels supply the long range, so
    # each level can stay local: 15x15 on the 160x160 level-0 latent, and the
    # 80x80 and 40x40 levels reach the rest. Cost is O(window^2) per pixel per
    # level, so 35 would be ~5.4x the attention work for reach the hierarchy
    # already provides.
    #
    # sim_fun="distance" is forced by the dtype: the phase-invariant
    # similarities need |<q,k>|, which FlexAttention cannot form on complex
    # features (models/prox.py raises). See tests/test_group_lpds.py.
    "mggrouplpds": dict(
        type="MGGroupLPDS",
        params=dict(LPDS_COMMON, K=LPDS_VCYCLE_K,
                    window=15, Mh=64, dK=5, nheads=1,
                    sim_fun="distance", attn_backend="flex",
                    flex_block_size=128),
        note=("nonlocal prox at every grid level: GroupThreshold in the LPDS "
              "prox slot. window=15 with the V-cycle supplying long range; "
              "sim_fun='distance' because flex cannot carry a phase-invariant "
              "similarity on complex features."),
    ),
    "altsplit":   _altsplit(6),
    "mgaltsplit": _altsplit([1, [4, 4, 6]]),

    # BASELINE. fastMRI's End-to-End VarNet, vendored unmodified under
    # models/e2evarnet/_fastmri/. Published defaults (12 cascades, chans=18,
    # sens_chans=8) -- deliberately NOT retuned to this grid, because a
    # baseline that has been fiddled with is not a baseline.
    #
    # Three asymmetries to carry with the numbers:
    #   * it ESTIMATES its own coil maps, where the unrolled cells are handed
    #     the dataset's. That is the method, not an oversight -- but it makes
    #     this the harder task, and the comparison favours the unrolled nets.
    #   * it returns an RSS MAGNITUDE image, so only magnitude metrics apply.
    #   * it has no noise-adaptive parameter, so sigma ~ U[0.01, 0.02] is a
    #     handicap the sigma-conditioned cells do not carry. This got HEAVIER
    #     when the range moved up from [0, 0.01]: the sigma-conditioned cells
    #     see a 1.5x spread they can adapt their thresholds to, and VarNet has
    #     to pick one compromise across it. Check whether its gap widens with
    #     sigma before attributing the gap to architecture.
    # It also has ~10x the parameters of the flat LPDSNet; `count_parameters`
    # in each config records it.
    #
    # It is scored under the organ mask on both anatomies, which removes an
    # advantage it would otherwise have for the wrong reason: an RSS magnitude
    # is non-negative and noise-biased, so its background is a positive floor
    # by construction, and unmasked that floor is charged to it as error in a
    # region nobody is reconstructing. If it underperforms, check convergence
    # before width -- VarNet is usually trained far longer than 300k steps.
    # `python -m tests.test_e2evarnet` re-checks the vendored code against
    # upstream and ImMAP's FFT convention against fastMRI's.
    #
    # Every torch/exp*.sbatch launcher lists the varnet cell for its own
    # (anatomy, R); torch/_mg_recon_body.sh launches each cell once however
    # many launchers list it (LAUNCH-ONCE there). This replaced
    # baseline_varnet.sbatch.
    "varnet": dict(
        type="E2EVarNet",
        params=dict(num_cascades=12, sens_chans=8, sens_pools=4,
                    chans=18, pools=4, mask_center=True),
        note=("fastMRI E2E-VarNet baseline at published defaults. Estimates "
              "its own sensitivity maps and returns RSS magnitude, so it is "
              "comparable on PSNR/SSIM/NRMSE and on nothing phase-sensitive."),
    ),
}

# ---------------------------------------------------------------------------
#  OPT-IN cells
#
#  Written by every regeneration, so a launcher's REGENERATE=1 always finds its
#  config, but listed by --list-cells ONLY when named in --only. Appending them
#  to the default list would renumber every R4/R16 cell of the full grid, which
#  mg_recon_{knee,brain}.sbatch index directly -- and exp1's "cells 0 and 6 are
#  lpdsnet R8 and R4" would silently point at other runs.
# ---------------------------------------------------------------------------

# MULTILEVEL LPDS (models/ml_lpds.py): the analysis-form multilevel prior
# sum_l lam_l ||A_l ... A_1 x||_1 under unrolled Condat-Vu. The same primal-dual
# iteration as `lpdsnet`, with its one clipped dual replaced by one dual per
# resolution level; `L=1` IS `lpdsnet`.
#
# Held against the two LPDS cells: L=3 (mglpds's three grid levels, hence the
# same pad_stride=8), s=2, P=7, degrees, lam0/tau0/theta0, and everything
# outside `model`. Dropped: alpha0 (a V-cycle coarse-correction weight) and
# resize_noise (acts only on spatial noise maps; sigma here is per-image).
#
# SIZED FOR A WALL-CLOCK BUDGET, NOT MATCHED TO `mglpds`. At K=6 and FLOP
# parity these cells trained at ~20 it/s, far faster than mglpds: FLOPs are not
# time, and mglpds spends ~6 GB per forward on memory-bound elementwise work
# (prox, restrict/prolong, coarse Grams) that the ML nets do not do. At K=18 and
# the widths below they still trained at ~5 it/s with the GPU near 100%, against
# ~2 it/s for this grid's fastest other nets. So K went to 30:
#
#     mllpdsw2  K=30  widen=2  channels 48/96/192    ~260 GFLOP (7.7x)  135.8M params
#     mglpds    K=[6,[4,4,6]], M=169                   34 GFLOP          3.62M params
#     lpdsnet   K=30, M=169                            21 GFLOP          1.00M params
#
# (GFLOPs per forward on a 160x160, 16-coil problem with preproc="kspace", conv +
# 5 n log2 n per FFT; measured at K=18 and scaled by 30/18 -- every term is per
# layer. Parameters are exact.)
#
# K=30 IS lpdsnet's K: layer 0 is the cold start in both, so both take 29 E^H E
# steps, and (lpdsnet vs mllpdsw2) no longer differs in iteration count -- only in
# the prior. Width stays where K=18 put it, on multiples of 8: level-to-level
# filters are dense M_{l-1} x M_l banks, so width costs quadratically and buys no
# data consistency. Expect ~3 it/s if time scales with K -- read the probe, and
# watch training memory, which also grows with K. tau0=0.5 stays inside the
# Condat-Vu bound at init for both cells (tau0 (1/2 + ||A||^2) = 0.82 and 0.86,
# against 1; the bound depends on the filters, not K).
# tau0 IS HALVED RELATIVE TO `lpdsnet`, and that is not a tuning choice.
# `MLLPDSNet` now defaults to `init_norm="cascade"`, which rescales each level so
# `||A_(1,l)||_2 = 1` instead of `||A_l||_2 = 1`. Per-level normalisation leaves
# the cascade collapsed -- measured at init on the 48/96/192 shape:
#
#     init_norm="level"     ||A_(1,l)|| = 0.998, 0.394, 0.164
#     init_norm="cascade"   ||A_(1,l)|| = 0.999, 1.001, 1.005
#
# so level 3's dual was receiving 16% of level 1's gain while `lam0` clipped
# both against the same threshold, and its push back into the primal carried
# that factor a second time. Fixing it means paying the depth tax the Condat-Vu
# bound always implied: `||A||^2` rises from ~1.1 toward L, so the admissible
# primal step `1/(1/2 + ||A||^2)` falls from ~0.61 to ~0.29. At tau0=0.5 these
# cells would now start OUTSIDE the bound; 0.25 lands at ~0.76 of it.
#
# The cascade rescale requires proj_mode="slice" (the default). Per-atom
# projection is tight enough to undo it -- 1.001/1.005 -> 0.599/0.377 after one
# `project()` -- so the two options are not independent. See
# `MLLPDSLayer.normalize_cascade`.
ML_LPDS_COMMON = dict(
    {k: v for k, v in LPDS_COMMON.items()
     if k not in ("M", "widen", "alpha0", "resize_noise")},
    K=30, L=3, tau0=2.5e-1)

# WIDTH AND DEPTH. P STAYS AT 7 IN EVERY CELL -- the 7x7 atom is the model, not
# a hyperparameter to trade away. Capacity is bought with channels and paid for
# with K, since
#
#     params, FLOPs  ~  K * P^2 * sum_l M_{l-1} M_l
#
# is linear in K and quadratic in width. Dropping K therefore funds width at
# constant cost, and these four cells walk that trade one step at a time:
#
#                     channels      K     params   vs w2   rel FLOP   DC steps
#     mllpdsw2       48/96/192     30     135.8M   1.00x     1.00x       29
#     mllpds64       64/128/256    30     241.2M   1.78x     1.77x       29
#     mllpds64k20    64/128/256    20     160.8M   1.18x     1.18x       19
#     mllpds128k20   128/256/512   20     642.8M   4.73x     4.68x       19
#
# K IS NOT A FREE PARAMETER. Layer 0 is the cold start, so K=30 takes 29 `E^H E`
# steps and K=20 takes 19. ML_LPDS_COMMON's K=30 exists precisely so ML-LPDS and
# `lpdsnet` take the SAME number, leaving the prior as the only difference. At
# R=16 -- ~6% of k-space -- data consistency is the scarce resource, which is
# why `mllpds64` holds K at 30 while widening: it is the only cell that changes
# width WITHOUT also changing how much data consistency the net gets.
#
# The chain reads in three adjacent pairs, each changing exactly one thing:
#
#     mllpdsw2     vs mllpds64      WIDTH, at K = 30
#     mllpds64     vs mllpds64k20   K, at 64/128/256
#     mllpds64k20  vs mllpds128k20  WIDTH, at K = 20
#
# 128/256/512 at K=30 would be 964.1M parameters and 7.02x the FLOPs -- 7x a
# model that currently loses to 1.00M-parameter `lpdsnet`. K=20 is what makes
# that width affordable at all.
MODELS.update({
    "mllpdsw2":     dict(type="MLLPDSNet",
                         params=dict(ML_LPDS_COMMON, M=48, widen=2)),
    "mllpds64":     dict(type="MLLPDSNet",
                         params=dict(ML_LPDS_COMMON, M=64, widen=2)),
    "mllpds64k20":  dict(type="MLLPDSNet",
                         params=dict(ML_LPDS_COMMON, M=64, widen=2, K=20)),
    "mllpds128k20": dict(type="MLLPDSNet",
                         params=dict(ML_LPDS_COMMON, M=128, widen=2, K=20)),
})

# DEPTH INSTEAD OF WIDTH (exp6). exp5 said cutting K costs more than width buys:
# 64/128/256 beat itself at K=20, and 128/256/512 at K=20 lost to it despite
# 2.7x the parameters. These cells spend the budget on K instead:
#
#                     channels      K     params   rel FLOP   DC steps
#     mllpds64       64/128/256    30     241.2M    1.00x        29   (exp4 arm)
#     mllpds48k54    48/96/192     54     244.4M    1.01x        53
#     mllpds64k40    64/128/256    40     321.7M    1.33x        39
#
# mllpds48k54 is ISO-BUDGET with mllpds64 -- same parameters and FLOPs, spent
# on depth rather than width -- so that pair is the clean width-vs-depth
# question. mllpds64k40 asks whether the winner keeps improving with K.
#
# There is deliberately NO lpdsnet control at matched K: the claim under test is
# that the multilevel formulation packs in parameters without costing time, not
# that the prior beats a flat one at equal iteration count.
#
# tau0 is unchanged: the Condat-Vu bound depends on the filters, not K.
MODELS.update({
    "mllpds48k54": dict(type="MLLPDSNet",
                        params=dict(ML_LPDS_COMMON, M=48, widen=2, K=54)),
    "mllpds64k40": dict(type="MLLPDSNet",
                        params=dict(ML_LPDS_COMMON, M=64, widen=2, K=40)),
})

# MULTILEVEL CDL (models/ml_cdlnet.py), the synthesis-form siblings of ML-LPDS:
#
#   mlcdlw2    MLCDLNet       unrolled ML-ISTA -- the only state carried between
#                             sweeps is the deepest code
#   mlsplitw2  MLSplitCDLNet  unrolled linearised ADMM -- every code kept, one
#                             dual per link between levels
#
# Matched to `mllpds64` in WIDTH -- channels 64/128/256, L=3, s=2, P=7, degrees,
# dtype and preproc -- and the S.T. threshold initialised at ML-LPDS's clip
# threshold, lam0=1e-3. (`tau0` in these classes IS that threshold; in the LPDS
# family `tau0` is the primal step.) readout='level1' is stated so the config
# records it. M/widen come from `mllpds64` -- exp4's ML-LPDS arm -- so
# re-widening that cell re-widens these, and exp4 stays width-matched.
#
# K IS PINNED AT 18, not taken from ML_LPDS_COMMON. Both nets were unstable at
# K=18 -- MLSplitCDLNet tripped the loss backtrack almost immediately, and
# MLCDLNet was unstable too -- so they stay at the size that was observed rather
# than silently growing to K=30 with ML-LPDS. Neither is matched to ML-LPDS in K
# any more. MLSplitCDLNet is out of exp4 for now; its configs are still written.
#
#     mlcdlw2    K=18  widen=2  channels 64/128/256  ~393 GFLOP  144.8M params
#     mlsplitw2  K=18  widen=2  channels 64/128/256  ~848 GFLOP  144.8M params
#
# (Were 48/96/192 at 221/477 GFLOP and 81.5M; scaled by sum M_{l-1} M_l.)
#
# A suspect for the instability, not yet tested: `uball_project` bounds each
# (out, in) 7x7 slice, not each atom, so a level-l atom may grow to norm
# sqrt(M_{l-1}) (~8-10 at levels 2-3). ML-ISTA's unit step and the split net's mu
# clamp both assume ||B_l A_l|| <= 1, which that does not keep.
ML_CDL_COMMON = dict(
    {k: ML_LPDS_COMMON[k]
     for k in ("C", "P", "s", "degrees", "is_complex", "preproc", "L")},
    K=18, tau0=ML_LPDS_COMMON["lam0"], readout="level1")
_W2_SHAPE = {k: MODELS["mllpds64"]["params"][k] for k in ("M", "widen")}

MODELS.update({
    "mlcdlw2":   dict(type="MLCDLNet",
                      params=dict(ML_CDL_COMMON, **_W2_SHAPE)),
    "mlsplitw2": dict(type="MLSplitCDLNet",
                      params=dict(ML_CDL_COMMON, **_W2_SHAPE)),
})

OPT_IN = ("mllpdsw2", "mlcdlw2", "mlsplitw2",
          # the ML-LPDS width/depth sweep (exp5) -- OPT_IN so adding them does
          # not renumber exp1-exp4, whose arrays index the default list.
          "mllpds64", "mllpds64k20", "mllpds128k20",
          # depth instead of width (exp6)
          "mllpds48k54", "mllpds64k40")

# Both settings hold acs_lines at 20, so the two accelerations differ only in
# how far apart the outer lines sit.
# 16 is exp3's deep-acceleration arm; 8 and 4 are exp1/exp2's. All three are
# in one list so every cell exists for every R and the launchers select with
# ACCELS -- which also RENUMBERS, so each experiment keeps a dense array.
ACCELS = [8, 4, 16]

ANATOMIES = {
    "knee": dict(
        anatomy="knee",
        scale_fac=5000.0,
        kspace_root="../datasets/fastmri/knee/multicoil_{split}",
        smap_root="../datasets/fastmri_preprocessed/knee_coil_combined/pd/{split}",
        slices=(12, 25),
    ),
    "brain": dict(
        anatomy="brain",
        scale_fac=2000.0,
        kspace_root="../datasets/fastmri/brain/multicoil_{split}",
        # ESPIRiT, not the original Walsh maps. Walsh wrote no exact zeros,
        # so the organ mask (`smaps.abs().sum(0) > 0`) was all-True on brain
        # and use_organ_mask was a no-op there; and mri_awgn assumes unit-RSS
        # maps, which Walsh is not. knee was already ESPIRiT -- that mismatch
        # confounded every knee-vs-brain comparison. Written by
        # torch/espirit_smaps.sbatch -- which writes each split BESIDE the old
        # one, as `<split>_espirit` (scripts/make_espirit_smaps.py appends the
        # suffix to the split directory), so that is the path read here. It
        # used to say `brain_T2W_coil_combined_espirit/{split}`, a directory
        # nothing ever wrote. The Walsh `train`/`val` beside it are deleted.
        smap_root="../datasets/fastmri_preprocessed/brain_T2W_coil_combined/{split}_espirit",
        slices=(0, 8),
    ),
}

# TRAINING NOISE, PER ANATOMY. sigma is added in the COIL-IMAGE domain
# (operators/noise.py::mri_awgn) on data already scaled by the anatomy's
# scale_fac, so these are fractions of unit signal scale.
#
#     brain   U[0.04, 0.06]   ~5%     val 0.05
#     knee    U[0.01, 0.02]   ~1.5%   val 0.015
#
# PER-ANATOMY AND NOT GLOBAL, because the two were moved independently and a
# single constant made one of them collateral damage. brain went to [0.01, 0.02]
# and is now back at [0.04, 0.06]; knee was never asked to move. Anything
# reading this must index by anatomy -- `noise_std()` below, and the staleness
# check in torch/_mg_recon_body.sh, both do.
#
# brain is at [0.04, 0.06] because at [0.01, 0.02] the grid stopped separating
# the models: a comparison run in a regime where the prior does not have to do
# any work measures nothing about the prior. That is the same reason the range
# was first raised from [0.0, 0.01].
#
# HOW THIS RELATES TO THE Sljiva REFERENCE, because it is not a straight copy.
# `config/synthmri_closure.yaml` TRAINS at noise_level [0.00, 0.001] -- lower
# than either range here -- and the fastMRI eval scripts then EVALUATE at a
# single pinned 0.05 (`scripts/eval_guidedfastmri.jl:44`). 0.05 is the centre of
# the brain range, so brain now moves the reference's TEST operating point into
# TRAINING and matches train to test, rather than reproducing its protocol.
#
# That is a defensible design and it is the one asked for, but it is a different
# experiment from the paper's: a net trained at the level it is tested at should
# beat one trained near-noiseless and tested at 0.05, so these numbers are not
# directly comparable to published ones. Say which protocol produced a number
# whenever one is quoted.
#
# CHANGING A RANGE INVALIDATES THAT ANATOMY'S RUN DIRS. The launch guard in
# torch/_mg_recon_body.sh compares configs and REFUSES a run dir whose stored
# config differs on a key like noise_std, rather than silently continuing a net
# trained at a different sigma. Move the affected dirs aside (or pass
# FORCE_RESTART=1) before re-submitting.
NOISE_STD = {
    "brain": [0.04, 0.06],
    "knee":  [0.01, 0.02],
}

# Derived, never written by hand: mrireco.jl:277 evaluates at the MEAN of the
# training range, and a val point that drifts off-centre silently changes what
# every val curve in the grid measures.
VAL_NOISE_STD = {a: round(sum(v) / 2.0, 6) for a, v in NOISE_STD.items()}


def noise_std(anatomy):
    """This anatomy's training sigma range, as a fresh list."""
    try:
        return list(NOISE_STD[anatomy])
    except KeyError:
        raise ValueError(
            "no NOISE_STD entry for anatomy %r; known: %s"
            % (anatomy, sorted(NOISE_STD))) from None


def val_noise_std(anatomy):
    """This anatomy's validation sigma -- the mean of its training range."""
    noise_std(anatomy)                      # same error for an unknown anatomy
    return VAL_NOISE_STD[anatomy]


VAL_SEED = 1234


def cells(anatomy=None, only=None, accels=None, every=False):
    """The canonical cell order, optionally for one anatomy.

    knee and brain have SEPARATE sbatch files, so each indexes its own list and
    the array bound is per-anatomy. Passing no anatomy gives every cell.

    OPT_IN cells appear only when named in `only`, or with `every=True` -- which
    is what the generator uses when WRITING configs, so an opt-in launcher's
    regeneration always produces its config without the default list (and its
    indices) ever changing.
    """
    out = []
    for a in ([anatomy] if anatomy else ANATOMIES):
        for r in ACCELS:
            if accels and r not in accels:
                continue
            for model in MODELS:
                if only and model not in only:
                    continue
                if model in OPT_IN and not (only or every):
                    continue
                out.append((a, r, model))
    return out


# ===========================================================================
#  Config assembly
# ===========================================================================
def _variant(params):
    """"mg" or "flat" (or "L<L>w<widen>"), read off the params -- not off the
    cell's name.

    The flat and multigrid arms of a pair share a MODEL CLASS: `MGLPDSNet` with
    `K=30` is the flat LPDS stack and with `K=[6,[4,4,6]]` is the V-cycle, and
    likewise for AltSplitCDLNet's `denoiser_kws.K`. That is the point of the
    design -- one key is the whole ablation -- but it means `spec["type"]` alone
    cannot name a run, and two cells would land on the SAME wandb name.

    Derived from K rather than from the cell key so the tag cannot drift if
    someone edits a K by hand. The multilevel nets carry their depth in `L`
    instead (their K is a plain layer count, which would misread as "flat"),
    and two of them can share L and differ only in widen -- so both go in.
    """
    p = params.get("denoiser_kws", params)
    if "L" in p:
        return f"L{int(p['L'])}w{p.get('widen', 1)}"
    K = params.get("K", params.get("denoiser_kws", {}).get("K"))
    return "flat" if isinstance(K, int) else "mg"


# What a run is CALLED, which is not what its class is called. `MGLPDSNet` with
# K=30 is the plain LPDS baseline and labelling it "MGLPDSNet" in wandb would be
# actively misleading; AltSplitCDLNet is named for its denoiser, which is a
# multigrid CDLNet only in the mg arm.
_DISPLAY_NAME = {
    ("MGLPDSNet",      "flat"): "LPDSNet",
    ("MGLPDSNet",      "mg"):   "MGLPDSNet",
    ("AltSplitCDLNet", "flat"): "AltSplitCDLNet",
    ("AltSplitCDLNet", "mg"):   "AltSplitMGCDLNet",
}


def _pad_multiple(spec, params):
    """The image grid this cell's network needs, i.e. its `pad_stride`.

    Derived from the model block rather than written by hand, because the two
    drifting apart is exactly the failure the embedding exists to remove: a
    pad_multiple smaller than pad_stride sends `kspace_pre_process` back to
    resampling the mask, silently. `training/recon.py::_embed` re-checks this
    against the live model and raises, so a mismatch cannot reach a run.

    An AltSplitCDLNet's grid comes from its DENOISER: the outer loop runs
    `preproc="identity"` and never pads, while the prox slot is the multigrid
    net with the levels.

    The multilevel nets (MLCDLNet / MLLPDSNet) carry their depth in `L` and
    their `K` is a plain layer count, so reading levels off K would give
    pad_multiple = s instead of s * 2^(L-1).
    """
    if spec.get("type") == "E2EVarNet":
        # Works at the measured size; its NormUnet pads internally to a
        # multiple of 16. No image-domain embedding, so no constraint.
        return 1
    p = params.get("denoiser_kws", params)
    s_ = int(p.get("s", 1) or 1)
    if "L" in p:
        levels = int(p["L"])
    else:
        K = p.get("K")
        levels = 1 if (K is None or isinstance(K, int)) else len(list(K[1]))
    return s_ * (2 ** (levels - 1))


def _display_name(spec_type, params):
    """The run's name. Falls back to `<class>_<variant>` for an unmapped model
    so a new cell stays distinguishable instead of silently colliding."""
    variant = _variant(params)
    if spec_type == "MLLPDSNet":
        # L and widen do not separate this grid's ML-LPDS cells: every one of
        # them is L3w2 and they differ in M, P and init_norm. Without this a
        # width/depth sweep collides on one wandb name, which the check
        # below catches -- but the fix belongs here, not in a lookup table that
        # would need an entry per size.
        #
        # These names DID change when init_norm's default became "cascade".
        # That is deliberate: the cascade rescale makes it a different network
        # from the one already trained under the old name, and mixing the two
        # under one run name is exactly the confusion the rename avoids.
        # `save_dir` is keyed on the CELL name, so checkpoints are untouched.
        tag = "%sM%dK%s" % (variant, params.get("M", 0), params.get("K", "?"))
        if params.get("P") not in (None, 7):
            tag += "P%d" % params["P"]
        if params.get("init_norm", "cascade") != "cascade":
            tag += "-lvl"
        return "%s_%s" % (spec_type, tag)
    return _DISPLAY_NAME.get((spec_type, variant), f"{spec_type}_{variant}")


# Metrics every generated eval config asks for. `lpips` downloads pretrained
# weights on first use -- warm the cache on a login node before an offline run,
# or drop it from a config's list.
EVAL_METRICS = ["psnr", "ssim", "nrmse", "lpips"]
EVAL_SEED = 1234


def make_eval_config(save_dir, out_csv, comment):
    """A single-run eval config for `scripts/evaluate.py`.

    Deliberately tiny: the task, the data block and the noise level all come
    from the RUN's own `config.json` at evaluation time, so duplicating any of
    them here would only create something that can disagree with the run.
    Omitting `sigmas` means "each run's own `val_noise_std`", which reproduces
    the operating point its wandb val curve was measured at.
    """
    return {
        "_comment": comment,
        "runs": save_dir,
        "out": out_csv,
        "metrics": list(EVAL_METRICS),
        "seed": EVAL_SEED,
    }


def _lr_for(spec, args):
    """The initial LR for one cell.

    The nonlocal (group-attention) cells get their own, lower value. They kept
    tripping the averaged-loss backtrack at the shared 5e-4: the attention
    projections are the only parameters in this grid whose gradients pass
    through a softmax over a 15x15 neighbourhood at three grid levels, and an
    early step large enough to saturate it moves the whole adjacency at once.
    Backtracking then restores the checkpoint and halves the LR, which costs
    the epoch and leaves the run's effective schedule undefined -- so start
    where it does not fire rather than letting the backtracker find it.

    Keyed on the cell CARRYING attention, not on a tag list, so a GroupCDL or
    MGGroupCDL cell added later inherits it without a second edit here. The
    non-attention cells are untouched at args.lr.
    """
    params = spec["params"]
    is_group = ("attn_backend" in params
                or "attn_backend" in params.get("denoiser_kws", {}))
    return args.lr_group if is_group else args.lr


def make_config(anatomy, r, model, args):
    a = ANATOMIES[anatomy]
    spec = MODELS[model]

    params = dict(spec["params"])
    if spec["type"] == "E2EVarNet":
        # The mask's ACS width. VarNet's SensitivityModel can infer it, but the
        # config knows it exactly, and the inference assumes a symmetric centre.
        params["acs_lines"] = 20
    # Notes accumulate: a cell can be both an LPDS variant and on a backend
    # that swapped its similarity, and losing either one in the run directory
    # is how a caveat stops travelling with its numbers.
    notes = [spec["note"]] if spec.get("note") else []
    attn = getattr(args, "attn", "flex")

    if "attn_backend" in params:
        params["attn_backend"] = attn
        if attn == "flex":
            params["flex_block_size"] = 128
            # These models are complex, so flex cannot carry pidistance -- see
            # the GROUP comment. Record the swap in the config rather than
            # letting the model raise on the first forward.
            if params.get("sim_fun") in ("pidistance", "pidot") \
                    and params.get("is_complex", True):
                notes.append(
                    f"attn_backend='flex' cannot fuse "
                    f"sim_fun='{params['sim_fun']}' for complex features "
                    f"(the score is Re<q,k>; the modulus needs Im<q,k> too), "
                    f"so the similarity is 'distance' here. --attn triton "
                    f"keeps pidistance and stays fused; --attn gather keeps it "
                    f"and materialises the window.")
                params["sim_fun"] = "distance"
        elif attn == "triton":
            if params.get("sim_fun") not in ("pidistance", "pidot"):
                raise SystemExit(
                    f"[configs] --attn triton with sim_fun="
                    f"'{params.get('sim_fun')}': the triton kernel implements "
                    f"only the phase-invariant similarities, and models/prox.py "
                    f"rejects anything else on that backend. Either use "
                    f"--attn flex (what these experiments do), or set the "
                    f"cell's sim_fun to 'pidistance'.")
            notes.append(
                "attn_backend='triton' -- the fused kernel in "
                "models/circulant_triton.py, carrying the reference's exact "
                "complex pidistance (which FlexAttention cannot express). "
                "Re-verify with `python -m tests.test_triton_attention` on a "
                "GPU node after any change to that file.")
    note = " ".join(notes) if notes else None

    pad_multiple = _pad_multiple(spec, params)

    def data(split, shuffle_slices):
        return {
            "name": "fastmri",
            "task": "recon",
            "anatomy": a["anatomy"],
            "batch_size": 1,
            "crop_size": None,
            "center_crop": None,
            "random_flips": False,
            "start_slice": a["slices"][0],
            "end_slice": a["slices"][1] if shuffle_slices else a["slices"][0] + 1,
            "scale_fac": a["scale_fac"],
            "kspace_root": a["kspace_root"].format(split=split),
            "smap_root": a["smap_root"].format(split=split),
            # Image-domain embedding: the loader reports the smallest grid
            # >= the image that divides by this, and training unrolls on it
            # via `E @ Truncate`. Padding the IMAGE keeps E exact; padding the
            # OPERATOR (the old path) resamples the mask. See
            # operators/truncate.py and notebooks/pad_stride_init_gap.ipynb.
            "pad_multiple": pad_multiple,
            # WHERE THE ORGAN MASK COMES FROM, recorded even though "rss" is
            # the loader default: it decides which pixels every masked metric
            # is computed over, so a run that does not state it cannot be
            # compared with one that does. "rss" thresholds the coil RSS
            # (physics/object_mask.py); "smaps" is the old coil-support test,
            # which is DILATED -- ESPIRiT's eigenvalue map is band-limited to
            # `kernel_size` k-space samples and cannot fall off faster than
            # ~N/ks pixels, so the support ran tens of pixels past the skull
            # and masked metrics were scoring air.
            #
            # CHANGING IT INVALIDATES EVERY MASKED RUN, exactly like a change
            # of noise_std: the numbers measure a different region. The launch
            # guard in torch/_mg_recon_body.sh compares configs and will refuse
            # the old run dirs, which is the intended behaviour.
            "organ_mask_source": "rss",
        }

    total_steps = args.num_epochs * args.steps_per_epoch

    return {
        "task": "recon",
        "experiment": {
            # Load-bearing, not decoration: the flat and multigrid arms share a
            # model class, so `spec["type"]` alone collides on one wandb run.
            "name": f"{_display_name(spec['type'], params)}_{anatomy}_R{r}_synth",
        },
        "model": dict(
            {"type": spec["type"], "params": params},
            **({"_comment": note} if note else {}),
        ),
        "paths": {
            "save_dir": f"trained_nets/mg_recon/{anatomy}/{model}_R{r}",
            "ckpt": None,
        },
        "data": {
            "train": data("train", shuffle_slices=True),
            # One fixed slice per val volume: combined with val_seed this makes
            # the validation set literally identical across every cell.
            "val": data("val", shuffle_slices=False),
        },
        "training": {
            "num_epochs": args.num_epochs,
            "steps_per_epoch": args.steps_per_epoch,
            "val_every_epochs": args.val_every_epochs,
            "noise_std": noise_std(anatomy),
            "noise_dist": "uniform",
            "val_noise_std": val_noise_std(anatomy),
            "val_seed": VAL_SEED,
            "loss_type": "magnitude-nl1-nl2",
            "clip_grad": 1.0,
            # Loss AND metrics AND the wandb panel -- see training/recon.py.
            "use_organ_mask": bool(getattr(args, "organ_mask", False)),
            # None disables the averaged-loss backtrack THRESHOLD (a non-finite
            # loss still triggers a protective restore). The margin is in loss
            # units and this grid's scale is not known in advance -- pick one
            # from the first runs rather than inheriting recon.json's 5, which
            # is large enough to never fire for mag_nl1_nl2.
            "backtrack_thresh": None,
            "backtrack_factor": 0.9,
        },
        "mri": {
            "R": r,
            "acs_lines": 20,
            "mask_dist": "uniform",
            "mask_offset": 0,
            "kspace_type": "simulated",
            "whiten_kspace": False,
        },
        "optimizer": {
            "type": "Adam",
            "params": {"lr": _lr_for(spec, args)},
        },
        "scheduler": {
            "type": "CosineAnnealingLR",
            "params": {"eta_min": 1.0e-6, "T_max": total_steps},
        },
        "wandb": {"project": "mg_recon", "id": None},
    }


# ===========================================================================
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="config", help="config root (default: config)")
    # 6000 x 50 = 300000 steps; the cosine schedule's T_max is derived from
    # these two, so the annealing always spans exactly one full run.
    p.add_argument("--num-epochs", type=int, default=6000)
    p.add_argument("--steps-per-epoch", type=int, default=50)
    p.add_argument("--val-every-epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=5.0e-4)
    p.add_argument("--lr-group", type=float, default=2.0e-4,
                   help="initial LR for the group-attention cells "
                        "(MGGroupLPDS, and any GroupCDL/MGGroupCDL added "
                        "later). Lower than --lr because they backtrack at "
                        "5e-4. Cosine T_max is unchanged, so these cells "
                        "anneal from a lower start over the same budget.")
    p.add_argument("--only", nargs="*", default=None,
                   help="restrict to these model tags")
    p.add_argument("--anatomy", choices=("knee", "brain"), default=None,
                   help="restrict to one anatomy. The sbatch files pass this so "
                        "each indexes its own cell list.")
    p.add_argument("--attn", choices=("triton", "flex", "gather"), default="flex",
                   help="attention backend for the group models. 'flex' (default) "
                        "fuses and is what the current experiments use, with "
                        "sim_fun='distance' -- it cannot carry a phase-invariant "
                        "similarity on complex features. 'triton' keeps the "
                        "reference's exact pidistance and stays fused, but REJECTS "
                        "sim_fun='distance', so switching backend also means "
                        "switching similarity. 'gather' keeps pidistance and "
                        "materialises the window (~39x slower, ~15x the memory).")
    p.add_argument("--dry-run", action="store_true",
                   help="print the configs instead of writing them (no torch needed)")
    p.add_argument("--list-cells", action="store_true",
                   help="print the sbatch index -> cell map and exit")
    p.add_argument("--organ-mask", action="store_true",
                   help="restrict the loss, the metrics and the logged panel "
                        "to the coil-sensitivity support (`organ_mask`). Use "
                        "for knee, where the air background is large and its "
                        "reconstruction is not what the comparison is about. "
                        "OFF by default and deliberately not per-anatomy: it "
                        "changes what PSNR/NRMSE/SSIM MEAN, so a masked run "
                        "cannot go in the same table as an unmasked one.")
    p.add_argument("--accels", nargs="*", type=int, default=None,
                   help="restrict to these accelerations (with --list-cells)")
    args = p.parse_args()

    if args.list_cells:
        # --only / --accels narrow the list AND renumber it, so an experiment
        # that runs a subset gets a dense 0..N-1 array range of its own rather
        # than having to know the full grid's indices.
        unknown = sorted(set(args.only or ()) - set(MODELS))
        if unknown:
            raise SystemExit(f"[cells] unknown model tag(s) {unknown}; "
                             f"known: {sorted(MODELS)}")
        for i, (anatomy, r, model) in enumerate(
                cells(args.anatomy, only=args.only, accels=args.accels)):
            print(f"{i}\t{anatomy}\tR{r}\t{model}")
        return

    if not args.dry_run:
        # write_config is the repo's canonical writer: it re-reads the file with
        # yaml.safe_load (which is how train.py loads it) and rejects anything
        # that would come back as a string, e.g. json's `1e-06` for eta_min.
        #
        # It lives in training/config_io.py, which imports nothing but the
        # standard library and PyYAML. `import training.config_io` would still
        # execute training/__init__.py (-> torchvision), so the module is loaded
        # BY PATH: writing a config has no business requiring a GPU training
        # environment, and requiring one meant this script could only run on the
        # cluster. training.common re-exports the same function for everyone else.
        import importlib.util
        _p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "training", "config_io.py")
        try:
            _s = importlib.util.spec_from_file_location("_immap_config_io", _p)
            _m = importlib.util.module_from_spec(_s)
            _s.loader.exec_module(_m)
            write_config = _m.write_config
        except Exception as e:                      # noqa: BLE001
            raise SystemExit(
                f"could not load write_config from {_p} ({e}).\n"
                f"It needs only PyYAML. Use --dry-run to inspect the configs "
                f"without writing.")

    written = []
    seen_names = {}
    eval_roots = {}
    # every=True: opt-in cells are WRITTEN on every regeneration (only their
    # listing is gated), so an opt-in launcher always finds its config.
    for anatomy, r, model in cells(args.anatomy, every=True):
        if args.only and model not in args.only:
            continue

        cfg = make_config(anatomy, r, model, args)
        path = os.path.join(args.out, anatomy, "mg", f"{model}_R{r}.json")

        # Two cells sharing a wandb name interleave their curves into one run,
        # and the loss is silent -- nothing errors, the plot is just wrong. The
        # flat/mg pairs share a model class, so this is one edit away at all
        # times. Fail here instead.
        name = cfg["experiment"]["name"]
        if name in seen_names:
            raise SystemExit(
                f"duplicate experiment name {name!r}: cells {seen_names[name]} "
                f"and {(anatomy, r, model)} would log into the same wandb run. "
                f"Add an entry to _DISPLAY_NAME.")
        seen_names[name] = (anatomy, r, model)

        if args.dry_run:
            print(f"--- {path} ---")
            print(json.dumps(cfg, indent=4))
        else:
            write_config(cfg, path)
            written.append(path)

            # One eval config per cell, emitted HERE rather than hand-written
            # so `runs` is the same string as this run's `paths.save_dir`. A
            # hand-maintained copy goes stale the moment a cell is renamed, and
            # the failure is quiet: evaluate.py just reports "no runs found".
            save_dir = cfg["paths"]["save_dir"]
            eval_cfg = make_eval_config(
                save_dir,
                f"results/eval/{anatomy}/{model}_R{r}.csv",
                f"Evaluate {name} alone, over its own validation set. "
                f"`metrics` is the toggle. Generated by "
                f"scripts/make_mg_recon_configs.py -- edit that, not this.")
            assert eval_cfg["runs"] == save_dir
            eval_path = os.path.join(args.out, "eval", anatomy,
                                     f"{model}_R{r}.json")
            write_config(eval_cfg, eval_path)
            written.append(eval_path)
            eval_roots.setdefault(anatomy, []).append(save_dir)

    # ...plus one aggregate per anatomy, for sweeping the whole grid at once.
    if not args.dry_run:
        for anatomy, dirs in eval_roots.items():
            root = os.path.commonpath(dirs).replace(os.sep, "/")
            path = os.path.join(args.out, "eval", f"{anatomy}.json")
            write_config(make_eval_config(
                root, f"results/eval/{anatomy}.csv",
                f"Evaluate every {anatomy} cell in one sweep ({len(dirs)} runs). "
                f"Per-cell configs are in {args.out}/eval/{anatomy}/."), path)
            written.append(path)

    if written:
        print(f"wrote {len(written)} configs under {args.out}/")
        for path in written:
            print(f"  {path}")


if __name__ == "__main__":
    main()
