#!/usr/bin/env python3
"""
Generate the multigrid fastMRI reconstruction grid.

PyTorch counterpart of Sljiva's `scripts/makeconfigs_multigrid.jl`: writes one
config per (anatomy, acceleration, model) cell to

    config/<anatomy>/mg/<model>_R<r>.json

4 models x 2 accelerations = 8 runs PER ANATOMY. knee and brain have separate
sbatch files (`slurm/mg_recon_{knee,brain}.sbatch`); each asks this script for
its own cell list via `--list-cells --anatomy <a>`, so the two stay in sync.

Every cell trains on SYNTHETIC k-space (`kspace_type: "simulated"`): the clean
coil-combined image is pushed through Sense -> Fourier -> mask with complex AWGN
at sigma ~ U[0, 0.01] added in the coil-image domain. See
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
    #   * it has no noise-adaptive parameter, so sigma ~ U[0, 0.01] is a
    #     handicap the sigma-conditioned cells do not carry.
    # It also has ~10x the parameters of the flat LPDSNet; `count_parameters`
    # in each config records it.
    "varnet": dict(
        type="E2EVarNet",
        params=dict(num_cascades=12, sens_chans=8, sens_pools=4,
                    chans=18, pools=4, mask_center=True),
        note=("fastMRI E2E-VarNet baseline at published defaults. Estimates "
              "its own sensitivity maps and returns RSS magnitude, so it is "
              "comparable on PSNR/SSIM/NRMSE and on nothing phase-sensitive."),
    ),
}

# Both settings hold acs_lines at 20, so the two accelerations differ only in
# how far apart the outer lines sit.
ACCELS = [8, 4]

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
        smap_root="../datasets/fastmri_preprocessed/brain_T2W_coil_combined/{split}",
        slices=(0, 8),
    ),
}

NOISE_STD = [0.0, 0.01]        # sigma ~ U[0, 0.01], the experiment's whole point
VAL_NOISE_STD = 0.005          # mean(NOISE_STD): mrireco.jl:277 evaluates there
VAL_SEED = 1234


def cells(anatomy=None, only=None, accels=None):
    """The canonical cell order, optionally for one anatomy.

    knee and brain have SEPARATE sbatch files, so each indexes its own list and
    the array bound is per-anatomy. Passing no anatomy gives every cell, which
    is what the generator uses when writing configs.
    """
    out = []
    for a in ([anatomy] if anatomy else ANATOMIES):
        for r in ACCELS:
            if accels and r not in accels:
                continue
            for model in MODELS:
                if only and model not in only:
                    continue
                out.append((a, r, model))
    return out


# ===========================================================================
#  Config assembly
# ===========================================================================
def _variant(params):
    """"mg" or "flat", read off the actual K -- not off the cell's name.

    The flat and multigrid arms of a pair share a MODEL CLASS: `MGLPDSNet` with
    `K=30` is the flat LPDS stack and with `K=[6,[4,4,6]]` is the V-cycle, and
    likewise for AltSplitCDLNet's `denoiser_kws.K`. That is the point of the
    design -- one key is the whole ablation -- but it means `spec["type"]` alone
    cannot name a run, and two cells would land on the SAME wandb name.

    Derived from K rather than from the cell key so the tag cannot drift if
    someone edits a K by hand.
    """
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
    """
    if spec.get("type") == "E2EVarNet":
        # Works at the measured size; its NormUnet pads internally to a
        # multiple of 16. No image-domain embedding, so no constraint.
        return 1
    p = params.get("denoiser_kws", params)
    K = p.get("K")
    s_ = int(p.get("s", 1) or 1)
    levels = 1 if (K is None or isinstance(K, int)) else len(list(K[1]))
    return s_ * (2 ** (levels - 1))


def _display_name(spec_type, params):
    """The run's name. Falls back to `<class>_<variant>` for an unmapped model
    so a new cell stays distinguishable instead of silently colliding."""
    variant = _variant(params)
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
            "noise_std": NOISE_STD,
            "noise_dist": "uniform",
            "val_noise_std": VAL_NOISE_STD,
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
            "params": {"lr": args.lr},
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
        # Imported here so --dry-run / --list-cells work without torch.
        # write_config is the repo's canonical writer: it re-reads the file with
        # yaml.safe_load (which is how train.py loads it) and rejects anything
        # that would come back as a string, e.g. json's `1e-06` for eta_min.
        try:
            from training.common import write_config
        except ImportError as e:
            raise SystemExit(
                f"could not import training.common.write_config ({e}).\n"
                f"Run this in the training env (conda activate gcdl), or use "
                f"--dry-run to inspect the configs without writing.")

    written = []
    seen_names = {}
    eval_roots = {}
    for anatomy, r, model in cells(args.anatomy):
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
