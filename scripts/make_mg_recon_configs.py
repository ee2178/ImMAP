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
MG_COMMON = dict(
    M=169, C=1, P=7, s=2, widen=1, degrees=1,
    is_complex=True, preproc="kspace", resize_noise=True,
    tau0=1.0e-3, alpha0=1.0e-1, alpha_conv=False,
)

# Group (nonlocal) prox. windowsize=9 / dK=2 come from the multigrid sweep;
# groupcdl.yaml's W=35 / dK=5 belong to the standalone denoising experiment.
# Held identical across V-cycle and baseline cells so the comparison isolates
# the V-cycle rather than the attention configuration.
#
# ATTENTION BACKEND -- the one place this grid cannot match the reference.
#
# The reference uses similarity=pidistance, the phase-invariant distance
#     -1/2||q||^2 + |<q,k>| - 1/2||k||^2.
# FlexAttention's score_mod sees ONE number per pair (the raw dot product). For
# REAL features <q,k> = q.k, so |<q,k>| is |score| and the fusion is exact --
# models/circulant_flex.py now does it. For COMPLEX features (these configs)
# the stacked score is only Re<q,k>, and the modulus needs Im<q,k> too: a
# second bilinear form no score_mod can reach. Julia gets away with it because
# its flash path is a bespoke kernel that accumulates both.
#
# `triton` (the DEFAULT) is the one that gets both: models/circulant_triton.py
# accumulates Re<q,k> AND Im<q,k>, so the modulus is available -- exactly what a
# score_mod cannot reach. Verified against the gather path on an L40S
# (tests/test_triton_attention.py, 115/115) at 0.29 ms / 69 MiB per apply versus
# gather's 11.2 ms / 1074 MiB, so there is no longer a reason to deviate.
#
#   --attn triton  (default) exact pidistance, fused
#   --attn flex              fused, but sim_fun drops to "distance"
#   --attn gather            exact pidistance, materialises (B, Mh, Q, W^2)
#
# flex remains useful as a fallback on a box where triton is unavailable; the
# similarity swap it forces is written into the emitted config so no run is
# ambiguous about which similarity it trained on.
GROUP = dict(
    Mh=64, W=9, dK=2, nheads=1, gamma0=0.8,
    sim_fun="pidistance", init_strategy="semi_orthogonal",
    subgrad_mode="rigorous", attn_backend="triton",
)

# One outer V-cycle over three levels, 8 smoothing sweeps each.
VCYCLE_K = [1, [8, 8, 8]]
# groupcdl.yaml's K: 30. NOTE this is the reference's own baseline and it is
# NOT iteration-matched to the V-cycle (30 sweeps vs 8 on the fine grid) nor
# parameter-matched (a V-cycle also carries its coarse levels, the dF copies
# and alpha). scripts/eval_mg_recon.py reports n_params next to PSNR so the
# asymmetry is visible rather than papered over.
BASELINE_K = 30

# READ THIS BEFORE COMPARING THE `mglpds` CELLS TO ANYTHING PUBLISHED.
#
# ImMAP's MGLPDS is `MGCDLNet(dual=True)`: the Fenchel/Moreau dual of the CDL
# prox -- clipping instead of shrinkage, read-out `y~ - Dz`. It is a one-flag
# variant of the same V-cycle CDLNet, sharing every other hyperparameter.
#
# The reference's multigrid MRI model *called* LPDS is `mg_lpdsnet`
# (Sljiva/src/networks/mg_lpds.jl): a primal-dual splitting network with its own
# sensenet, dual-channel widening, and separate lambda/theta step parameters. It
# is NOT ported. Same name, different architecture.
#
# So these two cells answer "does the dual read-out help OUR V-cycle CDLNet",
# which is a real question. They do not reproduce Sljiva's MGLPDS, and their
# numbers are not comparable to anything reported for it. The caveat rides along
# in each emitted config's `_comment` so it survives into the run directory.
_LPDS_NOTE = (
    "MGLPDS here is MGCDLNet(dual=True) -- the Fenchel/Moreau dual of the CDL "
    "prox (clipping instead of shrinkage, read-out y~ - Dz). It is NOT Sljiva's "
    "mg_lpdsnet (src/networks/mg_lpds.jl), a primal-dual splitting network with "
    "a sensenet, which is not ported to ImMAP. Same name, different "
    "architecture: these numbers measure the dual read-out on our V-cycle "
    "CDLNet and are not comparable to published MGLPDS results.")

# AltSplitCDLNet, from altsplit.yaml. `denoiser_kws.K` is the ONLY difference
# between the two LADMM cells: altsplit.yaml ships `K: [1, [4,4,8]]` with
# `# K: 6` commented directly above it, so both come from the reference.
# preproc stays "identity" -- with smap_update the unroll keeps raw k-space and
# re-forms E^H y per layer, so there is nothing for kspace preprocessing to act on.
def _altsplit(denoiser_K):
    return dict(
        type="AltSplitCDLNet",
        lr=2.0e-4,          # the CG solves make this one touchier than the rest
        params=dict(
            admm_iters=6, reuse_latent=True, smap_update=True, rho0=1.0,
            cg_maxit=10, cg_tol=1.0e-4, implicit_cg=True, preproc="identity",
            denoiser_type="mgcdlnet",
            denoiser_kws=dict(
                K=denoiser_K, M=169, C=1, P=7, s=2, degrees=1,
                tau0=1.0e-3, is_complex=True, dual=False, widen=1,
                alpha0=1.0, alpha_conv=False, resize_noise=True,
            ),
            smap_kws=dict(sigma_g=1.5, sigma_max=3.0, sigma_min=0.3,
                          mu0=1.0, gamma0=0.01),
        ),
    )


MODELS = {
    # models/lpdsnet.py::LPDSNet -- its own class, not a CDLNet variant. Its
    # l0 / eta_0 / theta_0 defaults already equal Sljiva lpdsnet.yaml's
    # lambda0 / tau0 / theta0. M / P / s / K are held at the grid's values
    # rather than lpdsnet.yaml's (M=225, p=9, K=40) so the only thing separating
    # it from mglpds is architecture, not capacity.
    "lpdsnet": dict(
        type="LPDSNet",
        params=dict(K=BASELINE_K, M=169, P=7, s=2, C=1,
                    l0=1.0e-3, eta_0=0.5, theta_0=0.0,
                    adaptive=True, init=True, preproc="kspace"),
    ),
    # dual=True is what build_model forces for this type anyway; stated here so
    # the config shows the flag that makes it the Fenchel/clipping read-out.
    "mglpds": dict(type="MGLPDS", note=_LPDS_NOTE,
                   params=dict(MG_COMMON, K=VCYCLE_K, dual=True)),
    "altsplit":   _altsplit(6),
    "mgaltsplit": _altsplit([1, [4, 4, 8]]),
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


def cells(anatomy=None):
    """The canonical cell order, optionally for one anatomy.

    knee and brain have SEPARATE sbatch files, so each indexes its own list and
    the array bound is per-anatomy. Passing no anatomy gives every cell, which
    is what the generator uses when writing configs.
    """
    out = []
    for a in ([anatomy] if anatomy else ANATOMIES):
        for r in ACCELS:
            for model in MODELS:
                out.append((a, r, model))
    return out


# ===========================================================================
#  Config assembly
# ===========================================================================
def make_config(anatomy, r, model, args):
    a = ANATOMIES[anatomy]
    spec = MODELS[model]

    params = dict(spec["params"])
    # Notes accumulate: a cell can be both an LPDS variant and on a backend
    # that swapped its similarity, and losing either one in the run directory
    # is how a caveat stops travelling with its numbers.
    notes = [spec["note"]] if spec.get("note") else []
    attn = getattr(args, "attn", "triton")

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
            notes.append(
                "attn_backend='triton' -- the fused kernel in "
                "models/circulant_triton.py, carrying the reference's exact "
                "complex pidistance (which FlexAttention cannot express). "
                "Re-verify with `python -m tests.test_triton_attention` on a "
                "GPU node after any change to that file.")
    note = " ".join(notes) if notes else None

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
        }

    total_steps = args.num_epochs * args.steps_per_epoch

    return {
        "task": "recon",
        "experiment": {
            "name": f"{spec['type']}_{anatomy}_R{r}_synth",
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
            "use_organ_mask": False,
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
            "params": {"lr": spec.get("lr", args.lr)},
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
    p.add_argument("--num-epochs", type=int, default=500)
    p.add_argument("--steps-per-epoch", type=int, default=200)
    p.add_argument("--val-every-epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=5.0e-4)
    p.add_argument("--only", nargs="*", default=None,
                   help="restrict to these model tags")
    p.add_argument("--anatomy", choices=("knee", "brain"), default=None,
                   help="restrict to one anatomy. The sbatch files pass this so "
                        "each indexes its own cell list.")
    p.add_argument("--attn", choices=("triton", "flex", "gather"), default="triton",
                   help="attention backend for the group models. 'triton' (default) "
                        "keeps the reference's exact pidistance and stays fused. "
                        "'flex' also fuses but cannot carry pidistance for complex "
                        "features, so the similarity drops to 'distance'; use it "
                        "where triton is unavailable. 'gather' keeps pidistance and "
                        "materialises the window (~39x slower, ~15x the memory).")
    p.add_argument("--dry-run", action="store_true",
                   help="print the configs instead of writing them (no torch needed)")
    p.add_argument("--list-cells", action="store_true",
                   help="print the sbatch index -> cell map and exit")
    args = p.parse_args()

    if args.list_cells:
        for i, (anatomy, r, model) in enumerate(cells(args.anatomy)):
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
    for anatomy, r, model in cells(args.anatomy):
        if args.only and model not in args.only:
            continue

        cfg = make_config(anatomy, r, model, args)
        path = os.path.join(args.out, anatomy, "mg", f"{model}_R{r}.json")

        if args.dry_run:
            print(f"--- {path} ---")
            print(json.dumps(cfg, indent=4))
        else:
            write_config(cfg, path)
            written.append(path)

    if written:
        print(f"wrote {len(written)} configs under {args.out}/")
        for path in written:
            print(f"  {path}")


if __name__ == "__main__":
    main()
