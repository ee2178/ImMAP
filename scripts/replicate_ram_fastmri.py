#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reproduce RAM's reported fastMRI numbers, to check the harness before trusting it.

    python scripts/replicate_ram_fastmri.py --task multicoil_x8
    python scripts/replicate_ram_fastmri.py --task singlecoil_x4 --volumes 40

WHAT THE PAPER SAYS (arXiv 2503.08915, Table 2 + Appendix A.1 / D / B.2)
-----------------------------------------------------------------------
    single-coil MRI x4     PSNR 34.39   SSIM 0.853
    single-coil MRI x8     PSNR 31.50   SSIM 0.813
    multi-coil  MRI x8     PSNR 35.5    SSIM 0.937

  * fastMRI **brain**, virtual coil-combination of the raw multicoil k-space.
    70748 training slices; the 21842 VALIDATION slices are what Table 2 reports.
  * single-coil: "the single-coil accelerated acquisition procedure from [67]"
    (the fastMRI paper), Cartesian RANDOM masks, R in {4, 8}.
  * multi-coil: "simulated L=15 coil maps", R=8.
  * Gaussian noise sigma = 5e-4.
  * PSNR/SSIM on magnitude; complex carried as 2 channels.
  * Training patches were (C, 128, 128), C in {1,2,3}.

TWO THINGS TO KNOW BEFORE READING ANY RESULT
--------------------------------------------
1. **These numbers are IN-DISTRIBUTION.** Appendix B.2 lists "FastMRI -
   Undersampled MRI Reconstruction" among the TRAINING inverse problems, and
   A.1 says the brain training split was used to train the model. "Zero-shot"
   here means "no test-time finetuning", NOT "unseen task". So this script is a
   harness check, not evidence about RAM on your data -- and if it does NOT
   reproduce, the likely fault is in the harness (layout, mask, normalisation),
   not in RAM.

2. **sigma = 5e-4 is below RAM's own floor.** `ram/models/ram.py::forward` does
   `sigma = max(sigma, 1e-3)` before building the conditioning map, so the
   network is told 1e-3 whatever the paper used. That is upstream behaviour,
   reproduced here rather than worked around.

WHAT THE PAPER DOES NOT SPECIFY
-------------------------------
Left as flags, defaulting to the most likely reading, so you can see which one
moves the number rather than silently baking a guess in:

    --acs            centre lines. Not stated. fastMRI's own convention is a
                     centre FRACTION (0.08 at R=4, 0.04 at R=8), which is what
                     --acs-frac uses by default.
    --normalise      not stated. Default `max`: divide by max|x| per slice.
    --coil-sim       how the L=15 maps were simulated. Not stated. Default
                     `birdcage`, the usual choice.
    --crop           evaluation resolution not stated; default is no crop.

Anything you change here should be recorded next to the number you quote.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Table 2, for the comparison line at the end.
PAPER = {
    "singlecoil_x4": dict(psnr=34.39, ssim=0.853, coils=1, R=4),
    "singlecoil_x8": dict(psnr=31.50, ssim=0.813, coils=1, R=8),
    "multicoil_x8": dict(psnr=35.5, ssim=0.937, coils=15, R=8),
}
PAPER_SIGMA = 5e-4


# ===========================================================================
#  masks -- the fastMRI procedure, not this repo's
# ===========================================================================
def fastmri_random_mask(W, R, center_frac, generator=None):
    """Cartesian RANDOM mask, fastMRI's `RandomMaskFunc`.

    Deliberately NOT `physics/mask.py::make_acc_mask`, which is equispaced --
    the paper cites the fastMRI acquisition procedure, and equispaced vs random
    is worth 0.5-1 dB on its own, so using ours here would quietly break the
    comparison.

    fastMRI's rule: keep `center_frac` of the lines at the centre, then choose
    the rest uniformly at random so the OVERALL acceleration is R.
    """
    num_low = int(round(W * center_frac))
    # solve for the probability that makes the total kept fraction 1/R
    prob = (W / R - num_low) / (W - num_low)
    if generator is None:
        pick = torch.rand(W) < prob
    else:
        pick = torch.rand(W, generator=generator) < prob
    pad = (W - num_low + 1) // 2
    pick[pad:pad + num_low] = True
    return pick.float()


def default_center_frac(R):
    """fastMRI's published pairing: 0.08 at R=4, 0.04 at R=8."""
    return 0.08 if R == 4 else 0.04


# ===========================================================================
#  coil maps -- "simulated L=15", method unstated
# ===========================================================================
def birdcage_maps(n_coils, H, W, device, relative_radius=1.5):
    """Smooth birdcage-like complex maps, normalised to unit RSS.

    The paper says only "simulated L=15 coil maps". This is the usual
    construction (as in sigpy's `birdcage_maps`): coils on a ring, each map a
    smooth complex field falling off with distance. If your reproduction lands
    close but not exact, this is one of the few free choices left.
    """
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W),
                            indexing="ij")
    maps = []
    for c in range(n_coils):
        ang = 2 * math.pi * c / n_coils
        cy, cx = relative_radius * math.sin(ang), relative_radius * math.cos(ang)
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2
        amp = 1.0 / (d2 + 1e-3)
        phase = torch.atan2(yy - cy, xx - cx)
        maps.append(amp * torch.exp(1j * phase))
    m = torch.stack(maps).to(torch.complex64)[None].to(device)
    return m / (m.abs().pow(2).sum(1, keepdim=True).sqrt() + 1e-8)


# ===========================================================================
#  data
# ===========================================================================
def load_val_slices(smap_root, n_volumes, n_slices, device, normalise="max",
                    crop=None):
    """Coil-combined complex slices from the preprocessed brain VAL volumes.

    The paper coil-combines the raw multicoil k-space itself; the `image`
    dataset here is this repo's equivalent, so it stands in for that. It is the
    closest available match, not the identical object -- flagged in the summary.
    """
    import h5py
    files = sorted(glob.glob(os.path.join(smap_root, "*.h5")))
    if not files:
        raise SystemExit(f"[rep] no volumes under {smap_root}")

    out = []
    for path in files[:n_volumes]:
        with h5py.File(path, "r") as f:
            n = f["image"].shape[0]
            take = list(range(min(n_slices, n)))
            img = np.asarray(f["image"][take])
        for k in range(img.shape[0]):
            x = torch.from_numpy(img[k]).to(torch.complex64)
            x = x.reshape(1, 1, *img.shape[-2:]).to(device)
            if crop:
                H, W = x.shape[-2:]
                t, l = (H - crop) // 2, (W - crop) // 2
                x = x[..., t:t + crop, l:l + crop]
            if normalise == "max":
                s = float(x.abs().max())
            elif normalise == "std":
                s = float(x.abs().std())
            elif normalise == "none":
                s = 1.0
            else:
                raise SystemExit(f"[rep] unknown --normalise {normalise!r}")
            if s <= 0:
                continue
            out.append(x / s)
    return out


# ===========================================================================
#  metrics -- magnitude, as the paper reports
# ===========================================================================
def psnr_ssim(est, ref):
    from training.metrics import compute_metrics
    m = compute_metrics(ref.abs(), est.abs())
    return float(m["psnr"]), float(m["ssim"])


# ===========================================================================
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=sorted(PAPER), default="multicoil_x8")
    p.add_argument("--config", default=None,
                   help="config to read the brain val root from")
    p.add_argument("--volumes", type=int, default=20)
    p.add_argument("--slices", type=int, default=4, help="slices per volume")
    p.add_argument("--sigma", type=float, default=PAPER_SIGMA)
    p.add_argument("--acs-frac", type=float, default=None,
                   help="centre fraction; default 0.08 at R=4, 0.04 at R=8")
    p.add_argument("--normalise", choices=("max", "std", "none"), default="max")
    p.add_argument("--coil-sim", choices=("birdcage",), default="birdcage")
    p.add_argument("--crop", type=int, default=None)
    p.add_argument("--noise-domain", choices=("measurement", "coil_image"),
                   default="measurement",
                   help="THEIR convention is 'measurement' (y = Ax + n); this "
                        "repo's is 'coil_image'. Default matches the paper.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="write a json summary here")
    args = p.parse_args(argv)

    spec = PAPER[args.task]
    R, n_coils = spec["R"], spec["coils"]
    center_frac = (args.acs_frac if args.acs_frac is not None
                   else default_center_frac(R))
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    cfg_path = args.config or os.path.join(ROOT, "config", "brain", "mg",
                                           "mglpds_R8.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    root = cfg["data"]["val"]["smap_root"]
    if not os.path.isabs(root):
        root = os.path.normpath(os.path.join(ROOT, root))

    print(f"[rep] task {args.task}:  R={R}  coils={n_coils}  "
          f"sigma={args.sigma}  centre_frac={center_frac}")
    print(f"[rep] brain VAL from {root}")
    print(f"[rep] normalise={args.normalise}  noise_domain={args.noise_domain}"
          + (f"  crop={args.crop}" if args.crop else ""))
    if args.sigma < 1e-3:
        print(f"[rep] note: RAM floors sigma at 1e-3 in forward(), so the "
              f"conditioning map sees 1e-3, not {args.sigma}.")

    slices = load_val_slices(root, args.volumes, args.slices, device,
                             normalise=args.normalise, crop=args.crop)
    if not slices:
        raise SystemExit("[rep] no slices loaded.")
    print(f"[rep] {len(slices)} slices "
          f"({args.volumes} volumes x {args.slices})")

    import deepinv as dinv
    from ram import RAM
    from operators import FFT2D, Mask, Sense
    from operators.noise import mri_awgn

    model = RAM(device=str(device))

    gen = torch.Generator().manual_seed(args.seed)
    tot_p = tot_s = 0.0
    n = 0
    for i, x in enumerate(slices):
        H, W = x.shape[-2:]
        line = fastmri_random_mask(W, R, center_frac, generator=gen)
        mask = line.view(1, 1, 1, W).expand(1, 1, H, W).to(device)

        if n_coils == 1:
            smaps = torch.ones(1, 1, H, W, dtype=torch.complex64, device=device)
            physics = dinv.physics.MRI(mask=mask.squeeze(1), img_size=(H, W),
                                       device=device)
        else:
            smaps = birdcage_maps(n_coils, H, W, device)
            physics = dinv.physics.MultiCoilMRI(
                mask=mask.squeeze(1), coil_maps=smaps, img_size=(H, W),
                device=device)
        physics.noise_model = dinv.physics.GaussianNoise(args.sigma)

        x_dinv = torch.cat([x.real, x.imag], dim=1)          # (B, 2, H, W)
        if args.noise_domain == "measurement":
            # y = A(x) + n  -- the paper's / deepinv's convention
            y = physics(x_dinv)
        else:
            # this repo's: noise into s.x BEFORE the transform and the mask
            y_c, _, _ = mri_awgn(x, mask, smaps, args.sigma, "uniform")
            y = torch.stack([y_c.real, y_c.imag], dim=2)

        with torch.no_grad():
            out = model(y, physics=physics)
        est = (torch.complex(out[:, :1], out[:, 1:2])
               if (not torch.is_complex(out) and out.shape[1] == 2) else out)

        pp, ss = psnr_ssim(est, x)
        tot_p += pp
        tot_s += ss
        n += 1
        if i < 3 or (i + 1) % 25 == 0:
            print(f"[rep]   slice {i + 1:>4}/{len(slices)}  "
                  f"PSNR {pp:6.2f}  SSIM {ss:.4f}   (running "
                  f"{tot_p / n:6.2f} / {tot_s / n:.4f})")

    got_p, got_s = tot_p / n, tot_s / n
    print()
    print(f"{'':<14}{'PSNR':>9}{'SSIM':>9}")
    print(f"{'paper':<14}{spec['psnr']:>9.2f}{spec['ssim']:>9.3f}")
    print(f"{'this run':<14}{got_p:>9.2f}{got_s:>9.4f}")
    print(f"{'delta':<14}{got_p - spec['psnr']:>+9.2f}{got_s - spec['ssim']:>+9.4f}")
    print()
    if abs(got_p - spec["psnr"]) < 0.5:
        print("[rep] within 0.5 dB -- the harness reproduces the paper.")
    else:
        print("[rep] NOT reproduced. Before concluding anything about RAM, the")
        print("      free choices are, in the order I would test them:")
        print("        --normalise {max,std,none}   (unstated in the paper)")
        print("        --acs-frac ...               (unstated; default is")
        print("                                      fastMRI's 0.08/0.04)")
        print("        --coil-sim                   (only 'birdcage' so far)")
        print("        --crop 320                   (eval resolution unstated)")
        print("      and check the k-space layout deepinv expects, which is the")
        print("      one failure that looks like a bad model rather than a bug.")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"task": args.task, "args": vars(args),
                       "paper": spec, "psnr": got_p, "ssim": got_s,
                       "n_slices": n}, f, indent=2)
        print(f"[rep] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
