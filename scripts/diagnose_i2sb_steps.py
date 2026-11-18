#!/usr/bin/env python3
"""
What has an i2sb recon regressor learned, step by step? (2026-11-18: sbunet_inv's single-pass
validation at t = 999 looked like its input.)

    python scripts/diagnose_i2sb_steps.py trained_nets/guided_recon/knee/<set>/<cell>
    python scripts/diagnose_i2sb_steps.py <run dir> --ckpt net.ckpt --slices 16

READ-ONLY. Loads the run's config.json and checkpoint (net_bestval.ckpt by default; the EMA
when the checkpoint carries one, as validation scores it), draws a few validation slices exactly
as validation does (same measurement, same x1 start), and for a ladder of bridge steps reports:

    rel_err(x0_hat)   ||x0_hat - x0||^2 / ||x0||^2          what the net predicts
    rel_err(x_t)      the same for x0_hat = x_t               the IDENTITY prediction
    gain              rel_err(x_t) / rel_err(x0_hat)          > 1: the net beats identity
    |eps_hat|/|eps|   for an eps net, the size of its predicted noise against the true label
                      (x_t - x0) / std_fwd -- ~0 means the output head is still (near) zero

plus the norm of the UNet's final conv (zero at init), and, at the top step, the single-pass
estimate followed by the run's CG data-consistency step (dc_iters, dc_lam) -- what the
measurement adds to a regressor that never sees y (I2SB-Inversion's).

Reading it: a net whose gain is ~1 at EVERY step and whose final conv is ~0 has not left its
initialisation (a training problem: steps, LR, collapse). A net that beats identity at low
steps but not at the top has learned denoising but not the x1 -> x0 map; for a guide start
without x1-conditioning or y that map is a pure contrast synthesis, the hardest part.
"""

import argparse
import json
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from datasets.registry import build_loader                                  # noqa: E402
from models import build_model                                              # noqa: E402
from sb.base import build_schedule, forward_sample, forward_std, n_steps    # noqa: E402
from sb.i2sb_recon import cg_data_consistency, predict_x0_recon             # noqa: E402
from training.i2sb_recon import _measure, load_sampling_weights            # noqa: E402

STEPS = (0, 50, 100, 250, 500, 750, 900, 990, 999)


def rel(a, b):
    return float((a - b).abs().pow(2).sum() / b.abs().pow(2).sum().clamp_min(1e-30))


def final_conv_norm(net):
    """Norm of the last Conv2d's weight -- SBUnet's output layer is zero-initialised."""
    last = None
    for m in net.modules():
        if isinstance(m, torch.nn.Conv2d):
            last = m
    return float(last.weight.norm()) if last is not None else float("nan")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[1].strip())
    p.add_argument("run_dir")
    p.add_argument("--ckpt", default="net_bestval.ckpt")
    p.add_argument("--slices", type=int, default=8)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    cfg = json.load(open(os.path.join(args.run_dir, "config.json")))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = build_model(cfg).to(dev).eval()
    ck = load_sampling_weights(net, os.path.join(args.run_dir, args.ckpt), dev)
    i2, mri, tr = cfg["i2sb"], cfg["mri"], cfg["training"]
    sched = build_schedule(kind=i2["kind"], tau=i2["tau"], n_points=i2["n_points"],
                           beta_max=i2.get("beta_max", 0.3), device=dev)
    N = n_steps(sched)
    meas = dict(R=mri["R"], acs_lines=mri.get("acs_lines"), mask_dist=mri.get("mask_dist", "uniform"),
                mask_offset=mri.get("mask_offset", 0), center_frac=mri.get("center_frac"),
                adjust_accel=mri.get("adjust_accel", False), noise_dist=tr.get("noise_dist", "uniform"),
                online_smaps=mri.get("online_smaps"), online_smaps_kws=mri.get("online_smaps_kws"),
                x1_source=i2.get("x1_source", "adjoint"), x0_source=i2.get("x0_source", "operator"),
                guide_R=mri.get("guide_R", 1), guide_acs_lines=mri.get("guide_acs_lines", 0),
                guide_mask_dist=mri.get("guide_mask_dist", "uniform"),
                guide_maps=i2.get("guide_maps", "operator"))
    loader = build_loader(dict(cfg["data"]["val"], num_workers=0), shuffle=False, drop_last=False)
    eps_net = getattr(net, "predict", "x0") == "eps"
    print(f"{cfg['paths']['save_dir']}  ckpt {args.ckpt} (step {ck.get('step', '?')})  "
          f"predict={getattr(net, 'predict', 'x0')}  final conv |W| = {final_conv_norm(net):.3e}")

    gen = torch.Generator(device=dev).manual_seed(args.seed)
    torch.manual_seed(args.seed)
    acc = {s: [0.0, 0.0, 0.0, 0.0] for s in STEPS}
    dc = [0.0, 0.0, 0.0]
    n = 0
    with torch.no_grad():
        for batch in loader:
            if n >= args.slices:
                break
            x0, x1, y, E, sigma_n, _, gt, guide = _measure(
                batch, net, dev, kspace_type=mri.get("kspace_type", "measurement_awgn"),
                noise_std=tr.get("val_noise_std", 0.0), generator=gen, guide_offset=0, **meas)
            b = x0.shape[0]
            for s in STEPS:
                st = torch.full((b,), min(s, N - 1), dtype=torch.long, device=dev)
                xt = forward_sample(sched, st, x0, x1)
                sd = forward_std(sched, st, xdim=x0.shape[1:])
                x0h = predict_x0_recon(net, xt, sd, x1=x1, y=y, E=E, sigma_n=sigma_n,
                                       guide=guide)
                a = acc[s]
                a[0] += rel(x0h, x0)
                a[1] += rel(xt, x0)
                if eps_net:
                    a[2] += float(((xt - x0h) / sd).abs().pow(2).mean().sqrt())
                    a[3] += float(((xt - x0) / sd).abs().pow(2).mean().sqrt())
                if s == STEPS[-1]:
                    x0dc = cg_data_consistency(x0h, y, E, float(i2.get("dc_lam", 7e-5)),
                                               int(i2.get("dc_iters", 7)) or 7)
                    dc[0] += rel(x0h.abs(), gt.abs())
                    dc[1] += rel(x0dc.abs(), gt.abs())
                    dc[2] += rel(E.adjoint(y).abs(), gt.abs())
            n += b
    n = max(n, 1)
    print(f"\n{n} val slices; relative squared error against the bridge target x0")
    print(f"{'step':>6}{'rel_err(x0_hat)':>17}{'rel_err(x_t)':>14}{'gain':>8}"
          + (f"{'|eps_hat|':>11}{'|eps|':>9}" if eps_net else ""))
    for s in STEPS:
        e, i, eh, et = (v / n for v in acc[s])
        print(f"{min(s, N - 1):>6}{e:>17.4f}{i:>14.4f}{i / max(e, 1e-12):>8.2f}"
              + (f"{eh:>11.3f}{et:>9.3f}" if eps_net else ""))
    print(f"\ntop step, magnitudes against the stored image: single pass {dc[0] / n:.4f}, "
          f"+ CG ({i2.get('dc_iters', 7)} it, lam {i2.get('dc_lam', 7e-5):g}) {dc[1] / n:.4f}, "
          f"zero-filled E^H y {dc[2] / n:.4f}")


if __name__ == "__main__":
    main()
