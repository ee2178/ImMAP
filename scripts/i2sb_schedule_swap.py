#!/usr/bin/env python3
"""
Is the bridge's power in its SAMPLER or in its NOISE SCHEDULE? Run a trained i2sb_recon regressor
as an ImMAP-style ascent (sb/ascent.py) under different sigma schedules and step rules -- no
retraining: the regressor, measurement, CG step and support are fixed, only the walk changes.

    python scripts/i2sb_schedule_swap.py --run-dir trained_nets/guided_recon/knee/<tag>/<cell>
    python scripts/i2sb_schedule_swap.py --run-dir ... --nfe 20 --schedules bridge geometric linear \\
        --zetas ddpm 0 0.5 1 --n-slices 20

Settings (every one on the SAME val slices, measurement and per-slice noise seed):
  i2sb               the bridge sampler itself (sb/i2sb_recon.py::i2sb_recon_sample) -- the
                     reference; "bridge / ddpm" must reproduce it (to float roundoff)
  <schedule>/<zeta>  the ascent: schedule "bridge" (uniform t, the I2SB grid), "geometric" or
                     "linear" in sigma_eff between the bridge grid's own endpoints (same NFE,
                     same one-shot first step at t = 1); zeta "ddpm" (= I2SB's step), 0
                     (deterministic), or any value in [0, 1] (ImMAP's zeta-rule)
The CG step (`--dc`, default the run's dc_iters / dc_lam) and the hard support (the run's
support_thresh) act on every prediction, as in the run's own sampler.

Scored like the dumps: magnitudes over the anatomy (where the stored reference is nonzero) by
default (--region frame for the whole image), the run's metric_range. Prints each schedule's
sigma_eff values and a table; writes <run-dir>/schedule_swap/results.json (or --out).
"""
import argparse
import json
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from datasets.registry import build_loader                         # noqa: E402
from sb.ascent import SCHEDULES, ascent_sample, schedule_steps, sigma_eff   # noqa: E402
from sb.i2sb_recon import bridge_noise_scale, i2sb_recon_sample    # noqa: E402
from training.forward_op import fixed_val_subset                    # noqa: E402
from training.i2sb_recon import BESTVAL_CKPT, _measure, measurement_support   # noqa: E402
from training.metrics import compute_metrics                        # noqa: E402


def _zeta(s):
    return "ddpm" if s == "ddpm" else float(s)


def _dc(s):
    it, lam = s.split(":")
    return int(it), float(lam)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ckpt", default=BESTVAL_CKPT)
    ap.add_argument("--nfe", type=int, default=None, help="default: the run's val_nfe")
    ap.add_argument("--schedules", nargs="+", default=list(SCHEDULES), choices=SCHEDULES)
    ap.add_argument("--zetas", type=_zeta, nargs="+", default=["ddpm", 0.0, 1.0],
                    help="step rules: 'ddpm' (I2SB's) or zeta in [0, 1] (0: deterministic)")
    ap.add_argument("--sigma-hi", type=float, default=None,
                    help="geometric / linear: first sigma after the one-shot step (default: the "
                         "bridge grid's)")
    ap.add_argument("--sigma-lo", type=float, default=None, help="... and the last one")
    ap.add_argument("--dc", type=_dc, default=None, metavar="ITERS:LAM",
                    help="the CG step on every prediction; default the run's dc_iters:dc_lam")
    ap.add_argument("--n-slices", type=int, default=20)
    ap.add_argument("--region", choices=("organ", "frame"), default="organ")
    ap.add_argument("--seed", type=int, default=None, help="default: the run's val_seed")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from scripts.eval_i2sb_recon_dc import load_run
    run = load_run(a.run_dir, a.ckpt)
    cfg, net, bridge, meas, i2, tr = (run[k] for k in ("cfg", "net", "bridge", "meas", "i2", "tr"))
    device = run["device"]
    nfe = a.nfe or run["nfe"]
    seed = run["seed"] if a.seed is None else int(a.seed)
    dc_it, dc_lam = a.dc or (int(i2.get("dc_iters", 0)), float(i2.get("dc_lam", 0.0) or 0.0))
    thresh = i2.get("support_thresh")
    tau_scale, tau_ref = i2.get("tau_scale", "none"), i2.get("tau_ref")
    loader = fixed_val_subset(build_loader(dict(cfg["data"]["val"], batch_size=1),
                                           shuffle=False, drop_last=False), a.n_slices, seed)

    sig = sigma_eff(bridge)
    sched_steps = {k: schedule_steps(bridge, k, nfe, a.sigma_hi, a.sigma_lo)
                   for k in a.schedules}
    print(f"{run['cfg']['experiment']['name']} ({a.ckpt}) | {len(loader.dataset)} val slices, "
          f"nfe {nfe}, tau {i2['tau']}, CG {dc_it} it lam {dc_lam:g}, support "
          f"{thresh if thresh is not None else 'none'}, scored over the {a.region}")
    for k, st in sched_steps.items():
        s = [float(sig[i]) for i in st[::-1]]
        print(f"  {k:9s} sigma_eff: " + " ".join(f"{v:.3g}" for v in s))

    names = ["i2sb"] + [f"{k}/{z}" for k in a.schedules for z in a.zetas]
    acc = {n: dict(psnr=0.0, ssim=0.0, nrmse=0.0, n=0) for n in names}
    gen = torch.Generator(device=device).manual_seed(seed)
    devs = [device] if device.type == "cuda" else []
    for bi, batch in enumerate(loader):
        x0, x1, y, E, sigma_n, organ, gt, guide = _measure(
            batch, net, device, kspace_type=run["kspace_type"], noise_std=run["noise_std"],
            generator=gen, guide_offset=0, **meas)
        kappa = bridge_noise_scale(x1, tau_scale, tau_ref)
        sup = None if thresh is None else measurement_support(y, E, meas, thresh)
        mask = (gt.abs() > 0) if a.region == "organ" else None
        for name in names:
            with torch.random.fork_rng(devices=devs):
                torch.manual_seed(seed * 100003 + bi)            # the same noise for every setting
                if name == "i2sb":
                    rec = i2sb_recon_sample(net, y, E, bridge, sigma_n=sigma_n, x1=x1, nfe=nfe,
                                            verbose=False, guide=guide, noise_scale=kappa,
                                            dc_iters=dc_it, dc_lam=dc_lam, support=sup,
                                            deterministic=bool(i2.get("deterministic", False)),
                                            posterior=i2.get("posterior", "ddpm"))[0]
                else:
                    k, z = name.split("/")
                    rec = ascent_sample(net, y, E, bridge, x1, sched_steps[k], zeta=_zeta(z),
                                        sigma_n=sigma_n, guide=guide, noise_scale=kappa,
                                        dc_iters=dc_it, dc_lam=dc_lam, support=sup)
            m = compute_metrics(gt.abs(), rec.abs(), data_range=tr.get("metric_range", 1.0),
                                mask=mask)
            for key in ("psnr", "ssim", "nrmse"):
                acc[name][key] += float(m[key])
            acc[name]["n"] += 1
        print(f"  slice {bi + 1}: " + "  ".join(
            f"{n} {acc[n]['psnr'] / acc[n]['n']:.2f}" for n in names[:4]) + " ...", flush=True)

    res = {n: {k: v / a_["n"] for k, v in a_.items() if k != "n"} for n, a_ in acc.items()}
    print(f"\n  {'setting':22s}{'PSNR':>8s}{'SSIM':>8s}{'NRMSE':>9s}   (vs i2sb PSNR)")
    for n, r in res.items():
        print(f"  {n:22s}{r['psnr']:8.2f}{r['ssim']:8.4f}{r['nrmse']:9.4f}   "
              f"{r['psnr'] - res['i2sb']['psnr']:+.2f}")
    out = a.out or os.path.join(a.run_dir, "schedule_swap")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(dict(run_dir=a.run_dir, ckpt=a.ckpt, nfe=nfe, dc=[dc_it, dc_lam],
                       support_thresh=thresh, region=a.region, n=len(loader.dataset),
                       steps={k: list(map(int, v)) for k, v in sched_steps.items()},
                       sigma={k: [float(sig[i]) for i in v] for k, v in sched_steps.items()},
                       results=res), f, indent=2)
    print(f"\nwrote {out}/results.json")


if __name__ == "__main__":
    main()
