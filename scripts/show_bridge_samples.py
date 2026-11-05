#!/usr/bin/env python3
"""
Look at the bridge states a recon I2SB regressor is trained on, for one or more tau.

For `--slices` val slices of an i2sb_recon config, the pair (x0, x1) is built exactly as training
builds it (training/i2sb_recon.py::_measure: that config's mask, protocol, online maps, noise and
x0_source), and the Brownian bridge state

    x_t = (1 - t) x0 + t x1 + std_sb(t) eps,     std_sb(t) = 2 tau sqrt(t (1 - t))

is drawn at a few t. The noise peaks at t = 1/2, where std_sb = tau: that column is the NOISIEST
state the regressor ever sees. One figure row per (slice, tau); every tau row of a slice uses the
SAME eps, so rows differ only by the noise scale. The last column is the injected noise alone,
|std_sb(1/2) eps|, on the same greyscale as the images.

Columns: x0 (target) | x_t at each --ts | x1 (zero-filled) | noise at t=1/2. Magnitudes, one
window per slice: [0, 99.5th percentile of |x0|]. The printed table gives, per tau, the peak noise
against the measurement noise sigma_n and against the image's RMS inside the object.

    python scripts/show_bridge_samples.py config/knee/guided/sbtgvn_R8_g1.json \\
        --taus 0.0234 0.06 --out figs/bridge_samples_knee_R8.png
"""

import argparse
import os
import sys
import types

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datasets                                    # noqa: E402,F401  (registers the loaders)
from datasets.registry import build_loader        # noqa: E402
from physics.object_mask import rss_object_mask   # noqa: E402
from sb.base import bridge_coeffs, build_schedule  # noqa: E402
from training.i2sb_recon import _measure          # noqa: E402


def bridge_rows(x0, x1, taus, ts, n_points=1000, kind="brownian", beta_max=0.3, seed=0):
    """{tau: [(t, x_t, std_sb(t))]} for one complex (1, H, W) pair, one shared eps."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    eps = torch.complex(torch.randn(x0.shape, generator=g), torch.randn(x0.shape, generator=g))
    eps = (eps / 2 ** 0.5).to(x0.device)           # E|eps|^2 = 1, as torch.randn_like(complex)
    out = {}
    for tau in taus:
        sched = build_schedule(kind=kind, tau=tau, n_points=n_points, beta_max=beta_max)
        mu0, mu1, std_sb = bridge_coeffs(sched)
        row = []
        for t in ts:
            k = int(round(t * (n_points - 1)))
            s = float(std_sb[k])
            row.append((t, float(mu0[k]) * x0 + float(mu1[k]) * x1 + s * eps, s))
        out[tau] = (row, float(std_sb.max()) * eps)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config")
    p.add_argument("--taus", nargs="*", type=float, default=None,
                   help="taus to show (default: the config's)")
    p.add_argument("--ts", nargs="*", type=float, default=[0.25, 0.5, 0.75])
    p.add_argument("--slices", type=int, default=2)
    p.add_argument("--split", default="val", choices=("train", "val"))
    p.add_argument("--out", default=None, help="PNG path (default figs/bridge_samples_<name>.png)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    mri, i2sb, tr = cfg["mri"], cfg.get("i2sb", {}), cfg["training"]
    taus = args.taus or [float(i2sb.get("tau", 0.1))]
    torch.manual_seed(args.seed)
    loader = build_loader(dict(cfg["data"][args.split], num_workers=0), shuffle=False,
                          drop_last=False)
    net = types.SimpleNamespace(uses_operator_smaps=True, pad_stride=1, guided=False)
    meas = {k: mri[k] for k in ("R", "acs_lines", "mask_dist", "mask_offset", "center_frac",
                                "adjust_accel", "online_smaps", "online_smaps_kws") if k in mri}
    meas.setdefault("online_smaps", None)
    meas.setdefault("online_smaps_kws", None)
    sig = tr.get("val_noise_std") or tr.get("noise_std")
    sig_val = float(np.mean(sig)) if isinstance(sig, (list, tuple)) else float(sig)

    cases = []
    with torch.no_grad():
        for batch in loader:
            x0, x1, *_, gt, _ = _measure(
                batch, net, args.device, kspace_type=mri["kspace_type"], noise_std=sig,
                noise_dist=tr.get("noise_dist", "uniform"),
                x1_source=i2sb.get("x1_source", "adjoint"),
                x0_source=i2sb.get("x0_source", "operator"), **meas)
            x0, x1, gt = x0[0].cpu(), x1[0].cpu(), gt[0].cpu()
            m = rss_object_mask(gt.abs()).expand_as(x0)
            cases.append((x0, x1, m, bridge_rows(x0, x1, taus, args.ts, seed=args.seed,
                                                  n_points=int(i2sb.get("n_points", 1000)),
                                                  kind=i2sb.get("kind", "brownian"),
                                                  beta_max=float(i2sb.get("beta_max", 0.3)))))
            if len(cases) >= args.slices:
                break

    print(f"{args.config}: sigma_n (val) = {sig_val:.4f}")
    print(f"{'slice':>5} {'tau':>8} {'peak std_sb':>11} {'/sigma_n':>9} {'/RMS(x0)':>9} "
          f"{'RMS(x0-x1)':>10}")
    for i, (x0, x1, m, rows) in enumerate(cases):
        rms0 = float(x0.abs()[m].pow(2).mean().sqrt())
        gap = float((x0 - x1).abs()[m].pow(2).mean().sqrt())
        for tau in taus:
            print(f"{i:>5} {tau:>8.4f} {tau:>11.4f} {tau / sig_val:>9.2f} {tau / rms0:>9.3f} "
                  f"{gap:>10.4f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncol = len(args.ts) + 3
    nrow = len(cases) * len(taus)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.4 * ncol, 2.6 * nrow), squeeze=False)
    r = 0
    for i, (x0, x1, m, rows) in enumerate(cases):
        vmax = float(torch.quantile(x0.abs().flatten(), 0.995))
        for tau in taus:
            states, noise = rows[tau]
            tiles = ([("x0 (target)", x0)]
                     + [(f"t={t:g}  std_sb={s:.4f}", xt) for t, xt, s in states]
                     + [("x1 (zero-filled)", x1), (f"noise at t=1/2 ({tau:g})", noise)])
            for c, (title, img) in enumerate(tiles):
                ax = axes[r, c]
                ax.imshow(img.abs()[0].numpy(), cmap="gray", vmin=0, vmax=vmax)
                ax.set_title(title, fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
            axes[r, 0].set_ylabel(f"slice {i}, tau={tau:g}", fontsize=9)
            r += 1
    fig.suptitle(f"{os.path.basename(args.config)}: bridge states (|.|), shared eps per slice",
                 fontsize=10)
    fig.tight_layout()
    out = args.out or os.path.join(
        "figs", f"bridge_samples_{os.path.splitext(os.path.basename(args.config))[0]}.png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
