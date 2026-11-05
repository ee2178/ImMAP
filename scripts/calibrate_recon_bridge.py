#!/usr/bin/env python3
"""
Size an i2sb_recon bridge from the data: tau = RMS(x0 - x1) / 2 inside the object.

The rule of scripts/calibrate_nyumets_bridge.py (centre the denoising ladder on the gap: the
Brownian bridge's std_fwd[-1] = 2 tau equals RMS(x0 - x1)), applied to a RECON config: the pair is
built exactly as training/i2sb_recon.py::_measure builds it -- that config's mask, protocol,
online maps, noise and x0_source -- on `--slices` slices of the chosen split, and the RMS is taken
inside the RSS object mask of the ground truth (the background would dilute it).

Prints the pooled value, the per-slice spread and a top-10% tail, and the tau each implies.

INTENSITY-RELATIVE BRIDGE (cfg["i2sb"]["tau_scale"] = "x1_rms"). Each slice's tau follows the RMS
of its zero-filled image over its Otsu foreground (sb/i2sb_recon.py::slice_scale),
tau_i = tau * s_i / tau_ref. The script always reports the median s over these slices -- that is
`tau_ref`, which makes `tau` the absolute tau of a typical slice -- and the gap rule redone in
those units: RMS of (x0 - x1) / kappa_i, kappa_i = s_i / tau_ref. Compare the per-slice spread of
the two gaps to see how much of it was intensity.

Nothing is written; put the values in the generator (e.g. scripts/make_guided_recon_configs.py
TAU and TAU_REF), which writes tau to BOTH cfg["i2sb"] and model.params.

    python scripts/calibrate_recon_bridge.py config/knee/guided/sbtgvn_R8_g1.json --slices 60
"""

import argparse
import os
import sys
import types

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datasets                                    # noqa: E402,F401  (registers the loaders)
from datasets.registry import build_loader        # noqa: E402
from physics.object_mask import rss_object_mask   # noqa: E402
from sb.i2sb_recon import slice_scale             # noqa: E402
from training.i2sb_recon import _measure          # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config")
    p.add_argument("--split", default="train", choices=("train", "val"))
    p.add_argument("--slices", type=int, default=60)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    mri, i2sb, tr = cfg["mri"], cfg.get("i2sb", {}), cfg["training"]
    torch.manual_seed(args.seed)
    loader = build_loader(dict(cfg["data"][args.split], num_workers=2), shuffle=True,
                          drop_last=False)
    # Stand-in for the regressor: _measure only asks whether it reads the operator's maps, its
    # grid stride, and whether it takes a guide (the guide does not enter x0 or x1).
    net = types.SimpleNamespace(uses_operator_smaps=True, pad_stride=1, guided=False)
    meas = {k: mri[k] for k in ("R", "acs_lines", "mask_dist", "mask_offset", "center_frac",
                                "adjust_accel", "online_smaps", "online_smaps_kws") if k in mri}
    meas.setdefault("online_smaps", None)
    meas.setdefault("online_smaps_kws", None)
    noise = tr.get("val_noise_std") or tr.get("noise_std")

    sq, n, per, diffs = 0.0, 0, [], []
    scales, rel = [], []            # slice_scale(x1) per slice, and its gap pixels
    with torch.no_grad():
        for batch in loader:
            x0, x1, *_, gt, _ = _measure(
                batch, net, args.device, kspace_type=mri["kspace_type"], noise_std=noise,
                noise_dist=tr.get("noise_dist", "uniform"),
                x1_source=i2sb.get("x1_source", "adjoint"),
                x0_source=i2sb.get("x0_source", "operator"), **meas)
            s_b = slice_scale(x1)
            for b in range(x0.shape[0]):
                m = rss_object_mask(gt[b].abs()).expand_as(x0[b])
                d = (x0[b] - x1[b]).abs()[m]
                sq += float(d.pow(2).sum())
                n += d.numel()
                per.append(float(d.pow(2).mean().sqrt()))
                diffs.append(d.cpu())
                scales.append(float(s_b[b]))
                rel.append(d.cpu())
            if len(per) >= args.slices:
                break

    vs = (sq / max(n, 1)) ** 0.5
    d = torch.cat(diffs)
    if d.numel() > 4_000_000:
        d = d[torch.randperm(d.numel())[:4_000_000]]
    q = d.quantile(0.9)
    tail = float(d[d >= q].pow(2).mean().sqrt())
    per = torch.tensor(per)
    print(f"{args.config}  ({args.split}, {len(per)} slices, R={mri['R']}, "
          f"{mri['kspace_type']}, x0_source={i2sb.get('x0_source', 'operator')})")
    print(f"  RMS(x0 - x1) in the object: {vs:.4f}  -> tau = {vs / 2:.4f}")
    print(f"  per slice: median {float(per.median()):.4f}, "
          f"range [{float(per.min()):.4f}, {float(per.max()):.4f}]")
    print(f"  top-10% tail RMS: {tail:.4f}  -> tau = {tail / 2:.4f}  (ceiling)")

    # intensity-relative: kappa_i = s_i / s_ref, s_ref = the median slice scale
    s = torch.tensor(scales)
    s_ref = float(s.median())
    kap = s / s_ref
    rel_sq = sum(float(d.pow(2).sum()) / float(k) ** 2 for d, k in zip(rel, kap))
    rel_per = torch.tensor([p / float(k) for p, k in zip(per.tolist(), kap)])
    vs_rel = (rel_sq / max(n, 1)) ** 0.5
    cv = lambda v: float(v.std() / v.mean())                         # noqa: E731
    print(f"  slice scale s = RMS|x1| on its Otsu foreground: median {s_ref:.4f} (-> tau_ref), "
          f"range [{float(s.min()):.4f}, {float(s.max()):.4f}]")
    print(f"  gap in tau_ref units, RMS((x0 - x1) / kappa): {vs_rel:.4f}  -> tau = "
          f"{vs_rel / 2:.4f}  (tau_scale='x1_rms')")
    print(f"  per-slice gap spread (std/mean): absolute {cv(per):.2f}, relative "
          f"{cv(rel_per):.2f}")
    print(f"  config's current tau: {i2sb.get('tau')}, tau_scale: "
          f"{i2sb.get('tau_scale', 'none')}, tau_ref: {i2sb.get('tau_ref')}")


if __name__ == "__main__":
    main()
