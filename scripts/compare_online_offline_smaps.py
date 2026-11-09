#!/usr/bin/env python3
"""
Are the online coil maps behind the horizontal streaks? Online ESPIRiT vs the offline big-ACS maps,
on sample knee slices.

    python scripts/compare_online_offline_smaps.py                       # val, 6 volumes
    python scripts/compare_online_offline_smaps.py --n-volumes 10 --variant k6:kernel_size=6 \\
        --variant walsh:method=walsh --variant clamp64:clamp=64

THE QUESTION. Every guided recon shows subtle horizontal streaks, LGGS included. The operator
the nets see uses ONLINE maps (ESPIRiT from the measurement's own ACS: ~15 phase-encode lines x a
readout band clamped to 32 rows, kernel 4, no support threshold), while the target is the SENSE
image under the OFFLINE maps (ACS 32 x 32 from the fully-sampled scan, kernel 8, thresh_eig 0.9).
If the online maps are wrong in a streak-shaped way, no network can fix it: the operator itself
cannot represent the target.

PER SLICE (the config's data and protocol: per-volume scaling, the training mask at offset 0,
measurement noise at val_noise_std, online-map settings from cfg.mri.online_smaps_kws):

    x_off     the TARGET: the stored offline SENSE image
    x_on      the best image under the online operator: S_on^H F^-1 k_FULL. With unit-RSS maps
              this is the least-squares solution of F S_on x = k_full, so |x_on| vs |x_off| is the
              error FLOOR the online operator puts under every reconstruction
    rss       the coil RSS of the full k-space (map-free reference)
    res_on/off  per-pixel model residual ||c - s (s^H c)|| / rss: how much of the coil data the
              maps cannot explain, at each pixel

    floor_psnr   PSNR(|x_on|, |x_off|) in the foreground, against the slice's own peak
    streak_h     STRIPE INDEX of d = |x_on| - |x_off| in the foreground: the variance of its ROW
                 means over the variance of d. ~1/W for unstructured error, -> 1 when d is
                 constant along rows, i.e. HORIZONTAL stripes. streak_v: the same for columns.
    off_vs_rss   the same stripe indices for |x_off| - rss: are the streaks already in the TARGET?

Every --variant (NAME:key=val,key=val -- online_smaps arguments on top of the config's) adds the
same numbers for another online estimate, e.g. a bigger kernel, Walsh, or a wider readout clamp.

Output under --out: summary.json, slices.csv, and per-slice panels (the first --panels slices):
target | online floor | diff x5 | residual maps | one coil's map, offline vs online | row/column
mean profiles of the diff. Needs a GPU for ESPIRiT on the knee grid.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from datasets.fastmri.paired import PairedKneeDataset                 # noqa: E402
from operators.fourier import ifftc                                   # noqa: E402
from operators.noise import kspace_awgn                               # noqa: E402
from physics.mask import get_mask_cached, resolve_acs_lines           # noqa: E402
from physics.object_mask import otsu_threshold                        # noqa: E402
from physics.online_smaps import online_smaps                         # noqa: E402


def _parse_variant(s):
    name, _, rest = s.partition(":")
    kws = {}
    for kv in filter(None, rest.split(",")):
        k, _, v = kv.partition("=")
        for cast in (int, float):
            try:
                v = cast(v)
                break
            except ValueError:
                pass
        kws[k.strip()] = v
    return name, kws


def stripe_index(d, fg):
    """(horizontal, vertical): variance of the row (column) means of d over var(d), foreground."""
    d = torch.where(fg, d, torch.zeros_like(d))
    n_row, n_col = fg.sum(1).clamp_min(1), fg.sum(0).clamp_min(1)
    rows = d.sum(1) / n_row                                          # (H,) mean along each row
    cols = d.sum(0) / n_col
    v = d[fg].var().clamp_min(1e-30)
    rr, cc = rows[fg.any(1)], cols[fg.any(0)]
    return float(rr.var() / v) if rr.numel() > 1 else 0.0, float(cc.var() / v) if cc.numel() > 1 else 0.0


def combine(smaps, coils):
    return (smaps.conj() * coils).sum(0)


def residual(smaps, coils, rss):
    x = combine(smaps, coils)
    r = (coils - smaps * x).abs().pow(2).sum(0).sqrt()
    return r / rss.clamp_min(1e-12 * float(rss.max()))


def setup(config, split="val", n_volumes=6, slices_per_volume=3, variants=(), noise=None,
          seed=0):
    """-> (ds, items, mri, variants, sigma): the config's paired dataset (per slice, the stored
    offline SENSE target and maps), the picked item indices (`slices_per_volume` evenly over each
    of `n_volumes` random volumes), its mri block, [(name, online_smaps kwargs)] -- the config's
    online settings first, as "online", then each variant ("NAME:key=val,..." or (name, dict))
    on top of them -- and the measurement noise std."""
    cfg = json.load(open(config))
    d, mri, tr = dict(cfg["data"][split]), cfg["mri"], cfg["training"]
    if d.get("target") != "sense":
        raise ValueError("the config's target must be 'sense' (offline maps) to compare against")
    keep = ("split_csv", "kspace_roots", "start_slice", "end_slice", "target", "smap_root",
            "scale_fac", "side_scale_fac", "volume_scales", "volume_stat", "side_volume_stat",
            "volume_target")
    ds = PairedKneeDataset(**{k: d[k] for k in keep if k in d}, enumerate_slices=True)
    rng = np.random.default_rng(seed)
    vols = sorted(rng.choice(len(ds.pairs), min(n_volumes, len(ds.pairs)), replace=False))
    items = []
    for v in vols:
        idx = [i for i, (pi, _) in enumerate(ds.index) if pi == v]
        pick = np.linspace(0, len(idx) - 1, min(slices_per_volume, len(idx))).round()
        items += [idx[int(j)] for j in sorted(set(pick))]
    base = dict(mri.get("online_smaps_kws") or {})
    method = mri.get("online_smaps") or "espirit"
    parsed = [_parse_variant(v) if isinstance(v, str) else v for v in variants]
    out = [("online", dict(base, method=method))] + [
        (n, {**base, "method": method, **kw}) for n, kw in parsed]
    sigma = float(tr.get("val_noise_std", 0.015) if noise is None else noise)
    return ds, items, mri, out, sigma


def compare_slice(ds, idx, mri, variants, sigma, gen, dev):
    """One slice -> (row of metrics, panel tensors for the FIRST variant, on CPU)."""
    k, sm_off, img_off, _, _, _ = ds[idx]
    name, sl = ds.item_id(idx)
    k, sm_off, img_off = (t[None].to(dev) for t in (k, sm_off, img_off))
    W = k.shape[-1]
    mask = get_mask_cached(img_off, R=mri["R"], acs_lines=mri.get("acs_lines"),
                           mode=mri.get("mask_dist", "uniform"), offset=0,
                           center_frac=mri.get("center_frac"),
                           adjust_accel=mri.get("adjust_accel", False))
    y, _ = kspace_awgn(k, mask, sigma, "uniform", generator=gen)
    acs = resolve_acs_lines(W, mri.get("acs_lines"), mri.get("center_frac"))

    coils = ifftc(k)[0]
    rss = coils.abs().pow(2).sum(0).sqrt()
    a = rss[None, None]
    fg = (a > otsu_threshold(a))[0, 0]
    x_off = img_off[0, 0]
    peak = float(x_off.abs()[fg].max())
    h0, v0 = stripe_index(x_off.abs() - rss, fg)
    row = dict(volume=name, slice=sl, acs_lines=acs, off_vs_rss_streak_h=h0,
               off_vs_rss_streak_v=v0,
               res_off=float(residual(sm_off[0], coils, rss)[fg].mean()))
    panel = None
    for vname, kws in variants:
        with torch.no_grad():
            sm_on = online_smaps(y, mask, acs_lines=acs, **kws)[0]
        x_on = combine(sm_on, coils)
        diff = x_on.abs() - x_off.abs()
        mse = float(diff[fg].pow(2).mean())
        h, v = stripe_index(diff, fg)
        row.update({f"{vname}_floor_psnr": 10 * np.log10(peak ** 2 / max(mse, 1e-30)),
                    f"{vname}_floor_nrmse": float(diff[fg].norm() / x_off.abs()[fg].norm()),
                    f"{vname}_streak_h": h, f"{vname}_streak_v": v,
                    f"{vname}_res": float(residual(sm_on, coils, rss)[fg].mean())})
        if panel is None:
            panel = tuple(t.detach().cpu() for t in (
                x_off, x_on, diff, residual(sm_off[0], coils, rss), residual(sm_on, coils, rss),
                sm_off[0], sm_on, fg))
    return row, panel


def summarize(rows):
    keys = [k for k in rows[0] if k not in ("volume", "slice", "acs_lines")]
    return {k: dict(mean=float(np.mean([r[k] for r in rows])),
                    median=float(np.median([r[k] for r in rows])),
                    min=float(np.min([r[k] for r in rows])),
                    max=float(np.max([r[k] for r in rows]))) for k in keys}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/knee/guided/tgvn_c8_R8_g1.json")
    ap.add_argument("--split", default="val", choices=("train", "val"))
    ap.add_argument("--n-volumes", type=int, default=6)
    ap.add_argument("--slices-per-volume", type=int, default=3)
    ap.add_argument("--variant", action="append", default=[],
                    help="NAME:key=val,... -- extra online_smaps settings on top of the config's")
    ap.add_argument("--noise", type=float, default=None, help="default: cfg val_noise_std")
    ap.add_argument("--panels", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/smaps_online_vs_offline")
    args = ap.parse_args()
    dev = torch.device(args.device)

    ds, items, mri, variants, sigma = setup(args.config, args.split, args.n_volumes,
                                            args.slices_per_volume, args.variant, args.noise,
                                            args.seed)
    print(f"{len(items)} slices; noise {sigma:g}; variants: "
          + "; ".join(f"{n} {kw}" for n, kw in variants))
    os.makedirs(args.out, exist_ok=True)
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    rows, n_panel = [], 0
    for idx in items:
        row, panel = compare_slice(ds, idx, mri, variants, sigma, gen, dev)
        rows.append(row)
        print(f"  {row['volume']} sl {row['slice']}: floor PSNR {row['online_floor_psnr']:.2f} "
              f"dB, stripe h/v {row['online_streak_h']:.3f}/{row['online_streak_v']:.3f} "
              f"(target vs RSS {row['off_vs_rss_streak_h']:.3f}/"
              f"{row['off_vs_rss_streak_v']:.3f}), residual on/off "
              f"{row['online_res']:.4f}/{row['res_off']:.4f}", flush=True)
        if n_panel < args.panels:
            _panel(os.path.join(args.out, f"{row['volume']}_sl{row['slice']}.png"),
                   f"{row['volume']} slice {row['slice']}", *panel)
            n_panel += 1

    summary = summarize(rows)
    print(f"\n{'quantity':<28}{'mean':>10}{'median':>10}{'min':>10}{'max':>10}")
    for k, st in summary.items():
        print(f"{k:<28}{st['mean']:10.4f}{st['median']:10.4f}{st['min']:10.4f}{st['max']:10.4f}")
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(dict(config=args.config, split=args.split, noise=sigma,
                       variants={n: kw for n, kw in variants}, summary=summary), f, indent=2)
    with open(os.path.join(args.out, "slices.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out}/summary.json, slices.csv and {n_panel} panels")
    print("reading it: streak_h well above streak_v (and above the target's own off_vs_rss "
          "value) means the online maps put horizontal stripes under every recon")


def _panel(path, title, x_off, x_on, diff, r_off, r_on, sm_off, sm_on, fg):
    """The per-slice figure; saved to `path`, or returned (for a notebook) when path is None."""
    import matplotlib
    if path is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    vmax = float(x_off.abs()[fg].quantile(0.99))
    c = int(sm_off.abs().pow(2).sum(dim=(-2, -1)).argmax())          # the strongest coil
    fgf = fg.float()
    fig = plt.figure(figsize=(20, 9))
    gs = fig.add_gridspec(2, 6)
    tiles = [(x_off.abs(), "target |x_off|", "gray", 0, vmax),
             (x_on.abs(), "online floor |x_on|", "gray", 0, vmax),
             (diff * fgf, "|x_on| - |x_off| (x5)", "RdBu_r", -0.2 * vmax, 0.2 * vmax),
             (r_off * fgf, "residual, offline maps (fg)", "magma", 0, 0.3),
             (r_on * fgf, "residual, online maps (fg)", "magma", 0, 0.3),
             (sm_off[c].abs(), f"|s_{c}| offline", "viridis", 0, 1),
             (sm_on[c].abs(), f"|s_{c}| online", "viridis", 0, 1),
             (torch.angle(sm_off[c] * sm_on[c].conj()) * (sm_off[c].abs() > 0),
              f"phase(s_off conj s_on), coil {c}", "twilight", -np.pi, np.pi)]
    for i, (img, t, cm, lo, hi) in enumerate(tiles):
        ax = fig.add_subplot(gs[i // 5, i % 5] if i < 5 else gs[1, i - 5])
        ax.imshow(img, cmap=cm, vmin=lo, vmax=hi)
        ax.set_title(t, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    dm = diff * fgf
    ax = fig.add_subplot(gs[0, 5])
    ax.plot((dm.sum(1) / fgf.sum(1).clamp_min(1)).numpy(), lw=0.8)
    ax.set_title("row means of the diff (horizontal stripes)", fontsize=9)
    ax = fig.add_subplot(gs[1, 3:6])
    ax.plot((dm.sum(0) / fgf.sum(0).clamp_min(1)).numpy(), lw=0.8, color="C1")
    ax.set_title("column means of the diff (vertical stripes)", fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    if path is None:
        return fig
    fig.savefig(path, dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    main()
