#!/usr/bin/env python3
"""
How noisy are the knee scans themselves -- PD against PDFS -- in the units the networks see?

    IMMAP_CLUSTER=bigpurple python scripts/scanner_noise_stats.py --split val
    python scripts/scanner_noise_stats.py --split train --volumes 60 --out noise_train.json

READ-ONLY, CPU, no training. For each paired volume of the split (TGVN's PD/PDFS pairs,
config/knee/guided/splits/<split>.csv) and a range of its slices, both contrasts get:

  Sigma          the inter-coil noise covariance PER K-SPACE SAMPLE (the units the added noise
                 sigma is defined in, operators/noise.py::kspace_awgn), estimated three ways --
                   corners     the outer k-space band in BOTH axes, acquired columns only (the
                               PRIMARY estimate)
                   background  coil-image pixels in the air outside the dilated anatomy, rescaled
                               by W / W_acquired (zero-padded phase-encode columns)
                   wavelet     physics/nle.py::ncov_est on the coil images (CDF 9/7 high-pass)
                 Checked on synthetic scans with known white coil noise (smooth anatomy, PD 5x
                 brighter): corners exact in both contrasts, wavelet ~9% low in both, background
                 HIGH for the bright contrast -- truncation (Gibbs) ringing and, in real scans,
                 motion ghosts land in the air. Read agreement between corners and the other two
                 as the confidence; corners also read high for very sharp, bright edges.
  sigma_scan     the noise std of a unit-norm SENSE combination, sqrt(s^H Sigma s), median over
                 the anatomy (Otsu on RSS) -- s from the coil images, smoothed and RSS-normalised
  p95            the foreground 95th percentile of the RSS image
  SNR            p95 / sigma_scan

and sigma_scan in TRAINING UNITS -- what the network's noise level is calibrated against:

  guided   the per-volume normalisation of config/knee/guided/volume_scales.json
           (PDFS target / acs_p95, PD guide / full_p95, as datasets/fastmri/paired.py)
  mg grid  x 5000 (the one-shot grid's fixed scale_fac)

Compare guided-unit sigma_scan with the added training noise: vlow U[0, 0.001], the earlier "low"
U[0, 0.01] and the grid's U[0.01, 0.02]. If sigma_scan is far above 0.001, the networks' noise-
adaptive thresholds are being told sigma ~ 0 on noisier data.

Writes a per-slice / per-volume JSON (--out) and, with matplotlib, a histogram figure (--plot).
"""

import argparse
import json
import os
import statistics
import sys

import h5py
import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from operators.fourier import ifftc                     # noqa: E402


def read_pairs(csv_path):
    """[(pd, pdfs)] file names from a TGVN split CSV (header `pd,pdfs`), as
    datasets/fastmri/paired.py reads it -- without importing the datasets package."""
    import csv
    with open(csv_path, newline="") as f:
        return [(os.path.basename(r["pd"]), os.path.basename(r["pdfs"]))
                for r in csv.DictReader(f)]

KNEE_ROOTS = {                                          # as scripts/make_guided_recon_configs.py
    "torch": ["../datasets/fastmri/knee/multicoil_train", "../datasets/fastmri/knee/multicoil_val"],
    "bigpurple": [f"../datasets/fastmri_multiknee/{s}/multicoil_{s}" for s in ("train", "val", "test")],
}
MG_SCALE_FAC = 5000.0


def find(name, roots):
    for r in roots:
        p = os.path.join(r, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{name} in none of {roots}")


def cmm(A, B):
    """Complex matrix product through REAL matmuls (some torch builds crash on complex BLAS)."""
    Ar, Ai, Br, Bi = A.real, A.imag, B.real, B.imag
    return torch.complex(Ar @ Br - Ai @ Bi, Ar @ Bi + Ai @ Br)


def herm(Z):
    """Z Z^H for a (C, N) complex matrix."""
    return cmm(Z, Z.conj().T)


def corner_cov(k, frac):
    """Noise covariance from the outer k-space band (both axes), acquired columns only."""
    C, H, W = k.shape
    cols = (k.abs().sum(dim=(0, 1)) > 0).nonzero().flatten()
    lo, hi = int(cols.min()), int(cols.max()) + 1
    w_acq = hi - lo
    nh, nw = max(int(frac * H), 4), max(int(frac * w_acq), 2)
    rows = torch.cat([torch.arange(nh), torch.arange(H - nh, H)])
    cs = torch.cat([torch.arange(lo, lo + nw), torch.arange(hi - nw, hi)])
    Z = k[:, rows][:, :, cs].reshape(C, -1)
    Z = Z[:, (Z.abs().sum(0) > 0)]
    return herm(Z) / max(Z.shape[1], 1)


def wavelet_cov(x):
    """physics/nle.py::ncov_est (CDF 9/7 high-pass on each coil image, Z Z^H / N_nonzero),
    with the outer product done by `herm`."""
    from physics.nle import _default_filter, _separable_conv
    z = _separable_conv(x[None], _default_filter(x[None]))[0] / 2.0
    Z = z.reshape(z.shape[0], -1)
    n = int((x.abs().pow(2).sum(0) > 0).sum())
    return herm(Z) / max(n, 1)


def unit_maps(x, box=9):
    """Coil sensitivities from the coil images: box-smoothed, then normalised to unit RSS."""
    def smooth(t):
        return F.avg_pool2d(t[None], box, stride=1, padding=box // 2, count_include_pad=False)[0]
    sm = torch.complex(smooth(x.real), smooth(x.imag))
    return sm / sm.abs().pow(2).sum(0, keepdim=True).sqrt().clamp_min(1e-30)


def background_cov(x, fg, k, dilate=15, border=4):
    """Noise covariance from coil-image pixels outside the dilated anatomy, per k-space sample."""
    grow = F.max_pool2d(fg[None, None].double(), 2 * dilate + 1, stride=1,
                        padding=dilate)[0, 0] > 0
    bg = ~grow
    bg[:border], bg[-border:], bg[:, :border], bg[:, -border:] = False, False, False, False
    if int(bg.sum()) < 1000:
        return None
    Z = x[:, bg]
    W = k.shape[-1]
    w_acq = int((k.abs().sum(dim=(0, 1)) > 0).sum())               # zero-padded columns out
    return herm(Z) / Z.shape[1] * (W / max(w_acq, 1))


def slice_stats(k, frac):
    """(sigma_background, sigma_corners, sigma_wavelet, p95) of one (C, H, W) fully-sampled
    k-space slice; sigma_background is None when the slice has too little air."""
    from physics.object_mask import rss_object_mask
    k = torch.as_tensor(k).to(torch.complex128)
    x = ifftc(k)                                                  # coil images, ortho FFT
    rss = x.abs().pow(2).sum(0).sqrt()
    fg = rss_object_mask(rss.float()).reshape(rss.shape).bool()
    if int(fg.sum()) < 100:
        return None
    s = unit_maps(x)
    out = []
    C, H, W = s.shape
    sf = s.reshape(C, -1)
    for Sig in (background_cov(x, fg, k), corner_cov(k, frac), wavelet_cov(x)):
        if Sig is None:
            out.append(float("nan"))
            continue
        var = (sf.conj() * cmm(Sig.to(sf.dtype), sf)).sum(0).real.clamp_min(0).reshape(H, W)
        out.append(float(var[fg].median().sqrt()))
    return out[0], out[1], out[2], float(torch.quantile(rss[fg].float(), 0.95))


def pct(v, q):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.percentile(v, q)) if v.size else float("nan")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[1].strip())
    p.add_argument("--split", default="val", choices=("train", "val", "test"))
    p.add_argument("--splits-dir", default="config/knee/guided/splits")
    p.add_argument("--roots", nargs="+", default=None,
                   help="k-space roots (default: by IMMAP_CLUSTER, as the config generator)")
    p.add_argument("--volume-scales", default="config/knee/guided/volume_scales.json")
    p.add_argument("--volumes", type=int, default=None, help="first N pairs (default: all)")
    p.add_argument("--slice-frac", type=float, nargs=2, default=(0.25, 0.75))
    p.add_argument("--slice-step", type=int, default=2)
    p.add_argument("--corner-frac", type=float, default=0.1,
                   help="outer fraction of each k-space axis used as noise (default 0.1)")
    p.add_argument("--out", default=None, help="JSON with per-slice and per-volume values")
    p.add_argument("--plot", default=None, help="PNG of the training-unit noise histograms")
    args = p.parse_args()

    os.chdir(REPO)
    roots = args.roots or KNEE_ROOTS[os.environ.get("IMMAP_CLUSTER", "torch")]
    vols = json.load(open(args.volume_scales))["volumes"]
    pairs = read_pairs(os.path.join(args.splits_dir, f"{args.split}.csv"))[:args.volumes]
    stat = {"pdfs": "acs_p95", "pd": "full_p95"}                # paired.py's defaults
    per_vol = []
    for vi, (pd_name, pdfs_name) in enumerate(pairs):
        rec = {"pd": pd_name, "pdfs": pdfs_name}
        for contrast, name in (("pd", pd_name), ("pdfs", pdfs_name)):
            with h5py.File(find(name, roots), "r") as f:
                ks = f["kspace"]
                n = ks.shape[0]
                rng = range(int(args.slice_frac[0] * n), int(args.slice_frac[1] * n),
                            args.slice_step)
                rows = [r for r in (slice_stats(ks[j], args.corner_frac) for j in rng) if r]
            if not rows:
                continue
            med = lambda v: float(np.nanmedian(np.asarray(v, dtype=float)))  # noqa: E731
            sb, sc, sw, p95 = (med(v) for v in zip(*rows))
            fac = 1.0 / vols[name][stat[contrast]] if name in vols else float("nan")
            rec[contrast] = dict(sigma_corners=sc, sigma_background=sb, sigma_wavelet=sw,
                                 p95=p95, snr=p95 / sc, snr_background=p95 / sb,
                                 snr_wavelet=p95 / sw, guided_sigma=sc * fac,
                                 guided_sigma_background=sb * fac, guided_sigma_wavelet=sw * fac,
                                 mg_sigma=sc * MG_SCALE_FAC, n_slices=len(rows))
        per_vol.append(rec)
        if "pd" in rec and "pdfs" in rec:
            print(f"[{vi + 1}/{len(pairs)}] {pdfs_name}: SNR PD {rec['pd']['snr']:.1f} "
                  f"PDFS {rec['pdfs']['snr']:.1f};  guided-unit sigma PD "
                  f"{rec['pd']['guided_sigma']:.4f} PDFS {rec['pdfs']['guided_sigma']:.4f}",
                  flush=True)

    def col(c, k):
        return [r[c][k] for r in per_vol if c in r]

    print(f"\n{len(per_vol)} volume pairs ({args.split}); per-volume medians over slices; "
          f"columns p5 / p50 / p95 across volumes")
    for k, label in (("snr", "SNR (corners)"), ("snr_background", "SNR, background"),
                     ("snr_wavelet", "SNR, wavelet"),
                     ("guided_sigma", "sigma_scan, guided units (corners)"),
                     ("guided_sigma_background", "sigma_scan, guided units (background)"),
                     ("guided_sigma_wavelet", "sigma_scan, guided units (wavelet)"),
                     ("mg_sigma", "sigma_scan, mg-grid units x5000 (corners)")):
        print(f"  {label:<38}" + "".join(
            f"   {c.upper():>4} {pct(col(c, k), 5):8.4f} {pct(col(c, k), 50):8.4f} "
            f"{pct(col(c, k), 95):8.4f}" for c in ("pd", "pdfs")))
    ratio = [r["pd"]["snr"] / r["pdfs"]["snr"] for r in per_vol
             if "pd" in r and "pdfs" in r and np.isfinite(r["pd"]["snr"] / r["pdfs"]["snr"])]
    print(f"  SNR ratio PD / PDFS per pair (corners): p5 {pct(ratio, 5):.2f}  median "
          f"{pct(ratio, 50):.2f}  p95 {pct(ratio, 95):.2f}")
    print("  reference training noise (guided units): vlow U[0, 0.001], low U[0, 0.01]; "
          "mg grid U[0.01, 0.02] in its own units")
    if args.out:
        json.dump({"args": vars(args), "volumes": per_vol}, open(args.out, "w"), indent=1)
        print(f"wrote {args.out}")
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for c, colr in (("pd", "C0"), ("pdfs", "C3")):
            ax[0].hist(col(c, "guided_sigma"), bins=20, alpha=0.6, color=colr, label=c.upper())
            ax[1].hist(col(c, "snr"), bins=20, alpha=0.6, color=colr, label=c.upper())
        for v, ls in ((0.001, "-"), (0.01, "--")):
            ax[0].axvline(v, color="k", ls=ls, lw=0.8)
        ax[0].set(title="scanner noise, guided training units (lines: 0.001, 0.01)",
                  xlabel="sigma_scan")
        ax[1].set(title="SNR = foreground p95 / sigma_scan", xlabel="SNR")
        for a in ax:
            a.legend()
        fig.tight_layout()
        fig.savefig(args.plot, dpi=130)
        print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
