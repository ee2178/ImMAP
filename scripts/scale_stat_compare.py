#!/usr/bin/env python3
"""
Which per-slice statistic should `scale_norm` divide by? Compare candidates against the
true tissue intensity, on the measurement a net actually sees.

    python scripts/scale_stat_compare.py \
        --configs trained_nets/mg_recon/knee/lpdsnetcf32_R8/config.json \
                  trained_nets/mg_recon/knee/lpdsnetcf32_R12/config.json \
        --n-volumes 40 --stride 4 --out results/scale_stats/knee

`scale_norm` (models/mg_lpds.py) makes LPDS / MGLPDS exactly scale-equivariant whatever
the statistic s; what the statistic decides is whether a learned threshold MEANS THE SAME
on every slice. So the right s is the one that tracks the tissue's intensity, and nothing
else (mask, coil count, matrix size, noise, object coverage of the FOV).

Candidates, per slice, from y and the embedded operator E exactly as the recon adapter
builds them (evaluation/tasks.py, the run's online maps, the run's mask, AWGN at the run's
val sigma):
  ram       mean(|Re y|, |Im y|) over every k-space entry, zeros included: RAM's `num`,
            what scale_norm uses now (models/cg_init.py::ram_signal_level)
  ksp_rms   sqrt(sum |y|^2 / number of SAMPLED entries): k-space energy, mask-corrected
  mu        |sum(E^H y) / sum(E^H E 1)|: the least-squares constant image, the DC that
            preprocessing/kspace.py already removes
  mu_mag    sum|E^H y| / sum(E^H E 1): mu without the phase, so it cannot cancel
  adj_p99   99th percentile of |E^H y| / mean(E^H E 1): how BRIGHT the tissue is rather than
            how much of it there is (the FOV means above scale with object coverage)
  acs_p99   the same on the fully sampled centre lines only: no aliasing, lower resolution
Truth, from the reference and the stored-map organ mask:
  tissue    mean |reference| inside the organ mask
  tissue99  99th percentile of |reference| inside it (the bright-edge scale)

Scored by the spread of log(candidate / truth) across slices (std, and the 10-90
percentile range as a factor): a candidate that tracks the truth exactly has spread 0,
whatever constant separates them. Also reported: |mu| / mu_mag (near 1 means the image
phase does not cancel; small values flag slices where |mu| would blow up the scaling),
and, with two configs at different R, each candidate's ratio between them on the same
slices (1 = independent of the mask).

Outputs under --out: stats.csv (one row per config x slice), summary.txt, stats.png.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datasets                                                  # noqa: F401,E402
from datasets.registry import build_loader                      # noqa: E402
from evaluation.tasks import build_adapter, default_sigma, force_organ_mask  # noqa: E402
from models.cg_init import ram_signal_level                      # noqa: E402

CANDIDATES = ("ram", "ksp_rms", "mu", "mu_mag", "adj_p99", "acs_p99")
TRUTHS = ("tissue", "tissue99")


class _Probe(nn.Module):
    """Stands in for a net in the recon adapter: stride 1, uses the operator's maps."""
    pad_stride = 1
    uses_operator_smaps = True

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))


def acs_mask(mask):
    """The fully sampled centre block of a Cartesian mask (lines along the last axis):
    the run of sampled columns through the centre."""
    m = (mask.abs() > 0)
    cols = m.reshape(-1, m.shape[-1]).any(0)
    c = cols.shape[0] // 2
    if not cols[c]:
        return None
    lo, hi = c, c
    while lo > 0 and cols[lo - 1]:
        lo -= 1
    while hi < cols.shape[0] - 1 and cols[hi + 1]:
        hi += 1
    out = torch.zeros_like(mask)
    out[..., lo:hi + 1] = mask[..., lo:hi + 1]
    return out


def p99_adjusted(y, E, q=0.99):
    """q-quantile of |E^H y| over the image, divided by mean(E^H E 1): a bright-tissue
    level that the sampled fraction does not change and object coverage barely does."""
    B = y.shape[0]
    xa = E.adjoint(y).abs().reshape(B, -1).float()
    g = E.adjoint(E(torch.ones_like(E.adjoint(y)))).real.reshape(B, -1).mean(1)
    return torch.quantile(xa, q, dim=1) / g


@torch.no_grad()
def statistics(y, E):
    B = y.shape[0]
    xa = E.adjoint(y)
    den = E.adjoint(E(torch.ones_like(xa))).real.reshape(B, -1).sum(1)
    sampled = (y.abs() > 0).reshape(B, -1).sum(1).clamp_min(1)
    return {
        "ram": ram_signal_level(y),
        "ksp_rms": (y.abs().pow(2).reshape(B, -1).sum(1) / sampled).sqrt(),
        "mu": (xa.reshape(B, -1).sum(1) / den).abs(),
        "mu_mag": xa.abs().reshape(B, -1).sum(1) / den,
        "adj_p99": p99_adjusted(y, E),
        "acs_p99": _acs_p99(y, E),
    }


def _acs_p99(y, E):
    """`p99_adjusted` on the centre block only: the unaliased low-resolution image."""
    from operators.accessors import get_mask, set_mask
    m = acs_mask(get_mask(E))
    if m is None:
        return torch.full((y.shape[0],), float("nan"), device=y.device)
    E_acs = set_mask(E, m)
    return p99_adjusted(y * (m.abs() > 0) if m.shape[-2:] == y.shape[-2:] else y, E_acs)


def at_accel(cfg, R):
    """`cfg` with its mask moved to acceleration R. Only the mask matters to these
    statistics, so no net trained at R is needed. A cf32-style config (center_frac set,
    adjust_accel off) keeps center_frac * R fixed, as make_mg_recon_configs.py does."""
    mri = dict(cfg["mri"])
    R0 = mri.get("R")
    if mri.get("center_frac") and not mri.get("adjust_accel", False) and R0:
        mri["center_frac"] = float(mri["center_frac"]) * float(R0) / float(R)
    mri["R"] = R
    return dict(cfg, mri=mri)


def run_config(path, a, device, R=None):
    with open(path) as f:
        cfg = json.load(f)
    if R is not None:
        cfg = at_accel(cfg, R)
    force_organ_mask(cfg)
    cfg.setdefault("training", {})["val_kspace_type"] = a.val_kspace_type
    sigma = a.sigma if a.sigma is not None else default_sigma(cfg)
    val = dict(cfg["data"]["val"])
    val.update(enumerate_slices=True, start_slice=0, end_slice=None,
               batch_size=1, num_workers=a.workers)
    if a.volumes:
        val["volumes"] = a.volumes
    base = build_loader(val, shuffle=False, drop_last=False)
    ds = base.dataset

    vols = sorted({ds.item_id(i)[0] for i in range(len(ds))})[:a.n_volumes or None]
    keep = set(vols)
    idx = [i for i in range(len(ds)) if ds.item_id(i)[0] in keep]
    idx = idx[::a.stride]
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx), batch_size=1,
                                         shuffle=False, num_workers=a.workers,
                                         collate_fn=base.collate_fn)
    adapter = build_adapter("recon")
    probe = _Probe().to(device)
    gen = torch.Generator(device=device).manual_seed(a.seed)
    R = cfg.get("mri", {}).get("R")
    rows, got = [], {}

    def call(net, y, E, s):
        got.update(statistics(y, E))
        return E.adjoint(y)

    for k, batch in enumerate(loader):
        vol, sl = ds.item_id(idx[k])
        gt, _, _ = adapter(probe, batch, cfg, device, sigma, gen, call=call)
        organ = batch[3].to(device).abs() > 0
        mag = gt.abs()[organ.expand_as(gt)]
        if mag.numel() < 100:
            continue
        name = os.path.basename(os.path.dirname(os.path.abspath(path)))
        r = dict(config=f"{name}@R{R}" if a.accels else name, R=R,
                 volume=vol, slice=int(sl),
                 tissue=float(mag.mean()), tissue99=float(torch.quantile(mag.float(), 0.99)))
        r.update({c: float(got[c].reshape(-1)[0]) for c in CANDIDATES})
        rows.append(r)
        print(f"\r  {r['config']}: {k + 1}/{len(idx)} slices", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    return rows, sigma


def spread(x):
    x = np.log(np.asarray(x, float))
    x = x[np.isfinite(x)]
    return float(np.std(x)), float(np.exp(np.percentile(x, 90) - np.percentile(x, 10)))


def summarize(rows):
    L = []
    for cfgname in sorted({r["config"] for r in rows}):
        rr = [r for r in rows if r["config"] == cfgname]
        L.append(f"{cfgname}  (R={rr[0]['R']}, {len(rr)} slices, "
                 f"{len({r['volume'] for r in rr})} volumes)")
        L.append(f"  spread of log(candidate / truth): std, and the 10-90% range as a factor")
        L.append(f"  {'candidate':<10}" + "".join(f"{'vs ' + t:>24}" for t in TRUTHS))
        for c in CANDIDATES:
            cells = []
            for t in TRUTHS:
                sd, f = spread([r[c] / r[t] for r in rr])
                cells.append(f"{sd:>12.3f}  x{f:>8.2f}")
            L.append(f"  {c:<10}" + "".join(f"{x:>24}" for x in cells))
        sd, f = spread([r["tissue"] for r in rr])
        L.append(f"  {'(none)':<10}{sd:>12.3f}  x{f:>8.2f}   <- the tissue intensity's own spread: "
                 f"what a constant s leaves")
        ph = np.array([r["mu"] / r["mu_mag"] for r in rr])
        L.append(f"  |mu| / mu_mag: median {np.median(ph):.3f}, 5th percentile "
                 f"{np.percentile(ph, 5):.3f}, min {ph.min():.3f}  (1 = no phase cancellation)")
        L.append("")
    cfgs = sorted({r["config"] for r in rows})
    if len(cfgs) == 2:
        A = {(r["volume"], r["slice"]): r for r in rows if r["config"] == cfgs[0]}
        B = {(r["volume"], r["slice"]): r for r in rows if r["config"] == cfgs[1]}
        both = sorted(set(A) & set(B))
        if both:
            L.append(f"mask dependence: {cfgs[0]} / {cfgs[1]} on the same {len(both)} slices "
                     f"(median ratio; 1 = independent of R)")
            for c in CANDIDATES + TRUTHS:
                ratio = np.array([A[k][c] / B[k][c] for k in both])
                L.append(f"  {c:<10} median {np.median(ratio):.3f}   10-90% "
                         f"{np.percentile(ratio, 10):.3f} - {np.percentile(ratio, 90):.3f}")
    return "\n".join(L)


def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cfgs = sorted({r["config"] for r in rows})
    fig, ax = plt.subplots(len(cfgs), len(CANDIDATES), figsize=(3.2 * len(CANDIDATES), 3.0 * len(cfgs)),
                           squeeze=False)
    for i, cfgname in enumerate(cfgs):
        rr = [r for r in rows if r["config"] == cfgname]
        t = np.array([r["tissue"] for r in rr])
        for j, c in enumerate(CANDIDATES):
            v = np.array([r[c] for r in rr])
            A = ax[i, j]
            A.loglog(t, v, "o", ms=3, color="#2f6db5", alpha=0.5, mec="none")
            k = np.exp(np.median(np.log(v / t)))
            lo, hi = t.min(), t.max()
            A.loglog([lo, hi], [k * lo, k * hi], color="#c4452f", lw=1.2)
            sd, f = spread(v / t)
            A.set_title(f"{cfgname}\n{c}: log-ratio std {sd:.2f}", fontsize=8)
            A.set_xlabel("tissue intensity (mean |ref| in organ)", fontsize=7)
            A.tick_params(labelsize=6)
            A.spines[["top", "right"]].set_visible(False)
            A.grid(alpha=0.25, which="both")
    fig.suptitle("A good scale statistic lies on the red line (proportional to tissue intensity)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--accels", type=int, nargs="*",
                   help="run each config at these R instead of its own (the mask is all the "
                        "statistics need); two values also report the mask dependence")
    p.add_argument("--configs", nargs="+", required=True,
                   help="run config.json files (their val data, mask, maps and sigma); two at "
                        "different R also report the mask dependence")
    p.add_argument("--n-volumes", type=int, default=40, help="first N val volumes (0: all)")
    p.add_argument("--volumes", nargs="*", help="explicit volume names instead")
    p.add_argument("--stride", type=int, default=4, help="every Nth slice")
    p.add_argument("--val-kspace-type", choices=["simulated", "measurement_awgn"],
                   default="measurement_awgn")
    p.add_argument("--sigma", type=float, help="default: each config's val_noise_std")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(a.out, exist_ok=True)

    rows = []
    missing = [c for c in a.configs if not os.path.isfile(c)]
    if missing:
        raise SystemExit(f"no such config: {' '.join(missing)}  (use --accels to run one "
                         f"config at several R)")
    for path in a.configs:
        for R in (a.accels or [None]):
            r, sigma = run_config(path, a, device, R)
            print(f"  {path}{'' if R is None else f' at R={R}'}: {len(r)} slices at sigma {sigma:g}")
            rows += r
    if not rows:
        raise SystemExit("no slices")
    with open(os.path.join(a.out, "stats.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    txt = summarize(rows)
    print("\n" + txt)
    with open(os.path.join(a.out, "summary.txt"), "w") as f:
        f.write(txt + "\n")
    plot(rows, os.path.join(a.out, "stats.png"))
    print(f"wrote stats.csv, summary.txt, stats.png to {a.out}")


if __name__ == "__main__":
    main()
