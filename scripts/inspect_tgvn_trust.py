#!/usr/bin/env python3
"""
What does a trained TGVN's trust-guidance term actually contribute? And does H(w) look like the
TARGET contrast, or like the guide?

    python scripts/inspect_tgvn_trust.py trained_nets/guided_recon/knee/s4_noise-vlow/tgvn_c6_R8_g1
    python scripts/inspect_tgvn_trust.py RUN --n 20 --figs 4 --out figures/out/tgvn_trust

Each TGVN cascade (models/tgvn/adapter.py::_GuardedTGVNBlock) updates its iterate x as

    x <- x - soft_dc - soft_asc - Phi(x)
         soft_dc  = |w_dc| S^H M (S x - y)                         data consistency
         soft_asc = |w_asc| P'_delta (x - H(w)),  P'_delta = (I + S^H M S / delta^2)^{-1}
         Phi(x)   = the regulariser U-Net
         H(w)     = a NormUnet on the guide coil-combined with the operator's maps

so H(w) is what the cascade pulls x TOWARDS in the space the measurement does not pin down
(P'_delta ~ 1 there, ~ delta^2 / (1 + delta^2) on the measured k-space). Nothing in training
asks H(w) to be a PDFS image: it is only whatever makes the output good.

This runs the run's net (net_bestval.ckpt) on its fixed val subset (scripts/eval_guided_recon.py,
as validation), with every cascade TRACED -- the block's own computation, recorded term by term
and checked against the untraced block (max relative difference printed). Per cascade, averaged
over slices:

  corr(H, target)   Pearson correlation of |H(w)| with |target| inside the target's support
  corr(H, guide)    ... with |guide| (the guide coil-combined with the operator's maps)
  fit(H, target)    relative L2 error of the best scale a |H(w)| against |target|
  fit(H, guide)     ... against |guide|
  |soft_asc|, |soft_dc|, |Phi|   each term's norm relative to the incoming |x|
  asc on measured   share of soft_asc's coil k-space energy on the MEASURED lines (P'_delta
                    should keep it small: the guide acts where the data do not)
  and the learned |w_asc|, |w_dc| per cascade and delta (shared).

corr(guide, target) is printed as the baseline: H(w) "looks like the target" only if it beats it.
`--figs N` saves, for the first N slices, one PNG each: the guide, target, recon, and per cascade
x_in, |H(w)|, |soft_asc|, |Phi(x)|, |soft_dc| (the knee flipped upright, as the viewer does).
"""
import argparse
import json
import os
import statistics
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from models.tgvn._upstream import inner, sens_expand, sens_reduce        # noqa: E402
from models.tgvn.adapter import _TINY, _GuardedTGVNBlock                 # noqa: E402


def _c(t):
    """(B, 1, H, W, 2) real view -> (B, H, W) complex."""
    return torch.view_as_complex(t.contiguous())[:, 0]


def traced_block_forward(block, rec, current_image, ref_kspace, mask, second_kspace, sens_maps,
                         delta, sens_maps_second=None, num_iter=10):
    """_GuardedTGVNBlock.forward, line for line, recording every term into `rec`."""
    current_kspace = sens_expand(current_image, sens_maps)
    zero = torch.zeros(1, 1, 1, 1, 1).to(current_kspace)
    soft_dc = torch.where(mask, current_kspace - ref_kspace, zero)
    soft_dc = soft_dc * torch.abs(block.dc_weight)
    soft_dc = sens_reduce(soft_dc, sens_maps)
    model_term = block.model(current_image)
    maps2 = sens_maps if sens_maps_second is None else sens_maps_second
    guide_img = sens_reduce(second_kspace, maps2)
    Hw = block.asc_model(guide_img)
    b = current_image - Hw

    mask_delta = 1 + mask / (delta ** 2).clamp_min(1e-12)
    mask_delta_inv = 1 / mask_delta
    x = sens_reduce(mask_delta_inv * sens_expand(b, sens_maps), sens_maps)
    r = b - sens_reduce(mask_delta * sens_expand(x, sens_maps), sens_maps)
    p = r.clone()
    rs_old = inner(r, r)
    for _ in range(num_iter):
        Ap = sens_reduce(mask_delta * sens_expand(p, sens_maps), sens_maps)
        alpha = rs_old / inner(p, Ap).clamp_min(_TINY)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = inner(r, r)
        p = r + rs_new / rs_old.clamp_min(_TINY) * p
        rs_old = rs_new
    soft_asc = torch.abs(block.asc_weight) * x
    out = current_image - soft_dc - soft_asc - model_term

    # the share of soft_asc's coil k-space energy on the measured lines
    k_asc = sens_expand(soft_asc, sens_maps)
    e = k_asc.pow(2).sum(-1)                                       # (B, C, H, W)
    m = mask[..., 0].expand_as(e)
    rec.append(dict(x_in=_c(current_image), guide=_c(guide_img), H=_c(Hw), soft_asc=_c(soft_asc),
                    soft_dc=_c(soft_dc), model=_c(model_term), x_out=_c(out),
                    asc_measured=float(e[m].sum() / e.sum().clamp_min(1e-30)),
                    w_asc=float(block.asc_weight.abs()), w_dc=float(block.dc_weight.abs()),
                    delta=float(delta.reshape(-1)[0])))
    return out


class TraceTGVN:
    """Context manager: every _GuardedTGVNBlock of `net` records its terms; each traced output
    is checked against the block's own forward (`max_rel_diff`)."""

    def __init__(self, net):
        self.blocks = [m for m in net.modules() if isinstance(m, _GuardedTGVNBlock)]
        if not self.blocks:
            raise ValueError("no TGVN cascades in this net (a guided TGVN is needed)")
        self.records, self.max_rel_diff = [], 0.0

    def __enter__(self):
        for blk in self.blocks:
            def fwd(*args, _blk=blk, **kw):
                out = traced_block_forward(_blk, self.records, *args, **kw)
                ref = _GuardedTGVNBlock.forward(_blk, *args, **kw)
                d = float((out - ref).abs().max() / ref.abs().max().clamp_min(1e-30))
                self.max_rel_diff = max(self.max_rel_diff, d)
                return ref
            blk.forward = fwd
        return self

    def __exit__(self, *exc):
        for blk in self.blocks:
            if "forward" in blk.__dict__:
                del blk.forward


def _corr(a, b, m):
    a, b = a[m], b[m]
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-30))


def _fit(a, b, m):
    """Relative L2 error of the best scale s*a against b, over m."""
    a, b = a[m], b[m]
    s = (a * b).sum() / (a * a).sum().clamp_min(1e-30)
    return float((s * a - b).norm() / b.norm().clamp_min(1e-30))


def cascade_stats(records, target, support):
    """Per cascade: the similarity and size numbers of the module docstring (one slice)."""
    t = target.abs()
    out = []
    for r in records:
        H, g, x = r["H"].abs(), r["guide"].abs(), r["x_in"]
        nx = x.abs().norm().clamp_min(1e-30)
        out.append({
            "corr_H_target": _corr(H, t, support), "corr_H_guide": _corr(H, g, support),
            "corr_guide_target": _corr(g, t, support),
            "fit_H_target": _fit(H, t, support), "fit_H_guide": _fit(H, g, support),
            "rel_soft_asc": float(r["soft_asc"].abs().norm() / nx),
            "rel_soft_dc": float(r["soft_dc"].abs().norm() / nx),
            "rel_model": float(r["model"].abs().norm() / nx),
            "asc_measured": r["asc_measured"], "w_asc": r["w_asc"], "w_dc": r["w_dc"],
            "delta": r["delta"],
        })
    return out


def save_figure(path, records, target, recon, flip=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f = (lambda a: a[::-1]) if flip else (lambda a: a)

    def show(ax, img, title, vmax=None):
        a = img.detach().abs().cpu().numpy()
        vmax = vmax if vmax is not None else max(float(torch.quantile(
            img.detach().abs().flatten().cpu().float()[::7], 0.995)), 1e-12)
        ax.imshow(f(a), cmap="gray", vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=7)
        ax.axis("off")

    K = len(records)
    cols = ("x_in", "H", "soft_asc", "model", "soft_dc")
    labels = ("x in", "|H(w)|", "|soft_asc|", "|Phi(x)|", "|soft_dc|")
    fig, axes = plt.subplots(K + 1, 5, figsize=(10, 2.1 * (K + 1)))
    t = target.abs()
    vmax = float(torch.quantile(t.flatten().cpu().float()[::7], 0.995))
    show(axes[0, 0], records[0]["guide"][0], "guide (S^H F^-1 w)")
    show(axes[0, 1], target, "target", vmax)
    show(axes[0, 2], recon, "recon", vmax)
    axes[0, 3].axis("off")
    axes[0, 4].axis("off")
    for k, r in enumerate(records):
        for j, (c, lab) in enumerate(zip(cols, labels)):
            show(axes[k + 1, j], r[c][0], f"cascade {k}: {lab}",
                 vmax if c == "x_in" else None)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="a trained guided TGVN run dir (config.json + net_bestval.ckpt)")
    ap.add_argument("--n", type=int, default=20, help="val slices")
    ap.add_argument("--figs", type=int, default=4, help="PNGs for the first N slices")
    ap.add_argument("--out", default=None, help="figure dir (default <run>/trust_inspect)")
    ap.add_argument("--ckpt", default="net_bestval.ckpt")
    ap.add_argument("--device", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    from scripts.eval_guided_recon import load_recon_run, measure, reconstruct, val_subset
    run = load_recon_run(a.run, ckpt=a.ckpt, device=a.device)
    ds = val_subset(run["cfg"])
    n = min(a.n, len(ds))
    out_dir = a.out or os.path.join(a.run, "trust_inspect")
    os.makedirs(out_dir, exist_ok=True)

    per, worst = [], 0.0
    with torch.no_grad():
        for i in range(n):
            m = measure(run, ds[i], seed=i)
            with TraceTGVN(run["net"]) as tr:
                recon = reconstruct(run, m)
            worst = max(worst, tr.max_rel_diff)
            gt = m["gt"][:, 0]
            support = (gt.abs() > 0) if (gt.abs() > 0).any() else torch.ones_like(gt.real, dtype=torch.bool)
            per.append(cascade_stats(tr.records, gt, support))
            if i < a.figs:
                p = os.path.join(out_dir, f"slice{i:02d}.png")
                save_figure(p, tr.records, gt[0], recon[0, 0] if recon.dim() == 4 else recon[0])
                print(f"  wrote {p}")

    K = len(per[0])
    keys = list(per[0][0])
    mean = [{k: statistics.mean(s[c][k] for s in per) for k in keys} for c in range(K)]
    print(f"\n== {run['name']}  ({n} val slices; traced vs untraced cascades: max rel diff "
          f"{worst:.1e})")
    print(f"  delta = {mean[0]['delta']:.4g};  corr(guide, target) = "
          f"{mean[0]['corr_guide_target']:.3f} (the bar H(w) must beat to 'look like PDFS')")
    print(f"  {'casc':>4s}{'corr H,tgt':>11s}{'corr H,gd':>10s}{'fit H,tgt':>10s}{'fit H,gd':>9s}"
          f"{'|asc|/|x|':>10s}{'|dc|/|x|':>9s}{'|Phi|/|x|':>10s}{'asc meas':>9s}"
          f"{'w_asc':>7s}{'w_dc':>7s}")
    for c, s in enumerate(mean):
        print(f"  {c:4d}{s['corr_H_target']:11.3f}{s['corr_H_guide']:10.3f}"
              f"{s['fit_H_target']:10.3f}{s['fit_H_guide']:9.3f}{s['rel_soft_asc']:10.3f}"
              f"{s['rel_soft_dc']:9.3f}{s['rel_model']:10.3f}{s['asc_measured']:9.3f}"
              f"{s['w_asc']:7.3f}{s['w_dc']:7.3f}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"run": a.run, "n": n, "max_rel_diff": worst, "per_cascade": mean,
                       "per_slice": per}, f, indent=2)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
