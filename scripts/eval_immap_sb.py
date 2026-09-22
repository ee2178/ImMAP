"""
Evaluate ImMAP-SB (sb/immap_sb.py: I2SB + learned data-consistency prox) against plain I2SB.

    python -m scripts.eval_immap_sb \
        --run-config trained_nets/nyumets/I2SB_Unet_NYUMets_CT1_from_all/config.json \
        --a-ckpt trained_nets/nyumets/forward_ladder_unet_cond_w8_CT1_to_T1/E_unet_w8_l3_x.pt \
        --c 0 0.25 1 4 --n-slices 200 --out figs/immap_sb

For every c in --c (c = 0 IS the plain sampler), the SAME val slices are sampled with the SAME
noise draws, so the differences between rows are the prox and nothing else. Reported, pooled over
pixels inside the brain mask:

    psnr / ssim            full reverse sampling vs CT1 (training.metrics, masked, the run's data_range)
    rmse, rmse_enh, rest   CT1 error overall / on the enhancement proxy (top ENH_Q of CT1 - T1) / rest
    t1_res                 RMS of M(T1 - A(recon)): consistency with the measurement. The prox should
                           pull this DOWN toward sigma_A -- not below it, which would be fitting A's error
    delta                  mean RMS change the prox made to x_hat, per active step
    seconds                wall clock for the whole subset at that c

A panel (one slice, one row per c: sample and sample - CT1) and results.json go to --out.
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import yaml

import datasets                                    # noqa: F401  (registers loaders)
from datasets.registry import build_loader
from models import build_model
from sb.base import build_schedule
from sb.immap_sb import ImMAPProx, immap_sb
from sb.learned_dc import _load_E
from training.common import load_ckpt
from training.forward_op import enh_region, fixed_val_subset
from training.i2sb import _split_batch
from training.metrics import compute_metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-config", required=True, help="a trained i2sb run's saved config.json")
    ap.add_argument("--a-ckpt", required=True, help="frozen A: ladder E_*.pt or forward_op net.ckpt")
    ap.add_argument("--sigma-a", type=float, default=None, help="default: the val rmse A stores")
    ap.add_argument("--a-cond-idx", type=int, nargs="*", default=[3, 0],
                    help="A's side-information contrasts, if A is conditioned")
    ap.add_argument("--c", type=float, nargs="+", default=[0.0, 0.25, 1.0, 4.0])
    ap.add_argument("--t-max", type=float, default=1.0)
    ap.add_argument("--cg-iters", type=int, default=10)
    ap.add_argument("--cg-tol", type=float, default=1e-4)
    ap.add_argument("--gn-iters", type=int, default=1)
    ap.add_argument("--mask", choices=["brain", "none"], default="brain",
                    help="fidelity region M for the prox")
    ap.add_argument("--nfe", type=int, default=None, help="default: the run's val_nfe")
    ap.add_argument("--n-slices", type=int, default=200)
    ap.add_argument("--slice-range", type=int, nargs=2, default=[40, 110], metavar=("LO", "HI"),
                    help="original slice indices [LO, HI) to evaluate on; overrides the run's "
                         "config (older runs have none). Pass -1 -1 to keep the run's own setting")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--enh-q", type=float, default=0.98)
    ap.add_argument("--out", default="figs/immap_sb")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.run_config) as f:
        cfg = yaml.safe_load(f)
    i2, tr = cfg["i2sb"], cfg["training"]
    data_range = float(tr.get("data_range", 1.0))

    net = build_model(cfg).to(device).eval()
    ckpt = cfg["paths"].get("ckpt") or os.path.join(cfg["paths"]["save_dir"], "net.ckpt")
    load_ckpt(ckpt, model=net, device=device)
    sched = build_schedule(kind=i2.get("kind", "brownian"), tau=i2.get("tau", 0.19),
                           n_points=i2.get("n_points", 1000), beta_max=i2.get("beta_max", 0.3),
                           device=device)
    if hasattr(net, "assert_schedule_matches"):
        net.assert_schedule_matches(sched)
    nfe = args.nfe or int(i2.get("val_nfe", 20))

    A, a_rmse = _load_E(args.a_ckpt, device)
    sigma_A = args.sigma_a if args.sigma_a is not None else a_rmse
    if sigma_A is None:
        raise ValueError("A's checkpoint stores no val rmse; pass --sigma-a")

    vcfg = dict(cfg["data"]["val"])
    run_cond = list(vcfg.get("cond_idx") or [])
    a_cond = list(args.a_cond_idx) if A.cond_channels else []
    all_cond = run_cond + [c for c in a_cond if c not in run_cond]
    net_sel, a_sel = list(range(len(run_cond))), [all_cond.index(c) for c in a_cond]
    vcfg.update(cond_idx=all_cond, batch_size=args.batch)
    if args.slice_range != [-1, -1]:
        vcfg["slice_range"] = list(args.slice_range)
    loader = fixed_val_subset(build_loader(vcfg, shuffle=False, drop_last=False),
                              args.n_slices, args.seed)

    print(f"denoiser {type(net).__name__} ({ckpt}) | A {args.a_ckpt} sigma_A={sigma_A:.4g}")
    print(f"{len(loader.dataset)} val slices, nfe={nfe}, c={args.c}, t_max={args.t_max}, "
          f"cg_iters={args.cg_iters}, gn_iters={args.gn_iters}, mask={args.mask}")

    cs = list(dict.fromkeys(args.c))
    acc = {c: dict(psnr=0., ssim=0., n=0, sse=0., npx=0., sse_e=0., npx_e=0., sse_r=0.,
                   npx_r=0., t1_sse=0., deltas=[], sec=0.) for c in cs}
    panel = {}
    for bi, batch in enumerate(loader):
        x0, x1, cond, mask, _, _ = _split_batch(batch, device)
        # the MEASUREMENT: this session's T1. The loader returns it as "y" when the bridge starts
        # elsewhere (x1_source="other_study"); for the T1 bridge it IS x1.
        y = batch["y"].to(device) if isinstance(batch, dict) and "y" in batch else x1
        c_net = None if (cond is None or not net_sel) else cond[:, net_sel]
        c_A = None if not a_sel else cond[:, a_sel]
        m = (mask > 0.5).float()
        enh = enh_region(x0, y, m, args.enh_q)
        rest = m * (1 - enh)
        M = m if args.mask == "brain" else None
        for c in cs:
            prox = ImMAPProx(sched, A, sigma_A, c=c, t_max=args.t_max, cg_iters=args.cg_iters,
                          gn_iters=args.gn_iters, cg_tol=args.cg_tol)
            torch.manual_seed(args.seed * 100003 + bi)         # same noise for every c
            t0 = time.time()
            recon, _, _, stats = immap_sb(
                net, x1, sched, prox, y=y, cond=c_net, a_cond=c_A, mask=M, nfe=nfe,
                deterministic=bool(i2.get("deterministic", False)),
                posterior=i2.get("posterior", "ddpm"), clip_denoise=bool(i2.get("clip_denoise", False)),
                verbose=False)
            if device.type == "cuda":
                torch.cuda.synchronize()
            a = acc[c]
            a["sec"] += time.time() - t0
            recon = recon.real
            mets = compute_metrics(x0 * m, recon * m, data_range=data_range, mask=m)
            bs = x0.shape[0]
            a["psnr"] += float(mets["psnr"]) * bs
            a["ssim"] += float(mets["ssim"]) * bs
            a["n"] += bs
            e2 = (recon - x0) ** 2
            a["sse"] += float((e2 * m).sum()); a["npx"] += float(m.sum())
            a["sse_e"] += float((e2 * enh).sum()); a["npx_e"] += float(enh.sum())
            a["sse_r"] += float((e2 * rest).sum()); a["npx_r"] += float(rest.sum())
            with torch.no_grad():
                a["t1_sse"] += float(((y - A(recon, c_A)) ** 2 * m).sum())
            a["deltas"] += [s["delta_rms"] for s in stats if s.get("active")]
            if bi == 0:
                panel[c] = (recon[0, 0].cpu(), x0[0, 0].cpu(), y[0, 0].cpu(), m[0, 0].cpu())
        print(f"batch {bi + 1}/{len(loader)} done")

    sq = lambda s, n: math.sqrt(s / n) if n else float("nan")
    rows = []
    for c in cs:
        a = acc[c]
        rows.append({"c": c, "psnr": a["psnr"] / a["n"], "ssim": a["ssim"] / a["n"],
                     "rmse": sq(a["sse"], a["npx"]), "rmse_enh": sq(a["sse_e"], a["npx_e"]),
                     "rmse_rest": sq(a["sse_r"], a["npx_r"]), "t1_res": sq(a["t1_sse"], a["npx"]),
                     "delta": float(np.mean(a["deltas"])) if a["deltas"] else 0.0,
                     "seconds": a["sec"]})

    print(f"\n{'c':>6s} {'psnr':>7s} {'ssim':>7s} {'rmse':>7s} {'enh':>7s} {'rest':>7s} "
          f"{'t1_res':>7s} {'delta':>7s} {'sec':>7s}     (sigma_A = {sigma_A:.4f})")
    for r in rows:
        print(f"{r['c']:6.3g} {r['psnr']:7.3f} {r['ssim']:7.4f} {r['rmse']:7.4f} {r['rmse_enh']:7.4f} "
              f"{r['rmse_rest']:7.4f} {r['t1_res']:7.4f} {r['delta']:7.4f} {r['seconds']:7.1f}")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump({"args": vars(args), "sigma_A": sigma_A, "nfe": nfe, "rows": rows}, f, indent=2)

    import matplotlib.pyplot as plt
    plt.switch_backend("Agg")
    from visualization.image import set_display_orient, subplot_images

    # NYUMets stores canonical-RAS axes; draw the panel eyes-up.
    set_display_orient("radiological")
    _, gt, t1, mk = panel[cs[0]]
    img_rows = ([[gt, None], [t1, None]]
                + [[panel[c][0], panel[c][0] - gt] for c in cs])
    labels = ["CT1 (target)", "T1 (y)"] + [f"c = {c:g}" + (" (plain)" if c == 0 else "")
                                            for c in cs]
    v = float(torch.quantile((panel[cs[0]][0] - gt)[mk > 0.5].abs().float(), 0.99)) or 1.0
    fig, _ = subplot_images(
        img_rows, row_labels=labels, col_titles=["image", "sample - CT1"],
        cmap=["gray", "RdBu_r"], vmin=[None, -v], vmax=[None, v],
        window_from=[gt], p=(1, 99), mask=mk, apply_mask=True, magnitude=False,
        colorbar="each", panel_size=(3.0, 3.0), show=False)
    fig.savefig(os.path.join(args.out, "panel.png"), dpi=130, bbox_inches="tight")
    print(f"\nwrote {os.path.join(args.out, 'results.json')} and panel.png")


if __name__ == "__main__":
    main()
