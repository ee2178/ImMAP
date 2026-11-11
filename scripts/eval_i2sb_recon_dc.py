"""
Sampling-time CG data consistency (I2SB-Recon, Wang et al., arXiv 2411.14269) on an ALREADY
TRAINED i2sb_recon run: no retraining, the regressor is the same, only the sampler changes.

    python -m scripts.eval_i2sb_recon_dc --run-dir trained_nets/i2sb_recon/knee/unet_R8
    python -m scripts.eval_i2sb_recon_dc --run-dir ... --dc 0:0 7:7e-5 7:1e-3 7:1e-2 20:1e-2

Each --dc entry is ITERS:LAM (0:0 = the plain sampler, always added). Every setting samples the
SAME val slices with the SAME measurement noise and the SAME bridge noise draws, so the rows
differ by the data-consistency step alone. Reported per setting, over the val subset the run
validates on (val_slices / val_seed / val_noise_std from its config):

    psnr / ssim / nrmse   magnitude vs the stored ground truth (organ mask if the run uses one)
    data_res              ||y - E x|| / ||y||: consistency of the final sample with the measurement
    seconds               wall clock for the subset

A panel (one slice: ground truth, then each setting's recon and |error|) and results.json go to
--out.
"""

import argparse
import contextlib
import json
import os
import time

import torch

import datasets                                    # noqa: F401  (registers loaders)
from datasets.registry import build_loader
from models import build_model
from sb.base import build_schedule
from sb.i2sb_recon import bridge_noise_scale, i2sb_inversion_sample, i2sb_recon_sample
import yaml
from training.forward_op import fixed_val_subset
from training.i2sb_recon import BESTVAL_CKPT, _measure, load_sampling_weights
from training.metrics import compute_metrics


def _setting(s):
    it, lam = s.split(":")
    return int(it), float(lam)


def load_run(run_dir, ckpt=BESTVAL_CKPT, device=None, config=None):
    """A trained i2sb_recon run, ready to sample: {"cfg", "net" (eval, weights loaded), "bridge"
    (its schedule), "meas" (training/i2sb_recon.py::_measure's settings, validation mask offset),
    "kspace_type", "noise_std" (the validation noise), "nfe", "seed", "i2", "tr", "device"}."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg_path = config or next(
        (p for p in (os.path.join(run_dir, f"config.{e}") for e in ("json", "yaml", "yml"))
         if os.path.exists(p)), None)
    if cfg_path is None:
        raise FileNotFoundError(f"no config.json / config.yaml in {run_dir}")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)                     # what train.py does on resume
    if cfg.get("task") != "i2sb_recon":
        raise ValueError(f"{cfg_path} is task {cfg.get('task')!r}, not i2sb_recon")
    i2, mri, tr = cfg["i2sb"], cfg["mri"], cfg["training"]
    net = build_model(cfg).to(device).eval()
    load_sampling_weights(net, os.path.join(run_dir, ckpt), device)   # net.ckpt -> its EMA
    bridge = build_schedule(kind=i2.get("kind", "brownian"), tau=i2["tau"],
                            n_points=i2.get("n_points", 1000), beta_max=i2.get("beta_max", 0.3),
                            device=device)
    if hasattr(net, "assert_schedule_matches"):
        net.assert_schedule_matches(bridge)
    meas = dict(R=mri["R"], acs_lines=mri.get("acs_lines", 24),
                mask_dist=mri.get("mask_dist", "uniform"), mask_offset=0,
                center_frac=mri.get("center_frac"), adjust_accel=mri.get("adjust_accel", False),
                noise_dist=tr.get("noise_dist", "uniform"), online_smaps=mri.get("online_smaps"),
                online_smaps_kws=mri.get("online_smaps_kws"),
                x1_source=i2.get("x1_source", "adjoint"),
                x0_source=i2.get("x0_source", "operator"),
                guide_R=mri.get("guide_R", 1), guide_acs_lines=mri.get("guide_acs_lines", 0),
                guide_mask_dist=mri.get("guide_mask_dist", "uniform"),
                guide_maps=i2.get("guide_maps", "operator"))
    noise_std = tr.get("val_noise_std")
    noise_std = tr.get("noise_std", (0.0, 0.01)) if noise_std is None else noise_std
    return dict(cfg=cfg, net=net, bridge=bridge, meas=meas, i2=i2, tr=tr, device=device,
                kspace_type=mri.get("val_kspace_type") or mri.get("kspace_type", "simulated"),
                noise_std=noise_std, nfe=int(i2.get("val_nfe", 20)),
                seed=int(tr.get("val_seed") or 0))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="an i2sb_recon run's save_dir (has config)")
    ap.add_argument("--config", default=None, help="default: <run-dir>/config.json")
    ap.add_argument("--ckpt", default=BESTVAL_CKPT,
                    help="file in run-dir: net_bestval.ckpt (best val, default) or net.ckpt "
                         "(the latest validation; its EMA weights are used)")
    ap.add_argument("--dc", type=_setting, nargs="+",
                    default=[(7, 7e-5), (7, 1e-3), (7, 1e-2)], metavar="ITERS:LAM")
    ap.add_argument("--nfe", type=int, default=None, help="default: the run's val_nfe")
    ap.add_argument("--noise", type=float, default=None,
                    help="added measurement noise std; default: the run's val_noise_std "
                         "(0 = the scan as acquired, its own noise only)")
    ap.add_argument("--n-slices", type=int, default=None, help="default: the run's val_slices")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None, help="default: the run's val_seed")
    ap.add_argument("--out", default=None, help="default: <run-dir>/eval_dc")
    ap.add_argument("--inversion", action="store_true",
                    help="also run the full I2SB-Inversion method (recon -> deterministic "
                         "inversion -> recon, sb/i2sb_recon.py::i2sb_inversion_sample) with the "
                         "run's own dc_iters / dc_lam; ~ n_points extra net calls per slice")
    ap.add_argument("--inversion-steps", type=int, default=None,
                    help="inversion length (their --inver_step); default the full interval")
    args = ap.parse_args()

    run = load_run(args.run_dir, args.ckpt, config=args.config)
    cfg, net, bridge, meas, i2, tr = (run[k] for k in ("cfg", "net", "bridge", "meas", "i2", "tr"))
    device, kspace_type, noise_std = run["device"], run["kspace_type"], run["noise_std"]
    if args.noise is not None:
        noise_std = float(args.noise)
    nfe = args.nfe or run["nfe"]
    seed = int(args.seed if args.seed is not None else run["seed"])
    tau_scale, tau_ref = i2.get("tau_scale", "none"), i2.get("tau_ref")
    vcfg = dict(cfg["data"]["val"])
    if args.batch:
        vcfg["batch_size"] = args.batch
    n = args.n_slices if args.n_slices is not None else tr.get("val_slices")
    loader = fixed_val_subset(build_loader(vcfg, shuffle=False, drop_last=False), n, seed)
    mask_metrics = bool(tr.get("use_organ_mask", False))

    settings = list(dict.fromkeys([(0, 0.0)] + list(args.dc)))
    names = ["plain" if it == 0 else f"dc{it}_lam{lam:g}" for it, lam in settings]
    inv_dc = (int(i2.get("dc_iters", 7)), float(i2.get("dc_lam", 7e-5)))
    if args.inversion:
        names.append(f"inversion_dc{inv_dc[0]}_lam{inv_dc[1]:g}")
    print(f"{type(net).__name__} {args.ckpt} | {len(loader.dataset)} val slices, nfe={nfe}, "
          f"seed={seed} | settings: {', '.join(names)}")

    acc = {k: dict(psnr=0., ssim=0., nrmse=0., data_res=0., sec=0., n=0) for k in names}
    panel = None
    gen = torch.Generator(device=device).manual_seed(seed)
    devs = [device] if device.type == "cuda" else []
    for bi, batch in enumerate(loader):
        x0, x1, y, E, sigma_n, organ_mask, gt, guide = _measure(
            batch, net, device, kspace_type=kspace_type, noise_std=noise_std, generator=gen,
            guide_offset=0, **meas)
        kappa = bridge_noise_scale(x1, tau_scale, tau_ref)
        mm = organ_mask if mask_metrics else None
        recs = []
        for name, (it, lam) in zip(names, settings + [inv_dc] * bool(args.inversion)):
            with torch.random.fork_rng(devices=devs):
                torch.manual_seed(seed * 100003 + bi)           # same bridge noise per setting
                t0 = time.time()
                if name.startswith("inversion"):
                    rec = i2sb_inversion_sample(net, y, E, bridge, x1, sigma_n=sigma_n, nfe=nfe,
                                                dc_iters=it, dc_lam=lam, guide=guide,
                                                n_steps_inv=args.inversion_steps,
                                                noise_scale=kappa,
                                                posterior=i2.get("posterior", "ddpm"))[0]
                else:
                    rec = i2sb_recon_sample(net, y, E, bridge, sigma_n=sigma_n, x1=x1, nfe=nfe,
                                            deterministic=bool(i2.get("deterministic", False)),
                                            posterior=i2.get("posterior", "ddpm"),
                                            verbose=False, guide=guide, noise_scale=kappa,
                                            dc_iters=it, dc_lam=lam)[0]
                if device.type == "cuda":
                    torch.cuda.synchronize()
            a, bs = acc[name], x0.shape[0]
            a["sec"] += time.time() - t0
            m = compute_metrics(gt.abs(), rec.abs(), mask=mm,
                                data_range=tr.get("metric_range", 1.0))
            for k in ("psnr", "ssim", "nrmse"):
                a[k] += float(m[k]) * bs
            res = (y - E.forward(rec)).abs().pow(2).flatten(1).sum(1).sqrt()
            a["data_res"] += float((res / y.abs().pow(2).flatten(1).sum(1).sqrt()).sum())
            a["n"] += bs
            recs.append(rec[:1].abs().cpu())
        if panel is None:
            panel = (gt[:1].abs().cpu(), recs, None if mm is None else mm[:1].cpu())
        print(f"  batch {bi}: " + "  ".join(
            f"{k} {acc[k]['psnr'] / acc[k]['n']:.2f}" for k in names))

    res = {k: {m: (v / a["n"] if m not in ("sec", "n") else v) for m, v in a.items()}
           for k, a in acc.items()}
    print(f"\n{'setting':<22}{'psnr':>8}{'ssim':>8}{'nrmse':>9}{'data_res':>10}{'sec':>8}")
    for k, r in res.items():
        print(f"{k:<22}{r['psnr']:8.2f}{r['ssim']:8.4f}{r['nrmse']:9.4f}{r['data_res']:10.4f}"
              f"{r['sec']:8.1f}")

    out = args.out or os.path.join(args.run_dir, "eval_dc")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(dict(run_dir=args.run_dir, ckpt=args.ckpt, nfe=nfe, seed=seed,
                       settings=dict(zip(names, settings + [inv_dc] * bool(args.inversion))),
                       results=res), f, indent=2)
    if panel is not None:
        _save_panel(os.path.join(out, "panel.png"), names, *panel)
    print(f"wrote {out}/results.json" + (" and panel.png" if panel is not None else ""))


def _save_panel(path, names, gt, recs, mask):
    with contextlib.suppress(ImportError):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        m = 1.0 if mask is None else mask[0, 0].float()
        g = gt[0, 0] * m
        vmax = float(g.max())
        fig, ax = plt.subplots(2, len(recs) + 1, figsize=(3 * (len(recs) + 1), 6))
        ax[0, 0].imshow(g, cmap="gray", vmin=0, vmax=vmax)
        ax[0, 0].set_title("ground truth")
        ax[1, 0].axis("off")
        for j, (name, r) in enumerate(zip(names, recs), start=1):
            r = r[0, 0] * m
            ax[0, j].imshow(r, cmap="gray", vmin=0, vmax=vmax)
            ax[0, j].set_title(name)
            ax[1, j].imshow((r - g).abs(), cmap="magma", vmin=0, vmax=0.2 * vmax)
            ax[1, j].set_title("|error| (x5)")
        for a in ax.flat:
            a.set_xticks([])
            a.set_yticks([])
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)


if __name__ == "__main__":
    main()
