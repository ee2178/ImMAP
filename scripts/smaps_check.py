#!/usr/bin/env python3
"""
Online vs offline coil maps: how much does the operator's map estimate cost, on its own?

    python scripts/smaps_check.py                                    # knee R8, 30 val slices
    python scripts/smaps_check.py config/knee/guided/lpdsnet_R8.json --n 10 --lams 1e-3 1e-2

Every guided-recon net here reconstructs through the ONLINE maps (ESPIRiT from the R-mask's
ACS, `mri.online_smaps`), but is scored against the OFFLINE reference (SENSE-combined with
maps from a 32-line ACS at thresh_eig 0.9, `data.val.smap_root`). Whatever separates the two is
an error floor no network -- guided or not -- can remove. This measures it with no network at
all, on the config's fixed val subset (scripts/eval_guided_recon.py::val_subset), on the
measured k-space as acquired (no added noise, as validation):

  full/offline   S_off^H F^-1 k, fully sampled: the reference rebuilt (a sanity check: ~exact)
  full/online    S_on^H F^-1 k, fully sampled: THE MAP FLOOR -- a perfect recon in the online
                 maps' convention, scored against the offline reference
  full/rss       coil RSS, fully sampled, for scale
  zf/online      E_on^H y at R: the zero-filled input the nets start from
  cg<lam>/<maps> Tikhonov CG-SENSE at R, (E^H E + lam I) x = E^H y, through either map set:
                 if online < offline here by more than the full/online floor, the online
                 maps also hurt the R-fold UNFOLDING (not only the coil combination)

Scored as training validates (magnitudes, the config's metric_range), over the whole FOV and
inside the offline maps' support (the reference is zero outside it; a recon that is not --
ESPIRiT at thresh_eig 0 keeps the background -- pays for that in the whole-FOV numbers only).
Also reported: the per-pixel map error inside the support, min_phi ||s_on - e^{i phi} s_off|| /
||s_off|| (one phase per pixel, common to the coils, is the ESPIRiT ambiguity).

Reads data only. CPU is fine for a few slices (the ESPIRiT estimate is the slow part).
"""
import argparse
import json
import os
import statistics
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from operators import FFT2D, Mask, Sense                               # noqa: E402
from operators.fourier import ifftc                                     # noqa: E402
from physics.mask import get_mask_cached, resolve_acs_lines             # noqa: E402
from training.common import prepare_measurement                         # noqa: E402
from training.metrics import compute_metrics                            # noqa: E402

DEFAULT_CONFIG = "config/knee/guided/lpdsnet_R8.json"


def coil_combine(k, maps):
    """S^H F^-1 k: (B, C, H, W) k-space -> (B, 1, H, W)."""
    return (maps.conj() * ifftc(k)).sum(dim=1, keepdim=True)


def cg_sense(y, mask, maps, lam, iters):
    from solvers.cg import batched_cg
    E = Mask(mask) @ FFT2D() @ Sense(maps)
    return batched_cg(lambda v: E.adjoint(E(v)) + lam * v, E.adjoint(y), tol=1e-8,
                      max_iter=int(iters))


def map_error(s_on, s_off, support):
    """Mean over the support of min_phi ||s_on - e^{i phi} s_off|| / ||s_off|| per pixel."""
    ip = (s_off.conj() * s_on).sum(dim=1)                       # <s_off, s_on> per pixel
    ph = torch.sgn(ip)
    err = (s_on - ph[:, None] * s_off).abs().pow(2).sum(1).sqrt()
    ref = s_off.abs().pow(2).sum(1).sqrt().clamp_min(1e-12)
    return float((err / ref)[support[:, 0]].mean())


def evaluate_slice(kspace, s_off, image, mri, training, lams=(1e-3, 1e-2), cg_iters=50,
                   seed=0):
    """All recons of one slice -> ({name: {region: metrics}}, extras). Tensors batched (1, ...)."""
    dev = kspace.device
    mask = get_mask_cached(image, R=mri["R"], acs_lines=mri.get("acs_lines"),
                           mode=mri.get("mask_dist", "uniform"), offset=mri.get("mask_offset", 0),
                           center_frac=mri.get("center_frac"),
                           adjust_accel=mri.get("adjust_accel", False))
    acs = resolve_acs_lines(image.shape[-1], mri.get("acs_lines"), mri.get("center_frac"))
    gen = torch.Generator(device=dev).manual_seed(int(seed))
    y, _, extra = prepare_measurement(
        image=image, kspace=kspace, mask=mask, smaps=s_off, kspace_type="measurement_awgn",
        noise_std=0.0, noise_dist=training.get("noise_dist", "uniform"), whiten_kspace=False,
        online_smaps=mri.get("online_smaps") or "espirit",
        online_smaps_kws=dict(mri.get("online_smaps_kws") or {}, acs_lines=acs),
        need_smaps=True, generator=gen)
    s_on = extra["smaps"]
    support = s_off.abs().sum(dim=1, keepdim=True) > 0

    recons = {
        "full/offline": coil_combine(kspace, s_off),
        "full/online": coil_combine(kspace, s_on),
        "full/rss": ifftc(kspace).abs().pow(2).sum(1, keepdim=True).sqrt(),
        "zf/online": (Mask(mask) @ FFT2D() @ Sense(s_on)).adjoint(y),
    }
    for lam in lams:
        recons[f"cg{lam:g}/offline"] = cg_sense(y, mask, s_off, lam, cg_iters)
        recons[f"cg{lam:g}/online"] = cg_sense(y, mask, s_on, lam, cg_iters)

    rng = training.get("metric_range", 1.0)
    out = {}
    for name, x in recons.items():
        out[name] = {}
        for region, m in (("fov", None), ("support", support)):
            r = compute_metrics(image.abs(), x.abs(), data_range=rng, mask=m)
            out[name][region] = {k: float(v.mean() if torch.is_tensor(v) else v)
                                 for k, v in r.items()}
    extras = {
        "map_error_support": map_error(s_on, s_off, support),
        "online_rss_outside_support": float(
            s_on.abs().pow(2).sum(1).sqrt()[~support[:, 0]].mean()) if (~support).any() else 0.0,
        "acs_lines": int(acs),
    }
    return out, extras


def use_cluster_roots(cfg):
    """A config generated on the other cluster names k-space roots that do not exist here: swap
    in THIS cluster's (IMMAP_CLUSTER; scripts/make_guided_recon_configs.py::KNEE_ROOTS) -- the
    only data path that differs between torch and bigpurple. Roots that exist are left alone."""
    cluster = os.environ.get("IMMAP_CLUSTER")
    if not cluster:
        return cfg
    from scripts.make_guided_recon_configs import KNEE_ROOTS
    for split in ("train", "val"):
        d = cfg.get("data", {}).get(split)
        if not d or "kspace_roots" not in d or all(os.path.isdir(r) for r in d["kspace_roots"]):
            continue
        print(f"[roots] {split}: {d['kspace_roots']} not found here -> {cluster}'s "
              f"{KNEE_ROOTS[cluster]}")
        d["kspace_roots"] = list(KNEE_ROOTS[cluster])
    return cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", default=DEFAULT_CONFIG,
                    help="a guided-recon config (its val data, mask and online-map settings)")
    ap.add_argument("--n", type=int, default=30, help="val slices")
    ap.add_argument("--lams", type=float, nargs="*", default=[1e-3, 1e-2],
                    help="Tikhonov weights for CG-SENSE (empty: none)")
    ap.add_argument("--cg-iters", type=int, default=50)
    ap.add_argument("--device", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    import yaml
    from scripts.eval_guided_recon import _batched, val_subset
    from training.recon import _unpack
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    use_cluster_roots(cfg)
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ds = val_subset(cfg)
    n = min(a.n, len(ds))
    print(f"{a.config}: R={cfg['mri']['R']}, online maps {cfg['mri'].get('online_smaps')} "
          f"{cfg['mri'].get('online_smaps_kws')}, reference maps {cfg['data']['val'].get('smap_root')}, "
          f"{n} val slices, metric_range {cfg['training'].get('metric_range', 1.0)}, device {dev}")

    per, ext = [], []
    for i in range(n):
        kspace, s_off, image, _, _, _ = _unpack(_batched(ds[i]))
        kspace, s_off, image = (t.to(dev) for t in (kspace, s_off, image))
        res, extras = evaluate_slice(kspace, s_off, image, cfg["mri"], cfg["training"],
                                     a.lams, a.cg_iters, seed=i)
        per.append(res)
        ext.append(extras)
        print(f"  slice {i + 1}/{n}: full/online PSNR {res['full/online']['fov']['psnr']:.2f} dB "
              f"(support {res['full/online']['support']['psnr']:.2f}), map error "
              f"{extras['map_error_support']:.3f}", flush=True)

    names = list(per[0])
    summary = {nm: {rg: {k: statistics.mean(p[nm][rg][k] for p in per)
                         for k in per[0][nm][rg]} for rg in ("fov", "support")} for nm in names}
    print(f"\n  {'recon':18s}{'PSNR fov':>10s}{'SSIM fov':>10s}{'PSNR supp':>11s}"
          f"{'SSIM supp':>11s}{'NRMSE supp':>12s}")
    for nm in names:
        s = summary[nm]
        print(f"  {nm:18s}{s['fov']['psnr']:10.2f}{s['fov']['ssim']:10.4f}"
              f"{s['support']['psnr']:11.2f}{s['support']['ssim']:11.4f}"
              f"{s['support']['nrmse']:12.4f}")
    me = statistics.mean(e["map_error_support"] for e in ext)
    bg = statistics.mean(e["online_rss_outside_support"] for e in ext)
    print(f"\n  online vs offline maps inside the support: mean relative error {me:.3f} "
          f"(per pixel, after the common phase); online map RSS outside it: {bg:.3f}")
    for lam in a.lams:
        d = (summary[f"cg{lam:g}/online"]["support"]["psnr"]
             - summary[f"cg{lam:g}/offline"]["support"]["psnr"])
        print(f"  CG-SENSE lam={lam:g}: online - offline maps = {d:+.2f} dB (support)")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"config": a.config, "n": n, "summary": summary, "per_slice": per,
                       "extras": ext}, f, indent=2)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
