#!/usr/bin/env python3
"""
Helpers for evaluating trained STANDARD (one-shot, task "recon") guided reconstruction runs --
TGVN, LGGS, LPDSNet -- on chosen val slices, exactly as training/recon.py validates them:

    run = load_recon_run("trained_nets/guided_recon/knee/s3/tgvn_c8_R8_g1")
    ds  = val_subset(run["cfg"])                  # the run's fixed val subset (val_slices, val_seed)
    m   = measure(run, ds[i], noise=0.0, seed=0)  # y, E, guide, target, zero-filled ... for slice i
    x   = reconstruct(run, m)                     # the net's recon on the measured grid

`measure` is training/recon.py's validation block (mask at the config's offset, prepare_measurement
with the config's protocol and online maps, the net's embedding, the guide routed the way the net
takes it), with the ADDED noise std as an argument (0 = the scan as acquired) and a seed, so every
run sees the same measurement of the same slice. notebooks/guided_recon_compare.ipynb uses these.
"""

import os
import sys

import torch
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import datasets                                   # noqa: E402,F401  (registers the loaders)
from datasets.registry import build_loader        # noqa: E402
from models import build_model                    # noqa: E402
from operators import FFT2D, Mask, Sense          # noqa: E402
from physics.mask import get_mask_cached, resolve_acs_lines  # noqa: E402
from training.common import embed_for_net, load_ckpt, prepare_measurement  # noqa: E402
from training.forward_op import fixed_val_subset  # noqa: E402
from training.recon import BESTVAL_CKPT, _forward, _guide, _unpack  # noqa: E402


def load_recon_run(run_dir, ckpt=BESTVAL_CKPT, device=None):
    """{"cfg", "net" (eval, weights loaded), "device", "name"} for a trained task-"recon" run."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = yaml.safe_load(f)                   # what train.py does on resume
    if cfg.get("task") != "recon":
        raise ValueError(f"{run_dir} is task {cfg.get('task')!r}, not recon")
    net = build_model(cfg).to(device).eval()
    load_ckpt(os.path.join(run_dir, ckpt), model=net, device=device)
    return dict(cfg=cfg, net=net, device=device, name=os.path.basename(run_dir.rstrip("/\\")))


def val_subset(cfg, batch_size=1):
    """The run's fixed val subset as a Dataset: val_slices slices drawn with val_seed (what its
    validation scores), batch_size 1."""
    tr = cfg["training"]
    vcfg = dict(cfg["data"]["val"], batch_size=batch_size)
    loader = fixed_val_subset(build_loader(vcfg, shuffle=False, drop_last=False),
                              tr.get("val_slices"), int(tr.get("val_seed") or 0))
    return loader.dataset


def _batched(item):
    return [t[None] if torch.is_tensor(t) else torch.as_tensor(t)[None] for t in item]


@torch.no_grad()
def measure(run, item, noise=None, seed=0):
    """One dataset item -> the measurement dict the run's net is validated on: {"y", "E" (on the
    net's grid), "T", "sigma", "guide", "gt" (stored target), "zf" (E^H y, measured grid), "mask",
    "smaps", "side", "acs_lines"}. `noise`: ADDED noise std (None = the run's val_noise_std)."""
    cfg, net, dev = run["cfg"], run["net"], run["device"]
    mri, tr = cfg["mri"], cfg["training"]
    kspace, smaps, image, organ, pad_hw, side = _unpack(_batched(item))
    kspace, smaps, image = (t.to(dev) for t in (kspace, smaps, image))
    noise = tr.get("val_noise_std", 0.0) if noise is None else noise
    mask = get_mask_cached(image, R=mri["R"], acs_lines=mri.get("acs_lines"),
                           mode=mri.get("mask_dist", "uniform"), offset=mri.get("mask_offset", 0),
                           center_frac=mri.get("center_frac"),
                           adjust_accel=mri.get("adjust_accel", False))
    acs = resolve_acs_lines(image.shape[-1], mri.get("acs_lines"), mri.get("center_frac"))
    gen = torch.Generator(device=dev).manual_seed(int(seed))
    y, sigma, extra = prepare_measurement(
        image=image, kspace=kspace, mask=mask, smaps=smaps,
        kspace_type=mri.get("val_kspace_type") or mri.get("kspace_type", "simulated"),
        noise_std=noise, noise_dist=tr.get("noise_dist", "uniform"),
        whiten_kspace=mri.get("whiten_kspace", False), generator=gen,
        online_smaps=mri.get("online_smaps"),
        online_smaps_kws=dict(mri.get("online_smaps_kws") or {}, acs_lines=acs),
        need_smaps=getattr(net, "uses_operator_smaps", True))
    maps = extra["smaps"]
    E0 = Mask(mask) @ FFT2D() @ Sense(maps)
    E, T = embed_for_net(net, E0, image, pad_hw)
    guide = _guide(net, side, maps, T, dev, mri.get("guide_R", 1), mri.get("guide_acs_lines", 0),
                   mri.get("guide_mask_dist", "uniform"), guide_offset=0)
    return dict(y=y, E=E, T=T, sigma=sigma, guide=guide, gt=image, zf=E0.adjoint(y), mask=mask,
                smaps=maps, side=side, acs_lines=acs)


@torch.no_grad()
def reconstruct(run, m):
    """The net's recon of a measure() dict, cropped back to the measured grid (complex)."""
    out, _ = _forward(run["net"], m["y"], m["E"], m["sigma"], m["guide"])
    return m["T"].forward(out)


def guide_image(m):
    """The guide as an image for display: |guide| for image-domain guides, the coil RSS of a
    k-space guide (TGVN takes k-space), the centre slice of several."""
    g = m["guide"]
    if g is None:
        return None
    if g.dim() == 5:
        g = g[:, g.shape[1] // 2]
    if g.shape[1] > 1:                            # k-space, coils on dim 1
        from operators.fourier import ifftc
        return ifftc(g).abs().pow(2).sum(1, keepdim=True).sqrt()
    return g.abs()
