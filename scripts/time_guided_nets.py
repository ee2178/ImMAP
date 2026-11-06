"""
Wall-clock per TRAINING STEP (forward + backward + clip + Adam) of the guided-recon nets at
fastMRI knee size, on a synthetic batch -- the number that decides what fits the budget.

    python scripts/time_guided_nets.py                                  # the TGVN sweep + LGGS
    python scripts/time_guided_nets.py --cells tgvn_c8_R8_g1 lggs_R8_g1 --iters 20
    python scripts/time_guided_nets.py --no-measure                     # skip the ESPIRiT row

Configs are generated fresh (scripts/make_guided_recon_configs.py into a temp dir), so each net
is built exactly as a launch would build it: same model block, same loss-free forward path
(training/recon.py::_forward for recon cells; the bridge regressor call for i2sb_recon cells),
same grad clip. Not timed: data loading and the loss (both negligible next to the nets). Timed
separately, once, as a shared overhead row: prepare_measurement with the config's online ESPIRiT
-- every cell pays it every step, so it shifts all rows equally.

Reported per cell: ms / step (median over --iters after --warmup), peak GPU memory, parameters,
and the hours that many steps would take at the config's num_epochs x steps_per_epoch.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from models import build_model                                       # noqa: E402
from operators import FFT2D, Mask, Sense                              # noqa: E402
from operators.fourier import fftc                                    # noqa: E402
from physics.mask import get_mask_cached, resolve_acs_lines           # noqa: E402
from training.common import embed_for_net, prepare_measurement        # noqa: E402
from training.recon import _forward, _guide                           # noqa: E402

DEFAULT_CELLS = ("tgvn_c12_R8_g1", "tgvn_c10_R8_g1", "tgvn_c8_R8_g1", "tgvn_c6_R8_g1",
                 "lggs_R8_g1", "lpdsnet_R8", "sbtgvn_c8_adjoint_R8_g1")
H, W, NC = 640, 368, 15


def gen_configs(cells, out):
    models = sorted({c.split("_R")[0] for c in cells})
    subprocess.run([sys.executable, os.path.join(REPO, "scripts", "make_guided_recon_configs.py"),
                    "--out", out, "--models", *models, "--guide-accels", "1"],
                   check=True, cwd=REPO, stdout=subprocess.DEVNULL)
    cfgs = {}
    for c in cells:
        with open(os.path.join(out, "knee", "guided", f"{c}.json")) as f:
            cfgs[c] = json.load(f)
    return cfgs


def synthetic_batch(device, bs, seed=0):
    g = torch.Generator().manual_seed(seed)
    lo = torch.randn(bs, NC, 8, 8, 2, generator=g)
    up = lambda t: torch.nn.functional.interpolate(t, size=(H, W), mode="bilinear")  # noqa: E731
    smaps = torch.complex(up(lo[..., 0]), up(lo[..., 1])) + 1.0
    smaps = smaps / smaps.abs().pow(2).sum(1, keepdim=True).sqrt()
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij")
    disk = ((yy ** 2 + xx ** 2) < 0.7).float()
    img = (disk * (0.1 + 0.2 * torch.rand(bs, 1, H, W, generator=g))).to(torch.complex64)
    side_img = (disk * (0.2 + 0.4 * torch.rand(bs, 1, H, W, generator=g))).to(torch.complex64)
    return (img.to(device), fftc(smaps * img).to(device), smaps.to(device),
            fftc(smaps * side_img).to(device))


def measurement(cfg, img, kspace, smaps, online=True):
    mri, tr = cfg["mri"], cfg["training"]
    mask = get_mask_cached(img, R=mri["R"], acs_lines=mri.get("acs_lines"),
                           mode=mri.get("mask_dist", "uniform"), offset=0,
                           center_frac=mri.get("center_frac"),
                           adjust_accel=mri.get("adjust_accel", False))
    kws = dict(mri.get("online_smaps_kws") or {},
               acs_lines=resolve_acs_lines(W, mri.get("acs_lines"), mri.get("center_frac")))
    return prepare_measurement(image=img, kspace=kspace, mask=mask, smaps=smaps,
                               kspace_type=(mri.get("kspace_type", "simulated") if online
                                            else "simulated"),
                               noise_std=tr.get("val_noise_std", 0.015),
                               noise_dist=tr.get("noise_dist", "uniform"), whiten_kspace=False,
                               online_smaps=mri.get("online_smaps") if online else None,
                               online_smaps_kws=kws), mask


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_cell(name, cfg, batch, meas, device, warmup, iters):
    net = build_model(cfg).to(device).train()
    params = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    clip = cfg["training"].get("clip_grad")
    img, kspace, smaps, side = batch
    (y, sigma_n, extra), mask = meas
    E = Mask(mask) @ FFT2D() @ Sense(extra["smaps"])
    E, T = embed_for_net(net, E, img, None)
    guide = _guide(net, side, extra["smaps"], T, device, 1)
    sigma = torch.as_tensor(sigma_n, device=device, dtype=torch.float32).reshape(-1, 1, 1, 1)
    sigma = sigma.expand(img.shape[0], 1, 1, 1)
    bridge = cfg["task"] == "i2sb_recon"
    if bridge:
        x1 = E.adjoint(y)
        x_t = 0.5 * (img if T.is_identity else T.adjoint(img)) + 0.5 * x1
        std = torch.full((img.shape[0], 1, 1, 1), 0.02, device=device)

    def step():
        opt.zero_grad(set_to_none=True)
        if bridge:
            kw = {} if guide is None else {"guide": guide}
            out, _ = net(x_t, sigma=std, x1=x1, y=y, E=E, sigma_n=sigma, **kw)
        else:
            out, _ = _forward(net, y, E, sigma, guide)
        loss = out.abs().pow(2).mean()          # the loss is negligible; any scalar will do
        loss.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(net.parameters(), float(clip))
        opt.step()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(warmup):
        step()
    times = []
    for _ in range(iters):
        sync(device)
        t0 = time.perf_counter()
        step()
        sync(device)
        times.append(time.perf_counter() - t0)
    peak = torch.cuda.max_memory_allocated(device) / 2 ** 30 if device.type == "cuda" else 0.0
    del net, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return statistics.median(times), peak, params


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", nargs="+", default=list(DEFAULT_CELLS))
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--no-measure", action="store_true",
                    help="skip online ESPIRiT entirely (the nets get the synthetic maps)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    with tempfile.TemporaryDirectory() as d:
        cfgs = gen_configs(args.cells, d)
    bs = int(next(iter(cfgs.values()))["data"]["train"]["batch_size"])
    batch = synthetic_batch(device, bs)
    print(f"knee {H}x{W}, {NC} coils, batch {bs}, device {device}"
          + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""))

    first = cfgs[args.cells[0]]
    meas = measurement(first, *batch[:3], online=not args.no_measure)     # warms ESPIRiT / FFT
    if not args.no_measure:
        ts = []
        for _ in range(max(3, args.iters // 2)):
            sync(device)
            t0 = time.perf_counter()
            measurement(first, *batch[:3])
            sync(device)
            ts.append(time.perf_counter() - t0)
        print(f"\nshared per-step overhead: prepare_measurement "
              f"({first['mri'].get('online_smaps') or 'stored'} maps) "
              f"{1e3 * statistics.median(ts):.0f} ms")

    print(f"\n{'cell':<18}{'ms/step':>9}{'peak GB':>9}{'params':>9}{'steps':>9}{'hours':>8}")
    for name in args.cells:
        cfg = cfgs[name]
        try:
            sec, peak, params = time_cell(name, cfg, batch, meas, device, args.warmup, args.iters)
        except Exception as e:                                            # noqa: BLE001
            print(f"{name:<18}  FAILED: {type(e).__name__}: {str(e)[:120]}")
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue
        tr = cfg["training"]
        steps = int(tr["num_epochs"]) * int(tr["steps_per_epoch"]) * int(tr.get("accum_steps", 1))
        print(f"{name:<18}{1e3 * sec:9.0f}{peak:9.2f}{params / 1e6:8.2f}M{steps:9d}"
              f"{steps * sec / 3600:8.1f}")
    print("\nhours = network fwd+bwd only; add the shared overhead x steps for the measurement.")


if __name__ == "__main__":
    main()
