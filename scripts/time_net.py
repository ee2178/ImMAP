#!/usr/bin/env python3
"""
Network speed, and nothing else: forward and backward, GPU-synchronised.

    python scripts/time_net.py --configs config/brain/mg/mglpds_R12.json \\
                                         config/brain/mg/lpdsnet_R12.json

The training progress bar cannot answer "how fast is this net": it lumps the
data loader, the online coil-map estimate, the optimizer, the projection and the
end-of-epoch work into one it/s. This builds the model from its config, puts ONE
measurement on the GPU, and times only the network.

The method is Sljiva's. Its eval closure wraps the net call alone --
`fwdtime = CUDA.@elapsed((x, st) = net((y, A), ps, st))` (src/closures/ssdu.jl)
-- and `CUDA.@elapsed` synchronises before and after; `main.jl` runs one val
and one train call first so compilation is never measured. Here that is
`torch.cuda.synchronize()` on both sides of each call, after `--warmup` calls.

Columns (ms, median over --reps):
    infer     eval mode, no autograd -- the deployment forward
    forward   train mode, building the autograd graph
    backward  loss (the config's loss_type) + loss.backward()
    fwd+bwd   their sum: the network's share of one training step
With --step:
    opt       grad clip + optimizer.step()   (Adam, fused when it applies)
    project   net.project()

--fig-out writes a figure of the same numbers (visualization/timing_chart.py):
one upright column per network, the forward pass with the backward pass
stacked on top of it. --from-json redraws it from a saved --json-out without
timing anything.

Settings come from each config: `training.complex_conv` (planar / gauss) and
`model.params` (K, M, coarse_op, ...). `cudnn.benchmark` is on, as in train.py.
Synthetic SENSE data by default -- timing depends on shapes, not values. A size
that is not a multiple of the model's stride is embedded exactly as training
does (`E @ Truncate`), which is the case where the rediscretized coarse Gram
falls back to Galerkin, so pass such a size to see what that costs.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



def _import_repo():
    """The model stack, imported only when something is timed: --from-json
    redraws a figure anywhere matplotlib runs (a laptop whose torch is too old
    to import the models)."""
    global build_model, set_complex_mode, FFT2D, Mask, Sense
    global make_acc_mask, embed_for_net, LOSS_REGISTRY
    from models import build_model
    from models.components import set_complex_mode
    from operators import FFT2D, Mask, Sense
    from physics.mask import make_acc_mask
    from training.common import embed_for_net
    from training.losses import LOSS_REGISTRY


def build_problem(hw, coils, mri, R, device, seed=0):
    """One SENSE measurement of the requested shape: (y, E, image)."""
    g = torch.Generator().manual_seed(seed)
    H, W = hw
    lo = torch.randn(1, coils, 8, 8, dtype=torch.complex64, generator=g)
    smaps = torch.complex(
        F.interpolate(lo.real, size=(H, W), mode="bilinear", align_corners=False),
        F.interpolate(lo.imag, size=(H, W), mode="bilinear", align_corners=False)) + 1.0
    smaps = smaps / smaps.abs().pow(2).sum(1, keepdim=True).sqrt()
    cf = mri.get("center_frac")
    mask = make_acc_mask(shape=(H, W), accel=R,
                         acs_lines=mri.get("acs_lines") if cf is None else None,
                         mode="uniform", offset=0, center_frac=cf,
                         adjust_accel=bool(mri.get("adjust_accel", False)))
    mask = mask.reshape(1, 1, H, W).float()
    image = torch.randn(1, 1, H, W, dtype=torch.complex64, generator=g)
    noise = torch.randn(1, coils, H, W, dtype=torch.complex64, generator=g)
    smaps, mask, image = (t.to(device) for t in (smaps, mask, image))
    E = Mask(mask) @ FFT2D() @ Sense(smaps)
    y = E.forward(image) + 0.01 * noise.to(device)
    return y, E, image


class Clock:
    """`CUDA.@elapsed`: wall time between two synchronisation points."""

    def __init__(self, device):
        self.cuda = device.type == "cuda"

    def now(self):
        if self.cuda:
            torch.cuda.synchronize()
        return time.perf_counter()


def stats(v):
    v = sorted(v)
    return dict(median=statistics.median(v), lo=v[0],
                p95=v[min(len(v) - 1, int(round(0.95 * (len(v) - 1))))], hi=v[-1])


def time_config(cfg_path, hw, coils, R, reps, warmup, device, with_step, ckpt):
    with open(cfg_path) as f:
        cfg = json.load(f)
    mode = set_complex_mode(cfg.get("training", {}).get("complex_conv") or "gauss")
    torch.manual_seed(0)
    net = build_model(cfg).to(device)
    if ckpt:
        state = torch.load(ckpt, map_location=device, weights_only=False)
        net.load_state_dict(state.get("model_state_dict", state))
    if getattr(net, "attn_backend", None) == "flex" and device.type == "cuda":
        net.compile_flex()
    n_par = sum(p.numel() for p in net.parameters())
    mri = cfg.get("mri", {})
    R = R if R is not None else mri.get("R", 8)

    y, E0, image = build_problem(hw, coils, mri, R, device)
    E, T = embed_for_net(net, E0, image, None)
    sigma = torch.full((1, 1, 1, 1), 0.01, device=device)
    loss_fn = LOSS_REGISTRY[cfg.get("training", {}).get("loss_type", "magnitude-nl1-nl2")]
    clock = Clock(device)

    opt = None
    if with_step:
        from train import build_optimizer
        opt = build_optimizer(net, cfg) if "optimizer" in cfg else torch.optim.Adam(
            net.parameters(), lr=1e-4)
    clip = cfg.get("training", {}).get("clip_grad", 1.0)

    def infer():
        net.eval()
        with torch.no_grad():
            t0 = clock.now()
            net(y, E=E, sigma=sigma)
            return {"infer": clock.now() - t0}

    def train_step():
        net.train()
        net.zero_grad(set_to_none=True)
        t0 = clock.now()
        recon, _ = net(y, E=E, sigma=sigma)
        recon = T.forward(recon)
        t1 = clock.now()
        loss = loss_fn(image, recon, sigma)
        loss.backward()
        t2 = clock.now()
        out = {"forward": t1 - t0, "backward": t2 - t1}
        if opt is not None:
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()
            t3 = clock.now()
            if hasattr(net, "project"):
                net.project()
            out.update(opt=t3 - t2, project=clock.now() - t3)
        return out

    # warm-up: every code path once (cuDNN autotune, lazy kernel compiles)
    for _ in range(max(warmup, 1)):
        infer()
        train_step()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    samples = {}
    for _ in range(reps):
        for k, v in {**infer(), **train_step()}.items():
            samples.setdefault(k, []).append(1e3 * v)
    peak = torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == "cuda" else float("nan")

    p = cfg["model"]["params"]
    K = p.get("K", p.get("denoiser_kws", {}).get("K"))
    if K is None and "num_cascades" in p:
        K = f"{p['num_cascades']}casc"
    return dict(name=os.path.splitext(os.path.basename(cfg_path))[0],
                type=cfg["model"]["type"], K=K, M=p.get("M"), params=n_par,
                coarse_op=p.get("coarse_op"), complex_conv=mode,
                embedded=not T.is_identity, peak_mb=peak,
                **{k: stats(v) for k, v in samples.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="+", default=None)
    ap.add_argument("--from-json", default=None,
                    help="skip timing: load a previous --json-out and only draw "
                         "the figure (needs --fig-out)")
    ap.add_argument("--fig-out", default=None,
                    help="save a figure of the timings (.png, .pdf or .svg)")
    ap.add_argument("--fig-theme", default="light", choices=("light", "dark"))
    ap.add_argument("--size", type=int, nargs="+", default=[640, 320], help="H [W]")
    ap.add_argument("--coils", type=int, default=20)
    ap.add_argument("--R", type=int, default=None, help="default: each config's mri.R")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--step", action="store_true",
                    help="also time grad clip + optimizer.step() and net.project()")
    ap.add_argument("--ckpt", default=None,
                    help="load these weights (single config only); timing does "
                         "not depend on them unless a prox saturates differently")
    ap.add_argument("--no-cudnn-benchmark", action="store_true")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.from_json:
        if not args.fig_out:
            raise SystemExit("--from-json only redraws the figure; pass --fig-out")
        with open(args.from_json) as f:
            saved = json.load(f)
        from visualization.timing_chart import save_timing_figure
        print("wrote", save_timing_figure(saved["rows"], saved, args.fig_out,
                                          theme=args.fig_theme))
        return
    if not args.configs:
        raise SystemExit("--configs is required (or --from-json to redraw a figure)")
    _import_repo()

    device = torch.device(args.device)
    hw = (args.size[0], args.size[-1])
    torch.backends.cudnn.benchmark = not args.no_cudnn_benchmark
    if args.ckpt and len(args.configs) > 1:
        raise SystemExit("--ckpt takes exactly one config")
    print(f"device {device}" + (f" ({torch.cuda.get_device_name(0)})"
                                if device.type == "cuda" else "")
          + f" | torch {torch.__version__} | {hw[0]}x{hw[1]}, {args.coils} coils"
          f" | cudnn.benchmark {torch.backends.cudnn.benchmark}"
          f" | {args.reps} reps after {args.warmup} warm-up\n")

    rows = [time_config(c, hw, args.coils, args.R, args.reps, args.warmup, device,
                        args.step, args.ckpt) for c in args.configs]

    cols = ["infer", "forward", "backward"] + (["opt", "project"] if args.step else [])
    head = (f"{'config':<14}{'K':<18}{'M':>4}{'params':>11} {'conv':<7}{'coarse':<13}"
            + "".join(f"{c:>10}" for c in cols) + f"{'fwd+bwd':>10}{'peak MB':>9}")
    print(head + "    (ms, median)")
    print("-" * len(head))
    for r in rows:
        fb = r["forward"]["median"] + r["backward"]["median"]
        print(f"{r['name']:<14}{str(r['K']):<18}{str(r['M'] or ''):>4}{r['params']:>11,} "
              f"{r['complex_conv']:<7}{str(r['coarse_op'] or '-'):<13}"
              + "".join(f"{r[c]['median']:>10.1f}" for c in cols)
              + f"{fb:>10.1f}{r['peak_mb']:>9.0f}"
              + ("   [embedded]" if r["embedded"] else ""))
    print("\nspread (min / p95 / max, ms) -- a wide one means the timing, not the "
          "net, is unstable:")
    for r in rows:
        print(f"  {r['name']:<14}" + "   ".join(
            f"{c} {r[c]['lo']:.1f}/{r[c]['p95']:.1f}/{r[c]['hi']:.1f}" for c in cols))
    if len(rows) > 1:
        base = rows[0]
        print(f"\nrelative to {base['name']}:")
        for r in rows[1:]:
            print(f"  {r['name']:<14} infer {r['infer']['median'] / base['infer']['median']:.2f}x"
                  f"   fwd+bwd "
                  f"{(r['forward']['median'] + r['backward']['median']) / (base['forward']['median'] + base['backward']['median']):.2f}x")
    meta = dict(size=list(hw), coils=args.coils, device=str(device),
                device_name=(torch.cuda.get_device_name(0) if device.type == "cuda"
                             else "CPU"),
                torch=torch.__version__, reps=args.reps, warmup=args.warmup,
                cudnn_benchmark=bool(torch.backends.cudnn.benchmark))
    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(dict(meta, rows=rows), f, indent=1)
        print(f"\nwrote {args.json_out}")
    if args.fig_out:
        from visualization.timing_chart import save_timing_figure
        os.makedirs(os.path.dirname(os.path.abspath(args.fig_out)), exist_ok=True)
        print("wrote", save_timing_figure(rows, meta, args.fig_out, theme=args.fig_theme))


if __name__ == "__main__":
    main()
