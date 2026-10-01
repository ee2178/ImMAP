"""
Time LGGS with LEARNED data consistency, across UNet sizes for the learned operator E.

    python -m scripts.time_lggs_learned_dc --size 128 --batch 4 --backward
    python -m scripts.time_lggs_learned_dc --unets 4:1:1 8:1:1 8:2:1 16:2:2 16:3:2 --size 224

On the cluster (the numbers only mean anything on the GPU the runs train on):

    CMD="python -m scripts.time_lggs_learned_dc --size 128 --batch 4 --backward \\
         --csv bigpurple/logs/lggs_dc_timing.csv" \\
        sbatch --job-name=lggs-dc-time bigpurple/gpu_cmd.sbatch

What it answers: what a learned CT1 -> T1 operator costs inside an LGGS bridge regressor, as a
function of how big that operator is, starting from the shallowest UNet that exists. Each of the
sweep after the cold start adds one data gradient -- an E forward plus its VJP
(operators/learned.py); sweep 0 only runs the prox. So the overhead over the no-DC LGGS should be
close to (K - 1) * E grad, which is the `pred` column; a measured overhead far above it means
something other than E is paying.

The setting is the MAGNITUDE approach with COMPLEX weights: LGGS runs `is_complex=True` on the
real, scale-only NYUMets intensities, and E sees |x| (operators/learned.MagnitudeOperator). That
needs data whose background is 0 -- the `img_raw / scales` normalisation -- so |x| is an image E
was trained on. It is wrapped in the same BridgeDCOperator the i2sb regressor uses at bridge
position `--t`, so the timed call is the real regressor call, not a stand-in.

Arms (always the same LGGS weights, frame, batch and guides):

    none          LGGS with no data operator -- today's synthesis LGGS (E = Identity)
    w:L:c         LGGS + learned DC, E = ForwardOp(kind="unet", width=w, levels=L, convs=c,
                  cond_channels=--cond). 4:1:1 is the shallowest UNet ForwardOp can build.

Columns, all medians over --reps after --warmup:

    E params / RF   size and receptive field of the operator
    E fwd           one forward of the operator alone, same frame and batch (ms)
    E grad          one data_grad: E forward + VJP, i.e. what each sweep pays (ms)
    infer           LGGS forward under no_grad -- one bridge step at sampling time (ms)
    over            infer - infer(none): what the learned DC adds (ms), and x it multiplies by
    pred            (K - 1) * E grad: what that overhead SHOULD be if E is all it costs
    train           forward + magnitude loss + backward + step, second order through the VJP
                    (--backward; the number that sets epoch time)
    peak            peak CUDA memory of the training step, else of inference (MB)
    sample/slice    infer / batch * --nfe: the cost of sampling one slice at val_nfe (ms)

Random data: a cost measurement, not a quality one. E's weights are random too -- the cost of a
UNet does not depend on what it learned.
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
import yaml

from models import build_model
from models.forward_ops import ForwardOp
from operators.learned import BridgeDCOperator, MagnitudeOperator
from sb.base import bridge_coeffs, build_schedule, n_steps


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_ms(fn, device, warmup, reps):
    """-> (median ms, peak MB). Peak is measured over the timed reps only."""
    for _ in range(warmup):
        fn()
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        _sync(device)
        ts.append((time.perf_counter() - t0) * 1e3)
    peak = torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == "cuda" else float("nan")
    return float(np.median(ts)), float(peak)


def guarded(fn, device, warmup, reps):
    """time_ms, but an OOM becomes a row entry instead of ending the sweep."""
    try:
        return time_ms(fn, device, warmup, reps)
    except RuntimeError as e:                      # torch.cuda.OutOfMemoryError is a RuntimeError
        if "out of memory" not in str(e).lower():
            raise
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return float("nan"), float("inf")


def parse_unet(spec):
    try:
        w, L, c = (int(v) for v in spec.split(":"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"--unets entries are width:levels:convs, got {spec!r}")
    if w < 1 or L < 1 or c < 1:
        raise argparse.ArgumentTypeError(f"{spec}: width, levels and convs must all be >= 1 "
                                         f"(4:1:1 is the shallowest UNet ForwardOp builds)")
    return w, L, c


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/NYUMets/i2sb_lggs_session.json",
                    help="the LGGS whose model.params are timed (is_complex is forced True)")
    ap.add_argument("--unets", nargs="+", type=parse_unet,
                    default=[parse_unet(s) for s in ("4:1:1", "8:1:1", "8:2:1", "16:2:1",
                                                     "16:2:2", "16:3:2")],
                    help="E sizes as width:levels:convs, shallowest first. 16:3:2 is the "
                         "existing ForwardOp (forward_op_unet_w16_l3_xc)")
    ap.add_argument("--cond", type=int, default=2,
                    help="E's side-information channels (2 = T2, FLAIR, as forward_op_unet_w16_l3_xc)")
    ap.add_argument("--size", type=int, default=128, help="square frame: 128 = the training crop, "
                                                          "224 = a full stored slice")
    ap.add_argument("--batch", type=int, default=4, help="4 = the LGGS training batch")
    ap.add_argument("--guides", type=int, default=1, help="guide planes per sample")
    ap.add_argument("--K", type=int, default=None, help="override the config's number of sweeps")
    ap.add_argument("--attn-backend", default=None, dest="attn_backend",
                    help="override the guided prox backend (gather | flex | triton ...)")
    ap.add_argument("--t", type=float, default=0.5, help="bridge position in [0, 1]")
    ap.add_argument("--sigma-E", type=float, default=0.05, dest="sigma_E",
                    help="E's residual std for BridgeDCOperator; does not affect cost")
    ap.add_argument("--plain", action="store_true",
                    help="time the bare MagnitudeOperator instead of wrapping it in the bridge "
                         "DC operator (no i2sb, just LGGS with a learned forward model)")
    ap.add_argument("--nfe", type=int, default=20, help="sampling steps, for the sample/slice column")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--backward", action="store_true", help="also time a full training step")
    ap.add_argument("--csv", default=None, help="also write the table here")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    cfg = ap.parse_args()

    device = torch.device("cpu" if cfg.cpu or not torch.cuda.is_available() else "cuda")
    torch.manual_seed(cfg.seed)

    with open(cfg.config) as f:
        mcfg = yaml.safe_load(f)
    params = dict(mcfg["model"]["params"])
    if mcfg["model"]["type"] not in ("LGGS", "LGGSNet", "GuidedLPDSNet"):
        raise SystemExit(f"{cfg.config}: model.type is {mcfg['model']['type']!r}, not LGGS")
    params["is_complex"] = True                    # the magnitude approach: complex weights
    if params.get("preproc") == "kspace":
        params["preproc"] = "image"                # a learned E acts in the image domain
    if cfg.K is not None:
        params["K"] = cfg.K
    if cfg.attn_backend is not None:
        params["attn_backend"] = cfg.attn_backend
    net = build_model({"model": {"type": "LGGS", "params": params}}).to(device)
    if getattr(net, "attn_backend", None) == "flex":
        net.compile_flex()
    K = net.K

    B, H = cfg.batch, cfg.size
    x_t = torch.rand(B, 1, H, H, device=device)               # bridge state (== T1-ish)
    t1 = torch.rand(B, 1, H, H, device=device)                # E's measurement
    target = torch.rand(B, 1, H, H, device=device)
    cond = torch.rand(B, cfg.cond, H, H, device=device) if cfg.cond else None
    guide = torch.rand(B, cfg.guides, 1, H, H, device=device)

    sched = build_schedule(kind="brownian", tau=0.1, n_points=1000, beta_max=0.3, device=device)
    mu0, mu1, std_sb = bridge_coeffs(sched)
    k = min(int(round(cfg.t * (n_steps(sched) - 1))), n_steps(sched) - 1)
    col = lambda tab: tab[k].reshape(1, 1, 1, 1).expand(B, 1, 1, 1)   # noqa: E731

    def make_dc(E):
        op = MagnitudeOperator(E, cond)
        if cfg.plain:
            return op, t1
        return BridgeDCOperator(op, x1=t1, t1=t1, mu0=col(mu0), mu1=col(mu1),
                                std_sb=col(std_sb), sigma_E=cfg.sigma_E), x_t

    print(f"device {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"LGGS from {cfg.config}: K={K} M={net.M} P={net.P} s={net.s} "
          f"guide_window={params.get('guide_window')} backend={params.get('attn_backend')} "
          f"complex=True preproc={params.get('preproc')}")
    print(f"frame {H}x{H}, batch {B}, {cfg.guides} guide(s), E cond={cfg.cond}, "
          f"{'bare MagnitudeOperator' if cfg.plain else f'BridgeDCOperator at t={cfg.t}'}, "
          f"median of {cfg.reps} after {cfg.warmup} warmup\n")

    opt = torch.optim.Adam(net.parameters(), lr=1e-6)

    def train_step(E_dc, y):
        net.train()
        opt.zero_grad(set_to_none=True)
        out, _ = net(y, guide=guide, E=E_dc)
        loss = ((out.abs() - target) ** 2).mean()          # magnitude-mse
        loss.backward()
        opt.step()

    rows = []

    # ---- baseline: no data operator --------------------------------------------------------
    def f_none():
        with torch.no_grad():
            net.eval()
            net(x_t, guide=guide)
    base_ms, base_mem = guarded(f_none, device, cfg.warmup, cfg.reps)
    base_train, base_tmem = (float("nan"), float("nan"))
    if cfg.backward:
        base_train, base_tmem = guarded(lambda: train_step(None, x_t), device,
                                        cfg.warmup, cfg.reps)
    rows.append(dict(arm="none", E_params=0, E_rf=0, E_fwd=0.0, E_grad=0.0, infer=base_ms,
                     over=0.0, over_x=1.0, pred=0.0, train=base_train,
                     peak=base_tmem if cfg.backward else base_mem,
                     sample=base_ms / B * cfg.nfe))

    # ---- one arm per UNet size ------------------------------------------------------------
    for (w, L, c) in cfg.unets:
        E = ForwardOp(kind="unet", width=w, levels=L, convs=c, cond_channels=cfg.cond)
        with torch.no_grad():                    # off the identity init, so the VJP is not trivial
            E.net.out.weight.normal_(0, 0.02)
        E = E.to(device).requires_grad_(False).eval()
        n_par = sum(p.numel() for p in E.parameters())
        mag = MagnitudeOperator(E, cond)
        xc = torch.complex(x_t, 0.01 * torch.randn_like(x_t))

        def f_E():
            with torch.no_grad():
                mag(xc)

        def f_vjp():
            with torch.no_grad():
                mag.data_grad(xc, t1)

        e_fwd, _ = guarded(f_E, device, cfg.warmup, cfg.reps)
        e_grad, _ = guarded(f_vjp, device, cfg.warmup, cfg.reps)      # fwd + VJP

        dc, y = make_dc(E)

        def f_infer():
            with torch.no_grad():
                net.eval()
                net(y, guide=guide, E=dc)

        inf_ms, inf_mem = guarded(f_infer, device, cfg.warmup, cfg.reps)
        tr_ms, tr_mem = (float("nan"), float("nan"))
        if cfg.backward:
            tr_ms, tr_mem = guarded(lambda: train_step(dc, y), device, cfg.warmup, cfg.reps)

        rows.append(dict(arm=f"{w}:{L}:{c}", E_params=n_par, E_rf=E.receptive_field,
                         E_fwd=e_fwd, E_grad=e_grad, infer=inf_ms,
                         over=inf_ms - base_ms, over_x=inf_ms / base_ms if base_ms else float("nan"),
                         pred=(K - 1) * e_grad, train=tr_ms,
                         peak=tr_mem if cfg.backward else inf_mem,
                         sample=inf_ms / B * cfg.nfe))
        del E, mag, dc
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- report -----------------------------------------------------------------------------
    hdr = (f"{'arm':>8} {'E params':>9} {'RF':>4} {'E fwd':>7} {'E grad':>7} {'infer':>9} "
           f"{'over':>9} {'x':>6} {'pred':>9} {'train':>9} {'peak MB':>9} {'sample/slice':>13}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['arm']:>8} {r['E_params']:>9,d} {r['E_rf']:>4} {r['E_fwd']:>7.2f} "
              f"{r['E_grad']:>7.2f} {r['infer']:>9.1f} {r['over']:>9.1f} {r['over_x']:>6.2f} "
              f"{r['pred']:>9.1f} {r['train']:>9.1f} {r['peak']:>9.0f} {r['sample']:>13.1f}")
    print(f"\nms throughout. over = infer - infer(none); pred = (K - 1) * E grad with K={K}; sweep 0 is the cold start.")
    print(f"sample/slice = infer / batch * nfe (nfe={cfg.nfe}). "
          f"OOM shows as nan time / inf memory.")
    if not cfg.backward:
        print("train is nan: pass --backward to time a training step.")

    if cfg.csv:
        os.makedirs(os.path.dirname(os.path.abspath(cfg.csv)), exist_ok=True)
        with open(cfg.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {cfg.csv}")


if __name__ == "__main__":
    sys.exit(main())
