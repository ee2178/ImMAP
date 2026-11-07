"""
Benchmark LGGS's guide branch on the triton backend: implicit (`TritonAdjacency`, the
attention re-run on every apply) vs materialised (`WindowAdjacency`, models/circulant_window.py:
softmax weights built once per refresh, then one weighted window sum per apply).

Three stages, in order; each stops the run if it fails, so no timing is printed for kernels
that are wrong:

  1. kernels   each new triton kernel (scores fwd/bwd, apply fwd/bwd, transposed) against the
               torch reference, on small problems.
  2. branch    one guide branch at the knee latent size over one dK cycle -- build, dK applies
               with fresh values, backward -- implicit vs materialised: ms, peak memory, and
               the relative difference of the pooled energy and of dq / dk.
  3. net       a full LGGSNet training step (forward + backward + Adam) built from the LGGS
               config, triton_materialize off vs on, same weights: ms / step, peak memory,
               and the relative difference of the output and the gradients.

    python scripts/bench_lggs_materialize.py                       # all three, knee sizes
    python scripts/bench_lggs_materialize.py --stages branch net --iters 20
    python scripts/bench_lggs_materialize.py --H 320 --W 160 --batch 4 --guides 3
    python scripts/bench_lggs_materialize.py --json out.json       # also write the numbers

`--H/--W` are the LATENT grid of stage 2 (knee: 640 x 368 image, s = 2 -> 320 x 184); stage 3
uses twice that as the image. Stage 3 runs the net with preproc='image' and E = Identity:
the k-space operator and ESPIRiT cost the same with either setting, so they are left out to
keep the difference visible (scripts/time_guided_nets.py times the full recon step).
CUDA + triton only. On Torch (L40S): `sbatch torch/bench_lggs_materialize.sbatch`; the
BigPurple A100 equivalent is bigpurple/bench_lggs_materialize.sbatch. Timings are per GPU type,
so compare implicit vs materialised within one run, not across clusters.
"""

import argparse
import json
import os
import statistics
import sys
import types

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.modules.setdefault("nibabel", types.ModuleType("nibabel"))

import models.circulant_window as cw                                  # noqa: E402
import models.guided_prox as gp                                       # noqa: E402
from models.circulant_triton import HAVE_TRITON, TritonAdjacency      # noqa: E402
from models.guided_lpds import LGGSNet                                # noqa: E402


def rel(a, b):
    """max |a - b| / max |b|: a single tensor's worst entry against its largest."""
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-30))


def nrel(a, b):
    """||a - b|| / ||b||: the whole tensor's relative difference."""
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def timed(fn, warmup, iters):
    """Median ms and peak MiB (over the timed calls) of fn() on the current device."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    peak = (torch.cuda.max_memory_allocated() - base) / 2 ** 20
    return statistics.median(ts), peak


# =============================================================================================
# 1. kernels vs reference
# =============================================================================================
def stage_kernels(dev):
    print("\n== 1. kernels vs torch reference ==")
    g = torch.Generator(device=dev).manual_seed(0)
    worst = 0.0
    for (H, W), win, cplx, knorm in [((40, 33), 15, True, True), ((40, 33), 15, False, True),
                                      ((24, 20), 5, True, False), ((17, 19), 3, True, True)]:
        B, C = 2, 64
        mk = lambda: (torch.randn(B, H * W, C, device=dev, generator=g) * 0.3).requires_grad_()  # noqa: E731
        qr, kr = mk(), mk()
        qi, ki = (mk(), mk()) if cplx else (None, None)
        ins = [t for t in (qr, qi, kr, ki) if t is not None]
        dS = torch.randn(B, win * win, H * W, device=dev, generator=g)
        res = []
        for force in (False, True):
            cw.FORCE_REFERENCE = force
            S = cw.window_scores(qr, qi, kr, ki, H, W, win, use_knorm=knorm)
            res.append((S.detach(),) + torch.autograd.grad(S, ins, dS))
        cw.FORCE_REFERENCE = False
        errs = [rel(a, b) for a, b in zip(*res)]
        worst = max(worst, *errs)
        print(f"  scores {H}x{W} win={win} complex={cplx} knorm={knorm}: "
              f"S {errs[0]:.1e}, grads " + " ".join(f"{e:.1e}" for e in errs[1:]))
    for (H, W), win, DV in [((40, 33), 15, 64), ((24, 20), 5, 33)]:
        B = 2
        P = torch.softmax(torch.randn(B, win * win, H * W, device=dev, generator=g), 1)
        P.requires_grad_()
        v = torch.randn(B, H * W, DV, device=dev, generator=g, requires_grad=True)
        dout = torch.randn(B, H * W, DV, device=dev, generator=g)
        for t in (False, True):
            res = []
            for force in (False, True):
                cw.FORCE_REFERENCE = force
                out = cw.window_apply(P, v, H, W, win, t)
                res.append((out.detach(),) + torch.autograd.grad(out, (P, v), dout))
            cw.FORCE_REFERENCE = False
            errs = [rel(a, b) for a, b in zip(*res)]
            worst = max(worst, *errs)
            print(f"  apply  {H}x{W} win={win} DV={DV} transpose={t}: out {errs[0]:.1e}, "
                  f"dP {errs[1]:.1e}, dv {errs[2]:.1e}")
    if not worst < 1e-3:
        raise SystemExit(f"kernel mismatch: worst relative error {worst:.2e} -- not timing")
    print(f"  OK (worst {worst:.1e})")
    return {"worst_rel_err": worst}


# =============================================================================================
# 2. one guide branch over one dK cycle
# =============================================================================================
def stage_branch(dev, a):
    H, W, B, C, win, dK = a.H, a.W, a.batch, a.Mh, a.guide_window, a.dK
    print(f"\n== 2. one guide branch, {H}x{W} latent, B={B}, Mh={C} complex, "
          f"{win}x{win} window, dK={dK} ==")
    g = torch.Generator(device=dev).manual_seed(0)
    cplx = lambda: torch.complex(torch.randn(B, C, H, W, device=dev, generator=g),  # noqa: E731
                                 torch.randn(B, C, H, W, device=dev, generator=g)) * 0.2
    q0, k0 = cplx(), cplx()
    vals = [torch.rand(B, C, H, W, device=dev, generator=g) for _ in range(dK)]

    def cycle(kind, grad=True):
        q, k = q0.detach().requires_grad_(grad), k0.detach().requires_grad_(grad)
        if kind == "materialized":
            adj = cw.WindowAdjacency(q, k, win, sim="pidistance")
        else:
            adj = TritonAdjacency(q, k, win, sim="pidistance", block_m=a.block_m)
        outs = [adj.apply(v) for v in vals]
        e = sum(o.sum() for o in outs) + adj._lse.sum()
        if grad:
            dq, dk = torch.autograd.grad(e, (q, k))
            return outs, dq, dk
        return outs, None, None

    res, out = {}, {}
    for kind in ("implicit", "materialized"):
        o, dq, dk = cycle(kind)
        out[kind] = (torch.stack([x.detach() for x in o]), dq, dk)
        with torch.no_grad():
            fwd_ms, fwd_mb = timed(lambda: cycle(kind, grad=False), a.warmup, a.iters)
        ms, mb = timed(lambda: cycle(kind), a.warmup, a.iters)
        res[kind] = dict(fwd_ms=fwd_ms, fwd_peak_mib=fwd_mb, step_ms=ms, step_peak_mib=mb)
        print(f"  {kind:12s}  fwd {fwd_ms:8.2f} ms  ({fwd_mb:7.0f} MiB)   "
              f"fwd+bwd {ms:8.2f} ms  ({mb:7.0f} MiB)")
    errs = [rel(x, y) for x, y in zip(out["materialized"], out["implicit"])]
    print(f"  difference: energy {errs[0]:.1e}, dq {errs[1]:.1e}, dk {errs[2]:.1e}")
    r = res["implicit"]["step_ms"] / res["materialized"]["step_ms"]
    print(f"  speed-up per dK cycle (fwd+bwd): {r:.2f}x")
    res["rel_err"] = dict(zip(("energy", "dq", "dk"), errs))
    return res


# =============================================================================================
# 3. a full LGGS training step
# =============================================================================================
def lggs_params(path):
    with open(path) as f:
        p = dict(json.load(f)["model"]["params"])
    p.update(preproc="image", attn_backend="triton")
    return p


def stage_net(dev, a):
    params = lggs_params(a.config)
    if a.K:
        params["K"] = a.K
    Hi, Wi = 2 * a.H, 2 * a.W
    print(f"\n== 3. LGGSNet training step, {Hi}x{Wi} image, B={a.batch}, G={a.guides}, "
          f"K={params['K']}, dK={params['dK']} ==")
    torch.manual_seed(0)
    proto = LGGSNet(**params, spectral_init=False)
    state = proto.state_dict()
    g = torch.Generator(device=dev).manual_seed(1)
    cplx = lambda: torch.complex(torch.randn(a.batch, 1, Hi, Wi, device=dev, generator=g),  # noqa: E731
                                 torch.randn(a.batch, 1, Hi, Wi, device=dev, generator=g))
    y, guides = cplx(), [cplx() for _ in range(a.guides)]
    sigma = torch.full((a.batch, 1, 1, 1), 0.015, device=dev)

    # Three nets from the same weights. "noise floor" is the implicit net with only the triton
    # kernel's query-block size changed -- which reorders its fp32 softmax reduction and nothing
    # else (GroupThreshold.triton_block_m) -- so its gap to "implicit" is what fp32 alone does
    # to the gradients through K thresholded layers. Materialised is judged against THAT.
    arms = [("implicit", False, a.block_m, True),
            ("materialized", True, a.block_m, True),
            ("noise floor", False, a.noise_block_m, False)]
    res, ref = {}, {}
    for name, mat, bm, time_it in arms:
        net = LGGSNet(**params, triton_materialize=mat, spectral_init=False).to(dev).train()
        net.load_state_dict(state)
        for m in net.modules():
            if hasattr(m, "triton_block_m"):
                m.triton_block_m = int(bm)
        opt = torch.optim.Adam(net.parameters(), lr=1e-5)

        def step(update=True):
            opt.zero_grad(set_to_none=True)
            xh, _ = net(y, guide=guides, sigma=sigma)
            xh.abs().pow(2).mean().backward()
            if update:
                opt.step()
            return xh

        xh = step(update=False)                         # same weights for the comparison
        ref[name] = (xh.detach(), {n: p.grad.detach().clone()
                                   for n, p in net.named_parameters() if p.grad is not None})
        if time_it:
            ms, mb = timed(step, a.warmup, a.iters)
            res[name] = dict(step_ms=ms, step_peak_mib=mb)
            print(f"  {name:12s}  {ms:8.1f} ms / step   peak {mb:8.0f} MiB")
        del net, opt
        torch.cuda.empty_cache()

    base_out, base_g = ref["implicit"]
    res["rel_err"] = {}
    rows = {}
    for name in ("materialized", "noise floor"):
        out, gr = ref[name]
        assert gr.keys() == base_g.keys(), f"{name}: a different set of parameters got gradients"
        rows[name] = {n: (nrel(gr[n], base_g[n]), rel(gr[n], base_g[n])) for n in base_g}
        res["rel_err"][name] = dict(
            output=rel(out, base_out),
            grad_norm_rel_median=statistics.median(v[0] for v in rows[name].values()),
            grad_norm_rel_worst=max(v[0] for v in rows[name].values()),
            grad_max_rel_worst=max(v[1] for v in rows[name].values()))
    print("\n  gradient difference vs implicit, ||dg|| / ||g|| per parameter tensor:")
    print(f"  {'':14s}{'output':>10s}{'median':>10s}{'worst':>10s}{'worst max-rel':>15s}")
    for name, e in res["rel_err"].items():
        print(f"  {name:14s}{e['output']:10.1e}{e['grad_norm_rel_median']:10.1e}"
              f"{e['grad_norm_rel_worst']:10.1e}{e['grad_max_rel_worst']:15.1e}")

    worst = sorted(base_g, key=lambda n: -rows["materialized"][n][0])[:a.top]
    print(f"\n  the {len(worst)} worst tensors (materialized), with the noise floor beside them:")
    print(f"  {'parameter':48s}{'||g||':>10s}{'mat':>10s}{'floor':>10s}")
    res["worst"] = []
    for n in worst:
        g = float(base_g[n].norm())
        m, f = rows["materialized"][n][0], rows["noise floor"][n][0]
        print(f"  {n:48s}{g:10.1e}{m:10.1e}{f:10.1e}")
        res["worst"].append(dict(param=n, grad_norm=g, materialized=m, noise_floor=f))

    r = res["implicit"]["step_ms"] / res["materialized"]["step_ms"]
    print(f"\n  speed-up per training step: {r:.2f}x")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stages", nargs="+", default=["kernels", "branch", "net"],
                    choices=["kernels", "branch", "net"])
    ap.add_argument("--H", type=int, default=320, help="latent rows (knee: 320)")
    ap.add_argument("--W", type=int, default=184, help="latent cols (knee: 184)")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--guides", type=int, default=1)
    ap.add_argument("--Mh", type=int, default=64)
    ap.add_argument("--guide-window", type=int, default=15)
    ap.add_argument("--dK", type=int, default=5)
    ap.add_argument("--K", type=int, default=None, help="override the config's K (stage 3)")
    ap.add_argument("--block-m", type=int, default=64, help="TritonAdjacency block_m")
    ap.add_argument("--noise-block-m", type=int, default=32,
                    help="block_m of stage 3's noise-floor arm (reorders fp32 only)")
    ap.add_argument("--top", type=int, default=8, help="stage 3: worst tensors to list")
    ap.add_argument("--config", default=os.path.join(REPO, "config/knee/guided/lggs_R8_g1.json"))
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--json", default=None, help="write the results here")
    a = ap.parse_args()

    if not (HAVE_TRITON and torch.cuda.is_available()):
        raise SystemExit("needs CUDA and triton (the Linux CUDA wheels ship it)")
    dev = torch.device("cuda")
    print(f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}, "
          f"matmul precision {torch.get_float32_matmul_precision()}")
    gp._FUSED_JOINT_CHECKED.clear()

    out = {"device": torch.cuda.get_device_name(0), "args": vars(a)}
    if "kernels" in a.stages:
        out["kernels"] = stage_kernels(dev)
    if "branch" in a.stages:
        out["branch"] = stage_branch(dev, a)
    if "net" in a.stages:
        out["net"] = stage_net(dev, a)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
