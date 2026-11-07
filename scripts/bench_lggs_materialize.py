"""
Benchmark LGGS's guide branch on the triton backend: implicit (`TritonAdjacency`, the
attention re-run on every apply) vs materialised (`WindowAdjacency`, models/circulant_window.py:
softmax weights built once per refresh, then one weighted window sum per apply).

Stages, in order; the first three run by default, and each stops the run if it fails, so
no timing is printed for kernels that are wrong:

  1. kernels   each new triton kernel (scores fwd/bwd, apply fwd/bwd, transposed) against the
               torch reference, on small problems.
  2. branch    one guide branch at the knee latent size over one dK cycle -- build, dK applies
               with fresh values, backward -- implicit vs materialised: ms, peak memory, and
               the relative difference of the pooled energy and of dq / dk.
  3. net       a full LGGSNet training step (forward + backward + Adam) built from the LGGS
               config, same weights: implicit, materialised, and materialised with the
               planar state + planar convs (`--no-planar` skips it): ms / step, peak memory,
               and the output / per-parameter gradient differences against implicit, beside
               a noise floor (implicit with only the triton block size changed).
  4. profile   (only when asked for) one training step of one arm under torch.profiler:
               forward / backward / optimizer wall time, GPU-busy vs wall (is the step
               launch-bound?), kernel launches per step, GPU time by kind of kernel (convs,
               attention, fused clip, elementwise, copies, reductions, optimizer) and the
               top kernels; `--trace` also writes a chrome trace.

    python scripts/bench_lggs_materialize.py                       # all three, knee sizes
    python scripts/bench_lggs_materialize.py --stages branch net --iters 20
    python scripts/bench_lggs_materialize.py --H 320 --W 160 --batch 4 --guides 3
    python scripts/bench_lggs_materialize.py --json out.json       # also write the numbers
    python scripts/bench_lggs_materialize.py --stages profile      # where a step's time goes
    python scripts/bench_lggs_materialize.py --stages profile --profile-arm implicit --trace t.json

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
from models import conv_gemm                                          # noqa: E402
import models.guided_prox as gp                                       # noqa: E402
from models.circulant_triton import HAVE_TRITON, TritonAdjacency      # noqa: E402
from models.components import _GaussConvNd, set_complex_mode         # noqa: E402
from models.guided_lpds import LGGSNet                                # noqa: E402
from models.mg_lpds import MGLPDSNet                                  # noqa: E402


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

    # Nets from the same weights. "noise floor" is the implicit net with only the triton
    # kernel's query-block size changed -- which reorders its fp32 softmax reduction and nothing
    # else (GroupThreshold.triton_block_m) -- so its gap to "implicit" is what fp32 alone does
    # to the gradients through K thresholded layers. The other arms are judged against THAT.
    # "mat + planar" adds the planar state and planar complex convs (training.planar_state /
    # training.complex_conv: models/mg_lpds.py::set_planar_state, models/components.py).
    arms = [dict(name="implicit", mat=False, bm=a.block_m, planar=False, time=True),
            dict(name="materialized", mat=True, bm=a.block_m, planar=False, time=True)]
    if not a.no_planar:
        arms.append(dict(name="mat + planar", mat=True, bm=a.block_m, planar=True, time=True))
    arms.append(dict(name="noise floor", mat=False, bm=a.noise_block_m, planar=False,
                     time=False))
    res, ref = {}, {}
    for arm in arms:
        name = arm["name"]
        keep = (MGLPDSNet.PLANAR_STATE, _GaussConvNd.COMPLEX_MODE)
        MGLPDSNet.PLANAR_STATE = arm["planar"]
        set_complex_mode("planar" if arm["planar"] else keep[1])
        try:
            net = LGGSNet(**params, triton_materialize=arm["mat"], spectral_init=False)
            net = net.to(dev).train()
            net.load_state_dict(state)
            if arm["planar"] and not net.planar_state_active():
                raise SystemExit("planar arm: LGGSNet.planar_state_active() is False")
            for m in net.modules():
                if hasattr(m, "triton_block_m"):
                    m.triton_block_m = int(arm["bm"])
            opt = torch.optim.Adam(net.parameters(), lr=1e-5)

            def step(update=True):
                opt.zero_grad(set_to_none=True)
                xh, _ = net(y, guide=guides, sigma=sigma)
                xh.abs().pow(2).mean().backward()
                if update:
                    opt.step()
                return xh

            xh = step(update=False)                     # same weights for the comparison
            ref[name] = (xh.detach(), {n: p.grad.detach().clone()
                                       for n, p in net.named_parameters()
                                       if p.grad is not None})
            if arm["time"]:
                ms, mb = timed(step, a.warmup, a.iters)
                res[name] = dict(step_ms=ms, step_peak_mib=mb)
                print(f"  {name:14s}{ms:8.1f} ms / step   peak {mb:8.0f} MiB")
            del net, opt
            torch.cuda.empty_cache()
        finally:
            MGLPDSNet.PLANAR_STATE = keep[0]
            set_complex_mode(keep[1])

    base_out, base_g = ref["implicit"]
    others = [arm["name"] for arm in arms[1:]]
    res["rel_err"] = {}
    rows = {}
    for name in others:
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

    judged = [n for n in others if n != "noise floor"]
    worst = sorted(base_g, key=lambda n: -max(rows[k][n][0] for k in judged))[:a.top]
    print(f"\n  the {len(worst)} worst tensors, with the noise floor beside them:")
    print(f"  {'parameter':48s}{'||g||':>10s}" + "".join(f"{k:>14s}" for k in others))
    res["worst"] = []
    for n in worst:
        g = float(base_g[n].norm())
        print(f"  {n:48s}{g:10.1e}" + "".join(f"{rows[k][n][0]:14.1e}" for k in others))
        res["worst"].append(dict(param=n, grad_norm=g, **{k: rows[k][n][0] for k in others}))

    print()
    for name in judged:
        r = res["implicit"]["step_ms"] / res[name]["step_ms"]
        print(f"  speed-up per training step, {name}: {r:.2f}x")
    return res


# =============================================================================================
# 4. profile: where one training step's GPU time goes
# =============================================================================================
# First match wins; names lower-cased. The triton kernels show up under their @jit function
# names (models/circulant_window.py, circulant_triton.py, clip_triton.py).
BUCKETS = [
    ("attention: window (materialized)", r"^_wscores|^_wapply|^_wdot"),
    ("attention: triton (implicit)", r"^_fwd$|^_bwd_preprocess|^_bwd_dq|^_bwd_dkdv"),
    ("fused clip / envelope (triton)",
     r"^_clip_kernel|^_prox_kernel|^_prox_backward|^_scale_kernel|^_scale_backward"
     r"|^_abs2_seq|^_sqrt_img"),
    ("conv (cuDNN, incl. its layout kernels)",
     r"conv|cudnn|winograd|implicit_gemm|xmma|cutlass|dgrad|wgrad|fprop|nchwtonhwc"
     r"|nhwctonchw|gemm|sm\d\d_"),
    ("copy / fill / layout", r"copy|catarray|memcpy|memset|transpose|permute|fill"),
    ("reduction / softmax", r"reduce|softmax|logsumexp"),
    ("optimizer", r"multi_tensor|adam|foreach"),
    ("elementwise (other)", r"elementwise|vectorized|unrolled"),
]


def _bucket(name):
    import re
    n = name.lower()
    for label, pat in BUCKETS:
        if re.search(pat, n):
            return label
    return "other"


def _dev_us(e):
    v = getattr(e, "self_device_time_total", None)
    return float(v if v is not None else getattr(e, "self_cuda_time_total", 0.0))


def _build_arm(arm, params, state, dev):
    """(net, restore): an LGGSNet for `arm` with the process switches it needs set."""
    keep = (MGLPDSNet.PLANAR_STATE, _GaussConvNd.COMPLEX_MODE)
    planar = arm == "planar"
    MGLPDSNet.PLANAR_STATE = planar
    set_complex_mode("planar" if planar else keep[1])
    net = LGGSNet(**params, triton_materialize=arm != "implicit", spectral_init=False)
    net = net.to(dev).train()
    net.load_state_dict(state)

    def restore():
        MGLPDSNet.PLANAR_STATE = keep[0]
        set_complex_mode(keep[1])
    return net, restore


def stage_profile(dev, a):
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile, record_function
    import models.clip_triton as clip_triton

    params = lggs_params(a.config)
    if a.K:
        params["K"] = a.K
    Hi, Wi = 2 * a.H, 2 * a.W
    print(f"\n== 4. profile: one LGGSNet training step, arm '{a.profile_arm}', {Hi}x{Wi} image, "
          f"B={a.batch}, G={a.guides}, K={params['K']}, dK={params['dK']} ==")
    torch.manual_seed(0)
    state = LGGSNet(**params, spectral_init=False).state_dict()
    g = torch.Generator(device=dev).manual_seed(1)
    cplx = lambda: torch.complex(torch.randn(a.batch, 1, Hi, Wi, device=dev, generator=g),  # noqa: E731
                                 torch.randn(a.batch, 1, Hi, Wi, device=dev, generator=g))
    y, guides = cplx(), [cplx() for _ in range(a.guides)]
    sigma = torch.full((a.batch, 1, 1, 1), 0.015, device=dev)

    net, restore = _build_arm(a.profile_arm, params, state, dev)
    try:
        if a.profile_arm == "planar" and not net.planar_state_active():
            raise SystemExit("planar arm: LGGSNet.planar_state_active() is False")
        opt = torch.optim.Adam(net.parameters(), lr=1e-5)

        def step(events=None):
            if events:
                events[0].record()
            opt.zero_grad(set_to_none=True)
            with record_function("forward"):
                xh, _ = net(y, guide=guides, sigma=sigma)
                loss = xh.abs().pow(2).mean()
            if events:
                events[1].record()
            with record_function("backward"):
                loss.backward()
            if events:
                events[2].record()
            with record_function("optimizer"):
                opt.step()
            if events:
                events[3].record()

        for _ in range(max(a.warmup, 2)):                # compile triton, settle cuDNN
            step()
        torch.cuda.synchronize()

        # unprofiled phase timings
        phases = {"forward": [], "backward": [], "optimizer": [], "step": []}
        for _ in range(a.iters):
            ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            step(ev)
            torch.cuda.synchronize()
            phases["forward"].append(ev[0].elapsed_time(ev[1]))
            phases["backward"].append(ev[1].elapsed_time(ev[2]))
            phases["optimizer"].append(ev[2].elapsed_time(ev[3]))
            phases["step"].append(ev[0].elapsed_time(ev[3]))
        ph = {k: statistics.median(v) for k, v in phases.items()}
        print(f"  wall (median of {a.iters}): step {ph['step']:.1f} ms = forward "
              f"{ph['forward']:.1f} + backward {ph['backward']:.1f} + optimizer "
              f"{ph['optimizer']:.1f}")

        n = a.profile_steps
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(n):
                step()
            torch.cuda.synchronize()
        if a.trace:
            prof.export_chrome_trace(a.trace)
            print(f"  chrome trace: {a.trace}  (open in chrome://tracing or ui.perfetto.dev)")
    finally:
        restore()

    avg = prof.key_averages()
    # Kernels only. The record_function ranges ("forward", ...) also appear on the device
    # timeline as user annotations spanning their kernels; counting them double-counts.
    kern = [e for e in avg if getattr(e, "device_type", None) == DeviceType.CUDA
            and _dev_us(e) > 0 and not getattr(e, "is_user_annotation", False)
            and e.key not in ("forward", "backward", "optimizer")]
    if not kern:                                     # older profilers: device-only rows
        kern = [e for e in avg if _dev_us(e) > 0 and getattr(e, "self_cpu_time_total", 0) == 0
                and e.key not in ("forward", "backward", "optimizer")]
    gpu_ms = sum(_dev_us(e) for e in kern) / 1e3 / n
    launches = sum(e.count for e in kern) / n
    busy = gpu_ms / ph["step"]
    print(f"  GPU busy {gpu_ms:.1f} ms of a {ph['step']:.1f} ms step ({100 * busy:.0f}%), "
          f"{launches:.0f} kernel launches / step "
          f"({1e3 * ph['step'] / max(launches, 1):.1f} us of wall per launch)")
    if busy < 0.85:
        print("  -> the GPU idles for a sizeable part of the step: launch / host bound "
              "(CUDA graphs, fewer kernels) before faster kernels")
    print(f"  fused planar kernels: {clip_triton.planar_kernel_report()}")

    buckets = {}
    for e in kern:
        b = buckets.setdefault(_bucket(e.key), [0.0, 0])
        b[0] += _dev_us(e) / 1e3 / n
        b[1] += e.count / n
    print(f"\n  {'GPU time by kind':42s}{'ms/step':>10s}{'share':>8s}{'launches':>10s}")
    for label, (ms, cnt) in sorted(buckets.items(), key=lambda kv: -kv[1][0]):
        print(f"  {label:42s}{ms:10.1f}{100 * ms / gpu_ms:7.1f}%{cnt:10.0f}")

    # The backward, charged to the forward op it differentiates: each autograd node's
    # device_time_total is the GPU time of every kernel launched while evaluating it.
    nodes = []
    for e in avg:
        if e.key.startswith("autograd::engine::evaluate_function:"):
            us = float(getattr(e, "device_time_total", 0.0) or 0.0)
            if us > 0:
                nodes.append((us / 1e3 / n, e.count / n,
                              e.key.split("evaluate_function:", 1)[1].strip()))
    nodes.sort(reverse=True)
    bwd_ms = sum(t for t, _, _ in nodes)
    print(f"\n  backward by autograd node ({bwd_ms:.1f} ms/step of GPU time attributed):")
    print(f"  {'node':60s}{'ms/step':>9s}{'share':>8s}{'calls':>7s}")
    for t, c, k in nodes[:a.profile_top]:
        print(f"  {k[:58]:60s}{t:9.2f}{100 * t / max(bwd_ms, 1e-9):7.1f}%{c:7.0f}")

    top = sorted(kern, key=lambda e: -_dev_us(e))[:a.profile_top]
    print(f"\n  top {len(top)} kernels:")
    print(f"  {'kernel':72s}{'ms/step':>9s}{'share':>8s}{'calls':>7s}  kind")
    rows = []
    for e in top:
        ms = _dev_us(e) / 1e3 / n
        name = e.key if len(e.key) <= 70 else e.key[:67] + "..."
        print(f"  {name:72s}{ms:9.2f}{100 * ms / gpu_ms:7.1f}%{e.count / n:7.0f}  {_bucket(e.key)}")
        rows.append(dict(kernel=e.key, ms_per_step=ms, calls_per_step=e.count / n,
                         kind=_bucket(e.key)))
    return dict(arm=a.profile_arm, wall_ms=ph, gpu_busy_ms=gpu_ms, launches_per_step=launches,
                backward_nodes=[dict(node=k, ms_per_step=t, calls_per_step=c)
                                for t, c, k in nodes[:a.profile_top]],
                fused_kernels=clip_triton.planar_kernel_report(),
                buckets={k: dict(ms_per_step=v[0], launches=v[1]) for k, v in buckets.items()},
                top=rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stages", nargs="+", default=["kernels", "branch", "net"],
                    choices=["kernels", "branch", "net", "profile"],
                    help="'profile' is not in the default set: ask for it")
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
    ap.add_argument("--profile-arm", default="planar",
                    choices=["implicit", "materialized", "planar"],
                    help="profile: which net (planar = materialized + planar state + convs)")
    ap.add_argument("--profile-steps", type=int, default=3, help="profile: steps recorded")
    ap.add_argument("--profile-top", type=int, default=25, help="profile: kernels listed")
    ap.add_argument("--trace", default=None, help="profile: write a chrome trace here")
    ap.add_argument("--no-cudnn-benchmark", action="store_true",
                    help="leave cuDNN on its heuristics. Default ON, as train.py runs "
                         "(training.cudnn_benchmark, default true)")
    ap.add_argument("--no-tf32", action="store_true",
                    help="IEEE fp32 convs and matmuls (cudnn.allow_tf32 / matmul precision "
                         "'highest'): is a gap down to TF32?")
    ap.add_argument("--no-planar", action="store_true",
                    help="stage 3: skip the planar-state + planar-conv arm")
    ap.add_argument("--config", default=os.path.join(REPO, "config/knee/guided/lggs_R8_g1.json"))
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--json", default=None, help="write the results here")
    a = ap.parse_args()

    if not (HAVE_TRITON and torch.cuda.is_available()):
        raise SystemExit("needs CUDA and triton (the Linux CUDA wheels ship it)")
    dev = torch.device("cuda")
    # What train.py does: cuDNN autotunes each conv shape (the earlier runs of this script
    # used its heuristics instead, so their conv times are not the training ones).
    torch.backends.cudnn.benchmark = not a.no_cudnn_benchmark
    if a.no_tf32:
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    print(f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}, "
          f"matmul precision {torch.get_float32_matmul_precision()}, "
          f"cudnn.benchmark {torch.backends.cudnn.benchmark}, "
          f"cudnn.allow_tf32 {torch.backends.cudnn.allow_tf32}, "
          f"conv GEMM {conv_gemm.MODE or 'off'} (IMMAP_CONV_GEMM)")
    gp._FUSED_JOINT_CHECKED.clear()

    out = {"device": torch.cuda.get_device_name(0), "args": vars(a)}
    if "kernels" in a.stages:
        out["kernels"] = stage_kernels(dev)
    if "branch" in a.stages:
        out["branch"] = stage_branch(dev, a)
    if "net" in a.stages:
        out["net"] = stage_net(dev, a)
    if "profile" in a.stages:
        out["profile"] = stage_profile(dev, a)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
