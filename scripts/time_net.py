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

--planar-state times every MGLPDSNet a second time with its iterates carried
as real [re; im] tensors (`MGLPDSNet.PLANAR_STATE`; same parameters, same map),
straight after the normal timing, and prints the pair. It is how to find out
whether that representation is worth adopting.

--compile times every net a second time under `torch.compile`, straight after
its eager timing on the same node, and prints the pair: the one switch that can
be applied identically to LPDSNet, the multigrid nets and E2E-VarNet. It also
reports what the compiler could not do (graph breaks and why), how long the
first calls took, and whether a NEW operator of the same shape forces a
recompile -- in training every step brings one.

Settings come from each config: `training.complex_conv` (planar / gauss) and
`model.params` (K, M, coarse_op, ...). `cudnn.benchmark` is on, as in train.py.
Synthetic SENSE data by default -- timing depends on shapes, not values. A size
that is not a multiple of the model's stride is embedded exactly as training
does (`E @ Truncate`); the rediscretized coarse Gram then runs on the measured
grid's half and quarter (operators/coarse.py), so pass such a size to time it.
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


def _first_line(e, width=160):
    msg = str(e).strip().splitlines()
    return f"{type(e).__name__}: {msg[0] if msg else ''}"[:width]


def time_config(cfg_path, hw, coils, R, reps, warmup, device, with_step, ckpt,
                compile_mode=None, planar_state=False):
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

    def measure(fwd):
        """Time `fwd` -- the module itself, or its compiled wrapper. Mode
        switches, gradients and the optimizer always go to `net`.
        -> `(stats per column, peak MB, seconds spent in the first calls)`."""
        def infer(y=y, E=E):
            net.eval()
            with torch.no_grad():
                t0 = clock.now()
                fwd(y, E=E, sigma=sigma)
                return {"infer": clock.now() - t0}

        def train_step():
            net.train()
            net.zero_grad(set_to_none=True)
            t0 = clock.now()
            recon, _ = fwd(y, E=E, sigma=sigma)
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

        # warm-up: every code path once (cuDNN autotune, lazy kernel compiles,
        # and under torch.compile the compilation itself -- hence `first_s`)
        t0 = clock.now()
        infer()
        train_step()
        first_s = clock.now() - t0
        for _ in range(max(warmup, 1) - 1):
            infer()
            train_step()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        samples = {}
        for _ in range(reps):
            for k, v in {**infer(), **train_step()}.items():
                samples.setdefault(k, []).append(1e3 * v)
        peak = (torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == "cuda"
                else float("nan"))
        return {k: stats(v) for k, v in samples.items()}, peak, first_s, infer

    cols, peak, first_s, _ = measure(net)

    p = cfg["model"]["params"]
    K = p.get("K", p.get("denoiser_kws", {}).get("K"))
    if K is None and "num_cascades" in p:
        K = f"{p['num_cascades']}casc"
    row = dict(name=os.path.splitext(os.path.basename(cfg_path))[0],
               type=cfg["model"]["type"], K=K, M=p.get("M"), params=n_par,
               coarse_op=p.get("coarse_op"), complex_conv=mode,
               embedded=not T.is_identity, peak_mb=peak, first_s=first_s, **cols)
    if planar_state:
        row["planar_state"] = time_planar_state(net, measure)
    if compile_mode:
        # With --planar-state as well, the compiled arm runs ON the planar
        # state where the net has one: real tensors are what the compiler can
        # generate code for, so that is the combination worth knowing about.
        on_planar = bool(planar_state and "skipped" not in row.get("planar_state", {}))
        row["compiled"] = time_compiled(net, measure, compile_mode, row,
                                        lambda: build_problem(hw, coils, mri, R, device, seed=1),
                                        planar=on_planar)
        print(f"  {row['name']}: " + (row["compiled"].get("error") or "compiled and timed"),
              flush=True)
    return row


def time_planar_state(net, measure):
    """The same measurement with the planar state on, or why it was not run."""
    import models.clip_triton as clip_mod
    from models.mg_lpds import MGLPDSNet

    if not isinstance(net, MGLPDSNet):
        return dict(skipped="not an MGLPDSNet")
    keep = MGLPDSNet.PLANAR_STATE
    MGLPDSNet.PLANAR_STATE = True
    try:
        if not net.planar_state_active():
            return dict(skipped="not implemented for this net (group prox, widen > 1 "
                                "or learned transfers)")
        cols, peak, first_s, _ = measure(net)
        kernel = {None: "not used", True: "active", False: "DISABLED (mismatch)"}[
            clip_mod._PLANAR_KERNEL_OK]
        return dict(peak_mb=peak, first_s=first_s, fused_clip=kernel, **cols)
    finally:
        MGLPDSNet.PLANAR_STATE = keep


def print_planar_state(rows):
    print("\nplanar state against the complex state, each pair timed back to back "
          "(ms, median; x = complex / planar, above 1 is a speedup):")
    keys = [k for k in ("infer", "forward", "backward", "opt", "project")
            if all(k in r for r in rows)]
    print(f"  {'config':<16}" + "".join(f"{k:^26}" for k in keys) + "  fused clip")
    for r in rows:
        p = r.get("planar_state") or {}
        if "skipped" in p or not p:
            print(f"  {r['name']:<16}{p.get('skipped', 'not run')}")
            continue
        cells = "".join(f"{r[k]['median']:>8.1f} > {p[k]['median']:<8.1f}"
                        f"{r[k]['median'] / p[k]['median']:>5.2f}x " for k in keys)
        print(f"  {r['name']:<16}{cells} {p['fused_clip']}")
    print("\n  Same parameters and the same map (to fp roundoff). `fused clip` is the "
          "planar Triton\n  kernel used at inference: it checks itself against the eager "
          "formula on first use.")


def time_compiled(net, measure, compile_mode, eager, new_problem, planar=False):
    """The same measurement under `torch.compile`, or `{"error": ...}`.

    Two hand-written host-side shortcuts are switched OFF for this arm, because
    the compiler cannot trace through them and would fall back to eager around
    each one: the fused Triton prox (`SoftThreshold.FUSED`) and the planar
    weight cache (`_GaussConvNd.PLANAR_WEIGHT_CACHE`, which reads `data_ptr`).
    Fusing those chains is the compiler's own job.
    """
    from models.components import _GaussConvNd
    from models.prox import SoftThreshold
    from training.common import embed_for_net as _embed

    from models.mg_lpds import MGLPDSNet

    saved = (SoftThreshold.FUSED, _GaussConvNd.PLANAR_WEIGHT_CACHE)
    saved_state = MGLPDSNet.PLANAR_STATE
    SoftThreshold.FUSED = _GaussConvNd.PLANAR_WEIGHT_CACHE = False
    MGLPDSNet.PLANAR_STATE = bool(planar)
    try:
        first_error = None
        for settings in COMPILE_ATTEMPTS:
            out = _compile_once(net, measure, compile_mode, eager, new_problem, _embed,
                                settings)
            out["planar_state"] = bool(planar)
            if "error" not in out:
                if first_error:
                    out["note"] = (f"compiled only with {settings}; as shipped: "
                                   f"{first_error}")
                return out
            if first_error and out["error"] != first_error:
                out["error"] = f"{first_error}  |  with {settings}: {out['error']}"
            first_error = first_error or out["error"]
        return out                                          # every attempt failed
    finally:
        SoftThreshold.FUSED, _GaussConvNd.PLANAR_WEIGHT_CACHE = saved
        MGLPDSNet.PLANAR_STATE = saved_state


# Tried in order until one compiles. Inductor's layout pass moves 4-D conv
# graphs to channels_last, and a COMPLEX tensor in that layout cannot be viewed
# as its real pairs ("self.stride(-1) must be 1 to view ComplexFloat as Float,
# but got <channels>") -- which is how every complex net here failed on torch
# 2.11. So the second attempt switches that pass off. A real-valued net (VarNet)
# compiles on the first attempt and keeps the pass.
COMPILE_ATTEMPTS = ({}, {"layout_optimization": False})


def _compile_once(net, measure, compile_mode, eager, new_problem, _embed, settings):
    import contextlib
    import traceback

    counters = None
    try:
        import torch._dynamo as dynamo
        dynamo.reset()
        try:
            from torch._dynamo.utils import counters
            counters.clear()
        except Exception:                                   # private API: best effort
            counters = None
        ctx = contextlib.nullcontext()
        if settings:
            import torch._inductor.config as inductor_config
            ctx = inductor_config.patch(**settings)
        with ctx:
            cnet = torch.compile(net,
                                 mode=None if compile_mode == "default" else compile_mode)
            cols, peak, first_s, infer = measure(cnet)
        out = dict(mode=compile_mode, peak_mb=peak, first_s=first_s,
                   inductor_settings=dict(settings), **cols)

        # A fresh operator of the SAME shape, as every training step and every
        # new slice brings: does the compiled code accept it, or recompile?
        y2, E02, image2 = new_problem()
        E2, _ = _embed(net, E02, image2, None)
        t_new = 1e3 * infer(y2, E2)["infer"]
        out["new_operator_ms"] = t_new
        out["new_operator_recompiles"] = bool(t_new > 5.0 * cols["infer"]["median"])

        if counters is not None:
            try:
                breaks = counters["graph_break"]
                out["graphs"] = int(counters["stats"].get("unique_graphs", 0))
                out["graph_breaks"] = int(sum(breaks.values()))
                out["graph_break_reasons"] = [[str(k)[:140], int(v)]
                                              for k, v in breaks.most_common(4)]
            except Exception:
                pass
        return out
    except Exception as e:                                  # report it, time the rest
        # the one-line message loses WHERE it failed: the full trace goes to
        # stderr (the job's .err log)
        print(f"\n[time_net] torch.compile failed for {eager['name']} with inductor "
              f"settings {dict(settings) or 'as shipped'}:", file=sys.stderr)
        traceback.print_exc()
        sys.stderr.flush()
        return dict(mode=compile_mode, inductor_settings=dict(settings),
                    error=f"torch.compile failed -- {_first_line(e)}")
    finally:
        try:
            import torch._dynamo as dynamo
            dynamo.reset()
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def print_compiled(rows, mode):
    """Eager against compiled, per network, from the paired measurement."""
    print(f"\ntorch.compile (mode={mode}) against eager, each pair timed back to back "
          f"(ms, median; x = eager / compiled, above 1 is a speedup):")
    head = (f"  {'config':<16}{'infer':^26}{'forward':^26}{'backward':^26}"
            f"{'first calls':>12}{'graphs':>8}{'breaks':>8}  new operator")
    print(head)
    notes = []
    for r in rows:
        c = r.get("compiled") or {}
        if "error" in c or not c:
            print(f"  {r['name']:<16}{c.get('error', 'not run')}")
            continue
        cells = "".join(
            f"{r[k]['median']:>8.1f} > {c[k]['median']:<8.1f}"
            f"{r[k]['median'] / c[k]['median']:>5.2f}x "
            for k in ("infer", "forward", "backward"))
        new = (f"{c['new_operator_ms']:.0f} ms: RECOMPILES" if c["new_operator_recompiles"]
               else f"{c['new_operator_ms']:.0f} ms: reused")
        print(f"  {r['name']:<16}{cells}{c['first_s']:>10.0f} s"
              f"{str(c.get('graphs', '?')):>8}{str(c.get('graph_breaks', '?')):>8}  {new}"
              + ("  [planar state]" if c.get("planar_state") else "")
              + ("  [*]" if c.get("note") else ""))
        if c.get("note"):
            notes.append(f"  [*] {r['name']}: {c['note']}")
    for line in notes:
        print(line)
    reasons = {}
    for r in rows:
        for why, n in (r.get("compiled") or {}).get("graph_break_reasons", []):
            reasons.setdefault(why, []).append(f"{r['name']} x{n}")
    if reasons:
        print("\n  what the compiler could not trace (it runs eager around each of these):")
        for why, who in reasons.items():
            print(f"    {why}\n        {', '.join(who)}")
    print("\n  The compiled arm runs with the fused Triton prox and the planar weight "
          "cache OFF\n  (untraceable; fusing them is the compiler's job). `first calls` is "
          "compile time.\n  A net that RECOMPILES for a new operator would recompile on "
          "every training step.")


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
    ap.add_argument("--planar-state", action="store_true",
                    help="also time each MGLPDSNet with its iterates carried as "
                         "real [re; im] tensors, right after its normal timing")
    ap.add_argument("--compile", nargs="?", const="default", default=None,
                    choices=("default", "reduce-overhead", "max-autotune"),
                    help="also time each net under torch.compile, right after its "
                         "eager timing; optionally a compile mode")
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

    if args.compile and not hasattr(torch, "compile"):
        raise SystemExit(f"--compile needs torch >= 2.0 (this is {torch.__version__})")
    rows = [time_config(c, hw, args.coils, args.R, args.reps, args.warmup, device,
                        args.step, args.ckpt, compile_mode=args.compile,
                        planar_state=args.planar_state)
            for c in args.configs]

    cols = ["infer", "forward", "backward"] + (["opt", "project"] if args.step else [])
    nw = max(len(r["name"]) for r in rows) + 2
    head = (f"{'config':<{nw}}{'K':<18}{'M':>4}{'params':>11} {'conv':<7}{'coarse':<13}"
            + "".join(f"{c:>10}" for c in cols) + f"{'fwd+bwd':>10}{'peak MB':>9}")
    print(head + "    (ms, median)")
    print("-" * len(head))
    for r in rows:
        fb = r["forward"]["median"] + r["backward"]["median"]
        print(f"{r['name']:<{nw}}{str(r['K']):<18}{str(r['M'] or ''):>4}{r['params']:>11,} "
              f"{r['complex_conv']:<7}{str(r['coarse_op'] or '-'):<13}"
              + "".join(f"{r[c]['median']:>10.1f}" for c in cols)
              + f"{fb:>10.1f}{r['peak_mb']:>9.0f}"
              + ("   [embedded]" if r["embedded"] else ""))
    print("\nspread (min / p95 / max, ms) -- a wide one means the timing, not the "
          "net, is unstable:")
    for r in rows:
        print(f"  {r['name']:<{nw}}" + "   ".join(
            f"{c} {r[c]['lo']:.1f}/{r[c]['p95']:.1f}/{r[c]['hi']:.1f}" for c in cols))
    if len(rows) > 1:
        base = rows[0]
        print(f"\nrelative to {base['name']}:")
        for r in rows[1:]:
            print(f"  {r['name']:<{nw}} infer {r['infer']['median'] / base['infer']['median']:.2f}x"
                  f"   fwd+bwd "
                  f"{(r['forward']['median'] + r['backward']['median']) / (base['forward']['median'] + base['backward']['median']):.2f}x")
    if args.planar_state:
        print_planar_state(rows)
    if args.compile:
        print_compiled(rows, args.compile)
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
