#!/usr/bin/env python3
"""
Inference profiler for the multigrid reconstruction grid.

Answers three questions that the accuracy sweep in `eval_mg_recon.py` does not:

  1. How long does a forward pass actually take, per config?
  2. Where does that time go -- FFT (physics), convolution (dictionary),
     grid transfer, or elementwise / Python overhead?
  3. What did the Tier-1 Gram optimisations buy, measured rather than argued?

Why it exists
-------------
`MGLPDSNet` at `K=[6,[4,4,6]]` runs ~2x slower than the `K=30` baseline. The
layer count only explains ~1.4x of that. The rest is the Gram: `galerkin` builds
the coarse operator as `E_c = E . R`, so a coarse level prolongs to FULL
resolution, runs the full multicoil FFT, and restricts back. Counting confirms
it -- 108 Grams per forward against the baseline's 30, and every one of them
executes its FFT on the fine grid no matter which level asked for it. The
`--counts` view prints that directly, as calls-per-grid alongside FFTs-per-grid;
when those two histograms disagree, the physics is not being coarsened.

Synthetic by default: coil maps, mask and k-space are generated to the shape you
ask for, so this runs without dataset access. Only the shapes matter for timing.

Usage
-----
    # the pair that motivated this, on real data dimensions
    python scripts/profile_mg.py --configs config/brain/mg/mglpds_R4.json \
                                            config/brain/mg/lpdsnet_R4.json \
                                 --size 320 --coils 16

    # measure what Tier 1 bought
    python scripts/profile_mg.py --configs config/brain/mg/mglpds_R4.json --ab

    # where the time goes
    python scripts/profile_mg.py --configs config/brain/mg/mglpds_R4.json \
                                 --breakdown --counts

Reading the output
------------------
`total` is uninstrumented and is the number to quote. The breakdown re-runs with
every FFT / conv / resample call individually timed, so its total is inflated by
the instrumentation -- compare the *proportions*, not its absolute figure.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import models.components as components_mod          # noqa: E402
import models.lpds as lpds_mod                      # noqa: E402
import models.mg_lpds as mg_mod                     # noqa: E402
import operators.base as base_mod                   # noqa: E402
from models import build_model                      # noqa: E402
from operators import FFT2D, Mask, Sense            # noqa: E402
from physics.mask import make_acc_mask              # noqa: E402


# ===========================================================================
#  synthetic measurement
# ===========================================================================
def build_problem(size, coils, batch, R, acs_lines, sigma, device, seed=0):
    """A SENSE measurement of the requested shape.

    The maps are smooth and unit-RSS (which is the assumption `mri_awgn` and
    the DC correction both rely on) and the mask is the same uniform + ACS
    pattern the configs train against. Values are arbitrary -- only shapes,
    dtypes and the sampling density affect timing.
    """
    g = torch.Generator().manual_seed(seed)
    H, W = size

    # smooth coil maps: low-frequency noise, normalised to unit RSS
    lo = torch.randn(batch, coils, 8, 8, dtype=torch.complex64, generator=g)
    smaps = torch.complex(
        F.interpolate(lo.real, size=(H, W), mode="bilinear", align_corners=False),
        F.interpolate(lo.imag, size=(H, W), mode="bilinear", align_corners=False))
    smaps = smaps + 1.0
    smaps = smaps / smaps.abs().pow(2).sum(1, keepdim=True).sqrt()

    mask = make_acc_mask(shape=(H, W), accel=R, acs_lines=acs_lines,
                         mode="uniform", offset=0).reshape(1, 1, H, W).float()

    image = torch.randn(batch, 1, H, W, dtype=torch.complex64, generator=g)

    # Drawn from `g`, not the global RNG: `--ab` builds the problem once per
    # mode, and a global draw would hand the two modes different measurements
    # (which is exactly what the equivalence check at the end would then flag).
    noise = torch.randn(batch, coils, H, W, dtype=torch.complex64, generator=g)

    smaps, mask, image = (t.to(device) for t in (smaps, mask, image))
    E = Mask(mask) @ FFT2D() @ Sense(smaps)
    y = E(image) + sigma * noise.to(device)
    return y, E, sigma


def load_model(cfg_path, device, seed=0):
    with open(cfg_path) as f:
        cfg = json.load(f)
    # Seeded so the two `--ab` modes build bit-identical weights: without this
    # the A/B would compare two different random draws, and the equivalence
    # check below would have nothing to say.
    torch.manual_seed(seed)
    model = build_model(cfg).to(device).eval()
    if getattr(model, "attn_backend", None) == "flex" and device.type == "cuda":
        model.compile_flex()
    p = cfg["model"]["params"]
    label = f"{cfg['model']['type']} K={p.get('K', p.get('denoiser_kws', {}).get('K'))}"
    n_par = sum(q.numel() for q in model.parameters())
    return model, label, n_par


# ===========================================================================
#  Tier-1 toggles  (for --ab)
# ===========================================================================
@contextlib.contextmanager
def tier1_disabled():
    """Restore the pre-optimisation behaviour of every Tier-1 change.

    Each is reverted at its seam rather than by editing the modules, so the
    A/B measures exactly the four changes and nothing else. Note the operator
    must be REBUILT inside this block: `_match_sense_gram` runs in
    `CompositeOperator.__init__`, so an operator constructed outside keeps its
    fused kernel.
    """
    orig_match = base_mod._match_sense_gram
    orig_layer_fwd = lpds_mod.LPDSLayer.forward
    orig_restrict = mg_mod._restrict_measurement
    orig_inplace = components_mod._GaussConvNd.INPLACE_COMBINE

    def no_fusion(ops):
        return None

    def no_handoff(self, state, y_tilde, E=None, sigma=None, pi=None, cache=None):
        if cache is not None:
            cache.pop("_gram_x", None)          # force the recompute
        return orig_layer_fwd(self, state, y_tilde, E=E, sigma=sigma, pi=pi,
                              cache=cache)

    def no_memo(y, E, sigma, R, static):
        return orig_restrict(y, E, sigma, R, None)

    base_mod._match_sense_gram = no_fusion
    lpds_mod.LPDSLayer.forward = no_handoff
    mg_mod._restrict_measurement = no_memo
    components_mod._GaussConvNd.INPLACE_COMBINE = False
    try:
        yield
    finally:
        base_mod._match_sense_gram = orig_match
        lpds_mod.LPDSLayer.forward = orig_layer_fwd
        mg_mod._restrict_measurement = orig_restrict
        components_mod._GaussConvNd.INPLACE_COMBINE = orig_inplace


# ===========================================================================
#  instrumentation
# ===========================================================================
class Patcher:
    """Wrap module-level functions, restoring them on exit."""

    def __init__(self):
        self._saved = []

    def wrap(self, module, name, factory):
        orig = getattr(module, name)
        setattr(module, name, factory(orig, name))
        self._saved.append((module, name, orig))

    def restore(self):
        for module, name, orig in reversed(self._saved):
            setattr(module, name, orig)
        self._saved.clear()


# The leaf families. They never nest inside one another, so their times add up
# and "elementwise / overhead" is an honest remainder. Analysis and synthesis
# are kept apart: with `--by-level` the key is the INPUT grid, and a level's
# analysis reads the image grid while its synthesis reads the latent grid --
# so at stride 2 they would otherwise collide with the next level down.
LEAF_OPS = [
    ("fft", torch.fft, ["fftn", "ifftn", "fft2", "ifft2"]),
    ("fftshift", torch.fft, ["fftshift", "ifftshift"]),
    ("conv (analysis)", F, ["conv2d"]),
    ("convT (synthesis)", F, ["conv_transpose2d"]),
    # NOTE: restrict / prolong are now conv2d / conv_transpose2d against a
    # windowed-sinc kernel, so their calls land in the two buckets above and
    # cannot be split out by op name alone. Use --by-level: a transfer reads
    # the grid it is leaving, so it shows up one level off from the smoother
    # convs it sits between.
    ("pad", F, ["pad"]),
]


def count_run(model, y, E, sigma):
    """Exact call counts and the grid each call ran at. Free; no timing."""
    counts, grids = Counter(), defaultdict(Counter)
    p = Patcher()

    def counter(bucket):
        def factory(orig, name):
            def inner(x, *a, **k):
                counts[bucket] += 1
                if torch.is_tensor(x) and x.dim() >= 2:
                    grids[bucket][tuple(x.shape[-2:])] += 1
                return orig(x, *a, **k)
            return inner
        return factory

    for bucket, module, names in LEAF_OPS:
        for n in names:
            if hasattr(module, n):
                p.wrap(module, n, counter(bucket))

    orig_gram = base_mod.CompositeOperator.gram

    def gram(self, x):
        counts["gram"] += 1
        grids["gram"][tuple(x.shape[-2:])] += 1
        return orig_gram(self, x)

    base_mod.CompositeOperator.gram = gram
    try:
        with torch.no_grad():
            model(y, E=E, sigma=sigma)
    finally:
        base_mod.CompositeOperator.gram = orig_gram
        p.restore()
    return counts, grids


def breakdown_run(model, y, E, sigma, device, by_level=False):
    """Per-family GPU time, via CUDA events (one sync at the end).

    Individually timing a few thousand kernels inflates the total; the split is
    what this is for.

    `by_level` additionally keys each family by the spatial size of its input,
    which is what separates a multigrid level from its parent. A family whose
    time does NOT fall as the grid shrinks is not doing the work you think it
    is -- either the physics is not being coarsened, or the kernel picked for
    that shape is a bad one.
    """
    spans = defaultdict(list)
    totals = defaultdict(float)
    counts = Counter()
    cuda = device.type == "cuda"
    p = Patcher()

    def key_for(bucket, a):
        if by_level and a and torch.is_tensor(a[0]) and a[0].dim() >= 2:
            return f"{bucket} @{a[0].shape[-2]}x{a[0].shape[-1]}"
        return bucket

    def timer(bucket):
        def factory(orig, name):
            if cuda:
                def inner(*a, **k):
                    key = key_for(bucket, a)
                    s = torch.cuda.Event(enable_timing=True)
                    e = torch.cuda.Event(enable_timing=True)
                    s.record()
                    out = orig(*a, **k)
                    e.record()
                    spans[key].append((s, e))
                    counts[key] += 1
                    return out
            else:
                def inner(*a, **k):
                    key = key_for(bucket, a)
                    t0 = time.perf_counter()
                    out = orig(*a, **k)
                    totals[key] += (time.perf_counter() - t0) * 1e3
                    counts[key] += 1
                    return out
            return inner
        return factory

    for bucket, module, names in LEAF_OPS:
        for n in names:
            if hasattr(module, n):
                p.wrap(module, n, timer(bucket))

    try:
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model(y, E=E, sigma=sigma)
        if cuda:
            torch.cuda.synchronize()
        wall = (time.perf_counter() - t0) * 1e3
    finally:
        p.restore()

    if cuda:
        for key, evts in spans.items():
            totals[key] = sum(s.elapsed_time(e) for s, e in evts)
    totals["elementwise / overhead"] = wall - sum(totals.values())
    return dict(totals), dict(counts), wall


def time_run(model, y, E, sigma, reps, warmup, device):
    """Uninstrumented wall clock. This is the number to quote."""
    cuda = device.type == "cuda"
    with torch.no_grad():
        for _ in range(warmup):
            model(y, E=E, sigma=sigma)
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        samples = []
        for _ in range(reps):
            t0 = time.perf_counter()
            model(y, E=E, sigma=sigma)
            if cuda:
                torch.cuda.synchronize()
            samples.append((time.perf_counter() - t0) * 1e3)

    samples.sort()
    peak = torch.cuda.max_memory_allocated() / 2**20 if cuda else float("nan")
    return dict(median=statistics.median(samples), lo=samples[0],
                hi=samples[-1], peak_mb=peak)


# ===========================================================================
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="+", required=True,
                    help="model config JSONs (first is treated as the subject, "
                         "the rest as baselines for the ratio)")
    ap.add_argument("--size", type=int, nargs="+", default=[320],
                    help="H [W]; W defaults to H")
    ap.add_argument("--coils", type=int, default=16)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--R", type=int, default=None,
                    help="acceleration; defaults to each config's mri.R")
    ap.add_argument("--acs-lines", type=int, default=None)
    ap.add_argument("--sigma", type=float, default=0.005)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--ab", action="store_true",
                    help="also run with the Tier-1 Gram optimisations disabled")
    ap.add_argument("--counts", action="store_true",
                    help="exact op counts and the grid each ran at")
    ap.add_argument("--breakdown", action="store_true",
                    help="split GPU time into fft / conv / resample / other")
    ap.add_argument("--by-level", action="store_true",
                    help="with --breakdown, key each family by its input grid "
                         "size, separating the multigrid levels")
    ap.add_argument("--cudnn-benchmark", action="store_true",
                    help="enable cuDNN autotuning (off by default everywhere in "
                         "this repo); shapes are fixed, so warmup pays for it")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    size = (args.size[0], args.size[-1])
    torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)
    if device.type == "cpu":
        print("WARNING: profiling on CPU. The FFT/conv balance and the launch "
              "overhead that drive GPU timings do not transfer -- use a GPU.\n")

    print(f"device   : {device}"
          + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"torch    : {torch.__version__}")
    print(f"problem  : {args.batch}x{args.coils} coils, {size[0]}x{size[1]}, "
          f"sigma={args.sigma}, {args.reps} reps after {args.warmup} warmup")
    print(f"cudnn.benchmark : {torch.backends.cudnn.benchmark}\n")

    rows, first_median, outputs = [], {}, {}
    for cfg_path in args.configs:
        with open(cfg_path) as f:
            mri = json.load(f).get("mri", {})
        R = args.R if args.R is not None else mri.get("R", 4)
        acs = args.acs_lines if args.acs_lines is not None else mri.get("acs_lines", 20)

        name = os.path.splitext(os.path.basename(cfg_path))[0]
        modes = [("optimised", contextlib.nullcontext)]
        if args.ab:
            modes.append(("pre-Tier1", tier1_disabled))

        for mode, ctx in modes:
            with ctx():
                # rebuilt inside the context: the fused-Gram matcher runs in
                # CompositeOperator.__init__
                y, E, sigma = build_problem(size, args.coils, args.batch, R, acs,
                                            args.sigma, device, args.seed)
                model, label, n_par = load_model(cfg_path, device, args.seed)
                stats = time_run(model, y, E, sigma, args.reps, args.warmup, device)
                if args.ab:
                    with torch.no_grad():
                        outputs[(name, mode)] = model(y, E=E, sigma=sigma)[0].clone()
            rows.append((name, label, n_par, mode, stats))
            first_median.setdefault((name, mode), stats["median"])

            if args.counts and mode == "optimised":
                counts, grids = count_run(model, y, E, sigma)
                print(f"--- counts: {name} ---")
                for k in ("gram", "fft", "fftshift", "conv", "resample"):
                    if counts.get(k):
                        print(f"    {k:<10} {counts[k]:>6}")
                for k in ("gram", "fft"):
                    if grids.get(k):
                        hist = dict(sorted(grids[k].items(), reverse=True))
                        print(f"    {k} by grid: {hist}")
                print("    (if `gram by grid` spans several sizes but `fft by grid` "
                      "does not,\n     the coarse levels are running full-resolution "
                      "physics)\n")

            if args.breakdown and mode == "optimised":
                totals, ncalls, wall = breakdown_run(model, y, E, sigma, device,
                                                     by_level=args.by_level)
                clean = stats["median"]
                print(f"--- breakdown: {name}  (instrumented {wall:.1f} ms vs "
                      f"clean {clean:.1f} ms; the {wall - clean:.1f} ms of wrapper "
                      f"overhead lands in the remainder) ---")
                head = f"    {'family':<28}{'ms':>9}{'%':>7}{'calls':>8}{'us/call':>10}"
                print(head)
                for k, v in sorted(totals.items(), key=lambda kv: -kv[1]):
                    n = ncalls.get(k, 0)
                    per = f"{1e3 * v / n:>10.1f}" if n else " " * 10
                    print(f"    {k:<28}{v:>9.2f}{100 * v / wall:>7.1f}"
                          f"{n if n else '':>8}{per}")
                print()

    print(f"{'config':<16}{'model':<28}{'params':>10}  {'mode':<11}"
          f"{'median':>9}{'min':>8}{'max':>8}{'peak MB':>10}")
    print("-" * 100)
    for name, label, n_par, mode, s in rows:
        print(f"{name:<16}{label:<28}{n_par:>10,}  {mode:<11}"
              f"{s['median']:>8.2f}{s['lo']:>8.2f}{s['hi']:>8.2f}{s['peak_mb']:>10.0f}")

    subject = os.path.splitext(os.path.basename(args.configs[0]))[0]
    for mode in ("optimised", "pre-Tier1"):
        base = [(n, m) for (n, m) in first_median if m == mode and n != subject]
        if (subject, mode) in first_median and base:
            print()
            for n, m in base:
                r = first_median[(subject, mode)] / first_median[(n, m)]
                print(f"  {mode:<11} {subject} / {n} = {r:.2f}x")

    if args.ab:
        print()
        for name in dict.fromkeys(r[0] for r in rows):
            a, b = first_median.get((name, "pre-Tier1")), first_median.get((name, "optimised"))
            if a and b:
                print(f"  Tier-1 speedup  {name:<16} {a / b:.2f}x   "
                      f"({a:.2f} -> {b:.2f} ms)")

        # The optimisations are meant to be exact, not approximate. Re-check
        # that here rather than trusting it: same seed, so the two modes differ
        # only by the four changes.
        print()
        for name in dict.fromkeys(r[0] for r in rows):
            u, v = outputs.get((name, "optimised")), outputs.get((name, "pre-Tier1"))
            if u is None or v is None:
                continue
            err = ((u - v).abs().max() / (v.abs().max() + 1e-12)).item()
            verdict = ("bit-identical" if err == 0.0 else
                       "exact to fp roundoff" if err < 1e-6 else "MISMATCH")
            print(f"  equivalence     {name:<16} max rel err = {err:.3e}  {verdict}")


if __name__ == "__main__":
    main()
