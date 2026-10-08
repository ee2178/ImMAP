#!/usr/bin/env python3
"""
FLOPs per forward pass, counted from the operations a network actually runs.

    python scripts/count_flops.py --configs config/brain/mg/lpdsnet_R12.json \\
        config/brain/mg/varnetmaps_R12.json config/brain/mg/mg81v6_R12.json --group

Builds each config's model, runs ONE inference forward on a synthetic SENSE
problem of the requested shape, and counts three families as they execute:

    conv        every `F.conv2d` / `F.conv_transpose2d` -- the dictionaries, the
                grid transfers, VarNet's U-Nets, the group prox's 1x1 maps.
                2 FLOPs per multiply-accumulate (MAC).
    FFT         every `torch.fft` transform: 5 n log2(n) per 2-D transform of n
                points, times the number of transforms in the call (coils).
    attention   the group prox's windowed adjacency: per application,
                pixels x window^2 x (query/key dims + value channels) MACs,
                2 FLOPs each.

That is the convention of the GFLOP figures already quoted in this repo
(scripts/make_mg_recon_configs.py: "conv + 5 n log2 n per FFT"); at 160 x 160
and 16 coils this script reproduces them (lpdsnet 21, mglpds 34).

What it does NOT count: elementwise arithmetic, thresholding, padding, norms.
Standard FLOP counters leave those out too, and for these nets that is the
point to remember when reading the numbers against measured time -- the
unrolled nets spend most of their time in exactly what is not counted here
(see scripts/profile_mg.py --ops / --host). FLOPs rank the arithmetic, not the
wall clock.

Three things that change a count without changing the network:

  * `training.complex_conv`. A complex conv runs as three real convs ("gauss")
    or as one real conv on stacked [re; im] channels ("planar"), which is four
    convs' worth of MACs. Same map, 4:3 in counted FLOPs. Counted as executed.
  * `coarse_op`. Galerkin coarse levels run their FFTs at FULL resolution,
    rediscretized ones on the coarse grids. The conv column moves as well: a
    Galerkin coarse Gram prolongs and restricts around the fine operator, and
    those grid transfers are convolutions.
  * the attention backend. `flex` and `triton` recompute the similarity on
    every application; `gather` computes it once per adjacency build. Counted
    as the config's backend would run it -- and as the ALGORITHM costs: flex
    additionally scores every key in each active 128-wide block before
    masking, which is executed work this does not include.

The attention itself is never executed here: the adjacency is replaced by a
stand-in that records its shapes and returns its input, so the group nets count
on a CPU and without FlexAttention. Everything else runs for real -- unless
--shapes-only, under which each convolution returns zeros of the shape the real
one would (taken from torch itself, by running it on one channel). Counts are
identical (tests/test_count_flops.py) and the convolution arithmetic is
skipped; model construction and the rest of the forward still run.

--group adds, for every MGLPDSNet config, its group-prox twin: the same
parameters as an `MGGroupLPDS`, with the group settings of the generator's
`mggrouplpds` cell (window, Mh, dK, heads, similarity, backend).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FLOPS_PER_MAC = 2
FFT_FLOPS_PER_NLOGN = 5


_CONV_ARGS = ("bias", "stride", "padding", "dilation", "groups")
_CONVT_ARGS = ("bias", "stride", "padding", "output_padding", "groups", "dilation")


def _shape_only(orig, x, weight, names, a, k, transposed):
    """Zeros shaped like `orig(x, weight, ...)`, without running it.

    The spatial size comes from the real op on ONE input and one output
    channel -- torch's own arithmetic, so padding / stride / output_padding
    cannot be mis-derived here -- and the channel count from the weight.
    """
    kw = dict(zip(names, a))
    kw.update(k)
    groups = int(kw.get("groups", 1) or 1)
    kw.update(bias=None, groups=1)
    probe = orig(x[:1, :1], weight[:1, :1], **kw)
    cout = weight.shape[1] * groups if transposed else weight.shape[0]
    return x.new_zeros((x.shape[0], cout) + tuple(probe.shape[-2:]))


class FlopCounter:
    """Count conv / FFT / attention work between `__enter__` and `__exit__`."""

    def __init__(self, shapes_only=False):
        self.shapes_only = bool(shapes_only)
        self.flops = Counter()            # family -> FLOPs
        self.calls = Counter()            # family -> calls
        self.fft_grids = Counter()        # (H, W) -> 2-D transforms
        self.attn_grids = Counter()       # (H, W) -> applications
        self._saved = []

    # -- patching -------------------------------------------------------------
    def _wrap(self, owner, name, make):
        orig = getattr(owner, name)
        setattr(owner, name, make(orig))
        self._saved.append((owner, name, orig))

    def __enter__(self):
        import models.prox as prox_mod
        from models.circulant_flex import FlexAdjacency

        counter = self

        def conv(orig):
            def inner(x, weight, *a, **k):
                out = (_shape_only(orig, x, weight, _CONV_ARGS, a, k, False)
                       if counter.shapes_only else orig(x, weight, *a, **k))
                # weight (Cout, Cin/groups, kh, kw): MACs per OUTPUT element
                counter.flops["conv"] += FLOPS_PER_MAC * out.numel() * weight[0].numel()
                counter.calls["conv"] += 1
                return out
            return inner

        def conv_t(orig):
            def inner(x, weight, *a, **k):
                out = (_shape_only(orig, x, weight, _CONVT_ARGS, a, k, True)
                       if counter.shapes_only else orig(x, weight, *a, **k))
                # weight (Cin, Cout/groups, kh, kw): MACs per INPUT element
                counter.flops["conv"] += FLOPS_PER_MAC * x.numel() * weight[0].numel()
                counter.calls["conv"] += 1
                return out
            return inner

        def fft(default_dims):
            def make(orig):
                def inner(x, *a, **k):
                    out = orig(x, *a, **k)
                    dim = k.get("dim", a[1] if len(a) > 1 else None)
                    if dim is None:
                        dim = default_dims(x)
                    dim = (dim,) if isinstance(dim, int) else tuple(dim)
                    n = 1
                    for d in dim:
                        n *= x.shape[d]
                    transforms = x.numel() // max(n, 1)
                    counter.flops["fft"] += (FFT_FLOPS_PER_NLOGN * n * math.log2(max(n, 2))
                                             * transforms)
                    counter.calls["fft"] += 1
                    if len(dim) == 2:
                        counter.fft_grids[tuple(x.shape[d] for d in dim)] += transforms
                    return out
                return inner
            return make

        self._wrap(F, "conv2d", conv)
        self._wrap(F, "conv_transpose2d", conv_t)
        for name in ("fftn", "ifftn"):
            self._wrap(torch.fft, name, fft(lambda x: tuple(range(x.dim()))))
        for name in ("fft2", "ifft2"):
            self._wrap(torch.fft, name, fft(lambda x: (-2, -1)))

        class Recording(FlexAdjacency):
            """Stands in for the adjacency: records one application's cost and
            returns its input. A FlexAdjacency subclass so `GroupThreshold`
            takes the fused-backend branches (no blend, real values only)."""

            def __init__(self, pixels, grid, window2, d_qk, per_apply_scores):
                self._c = (pixels, grid, window2, d_qk, per_apply_scores)

            def apply(self, x, transpose=False):
                pixels, grid, w2, d_qk, per_apply = self._c
                macs = pixels * w2 * (x.shape[1] + (d_qk if per_apply else 0))
                counter.flops["attention"] += FLOPS_PER_MAC * macs
                counter.calls["attention"] += 1
                counter.attn_grids[grid] += 1
                return x

        def build_gamma(orig):
            def inner(self, z, sigma):
                q, _k = self._scaled_qk(z, sigma)        # W_theta / W_phi run (and count)
                B, D, H, W = q.shape
                d_qk = D * (2 if q.is_complex() else 1)  # complex features stack [re; im]
                w2 = self.window ** 2
                fused = self.attn_backend in ("flex", "triton")
                if not fused:                            # gather: scores once per build
                    counter.flops["attention"] += FLOPS_PER_MAC * B * H * W * w2 * d_qk
                rec = object.__new__(Recording)
                rec.__init__(B * H * W, (H, W), w2, d_qk, fused)
                return rec
            return inner

        self._wrap(prox_mod.GroupThreshold, "_build_gamma", build_gamma)
        return self

    def __exit__(self, *exc):
        for owner, name, orig in reversed(self._saved):
            setattr(owner, name, orig)
        self._saved.clear()

    @property
    def total(self):
        return sum(self.flops.values())


def count_config(cfg, name, hw, coils, R, device, shapes_only=False):
    """-> one record for a config dict (already carrying any overrides)."""
    from models import build_model
    from models.components import set_complex_mode
    from scripts.time_net import build_problem, _import_repo
    from training.common import embed_for_net
    import scripts.time_net as tn

    if not hasattr(tn, "Mask"):
        _import_repo()
    mode = set_complex_mode(cfg.get("training", {}).get("complex_conv") or "gauss")
    torch.manual_seed(0)
    net = build_model(cfg).to(device).eval()
    mri = cfg.get("mri", {})
    R = R if R is not None else mri.get("R", 8)
    y, E0, image = build_problem(hw, coils, mri, R, device)
    E, T = embed_for_net(net, E0, image, None)
    sigma = torch.full((1, 1, 1, 1), 0.01, device=device)

    with torch.no_grad(), FlopCounter(shapes_only) as c:
        net(y, E=E, sigma=sigma)

    p = cfg["model"]["params"]
    K = p.get("K", p.get("denoiser_kws", {}).get("K"))
    if K is None and "num_cascades" in p:
        K = f"{p['num_cascades']}casc"
    return dict(name=name, type=cfg["model"]["type"], K=K, M=p.get("M"),
                params=sum(q.numel() for q in net.parameters()),
                complex_conv=mode, coarse_op=p.get("coarse_op"),
                window=p.get("window"), Mh=p.get("Mh"),
                attn_backend=p.get("attn_backend"), embedded=not T.is_identity,
                flops={k: float(v) for k, v in c.flops.items()}, total=float(c.total),
                calls=dict(c.calls),
                fft_grids={f"{h}x{w}": int(n) for (h, w), n in sorted(c.fft_grids.items(),
                                                                     reverse=True)},
                attn_grids={f"{h}x{w}": int(n) for (h, w), n in sorted(c.attn_grids.items(),
                                                                      reverse=True)})


def group_twin(cfg):
    """The same net with the group prox in the prox slot, or None.

    Only for `MGLPDSNet` configs. The group settings are the generator's
    `mggrouplpds` cell's -- whatever that cell adds to `mglpds`.
    """
    if cfg["model"]["type"] != "MGLPDSNet":
        return None
    from scripts.make_mg_recon_configs import MODELS

    base, grp = MODELS["mglpds"]["params"], MODELS["mggrouplpds"]["params"]
    extra = {k: v for k, v in grp.items() if k not in base}
    twin = json.loads(json.dumps(cfg))
    twin["model"] = dict(type=MODELS["mggrouplpds"]["type"],
                         params=dict(cfg["model"]["params"], **extra))
    return twin


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--group", action="store_true",
                    help="also count each MGLPDSNet config's group-prox twin")
    ap.add_argument("--size", type=int, nargs="+", default=[640, 320], help="H [W]")
    ap.add_argument("--coils", type=int, default=20)
    ap.add_argument("--R", type=int, default=None, help="default: each config's mri.R")
    ap.add_argument("--shapes-only", action="store_true",
                    help="convolutions return zeros of the right shape instead of "
                         "running: identical counts without the conv arithmetic")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--device", default="cpu",
                    help="counts do not depend on it; cpu needs no GPU allocation")
    args = ap.parse_args()

    device = torch.device(args.device)
    hw = (args.size[0], args.size[-1])
    todo = []
    for path in args.configs:
        with open(path) as f:
            cfg = json.load(f)
        name = os.path.splitext(os.path.basename(path))[0]
        todo.append((name, cfg))
    if args.group:
        for name, cfg in list(todo):
            twin = group_twin(cfg)
            if twin is not None:
                todo.append((name + "+group", twin))

    print(f"{hw[0]}x{hw[1]}, {args.coils} coils | one inference forward | "
          f"conv {FLOPS_PER_MAC} FLOPs/MAC, FFT {FFT_FLOPS_PER_NLOGN} n log2 n, "
          f"attention {FLOPS_PER_MAC} FLOPs/MAC | elementwise work NOT counted\n",
          flush=True)
    rows = []
    for name, cfg in todo:
        rows.append(count_config(cfg, name, hw, args.coils, args.R, device,
                                 shapes_only=args.shapes_only))
        print(f"  counted {name}", flush=True)

    base = rows[0]["total"]
    G = 1e9
    head = (f"{'config':<22}{'K':<18}{'M':>4}{'params':>12} {'conv':<7}{'coarse':<13}"
            f"{'conv':>9}{'FFT':>8}{'attn':>9}{'total':>9}{'x ' + rows[0]['name']:>16}")
    print("\n" + head + "    (GFLOPs)")
    print("-" * len(head))
    for r in rows:
        f = r["flops"]
        print(f"{r['name']:<22}{str(r['K']):<18}{str(r['M'] or ''):>4}{r['params']:>12,} "
              f"{r['complex_conv']:<7}{str(r['coarse_op'] or '-'):<13}"
              f"{f.get('conv', 0) / G:>9.1f}{f.get('fft', 0) / G:>8.1f}"
              f"{f.get('attention', 0) / G:>9.1f}{r['total'] / G:>9.1f}"
              f"{r['total'] / base:>15.2f}x" + ("   [embedded]" if r["embedded"] else ""))
    print("\nFFT transforms by grid (2-D transforms per forward; coils included):")
    for r in rows:
        print(f"  {r['name']:<22}{r['fft_grids']}")
    if any(r["attn_grids"] for r in rows):
        print("\nattention applications by latent grid:")
        for r in rows:
            if r["attn_grids"]:
                print(f"  {r['name']:<22}{r['attn_grids']}   window {r['window']}, "
                      f"Mh {r['Mh']}, {r['attn_backend']}")
    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w") as fh:
            json.dump(dict(size=list(hw), coils=args.coils, flops_per_mac=FLOPS_PER_MAC,
                           fft_flops_per_nlogn=FFT_FLOPS_PER_NLOGN, rows=rows), fh, indent=1)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
