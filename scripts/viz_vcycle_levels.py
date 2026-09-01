#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualise what a trained V-cycle carries at EVERY grid level.

    python scripts/viz_vcycle_levels.py \
        --config trained_nets/mg_recon/knee/mggrouplpds_R8/config.gen.json \
        --ckpt   trained_nets/mg_recon/knee/mggrouplpds_R8/net.ckpt \
        --out    figs/vcycle_levels.png

Every panel is on its OWN grid at its own resolution -- the point is to see the
hierarchy, not a set of upsampled thumbnails, so nothing is resampled for
display.

What is captured, per outer V-cycle and per level
-------------------------------------------------
    x        the primal iterate ENTERING that level (image grid, restricted)
    x_c      what the level below receives  (= R x)
    pi_x     the FAS correction formed for that level
    corr     the prolonged coarse correction, alpha_x P (wx_c - x_c), i.e. what
             the level below actually contributed back

`corr` is the panel worth staring at: it is the ONLY thing the coarse grid adds
to the fine solution. If it is visually structureless, or dominated by boundary
ringing, the coarse level is not paying for itself and the V-cycle is an
expensive way to run a flat stack.

Display window
--------------
Magnitudes share one window per ROW (per quantity) so levels are comparable
down a column; corrections get a symmetric diverging window around zero,
since their sign is the whole content. `--per-panel` breaks that if you would
rather see faint levels at full contrast, but then do not compare panels.

No training, no gradients, one forward.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def _load_ckpt_fn():
    try:
        from training.common import load_ckpt
        return load_ckpt
    except ImportError:
        # training/__init__ imports torchvision, which a bare checkout may not have,
        # and training/common.py in turn imports training.config_io -- so loading
        # common.py by path is not enough on its own. Register a STUB `training`
        # package first, so the submodule imports inside it resolve without ever
        # running the real __init__.
        import importlib.util as u
        import types
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if "training" not in sys.modules:
            pkg = types.ModuleType("training")
            pkg.__path__ = [os.path.join(root, "training")]
            sys.modules["training"] = pkg
        sp = u.spec_from_file_location(
            "training.common", os.path.join(root, "training", "common.py"))
        m = u.module_from_spec(sp)
        sys.modules["training.common"] = m
        sp.loader.exec_module(m)
        return m.load_ckpt


def build_inputs(cfg, H, W, coils, device, seed=0):
    """A measurement consistent with the config's own physics."""
    from operators import FFT2D, Mask, Sense
    from physics.mask import make_acc_mask

    mri = cfg["mri"]
    g = torch.Generator().manual_seed(seed)
    sm = torch.randn(1, coils, H, W, generator=g, dtype=torch.complex64)
    sm = (sm / (sm.abs().pow(2).sum(1, keepdim=True).sqrt() + 1e-8)).to(device)
    mask = make_acc_mask(shape=(H, W), accel=mri["R"], acs_lines=mri["acs_lines"],
                         mode=mri.get("mask_dist", "uniform"))
    mask = mask.reshape(1, 1, H, W).to(torch.complex64).to(device)
    E = Mask(mask) @ FFT2D() @ Sense(sm)

    yy, xx = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W),
                            indexing="ij")
    x0 = (((xx ** 2 + yy ** 2) < 0.55).float()
          + 0.45 * ((xx ** 2 + (yy - 0.25) ** 2) < 0.12).float()
          + 0.30 * ((xx.abs() < 0.5) & (yy.abs() < 0.06)).float())
    x0 = (x0 + 0.02 * torch.randn(H, W, generator=g)).reshape(1, 1, H, W)
    x0 = x0.to(torch.complex64).to(device)
    return E, x0, E(x0)


def capture(net, y, E, sigma):
    """Run one forward, recording (outer, level) -> tensors via hooks."""
    from models.lpds import LPDSStack
    from models.mg_lpds import PDVCycle, PDObjectiveDownsample

    rec = {}
    handles = []
    # depth of each PDVCycle = its level index counted from the finest
    level_of = {}

    def depth_of(mod):
        d = 0
        m = mod
        while isinstance(getattr(m, "mglayer", None), PDVCycle):
            m = m.mglayer
            d += 1
        return d

    vcycles = [m for m in net.modules() if isinstance(m, PDVCycle)]
    maxd = max((depth_of(v) for v in vcycles), default=0)
    for v in vcycles:
        level_of[id(v)] = maxd - depth_of(v)

    counter = {"outer": -1}

    def vc_pre(mod, args):
        lvl = level_of[id(mod)]
        if lvl == 0:
            counter["outer"] += 1
        state = args[0]
        # The OUTERMOST cycle's first call is the cold start (state=None): there is
        # no incoming x yet, the layer sets x = y~ itself. Record nothing rather
        # than inventing one; later outer cycles have it.
        if isinstance(state, tuple) and torch.is_tensor(state[0]):
            rec.setdefault((counter["outer"], lvl), {})["x"] = state[0].detach()

    def vc_post(mod, args, out):
        lvl = level_of[id(mod)]
        st = out[0] if isinstance(out, tuple) else out
        if isinstance(st, tuple) and torch.is_tensor(st[0]):
            rec.setdefault((counter["outer"], lvl), {})["x_out"] = st[0].detach()

    def df_post(mod, args, out):
        # PDObjectiveDownsample -> (x_c, z_c, pi_x, pi_z, y_c, E_c, sigma_c)
        vals = [t for t in out if torch.is_tensor(t)]
        owner = getattr(mod, "_viz_level", None)
        if owner is None:
            return
        d = rec.setdefault((counter["outer"], owner), {})
        if len(vals) >= 1:
            d["x_c"] = vals[0].detach()
        if len(vals) >= 3:
            d["pi_x"] = vals[2].detach()

    # The COARSEST level is an LPDSStack, not a PDVCycle, so the loop above never
    # sees it -- and it is a resolution worth looking at. Hook it as level maxd+1.
    def stack_pre(mod, args):
        state = args[0]
        if isinstance(state, tuple) and torch.is_tensor(state[0]):
            rec.setdefault((counter["outer"], maxd + 1), {})["x"] = state[0].detach()

    def stack_post(mod, args, out):
        st = out[0] if isinstance(out, tuple) else out
        if isinstance(st, tuple) and torch.is_tensor(st[0]):
            rec.setdefault((counter["outer"], maxd + 1), {})["x_out"] = st[0].detach()

    for v in vcycles:
        if isinstance(getattr(v, "mglayer", None), LPDSStack):
            handles.append(v.mglayer.register_forward_pre_hook(stack_pre))
            handles.append(v.mglayer.register_forward_hook(stack_post,
                                                           with_kwargs=False))
    for v in vcycles:
        handles.append(v.register_forward_pre_hook(vc_pre))
        handles.append(v.register_forward_hook(vc_post, with_kwargs=False))
        if isinstance(getattr(v, "dF", None), PDObjectiveDownsample):
            v.dF._viz_level = level_of[id(v)]
            handles.append(v.dF.register_forward_hook(df_post, with_kwargs=False))

    with torch.no_grad():
        net(y, E=E, sigma=sigma)
    for h in handles:
        h.remove()

    # the coarse contribution: what level l+1 gave back to level l
    for (o, l), d in list(rec.items()):
        nxt = rec.get((o, l + 1))
        if nxt is not None and "x_out" in nxt and "x_c" in d:
            if nxt["x_out"].shape == d["x_c"].shape:
                d["coarse_delta"] = (nxt["x_out"] - d["x_c"])
    return rec


def draw(rec, outer, rows, out_path, per_panel=False, cmap="gray"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    levels = sorted({l for (o, l) in rec if o == outer})
    if not levels:
        raise SystemExit(f"[viz] no records for outer cycle {outer}.")
    rows = [r for r in rows
            if any(r in rec.get((outer, l), {}) for l in levels)]
    if not rows:
        raise SystemExit("[viz] none of the requested quantities were captured.")

    fig, ax = plt.subplots(len(rows), len(levels),
                           figsize=(3.1 * len(levels), 3.0 * len(rows)),
                           squeeze=False)
    DIVERGING = {"pi_x", "coarse_delta"}
    for i, key in enumerate(rows):
        imgs = []
        for l in levels:
            t = rec.get((outer, l), {}).get(key)
            imgs.append(None if t is None else
                        (t.abs() if t.is_complex() else t)[0, 0].float().cpu())
        finite = [im for im in imgs if im is not None]
        if key in DIVERGING:
            v = max(float(im.abs().max()) for im in finite) or 1.0
            kw = dict(cmap="bwr", vmin=-v, vmax=v)
        else:
            v = max(float(im.max()) for im in finite) or 1.0
            kw = dict(cmap=cmap, vmin=0.0, vmax=v)
        for j, (l, im) in enumerate(zip(levels, imgs)):
            a = ax[i][j]
            a.set_xticks([]); a.set_yticks([])
            if im is None:
                a.text(.5, .5, "n/a", ha="center", va="center", fontsize=9,
                       color="0.6"); continue
            kk = dict(kw)
            if per_panel:
                kk["vmax"] = float(im.abs().max()) or 1.0
                if key in DIVERGING:
                    kk["vmin"] = -kk["vmax"]
            a.imshow(im, **kk, interpolation="nearest", aspect="equal")
            if i == 0:
                a.set_title(f"level {l}   {tuple(im.shape)}", fontsize=10)
            if j == 0:
                a.set_ylabel(key, fontsize=11)
    fig.suptitle(f"V-cycle level by level -- outer cycle {outer}"
                 + ("   (per-panel contrast)" if per_panel else
                    "   (shared window per row)"), fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"[viz] wrote {out_path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="the run's config.gen.json")
    ap.add_argument("--ckpt", default=None, help="net.ckpt; omit for an untrained net")
    ap.add_argument("--out", default="figs/vcycle_levels.png")
    ap.add_argument("--img", default=None, help="HxW; default = the config's pad stride x 40")
    ap.add_argument("--coils", type=int, default=15)
    ap.add_argument("--sigma", type=float, default=None,
                    help="default = the config's val_noise_std")
    ap.add_argument("--outer", type=int, default=-1,
                    help="which outer V-cycle to draw; -1 = the last, which is both "
                         "the most converged and the one whose level-0 x exists "
                         "(outer 0 is the cold start, where x is set to y~ inside "
                         "the layer rather than passed in)")
    ap.add_argument("--rows", default="x,x_c,pi_x,coarse_delta")
    ap.add_argument("--per-panel", action="store_true",
                    help="rescale every panel independently (breaks comparability)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    from models import build_model

    cfg = json.load(open(args.config))
    params = cfg["model"]["params"]
    K = params.get("K")
    if K is None or isinstance(K, int):
        raise SystemExit(f"[viz] K={K!r} is a FLAT stack -- there are no levels to show.")
    pad = params["s"] * 2 ** (len(K[1]) - 1)

    if args.img:
        H, W = (int(v) for v in args.img.lower().split("x"))
    else:
        H = W = pad * 40
    if H % pad or W % pad:
        raise SystemExit(f"[viz] {H}x{W} is not a multiple of pad_stride={pad}.")

    dev = torch.device(args.device)
    net = build_model(cfg).to(dev).eval()
    if args.ckpt:
        _load_ckpt_fn()(args.ckpt, model=net, device=dev)
        print(f"[viz] loaded {args.ckpt}")
    else:
        print("[viz] NO --ckpt given: this is an UNTRAINED net. The panels show "
              "what the architecture does at initialisation, not what it learned.")

    sigma = args.sigma if args.sigma is not None else \
        cfg.get("training", {}).get("val_noise_std", 0.005)
    E, x0, y = build_inputs(cfg, H, W, args.coils, dev)
    print(f"[viz] {H}x{W}, {args.coils} coils, R={cfg['mri']['R']}, sigma={sigma}")

    rec = capture(net, y, E, sigma)
    got = sorted({(o, l) for (o, l) in rec})
    outers = sorted({o for o, _ in got})
    if args.outer < 0:
        args.outer = outers[args.outer]
    print(f"[viz] captured {len(got)} (outer, level) records; "
          f"levels={sorted({l for _, l in got})}, "
          f"outer cycles={sorted({o for o, _ in got})}")
    for l in sorted({l for _, l in got}):
        d = rec.get((args.outer, l), {})
        shapes = {k: tuple(v.shape[-2:]) for k, v in d.items()}
        print(f"        level {l}: {shapes}")

    draw(rec, args.outer, [r.strip() for r in args.rows.split(",")],
         args.out, per_panel=args.per_panel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
