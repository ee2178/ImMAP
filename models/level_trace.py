# -*- coding: utf-8 -*-
"""
`forward_levels(...)` -- a forward pass that also hands back every intermediate,
keyed by (outer cycle, grid level).

    x_hat, trace = net.forward_levels(y, E=E, sigma=sigma)
    trace.summary()
    trace.get(outer=-1, level=1, key="coarse_delta_x")

Both multigrid families expose it and return the same `LevelTrace` container:

    models/multigrid.py::MGCDLNet   -- one variable, z          (VCycle)
    models/mg_lpds.py::MGLPDSNet    -- the pair (x, z)           (PDVCycle)
                                       `MGGroupLPDS` is this class with a
                                       group prox, so it comes along for free.

Why hooks rather than a `trace=` argument threaded through forward
------------------------------------------------------------------
`PDVCycle.forward` and `LPDSStack.forward` are deliberately interchangeable --
either can be a level's `mglayer` -- so a new keyword would have to be added to
both, and to `VCycle` / `LISTA` on the CDLNet side, and then branched on inside
the hot loop of an 84-layer unroll. Hooks cost nothing when they are not
installed, which is every training step. `forward_levels` installs them, runs
one forward under `no_grad`, and removes them in a `finally`, so an exception
mid-forward cannot leave a model instrumented.

What gets recorded
------------------
Per (outer, level), whichever of these exist for that family:

    x, z                incoming iterate at that level     (None on the cold
                        start, where the layer sets x = y~ itself)
    x_pre, z_pre        after the PRE-smoother: what is restricted down
    x_corr, z_corr      after the coarse correction: what the POST-smoother
                        starts from              (these two: MGLPDSNet only)
    x_out, z_out        outgoing iterate at that level
    x_c, z_c            what the level BELOW receives (= R x, widen_z R z)
    pi_x, pi_z          the FAS correction formed for the level below
    coarse_delta_x      (w_c - x_c): what the level below actually gave back,
    coarse_delta_z      BEFORE alpha and prolongation. This is the only thing
                        the coarse grid contributes; if it is structureless the
                        V-cycle is an expensive flat stack.

`coarse_delta_*` lives on the COARSE grid (it is differenced before prolong),
so it has the shape of level l+1 while being attributed to level l.

The coarsest level is an `LPDSStack` / `LISTA`, not a V-cycle, so it has no
`dF`, no `pi` and no `coarse_delta` -- only `x`/`z` in and out. It is recorded
as one level past the deepest V-cycle.
"""

from __future__ import annotations

import torch


class LevelTrace:
    """Intermediates from one forward, keyed by `(outer, level)`."""

    def __init__(self, records, n_levels, family):
        self.records = records
        self.n_levels = n_levels
        self.family = family

    # -- access -------------------------------------------------------------
    @property
    def levels(self):
        return sorted({l for _, l in self.records})

    @property
    def outers(self):
        return sorted({o for o, _ in self.records})

    def _norm_outer(self, outer):
        return self.outers[outer] if outer < 0 else outer

    def at(self, outer=-1, level=0):
        """Every recorded tensor for one (outer, level), as a dict."""
        return self.records.get((self._norm_outer(outer), level), {})

    def get(self, outer=-1, level=0, key="x"):
        return self.at(outer, level).get(key)

    def keys(self, level=None):
        out = set()
        for (o, l), d in self.records.items():
            if level is None or l == level:
                out |= set(d)
        return sorted(out)

    def grid(self, outer=-1, level=0, key=None):
        d = self.at(outer, level)
        t = d.get(key) if key else next(iter(d.values()), None)
        return None if t is None else tuple(t.shape[-2:])

    # -- reporting ----------------------------------------------------------
    def summary(self):
        lines = [f"LevelTrace({self.family}): {len(self.outers)} outer cycles, "
                 f"{len(self.levels)} levels"]
        o = self.outers[-1] if self.outers else 0
        for l in self.levels:
            d = self.records.get((o, l), {})
            shapes = ", ".join(f"{k}{tuple(v.shape[-2:])}" for k, v in sorted(d.items()))
            lines.append(f"  outer {o}, level {l}: {shapes or '(nothing)'}")
        return "\n".join(lines)

    def __repr__(self):
        return (f"LevelTrace(family={self.family!r}, outers={self.outers}, "
                f"levels={self.levels})")


def _level_map(net, vcycle_cls):
    """`id(vcycle) -> level index`, counting from the finest grid down."""
    def depth(mod):
        d, m = 0, mod
        while isinstance(getattr(m, "mglayer", None), vcycle_cls):
            m, d = m.mglayer, d + 1
        return d
    vcs = [m for m in net.modules() if isinstance(m, vcycle_cls)]
    if not vcs:
        return {}, -1, []
    maxd = max(depth(v) for v in vcs)
    return {id(v): maxd - depth(v) for v in vcs}, maxd, vcs


def _as_state(obj):
    """`(x, z)` for the primal-dual family, `(None, z)` for the CDLNet one."""
    if isinstance(obj, tuple) and len(obj) == 2 and all(
            t is None or torch.is_tensor(t) for t in obj):
        return obj
    if torch.is_tensor(obj):
        return None, obj
    return None, None


def trace_forward(net, *args, **kwargs):
    """Run `net` once with instrumentation. -> `(output, LevelTrace)`.

    Positional / keyword arguments are passed to `net.forward` unchanged.
    """
    from models.lista import LISTA
    from models.lpds import LPDSStack
    from models.mg_lpds import PDObjectiveDownsample, PDVCycle
    from models.multigrid import ObjectiveDownsample, VCycle

    lvl_pd, maxd_pd, vcs_pd = _level_map(net, PDVCycle)
    lvl_vc, maxd_vc, vcs_vc = _level_map(net, VCycle)
    if vcs_pd:
        family, lvl, maxd, vcs = "MGLPDSNet", lvl_pd, maxd_pd, vcs_pd
        leaf_cls, df_cls = LPDSStack, PDObjectiveDownsample
    elif vcs_vc:
        family, lvl, maxd, vcs = "MGCDLNet", lvl_vc, maxd_vc, vcs_vc
        leaf_cls, df_cls = LISTA, ObjectiveDownsample
    else:
        raise TypeError(
            f"{type(net).__name__} contains no VCycle / PDVCycle, so it has no "
            f"grid levels to trace. `K` must be [K_outer, [i0, i1, ...]]; a bare "
            f"int builds a flat stack.")

    rec, handles = {}, []
    ctr = {"outer": -1}

    def slot(level):
        return rec.setdefault((ctr["outer"], level), {})

    def pre(mod, args_):
        level = lvl[id(mod)]
        if level == 0:
            ctr["outer"] += 1
        x, z = _as_state(args_[0])
        d = slot(level)
        if x is not None:
            d["x"] = x.detach()
        if z is not None:
            d["z"] = z.detach()

    def post(mod, args_, out):
        x, z = _as_state(out[0] if isinstance(out, tuple) else out)
        d = slot(lvl[id(mod)])
        if x is not None:
            d["x_out"] = x.detach()
        if z is not None:
            d["z_out"] = z.detach()

    def df_post(mod, args_, out):
        level = getattr(mod, "_trace_level", None)
        if level is None:
            return
        d = slot(level)
        if isinstance(mod, PDObjectiveDownsample):
            # (x_c, z_c, pi_x, pi_z, y_c, E_c, sigma_c, gram_x_c)
            for name, i in (("x_c", 0), ("z_c", 1), ("pi_x", 2), ("pi_z", 3)):
                if i < len(out) and torch.is_tensor(out[i]):
                    d[name] = out[i].detach()
        else:
            # (z_c, pi, y_c, E_c, sigma_c)
            for name, i in (("z_c", 0), ("pi_z", 1)):
                if i < len(out) and torch.is_tensor(out[i]):
                    d[name] = out[i].detach()

    # The two states INSIDE a level that the V-cycle's own in/out hooks miss:
    # after the pre-smoother (what gets restricted down) and after the coarse
    # correction (what the post-smoother starts from). Primal-dual family only:
    # its smoothers are the `lpdsA` / `lpdsB` stacks.
    def smoothed(level):
        def hook(mod, args_, out):
            x, z = _as_state(out[0] if isinstance(out, tuple) else out)
            d = slot(level)
            if x is not None:
                d["x_pre"] = x.detach()
            if z is not None:
                d["z_pre"] = z.detach()
        return hook

    def corrected(level):
        def hook(mod, args_):
            x, z = _as_state(args_[0])
            d = slot(level)
            if x is not None:
                d["x_corr"] = x.detach()
            if z is not None:
                d["z_corr"] = z.detach()
        return hook

    def leaf_pre(mod, args_):
        x, z = _as_state(args_[0])
        d = slot(maxd + 1)
        if x is not None:
            d["x"] = x.detach()
        if z is not None:
            d["z"] = z.detach()

    def leaf_post(mod, args_, out):
        x, z = _as_state(out[0] if isinstance(out, tuple) else out)
        d = slot(maxd + 1)
        if x is not None:
            d["x_out"] = x.detach()
        if z is not None:
            d["z_out"] = z.detach()

    try:
        for v in vcs:
            handles.append(v.register_forward_pre_hook(pre))
            handles.append(v.register_forward_hook(post))
            if hasattr(v, "lpdsA") and hasattr(v, "lpdsB"):
                handles.append(v.lpdsA.register_forward_hook(smoothed(lvl[id(v)])))
                handles.append(v.lpdsB.register_forward_pre_hook(corrected(lvl[id(v)])))
            df = getattr(v, "dF", None)
            if isinstance(df, df_cls):
                df._trace_level = lvl[id(v)]
                handles.append(df.register_forward_hook(df_post))
            # the coarsest level is a plain stack, not a V-cycle
            if isinstance(getattr(v, "mglayer", None), leaf_cls):
                handles.append(v.mglayer.register_forward_pre_hook(leaf_pre))
                handles.append(v.mglayer.register_forward_hook(leaf_post))
        with torch.no_grad():
            out = net(*args, **kwargs)
    finally:
        for h in handles:
            h.remove()
        for v in vcs:
            if hasattr(getattr(v, "dF", None), "_trace_level"):
                del v.dF._trace_level

    # What the level below handed back, before alpha and prolongation. Formed
    # here rather than hooked because it is a DIFFERENCE of two records that
    # live in different slots.
    for (o, l), d in list(rec.items()):
        below = rec.get((o, l + 1))
        if not below:
            continue
        for var in ("x", "z"):
            c, w = d.get(f"{var}_c"), below.get(f"{var}_out")
            if c is not None and w is not None and c.shape == w.shape:
                d[f"coarse_delta_{var}"] = w - c

    return out, LevelTrace(rec, maxd + 2, family)


def _transfer_kws(net):
    """The fixed (image-grid) transfer of `net`'s V-cycles, as `restrict` kwargs."""
    from models.mg_lpds import PDObjectiveDownsample
    for m in net.modules():
        if isinstance(m, PDObjectiveDownsample):
            return dict(julia_compat=m.julia_compat, filter=m.transfer_filter)
    return None


def level_images(net, trace, out, outer=-1, tol=1e-3):
    """The image each grid level holds once V-cycle `outer` has come back up
    through it (`x_out`, the decoder side), finest first, in IMAGE units.

    -> `[x_0, x_1, ...]`, each `(B, C, H / 2^l, W / 2^l)` complex on the
    EMBEDDED grid (crop with the training loop's `T`), or None when there is no
    primal image per level (the CDLNet family carries only a code) or the net's
    output is not on the finest iterate's grid.

    DC. MGLPDSNet works on the DC-removed image (kspace_pre_process subtracts
    mu E^H E 1, kspace_post_process adds mu back), so level l holds
    R^l(x - mu). mu is recovered per image as output minus the finest iterate
    (a constant -- a larger misfit than `tol` returns None rather than a wrong
    picture) and R^l(mu) is added back with the net's own transfer.
    """
    from models.components import to_complex
    from operators.resample import restrict

    kws = _transfer_kws(net)
    if trace.family != "MGLPDSNet" or kws is None:
        return None
    o = trace.outers[outer]
    xs = []
    for l in trace.levels:
        x = trace.get(o, l, "x_out")
        if x is None:
            break
        xs.append(to_complex(x) if torch.is_complex(out) else x)
    if not xs or xs[0].shape != out.shape:
        return None
    diff = out - xs[0]
    mu = diff.mean(dim=(-2, -1), keepdim=True)
    if float((diff - mu).norm()) > tol * float(out.norm()) + 1e-12:
        return None
    dc = mu.expand_as(xs[0]).contiguous()
    images = []
    for x in xs:
        images.append(x + dc)
        dc = restrict(dc, **kws)
    return images


# ---------------------------------------------------------------------------
# per-level SUPERVISION: the coarse levels' final iterates, WITH autograd
# ---------------------------------------------------------------------------
LEVEL_LOSS_OUTERS = ("last", "all")


def _pd_chains(net):
    """`{(outer, level): module}` for an MGLPDSNet: every `PDVCycle` and the
    coarsest level's `LPDSStack`, level 0 finest, outer = V-cycle index."""
    from models.mg_lpds import PDVCycle
    cycles = [m for m in net.modules() if isinstance(m, PDVCycle)]
    inner = {id(c.mglayer) for c in cycles}
    out = {}
    for k, V in enumerate(c for c in cycles if id(c) not in inner):
        level = 0
        while isinstance(V, PDVCycle):
            out[(k, level)] = V
            V, level = V.mglayer, level + 1
        out[(k, level)] = V                       # the coarsest level's stack
    return out


class collect_level_outputs:
    """Context manager: during a forward of `net`, keep each COARSE level's
    final iterate x (after that level's last sweep -- what it hands back to
    the level above), with its autograd graph.

        with collect_level_outputs(net, "last") as rec:
            x_hat, (x0, z0) = net(y, E=E, sigma=sigma)
        rec.levels  ->  {level: [x per selected V-cycle, execution order]}

    `outers` = "last" (the final V-cycle) or "all". Level 0 is not collected:
    it IS the output. Hooks only -- nothing is detached, the forward is
    unchanged, and they are removed on exit (also on an exception)."""

    def __init__(self, net, outers="last"):
        if outers not in LEVEL_LOSS_OUTERS:
            raise ValueError(f"outers must be one of {LEVEL_LOSS_OUTERS}, got {outers!r}")
        chains = _pd_chains(net)
        if not chains:
            raise TypeError(f"{type(net).__name__} has no primal-dual V-cycle: per-level "
                            f"supervision is for multigrid MGLPDS nets only")
        n_outer = 1 + max(k for k, _ in chains)
        keep = {n_outer - 1} if outers == "last" else set(range(n_outer))
        self._targets = [(l, m) for (k, l), m in sorted(chains.items()) if l > 0 and k in keep]
        self.levels, self._handles = {}, []

    def __enter__(self):
        for level, m in self._targets:
            def hook(mod, args, out, level=level):
                state = out[0] if isinstance(out, tuple) else out
                self.levels.setdefault(level, []).append(state[0])
            self._handles.append(m.register_forward_hook(hook))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False


def supervised_level_images(net, levels, x_hat, x0):
    """The collected coarse iterates in IMAGE units on the embedded grid ->
    `{level: [image, ...]}`, differentiable.

    `x_hat` is the net's output (before any crop) and `x0` the finest iterate
    it returned; the net works DC-removed, so `x_hat - x0` is the constant mu
    (as `level_images`), and level l holds R^l(x - mu): R^l(mu) is added back
    with the net's own image-grid transfer. A planar iterate comes back complex."""
    from models.components import to_complex
    from operators.resample import restrict

    kws = _transfer_kws(net)
    if x0.shape != x_hat.shape:
        raise ValueError(f"output {tuple(x_hat.shape)} and finest iterate {tuple(x0.shape)} "
                         f"differ: the net padded beyond the embedding, which per-level "
                         f"supervision does not handle")
    mu = (x_hat - x0).detach().mean(dim=(-2, -1), keepdim=True)
    dc = mu.expand_as(x0).contiguous()
    out = {}
    for level in sorted(levels):
        for _ in range(level - (max(out) if out else 0)):
            dc = restrict(dc, **kws)
        xs = [to_complex(x) if (x_hat.is_complex() and not x.is_complex()) else x
              for x in levels[level]]
        out[level] = [x + dc for x in xs]
    return out


class LevelTraceMixin:
    """Adds `forward_levels` to a multigrid container."""

    def forward_levels(self, *args, **kwargs):
        """One forward, plus every per-level intermediate.

        -> `(output, LevelTrace)`, where `output` is exactly what `forward`
        returns. Runs under `no_grad`; for training, call `forward`.
        """
        return trace_forward(self, *args, **kwargs)

    def forward_level_images(self, *args, outer=-1, **kwargs):
        """One traced forward -> `(output, level_images(...))`: what `forward`
        returns, and the image every grid level holds after V-cycle `outer`
        (finest first, image units, embedded grid; None when not available).
        Runs under `no_grad` (training/recon.py's `log_levels` panel)."""
        out, trace = trace_forward(self, *args, **kwargs)
        x_hat = out[0] if isinstance(out, tuple) else out
        return out, level_images(self, trace, x_hat, outer=outer)
