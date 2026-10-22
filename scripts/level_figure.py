#!/usr/bin/env python3
"""
What a trained multigrid net holds at EVERY GRID LEVEL on one real val slice --
next to what a U-Net decoder holds at the same scales.

    python scripts/level_figure.py --run trained_nets/mg_recon/knee/mg169v6_R8 \\
        --volume file1000108 --slice 22 \\
        --compare trained_nets/mg_recon/knee/varnetmaps_R8 --out figures/out/levels

    # re-render without torch / data / GPU (e.g. after copying the .npz down):
    python scripts/level_figure.py --from-npz figures/out/levels/mg169v6_R8_knee_1000108_22.npz

THE ARGUMENT IT IS BUILT FOR. In MGLPDS every level of the V-cycle carries a
primal IMAGE x_l that solves a coarse version of the SAME reconstruction
problem (FAS: the coarse variable is the full restricted solution, not an
error), so it can be checked directly -- against the ground truth restricted
by the net's own transfer operator R. The figure prints that NRMSE per level.
A U-Net decoder (E2E-VarNet's regularisers, RAM, any U-Net) carries tens to
hundreds of feature CHANNELS per scale with no defined relation to an image;
showing them needs a summary (channel RMS, first principal component), and
nothing about the reconstruction problem says what they should be.

The measurement is eval_per_volume's for that slice -- same loader, same
noise seed (`slice_seed`), same operator (online ESPIRiT, embedding), same
sigma -- so the final-level numbers match results/<tree>/slices.csv.

Rows of the MGLPDS figure, one column per grid level, coarsest -> finest (the
DECODER side: what each level holds once the V-cycle has come back up through
it, last outer cycle):
    x_l              the level's iterate, |.|
    R^l(gt)          the ground truth through the net's restriction, l times
    |x_l - R^l gt|   residual, windowed at vmax / 5 (as the viewer's)
    |dx| handed up   what the level below returned (coarse_delta_x), on the
                     grid it lives on; the finest column has none
plus a second figure: x_l across the outer V-cycles.

DC: MGLPDS iterates on the DC-removed image (preprocessing/kspace.py: the
constant mu is subtracted before the unroll and added back after), so every
level holds R^l(x - mu). mu is recovered as output minus finest iterate (a
constant; the misfit is printed) and R^l(mu) added back to each level -- the
images shown are exactly what the V-cycle holds, in image units.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)


# ---------------------------------------------------------------------------
# compute (torch, data, checkpoint)
# ---------------------------------------------------------------------------
def _as_complex(t):
    """A traced iterate as a complex (B, 1, H, W) tensor. Under the planar
    state it is real with [re; im] stacked on the channel axis."""
    import torch
    if t is None or t.is_complex():
        return t
    if t.shape[1] == 2:
        return torch.complex(t[:, 0:1], t[:, 1:2])
    return t


def _mg_call(store):
    def call(net, y, E, sigma):
        out, trace = net.forward_levels(y, E=E, sigma=sigma)
        out = out[0] if isinstance(out, tuple) else out
        store.update(trace=trace, out_embedded=out.detach())
        return out
    return call


def _unet_call(store, cascade):
    """Record the decoder blocks (`up_conv`) of one VarNet cascade's U-Net."""
    def call(net, y, E, sigma):
        cascades = net.net.cascades
        unet = cascades[cascade].model.unet
        feats, hs = [], []
        for blk in unet.up_conv:
            hs.append(blk.register_forward_hook(
                lambda m, a, o: feats.append(o.detach())))
        try:
            import torch
            with torch.no_grad():
                out = net(y, E=E, sigma=sigma)
        finally:
            for h in hs:
                h.remove()
        store.update(features=feats, n_cascades=len(cascades),
                     cascade=cascade % len(cascades))
        return out[0] if isinstance(out, tuple) else out
    return call


def _transfer_kws(net):
    from models.mg_lpds import PDObjectiveDownsample
    for m in net.modules():
        if isinstance(m, PDObjectiveDownsample):
            return dict(julia_compat=m.julia_compat, filter=m.transfer_filter)
    raise TypeError(f"{type(net).__name__} has no PDObjectiveDownsample")


def _resolve_volume(cfg, name):
    import dump_eval as de
    key = json.dumps(cfg["data"]["val"], sort_keys=True, default=str)
    if key not in _VOLUMES:
        _VOLUMES[key] = de.resolve_volumes(cfg["data"]["val"], None, 0)
    names = _VOLUMES[key]
    if name in names:
        return name
    hits = [n for n in names if name in n]
    if len(hits) != 1:
        raise SystemExit(f"--volume {name!r} matches {len(hits)} val volumes"
                         + (f": {hits[:5]}" if hits else ""))
    return hits[0]


# ---------------------------------------------------------------------------
# cache: each checkpoint, slice batch and volume list is loaded ONCE per process,
# so a notebook can re-trace with other knobs without reloading the nets
# ---------------------------------------------------------------------------
_NETS, _BATCHES, _VOLUMES = {}, {}, {}


def clear_cache():
    """Drop every cached net, batch and volume list (frees GPU memory)."""
    _NETS.clear(); _BATCHES.clear(); _VOLUMES.clear()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:                                              # noqa: BLE001
        pass


def cache_info():
    """One line per cached net: run, device, the knobs it is currently set to."""
    if not _NETS:
        return "no nets cached"
    return "\n".join(f"  {os.path.basename(k[0])} on {k[1]}  (knobs now: {v['knobs']})"
                     for k, v in _NETS.items())


def _net(run_dir, device):
    """`dump_eval.load_run`, once per (run, device) -- reloaded only if net.ckpt
    changed on disk. Remembers what the net was BUILT with (coarse operator,
    transfer filter) so every trace starts from that, whatever an earlier one set."""
    import dump_eval as de
    from models.mg_lpds import PDObjectiveDownsample

    key = (os.path.abspath(run_dir), str(device))
    stamp = os.path.getmtime(os.path.join(run_dir, "net.ckpt"))
    hit = _NETS.get(key)
    if hit is None or hit["stamp"] != stamp:
        cfg, model, n_params, step = de.load_run(run_dir, device)
        hit = _NETS[key] = dict(
            cfg=cfg, model=model, n_params=n_params, step=step, stamp=stamp, knobs=None,
            coarse=[(m, m.coarse_op) for m in model.modules()
                    if isinstance(m, PDObjectiveDownsample)],
            filters=[(m, m.transfer_filter) for m in model.modules()
                     if hasattr(m, "transfer_filter")])
        print(f"[{os.path.basename(os.path.normpath(run_dir))}] loaded "
              f"({n_params:,} params, step {step})")
    return hit


def _modes(cfg):
    """The process-wide switches `load_run` sets, for THIS net: another cached
    net may have set them since (they change no parameter, ~1e-7 in the output,
    but each net should run the way it is evaluated)."""
    from models.components import set_complex_mode
    from models.mg_lpds import set_planar_state
    t = cfg.get("training", {})
    set_complex_mode(t.get("complex_conv") or "gauss")      # "gauss": the class default
    set_planar_state(bool(t.get("planar_state", False)))


def _set_knobs(entry, coarse_op=None, transfer_filter=None):
    """Back to the BUILT coarse operator / transfer filter, then these swaps."""
    from operators.resample import set_transfer_filter
    for m, op in entry["coarse"]:
        m.coarse_op = op
    for m, f in entry["filters"]:
        m.transfer_filter = f
    if coarse_op:
        for m, _ in entry["coarse"]:
            m.coarse_op = coarse_op
    if transfer_filter:
        set_transfer_filter(entry["model"], transfer_filter)
    entry["knobs"] = dict(coarse_op=coarse_op, transfer_filter=transfer_filter)


@contextlib.contextmanager
def as_traced(live, which="mg"):
    """Put a cached net back in the state trace `live` ran it with -- coarse
    operator, transfer filter, sinc length, conv/planar modes -- for working
    with `live[which]["model"]` after other traces have re-used that net."""
    from operators.resample import use_filter_length
    L = live[which]
    k = L.get("knobs", {})
    _modes(L["cfg"])
    _set_knobs(L["entry"], k.get("coarse_op"), k.get("transfer_filter"))
    with (use_filter_length("sinc", k["transfer_taps"]) if k.get("transfer_taps")
          else contextlib.nullcontext()):
        yield L["model"]


def _batch(cfg, vol, sl):
    key = (json.dumps(cfg["data"]["val"], sort_keys=True, default=str), vol, int(sl))
    if key not in _BATCHES:
        from datasets.registry import build_loader
        val = dict(cfg["data"]["val"])
        val.update(enumerate_slices=True, start_slice=sl, end_slice=sl + 1,
                   volumes=[vol], batch_size=1, num_workers=0)
        _BATCHES[key] = next(iter(build_loader(val, shuffle=False, drop_last=False)))
    return _BATCHES[key]


def _run_one(run_dir, vol, sl, args, device, call_for):
    import copy

    import torch
    from eval_per_volume import slice_seed
    from evaluation.tasks import build_adapter, default_sigma, force_organ_mask
    from operators.resample import transfer_filter_of

    entry = _net(run_dir, device)          # cached: loaded once per process
    cfg, model = copy.deepcopy(entry["cfg"]), entry["model"]
    n_params, step = entry["n_params"], entry["step"]
    _modes(cfg)
    # Inference-time swaps (no parameter depends on either), applied from the
    # net's BUILT state so nothing an earlier trace set survives: the coarse
    # forward model (e.g. a rediscretize-trained net with the Galerkin Gram) and
    # the grid-transfer filter (e.g. "box", or "none" -- the sinc off).
    _set_knobs(entry)
    filter_trained = transfer_filter_of(model)
    _set_knobs(entry, getattr(args, "coarse_op", None), getattr(args, "transfer_filter", None))
    name = os.path.basename(os.path.normpath(run_dir))
    if getattr(args, "coarse_op", None) and entry["coarse"]:
        print(f"[{name}] coarse_op -> {args.coarse_op} ({len(entry['coarse'])} levels)")
    if getattr(args, "transfer_filter", None):
        print(f"[{name}] transfer filter {filter_trained} -> {args.transfer_filter}")
    force_organ_mask(cfg)
    if args.val_kspace_type:
        cfg.setdefault("training", {})["val_kspace_type"] = args.val_kspace_type
    sigma = args.sigma if args.sigma is not None else default_sigma(cfg)
    batch = _batch(cfg, vol, sl)          # cached: same slice, same tensors
    gen = torch.Generator(device=device).manual_seed(slice_seed(args.seed, vol, sl))
    store = {}
    gt, recon, organ = build_adapter(cfg["task"])(
        model, batch, cfg, device, sigma, gen, call=call_for(store))
    if organ is None:
        organ = batch[3].to(device)
    return dict(cfg=cfg, model=model, gt=gt, recon=recon, organ=organ, store=store,
                sigma=float(sigma), n_params=int(n_params), step=int(step or 0),
                run=os.path.basename(os.path.normpath(run_dir)),
                filter_trained=filter_trained, entry=entry)


def mg_levels(tr, out_e, gt, organ, kws):
    """The per-level arrays of one traced forward, in image units.

    -> `(npz entries, mu, misfit)`. The net works on the DC-removed image
    (kspace_pre_process subtracts mu * E^H E 1, kspace_post_process adds mu
    back), so every level holds R^l(x - mu): mu is recovered as output minus the
    finest iterate -- a constant, `misfit` says how nearly -- and R^l(mu) is
    added back per level. The ground truth and organ mask are embedded like the
    iterate and restricted with the net's own transfer (`kws`).
    """
    import torch
    from operators.resample import restrict
    from operators.truncate import Truncate

    x0 = _as_complex(tr.get(-1, 0, "x_out"))
    if x0.shape != out_e.shape:
        raise SystemExit(f"finest iterate {tuple(x0.shape)} vs output {tuple(out_e.shape)}: "
                         f"the net pads beyond the embedding; not handled here")
    diff = out_e - x0
    mu = diff.mean()
    fit = float((diff - mu).norm() / out_e.norm())
    if fit > 1e-3:
        print(f"WARNING: output minus finest iterate is not a constant (rel. misfit "
              f"{fit:.2e}); per-level images are approximate", file=sys.stderr)

    npz = {}
    T = Truncate(tuple(x0.shape[-2:]), tuple(gt.shape[-2:]))
    g, m = T.adjoint(gt), T.adjoint(organ.to(gt.dtype)).abs()
    dc = torch.full_like(x0, mu)
    for l in tr.levels:
        npz[f"mg/gt/l{l}"] = g[0, 0].cpu().numpy().astype(np.complex64)
        npz[f"mg/organ/l{l}"] = m[0, 0].real.cpu().numpy().astype(np.float32)
        for o in tr.outers:
            # x_out -> mg/x (the decoder side); the other V-cycle stages as
            # mg/xin (entering the level), mg/xpre (pre-smoothed, restricted
            # next), mg/xcorr (coarse-corrected, post-smoothed next)
            for key, name in (("x_out", "x"), ("x", "xin"), ("x_pre", "xpre"),
                              ("x_corr", "xcorr")):
                x = _as_complex(tr.get(o, l, key))
                if x is not None:
                    npz[f"mg/{name}/o{o}/l{l}"] = (x + dc)[0, 0].cpu().numpy().astype(np.complex64)
            d = _as_complex(tr.get(o, l, "coarse_delta_x"))
            if d is not None:      # a difference: no DC; lives on level l+1's grid
                npz[f"mg/delta/o{o}/l{l + 1}"] = d[0, 0].cpu().numpy().astype(np.complex64)
        g, m, dc = restrict(g, **kws), restrict(m, **kws).abs(), restrict(dc, **kws)
    return npz, mu, fit


def trace_slice(run, volume, slice, compare=None, cascade=-1, sigma=None, seed=1234,
                val_kspace_type="measurement_awgn", device=None, coarse_op=None,
                transfer_taps=None, transfer_filter=None):
    """Run `run` (a multigrid net) -- and optionally `compare` (a VarNet) -- on
    one val slice, exactly as eval_per_volume measures it.

    -> `(z, meta, live)`:
        z     {name: numpy array}, the layout `save_npz` writes (mg/x/o<o>/l<l>,
              mg/gt/l<l>, mg/delta/o<o>/l<l>, mg/organ/l<l>, vn/feat/<k>, gt, recon)
        meta  plain dict: volume, slice, run, M, K, levels, outers, sigma, ...
        coarse_op  None: as trained; "galerkin" / "rediscretize" swaps the
              coarse forward model of `run` at inference (the trace records it)
        transfer_taps  None: as trained (the 8-tap sinc at factor 2); e.g. 12 or
              16 implants a LONGER sinc into every transfer of `run` at
              inference -- restriction, prolongation, the coarse maps, and the
              R^l(gt) the levels are scored against -- same weights
              (operators.resample.use_filter_length; the trace records it)
        transfer_filter  None: as trained (the sinc); "box", "bilinear", ...
              or "none" (no filter at all: plain decimation down, one-sample
              scatter up) swaps the FILTER of every transfer of `run` at
              inference -- same weights (operators.resample.set_transfer_filter;
              the trace records it). R^l(gt) follows the net's transfer, so
              coarse-level NRMSEs are only comparable at equal transfers.
        live  the objects, for poking at further in a notebook: live["mg"] has
              model, cfg, trace (a LevelTrace), gt, recon, organ, out_embedded,
              knobs; live["vn"] the same for the comparison plus `features`.
              The nets are CACHED (loaded once per process; `clear_cache`), so a
              later trace may re-set the same net's knobs: wrap work on
              live["mg"]["model"] in `with as_traced(live):`
    """
    import types

    import torch

    device = torch.device(device) if device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    args = types.SimpleNamespace(sigma=sigma, seed=seed, val_kspace_type=val_kspace_type,
                                 coarse_op=coarse_op, transfer_filter=transfer_filter)
    with open(os.path.join(run, "config.json")) as f:
        vol = _resolve_volume(json.load(f), volume)
    sl = int(slice)

    from operators.resample import filter_length, use_filter_length
    taps_trained = filter_length("sinc")
    with (use_filter_length("sinc", transfer_taps) if transfer_taps
          else contextlib.nullcontext()):
        R = _run_one(run, vol, sl, args, device, _mg_call)
        tkw = _transfer_kws(R["model"])
        if transfer_taps and tkw.get("filter") != "sinc":
            raise ValueError(f"{run} uses transfer filter {tkw.get('filter')!r}; "
                             f"transfer_taps only lengthens the 'sinc'")
        tr, out_e = R["store"]["trace"], R["store"]["out_embedded"]
        gt, organ = R["gt"], R["organ"]
        z, mu, fit = mg_levels(tr, out_e, gt, organ, tkw)
    meta = dict(volume=vol, slice=sl, anatomy=R["cfg"]["data"]["val"].get("anatomy", ""),
                run=R["run"], n_params=R["n_params"], sigma=R["sigma"], step=R["step"],
                K=R["cfg"]["model"]["params"].get("K"), M=R["cfg"]["model"]["params"].get("M"),
                levels=list(tr.levels), outers=list(tr.outers),
                dc=[float(mu.real), float(mu.imag)], scale_misfit=fit,
                R=R["cfg"]["mri"].get("R"),
                kspace_type=R["cfg"].get("training", {}).get("val_kspace_type"),
                run_path=run,
                coarse_op=coarse_op or R["cfg"]["model"]["params"].get("coarse_op", "galerkin"),
                coarse_op_trained=R["cfg"]["model"]["params"].get("coarse_op", "galerkin"),
                coarse_op_swapped=coarse_op if coarse_op and coarse_op != R["cfg"][
                    "model"]["params"].get("coarse_op", "galerkin") else None,
                transfer_taps=int(transfer_taps or taps_trained),
                transfer_taps_trained=int(taps_trained),
                transfer_taps_swapped=(int(transfer_taps) if transfer_taps
                                       and int(transfer_taps) != taps_trained else None),
                transfer_filter=str(transfer_filter or R["filter_trained"]),
                transfer_filter_trained=str(R["filter_trained"]),
                transfer_filter_swapped=(transfer_filter if transfer_filter
                                         and transfer_filter != R["filter_trained"] else None))
    live = {"mg": dict(R, trace=tr, out_embedded=out_e,
                       knobs=dict(coarse_op=coarse_op, transfer_filter=transfer_filter,
                                  transfer_taps=transfer_taps))}

    if compare:
        args.coarse_op = None            # the swaps are for the multigrid net only
        args.transfer_filter = None
        V = _run_one(compare, vol, sl, args, device, lambda st: _unet_call(st, cascade))
        feats = V["store"]["features"]
        for k, f in enumerate(feats):          # coarse -> fine
            z[f"vn/feat/{k}"] = f[0].float().cpu().numpy().astype(np.float16)
        z["vn/recon"] = V["recon"][0, 0].cpu().numpy().astype(np.complex64)
        meta.update(compare=V["run"], compare_n_params=V["n_params"],
                    cascade=V["store"]["cascade"], n_cascades=V["store"]["n_cascades"],
                    compare_path=compare)
        live["vn"] = dict(V, features=feats, knobs={})
    z["gt"] = gt[0, 0].cpu().numpy().astype(np.complex64)
    z["recon"] = R["recon"][0, 0].cpu().numpy().astype(np.complex64)
    return z, meta, live


def _layers_call(store):
    """Record the primal iterate after EVERY `LPDSLayer`, in execution order --
    the flat LPDSNet's K layers, or every smoother layer of a V-cycle (each on
    its own level's grid)."""
    def call(net, y, E, sigma):
        import torch
        from models.lpds import LPDSLayer
        xs, hs = [], []

        def hook(mod, args_, out):
            st = out[0] if isinstance(out, tuple) else out
            x = st[0] if isinstance(st, tuple) else st
            xs.append(_as_complex(x.detach()))

        for m in net.modules():
            if isinstance(m, LPDSLayer):
                hs.append(m.register_forward_hook(hook))
        try:
            with torch.no_grad():
                out = net(y, E=E, sigma=sigma)
        finally:
            for h in hs:
                h.remove()
        out = out[0] if isinstance(out, tuple) else out
        store.update(xs=xs, out_embedded=out.detach())
        return out
    return call


def trace_layers(run, volume, slice, sigma=None, seed=1234,
                 val_kspace_type="measurement_awgn", device=None, coarse_op=None,
                 transfer_filter=None):
    """The iterate after every LPDS layer of `run` on one val slice (same
    measurement as `trace_slice`; cached nets, same knobs).

    -> `(layers, gts, meta)`: `layers` a list of {"k", "shape", "x"} in
    execution order, x complex in IMAGE units (the DC the net removes added
    back -- under the "dc" transfer normalisation a constant restricts to
    itself, so it is the same constant on every level); `gts` {shape: R^l(gt)}
    for every grid size that occurs, via the net's own restriction.
    """
    import types

    import torch
    from operators.resample import restrict

    device = torch.device(device) if device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    args = types.SimpleNamespace(sigma=sigma, seed=seed, val_kspace_type=val_kspace_type,
                                 coarse_op=coarse_op, transfer_filter=transfer_filter)
    with open(os.path.join(run, "config.json")) as f:
        vol = _resolve_volume(json.load(f), volume)
    R = _run_one(run, vol, int(slice), args, device, _layers_call)
    xs, out_e = R["store"]["xs"], R["store"]["out_embedded"]
    last = next(x for x in reversed(xs) if x.shape == out_e.shape)
    diff = out_e - last
    mu = diff.mean()
    fit = float((diff - mu).norm() / out_e.norm())
    from operators.truncate import Truncate
    g = Truncate(tuple(out_e.shape[-2:]), tuple(R["gt"].shape[-2:])).adjoint(R["gt"])
    gts = {tuple(g.shape[-2:]): g[0, 0].cpu().numpy().astype(np.complex64)}
    # One R^l(gt) per grid the net actually ran on -- not "until it is small":
    # a grid that is even only down to the coarsest level (e.g. 768 x 400 ->
    # 96 x 50 at four levels) cannot be restricted once more.
    need = {tuple(x.shape[-2:]) for x in xs}
    try:
        kws = _transfer_kws(R["model"])
        while not need <= set(gts) and all(n % 2 == 0 for n in g.shape[-2:]):
            g = restrict(g, **kws)
            gts[tuple(g.shape[-2:])] = g[0, 0].cpu().numpy().astype(np.complex64)
    except TypeError:                       # a flat net: no transfers, one grid
        pass
    layers = [dict(k=i, shape=tuple(x.shape[-2:]),
                   x=(x + mu)[0, 0].cpu().numpy().astype(np.complex64))
              for i, x in enumerate(xs)]
    meta = dict(run=R["run"], volume=vol, slice=int(slice), n_layers=len(xs),
                dc=[float(mu.real), float(mu.imag)], dc_misfit=fit,
                anatomy=R["cfg"]["data"]["val"].get("anatomy", ""), run_path=run)
    if fit > 1e-3:
        print(f"WARNING: output minus last finest iterate is not a constant "
              f"(rel. misfit {fit:.1e}); per-layer images are approximate", file=sys.stderr)
    return layers, gts, meta


def nyquist_share(a, frac=0.8):
    """`(rows, cols)`: the share of |a|'s energy (mean removed) at or above
    `frac` of the Nyquist frequency along axis 0 / axis 1 of a's own grid. A
    period-2 pattern along an axis (stripes ACROSS it) puts its energy there;
    natural images put very little."""
    m = np.abs(a).astype(np.float64)
    m = m - m.mean()
    P = np.abs(np.fft.fft2(m)) ** 2
    fy = np.abs(np.fft.fftfreq(m.shape[0])) * 2       # 0 .. 1 = Nyquist
    fx = np.abs(np.fft.fftfreq(m.shape[1])) * 2
    tot = P.sum() or 1.0
    return float(P[fy >= frac, :].sum() / tot), float(P[:, fx >= frac].sum() / tot)


# ---------------------------------------------------------------------------
# the stride-2 convs' polyphase leak (notebook section 12)
#
# A stride-2 conv with an odd kernel is FOUR filters, one per output parity
# (row, col): with P = 7, an even output row gets taps {1, 3, 5}, an odd one
# {0, 2, 4, 6}. The synthesis (ConvTranspose2d) therefore maps a latent DC to
#     W(0, 0)  on every pixel, plus   W(pi, 0) (-1)^row + W(0, pi) (-1)^col
#                                     + W(pi, pi) (-1)^(row + col)
# where W(pi, 0) = sum_ij (-1)^i w_ij is (even-row taps - odd-row taps): a
# period-2 pattern along axis 0 (horizontal stripes, `nyquist_share`'s "rows")
# with that gain. The analysis conv is its adjoint: image content at those
# frequencies ALIASES onto the latent DC with the same gains. Nothing in
# training pins them to zero; a random (init) kernel has them ~ W(0, 0).
# ---------------------------------------------------------------------------
def cached_model(run, device=None):
    """The cached net of `run` -- the SAME object `trace_slice` / `trace_layers`
    run -- for inspecting or (temporarily) editing its weights."""
    import torch
    device = torch.device(device) if device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    return _net(run, device)["model"]


def _vcycle_walk(model):
    """Every LPDS layer and FAS downsampler of `model` in execution order, as
    `(kind, level, where, outer, module)`: kind "layer" (an `LPDSLayer`) or
    "dF" (a `PDObjectiveDownsample`, between `level` and `level + 1`); `outer`
    the outer V-cycle (None for a flat stack, whose layers are all level 0)."""
    from models.lpds import LPDSLayer
    from models.mg_lpds import PDVCycle

    def cycle(V, level, k):
        for i, L in enumerate(V.lpdsA.layers):
            yield "layer", level, f"V{k} L{level} pre{i}", k, L
        yield "dF", level, f"V{k} L{level}->L{level + 1} dF", k, V.dF
        if isinstance(V.mglayer, PDVCycle):
            yield from cycle(V.mglayer, level + 1, k)
        else:
            for i, L in enumerate(V.mglayer.layers):
                yield "layer", level + 1, f"V{k} L{level + 1} solve{i}", k, L
        for i, L in enumerate(V.lpdsB.layers):
            yield "layer", level, f"V{k} L{level} post{i}", k, L

    cycles = [m for m in model.modules() if isinstance(m, PDVCycle)]
    inner = {id(c.mglayer) for c in cycles}
    seen = set()
    for k, V in enumerate(c for c in cycles if id(c) not in inner):
        for item in cycle(V, 0, k):
            seen.add(id(item[-1]))
            yield item
    flat = [m for m in model.modules() if isinstance(m, LPDSLayer) and id(m) not in seen]
    for i, L in enumerate(flat):
        yield "layer", 0, f"layer{i}", None, L


def conv_sites(model):
    """Every analysis / synthesis conv of an LPDS-family net with its grid level:
    [{"level", "where", "conv": "analysis" | "synthesis", "module"}], V-cycle by
    V-cycle (pre-smoother, the FAS downsampler's copies, the coarser levels,
    post-smoother); a flat stack is all level 0."""
    sites, seen = [], set()

    def add(level, where, conv, module):
        if id(module) not in seen:
            seen.add(id(module))
            sites.append(dict(level=level, where=where, conv=conv, module=module))

    for kind, level, where, _, m in _vcycle_walk(model):
        if kind == "layer":
            add(level, where, "analysis", m.analysis)
            add(level, where, "synthesis", m.synthesis)
        else:
            add(level, where, "analysis", m.analysis_fine)
            add(level, where, "synthesis", m.synthesis_fine)
            add(level + 1, where, "analysis", m.analysis_coarse)
            add(level + 1, where, "synthesis", m.synthesis_coarse)
    if not sites:
        raise TypeError(f"{type(model).__name__} has no LPDS layers")
    return sites


def _parity_signs(w):
    import torch
    P0, P1 = w.shape[-2:]
    sr = (1 - 2 * (torch.arange(P0, device=w.device) % 2)).to(w.real.dtype)[:, None]
    sc = (1 - 2 * (torch.arange(P1, device=w.device) % 2)).to(w.real.dtype)[None, :]
    return sr, sc


def phase_leak(w):
    """`(rows, cols, diag)` for one conv weight (..., P, P), stride 2:
    ||W(pi, 0)|| / ||W(0, 0)||, ||W(0, pi)|| / ||W(0, 0)||, ||W(pi, pi)|| / ||W(0, 0)||,
    each W(.) the (out x in) matrix of kernel sums at that frequency, Frobenius
    norms. 0 = no period-2 leak; ~1 = a random kernel's."""
    sr, sc = _parity_signs(w)
    W00 = w.sum((-2, -1))
    den = float(W00.abs().pow(2).sum().sqrt()) or 1e-30
    return tuple(float((w * s).sum((-2, -1)).abs().pow(2).sum().sqrt()) / den
                 for s in (sr, sc, sr * sc))


def leak_table(model):
    """`phase_leak` of every conv of `model` -> [{level, where, conv, rows,
    cols, diag}], in `conv_sites` order."""
    import torch
    out = []
    with torch.no_grad():
        for s in conv_sites(model):
            r, c, d = phase_leak(s["module"].weight)
            out.append(dict(level=s["level"], where=s["where"], conv=s["conv"],
                            rows=r, cols=c, diag=d))
    return out


def leak_summary(rows):
    """Mean / max of the leak per (level, conv), coarse levels last ->
    [(level, conv, n, mean (rows, cols, diag), max (rows, cols, diag))]."""
    groups = {}
    for r in rows:
        groups.setdefault((r["level"], r["conv"]), []).append(r)
    out = []
    for (level, conv), rs in sorted(groups.items()):
        a = np.array([[r["rows"], r["cols"], r["diag"]] for r in rs])
        out.append((level, conv, len(rs), tuple(a.mean(0)), tuple(a.max(0))))
    return out


def equalize_phases(w, axes="both"):
    """`w` (..., P, P) with its stride-2 parity classes balanced -- the
    smallest (L2) change, spread evenly over each class's taps -- so that
      "both"  W(pi, 0) = W(0, pi) = W(pi, pi) = 0     (all four classes equal)
      "rows"  W(pi, 0) = W(pi, pi) = 0                (row parities equal)
      "cols"  W(0, pi) = W(pi, pi) = 0                (col parities equal)
    while W(0, 0) -- the kernel's DC gain, per (out, in) pair -- is unchanged."""
    import torch
    if axes not in ("both", "rows", "cols"):
        raise ValueError(f"axes must be 'both', 'rows' or 'cols', got {axes!r}")
    P0, P1 = w.shape[-2:]
    pr = (torch.arange(P0, device=w.device) % 2)[:, None]
    pc = (torch.arange(P1, device=w.device) % 2)[None, :]
    masks = {(a, b): ((pr == a) & (pc == b)).to(w.real.dtype) for a in (0, 1) for b in (0, 1)}
    sums = {k: (w * m).sum((-2, -1), keepdim=True) for k, m in masks.items()}
    out = w.clone()
    for (a, b), m in masks.items():
        if axes == "both":
            target = sum(sums.values()) / 4
        elif axes == "rows":
            target = (sums[(0, b)] + sums[(1, b)]) / 2
        else:
            target = (sums[(a, 0)] + sums[(a, 1)]) / 2
        out = out + m * (target - sums[(a, b)]) / m.sum()
    return out


@contextlib.contextmanager
def equalized_convs(model, convs=("analysis", "synthesis"), levels=None, axes="both"):
    """Inside the block, every selected conv of `model` has its polyphase leak
    removed (`equalize_phases`); the trained weights come back on exit, even on
    an error. `levels`: None = all, else an iterable of grid levels (0 = finest).
    Yields the number of convs changed. Same weights otherwise -- a diagnostic,
    like a transfer swap."""
    import torch
    from models.base import set_weight

    sel = [s for s in conv_sites(model) if s["conv"] in convs
           and (levels is None or s["level"] in set(levels))]
    saved = []
    try:
        with torch.no_grad():
            for s in sel:
                w = s["module"].weight
                saved.append((s["module"], w.clone()))
                set_weight(s["module"], equalize_phases(w, axes))
        yield len(saved)
    finally:
        with torch.no_grad():
            for m, w in saved:
                set_weight(m, w)


# ---------------------------------------------------------------------------
# which term of a sweep injects the period-2 energy (notebook section 12d)
#
# One LPDS sweep is
#     x+ = x - tau r,    r = (E^H E x - y~) + D z - pi_x
#     z+ = prox(u),      u = z + A xbar - pi_z     (pi_z after the prox under
#                                                    fas_dual="fixed_point")
# Every term of r is linear, so the near-Nyquist part of the STEP splits
# exactly into theirs. On a coarse level pi_x = r_c(entry) - R r_fine is split
# too: minus the level's own residual where it entered, plus the restricted
# fine residual. The image's Nyquist band is reached from the latent code only
# through D's W(pi) (latent near-DC -> image period 2), so the DUAL side is
# reported as D(each piece of u): which piece carries the latent DC that D
# leaks (before the clip, which is the one nonlinearity).
# ---------------------------------------------------------------------------
def _c2d(t):
    """A traced tensor as one complex (H, W) numpy image (planar-safe)."""
    return _as_complex(t.detach())[0, 0].cpu().numpy().astype(np.complex64)


def _terms_call(store, select):
    def call(net, y, E, sigma):
        import torch
        from models.lpds import _per_channel
        from models.mg_lpds import gram_or_x
        from operators.resample import restrict

        recs, rrx, patched = [], {}, []

        def wrap_dF(m):
            orig = m.forward

            def fwd(x, z, y_t, E_, sig, cache, static=None):
                out = orig(x, z, y_t, E_, sig, cache, static)
                planar = getattr(m, "is_complex", True) and not x.is_complex()
                r_f = gram_or_x(E_, x, planar) - y_t + m.synthesis_fine(z)
                # keep pi_x alive with its R r_fine, so the id cannot be reused
                rrx[id(out[2])] = (out[2], restrict(r_f, julia_compat=m.julia_compat,
                                                    filter=m.transfer_filter))
                return out
            m.forward = fwd
            patched.append(m)

        def wrap_layer(L, level, where, outer):
            orig = L.forward

            def fwd(state, y_tilde, E=None, sigma=None, pi=None, cache=None):
                if state is None:
                    return orig(state, y_tilde, E=E, sigma=sigma, pi=pi, cache=cache)
                x, z = state
                planar = L.is_complex and not x.is_complex()
                pi_x, pi_z = (None, None) if pi is None else pi
                T = {"data  E^H E x - y~": gram_or_x(E, x, planar) - y_tilde,
                     "synth D z": L.synthesis(z)}
                if pi_x is not None:
                    hit = rrx.get(id(pi_x))
                    if hit is not None and hit[0] is pi_x:
                        T["FAS   -r_c(entry)"] = -(pi_x + hit[1])
                        T["FAS   +R r_fine"] = hit[1]
                    else:
                        T["FAS   -pi_x"] = -pi_x
                out = orig(state, y_tilde, E=E, sigma=sigma, pi=pi, cache=cache)
                x_new = out[0][0]
                theta = L.theta(sigma, ref=x)
                theta = _per_channel(theta, x) if planar else theta
                x_bar = x_new + theta * (x_new - x)
                U = {"D z": L.synthesis(z), "D A xbar": L.synthesis(L.analysis(x_bar))}
                if pi_z is not None:
                    sign = 1.0 if getattr(L, "fas_dual", "legacy") == "fixed_point" else -1.0
                    U[f"D ({'+' if sign > 0 else '-'}pi_z)"] = L.synthesis(sign * pi_z)
                recs.append(dict(level=level, where=where, outer=outer,
                                 x_before=_c2d(x), x_after=_c2d(x_new),
                                 terms={k: _c2d(v) for k, v in T.items()},
                                 dual={k: _c2d(v) for k, v in U.items()},
                                 fas_dual=getattr(L, "fas_dual", "legacy")))
                return out
            L.forward = fwd
            patched.append(L)

        for kind, level, where, outer, m in _vcycle_walk(net):
            if kind == "dF":
                wrap_dF(m)
            elif select(level, outer):
                wrap_layer(m, level, where, outer)
        try:
            with torch.no_grad():
                out = net(y, E=E, sigma=sigma)
        finally:
            for m in patched:
                del m.forward                    # back to the class method
        store.update(recs=recs)
        return out[0] if isinstance(out, tuple) else out
    return call


def trace_terms(run, volume, slice, levels=(1,), outers=None, sigma=None, seed=1234,
                val_kspace_type="measurement_awgn", device=None, coarse_op=None,
                transfer_filter=None):
    """Every sweep of `run` on `levels` (0 = finest) in V-cycles `outers` (None:
    the LAST; a flat net: ignored), split into its terms -- same measurement
    as `trace_slice`, cached net, no weight changed.

    -> `(recs, meta)`, one record per sweep in execution order: level, where,
    outer, x_before / x_after (complex (H, W)), `terms` {name: (H, W)} summing
    to the primal residual r (the step is -tau r), `dual` {name: D(piece of
    the dual argument u)}, fas_dual."""
    import types

    import torch

    device = torch.device(device) if device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    model = cached_model(run, device)
    tops = sorted({o for kind, _, _, o, _ in _vcycle_walk(model) if o is not None})
    want = None if not tops else (
        {tops[-1]} if outers is None else {tops[o] for o in outers})
    levels = set(levels)

    def select(level, outer):
        return level in levels and (outer is None or want is None or outer in want)

    args = types.SimpleNamespace(sigma=sigma, seed=seed, val_kspace_type=val_kspace_type,
                                 coarse_op=coarse_op, transfer_filter=transfer_filter)
    with open(os.path.join(run, "config.json")) as f:
        vol = _resolve_volume(json.load(f), volume)
    R = _run_one(run, vol, int(slice), args, device, lambda st: _terms_call(st, select))
    recs = R["store"]["recs"]
    meta = dict(run=R["run"], volume=vol, slice=int(slice), levels=sorted(levels),
                outers=sorted(want) if want else None, n_sweeps=len(recs))
    if not recs:
        print(f"WARNING: no sweep of {R['run']} matched levels={sorted(levels)}, "
              f"outers={outers}", file=sys.stderr)
    return recs, meta


def _nyq_mask(shape, frac, axis):
    fy = np.abs(np.fft.fftfreq(shape[0])) * 2
    fx = np.abs(np.fft.fftfreq(shape[1])) * 2
    if axis == "rows":
        return np.broadcast_to((fy >= frac)[:, None], shape)
    if axis == "cols":
        return np.broadcast_to((fx >= frac)[None, :], shape)
    raise ValueError(f"axis must be 'rows' or 'cols', got {axis!r}")


def nyquist_part(a, frac=0.8, axis="rows"):
    """The part of the complex image `a` at or above `frac` of Nyquist along
    `axis` -- what `nyquist_share` counts, as an image."""
    A = np.fft.fft2(a)
    return np.fft.ifft2(np.where(_nyq_mask(a.shape, frac, axis), A, 0))


def field_share(a, frac=0.8):
    """`nyquist_share` of the complex FIELD (mean removed) rather than its
    magnitude: linear, so the shares of a sum's terms are comparable."""
    a = np.asarray(a, np.complex128)
    P = np.abs(np.fft.fft2(a - a.mean())) ** 2
    tot = P.sum() or 1.0
    return tuple(float(P[_nyq_mask(a.shape, frac, ax)].sum() / tot) for ax in ("rows", "cols"))


def term_attribution(parts, frac=0.8):
    """How the near-Nyquist part of `sum(parts.values())` splits over `parts`.

    -> {"rows" | "cols": [(name, share, size)]}: `share` = the term's projection
    onto the sum's near-Nyquist part over that part's energy -- the shares add
    to 1, a term can be negative (it cancels others); `size` = |term's
    near-Nyquist part| / |the sum's| (two big terms of opposite share cancel)."""
    total = sum(parts.values())
    out = {}
    for ax in ("rows", "cols"):
        m = _nyq_mask(total.shape, frac, ax)
        S = np.fft.fft2(total)[m]
        den = float((np.abs(S) ** 2).sum()) or 1e-30
        rows = []
        for name, t in parts.items():
            Tf = np.fft.fft2(t)[m]
            rows.append((name, float((np.conj(S) * Tf).real.sum() / den),
                         float(np.sqrt((np.abs(Tf) ** 2).sum() / den))))
        out[ax] = rows
    return out


def save_npz(z, meta, out_dir):
    """-> the path written:
    <out_dir>/<run>[_<coarse_op swap>][_taps<n>][_filt<name>]_<anatomy>_<volume tail>_<slice>.npz"""
    os.makedirs(out_dir, exist_ok=True)
    swap = meta.get("coarse_op_swapped")
    taps = meta.get("transfer_taps_swapped")
    filt = meta.get("transfer_filter_swapped")
    stem = (f"{meta['run']}{'_' + swap if swap else ''}{f'_taps{taps}' if taps else ''}"
            f"{f'_filt{filt}' if filt else ''}_"
            f"{meta['anatomy'] or 'slice'}_"
            f"{meta['volume'].split('_')[-1]}_{meta['slice']}")
    path = os.path.join(out_dir, stem + ".npz")
    np.savez_compressed(path, meta=np.array(json.dumps(meta)), **z)
    return path


def load_npz(path):
    """-> `(z, meta)`, as `trace_slice` returns them. numpy only."""
    with np.load(path, allow_pickle=False) as f:
        z = {k: f[k] for k in f.files if k != "meta"}
        meta = json.loads(str(f["meta"]))
    return z, meta


# ---------------------------------------------------------------------------
# figures (numpy + matplotlib only) -- each returns its Figure
# ---------------------------------------------------------------------------
# "nearest" shows raw pixels but ALIASES whenever a panel has fewer screen pixels
# than the image (a 640x372 slice in a 3-inch panel): edges turn jagged and
# blocky. "antialiased" filters when shrinking and stays sharp when enlarging.
INTERP = "antialiased"


def _flip(a, how):
    return a[::-1 if how in ("ud", "both") else 1, ::-1 if how in ("lr", "both") else 1]


def default_flip(meta):
    """'ud' for knee, as the viewer (figures/configs FLIP); '' otherwise."""
    return "ud" if "knee" in (meta.get("anatomy", "") + meta.get("run_path", "")) else ""


def nrmse(x, g):
    """||(|x| - |g|)|| / |||g|||, magnitudes -- the metric's definition."""
    return float(np.linalg.norm(np.abs(x) - np.abs(g)) / (np.linalg.norm(np.abs(g)) + 1e-12))


def pc1(f):
    """First principal component of a (C, h, w) feature stack, as an (h, w) map
    signed to correlate positively with the channel RMS, and its variance share."""
    C = f.shape[0]
    X = f.reshape(C, -1).astype(np.float64)
    X = X - X.mean(1, keepdims=True)
    _, s, vt = np.linalg.svd(X, full_matrices=False)
    pc = vt[0].reshape(f.shape[1:])
    rms = np.sqrt((f.astype(np.float64) ** 2).mean(0))
    if np.corrcoef(pc.ravel(), rms.ravel())[0, 1] < 0:
        pc = -pc
    return pc, float(s[0] ** 2 / (s ** 2).sum())


def level_table(z, meta):
    """[(level, (h, w), [NRMSE per outer V-cycle])], coarsest first."""
    rows = []
    for l in sorted(meta["levels"], reverse=True):
        g = z[f"mg/gt/l{l}"]
        rows.append((l, g.shape, [nrmse(z[f"mg/x/o{o}/l{l}"], g)
                                  for o in meta["outers"] if f"mg/x/o{o}/l{l}" in z]))
    return rows


def fig_levels(z, meta, outer=-1, flip=None, resid_gain=5.0, panel=3.2):
    """The decoder side: per level (coarsest -> finest) x_l, R^l(gt), the
    residual, and what the level below handed up, after outer cycle `outer`."""
    import matplotlib.pyplot as plt
    flip = default_flip(meta) if flip is None else flip
    F = (lambda a: _flip(a, flip)) if flip else (lambda a: a)
    o = meta["outers"][outer]
    order = sorted(meta["levels"], reverse=True)
    rows = ["x_l", "R^l(gt)", "|x_l - R^l gt|", "|dx| handed up"]
    fig, ax = plt.subplots(len(rows), len(order), squeeze=False,
                           figsize=(panel * len(order), 0.94 * panel * len(rows)))
    for j, l in enumerate(order):
        g, x = z[f"mg/gt/l{l}"], z[f"mg/x/o{o}/l{l}"]
        vmax = float(np.percentile(np.abs(g), 99.5)) or 1.0
        d = z.get(f"mg/delta/o{o}/l{l}")
        panels = [(np.abs(x), "gray", vmax), (np.abs(g), "gray", vmax),
                  (np.abs(np.abs(x) - np.abs(g)), "inferno", vmax / resid_gain),
                  (None if d is None else np.abs(d), "magma",
                   None if d is None else float(np.percentile(np.abs(d), 99.5)) or 1.0)]
        for i, (im, cmap, vm) in enumerate(panels):
            a = ax[i][j]
            a.set_xticks([]); a.set_yticks([])
            if im is None:
                a.text(.5, .5, "-- (finest level)", ha="center", va="center",
                       color="0.5", transform=a.transAxes)
                a.set_frame_on(False)
                continue
            a.imshow(F(im), cmap=cmap, vmin=0, vmax=vm, interpolation=INTERP)
        ax[0][j].set_title(f"level {l}  {g.shape[0]}x{g.shape[1]}\n"
                           f"NRMSE vs R^l(gt) {100 * nrmse(x, g):.1f}%", fontsize=10)
    for i, r in enumerate(rows):
        ax[i][0].set_ylabel(r, fontsize=11)
    fig.suptitle(f"{meta['run']} (M={meta['M']}, K={meta['K']}, {meta['n_params']:,} params)   "
                 f"{meta['volume']} slice {meta['slice']}   R={meta['R']}  sigma={meta['sigma']:g}\n"
                 f"decoder side after V-cycle {o + 1}/{len(meta['outers'])} "
                 f"(coarsest -> finest; panels at their own grid size)", fontsize=10)
    fig.tight_layout()
    return fig


def fig_outers(z, meta, flip=None, panel=2.4):
    """Every level after each outer V-cycle; title = NRMSE vs R^l(gt)."""
    import matplotlib.pyplot as plt
    flip = default_flip(meta) if flip is None else flip
    F = (lambda a: _flip(a, flip)) if flip else (lambda a: a)
    order, outers = sorted(meta["levels"], reverse=True), meta["outers"]
    fig, ax = plt.subplots(len(order), len(outers), squeeze=False,
                           figsize=(panel * len(outers), panel * len(order)))
    for i, l in enumerate(order):
        g = z[f"mg/gt/l{l}"]
        vmax = float(np.percentile(np.abs(g), 99.5)) or 1.0
        for j, o in enumerate(outers):
            a = ax[i][j]; a.set_xticks([]); a.set_yticks([])
            k = f"mg/x/o{o}/l{l}"
            if k not in z:
                continue
            a.imshow(F(np.abs(z[k])), cmap="gray", vmin=0, vmax=vmax, interpolation=INTERP)
            a.set_title(f"{100 * nrmse(z[k], g):.1f}%", fontsize=9)
        ax[i][0].set_ylabel(f"level {l}", fontsize=11)
    for j, o in enumerate(outers):
        ax[-1][j].set_xlabel(f"V-cycle {o + 1}", fontsize=10)
    fig.suptitle(f"{meta['run']}: x_l after each outer V-cycle (title: NRMSE vs R^l(gt))",
                 fontsize=10)
    fig.tight_layout()
    return fig


def fig_unet(z, meta, flip=None, panel=3.2):
    """The compared U-Net's decoder outputs, coarse -> fine: channel RMS and the
    first principal component. None when nothing was compared."""
    import matplotlib.pyplot as plt
    feats = sorted((k for k in z if k.startswith("vn/feat/")),
                   key=lambda k: int(k.rsplit("/", 1)[1]))
    if not feats:
        return None
    flip = default_flip(meta) if flip is None else flip
    F = (lambda a: _flip(a, flip)) if flip else (lambda a: a)
    fig, ax = plt.subplots(2, len(feats), squeeze=False,
                           figsize=(panel * len(feats), 1.94 * panel))
    for j, k in enumerate(feats):
        f = z[k].astype(np.float32)
        rms = np.sqrt((f ** 2).mean(0))
        pc, var = pc1(f)
        a = ax[0][j]; a.set_xticks([]); a.set_yticks([])
        a.imshow(F(rms), cmap="gray", vmin=0, vmax=float(np.percentile(rms, 99.5)) or 1.0,
                 interpolation=INTERP)
        a.set_title(f"{f.shape[0]} channels  {f.shape[1]}x{f.shape[2]}", fontsize=10)
        v = float(np.percentile(np.abs(pc), 99.5)) or 1.0
        a = ax[1][j]; a.set_xticks([]); a.set_yticks([])
        a.imshow(F(pc), cmap="RdBu_r", vmin=-v, vmax=v, interpolation=INTERP)
        a.set_xlabel(f"PC1 explains {100 * var:.0f}% of channel variance", fontsize=9)
    ax[0][0].set_ylabel("channel RMS", fontsize=11)
    ax[1][0].set_ylabel("1st principal comp.", fontsize=11)
    fig.suptitle(f"{meta['compare']} ({meta['compare_n_params']:,} params): U-Net decoder "
                 f"of cascade {meta['cascade'] + 1}/{meta['n_cascades']}, coarse -> fine "
                 f"-- feature channels, not images", fontsize=10)
    fig.tight_layout()
    return fig


def crop_box(z, meta, flip=None, pad=8):
    """(r0, r1, c0, c1) on the finest DISPLAYED grid around the anatomy (the
    organ mask), padded and snapped to whole coarsest-grid pixels, so
    `a[r0 // 2**l : r1 // 2**l, ...]` is the same region at every level."""
    flip = default_flip(meta) if flip is None else flip
    m = _flip(z["mg/organ/l0"], flip) > 0.5 if flip else z["mg/organ/l0"] > 0.5
    H, W = m.shape
    top = 2 ** max(meta["levels"])
    if not m.any():
        return 0, H, 0, W
    rr, cc = np.where(m.any(1))[0], np.where(m.any(0))[0]
    snap_lo = lambda v: max(v - pad, 0) // top * top
    snap_hi = lambda v, n: min(-(-(v + 1 + pad) // top) * top, n)
    return snap_lo(rr[0]), snap_hi(rr[-1], H), snap_lo(cc[0]), snap_hi(cc[-1], W)


def vcycle_stages(meta):
    """The V-cycle in time order as (level, key, label): down the left arm,
    the coarsest solve, back up the right arm."""
    L = max(meta["levels"])
    seq = []
    for l in range(L):
        seq += [(l, "xin", "enters" if l == 0 else "restricted down"),
                (l, "xpre", "pre-smoothed")]
    seq += [(L, "xin", "restricted down"), (L, "x", "coarsest solve")]
    for l in range(L - 1, -1, -1):
        seq += [(l, "xcorr", "+ coarse correction"), (l, "x", "post-smoothed")]
    return seq


def fig_vcycle(z, meta, outer=-1, show="x", flip=None, crop=True, pad=8,
               panel=2.0, resid_gain=5.0, dpi=None):
    """One V-cycle drawn as a V: a row per level (finest on top), time left to
    right -- enter, pre-smooth, restrict down, ..., coarsest solve, ..., coarse
    correction, post-smooth. `show` = "x" (the iterate) or "resid"
    (|x - R^l gt|). Yellow: NRMSE vs R^l(gt), whole frame. Windows are per
    level, shared along the row."""
    import matplotlib.pyplot as plt
    flip = default_flip(meta) if flip is None else flip
    F = (lambda a: _flip(a, flip)) if flip else (lambda a: a)
    r0, r1, c0, c1 = crop_box(z, meta, flip, pad) if crop else (0, None, 0, None)
    cut = lambda a, l: F(a)[r0 // 2 ** l:(None if r1 is None else r1 // 2 ** l),
                            c0 // 2 ** l:(None if c1 is None else c1 // 2 ** l)]
    o = meta["outers"][outer]
    levels = sorted(meta["levels"])
    seq = vcycle_stages(meta)
    g0 = cut(z["mg/gt/l0"], 0)
    aspect = g0.shape[0] / g0.shape[1]
    fig, ax = plt.subplots(len(levels), len(seq), squeeze=False, dpi=dpi,
                           figsize=(panel * len(seq) + 0.8, panel * aspect * len(levels) + 1.0),
                           gridspec_kw=dict(wspace=0.04, hspace=0.18))
    for a in ax.ravel():
        a.set_axis_off()
    vmax = {l: float(np.percentile(np.abs(cut(z[f"mg/gt/l{l}"], l)), 99.5)) or 1.0
            for l in levels}
    for j, (l, key, label) in enumerate(seq):
        a = ax[levels.index(l)][j]
        a.set_axis_on(); a.set_xticks([]); a.set_yticks([])
        k = f"mg/{key}/o{o}/l{l}"
        if k not in z:                      # the cold start: x = y~ is set inside
            a.text(.5, .5, "cold start\n(x = y~)", ha="center", va="center",
                   color="0.5", transform=a.transAxes, fontsize=8)
        else:
            x, g = z[k], z[f"mg/gt/l{l}"]
            if show == "resid":
                im = np.abs(np.abs(x) - np.abs(g))
                a.imshow(cut(im, l), cmap="inferno", vmin=0, vmax=vmax[l] / resid_gain,
                         interpolation=INTERP)
            else:
                a.imshow(cut(np.abs(x), l), cmap="gray", vmin=0, vmax=vmax[l],
                         interpolation=INTERP)
            a.text(0.03, 0.97, f"{100 * nrmse(x, g):.1f}%", transform=a.transAxes,
                   ha="left", va="top", fontsize=7, color="yellow")
        a.set_title(f"{j + 1}. {label}", fontsize=8)
    for i, l in enumerate(levels):
        g = z[f"mg/gt/l{l}"]
        ax[i][0].set_axis_on(); ax[i][0].set_xticks([]); ax[i][0].set_yticks([])
        ax[i][0].set_frame_on(False)
        ax[i][0].set_ylabel(f"level {l}\n{g.shape[0]}x{g.shape[1]}", fontsize=9)
    fig.suptitle(f"{meta['run']}  {meta['volume']} slice {meta['slice']}: V-cycle "
                 f"{o + 1}/{len(meta['outers'])}, every branch "
                 f"({'|x - R^l gt|' if show == 'resid' else 'x_l'}"
                 f"{', cropped' if crop else ''}; yellow = NRMSE vs R^l(gt))", fontsize=10)
    return fig


def render(npz_path, flip=None, out=None):
    """All three figures from a saved trace, to <stem>_{mg_levels,mg_outers,unet_decoder}.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z, meta = load_npz(npz_path)
    stem = out or os.path.splitext(npz_path)[0]
    written = []
    for name, fig in (("mg_levels", fig_levels(z, meta, flip=flip)),
                      ("mg_outers", fig_outers(z, meta, flip=flip)),
                      ("unet_decoder", fig_unet(z, meta, flip=flip))):
        if fig is None:
            continue
        fig.savefig(f"{stem}_{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(f"{stem}_{name}.png")

    print(f"{meta['run']}  {meta['volume']} slice {meta['slice']}  "
          f"(DC misfit {meta['scale_misfit']:.1e})")
    for l, shape, errs in level_table(z, meta):
        print(f"  level {l}  {shape[0]:>4}x{shape[1]:<4}  NRMSE vs R^l(gt) by V-cycle: "
              + "  ".join(f"{100 * e:5.1f}%" for e in errs))
    print("wrote " + ", ".join(written))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", help="multigrid run dir (config.json + net.ckpt)")
    p.add_argument("--volume", help="val volume name, or any unique part of it")
    p.add_argument("--slice", type=int, help="source slice number")
    p.add_argument("--compare", help="a VarNet run dir: draw its U-Net decoder too")
    p.add_argument("--cascade", type=int, default=-1,
                   help="which VarNet cascade's U-Net (default: the last)")
    p.add_argument("--sigma", type=float, help="default: the run's val_noise_std")
    p.add_argument("--seed", type=int, default=1234, help="eval_per_volume's seed")
    p.add_argument("--val-kspace-type", default="measurement_awgn",
                   help="as the main table (results/main); 'simulated' for the "
                        "simulated-k-space tree")
    p.add_argument("--transfer-taps", type=int,
                   help="implant a longer sinc (e.g. 12, 16) into every transfer of "
                        "--run at inference; default: as trained (8)")
    p.add_argument("--transfer-filter",
                   help="swap the grid-transfer filter of --run at inference: box, "
                        "bilinear, ..., or none (the sinc off); default: as trained")
    p.add_argument("--out", default=os.path.join(ROOT, "figures", "out", "levels"))
    p.add_argument("--from-npz", help="only re-render this .npz (no torch, no data)")
    p.add_argument("--flip", choices=["", "ud", "lr", "both"],
                   help="display flip (default: 'ud' for knee, as the viewer)")
    a = p.parse_args(argv)
    if a.from_npz:
        return render(a.from_npz, a.flip)
    if not (a.run and a.volume and a.slice is not None):
        p.error("--run, --volume and --slice are required (or --from-npz)")
    z, meta, _ = trace_slice(a.run, a.volume, a.slice, compare=a.compare, cascade=a.cascade,
                             sigma=a.sigma, seed=a.seed, val_kspace_type=a.val_kspace_type,
                             transfer_taps=a.transfer_taps, transfer_filter=a.transfer_filter)
    path = save_npz(z, meta, a.out)
    print(f"wrote {path}")
    render(path, a.flip)


if __name__ == "__main__":
    main()
