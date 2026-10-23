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
        # the call itself too: `coarse_solve_test` re-runs the net on it
        store.update(trace=trace, out_embedded=out.detach(), inputs=(y, E, sigma))
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

            def fwd(x, z, y_t, E_, sig, cache, static=None, pi=None):
                out = orig(x, z, y_t, E_, sig, cache, static, pi=pi)
                planar = getattr(m, "is_complex", True) and not x.is_complex()
                r_f = gram_or_x(E_, x, planar) - y_t + m.synthesis_fine(z)
                if pi is not None and getattr(m, "fas_parent_pi", False):
                    r_f = r_f - pi[0]            # the level's own residual
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


# ---------------------------------------------------------------------------
# the coarse-correction step: what a finer level actually receives
# ---------------------------------------------------------------------------
# A finer level never sees the coarse recon w_c itself. PDVCycle adds
#     x_corr = x_pre + alpha_x * P(w_c - u_c),     u_c = R x_pre,
# so only the coarse CHANGE crosses, and only through P: whatever P does not
# pass (the coarse grid's top band) or whatever w_c shares with its start u_c
# is invisible to the level above -- and to the loss.
CORR_COLORS = dict(coarse="#0072B2", prolonged="#E69F00", gt="#7F7F7F",
                   err_c="#009E73", err_p="#D55E00")


def _prolong_np(a, meta):
    """`P` of one (h, w) numpy image, with the net's image-grid transfer."""
    import torch
    from operators.resample import prolong
    t = torch.from_numpy(np.ascontiguousarray(a))[None, None]
    return prolong(t, filter=meta["transfer_filter"])[0, 0].numpy()


def _rms(a):
    return float(np.sqrt(np.mean(np.abs(a) ** 2)))


def correction_parts(z, meta, outer=-1):
    """Every coarse -> fine transition of V-cycle `outer`, coarsest first.

    -> list of dicts, one per fine level l (the level RECEIVING the correction):
        l, lc            fine and coarse level indices (0 = finest)
        w_c              coarse recon after its solve / post-smooth (mg/x, l+1)
        u_c              what the coarse level started from, R x_pre (mg/xin, l+1)
        gt_c, gt         R^{l+1} gt and R^l gt
        delta            w_c - u_c, the only thing the coarse level hands up
        P_w, P_delta     P(w_c), P(delta) on level l's grid
        P_gt_c           P(R^{l+1} gt): P_w - P_gt_c is the coarse error P lets through
        applied          x_corr - x_pre: what level l actually added (= alpha_x P(delta))
        alpha            the scalar fit applied ~ alpha * P(delta), and `alpha_misfit`
        x_pre, x_corr    level l before / after the correction
        visible          RMS(P e_c) / RMS(e_c), e_c = w_c - gt_c: the share of the
                         coarse error that survives P (P preserves RMS on its passband)
    Live mode only for P (torch); the arrays come from `z` (image units, DC in).
    """
    o = meta["outers"][outer]
    L = max(meta["levels"])
    parts = []
    for l in range(L - 1, -1, -1):
        lc = l + 1
        get = lambda name, lev: z.get(f"mg/{name}/o{o}/l{lev}")
        w_c, u_c = get("x", lc), get("xin", lc)
        x_pre, x_corr = get("xpre", l), get("xcorr", l)
        if any(a is None for a in (w_c, u_c, x_pre, x_corr)):
            continue
        gt_c, gt = z[f"mg/gt/l{lc}"], z[f"mg/gt/l{l}"]
        delta = w_c - u_c
        P_w, P_delta, P_gt_c = (_prolong_np(a, meta) for a in (w_c, delta, gt_c))
        applied = x_corr - x_pre
        den = np.vdot(P_delta, P_delta).real
        alpha = np.vdot(P_delta, applied) / den if den > 0 else np.nan
        misfit = (np.linalg.norm(applied - alpha * P_delta) / (np.linalg.norm(applied) + 1e-12)
                  if den > 0 else np.nan)
        e_c = w_c - gt_c
        parts.append(dict(l=l, lc=lc, w_c=w_c, u_c=u_c, gt_c=gt_c, gt=gt, delta=delta,
                          P_w=P_w, P_delta=P_delta, P_gt_c=P_gt_c, applied=applied,
                          alpha=complex(alpha), alpha_misfit=float(misfit),
                          x_pre=x_pre, x_corr=x_corr,
                          visible=_rms(_prolong_np(e_c, meta)) / (_rms(e_c) + 1e-12)))
    return parts


def correction_table(z, meta):
    """One row per (V-cycle, transition): how wrong the coarse recon is, how much
    of that P lets through, and how big the step the fine level takes is."""
    rows = []
    for k in range(len(meta["outers"])):
        for p in correction_parts(z, meta, outer=k):
            rows.append(dict(
                outer=k + 1, transition=f"L{p['lc']}->L{p['l']}",
                # where the coarse level STARTED (R x_pre) vs where it ended: a
                # solve that ends worse than its start moved away from the
                # restricted fine iterate
                nrmse_u_c=nrmse(p["u_c"], p["gt_c"]),
                nrmse_w_c=nrmse(p["w_c"], p["gt_c"]),
                # the fine level just before / after it adds the correction
                nrmse_x_pre=nrmse(p["x_pre"], p["gt"]),
                nrmse_x_corr=nrmse(p["x_corr"], p["gt"]),
                nrmse_P_w=nrmse(p["P_w"], p["P_gt_c"]),
                visible=p["visible"],
                step_rel=_rms(p["applied"]) / (_rms(p["x_pre"]) + 1e-12),
                alpha=p["alpha"], alpha_misfit=p["alpha_misfit"]))
    return rows


def _crop_fn(z, meta, flip, crop, pad):
    F = (lambda a: _flip(a, flip)) if flip else (lambda a: a)
    r0, r1, c0, c1 = crop_box(z, meta, flip, pad) if crop else (0, None, 0, None)
    return lambda a, l: F(a)[r0 // 2 ** l:(None if r1 is None else r1 // 2 ** l),
                             c0 // 2 ** l:(None if c1 is None else c1 // 2 ** l)]


def fig_correction(z, meta, outer=-1, flip=None, crop=True, pad=8, panel=2.0,
                   resid_gain=5.0, dpi=None):
    """The coarse-correction step of V-cycle `outer`, one row per transition
    (coarsest first). Columns: the coarse recon w_c, R gt, its error; P(w_c), the
    part of that error P lets through; what the coarse level hands up, P(w_c - u_c);
    what the fine level adds, alpha P(w_c - u_c); the fine level before and after.
    Images on one window per row (|.|); error and step panels on vmax/resid_gain."""
    import matplotlib.pyplot as plt
    flip = default_flip(meta) if flip is None else flip
    cut = _crop_fn(z, meta, flip, crop, pad)
    parts = correction_parts(z, meta, outer)
    cols = [("w_c", "coarse recon w_c", "img", "c"),
            ("gt_c", "R gt (coarse)", "img", "c"),
            ("err_c", "|w_c - R gt|", "res", "c"),
            ("P_w", "P(w_c)  [% vs P(R gt)]", "img", "f"),
            ("err_p", "|P(w_c - R gt)|", "res", "f"),
            ("P_delta", "|P(w_c - u_c)|  handed up", "step", "f"),
            ("applied", "|alpha P(.)|  added", "step", "f"),
            ("x_pre", "fine, before", "img", "f"),
            ("x_corr", "fine, after", "img", "f")]
    g0 = cut(z["mg/gt/l0"], 0)
    aspect = g0.shape[0] / g0.shape[1]
    fig, ax = plt.subplots(len(parts), len(cols), squeeze=False, dpi=dpi,
                           figsize=(panel * len(cols) + 0.8, panel * aspect * len(parts) + 1.0),
                           gridspec_kw=dict(wspace=0.04, hspace=0.25))
    for i, p in enumerate(parts):
        p = dict(p, err_c=p["w_c"] - p["gt_c"], err_p=p["P_w"] - p["P_gt_c"])
        vmax = float(np.percentile(np.abs(cut(p["gt"], p["l"])), 99.5)) or 1.0
        # the two step panels share their OWN window: a step is far smaller
        # than an error, and on the error window it reads as black
        vstep = float(np.percentile(np.abs(cut(p["P_delta"], p["l"])), 99.5)) or 1.0
        for j, (key, title, kind, grid) in enumerate(cols):
            a = ax[i][j]
            lev = p["lc"] if grid == "c" else p["l"]
            im = np.abs(cut(p[key], lev))
            if kind == "img":
                a.imshow(im, cmap="gray", vmin=0, vmax=vmax, interpolation=INTERP)
            elif kind == "res":
                a.imshow(im, cmap="inferno", vmin=0, vmax=vmax / resid_gain,
                         interpolation=INTERP)
            else:
                a.imshow(im, cmap="inferno", vmin=0, vmax=vstep, interpolation=INTERP)
            a.set_xticks([]); a.set_yticks([])
            if key in ("w_c", "P_w", "x_pre", "x_corr"):
                # P(w_c) against P(R gt): the coarse error P lets through, not the
                # fine band no coarse grid can hold
                g = {"w_c": p["gt_c"], "P_w": p["P_gt_c"]}.get(key, p["gt"])
                a.text(0.03, 0.97, f"{100 * nrmse(p[key], g):.1f}%", transform=a.transAxes,
                       ha="left", va="top", fontsize=7, color="yellow")
            elif kind == "step":         # RMS relative to the fine image, own window
                a.text(0.03, 0.97, f"rms {100 * _rms(p[key]) / (_rms(p['x_pre']) + 1e-12):.2f}%"
                       f" of x\nwindow 0-{vstep:.2g}", transform=a.transAxes,
                       ha="left", va="top", fontsize=6, color="cyan")
            if i == 0:
                a.set_title(title, fontsize=8)
        ax[i][0].set_ylabel(f"L{p['lc']} -> L{p['l']}\nvisible {100 * p['visible']:.0f}%\n"
                            f"alpha {p['alpha'].real:+.3g}", fontsize=8)
    o = meta["outers"][outer]
    fig.suptitle(f"{meta['run']}  {meta['volume']} slice {meta['slice']}: coarse correction, "
                 f"V-cycle {o + 1}/{len(meta['outers'])}  (L0 = finest; yellow = NRMSE vs "
                 f"R^l gt; 'visible' = RMS of P(coarse error) / RMS of coarse error)",
                 fontsize=10)
    return fig


def _spectrum2d(a):
    return np.fft.fftshift(np.fft.fft2(a, norm="forward"))


def _axis_power(a, axis):
    """Power vs frequency along `axis` (summed over the other axis), with the
    frequencies in cycles per pixel OF THIS GRID. `norm="forward"` and the sum
    make an image and its interpolation P(.) agree on the passband."""
    S = np.abs(_spectrum2d(a)) ** 2
    pw = S.sum(axis=1 - axis)
    f = np.fft.fftshift(np.fft.fftfreq(a.shape[axis]))
    return f, pw


def transfer_response(meta, n=64):
    """|H(f)| of P along one axis, normalised to 1 at DC; f in cycles per
    FINE pixel."""
    d = np.zeros((n // 2, n // 2), np.complex64)
    d[n // 4, n // 4] = 1.0
    f, pw = _axis_power(_prolong_np(d, meta), axis=0)
    h = np.sqrt(pw / pw[np.argmin(np.abs(f))])
    return f, h


def fig_correction_spectra(z, meta, outer=-1, flip=None, db_floor=-110.0, panel=3.0,
                           dpi=None):
    """Frequency content of the coarse recon and of what crosses the transfer.

    Per transition (rows, coarsest first): log|F| of w_c (coarse grid) and of
    P(w_c) (fine grid; dashed box = the coarse grid's band), the same for the
    coarse ERROR w_c - R gt, then 1-D power along each axis on a common axis in
    cycles per FINE pixel (coarse Nyquist at +-0.25), with |H| of P underneath.
    `flip` only orients the 2-D panels."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    flip = default_flip(meta) if flip is None else flip
    F = (lambda a: _flip(a, flip)) if flip else (lambda a: a)
    parts = correction_parts(z, meta, outer)
    n = len(parts)
    fig, ax = plt.subplots(n + 1, 6, squeeze=False, dpi=dpi,
                           figsize=(panel * 6 + 0.6, panel * n + 0.45 * panel + 1.6),
                           gridspec_kw=dict(wspace=0.3, hspace=0.5,
                                            height_ratios=[1] * n + [0.45]))
    for A in ax[n]:
        A.set_axis_off()
    C = CORR_COLORS
    fH, H = transfer_response(meta)
    for i, p in enumerate(parts):
        e_c = p["w_c"] - p["gt_c"]
        e_p = p["P_w"] - p["P_gt_c"]
        spec = [(p["w_c"], "w_c", "c"), (p["P_w"], "P(w_c)", "f"),
                (e_c, "w_c - R gt", "c"), (e_p, "P(w_c - R gt)", "f")]
        logs = [20 * np.log10(np.abs(_spectrum2d(F(a))) + 1e-12) for a, _, _ in spec]
        vmax = max(float(np.percentile(s, 99.9)) for s in logs)
        for j, ((a, title, grid), s) in enumerate(zip(spec, logs)):
            h, w = a.shape
            A = ax[i][j]
            A.set_facecolor("black")
            # every panel in cycles per FINE pixel: the coarse grid's spectrum
            # covers +-0.25, the dashed box
            e = .25 if grid == "c" else .5
            A.imshow(s, cmap="magma", vmin=vmax - 80, vmax=vmax,
                     extent=(-e, e, e, -e), interpolation="nearest")
            A.add_patch(Rectangle((-.25, -.25), .5, .5, fill=False, ls="--",
                                  lw=1.0, ec="white"))
            A.set_xlim(-.5, .5); A.set_ylim(.5, -.5)
            A.set_xticks([-.5, -.25, 0, .25, .5]); A.set_yticks([-.5, -.25, 0, .25, .5])
            A.tick_params(labelsize=6)
            A.set_title(f"{title}  ({h}x{w})", fontsize=8)
            if j == 0:
                A.set_ylabel(f"L{p['lc']} -> L{p['l']}\nky (cycles / fine px)", fontsize=8)
        # 1-D: rows (axis 0) and columns (axis 1), cycles per FINE pixel
        for j, axis in ((4, 0), (5, 1)):
            A = ax[i][j]
            for a, grid, label, color, ls in (
                    (p["gt"], "f", "R gt (fine)", C["gt"], "-"),
                    (p["w_c"], "c", "w_c", C["coarse"], "-"),
                    (p["P_w"], "f", "P(w_c)", C["prolonged"], "-"),
                    (e_c, "c", "w_c - R gt", C["err_c"], "--"),
                    (e_p, "f", "P(w_c - R gt)", C["err_p"], "--")):
                f, pw = _axis_power(a, axis)
                f = f / 2 if grid == "c" else f
                A.plot(f, 10 * np.log10(pw + 1e-30), color=color, lw=1.2, ls=ls, label=label)
            A.axvline(-.25, color="0.6", lw=0.8, ls=":"); A.axvline(.25, color="0.6", lw=0.8, ls=":")
            A.set_xlim(-.5, .5)
            lo = max(db_floor, A.get_ylim()[0])
            A.set_ylim(lo, A.get_ylim()[1])
            A.tick_params(labelsize=6); A.grid(alpha=.25)
            A.set_title(f"power along axis {axis} ({'rows' if axis == 0 else 'cols'})",
                        fontsize=8)
            A.set_xlabel("cycles / fine pixel", fontsize=7)
            if j == 5 and i == 0:
                A.legend(fontsize=6, frameon=False, loc="lower center")
    o = meta["outers"][outer]
    fig.suptitle(f"{meta['run']}: spectra at the coarse correction, V-cycle {o + 1}/"
                 f"{len(meta['outers'])}  (2-D: 20 log10 |F|, 80 dB window per row, all in "
                 f"cycles / fine px; dashed box / dotted lines = the coarse grid's band)",
                 fontsize=10)
    # the transfer itself, under the 1-D panels: what P does to each frequency
    for j in (4, 5):
        A = ax[n][j]
        A.set_axis_on()
        A.plot(fH, 20 * np.log10(H + 1e-12), color="0.2", lw=1.2)
        A.axvline(-.25, color="0.6", lw=0.8, ls=":"); A.axvline(.25, color="0.6", lw=0.8, ls=":")
        A.set_xlim(-.5, .5); A.set_ylim(-60, 5); A.tick_params(labelsize=6)
        A.set_title(f"|H| of P ({meta['transfer_filter']}), dB", fontsize=8)
        A.set_xlabel("cycles / fine pixel", fontsize=7); A.grid(alpha=.25)
    return fig


# ---------------------------------------------------------------------------
# is the coarse level solving its problem -- and is that the right problem?
# ---------------------------------------------------------------------------
# The learned coarse level steps on ITS OWN problem: its own analysis A_1,
# synthesis B_1 and threshold lambda_1, tilted by the FAS correction pi. The
# fine problem restricted to coarse images (Galerkin) is a different one:
#     min_{x_c}  1/2 ||E P x_c - y||^2 + lambda_0 ||A_0 P x_c||_1   (x 1/4)
# `coarse_solve_test` runs, from the SAME restricted start u_c and on the same
# slice, several fixed coarse solvers to (near) convergence and scores every
# iterate against R gt -- and what it would do to the fine level, x_pre + P(x_c - u_c):
#   "C1"          the learned coarse level's FIRST layer, iterated with the net's
#                 own pi: where the learned coarse problem's solution is
#   "galerkin"    the fine first layer F_1 on P x_c (z on the FINE latent grid):
#                 exactly the fine problem restricted to coarse images, FAS-tilted
#                 with the correct nested pi
#   "galerkin_R"  as "galerkin" with the coefficients restricted too,
#                 A = R_z A_0 P, B = R B_0 P_z (z on the coarse latent grid):
#                 what a Galerkin-prior coarse level would compute in the net
#   "galerkin_untilted"  "galerkin" with pi = 0: the best coarse image under
#                 the fine model, with no FAS correction at all
# Every solver is an LPDS layer with fixed weights (the learned tau / theta of
# the layer it came from), iterated `n_iter` times; `step` reports whether it
# converged. Galerkin modes use the exact Galerkin data term R E^H(E P x - y),
# whatever coarse operator the run was trained with.
COARSE_SOLVERS = ("C1", "galerkin", "galerkin_R", "galerkin_untilted")


def _vcycle_index(model):
    """`{(outer, level): PDVCycle}`, level 0 finest."""
    from models.mg_lpds import PDVCycle
    cycles = [m for m in model.modules() if isinstance(m, PDVCycle)]
    inner = {id(c.mglayer) for c in cycles}
    out = {}
    for k, V in enumerate(c for c in cycles if id(c) not in inner):
        level = 0
        while isinstance(V, PDVCycle):
            out[(k, level)] = V
            V, level = V.mglayer, level + 1
    return out


def _capture_vcycles(model, y, E, sigma):
    """One forward; per (outer, level) what the FAS downsampler received and
    returned, and what the coarse level handed back. -> `{(k, l): dict}`."""
    import torch
    index = _vcycle_index(model)
    rec, patched = {}, []

    def det(t):
        return t.detach() if torch.is_tensor(t) else t

    for key, V in index.items():
        d = rec.setdefault(key, {})

        def wrap_dF(m, d=d):
            orig = m.forward

            def fwd(x, z, y_t, E_, sig, cache, static=None, pi=None):
                out = orig(x, z, y_t, E_, sig, cache, static, pi=pi)
                d.update(x=det(x), z=det(z), y=det(y_t), E=E_, sigma=det(sig),
                         pi=None if pi is None else tuple(det(p) for p in pi),
                         x_c=det(out[0]), z_c=det(out[1]), pi_c=(det(out[2]), det(out[3])),
                         y_c=det(out[4]), E_c=out[5], sigma_c=det(out[6]))
                return out
            m.forward = fwd
            patched.append(m)

        def wrap_coarse(m, d=d):
            orig = m.forward

            def fwd(*a, **kw):
                out = orig(*a, **kw)
                state = out[0] if isinstance(out, tuple) else out
                d.update(w_x=det(state[0]), w_z=det(state[1]))
                return out
            m.forward = fwd
            patched.append(m)

        wrap_dF(V.dF)
        wrap_coarse(V.mglayer)
    try:
        with torch.no_grad():
            model(y, E=E, sigma=sigma)
    finally:
        for m in patched:
            del m.forward
    return rec, index


def _np_img(t, mu):
    """A DC-removed traced image as complex numpy (H, W), DC added back."""
    return (_as_complex(t)[0, 0].cpu().numpy() + mu).astype(np.complex64)


def _pd_run(x, z, data_grad, A, B, prox, tau, theta, pi, mode, n_iter, every, score):
    """`n_iter` LPDS steps with fixed operators (LPDSLayer.forward's update,
    `mode` = where pi_z enters). -> (x, z, records)."""
    pi_x, pi_z = (None, None) if pi is None else pi
    out = (lambda zn: zn + pi_z) if (pi_z is not None and mode == "fixed_point") else None
    recs = []
    for t in range(1, n_iter + 1):
        r = data_grad(x) + B(z)
        if pi_x is not None:
            r = r - pi_x
        xn = x - tau * r
        xb = xn + theta * (xn - x)
        u = z + A(xb)
        if pi_z is not None and out is None:
            u = u - pi_z
        zn = prox(u)
        if out is not None:
            zn = out(zn)
        if t == 1 or t % every == 0 or t == n_iter:
            step = float((xn - x).norm() / (x.norm() + 1e-12))
            step_z = float((zn - z).norm() / (z.norm() + 1e-12))
            recs.append(dict(it=t, step=step, step_z=step_z, **score(xn)))
        x, z = xn, zn
    return x, z, recs


def _residual(x, z, data_grad, A, B, prox, normal_cone, mode, pi=None):
    """The level's FAS residual (primal, dual) at (x, z) -- net of `pi` when
    given, in the form that is zero at the fixed point of `mode`'s sweep
    (models/mg_lpds.py::_check_fas_parent_pi)."""
    pi_x, pi_z = (None, None) if pi is None else pi
    rx = data_grad(x) + B(z)
    if pi_x is not None:
        rx = rx - pi_x
    Ax = A(x)
    if mode == "gradient":
        w = Ax if pi_z is None else Ax - pi_z
        rz = w - normal_cone(z, w)
    elif mode == "fixed_point":
        rz = z - prox(z + Ax)
        if pi_z is not None:
            rz = rz - pi_z
    else:                                                        # legacy
        rz = z - prox(z + Ax if pi_z is None else z + Ax - pi_z)
    return rx, rz


def coarse_solve_test(z, meta, live, outer=-1, level=0, n_iter=300, every=5,
                      solvers=COARSE_SOLVERS):
    """Fixed coarse solvers from the learned coarse level's own start, on the
    transition `level` -> `level + 1` of V-cycle `outer` (live mode).

    -> dict: `refs` (NRMSEs of u_c, the learned w_c, x_pre, x_corr), `own`
    (the learned coarse level's one-step change under its first layer at its
    start and at its end -- did the learned solve reduce its own residual?),
    and `runs[solver]` = {"recs": per-iteration NRMSE / step, "x_c", "x_f"}.
    """
    import torch
    from models.lpds import _per_channel
    from models.mg_lpds import first_pd_layer, gram_or_x
    from operators.resample import prolong, restrict

    model = live["mg"]["model"]
    y, E, sigma = live["mg"]["store"]["inputs"]
    ctx = as_traced(live) if "entry" in live["mg"] else contextlib.nullcontext()
    with ctx:
        rec, index = _capture_vcycles(model, y, E, sigma)
        k = sorted({kk for kk, _ in rec})[outer]
        if (k, level) not in rec:
            raise ValueError(f"no V-cycle at level {level} (levels "
                             f"{sorted({l for _, l in rec})})")
        d, V = rec[(k, level)], index[(k, level)]
        if V.widen != 1:
            raise NotImplementedError("widen > 1: the coarse code has more channels")
        mode = V.fas_dual
        filt = V.transfer_filter
        Ri = lambda t: restrict(t, julia_compat=V.julia_compat, filter=filt)  # noqa: E731
        Pi = lambda t: prolong(t, filter=filt)                                # noqa: E731
        if V.transfer is not None:
            Rz, Pz = V.transfer.restrict, V.transfer.prolong
        else:
            Rz, Pz = Ri, Pi
        F1, C1 = V.lpdsA.first, first_pd_layer(V.mglayer)
        x, zf, sig, sig_c = d["x"], d["z"], d["sigma"], d["sigma_c"]
        planar = F1.is_complex and not x.is_complex()
        mu = complex(*meta["dc"])
        gt_c, gt_f = z[f"mg/gt/l{level + 1}"], z[f"mg/gt/l{level}"]

        def steps(layer, s, ref):
            tau, theta = layer.tau(s, ref=ref), layer.theta(s, ref=ref)
            if planar:
                tau, theta = _per_channel(tau, ref), _per_channel(theta, ref)
            return tau, theta

        def score_for(u_x):
            def score(xc):
                xf = x + Pi(xc - u_x)
                return dict(coarse=nrmse(_np_img(xc, mu), gt_c),
                            fine=nrmse(_np_img(xf, mu), gt_f))
            return score

        with torch.no_grad():
            # ---- references: what the net did ------------------------------
            u_x, u_z, w_x, w_z = d["x_c"], d["z_c"], d["w_x"], d["w_z"]
            x_corr = x + V.alpha_x(Pi(w_x - u_x))
            refs = dict(u_c=nrmse(_np_img(u_x, mu), gt_c),
                        w_c=nrmse(_np_img(w_x, mu), gt_c),
                        x_pre=nrmse(_np_img(x, mu), gt_f),
                        x_corr=nrmse(_np_img(x_corr, mu), gt_f),
                        x_pre_plus_P=nrmse(_np_img(x + Pi(w_x - u_x), mu), gt_f))

            # ---- the learned coarse level's own residual (first layer) -------
            def one_step(xs, zs):
                (xn, zn), _ = C1((xs, zs), d["y_c"], E=d["E_c"], sigma=sig_c,
                                 pi=d["pi_c"])
                return (float((xn - xs).norm() / (xs.norm() + 1e-12)),
                        float((zn - zs).norm() / (zs.norm() + 1e-12)))
            own = dict(start=one_step(u_x, u_z), end=one_step(w_x, w_z))

            # ---- the fine level's own residual, net of its own pi ------------
            fprox = lambda u: F1.prox(u, sig, {})[0]                         # noqa: E731
            fcone = lambda zz, w: F1.prox.normal_cone(zz, w, sig)            # noqa: E731
            fdata = lambda xx: gram_or_x(d["E"], xx, planar) - d["y"]        # noqa: E731
            s_x, s_z = _residual(x, zf, fdata, F1.analysis, F1.synthesis, fprox, fcone,
                                 mode, pi=d["pi"])
            # exact Galerkin data term on coarse images: R E^H (E P x - y)
            gdata = lambda xc: Ri(gram_or_x(d["E"], Pi(xc), planar) - d["y"])  # noqa: E731

            runs = {}
            for name in solvers:
                if name == "C1":
                    cprox = lambda u: C1.prox(u, sig_c, {})[0]               # noqa: E731
                    cdata = lambda xc: gram_or_x(d["E_c"], xc, planar) - d["y_c"]  # noqa: E731
                    tau, theta = steps(C1, sig_c, u_x)
                    xc, zc, recs = _pd_run(u_x, u_z, cdata, C1.analysis, C1.synthesis, cprox,
                                           tau, theta, d["pi_c"], C1.fas_dual, n_iter, every,
                                           score_for(u_x))
                else:
                    restricted = name == "galerkin_R"
                    if restricted:       # z on the COARSE latent grid
                        A = lambda xc: Rz(F1.analysis(Pi(xc)))                # noqa: E731
                        B = lambda zc: Ri(F1.synthesis(Pz(zc)))               # noqa: E731
                        z0, Rd = Rz(zf), Rz
                    else:                # z stays on the FINE latent grid
                        A = lambda xc: F1.analysis(Pi(xc))                    # noqa: E731
                        B = lambda zc: Ri(F1.synthesis(zc))                   # noqa: E731
                        z0, Rd = zf, (lambda t: t)
                    if name == "galerkin_untilted":
                        pi = None
                    else:
                        cone = lambda zz, w: F1.prox.normal_cone(zz, w, sig)  # noqa: E731
                        r_x, r_z = _residual(u_x, z0, gdata, A, B, fprox, cone, mode)
                        pi = (r_x - Ri(s_x), r_z - Rd(s_z))
                    tau, theta = steps(F1, sig, u_x)
                    xc, zc, recs = _pd_run(u_x, z0, gdata, A, B, fprox, tau, theta, pi,
                                           mode, n_iter, every, score_for(u_x))
                runs[name] = dict(recs=recs, x_c=_np_img(xc, mu),
                                  x_f=_np_img(x + Pi(xc - u_x), mu))
            imgs = dict(u_c=_np_img(u_x, mu), w_c=_np_img(w_x, mu), x_pre=_np_img(x, mu),
                        x_corr=_np_img(x_corr, mu), gt_c=gt_c, gt_f=gt_f)
    return dict(outer=k, level=level, mode=mode, n_iter=n_iter, refs=refs, own=own,
                runs=runs, imgs=imgs, run=meta["run"],
                fas_parent_pi=bool(getattr(V, "fas_parent_pi", False)))


SOLVER_COLORS = {"C1": "#0072B2", "galerkin": "#D55E00", "galerkin_R": "#E69F00",
                 "galerkin_untilted": "#009E73"}
SOLVER_LABELS = {"C1": "learned coarse layer 1, iterated",
                 "galerkin": "Galerkin (fine F1 on P x_c)",
                 "galerkin_R": "Galerkin, restricted coeffs",
                 "galerkin_untilted": "Galerkin, no FAS tilt"}


def coarse_solve_table(res):
    """Print the comparison; returns nothing."""
    L, Lc = res["level"], res["level"] + 1
    r = res["refs"]
    print(f"{res['run']}  V-cycle {res['outer'] + 1}, L{Lc} -> L{L}  "
          f"(fas_dual={res['mode']}, fas_parent_pi={res['fas_parent_pi']})")
    print(f"  learned coarse solve: L{Lc} NRMSE {100 * r['u_c']:.2f}% at its start (R x_pre) "
          f"-> {100 * r['w_c']:.2f}% at its end")
    print(f"  L{L}: x_pre {100 * r['x_pre']:.2f}%  -> + alpha P(w - u) {100 * r['x_corr']:.2f}%"
          f"   (with alpha = 1: {100 * r['x_pre_plus_P']:.2f}%)")
    (sx0, sz0), (sx1, sz1) = res["own"]["start"], res["own"]["end"]
    print(f"  its OWN residual (one step of its first layer, relative change x / z): "
          f"start {sx0:.2e} / {sz0:.2e}  ->  end {sx1:.2e} / {sz1:.2e}")
    print(f"\n  {'solver (n_iter=' + str(res['n_iter']) + ')':<36}{'L' + str(Lc) + ' best':>9}"
          f"{'@it':>5}{'final':>8}{'L' + str(L) + ' (alpha=1)':>14}{'final step x/z':>18}")
    for name, run in res["runs"].items():
        recs = run["recs"]
        best = min(recs, key=lambda q: q["coarse"])
        last = recs[-1]
        print(f"  {SOLVER_LABELS.get(name, name):<36}{100 * best['coarse']:>8.2f}%{best['it']:>5}"
              f"{100 * last['coarse']:>7.2f}%{100 * last['fine']:>13.2f}%"
              f"{last['step']:>10.1e}/{last['step_z']:.1e}")


def fig_coarse_solve(res, z=None, meta=None, flip=None, crop=True, pad=8, panel=2.2,
                     dpi=None):
    """Curves (coarse NRMSE vs R gt, fine NRMSE with alpha = 1, step size) per
    solver against the learned level's references, then the final images."""
    import matplotlib.pyplot as plt
    L, Lc = res["level"], res["level"] + 1
    if z is not None and meta is not None:
        flip = default_flip(meta) if flip is None else flip
        cut = _crop_fn(z, meta, flip, crop, pad)
    else:
        cut = lambda a, l: a                                                 # noqa: E731
    names = list(res["runs"])
    ncol = 3 + len(names)
    fig = plt.figure(figsize=(panel * ncol + 0.6, 3 * panel + 1.2), dpi=dpi)
    gs = fig.add_gridspec(3, ncol, height_ratios=[1.15, 1, 1], hspace=0.35, wspace=0.08)
    r = res["refs"]
    for j, (key, ylabel, refs) in enumerate((
            ("coarse", f"L{Lc} NRMSE vs R gt (%)",
             [(r["u_c"], "start u_c = R x_pre", ":"), (r["w_c"], "learned w_c", "--")]),
            ("fine", f"L{L} NRMSE, x_pre + P(x_c - u_c) (%)",
             [(r["x_pre"], "x_pre", ":"), (r["x_corr"], "net: + alpha P(w - u)", "--")]),
            ("step", "relative step |x+ - x| / |x|", []))):
        A = fig.add_subplot(gs[0, j * ncol // 3:(j + 1) * ncol // 3])
        for name in names:
            recs = res["runs"][name]["recs"]
            it = [q["it"] for q in recs]
            v = [q[key] * (1 if key == "step" else 100) for q in recs]
            A.plot(it, v, color=SOLVER_COLORS.get(name, "0.3"), lw=1.6,
                   label=SOLVER_LABELS.get(name, name))
        for val, lab, ls in refs:
            A.axhline(100 * val, color="0.35", lw=1.0, ls=ls, label=lab)
        if key == "step":
            A.set_yscale("log")
        A.set_xlabel("iteration", fontsize=8); A.set_title(ylabel, fontsize=9)
        A.tick_params(labelsize=7); A.grid(alpha=.25)
        A.legend(fontsize=6.5, frameon=False)
    im = res["imgs"]
    vmax = {l: float(np.percentile(np.abs(cut(im["gt_f" if l == L else "gt_c"], l)), 99.5)) or 1.0
            for l in (L, Lc)}
    rows = [(Lc, [("R gt", im["gt_c"]), ("start u_c", im["u_c"]), ("learned w_c", im["w_c"])]
             + [(SOLVER_LABELS.get(n, n), res["runs"][n]["x_c"]) for n in names], im["gt_c"]),
            (L, [("gt", im["gt_f"]), ("x_pre", im["x_pre"]), ("net: x_corr", im["x_corr"])]
             + [("x_pre + P(.)", res["runs"][n]["x_f"]) for n in names], im["gt_f"])]
    for i, (lev, cells, g) in enumerate(rows):
        for j, (title, a) in enumerate(cells):
            A = fig.add_subplot(gs[1 + i, j])
            A.imshow(np.abs(cut(a, lev)), cmap="gray", vmin=0, vmax=vmax[lev],
                     interpolation=INTERP)
            A.set_xticks([]); A.set_yticks([])
            if j > 0:
                A.text(0.03, 0.97, f"{100 * nrmse(a, g):.1f}%", transform=A.transAxes,
                       ha="left", va="top", fontsize=7, color="yellow")
            A.set_title(title, fontsize=7)
            if j == 0:
                A.set_ylabel(f"L{lev}", fontsize=9)
    fig.suptitle(f"{res['run']}: the coarse problem, V-cycle {res['outer'] + 1}, L{Lc} -> L{L}"
                 f"  (fixed solvers from the learned level's start; images at the last "
                 f"iteration, n={res['n_iter']})", fontsize=10)
    return fig


# ---------------------------------------------------------------------------
# is the dual code saturated? (the clip boundary, per coefficient)
# ---------------------------------------------------------------------------
# The dual step z+ = clip(z + A xbar) is a sparsity mechanism only while the
# per-sweep increments A xbar are small against lambda: a noise coefficient's
# increments then average out and z stays INSIDE the box (= thresholded to
# zero). Once |A xbar| > |z| + lambda the sum leaves the box whatever z was --
# the clip returns lambda * phase(A xbar), the code's history is erased, and
# D z becomes lambda * D sign(A xbar): a noise coefficient pushes on x as hard as
# an edge. `trace_saturation` measures this per LPDS sweep, no weight changed.
SAT_LOG_BINS = np.linspace(-4.0, 3.0, 141)        # log10(|inc| / lambda)


def _code_modulus(t, channels):
    """|t| per coefficient: complex, PLANAR ([re; im] on the channel axis) or real."""
    import torch
    if t.is_complex():
        return t.abs()
    if t.shape[1] == 2 * channels:
        return torch.hypot(t[:, :channels], t[:, channels:])
    return t.abs()


def _saturation_call(store, select, masks):
    def call(net, y, E, sigma):
        import torch
        import torch.nn.functional as F
        from models.lpds import _per_channel

        recs, patched = [], []
        warned = set()
        last_clip = {}           # (outer, level) -> the last sweep's clip output there

        def stats(level, mod_out, mod_ax, mod_z, mod_pi, lam, dot=None, mod_prev=None,
                  extra=None):
            out = {}
            m = masks.get(level) if masks else None
            for scope in ("all", "anatomy"):
                if scope == "anatomy":
                    if m is None or m.shape[-2:] != mod_ax.shape[-2:]:
                        continue
                    sel = m.expand_as(mod_ax)
                else:
                    sel = torch.ones_like(mod_ax, dtype=torch.bool)
                lam_b = lam.expand_as(mod_ax)[sel]
                a = mod_ax[sel] / lam_b
                zr = mod_z[sel] / lam_b
                on = mod_out[sel] >= 0.99 * lam_b
                reset = mod_ax[sel] > mod_z[sel] + lam_b
                d = dict(sat=float(on.float().mean()),
                         reset=float(reset.float().mean()),
                         # self-checks, both 0 unless the clip input / output
                         # were mis-identified: an output past the box, and a
                         # reset coefficient (|inc| > |z| + lambda) not on it
                         over=float((mod_out[sel] > 1.01 * lam_b).float().mean()),
                         reset_off=float((reset & ~on).float().mean()),
                         a_med=float(a.median()),
                         a_p90=float(torch.quantile(a.float()[:2 ** 24 - 1], 0.9)),
                         z_med=float(zr.median()),
                         hist=np.histogram(np.log10(a.float().clamp_min(1e-30).cpu().numpy()),
                                           SAT_LOG_BINS)[0])
                if mod_pi is not None:
                    pr = mod_pi[sel] / lam_b
                    d.update(pi_over=float((pr > 1).float().mean()), pi_med=float(pr.median()))
                if dot is not None:
                    # phase reversals: saturated now AND in the previous sweep's
                    # clip output at this level, with the phase turned > 90 deg
                    both = on & (mod_prev[sel] >= 0.99 * lam_b)
                    n = int(both.sum())
                    d["flip"] = float((dot[sel][both] < 0).float().sum() / n) if n else float("nan")
                d.update(extra or {})
                out[scope] = d
            return out

        def wrap(L, level, where, outer):
            orig = L.forward
            soft = getattr(getattr(L, "prox", None), "prox", None)
            if not hasattr(soft, "threshold"):
                if type(soft).__name__ not in warned:
                    warned.add(type(soft).__name__)
                    print(f"trace_saturation: {type(soft).__name__} has no scalar threshold "
                          f"(group prox?) -- its sweeps are skipped", file=sys.stderr)
                return

            def fwd(state, y_tilde, E=None, sigma=None, pi=None, cache=None):
                out = orig(state, y_tilde, E=E, sigma=sigma, pi=pi, cache=cache)
                if state is None:
                    return out
                x, z = state
                x_new, z_new = out[0]
                planar = L.is_complex and not x.is_complex()
                theta = L.theta(sigma, ref=x)
                theta = _per_channel(theta, x) if planar else theta
                Ax = L.analysis(x_new + theta * (x_new - x))
                pi_z = None if pi is None else pi[1]
                fixed = pi_z is not None and getattr(L, "fas_dual", "legacy") == "fixed_point"
                clip_out = z_new - pi_z if fixed else z_new          # option A adds pi_z AFTER
                # the increment the clip adds to z: A xbar, minus pi_z where pi_z
                # goes INSIDE the clip ("gradient", "legacy")
                inc = Ax if (pi_z is None or fixed) else Ax - pi_z
                lam = soft.threshold(z, sigma)
                M = L.M
                key = (outer, level)
                prev = last_clip.get(key)
                if prev is not None and prev.shape != clip_out.shape:
                    prev = None
                dot = mod_prev = None
                extra = dict(step=float((x_new - x).norm() / (x.norm() + 1e-12)))
                if prev is not None:
                    if clip_out.is_complex():
                        dot = (prev.conj() * clip_out).real
                    elif clip_out.shape[1] == 2 * M:
                        dot = prev[:, :M] * clip_out[:, :M] + prev[:, M:] * clip_out[:, M:]
                    else:
                        dot = prev * clip_out
                    mod_prev = _code_modulus(prev, M)
                    # how hard the code CHANGE pushes on x (through this layer's
                    # D and tau -- the next sweep applies its own, nearly equal)
                    tau = L.tau(sigma, ref=x)
                    tau = _per_channel(tau, x) if planar else tau
                    extra["push"] = float((tau * L.synthesis(clip_out - prev)).norm()
                                          / (x.norm() + 1e-12))
                last_clip[key] = clip_out
                recs.append(dict(
                    level=level, where=where, outer=outer,
                    fas_dual=getattr(L, "fas_dual", "legacy"),
                    lam=float(lam.float().median()),
                    **stats(level, _code_modulus(clip_out, M), _code_modulus(inc, M),
                            _code_modulus(z, M),
                            None if pi_z is None else _code_modulus(pi_z, M), lam,
                            dot=dot, mod_prev=mod_prev, extra=extra)))
                return out
            L.forward = fwd
            patched.append(L)

        for kind, level, where, outer, m in _vcycle_walk(net):
            if kind == "layer" and select(level, outer):
                wrap(m, level, where, outer)
        try:
            with torch.no_grad():
                out = net(y, E=E, sigma=sigma)
        finally:
            for m in patched:
                del m.forward
        store.update(recs=recs)
        return out[0] if isinstance(out, tuple) else out
    return call


def trace_saturation(run, volume, slice, levels=(0, 1, 2), outers=None, z=None, sigma=None,
                     seed=1234, val_kspace_type="measurement_awgn", device=None,
                     coarse_op=None, transfer_filter=None):
    """Per LPDS sweep of `run` on `levels` in V-cycles `outers` (None: the last;
    indices into the outer cycles, e.g. [0, -1]): how saturated the dual code is.

    -> `(recs, meta)`, one record per sweep in execution order: level, where,
    outer, fas_dual, `lam` (median threshold), and per scope "all" (every
    coefficient) / "anatomy" (latent positions inside the organ mask; needs `z`
    from `trace_slice` on the same slice -- the masks are `mg/organ/l<l>`):
        sat     fraction with |clip output| >= 0.99 lambda  (on the boundary)
        reset   fraction with |inc| > |z| + lambda: saturated WHATEVER z was --
                the code's history erased this sweep. inc = what the clip adds
                to z: A xbar, minus pi_z where pi_z goes inside the clip
                ("gradient", "legacy"); A xbar under "fixed_point"
        a_med, a_p90   median / 90th percentile of |inc| / lambda
        z_med   median |z| / lambda of the incoming code
        pi_over, pi_med  (coarse sweeps) fraction with |pi_z| > lambda, median
        hist    counts of log10(|inc| / lambda) over SAT_LOG_BINS
        flip    of the coefficients saturated now AND in the previous sweep's
                clip output at the same (V-cycle, level), the fraction whose
                phase turned > 90 deg (a code move of ~2 lambda); NaN on a
                level's first traced sweep
        push    |tau D (clip output - previous clip output)| / |x|: how hard the
                code's change pushes on x (this layer's D and tau)
        step    |x+ - x| / |x|, the whole primal step
    Under fas_dual="fixed_point" the clip output is z+ - pi_z (pi_z is added
    after the clip). Local prox only; group-prox sweeps are skipped."""
    import types

    import torch
    import torch.nn.functional as F

    device = torch.device(device) if device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    model = cached_model(run, device)
    tops = sorted({o for kind, _, _, o, _ in _vcycle_walk(model) if o is not None})
    want = None if not tops else (
        {tops[-1]} if outers is None else {tops[o] for o in outers})
    levels = set(levels)

    def select(level, outer):
        return level in levels and (outer is None or want is None or outer in want)

    # anatomy masks on each level's LATENT grid: the organ mask, averaged over
    # the analysis stride (the latent sample i sits on image pixel s*i)
    masks = {}
    if z is not None:
        stride = next((L.stride for kind, _, _, _, L in _vcycle_walk(model) if kind == "layer"), 2)
        for l in levels:
            o = z.get(f"mg/organ/l{l}")
            if o is not None:
                t = torch.from_numpy(np.ascontiguousarray(o, np.float32))[None, None].to(device)
                masks[l] = F.avg_pool2d(t, stride) > 0.5
    args = types.SimpleNamespace(sigma=sigma, seed=seed, val_kspace_type=val_kspace_type,
                                 coarse_op=coarse_op, transfer_filter=transfer_filter)
    with open(os.path.join(run, "config.json")) as f:
        vol = _resolve_volume(json.load(f), volume)
    R = _run_one(run, vol, int(slice), args, device,
                 lambda st: _saturation_call(st, select, masks))
    recs = R["store"]["recs"]
    meta = dict(run=R["run"], volume=vol, slice=int(slice), levels=sorted(levels),
                outers=sorted(want) if want else None, n_sweeps=len(recs),
                scopes=sorted({s for r in recs for s in ("all", "anatomy") if s in r}))
    if not recs:
        print(f"WARNING: no sweep of {R['run']} matched levels={sorted(levels)}, "
              f"outers={outers}", file=sys.stderr)
    return recs, meta


def saturation_table(recs, meta, scope="anatomy", per_sweep=True):
    """Print every sweep (optional) and a per-(V-cycle, level) mean."""
    scope = scope if scope in meta["scopes"] else "all"
    print(f"{meta['run']}  {meta['volume']} slice {meta['slice']}  ({scope} coefficients; "
          f"sat = on the clip boundary, reset = saturated whatever z was;\n"
          f"  flip = saturated codes whose phase turned > 90 deg since the previous sweep "
          f"at this level; push = |tau D (code change)| / |x|; step = |x+ - x| / |x|)")
    hdr = (f"  {'sweep':<22}{'lambda':>9}{'sat':>7}{'reset':>7}{'|inc|/lam med':>14}"
           f"{'p90':>7}{'|z|/lam':>8}{'|pi|>lam':>9}{'flip':>7}{'push':>8}{'step':>8}")
    if per_sweep:
        print(hdr)
        bad = [r["where"] for r in recs
               if r[scope].get("over", 0) > 0 or r[scope].get("reset_off", 0) > 0]
        if bad:
            print(f"  WARNING: self-check failed in {bad} (clip input/output mis-identified)")
        for r in recs:
            d = r[scope]
            pi = f"{100 * d['pi_over']:8.1f}%" if "pi_over" in d else f"{'-':>9}"
            fl = f"{100 * d['flip']:6.1f}%" if np.isfinite(d.get("flip", np.nan)) else f"{'-':>7}"
            pu = f"{100 * d['push']:7.2f}%" if "push" in d else f"{'-':>8}"
            print(f"  {r['where']:<22}{r['lam']:>9.2e}{100 * d['sat']:6.1f}%{100 * d['reset']:6.1f}%"
                  f"{d['a_med']:>14.2f}{d['a_p90']:>7.1f}{d['z_med']:>8.2f}{pi}{fl}{pu}"
                  f"{100 * d['step']:7.2f}%")
    print(f"\n  mean over sweeps")
    print(f"  {'V-cycle, level':<22}{'lambda':>9}{'sat':>7}{'reset':>7}{'|inc|/lam med':>14}"
          f"{'p90':>7}{'|z|/lam':>8}{'|pi|>lam':>9}{'flip':>7}{'push':>8}{'step':>8}")
    groups = {}
    for r in recs:
        groups.setdefault((r["outer"], r["level"]), []).append(r)
    for (o, l), rs in sorted(groups.items(), key=lambda kv: (kv[0][0] or 0, kv[0][1])):
        ds = [r[scope] for r in rs]
        mean = lambda k: float(np.mean([d[k] for d in ds]))                   # noqa: E731
        pis = [d["pi_over"] for d in ds if "pi_over" in d]
        pi = f"{100 * np.mean(pis):8.1f}%" if pis else f"{'-':>9}"
        fls = [d["flip"] for d in ds if np.isfinite(d.get("flip", np.nan))]
        fl = f"{100 * np.mean(fls):6.1f}%" if fls else f"{'-':>7}"
        pus = [d["push"] for d in ds if "push" in d]
        pu = f"{100 * np.mean(pus):7.2f}%" if pus else f"{'-':>8}"
        label = f"V{(o or 0) + 1} L{l} ({len(rs)} sweeps)"
        print(f"  {label:<22}{np.median([r['lam'] for r in rs]):>9.2e}{100 * mean('sat'):6.1f}%"
              f"{100 * mean('reset'):6.1f}%{mean('a_med'):>14.2f}{mean('a_p90'):>7.1f}"
              f"{mean('z_med'):>8.2f}{pi}{fl}{pu}{100 * mean('step'):7.2f}%")


SAT_RUN_COLORS = ("#0072B2", "#D55E00", "#009E73", "#E69F00")


def fig_saturation(results, scope="anatomy", panel=3.2, dpi=None):
    """`results` = {label: (recs, meta)} from `trace_saturation` on the SAME
    levels / outers (e.g. vtxA and vtxB). Top: saturated and reset fractions
    per sweep, execution order (level spans shaded). Bottom: distribution of
    log10(|inc| / lambda) per level (inc = what the clip adds to z), pooled
    over the traced sweeps -- right of the dashed line (1) the clip is active
    for a zero code; right of the dotted one (2) for ANY code in the box."""
    import matplotlib.pyplot as plt
    labels = list(results)
    first = results[labels[0]][0]
    levels = sorted({r["level"] for r in first})
    fig = plt.figure(figsize=(panel * max(3, len(levels)) + 0.8, 2 * panel + 1.0), dpi=dpi)
    ncol = max(2, len(levels))
    gs = fig.add_gridspec(2, ncol, hspace=0.45, wspace=0.28)
    top = gs[0, :].subgridspec(1, 2, wspace=0.2)
    for j, (key, title) in enumerate((("sat", "on the clip boundary (sat)"),
                                      ("reset", "saturated whatever z was (reset)"))):
        A = fig.add_subplot(top[0, j])
        for c, lab in enumerate(labels):
            recs, meta = results[lab]
            sc = scope if scope in meta["scopes"] else "all"
            A.plot(range(len(recs)), [100 * r[sc][key] for r in recs], marker="o", ms=3,
                   lw=1.4, color=SAT_RUN_COLORS[c % 4], label=lab)
        start = 0
        for i in range(1, len(first) + 1):     # shade runs of the same level
            if i == len(first) or (first[i]["level"], first[i]["outer"]) != \
                    (first[start]["level"], first[start]["outer"]):
                lv = first[start]["level"]
                A.axvspan(start - .5, i - .5, color="0.5", alpha=0.06 + 0.06 * lv, lw=0)
                A.text((start + i - 1) / 2, 1.0, f"L{lv}", transform=A.get_xaxis_transform(),
                       ha="center", va="bottom", fontsize=7, color="0.3")
                start = i
        A.set_ylim(0, 100); A.set_ylabel("% of coefficients", fontsize=8)
        A.set_xlabel("LPDS sweep (execution order)", fontsize=8)
        A.set_title(title, fontsize=9, pad=12); A.tick_params(labelsize=7); A.grid(alpha=.25)
        A.legend(fontsize=7, frameon=False)
    centers = (SAT_LOG_BINS[:-1] + SAT_LOG_BINS[1:]) / 2
    for j, lv in enumerate(levels):
        A = fig.add_subplot(gs[1, j])
        for c, lab in enumerate(labels):
            recs, meta = results[lab]
            sc = scope if scope in meta["scopes"] else "all"
            h = sum(r[sc]["hist"] for r in recs if r["level"] == lv)
            if np.sum(h):
                A.plot(centers, 100 * h / h.sum(), color=SAT_RUN_COLORS[c % 4], lw=1.4, label=lab)
        A.axvline(0, color="0.4", ls="--", lw=0.9)
        A.axvline(np.log10(2), color="0.4", ls=":", lw=0.9)
        A.set_xlabel("log10(|increment| / lambda)", fontsize=8)
        A.set_title(f"L{lv}", fontsize=9); A.tick_params(labelsize=7); A.grid(alpha=.25)
        if j == 0:
            A.set_ylabel("% of coefficients per bin", fontsize=8)
            A.legend(fontsize=7, frameon=False)
    m0 = results[labels[0]][1]
    fig.suptitle(f"dual-code saturation, {m0['volume']} slice {m0['slice']}  ({scope} "
                 f"coefficients; increment = what the clip adds to z; dashed: = lambda, dotted: 2 lambda)", fontsize=10)
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
