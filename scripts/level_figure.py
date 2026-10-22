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
