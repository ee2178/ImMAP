#!/usr/bin/env python3
"""
Which coarse correction carries what, in a TRAINED multigrid net?

    python scripts/coarse_diag.py --run trained_nets/mg_recon/knee/mg81v6vtxAnestUcf32_R8 \
        --flags results/alias_slices_cf32_all/knee/slices_R8.csv --out results/coarse_diag/knee_R8

The worry it tests: the net uses its coarse levels for detail rather than to solve
the coarse problem. A V-cycle hands back two corrections, and they reach the image
differently:

    x <- x + alpha_x P(dx)          dx = wx_c - x_c       P low-passes it
    z <- z + alpha_z P(dz)          dz = wz_c - z_c       the next primal step then moves
    dx_z = -tau B_f alpha_z P(dz)                         x by this, where B_f can put
                                                          fine atoms under a smooth envelope

SPECTRA, on the unmodified forward. Per V-cycle and level, the applied alpha_x P(dx)
("x") and dx_z above ("z") on that level's image grid; "state" is the iterate they
correct. Energy in 8 bins of max(|w_h|, |w_w|), bin width pi/8, with `hf` the share
at or above pi/2: what the coarse grid below could not have represented, so anything
there came through P's transition band or, for "z", through B_f. `rel_norm` is
||correction|| / ||state||. "z" is the first post-smoothing step only: a z correction
keeps acting through every later sweep, so it is a lower bound on that path.

KNOCKOUTS, one forward each, same measurement and noise. The corrections' outputs are
zeroed by forward hooks, so the net itself is not touched:

    full     as trained
    az0      alpha_z off everywhere         az0_L0, az0_L1   only L1 -> L0, only L2 -> L1
    ax0      alpha_x off everywhere         ax0_L0, ax0_L1   likewise
    ax0az0   both off everywhere            (the coarse levels contribute nothing)

each with PSNR / SSIM / NRMSE and alias_check.py's rho against the zero-filled image.
A knockout on a trained net overstates that path's value, since the later sweeps were
trained to expect it: read it against the other knockouts, and against a net TRAINED
without the path (exp19's az0 cell, torch/exp19_coarse_op.sbatch).

TRAJECTORY: does a correction ADD aliasing? Knockouts cannot say, since the later sweeps
expect the path they remove. On the unmodified forward, the fine iterate of each V-cycle at

    pre    after pre-smoothing      +x   pre + alpha_x P(dx)
    +z     +x + dx_z                post after post-smoothing

and how much ghost each holds, by projection. HP is alias_check.py's high-pass, then a 4-tap
average on both axes: exact nulls at pi and pi/2, because the mid-net iterates carry lattice
tones there (see SPECTRA) that are nearly orthogonal to a ghost but far larger, and leaked
into this number before they were filtered. With H = HP(|ref|) and T = HP(|zero-filled|), the
template G is T with its component along H removed, so a true edge does not count, and

    beta = <HP(|x| - |ref|), G> / <G, G>

is 1 for the zero-filled image and 0 for the reference, signed, in magnitudes masked as the
dumps are. It can exceed 1: the zero-filled image is scaled down by the mask, so raising the
image's amplitude raises its ghost with it. `beta` is over the eroded organ, `beta_win` over
the window where the full output's rho peaks, which asks when the aliasing left in the output
appeared; rows with no such window leave it empty. Treat small differences as nothing.

SLICES. --sample-volumes volumes at random (the same draw for every run), each with
--slices-per-volume spaced over its training range, plus the top --top-flagged flagged
slices of --flags (an alias_check.py slices_R<R>.csv), so a net's numbers exist both
in general and where the aliasing is. The `source` column says which. --coarse-op swaps
the coarse operator at inference (no parameter depends on it): a diagnostic, since the
net learned around the operator it trained with.

Outputs under --out: <anatomy>/<run>[_op<op>].{knockouts,spectra,trajectory}.csv, and with
--combine (the sbatch runs it after each task) knockouts.csv, spectra.csv, trajectory.csv,
summary.txt.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import glob
import os
import sys
import zlib

import numpy as np
import torch
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import datasets                                  # noqa: F401,E402  registers the loaders
from alias_check import hp_pe, local_corr, slice_sigma, strength_map, top_patches  # noqa: E402
from datasets.registry import build_loader       # noqa: E402
from dump_eval import load_run, parse_slices, resolve_volumes  # noqa: E402
from eval_per_volume import sample_volumes, slice_seed, train_slices  # noqa: E402
from evaluation.metrics import build_metrics     # noqa: E402
from evaluation.tasks import (EVAL_KSPACE_TYPE, build_adapter, default_sigma,  # noqa: E402
                              force_organ_mask)
from models.components import to_complex         # noqa: E402
from operators.truncate import Truncate          # noqa: E402

N_BINS = 8                    # of max(|w_h|, |w_w|) over [0, pi]; pi/2 starts bin 4
HF_BIN = N_BINS // 2
METRICS = ("psnr", "ssim", "nrmse")


def _knockout(off_var=(), off_level=None):
    return lambda var, level: var in off_var and (off_level is None or level == off_level)


KNOCKOUTS = {
    "full": _knockout(),
    "az0": _knockout("z"),
    "ax0": _knockout("x"),
    "ax0az0": _knockout("xz"),
    "az0_L0": _knockout("z", 0),
    "az0_L1": _knockout("z", 1),
    "ax0_L0": _knockout("x", 0),
    "ax0_L1": _knockout("x", 1),
}


def band_energy(t):
    """Energy of a `(1, 1, H, W)` image per bin of max(|w_h|, |w_w|) -> numpy, N_BINS."""
    c = to_complex(t) if not t.is_complex() else t
    power = torch.fft.fft2(c[0, 0]).abs().pow(2)
    H, W = power.shape
    fh = torch.fft.fftfreq(H, device=c.device).abs()[:, None]
    fw = torch.fft.fftfreq(W, device=c.device).abs()[None, :]
    w = 2 * torch.maximum(fh.expand(H, W), fw.expand(H, W))      # in units of pi
    idx = (w * N_BINS).long().clamp_max(N_BINS - 1).flatten()
    e = torch.zeros(N_BINS, dtype=power.dtype, device=c.device)
    e.scatter_add_(0, idx, power.flatten())
    return e.cpu().numpy()


def spectrum_row(t, state):
    """`{rel_norm, hf, e0..e7}`: bins as shares of the energy."""
    e = band_energy(t)
    total = float(e.sum())
    share = e / max(total, 1e-30)
    return dict(rel_norm=float(t.norm() / state.norm().clamp_min(1e-30)),
                hf=float(share[HF_BIN:].sum()),
                energy=total, **{f"e{i}": float(s) for i, s in enumerate(share)})


@contextlib.contextmanager
def instrumented(net, knock=None, spectra=None):
    """Hooks on every V-cycle's corrections. `knock(var, level)` True zeroes one; `spectra`,
    a list, collects the unknocked corrections' rows. Removed on exit, also on an exception."""
    handles = []

    def add(mod, fn):
        handles.append(mod.register_forward_hook(fn))

    for k, level, V in net.vcycles():
        pending = {}

        def on_pre(m, a, out, k=k, level=level, pending=pending):
            x = out[0][0]
            pending["state"] = x
            if spectra is not None:
                spectra.append(dict(outer=k, level=level, path="state",
                                    **spectrum_row(x, x)))

        def on_x(m, a, out, k=k, level=level, pending=pending):
            if spectra is not None:
                spectra.append(dict(outer=k, level=level, path="x",
                                    **spectrum_row(out, pending["state"])))
            return torch.zeros_like(out) if knock and knock("x", level) else None

        def on_z(m, a, out, level=level, pending=pending):
            pending["dz"] = out
            return torch.zeros_like(out) if knock and knock("z", level) else None

        # the post-smoother's first tau is read just before that step uses the
        # corrected z, so its value here is the one dx_z is scaled by
        def on_tau(m, a, tau, k=k, level=level, V=V, pending=pending):
            dz = pending.pop("dz", None)
            if spectra is None or dz is None:
                return
            step = torch.as_tensor(tau).real.reshape(-1)[0]
            dx_z = -step * V.lpdsB.first.synthesis(dz)
            spectra.append(dict(outer=k, level=level, path="z",
                                **spectrum_row(dx_z, pending["state"])))

        add(V.lpdsA, on_pre)
        add(V.alpha_x, on_x)
        add(V.alpha_z, on_z)
        add(V.lpdsB.first.tau, on_tau)
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def run_knockouts(net, y, E, sigma, names=tuple(KNOCKOUTS)):
    """One forward per knockout on one measurement -> `({name: image}, spectra rows)`."""
    outs, spectra = {}, None
    with torch.no_grad():
        for name in names:
            rows = [] if name == "full" else None
            with instrumented(net, KNOCKOUTS[name], rows):
                out = net(y, E=E, sigma=sigma)
            outs[name] = (out[0] if isinstance(out, tuple) else out).detach()
            if rows is not None:
                spectra = rows
    return outs, spectra


def run_trajectory(net, y, E, sigma):
    """One forward -> `[(outer, stage, x)]`: the fine iterate around each correction."""
    rows, handles = [], []

    def add(mod, fn):
        handles.append(mod.register_forward_hook(fn))

    for k, level, V in net.vcycles():
        if level:
            continue
        held = {}

        def on_pre(m, a, out, k=k, held=held):
            held["x"] = out[0][0]
            rows.append((k, "pre", held["x"]))

        def on_x(m, a, out, k=k, held=held):
            held["x"] = held["x"] + out
            rows.append((k, "+x", held["x"]))

        def on_z(m, a, out, held=held):
            held["dz"] = out

        def on_tau(m, a, tau, k=k, V=V, held=held):
            dz = held.pop("dz", None)
            if dz is not None:
                step = torch.as_tensor(tau).real.reshape(-1)[0]
                rows.append((k, "+z", held["x"] - step * V.lpdsB.first.synthesis(dz)))

        def on_post(m, a, out, k=k):
            rows.append((k, "post", out[0][0]))

        add(V.lpdsA, on_pre)
        add(V.alpha_x, on_x)
        add(V.alpha_z, on_z)
        add(V.lpdsB.first.tau, on_tau)
        add(V, on_post)
    try:
        with torch.no_grad():
            out = net(y, E=E, sigma=sigma)
    finally:
        for h in handles:
            h.remove()
    return (out[0] if isinstance(out, tuple) else out).detach(), rows


def hp_no_tones(x):
    """`hp_pe`, then a 4-tap average per axis: exact nulls at pi and pi/2."""
    return ndi.uniform_filter1d(ndi.uniform_filter1d(hp_pe(x), 4, axis=0, mode="nearest"),
                                4, axis=1, mode="nearest")


def ghost_template(ref, zf):
    """`G`: the zero-filled image's high-pass with its component along the reference's removed."""
    h, t = hp_no_tones(ref), hp_no_tones(zf)
    return t - h * float((t * h).sum() / max((h * h).sum(), 1e-30))


def ghost_fraction(rec, ref, G, where):
    """`beta` of the docstring: 1 for the zero-filled image, 0 for the reference."""
    g = G[where]
    return float((hp_no_tones(rec - ref)[where] * g).sum() / max((g * g).sum(), 1e-30))


def score_trajectory(gt, zf, organ, out, rows, T, a):
    """`[{outer, stage, beta, beta_win}]` for `run_trajectory`'s rows; `out` on the embedded grid."""
    keep = organ[0, 0].abs().cpu().numpy() > 0
    mag = lambda t: np.abs(t[0, 0].cpu().numpy()) * keep       # noqa: E731
    ref, zf_m = mag(gt), mag(zf)
    inner = ndi.binary_erosion(keep, iterations=4)
    if inner.sum() < 100:
        return []
    final = mag(T.forward(out))
    # x is DC-removed (preprocessing/kspace.py); the output minus the last iterate is mu
    last = rows[-1][2]
    mu = (out - (to_complex(last) if not last.is_complex() else last)).mean()
    sigma = slice_sigma(ref, [final], inner)
    win = (a.win_rows, a.win_cols)
    e = hp_pe(final - ref)
    rho = np.where(strength_map(e, sigma, win) > a.min_strength,
                   local_corr(e, hp_pe(zf_m), win), -np.inf)
    top = top_patches(rho, np.abs(e) / (sigma + 1e-12), inner, a, n=1)
    box = None
    if top:
        box = np.zeros_like(keep)
        i, j = top[0][2], top[0][3]
        box[max(i - win[0] // 2, 0):i + win[0] // 2 + 1, max(j - win[1] // 2, 0):j + win[1] // 2 + 1] = True
        box &= inner
    G = ghost_template(ref, zf_m)
    out_rows = []
    for outer, stage, x in rows:
        im = mag(T.forward((to_complex(x) if not x.is_complex() else x) + mu))
        out_rows.append(dict(outer=outer, stage=stage, beta=ghost_fraction(im, ref, G, inner),
                             beta_win=ghost_fraction(im, ref, G, box) if box is not None else ""))
    return out_rows


def top_rho(rec, ref, zf, inner, sigma, a):
    """alias_check.py's slice score for one image: `(rho, strength)`, -1 with no strong place."""
    win = (a.win_rows, a.win_cols)
    e = hp_pe(rec - ref)
    st = strength_map(e, sigma, win)
    rho = np.where(st > a.min_strength, local_corr(e, hp_pe(zf), win), -np.inf)
    z = np.abs(e) / (sigma + 1e-12)
    top = top_patches(rho, z, inner, a, n=1)
    return (top[0][0], float(st[top[0][2], top[0][3]])) if top else (-1.0, 0.0)


def score_slice(gt, outs, zf, organ, metrics, a):
    """`[{knockout, psnr, ssim, nrmse, rho, strength}]`, on the masked magnitudes the dumps
    hold, so rho reads like alias_check.py's. The residual scale is the full net's, so a
    knockout's extra error shows as strength rather than being normalised away."""
    keep = organ[0, 0].abs().cpu().numpy() > 0
    mag = lambda t: np.abs(t[0, 0].cpu().numpy()) * keep       # noqa: E731
    ref, zf_m = mag(gt), mag(zf)
    inner = ndi.binary_erosion(keep, iterations=4)
    if inner.sum() < 100:
        return []
    sigma = slice_sigma(ref, [mag(outs["full"])], inner)
    rows = []
    for name, rec in outs.items():
        rho, strength = top_rho(mag(rec), ref, zf_m, inner, sigma, a)
        rows.append(dict(knockout=name, rho=rho, strength=strength,
                         **{m: float(fn(gt, rec, mask=organ).reshape(-1)[0])
                            for m, fn in metrics.items()}))
    return rows


def pick_slices(cfg, a):
    """`[(volume, slice, source)]`: the draw depends only on the arguments and val set."""
    vols = sample_volumes(resolve_volumes(cfg["data"]["val"], None, None),
                          a.sample_volumes, a.sample_seed)
    lo, hi = parse_slices(train_slices(cfg))
    if hi is None:
        raise SystemExit("this run's training range has no end; cannot space slices over it")
    picks = [(v, int(s), "random") for v in vols
             for s in np.unique(np.linspace(lo, hi - 1, a.slices_per_volume).round())]
    if a.flags and a.top_flagged:
        with open(a.flags, newline="") as f:
            flagged = [r for r in csv.DictReader(f) if int(r["flagged"])]
        flagged.sort(key=lambda r: int(r["rank"]))
        have = {(v, s): i for i, (v, s, _) in enumerate(picks)}
        for r in flagged[:a.top_flagged]:
            key = (r["volume"], int(r["slice"]))
            if key in have:
                picks[have[key]] = (*key, "flagged")
            else:
                picks.append((*key, "flagged"))
    return picks


def load_batch(cfg, volume, sl):
    val = dict(cfg["data"]["val"])
    val.update(enumerate_slices=True, start_slice=sl, end_slice=sl + 1, volumes=[volume],
               batch_size=1, num_workers=0)
    return next(iter(build_loader(val, shuffle=False, drop_last=False)), None)


def run_diag(run_dir, a, device):
    cfg, model, _, step = load_run(run_dir, device)
    run = os.path.basename(os.path.normpath(run_dir))
    if not (hasattr(model, "vcycles") and model.vcycles()):
        raise SystemExit(f"{run} has no primal-dual V-cycles to diagnose")
    if a.coarse_op:
        mods = [m for m in model.modules() if hasattr(m, "coarse_op")]
        print(f"[{run}] coarse_op {mods[0].coarse_op} -> {a.coarse_op} ({len(mods)} modules)")
        for m in mods:
            m.coarse_op = a.coarse_op
        run = f"{run}_op{a.coarse_op}"
    force_organ_mask(cfg)
    cfg.setdefault("training", {})["val_kspace_type"] = a.val_kspace_type
    sigma = a.sigma if a.sigma is not None else default_sigma(cfg)
    anatomy = cfg["data"]["val"]["anatomy"]
    picks = pick_slices(cfg, a)
    print(f"[{run}] step {step}, sigma {sigma:g}, {cfg['training']['val_kspace_type']}, "
          f"{len(picks)} slices", flush=True)

    metrics = build_metrics(METRICS)
    adapter = build_adapter(cfg["task"])
    gen = torch.Generator(device=device)
    kn_rows, sp_rows, tr_rows = [], [], []
    for i, (vol, sl, source) in enumerate(picks):
        batch = load_batch(cfg, vol, sl)
        if batch is None:
            print(f"  {vol} has no slice {sl}; skipped", file=sys.stderr)
            continue
        gen.manual_seed(slice_seed(a.seed, vol, sl))
        got = {}

        def call(net, y, E, sig):
            got["outs"], got["spectra"] = run_knockouts(net, y, E, sig)
            got["zf"] = E.adjoint(y)
            got["traj_out"], got["traj"] = run_trajectory(net, y, E, sig)
            return got["outs"]["full"]

        gt, recon, organ = adapter(model, batch, cfg, device, sigma, gen, call=call)
        T = Truncate(tuple(got["zf"].shape[-2:]), tuple(gt.shape[-2:]))
        outs = {k: T.forward(v) for k, v in got["outs"].items()}
        if not torch.equal(outs["full"], recon):
            raise SystemExit(f"{vol}:{sl}: the embedding here is not the adapter's")
        base = dict(run=run, step=int(step or 0), volume=vol, slice=sl, source=source)
        kn_rows += [dict(base, **r) for r in score_slice(gt, outs, T.forward(got["zf"]),
                                                         organ, metrics, a)]
        sp_rows += [dict(base, **r) for r in got["spectra"]]
        tr_rows += [dict(base, **r) for r in score_trajectory(gt, T.forward(got["zf"]), organ,
                                                              got["traj_out"], got["traj"], T, a)]
        if (i + 1) % 10 == 0 or i + 1 == len(picks):
            print(f"  {i + 1}/{len(picks)}", flush=True)

    dest = os.path.join(a.out, anatomy)
    os.makedirs(dest, exist_ok=True)
    for kind, rows in (("knockouts", kn_rows), ("spectra", sp_rows), ("trajectory", tr_rows)):
        write_csv(os.path.join(dest, f"{run}.{kind}.csv"), rows)
    return kn_rows, sp_rows


def write_csv(path, rows):
    tmp = f"{path}.partial{zlib.crc32(path.encode())}"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["run"])
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def _floats(rows, key):
    return np.array([float(r[key]) for r in rows])


def trajectory_lines(rows, key, raise_by=0.02):
    """Mean `key` per V-cycle and stage, and how often each correction raises it by `raise_by`."""
    rows = [r for r in rows if r[key] != ""]
    if not rows:
        return [f"  no slice has {key}"]
    at = {}
    for r in rows:
        at.setdefault((int(r["outer"]), r["volume"], int(r["slice"])), {})[r["stage"]] = float(r[key])
    lines = [f"  {'V':>3}{'pre':>9}{'+x':>9}{'+z':>9}{'post':>9}{'x raises':>10}{'z raises':>10}{'n':>5}"]
    for outer in sorted({k[0] for k in at}):
        s = [v for k, v in at.items() if k[0] == outer]
        mean = lambda st: np.mean([v[st] for v in s if st in v]) if any(st in v for v in s) else np.nan  # noqa: E731
        up_x = np.mean([v["+x"] - v["pre"] > raise_by for v in s])
        with_z = [v for v in s if "+z" in v]
        up_z = np.mean([v["+z"] - v["+x"] > raise_by for v in with_z]) if with_z else np.nan
        lines.append(f"  {outer + 1:>3}{mean('pre'):>9.3f}{mean('+x'):>9.3f}{mean('+z'):>9.3f}{mean('post'):>9.3f}"
                     f"{100 * up_x:>9.0f}%{100 * up_z:>9.0f}%{len(s):>5}")
    return lines


def summarize(kn_rows, sp_rows, tr_rows=()):
    """The tables a reader wants, one block per run. Means over slices."""
    lines = []
    for run in sorted({r["run"] for r in kn_rows}):
        kr = [r for r in kn_rows if r["run"] == run]
        lines += [f"== {run}", ""]
        lines.append(f"{'source':<9}{'knockout':<10}{'n':>4}{'PSNR':>8}{'dPSNR':>8}{'dSSIM':>9}"
                     f"{'rho':>7}{'drho':>7}{'strength':>10}")
        for source in sorted({r["source"] for r in kr}, reverse=True):
            sr = [r for r in kr if r["source"] == source]
            ref = {(r["volume"], r["slice"]): r for r in sr if r["knockout"] == "full"}
            for name in KNOCKOUTS:
                x = [r for r in sr if r["knockout"] == name]
                if not x:
                    continue
                base = [ref[(r["volume"], r["slice"])] for r in x]
                # rho -1 means no place strong enough, which a mean would read as "low"
                strong = [(float(r["rho"]), float(b["rho"])) for r, b in zip(x, base)
                          if float(r["rho"]) > -1 and float(b["rho"]) > -1]
                rho = np.mean([s[0] for s in strong]) if strong else np.nan
                drho = np.mean([s[0] - s[1] for s in strong]) if strong else np.nan
                lines.append(
                    f"{source:<9}{name:<10}{len(x):>4}{_floats(x, 'psnr').mean():>8.2f}"
                    f"{(_floats(x, 'psnr') - _floats(base, 'psnr')).mean():>+8.2f}"
                    f"{(_floats(x, 'ssim') - _floats(base, 'ssim')).mean():>+9.4f}"
                    f"{rho:>7.3f}{drho:>+7.3f}{_floats(x, 'strength').mean():>10.1f}")
        lines.append("")
        sp = [r for r in sp_rows if r["run"] == run]
        lines.append("x = alpha_x P(dx), z = first-step dx_z; hf = share of each at or above pi/2; "
                     "z share = z's part of the two corrections' energy there")
        lines.append(f"{'V':>3}{'level':>7}{'state hf':>10}{'x rel':>9}{'x hf':>8}"
                     f"{'z rel':>9}{'z hf':>8}{'z share hf':>12}")
        for outer, level in sorted({(int(r["outer"]), int(r["level"])) for r in sp}):
            at = {p: [r for r in sp if int(r["outer"]) == outer and int(r["level"]) == level
                      and r["path"] == p] for p in ("state", "x", "z")}
            sx, sz = at["x"], at["z"]
            hfx = (_floats(sx, "hf") * _floats(sx, "energy")).mean()
            hfz = (_floats(sz, "hf") * _floats(sz, "energy")).mean() if sz else 0.0
            zr = f"{_floats(sz, 'rel_norm').mean():>9.4f}{_floats(sz, 'hf').mean():>8.3f}" if sz \
                else f"{'-':>9}{'-':>8}"
            lines.append(f"{outer + 1:>3}{f'L{level + 1}->L{level}':>7}"
                         f"{_floats(at['state'], 'hf').mean():>10.3f}"
                         f"{_floats(sx, 'rel_norm').mean():>9.4f}{_floats(sx, 'hf').mean():>8.3f}"
                         f"{zr}{hfz / max(hfx + hfz, 1e-30):>12.3f}")
        lines.append("")
        tr = [r for r in tr_rows if r["run"] == run]
        if tr:
            lines.append("ghost in the fine iterate by stage: 1 = the zero-filled image's, 0 = the reference's; "
                         "'raises' = share of slices up by more than 0.02")
            for source in sorted({r["source"] for r in tr}, reverse=True):
                sr = [r for r in tr if r["source"] == source]
                lines.append(f"{source}, beta_win (where the output's rho peaks)")
                lines += trajectory_lines(sr, "beta_win")
                lines.append(f"{source}, beta (whole organ)")
                lines += trajectory_lines(sr, "beta")
            lines.append("")
    return lines


def combine(out):
    """Merge every run under `out`; written through a temporary file, so a reader never
    sees half of one."""
    merged = {}
    for kind in ("knockouts", "spectra", "trajectory"):
        rows = []
        for p in sorted(glob.glob(os.path.join(out, "*", f"*.{kind}.csv"))):
            with open(p, newline="") as f:
                rows += [dict(r, anatomy=os.path.basename(os.path.dirname(p)))
                         for r in csv.DictReader(f)]
        write_csv(os.path.join(out, f"{kind}.csv"), rows)
        merged[kind] = rows
    tmp = os.path.join(out, "summary.txt.partial")
    with open(tmp, "w") as f:
        f.write("\n".join(summarize(merged["knockouts"], merged["spectra"], merged["trajectory"])) + "\n")
    os.replace(tmp, os.path.join(out, "summary.txt"))
    print(f"[coarse_diag] merged {len({r['run'] for r in merged['knockouts']})} runs into {out}")


def parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", nargs="*", default=[], help="run dirs (config.json + net.ckpt)")
    p.add_argument("--out", default="results/coarse_diag")
    p.add_argument("--combine", action="store_true", help="merge --out, after any --run")
    p.add_argument("--sample-volumes", type=int, default=20)
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument("--slices-per-volume", type=int, default=3)
    p.add_argument("--flags", help="an alias_check.py slices_R<R>.csv")
    p.add_argument("--top-flagged", type=int, default=30)
    p.add_argument("--seed", type=int, default=1234, help="eval_per_volume's seed")
    p.add_argument("--sigma", type=float, help="default: the run's val_noise_std")
    p.add_argument("--val-kspace-type", default=EVAL_KSPACE_TYPE)
    p.add_argument("--coarse-op", choices=("galerkin", "rediscretize", "folded"),
                   help="swap the coarse operator at inference; default: as trained")
    p.add_argument("--win-rows", type=int, default=25, help="as alias_check.py")
    p.add_argument("--win-cols", type=int, default=9, help="as alias_check.py")
    p.add_argument("--min-strength", type=float, default=2.5, help="as alias_check.py")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    if not a.run and not a.combine:
        parser().error("nothing to do: --run and/or --combine")
    device = torch.device(a.device)
    for run_dir in a.run:
        run_diag(run_dir, a, device)
    if a.combine:
        combine(a.out)


if __name__ == "__main__":
    main()
