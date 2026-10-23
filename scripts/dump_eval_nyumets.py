#!/usr/bin/env python3
"""
Dump per-volume CT1 syntheses of NYUMets runs to HDF5, for the figure viewer, and score them.

    python scripts/dump_eval_nyumets.py                       # the default runs, 60 volumes, 80:100
    python scripts/dump_eval_nyumets.py --n-volumes 10 --slices 80:100 \
        --runs trained_nets/nyumets/I2SB_Unet_NYUMets_CT1_from_all_bs64

The NYUMets counterpart of `scripts/dump_eval.py` (fastMRI): the same HDF5 contract, so
`figures/viewer.py`, `figures/webviewer.py` and `figures/make_figure.py` read these dumps without
knowing the task. One file per (run, volume):

    <run_dir>/eval_dump/<volume>.h5     (or mirrored under --out/<run>/)

    reference      (n, H, W) float32   this session's CT1 -- the target       } x organ_mask
    recon          (n, H, W) float32   the run's synthesis                    } (images_masked)
    zero_filled    (n, H, W) float32   this session's T1 -- the INPUT, the role the zero-filled
                                       adjoint plays for recon (`@zero_filled` in a viewer config)
    prior_ct1      (n, H, W) float32   the prior study's CT1 at the matched slice (`@prior_ct1`)
    organ_mask     (n, H, W) uint8     the loader's mask (mask_source, e.g. T1 & CT1)

Every image is stored turned into the radiological view (`--orient`, attr `display_orient`):
the h5s are canonical RAS, which draws the eyes on the right, and the viewer's FLIP cannot
transpose. Display only -- the metrics do not depend on it.
    slice_index    (n,)      int32     ORIGINAL slice number in the volume
    <metric>_slice (n,)      float32   psnr / ssim / nrmse inside organ_mask

and a CSV of every per-slice metric (--csv), with per-run means printed at the end.

What is held fixed, and why
---------------------------
* THE SAME SAMPLES for every run. One loader configuration serves every run: the guided loader
  (`guide_mode: other_study`) with the union of the runs' conditioning contrasts and guide
  planes, as notebooks/immap_sb_longitudinal.ipynb builds it. So every run is scored on the
  multi-study patients only -- including a no-prior run, which trained on everyone -- and on the
  identical prior study.
* ONE PRIOR STUDY PER VOLUME. The loader picks a patient's other study per SLICE (from the sample
  index), so a patient with three studies would get a different prior on different slices of one
  stack. Here the pick is pinned to the volume's first dumped slice; the chosen study is the
  `prior_volume` attribute.
* ONE SET OF UNITS. Runs trained on the fixed per-contrast `scales` and on `volume_scale` are run
  on their own pixels (one loader per scaling -- it picks samples by index, never by pixel, so both
  serve the same ones) and every stored image is in `--units` (default volume_scale: each volume
  over its own in-brain median, prior and current on one scale). The conversion is the per-slice
  factor between the two loaders' targets, exact since both are the stored image over a constant;
  a slice one factor does not explain stops the dump. Metrics are computed in those units with the
  run's `data_range`.
* REGISTRATION WITHOUT THE TARGET (`--register`, default prior_to_current). The build aligned the
  studies with masks that saw the current CT1, which inference cannot. By default each (current,
  prior) pair is re-registered from the FLAIR, T1 and T2 of both studies only, and the prior --
  all four contrasts -- resampled onto the current study (`current_to_prior` warps the current
  study onto the prior instead; `none` keeps the stored alignment). One fit per pair, shared by
  every run and scaling, kept in --reg-cache for reruns. Each file records `registration` and the
  fit's correlation before / after (`reg_ncc_init`, `reg_ncc`, `reg_fit_ok`).
* A MASK INFERENCE COULD MAKE (`--mask-source`, default auto). The runs' own `t1ct1` mask is the
  support of the current T1 AND the current CT1 -- the scan being predicted. By default the dump
  masks with the current T1 support AND the registered prior CT1's (`t1_prior_ct1`), or the
  current T1 alone without registration (`t1`). It multiplies every stored image and bounds every
  metric, so dumps with different masks are not comparable (the viewer checks `mask_source`).
* PAIRED NOISE. The torch RNG is reset to the same value per (volume, chunk) before every run, so
  bridge runs draw identical noise; what separates them is the net.

Slice range
-----------
`--slices 80:100` are ORIGINAL slice indices (the loader's `slice_range`), not stack positions.
The runs trained on [50, 90), so 90-99 are slices they never saw.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import os
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datasets                                    # noqa: F401  registers the loaders
from datasets.registry import build_loader
from datasets.NYUMets.inference_registration import describe
from visualization.image import orient_tensor

DUMP_VERSION = 1
NYU = "trained_nets/nyumets"
DEFAULT_RUNS = [
    NYU + "/I2SB_Unet_NYUMets_CT1_from_all_bs64",                        # UNet bridge, no prior
    NYU + "/I2SB_Unet_NYUMets_CT1_from_all_plus_otherT1CT1_w2_bs64",     # + prior T1/CT1 z-2..z+2
]
METRICS = ("psnr", "ssim", "nrmse")


# ---------------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------------
def norm_key(v):
    """A loader block's intensity scaling, as a hashable key: ("volume_scale",) -- `scales` is
    ignored then -- or ("none", scales)."""
    n = v.get("intensity_norm") or "none"
    return (n,) if n == "volume_scale" else (n, tuple(float(s) for s in (v.get("scales") or ())))


def load_run(run_dir, device, weights):
    """A trained i2sb / synthesis run -> everything needed to run it and to rebuild its inputs."""
    from models import build_model                 # late: pulls in CUDA / flex bits
    from training.common import load_ckpt

    path = os.path.join(run_dir, "config.json")
    with open(path) as f:
        cfg = yaml.safe_load(f)                    # train.py writes it through YAML
    task = cfg.get("task")
    if task not in ("i2sb", "synthesis"):
        raise ValueError(f"task {task!r} is not i2sb or synthesis")

    ema = os.path.join(run_dir, "net_ema.ckpt")
    live = os.path.join(run_dir, "net.ckpt")
    use_ema = weights == "ema" and os.path.exists(ema)
    ckpt = ema if use_ema else live
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"no checkpoint at {ckpt}")
    net = build_model(cfg).to(device).eval()
    _, _, _, step = load_ckpt(ckpt, model=net, device=device)
    if getattr(net, "attn_backend", None) == "flex" and device.type == "cuda" \
            and hasattr(net, "compile_flex"):
        net.compile_flex()

    v = cfg["data"]["val"]
    modes = v.get("guide_mode") or "none"
    modes = [modes] if isinstance(modes, str) else list(modes)
    guided = "other_study" in modes
    if guided and v.get("guide_enhancement"):
        raise ValueError("guide_enhancement runs are not supported here")
    run = dict(dir=run_dir, name=os.path.basename(os.path.normpath(run_dir)), cfg=cfg, net=net,
               task=task, weights="ema" if use_ema else "live", step=int(step or 0),
               n_params=sum(p.numel() for p in net.parameters() if p.requires_grad),
               cond_idx=list(v.get("cond_idx") or []), guided=guided,
               guide_idx=list(v.get("guide_idx") or []) if guided else [],
               guide_window=int(v.get("guide_window") or 0) if guided else 0,
               guide_slice=v.get("guide_slice"),
               guide_as_cond=bool(v.get("guide_as_cond")) if task == "i2sb" else True,
               x0_idx=int(v.get("x0_idx", 2)), x1_idx=int(v.get("x1_idx", 1)),
               data_range=float(cfg["training"].get("data_range", 1.0)),
               pixels=(v.get("image_key"), v.get("mask_source"), bool(v.get("mask_apply")),
                       v.get("mask_guides")),
               norm=norm_key(v))
    if task == "i2sb":
        from sb.base import build_schedule
        i2 = cfg["i2sb"]
        run["sched"] = build_schedule(kind=i2.get("kind", "brownian"), tau=i2.get("tau", 0.1),
                                      n_points=i2.get("n_points", 1000),
                                      beta_max=i2.get("beta_max", 0.3), device=device)
        run["samp"] = dict(nfe=int(i2.get("val_nfe", 20)),
                           deterministic=bool(i2.get("deterministic", False)),
                           posterior=i2.get("posterior", "ddpm"),
                           clip_denoise=bool(i2.get("clip_denoise", False)), verbose=False)
        run["n_in"] = getattr(net, "C", None)
    else:
        p = cfg["model"]["params"]
        p = p["forward"]["params"] if "forward" in p else p
        run["n_in"] = p.get("in_chans")
    return run


# ---------------------------------------------------------------------------
# the shared loader and each run's slice of it
# ---------------------------------------------------------------------------
class Shared:
    """One loader configuration for every run, at every scaling the runs need."""

    def __init__(self, runs, base, slice_range, units, register=None, mask_source=None):
        self.cond = []
        for r in runs:
            for c in r["cond_idx"]:
                if c not in self.cond:
                    self.cond.append(c)
        guided = [r for r in runs if r["guided"]]
        bad = {r["name"]: r["guide_slice"] for r in guided if r["guide_slice"] != "index"}
        if bad:
            raise ValueError(f"guided runs must use guide_slice='index'; got {bad}")
        self.x0 = int(base.get("x0_idx", 2))
        self.gidx = sorted({c for r in guided for c in r["guide_idx"]} | {self.x0})
        self.W = max([r["guide_window"] for r in guided], default=0)
        self.nc = len(self.gidx)
        self.oct1 = self.W * self.nc + self.gidx.index(self.x0)   # prior CT1 at offset 0
        vcfg = dict(base)
        vcfg.update(name="nyumets_guided", cond_idx=self.cond, batch_size=1, num_workers=0,
                    guide_mode="other_study", guide_idx=self.gidx, guide_window=self.W,
                    guide_as_cond=False, n_guides=1, guide_slice="index",
                    guide_enhancement=False, x1_source="contrast", deterministic=True,
                    random_flips=False, crop_size=None, center_crop=None,
                    slice_range=list(slice_range),
                    # the SAME registration for every run and scaling: one fit per volume pair,
                    # shared in-process (and on disk with a cache_dir)
                    register=register)
        if mask_source is not None:
            # the mask every image is multiplied by and every metric averages over -- one for all
            # runs; mask_idx goes back to that source's default (t1 / t1_prior_ct1: [T1])
            vcfg.update(mask_source=mask_source, mask_idx=None)
        self.vcfg = vcfg
        self.units = (units,) if units == "volume_scale" else ("none", tuple(base.get("scales")))
        self.ds = {}
        for nk in dict.fromkeys([self.units] + [r["norm"] for r in runs]):
            c = dict(vcfg, intensity_norm=nk[0])
            if nk[0] != "volume_scale":
                c["scales"] = list(nk[1])
            self.ds[nk] = self._pinned(build_loader(c, shuffle=False, drop_last=False).dataset)
        ref = self.ds[self.units]
        for nk, ds in self.ds.items():
            if len(ds) != len(ref) or not np.array_equal(ds.file_id, ref.file_id):
                raise RuntimeError(f"the loader at scaling {nk[0]} holds different slices")

    @staticmethod
    def _pinned(ds):
        """Pin the prior-study pick to one index per VOLUME (see the module docstring)."""
        orig = ds._other_study_slice
        ds.pin = {}

        def pinned(fi, li, idx, salt):
            return orig(fi, li, ds.pin.get(fi, idx), salt)
        ds._other_study_slice = pinned
        return ds

    def volumes(self):
        """-> {volume name: [dataset indices in slice order]}, sorted by name."""
        ds = self.ds[self.units]
        out = {}
        for i, fi in enumerate(ds.file_id):
            out.setdefault(volume_name(ds.img_paths[int(fi)]), []).append(i)
        for name, idx in out.items():
            fi = int(ds.file_id[idx[0]])
            z = ds._slice_index(fi)
            idx.sort(key=lambda i: int(z[int(ds.local[i])]))
            for d in self.ds.values():
                d.pin[fi] = idx[0]
        return dict(sorted(out.items()))

    def batch(self, nk, idx, device):
        from training.i2sb import _split_batch
        items = [self.ds[nk][i] for i in idx]
        b = {k: torch.stack([it[k] for it in items]) for k in items[0]}
        x0, x1, cond, mask, _, guide = _split_batch(b, device)
        guide = guide.flatten(1, 2) if guide.dim() == 5 else guide     # (B,G,1,H,W) -> (B,G,H,W)
        return dict(x0=x0, x1=x1, cond=cond, m=(mask > 0.5).float(), guide=guide)

    def guide_for(self, run, guide):
        if not run["guided"]:
            return None
        w = run["guide_window"]
        sel = [(self.W + d) * self.nc + self.gidx.index(c)
               for d in range(-w, w + 1) for c in run["guide_idx"]]
        return guide[:, sel]

    def inputs(self, run, b):
        """-> (cond, guide=) exactly as this run was trained: guides as trailing cond channels
        (guide_as_cond, and every synthesis net) or through `guide=` as (B, G, 1, H, W)."""
        parts = [b["cond"][:, [self.cond.index(c) for c in run["cond_idx"]]]] \
            if run["cond_idx"] else []
        g = self.guide_for(run, b["guide"])
        gkw = None
        if g is not None and run["guide_as_cond"]:
            parts.append(g)
        elif g is not None:
            gkw = g.unsqueeze(2)
        return (torch.cat(parts, dim=1) if parts else None), gkw

    def check_channels(self, run):
        planes = len(run["guide_idx"]) * (2 * run["guide_window"] + 1) if run["guided"] else 0
        got = (1 if run["task"] == "i2sb" else 0) + len(run["cond_idx"]) \
            + (planes if run["guide_as_cond"] else 0)
        if run["n_in"] is not None and run["n_in"] != got:
            raise ValueError(f"{run['name']}: rebuilt {got} input channels, the net wants "
                             f"{run['n_in']}")


def volume_name(path):
    """`<root>/<case>/<case>_img.h5` -> `<case>`."""
    b = os.path.basename(path)
    return b[:-len("_img.h5")] if b.endswith("_img.h5") else os.path.splitext(b)[0]


def unit_factor(x_to, x_from, m, what):
    """(B, 1, 1, 1) per-slice c with x_to = c * x_from inside m (least squares), after checking
    that one factor really explains the slice -- i.e. both loaders served the same sample."""
    w = m * (x_from.abs() > 0) * (x_to.abs() > 0)
    num = (x_to * x_from * w).flatten(1).sum(1)
    den = (x_from * x_from * w).flatten(1).sum(1).clamp_min(1e-12)
    c = (num / den).view(-1, 1, 1, 1)
    rel = ((x_to - c * x_from) * w).flatten(1).norm(dim=1) \
        / (x_to * w).flatten(1).norm(dim=1).clamp_min(1e-12)
    if float(rel.max()) > 1e-3:
        raise RuntimeError(f"{what}: the loaders at different scalings served different "
                           f"samples (relative residual {float(rel.max()):.3g})")
    return c


# ---------------------------------------------------------------------------
# the dump
# ---------------------------------------------------------------------------
def run_net(run, sh, b, seed):
    from sb.i2sb import i2sb_sample
    cond, gkw = sh.inputs(run, b)
    torch.manual_seed(seed)                       # paired noise across runs
    if run["task"] == "i2sb":
        out, _, _ = i2sb_sample(run["net"], b["x1"], run["sched"], cond=cond, guide=gkw,
                                **run["samp"])
    else:
        out = run["net"](cond)
        out = out[0] if isinstance(out, (tuple, list)) else out
    out = out.real if torch.is_complex(out) else out
    return out[:, :1]


def write_volume(path, buf, attrs):
    import h5py
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        for k, v in attrs.items():
            f.attrs[k] = v
        for k, v in buf.items():
            f.create_dataset(k, data=np.concatenate(v), compression="gzip", compression_opts=4)


@torch.no_grad()
def dump(runs, sh, vols, args, device):
    from training.metrics import compute_metrics
    ref_ds = sh.ds[sh.units]
    rows = []
    for vi, (vol, idx) in enumerate(vols.items()):
        t0 = time.time()
        fi = int(ref_ds.file_id[idx[0]])
        bufs = {r["name"]: {} for r in runs}
        prior_fi = None
        for ci in range(0, len(idx), args.batch):
            chunk = idx[ci:ci + args.batch]
            B = {nk: sh.batch(nk, chunk, device) for nk in sh.ds}
            S = B[sh.units]
            m = S["m"]
            fac = {nk: unit_factor(S["x0"], b["x0"], m, f"{vol} {nk[0]}->{sh.units[0]}")
                   for nk, b in B.items()}
            z = ref_ds._slice_index(fi)[[int(ref_ds.local[i]) for i in chunk]].astype(np.int32)
            if prior_fi is None:
                prior_fi = ref_ds._other_study_slice(fi, int(ref_ds.local[chunk[0]]),
                                                     chunk[0], 1000)[0]
            seed = args.seed * 100003 + vi * 1000 + ci
            # stored images are turned into the display view (attr display_orient); the
            # metrics below use the unturned tensors, which no rotation changes anyway
            img = lambda t: orient_tensor(t, args.orient).float().cpu().numpy()   # noqa: E731
            mk = img(m[:, 0]).astype(np.uint8)
            shared = dict(reference=img((S["x0"] * m)[:, 0]), zero_filled=img((S["x1"] * m)[:, 0]),
                          prior_ct1=img(S["guide"][:, sh.oct1] * m[:, 0]))
            for r in runs:
                out = run_net(r, sh, B[r["norm"]], seed) * fac[r["norm"]]
                buf = bufs[r["name"]]
                for k, v in shared.items():
                    buf.setdefault(k, []).append(v)
                buf.setdefault("recon", []).append(img((out * m)[:, 0]))
                buf.setdefault("organ_mask", []).append(mk)
                buf.setdefault("slice_index", []).append(z)
                for j in range(out.shape[0]):
                    mj = m[j:j + 1]
                    met = compute_metrics(S["x0"][j:j + 1] * mj, out[j:j + 1] * mj,
                                          data_range=r["data_range"], mask=mj) \
                        if float(mj.sum()) > 0 else {k: float("nan") for k in METRICS}
                    vals = {k: float(met[k]) for k in METRICS}
                    for k in METRICS:
                        buf.setdefault(k + "_slice", []).append(np.float32([vals[k]]))
                    rows.append(dict(run=r["name"], volume=vol, slice=int(z[j]), **vals))
        prior = volume_name(ref_ds.img_paths[prior_fi]) if prior_fi is not None else ""
        reg = ref_ds.registration_info(fi, prior_fi) if prior_fi is not None else None
        reg_attrs = {} if reg is None else dict(
            reg_fit_ok=bool(reg["fit_ok"]), reg_ncc_init=float(reg["ncc_init"]),
            reg_ncc=float(reg["ncc"]), reg_note=str(reg.get("why", "")))
        for r in runs:
            attrs = dict(
                dump_version=DUMP_VERSION, run=r["name"], run_dir=r["dir"], task=r["task"],
                model_type=r["cfg"]["model"]["type"],
                experiment=r["cfg"].get("experiment", {}).get("name", ""),
                volume=vol, prior_volume=prior, anatomy="brain", seed=int(args.seed),
                sigma=0.0, R=0,
                intensity_units=sh.units[0], trained_intensity_norm=r["norm"][0],
                slice_range=np.int32(args.slices), data_range=r["data_range"],
                nfe=int(r["samp"]["nfe"]) if r["task"] == "i2sb" else 1,
                deterministic=bool(r["samp"]["deterministic"]) if r["task"] == "i2sb" else True,
                weights=r["weights"], step=int(r["step"]), n_params=int(r["n_params"]),
                use_organ_mask=True, images_masked=True, display_orient=args.orient or "",
                mask_source=str(sh.vcfg.get("mask_source") or ""), metrics=list(METRICS),
                registration=describe(ref_ds.register), **reg_attrs,
                created=datetime.datetime.now().isoformat(timespec="seconds"))
            write_volume(os.path.join(dest_dir(r, args), vol + ".h5"), bufs[r["name"]], attrs)
        line = "  ".join("%s %.2f" % (r["name"][-14:], np.nanmean(
            [x["psnr"] for x in rows if x["run"] == r["name"] and x["volume"] == vol]))
            for r in runs)
        regl = ("" if reg is None else
                f"reg ncc {reg['ncc_init']:.3f}->{reg['ncc']:.3f}{'' if reg['fit_ok'] else ' (kept identity)'}  ")
        print(f"[{vi + 1}/{len(vols)}] {vol}: {len(idx)} slices, prior {prior}  {regl}"
              f"psnr {line}  ({time.time() - t0:.1f} s)", flush=True)
    return rows


def dest_dir(run, args):
    return os.path.join(args.out, run["name"]) if args.out else os.path.join(run["dir"], "eval_dump")


def summarize(rows, runs, path):
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["run", "volume", "slice", *METRICS])
            w.writeheader()
            w.writerows(rows)
        print(f"\nper-slice metrics -> {path}")
    names = [r["name"] for r in runs]
    print("\n%-58s %s" % ("run (mean +- std over slices; median)", "   ".join(
        "%-22s" % k for k in METRICS)))
    per = {}
    for n in names:
        sel = [x for x in rows if x["run"] == n]
        per[n] = {(x["volume"], x["slice"]): x for x in sel}
        cells = []
        for k in METRICS:
            v = np.array([x[k] for x in sel], dtype=float)
            v = v[np.isfinite(v)]
            cells.append("%7.4f +- %6.4f; %7.4f" % (v.mean(), v.std(), np.median(v)))
        print("%-58s %s" % (n[:58], "   ".join(cells)))
    base = names[0]
    for n in names[1:]:
        keys = sorted(set(per[n]) & set(per[base]))
        d = np.array([per[n][k]["psnr"] - per[base][k]["psnr"] for k in keys])
        d = d[np.isfinite(d)]
        vols = {}
        for k in keys:
            vols.setdefault(k[0], []).append(per[n][k]["psnr"] - per[base][k]["psnr"])
        vm = np.array([np.nanmean(v) for v in vols.values()])
        print(f"\n{n}\n  vs {base}: PSNR {d.mean():+.3f} dB per slice (better on "
              f"{100 * (d > 0).mean():.0f}% of {d.size}); per volume {vm.mean():+.3f} dB, better "
              f"on {(vm > 0).sum()}/{vm.size}")


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", default=DEFAULT_RUNS,
                   help="run directories (each with config.json and net[_ema].ckpt)")
    p.add_argument("--n-volumes", type=int, default=60,
                   help="first N volumes in sorted order (0 = every volume)")
    p.add_argument("--volumes", nargs="*", help="explicit volume names instead")
    p.add_argument("--slices", default="80:100",
                   help="ORIGINAL slice indices lo:hi (hi exclusive)")
    p.add_argument("--split-root", help="val h5 root; default = the first run's val root")
    p.add_argument("--register", choices=["prior_to_current", "current_to_prior", "none"],
                   default="prior_to_current",
                   help="inference-time registration of each (current, prior) pair from FLAIR, "
                        "T1, T2 only (datasets/NYUMets/inference_registration.py); 'none' = the "
                        "build's stored alignment, which saw the current CT1")
    p.add_argument("--reg-cache", default="cache/nyumets_reg",
                   help="directory the per-pair transforms are kept in, so a rerun does not refit "
                        "('' = this process only)")
    p.add_argument("--reg-device", help="device for the registration fits (default: --device)")
    p.add_argument("--mask-source", choices=["auto", "config", "t1", "t1_prior_ct1", "t1ct1"],
                   default="auto",
                   help="the mask images are multiplied by and metrics averaged over. auto "
                        "(default): what inference has -- t1_prior_ct1 (current T1 support AND "
                        "the registered prior CT1's) with --register, t1 without; config: the "
                        "runs' own setting (t1ct1 uses the current CT1)")
    p.add_argument("--units", choices=["volume_scale", "none"], default="volume_scale",
                   help="the intensity units every stored image and metric is in")
    p.add_argument("--weights", choices=["ema", "live"], default="ema",
                   help="net_ema.ckpt when a run has one (default), else net.ckpt")
    p.add_argument("--nfe", type=int, help="override every bridge run's val_nfe")
    p.add_argument("--deterministic", action="store_true",
                   help="noise-free (OT-ODE) reverse process for every bridge run")
    p.add_argument("--orient", choices=["radiological", "neurological", "none"],
                   default="radiological",
                   help="view the stored images are turned into (the h5s are canonical RAS, "
                        "which draws the eyes on the right); 'none' stores them as read")
    p.add_argument("--batch", type=int, default=10, help="slices per forward call")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", help="write <out>/<run>/<volume>.h5 instead of <run>/eval_dump/")
    p.add_argument("--csv", default="results/nyumets_eval/per_slice.csv",
                   help="per-slice metrics of every run ('' = none)")
    p.add_argument("--device")
    args = p.parse_args()

    lo, _, hi = args.slices.partition(":")
    args.slices = [int(lo), int(hi)]
    args.orient = None if args.orient == "none" else args.orient
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    runs = []
    for d in args.runs:
        try:
            r = load_run(d, device, args.weights)
        except (FileNotFoundError, OSError, ValueError) as e:
            print(f"[skip] {d}: {type(e).__name__}: {e}")
            continue
        if r["task"] == "i2sb":
            if args.nfe:
                r["samp"]["nfe"] = int(args.nfe)
            if args.deterministic:
                r["samp"]["deterministic"] = True
        runs.append(r)
        print(f"loaded {r['name']}: {r['task']} {type(r['net']).__name__} {r['weights']} "
              f"step {r['step']} | trained on {r['norm'][0]} | "
              + (f"guide {r['guide_idx']} x {2 * r['guide_window'] + 1} "
                 f"{'as channels' if r['guide_as_cond'] else 'through guide='}"
                 if r["guided"] else "no prior"))
    if not runs:
        raise SystemExit("no run could be loaded")
    ref = runs[0]
    for r in runs[1:]:
        for k in ("pixels", "x0_idx", "x1_idx"):
            if r[k] != ref[k]:
                raise SystemExit(f"{r['name']}: {k} {r[k]} != {ref[k]} ({ref['name']}); the "
                                 f"runs would not be served the same pixels")
        if r["data_range"] != ref["data_range"]:
            raise SystemExit(f"{r['name']}: data_range differs; the metrics would not compare")
    base = dict(ref["cfg"]["data"]["val"])
    if args.split_root:
        base["root"] = args.split_root
    roots = {r["name"]: r["cfg"]["data"]["val"].get("root") for r in runs}
    if len(set(roots.values())) > 1:
        print(f"NOTE: runs trained on different roots; all are scored on {base['root']}: {roots}")

    register = None if args.register == "none" else dict(
        direction=args.register, device=str(args.reg_device or device),
        cache_dir=args.reg_cache or None)
    mask_source = {"auto": "t1_prior_ct1" if register else "t1",
                   "config": None}.get(args.mask_source, args.mask_source)
    sh = Shared(runs, base, args.slices, args.units, register, mask_source)
    for r in runs:
        sh.check_channels(r)
    vols = sh.volumes()
    n_all = len(vols)
    if args.volumes:
        missing = sorted(set(args.volumes) - set(vols))
        if missing:
            raise SystemExit(f"not in the split / slice range: {missing}")
        vols = {k: vols[k] for k in args.volumes}
    elif args.n_volumes:
        vols = dict(list(vols.items())[:args.n_volumes])
    print(f"\n{len(runs)} run(s) | {len(vols)} volume(s) of {n_all} multi-study | "
          f"slices {args.slices[0]}:{args.slices[1]} | units {sh.units[0]} | device {device} | "
          f"registration {describe(sh.ds[sh.units].register)} | mask {sh.vcfg.get('mask_source')}")

    rows = dump(runs, sh, vols, args, device)
    for r in runs:
        print(f"  {r['name']} -> {dest_dir(r, args)}")
    summarize(rows, runs, args.csv)


if __name__ == "__main__":
    main()
