#!/usr/bin/env python3
"""
Tiles for Figure 1 (the I2SB chain with the conditioning injected at every step): pick a slice with
clear enhancement in BOTH the current and the prior CT1, then export its real images and the real
reverse trajectory of the trained longitudinal bridge.

    # 1. rank candidate slices (CPU, reads the eval dumps only)
    python scripts/fig1_i2sb_tiles.py --rank -k 20

    # 2. export one slice (GPU): every tile as a PNG on one shared window, plus the raw arrays
    python scripts/fig1_i2sb_tiles.py --volume 10616619_2205499716 --slice 81

RANK scores every dumped slice of the longitudinal run (images already registered, masked and
turned radiological by scripts/dump_eval_nyumets.py) by its enhancement, in per-volume units
(tissue ~1):

    enh_cur    pixels where current CT1 - current T1 > --enh-thr
    enh_prior  pixels where registered prior CT1 - current T1 > --enh-thr  (the prior's
               enhancement against the same anatomy; the dump has no prior T1)
    score      min(enh_cur, enh_prior): both must show it

restricted to slices whose mask covers >= --min-brain of the image and where the longitudinal run
beats the baseline (a Figure 1 case should not be a failure). Vessels and dura enhance too, so
look at the top few in the viewer before choosing (the command for each is printed).

EXPORT rebuilds that one sample exactly as the dump did (scripts/dump_eval_nyumets.py: the same
loader, inference-time registration prior_to_current, mask, per-volume units) and reruns the
longitudinal bridge with every step logged. Writes, under --out/<volume>_s<slice>/:

    current_{flair,t1,t2,ct1}.png       the current exam (ct1 = acquired, for reference)
    prior_{t1,ct1}_z{-2..+2}.png        the registered prior at the five guide slices
    step_KK_t0.XX.png                   the bridge state x_t after each reverse step (KK = 00 is
                                        the start, x_1 = current T1)
    pred_KK_t0.XX.png                   the network's x0 estimate at that step
    synth_ct1.png                       the final synthesis
    tiles.npz, manifest.json            the raw (unwindowed) arrays, t values, window, PSNR

All intensity tiles share ONE window (0 to the --pct percentile of the acquired CT1 in the mask),
so brightness is comparable across the figure; images are masked like the dump and turned
radiological. The stochastic sampler is seeded (--seed), so the trajectory is reproducible.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
NYU = os.path.join("trained_nets", "nyumets")
BASE = "I2SB_Unet_NYUMets_CT1_from_all_bs64"
LONGI = "I2SB_Unet_NYUMets_CT1_from_all_plus_otherT1CT1_w2_bs64"


def dump_dir(root, run):
    for d in (os.path.join(root, run, "eval_dump"), os.path.join(root, run)):
        if os.path.isdir(d) and any(f.endswith(".h5") for f in os.listdir(d)):
            return d
    raise SystemExit(f"no dump for {run} under {root}")


# ---------------------------------------------------------------------------------------------
def rank(a):
    dl, db = dump_dir(a.root, a.longi), dump_dir(a.root, a.base)
    vols = sorted({f[:-3] for f in os.listdir(dl) if f.endswith(".h5")}
                  & {f[:-3] for f in os.listdir(db) if f.endswith(".h5")})
    out = []
    for i, v in enumerate(vols):
        with h5py.File(os.path.join(dl, v + ".h5"), "r") as fl, \
                h5py.File(os.path.join(db, v + ".h5"), "r") as fb:
            ct1, t1, pri = fl["reference"][:], fl["zero_filled"][:], fl["prior_ct1"][:]
            m = fl["organ_mask"][:] > 0
            z = fl["slice_index"][:].astype(int)
            pl, pb = fl["psnr_slice"][:], fb["psnr_slice"][:]
        for j in range(len(z)):
            mj = m[j]
            if mj.mean() < a.min_brain or not pl[j] > pb[j]:
                continue
            ec = int(((ct1[j] - t1[j]) > a.enh_thr)[mj].sum())
            ep = int(((pri[j] - t1[j]) > a.enh_thr)[mj].sum())
            out.append(dict(volume=v, slice=int(z[j]), enh_cur=ec, enh_prior=ep,
                            score=min(ec, ep), psnr_base=float(pb[j]), psnr_longi=float(pl[j])))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(vols)} volumes", flush=True)
    seen, top = set(), []
    for r in sorted(out, key=lambda r: -r["score"]):
        if r["volume"] in seen:
            continue
        seen.add(r["volume"])
        top.append(r)
        if len(top) == a.k:
            break
    print(f"\n{len(out)} slices scored; top {len(top)} (one per volume) by min(enh_cur, enh_prior), "
          f"enhancement = > {a.enh_thr} above the current T1:")
    print(f"  {'volume':<22} {'slice':>5} {'enh cur':>8} {'enh prior':>9} {'PSNR UNet':>9} {'w2':>6}")
    for r in top:
        print(f"  {r['volume']:<22} {r['slice']:>5} {r['enh_cur']:>8} {r['enh_prior']:>9} "
              f"{r['psnr_base']:>9.2f} {r['psnr_longi']:>6.2f}")
    if top:
        print("\nlook at one in the viewer:\n  python -m figures.webviewer figures/configs/"
              f"nyumets_longi.py --volume {top[0]['volume']} --slice {top[0]['slice']}\n"
              f"export it:\n  python scripts/fig1_i2sb_tiles.py --volume {top[0]['volume']} "
              f"--slice {top[0]['slice']}")


# ---------------------------------------------------------------------------------------------
def export(a):
    import torch
    from PIL import Image
    import scripts.dump_eval_nyumets as D
    from sb.i2sb import i2sb_sample
    from sb.base import n_steps
    from visualization.image import orient_tensor

    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    run = D.load_run(os.path.join(a.root, a.longi), device, "ema")
    if not run["guided"]:
        raise SystemExit(f"{a.longi} takes no prior study; Figure 1 needs the longitudinal run")
    base = dict(run["cfg"]["data"]["val"])
    if a.split_root:
        base["root"] = a.split_root
    reg = None if a.register == "none" else dict(direction=a.register, device=str(device),
                                                 cache_dir=a.reg_cache or None)
    mask_source = "t1_prior_ct1" if reg else "t1"
    sh = D.Shared([run], base, [a.slice, a.slice + 1], "volume_scale", reg, mask_source)
    vols = sh.volumes()
    if a.volume not in vols:
        # a unique prefix will do (the patient id alone, when that patient has one such exam)
        hits = [v for v in vols if v.startswith(a.volume)]
        if len(hits) != 1:
            raise SystemExit(f"{a.volume!r} slice {a.slice}: " + (
                f"matches {len(hits)} exams -- give one of: {', '.join(hits)}" if hits else
                "no such exam in the split (multi-study patients, with that slice)"))
        print(f"{a.volume} -> {hits[0]}")
        a.volume = hits[0]
    idx = vols[a.volume]
    b = sh.batch(sh.units, idx, device)
    cond, gkw = sh.inputs(run, b)

    nfe = int(run["samp"]["nfe"])
    torch.manual_seed(a.seed)
    samp = dict(run["samp"], log_count=nfe)
    with torch.no_grad():
        recon, xs, preds = i2sb_sample(run["net"], b["x1"], run["sched"], cond=cond, guide=gkw,
                                       **samp)
    # the visited steps (ascending); xs / preds are newest-first -> reversed to run from t = 1
    from sb.base import space_indices
    steps = np.asarray(space_indices(n_steps(run["sched"]), nfe + 1))
    t_of = lambda s: float(s) / float(n_steps(run["sched"]) - 1)        # noqa: E731
    traj = [b["x1"][0, 0].cpu()] + [xs[0, k, 0] for k in range(xs.shape[1] - 1, -1, -1)]
    pred = [preds[0, k, 0] for k in range(preds.shape[1] - 1, -1, -1)]
    t_traj = [t_of(steps[-1])] + [t_of(s) for s in steps[:-1][::-1]]
    t_pred = [t_of(s) for s in steps[1:][::-1]]

    m = b["m"][0, 0].cpu()
    turn = lambda x: orient_tensor((x * m)[None], "radiological")[0].numpy()   # noqa: E731
    cidx = {c: sh.cond.index(c) for c in sh.cond}
    gi = lambda d, c: (sh.W + d) * sh.nc + sh.gidx.index(c)              # noqa: E731
    G = b["guide"][0].cpu()
    arrays = {
        "current_flair": turn(b["cond"][0, cidx[0]].cpu()),
        "current_t1": turn(b["x1"][0, 0].cpu()),
        "current_t2": turn(b["cond"][0, cidx[3]].cpu()),
        "current_ct1": turn(b["x0"][0, 0].cpu()),
        "synth_ct1": turn(recon[0, 0].cpu()),
    }
    for d in range(-sh.W, sh.W + 1):
        for c, n in ((1, "t1"), (2, "ct1")):
            if c in sh.gidx:
                arrays[f"prior_{n}_z{d:+d}"] = turn(G[gi(d, c)])
    for k, (x, t) in enumerate(zip(traj, t_traj)):
        arrays[f"step_{k:02d}_t{t:.2f}"] = turn(x)
    for k, (x, t) in enumerate(zip(pred, t_pred)):
        arrays[f"pred_{k + 1:02d}_t{t:.2f}"] = turn(x)

    ref = arrays["current_ct1"]
    hi = float(np.percentile(ref[ref > 0], a.pct)) if (ref > 0).any() else 1.0
    dest = os.path.join(a.out, f"{a.volume}_s{a.slice}")
    os.makedirs(dest, exist_ok=True)
    for name, x in arrays.items():
        Image.fromarray((np.clip(x / hi, 0, 1) * 255 + 0.5).astype(np.uint8)).save(
            os.path.join(dest, name + ".png"))
    np.savez_compressed(os.path.join(dest, "tiles.npz"), **arrays)
    mk = turn(torch.ones_like(m)) > 0
    psnr = lambda x: float(10 * np.log10(run["data_range"] ** 2                 # noqa: E731
                                         / max(((x - ref) ** 2)[mk].mean(), 1e-12)))
    man = dict(volume=a.volume, slice=a.slice, run=a.longi, seed=a.seed, nfe=nfe,
               window=[0.0, hi], window_rule=f"0 .. p{a.pct} of the acquired CT1 in the mask",
               registration=D.describe(sh.ds[sh.units].register), mask_source=mask_source,
               psnr_synth=psnr(arrays["synth_ct1"]), t_steps=t_traj, t_preds=t_pred,
               files=sorted(n + ".png" for n in arrays))
    with open(os.path.join(dest, "manifest.json"), "w") as f:
        json.dump(man, f, indent=2)
    print(f"wrote {len(arrays)} tiles to {dest}  (window 0..{hi:.3f}, synthesis PSNR "
          f"{man['psnr_synth']:.2f} dB, {nfe} steps)")
    print(f"copy it down:\n  scp -r <cluster>:<ImMAP>/{dest} docs/ismrm_longi_synthesis/fig1/tiles/")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=os.path.join(HERE, NYU))
    p.add_argument("--base", default=BASE)
    p.add_argument("--longi", default=LONGI)
    p.add_argument("--rank", action="store_true", help="rank candidate slices and stop")
    p.add_argument("-k", type=int, default=15)
    p.add_argument("--enh-thr", type=float, default=0.3,
                   help="enhancement: CT1 above the current T1 by this much (per-volume units)")
    p.add_argument("--min-brain", type=float, default=0.25)
    p.add_argument("--volume")
    p.add_argument("--slice", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pct", type=float, default=99.5, help="window ceiling percentile")
    p.add_argument("--register", choices=["prior_to_current", "current_to_prior", "none"],
                   default="prior_to_current")
    p.add_argument("--reg-cache", default="cache/nyumets_reg")
    p.add_argument("--split-root")
    p.add_argument("--device")
    p.add_argument("--out", default=os.path.join("docs", "ismrm_longi_synthesis", "fig1", "tiles"))
    a = p.parse_args(argv)
    if a.rank:
        rank(a)
    elif a.volume and a.slice is not None:
        export(a)
    else:
        p.error("give --rank, or --volume and --slice")


if __name__ == "__main__":
    main()
