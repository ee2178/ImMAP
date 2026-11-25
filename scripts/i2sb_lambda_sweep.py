"""I2SB-Recon: how the CG data-consistency weight lam changes the reconstruction, on a few random
validation slices -- a quick look before committing to a dump.

Every x0 prediction x_hat of the reverse-bridge sampler goes through

    argmin_x ||y - E x||^2 + lam ||x - x_hat||^2        (--dc-iters CG steps from x_hat)

(sb/i2sb_recon.py::cg_data_consistency). Small lam fits the measured k-space exactly (the runs use
7e-5, I2SB-Inversion's value); large lam keeps the net's guess; "no CG" is the plain sampler, which
never sees the measurement (the bridge UNet takes x_t and, for _x1, the guide only).

Each slice goes through the dump's own code path (evaluation/tasks.py::i2sb_recon, the run's
sampler settings, measured k-space, online maps) and is scored as scripts/dump_eval.py scores it:
PSNR / SSIM with the peak fixed at 1, inside the organ mask of the stored coil maps. Every setting
of every run sees the same measurement and the same bridge noise on a slice (one generator seed per
slice), so two columns differ in lam alone.

    python scripts/i2sb_lambda_sweep.py                                # both sbunet_small runs
    python scripts/i2sb_lambda_sweep.py --n-samples 12 --lams 1e-3 1e-2 1e-1 1
    python scripts/i2sb_lambda_sweep.py --sample-seed 3                # the same slices again
    bash bigpurple/i2sb_lambda_sweep.sh [args]                         # on bigpurple, in an
                                                                       # interactive GPU job

Writes <out>/summary.txt (the table), per_slice.csv, and <run>.png (rows = slices, columns = no CG,
each lam, ground truth; then the residuals |gt - recon| x 4).
"""

import argparse
import csv
import datetime
import os
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.join(REPO, "scripts")]
os.chdir(REPO)                       # run configs hold repo-relative paths (split CSVs, maps)

import dump_eval as de                                                          # noqa: E402
from datasets.registry import build_loader                                      # noqa: E402
from evaluation.metrics import build_metrics                                    # noqa: E402
from evaluation.tasks import (EVAL_KSPACE_TYPE, build_adapter, default_sigma,   # noqa: E402
                              force_organ_mask)

RUNS = ["sbunet_small_R8_g1_tau0.19", "sbunet_small_x1_R8_g1_tau0.19"]
LAMS = [7e-5, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
METRICS = ("psnr", "ssim", "nrmse")


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-root", default="trained_nets/guided_recon/knee/s4_noise-vlow")
    p.add_argument("--runs", nargs="+", default=RUNS, help="i2sb_recon run directories")
    p.add_argument("--ckpt", default="net_bestval.ckpt")
    p.add_argument("--lams", nargs="+", type=float, default=LAMS)
    p.add_argument("--dc-iters", type=int, default=7, help="CG steps per prediction")
    p.add_argument("--nfe", type=int, default=None, help="sampler steps (default: the run's)")
    p.add_argument("--no-nodc", action="store_true", help="skip the plain sampler (no CG)")
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--slices", default="12:25",
                   help="slice range to draw from (12:25 = the range the nets trained on)")
    p.add_argument("--sample-seed", type=int, default=None,
                   help="which slices (default: a fresh random draw, printed for reuse)")
    p.add_argument("--seed", type=int, default=1234, help="measurement / bridge-noise seed")
    p.add_argument("--out", default=None,
                   help="output directory (default bigpurple/logs/lambda_sweep_<time>)")
    p.add_argument("--no-figures", action="store_true")
    return p.parse_args()


def settings(args):
    """[(label, dc_iters, lam)]: the plain sampler first, then each lam."""
    out = [] if args.no_nodc else [("no CG", 0, 0.0)]
    return out + [(f"lam {lam:g}", args.dc_iters, float(lam)) for lam in args.lams]


def val_dataset(cfg, lo, hi):
    val = dict(cfg["data"]["val"])
    val.update(enumerate_slices=True, start_slice=lo, end_slice=hi, batch_size=1, num_workers=0)
    return build_loader(val, shuffle=False, drop_last=False).dataset


def pick_slices(ds, n, seed):
    """n (volume, slice) pairs: distinct volumes first (one random slice each), then repeats."""
    by_vol = {}
    for i in range(len(ds)):
        v, s = ds.item_id(i)
        by_vol.setdefault(v, []).append(s)
    rng = random.Random(seed)
    vols = sorted(by_vol)
    rng.shuffle(vols)
    out = []
    while len(out) < n:
        for v in vols:
            free = [s for s in by_vol[v] if (v, s) not in out]
            if free and len(out) < n:
                out.append((v, rng.choice(free)))
        if all(not [s for s in by_vol[v] if (v, s) not in out] for v in vols):
            break                                        # fewer slices than asked for
    return out


def _np(t):
    return t.detach().cpu().numpy().squeeze()


def figure(path, title, picks, cols, ref, mask):
    """rows = slices; columns = each setting, then ground truth; images above, residuals below."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_r, n_c = len(picks), len(cols) + 1
    fig, axs = plt.subplots(2 * n_r, n_c, figsize=(1.9 * n_c, 2 * 2.9 * n_r), squeeze=False)
    for r, key in enumerate(picks):
        m = mask[key]
        g = np.flipud(np.abs(ref[key]))
        mm = np.flipud(m).astype(bool) if m is not None else None
        if mm is not None and mm.any():
            rows, cc = np.where(mm)
            rs = slice(max(rows.min() - 6, 0), rows.max() + 7)
            cs = slice(max(cc.min() - 6, 0), cc.max() + 7)
        else:
            rs = cs = slice(None)
        vmax = float(np.percentile(g[mm] if mm is not None and mm.any() else g, 99.9)) * 0.85
        for c, (lab, imgs, mets) in enumerate(cols + [("Ground truth", ref, None)]):
            im = np.flipud(np.abs(imgs[key]))
            for half, img, cmap in ((0, im, "gray"), (1, np.abs(im - g) * 4, "inferno")):
                a = axs[2 * r + half, c]
                if mets is None and half == 1:
                    a.axis("off")
                    continue
                a.imshow(img[rs, cs], cmap=cmap, vmin=0, vmax=vmax)
                a.set_xticks([]), a.set_yticks([])
                if mets is not None:
                    x = mets[key]
                    a.set_xlabel(f"{x['psnr']:.2f} dB / {x['ssim']:.3f}" if half == 0
                                 else f"NRMSE {x['nrmse']:.3f}", fontsize=7)
                if r == 0 and half == 0:
                    a.set_title(lab, fontsize=9)
        axs[2 * r, 0].set_ylabel(f"{key[0]} / {key[1]}", fontsize=8)
        axs[2 * r + 1, 0].set_ylabel("|gt - recon| x4", fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    args = parse()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type != "cuda":
        print("[warn] no GPU: the sampler will be slow")
    sample_seed = args.sample_seed if args.sample_seed is not None else random.randrange(10 ** 6)
    out = args.out or os.path.join(
        "bigpurple", "logs", "lambda_sweep_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out, exist_ok=True)
    lo, hi = de.parse_slices(args.slices)
    metric_fns = build_metrics(list(METRICS))
    sets = settings(args)

    picks, results, lines = None, {}, []
    for run in args.runs:
        run_dir = os.path.join(args.runs_root, run)
        if not os.path.exists(os.path.join(run_dir, args.ckpt)):
            print(f"[skip] {run}: no {args.ckpt} in {run_dir}")
            continue
        cfg, net, _, step = de.load_run(run_dir, dev, args.ckpt)
        if cfg["task"] != "i2sb_recon":
            print(f"[skip] {run}: task {cfg['task']!r}, not i2sb_recon")
            continue
        force_organ_mask(cfg)                                   # as dump_eval's default
        cfg.setdefault("training", {})["val_kspace_type"] = EVAL_KSPACE_TYPE
        i2 = cfg["i2sb"]
        ds = val_dataset(cfg, lo, hi)
        if picks is None:
            picks = pick_slices(ds, args.n_samples, sample_seed)
            print(f"slices (--sample-seed {sample_seed}): "
                  + ", ".join(f"{v}/{s}" for v, s in picks))
        where = {ds.item_id(i): i for i in range(len(ds))}
        adapter = build_adapter("i2sb_recon")
        sigma = default_sigma(cfg)
        nfe = args.nfe or i2.get("val_nfe", 20)
        print(f"\n{run} (step {step}): nfe {nfe}, the run's own dc {i2.get('dc_iters', 0)} "
              f"iters / lam {i2.get('dc_lam', 7e-5):g}, x1_source "
              f"{i2.get('x1_source', 'adjoint')}, cond_x1 {getattr(net, 'cond_x1', '?')}")

        ref, mask = {}, {}
        cols = {lab: ({}, {}) for lab, _, _ in sets}             # label -> (images, metrics)
        t0 = time.time()
        for k, key in enumerate(picks):
            batch = next(iter(DataLoader(Subset(ds, [where[key]]), batch_size=1)))
            for lab, it, lam in sets:
                cfg["eval"] = dict(dc_iters=it, dc_lam=lam, also_nodc=False, inversion=False,
                                   **({"nfe": int(args.nfe)} if args.nfe else {}))
                gen = torch.Generator(device=dev).manual_seed(args.seed + k)   # per slice
                with torch.no_grad():
                    gt, rec, mmask = adapter(net, batch, cfg, dev, sigma, gen, extras=None)
                cols[lab][1][key] = {n: float(fn(gt, rec, mask=mmask).reshape(-1)[0])
                                     for n, fn in metric_fns.items()}
                organ = (batch[3] != 0).float().to(dev) if len(batch) >= 4 else None
                keep = (lambda a: a) if organ is None else (lambda a: a * organ)  # noqa: E731
                cols[lab][0][key] = _np(keep(rec))
                ref[key] = _np(keep(gt))
                mask[key] = None if organ is None else _np(organ)
            print(f"  [{k + 1}/{len(picks)}] {key[0]}/{key[1]}: "
                  + "  ".join(f"{lab} {cols[lab][1][key]['psnr']:.2f}" for lab, _, _ in sets)
                  + f"   ({time.time() - t0:.0f} s)", flush=True)
        results[run] = cols
        if not args.no_figures:
            figure(os.path.join(out, f"{run}.png"),
                   f"{run}: CG weight lam ({args.dc_iters} CG iters, nfe {nfe}); "
                   "PSNR peak 1 / SSIM, inside the organ mask", picks,
                   [(lab, cols[lab][0], cols[lab][1]) for lab, _, _ in sets], ref, mask)
        del net
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    if not results:
        sys.exit("no run could be loaded")

    # ---- the table: mean +- std over the slices, the best lam per metric marked
    lines.append(f"slices ({len(picks)}, --sample-seed {sample_seed}, seed {args.seed}): "
                 + ", ".join(f"{v}/{s}" for v, s in picks))
    lines.append(f"PSNR (dB, peak 1) / SSIM / NRMSE inside the organ mask; mean +- std over slices;"
                 f" {args.dc_iters} CG iters per prediction\n")
    for run, cols in results.items():
        lines.append(run)
        lines.append(f"  {'setting':<12}{'psnr':>15}{'ssim':>17}{'nrmse':>17}")
        means = {lab: {m: np.mean([cols[lab][1][key][m] for key in picks]) for m in METRICS}
                 for lab in cols}
        best = {m: (max if m != "nrmse" else min)(
            [lab for lab, it, _ in sets if it], key=lambda lab: means[lab][m]) for m in METRICS}
        for lab, _, _ in sets:
            cells = []
            for m, fmt in (("psnr", "{:6.2f} +- {:4.2f}"), ("ssim", "{:6.4f} +- {:5.3f}"),
                           ("nrmse", "{:6.4f} +- {:5.3f}")):
                v = [cols[lab][1][key][m] for key in picks]
                cells.append((fmt.format(np.mean(v), np.std(v))
                              + ("*" if best[m] == lab else " ")).rjust(17))
            lines.append(f"  {lab:<12}" + "".join(cells))
        lines.append("  * = best lam for that metric\n")
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(out, "summary.txt"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(out, "per_slice.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "volume", "slice", "setting", "dc_iters", "lam", *METRICS])
        for run, cols in results.items():
            for lab, it, lam in sets:
                for key in picks:
                    w.writerow([run, *key, lab, it, lam,
                                *(f"{cols[lab][1][key][m]:.6g}" for m in METRICS)])
    print(f"\nwrote {out}/  (summary.txt, per_slice.csv"
          + ("" if args.no_figures else ", " + ", ".join(f"{r}.png" for r in results)) + ")")


if __name__ == "__main__":
    main()
