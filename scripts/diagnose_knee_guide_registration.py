#!/usr/bin/env python3
"""
Are the paired fastMRI knee guides (PD) registered to their targets (PDFS)?

    python scripts/diagnose_knee_guide_registration.py config/knee/guided/lggs_R8_g1.json
    python scripts/diagnose_knee_guide_registration.py --self-test        # no data needed

READ-ONLY. The guided loader (datasets/fastmri/paired.py) pairs PD and PDFS slices BY INDEX, as
TGVN does, and never registers them. On the pairs of the config's split this measures:

  1. IN-PLANE SHIFT. The sub-pixel (dy, dx) that best aligns each guide slice to its target
     slice. dx is the PHASE-ENCODE direction (the image width), the axis the R=8 masks
     undersample and the one the horizontal recon blur runs along.
  2. THROUGH-PLANE. Whether the guide slice that best matches target slice n is n (the index
     pairing) or a neighbour n +/- k. Target-vs-target neighbours give the scale: how much
     similarity a one-slice offset costs.

Both images are the RSS of the FULLY-SAMPLED k-space of each contrast, centre-cropped to
320 x 320. No coil maps, scaling or masks are involved, so the test checks the pairing itself.

SIMILARITY: PD and PDFS differ in contrast (fat is bright in PD, suppressed in PDFS), so plain
correlation is the wrong score. The test uses normalized gradient fields (Haber & Modersitzki
2006): n(I) = grad I / sqrt(|grad I|^2 + eps^2), score = mean of (n_target . n_guide)^2 over the
target's foreground. It rewards edges that share an orientation, whatever their contrast or
polarity. Shifts are applied in the Fourier domain (exact sub-pixel translation). The search is
an integer grid to +/- --max-shift px, then a 1/8-px grid within +/- 1 px of the best.

`--self-test` checks the estimator on a synthetic two-contrast phantom with known shifts.
"""

import argparse
import json
import math
import os
import statistics
import sys

import h5py
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

CROP = 320


# ---------------------------------------------------------------------------------------------
# images
# ---------------------------------------------------------------------------------------------
def rss_image(kspace):
    """(C, H, W) complex k-space -> (H, W) RSS of its centred inverse FFT, centre-cropped."""
    from operators.fourier import ifftc
    k = torch.as_tensor(kspace).to(torch.complex64)
    img = ifftc(k).abs().pow(2).sum(0).sqrt()
    return center_crop(img, CROP)


def center_crop(img, n):
    H, W = img.shape[-2:]
    h, w = min(n, H), min(n, W)
    t, l = (H - h) // 2, (W - w) // 2
    return img[..., t:t + h, l:l + w]


def fourier_shift(img, dy, dx):
    """Translate a real (H, W) image by (dy, dx) pixels (positive = down / right)."""
    H, W = img.shape
    ky = torch.fft.fftfreq(H, device=img.device)[:, None]
    kx = torch.fft.fftfreq(W, device=img.device)[None, :]
    ph = torch.exp(-2j * math.pi * (ky * dy + kx * dx))
    return torch.fft.ifft2(torch.fft.fft2(img) * ph).real


def ngf(img, eps_rel=0.2, mask=None):
    """Normalized gradient field, (2, H, W). eps is eps_rel times the median gradient magnitude
    over `mask` (the target's foreground), so the noise floor counts as 'no edge'."""
    gy = torch.zeros_like(img)
    gx = torch.zeros_like(img)
    gy[1:-1] = (img[2:] - img[:-2]) / 2
    gx[:, 1:-1] = (img[:, 2:] - img[:, :-2]) / 2
    mag = (gy ** 2 + gx ** 2).sqrt()
    ref = mag[mask] if mask is not None and mask.any() else mag.flatten()
    eps = eps_rel * float(ref.median()) + 1e-12
    d = (mag ** 2 + eps ** 2).sqrt()
    return torch.stack([gy / d, gx / d])


def foreground(img, frac=0.1):
    """Pixels above `frac` of the image's 99th percentile, minus a 4-px border."""
    m = img > frac * torch.quantile(img.flatten(), 0.99)
    m[:4], m[-4:], m[:, :4], m[:, -4:] = False, False, False, False
    return m


def score(n_ref, mask, mov, eps_rel):
    """NGF similarity of a moving image to a reference's precomputed NGF, over `mask`."""
    n_mov = ngf(mov, eps_rel, mask)
    dot = (n_ref * n_mov).sum(0)
    return float((dot ** 2)[mask].mean())


def best_shift(ref, mov, max_shift=4, eps_rel=0.2):
    """Sub-pixel (dy, dx) that best aligns `mov` to `ref`, its score, and the zero-shift score.
    Returns the shift to APPLY to mov: mov moved by (dy, dx) matches ref."""
    mask = foreground(ref)
    n_ref = ngf(ref, eps_rel, mask)
    s0 = score(n_ref, mask, mov, eps_rel)
    best = (s0, 0.0, 0.0)
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            s = score(n_ref, mask, fourier_shift(mov, dy, dx), eps_rel)
            if s > best[0]:
                best = (s, float(dy), float(dx))
    _, cy, cx = best
    fine = np.arange(-1.0, 1.0 + 1e-9, 0.125)
    for ddy in fine:
        for ddx in fine:
            dy, dx = cy + ddy, cx + ddx
            s = score(n_ref, mask, fourier_shift(mov, dy, dx), eps_rel)
            if s > best[0]:
                best = (s, dy, dx)
    return best[1], best[2], best[0], s0


# ---------------------------------------------------------------------------------------------
# self-test: a synthetic two-contrast phantom with known shifts
# ---------------------------------------------------------------------------------------------
def phantom(n=CROP, seed=0):
    """Tissue labels from random ellipses, rendered in two contrasts (one inverts a class, as
    fat suppression does), with independent noise in each (magnitude, so Rician-like)."""
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, n), torch.linspace(-1, 1, n), indexing="ij")
    labels = torch.zeros(n, n, dtype=torch.long)
    labels[(yy / 0.85) ** 2 + (xx / 0.6) ** 2 < 1] = 1
    for k in range(40):
        cy, cx = (torch.rand(2, generator=g) * 1.2 - 0.6).tolist()
        ry, rx = (torch.rand(2, generator=g) * 0.15 + 0.03).tolist()
        th = float(torch.rand(1, generator=g)) * math.pi
        u = (yy - cy) * math.cos(th) + (xx - cx) * math.sin(th)
        v = -(yy - cy) * math.sin(th) + (xx - cx) * math.cos(th)
        labels[((u / ry) ** 2 + (v / rx) ** 2 < 1) & (labels > 0)] = 2 + k % 3
    a_val = torch.tensor([0.0, 0.5, 0.9, 0.3, 0.7])      # "PD": class 2 (fat) bright
    b_val = torch.tensor([0.0, 0.4, 0.05, 0.6, 0.2])     # "PDFS": fat suppressed, others remapped
    a = a_val[labels] + 0.02 * torch.randn(n, n, generator=g)
    b = b_val[labels] + 0.02 * torch.randn(n, n, generator=g)
    return a.abs(), b.abs()


def self_test(tol=0.25):
    """Same contrast must align exactly. Across contrasts the estimate scatters by about a grid
    step or two with the noise draw (sharp synthetic edges); `tol` bounds it. The data summary
    only flags shifts above 0.5 px."""
    worst, ok = 0.0, True
    for seed in range(3):
        guide, target = phantom(seed=seed)
        sy, sx, _, _ = best_shift(guide, guide)
        ok &= (sy, sx) == (0.0, 0.0)
        for dy, dx in [(0.0, 0.0), (0.5, -1.25), (-2.0, 0.375), (1.75, 3.0)]:
            moved = fourier_shift(guide, -dy, -dx)        # misregister by (-dy, -dx)
            ey, ex, s, s0 = best_shift(target, moved)
            err = max(abs(ey - dy), abs(ex - dx))
            worst = max(worst, err)
            print(f"  phantom {seed}: true ({dy:+.3f}, {dx:+.3f})  estimated ({ey:+.3f}, "
                  f"{ex:+.3f})  score {s0:.3f} -> {s:.3f}  error {err:.3f} px")
    ok &= worst <= tol + 1e-9
    print(f"self-test {'PASSED' if ok else 'FAILED'}: same-contrast exact; cross-contrast worst "
          f"error {worst:.3f} px (tolerance {tol}, grid step 0.125)")
    return ok


# ---------------------------------------------------------------------------------------------
# the data
# ---------------------------------------------------------------------------------------------
def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, max(0, int(round(q / 100 * (len(v) - 1)))))]


def describe(name, v, fmt="{:+.3f}"):
    if not v:
        return f"{name}: no values"
    return (f"{name}: median " + fmt.format(statistics.median(v)) + ", p5 " + fmt.format(pct(v, 5))
            + ", p95 " + fmt.format(pct(v, 95)) + f"  (n={len(v)})")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[1].strip(),
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", nargs="?", help="a guided config, e.g. config/knee/guided/lggs_R8_g1.json")
    p.add_argument("--split", default="val", choices=("train", "val", "test"))
    p.add_argument("--volumes", type=int, default=None, help="first N pairs only (default: all)")
    p.add_argument("--slice-frac", type=float, nargs=2, default=(0.25, 0.75),
                   help="slices used, as fractions of each volume (default: the middle half)")
    p.add_argument("--slice-step", type=int, default=2, help="every k-th slice in that range")
    p.add_argument("--max-shift", type=int, default=4)
    p.add_argument("--reach", type=int, default=2, help="through-plane offsets checked, +/-")
    p.add_argument("--eps-rel", type=float, default=0.2)
    p.add_argument("--out", default=None, help="per-slice JSON (default: next to nothing)")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        sys.exit(0 if self_test() else 1)
    if not args.config:
        p.error("give a config (or --self-test)")

    cfg = json.load(open(args.config))
    d = cfg["data"][args.split] if args.split in cfg["data"] else cfg["data"]["val"]
    from datasets.fastmri.paired import read_pairs
    roots = d["kspace_roots"] if isinstance(d["kspace_roots"], list) else [d["kspace_roots"]]

    def find(name):
        for r in roots:
            q = os.path.join(r, name)
            if os.path.exists(q):
                return q
        raise FileNotFoundError(f"{name} in none of {roots}")

    pairs = read_pairs(d["split_csv"])[:args.volumes]
    recs = []
    for vi, (pd_name, pdfs_name) in enumerate(pairs):
        with h5py.File(find(pd_name), "r") as fa, h5py.File(find(pdfs_name), "r") as fb:
            ka, kb = fa["kspace"], fb["kspace"]
            if ka.shape != kb.shape:
                print(f"[skip] {pdfs_name}: shapes {ka.shape} vs {kb.shape} (the loader drops it)")
                continue
            n = ka.shape[0]
            lo, hi = int(args.slice_frac[0] * n), int(math.ceil(args.slice_frac[1] * n))
            want = range(max(lo, args.reach), min(hi, n - args.reach), args.slice_step)
            need = sorted({j for s in want for j in range(s - args.reach, s + args.reach + 1)})
            guide = {j: rss_image(ka[j]) for j in need}
            target = {j: rss_image(kb[j]) for j in need}
        for s in want:
            t = target[s]
            dy, dx, sc, s0 = best_shift(t, guide[s], args.max_shift, args.eps_rel)
            mask = foreground(t)
            n_t = ngf(t, args.eps_rel, mask)
            through = {k: score(n_t, mask, guide[s + k], args.eps_rel)
                       for k in range(-args.reach, args.reach + 1)}
            self_nb = {k: score(n_t, mask, target[s + k], args.eps_rel)
                       for k in range(-args.reach, args.reach + 1) if k != 0}
            recs.append(dict(volume=pdfs_name, slice=s, dy=dy, dx=dx, score=sc, score0=s0,
                             through=through, best_k=max(through, key=through.get),
                             target_neighbours=self_nb))
        last = recs[-1] if recs else None
        if last:
            print(f"[{vi + 1}/{len(pairs)}] {pdfs_name}: {len([r for r in recs if r['volume'] == pdfs_name])} "
                  f"slices; last dy {last['dy']:+.3f} dx {last['dx']:+.3f} best_k {last['best_k']:+d}",
                  flush=True)

    if not recs:
        sys.exit("no slices measured")
    dys, dxs = [r["dy"] for r in recs], [r["dx"] for r in recs]
    gain = [r["score"] - r["score0"] for r in recs]
    print(f"\n{len(recs)} slices from {len({r['volume'] for r in recs})} volume pairs "
          f"({args.split}); shift = what to apply to the guide to match the target")
    print("  " + describe("dy (readout, vertical)      ", dys))
    print("  " + describe("dx (phase-encode, horizontal)", dxs))
    big = sum(1 for a, b in zip(dys, dxs) if max(abs(a), abs(b)) > 0.5)
    print(f"  slices shifted by more than 0.5 px: {big} / {len(recs)}")
    print("  " + describe("score gain from the best shift", gain, "{:.4f}"))
    ks = {}
    for r in recs:
        ks[r["best_k"]] = ks.get(r["best_k"], 0) + 1
    print("  through-plane, best-matching guide slice offset: "
          + ", ".join(f"{k:+d}: {ks.get(k, 0)}" for k in range(-args.reach, args.reach + 1)))
    s_same = [r["through"][0] for r in recs]
    s_nb = [statistics.mean([r["target_neighbours"][k] for k in (-1, 1)]) for r in recs]
    print("  " + describe("NGF score, guide slice n vs target n   ", s_same, "{:.3f}"))
    print("  " + describe("NGF score, target n+-1 vs target n      ", s_nb, "{:.3f}"))
    per_vol = {}
    for r in recs:
        per_vol.setdefault(r["volume"], []).append(r["dx"])
    worst = sorted(per_vol.items(), key=lambda kv: -abs(statistics.median(kv[1])))[:5]
    print("  volumes with the largest median |dx|: "
          + "; ".join(f"{v} {statistics.median(x):+.3f}" for v, x in worst))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"args": vars(args), "slices": recs}, f, indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
