"""
Draw the h5 masks over NYUMets slices, to judge what a data-fidelity region should be.

    python -m scripts.show_nyumets_masks --root ../datasets/NYUMets_h5_val --n 4 \
           --out figs/nyumets_masks.png

or, in a notebook (returns the figure, nothing is written):

    from scripts.show_nyumets_masks import mask_panel
    fig, rows = mask_panel("../datasets/NYUMets_h5_val", n=3, seed=0)

Each row is one slice: FLAIR, T1, CT1, T2 and the T1 - CT1 difference, with

    lime   `mask`     the h5 brain mask -- brain_mask(): mean_c(v / p99.5_c) > bg_frac. This is
                      what the loader returns and what use_mask applies to every loss, INCLUDING
                      the forward operator's. Averaging makes it generous: bright orbital fat on
                      T1 clears the threshold on its own, so the eyes are usually inside it.
    orange `support`  the common acquisition support -- the INTERSECTION of all four contrasts'
                      supports. It answers "was this voxel acquired", not "does this contrast
                      show tissue", so orbits inside every FOV stay in. Under the current
                      builder default (`--support store`) it is RECORDED ONLY and the pixels
                      outside it are intact; h5s built before 2026-09-22 have it multiplied
                      into every channel instead. The `support_applied` attr says which.

The printed table gives each region's area fraction and, inside `mask`, the share of squared
T1 - CT1 that falls OUTSIDE an eroded brain mask -- i.e. how much of the fidelity budget the rim
(orbits, scalp fat, skull base) would take. Post-contrast T1 is often fat-suppressed while T1 is
not, so that rim is where E: CT1 -> T1 cannot win.

`--erode` adds a third contour (cyan): `mask` eroded by that many pixels, the cheapest candidate
fidelity region to eyeball before wiring one in.
"""

import argparse
import glob
import os

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from visualization import image as _vis
from visualization.image import orient_image, set_display_orient, subplot_images

# This viewer only ever draws NYUMets, whose h5s store canonical-RAS axes:
# without this the eyes come out on the image's right.
set_display_orient("radiological")

CONTRASTS = ["FLAIR", "T1", "CT1", "T2"]


def _erode(m, r):
    """Binary erosion by a (2r+1) square, on a (H, W) float array."""
    if r <= 0:
        return m
    t = torch.as_tensor(m, dtype=torch.float32)[None, None]
    return (-F.max_pool2d(-t, 2 * r + 1, stride=1, padding=r))[0, 0].numpy()


def _sessions(root, n, seed):
    paths = sorted(glob.glob(os.path.join(root, "*", "*_img.h5")))
    if not paths:
        raise RuntimeError(f"no */*_img.h5 under {root}")
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(paths), size=min(n, len(paths)), replace=False)
    return [paths[i] for i in sorted(pick)]


def mask_panel(root, n=4, seed=0, image_key="img_median_mad", scale=3.0, erode=0, z=None,
               show=False):
    """-> (fig, rows). `rows` holds the per-slice statistics the CLI prints."""
    imgs, ovls, labels, rows = [], [], [], []
    for path in _sessions(root, n, seed):
        with h5py.File(path, "r") as f:
            nz = f[image_key].shape[0]
            k = nz // 2 if z is None else min(max(int(z), 0), nz - 1)
            img = np.asarray(f[image_key][k]) / scale              # (H, W, 4)
            mask = np.asarray(f["mask"][k]).squeeze(-1).astype(np.float32)
            sup = (np.asarray(f["support"][k]).squeeze(-1).astype(np.float32)
                   if "support" in f else None)
            sid = f.attrs.get("subject_id", os.path.basename(os.path.dirname(path)))

        t1, ct1 = img[..., 1], img[..., 2]
        er = _erode(mask, erode) if erode else None
        d2 = (t1 - ct1) ** 2
        rim = mask - (er if er is not None else _erode(mask, 3))
        rows.append({
            "session": str(sid), "slice": k,
            "mask_frac": float(mask.mean()),
            "support_frac": float(sup.mean()) if sup is not None else float("nan"),
            # share of the (T1 - CT1)^2 budget inside `mask` that sits in the eroded-away rim
            "rim_share_of_sq_diff": float((d2 * rim).sum() / max((d2 * mask).sum(), 1e-8)),
            "rim_area_share": float(rim.sum() / max(mask.sum(), 1e-8)),
        })

        imgs.append([img[..., 0], t1, ct1, img[..., 3], t1 - ct1])
        ovls.append([mask, sup, er])
        labels.append(f"{sid}  z={k}")

    v = float(np.percentile(np.abs(imgs[0][4]), 99)) if imgs else 1.0
    fig, axes = subplot_images(
        imgs, row_labels=labels,
        col_titles=[*CONTRASTS, "T1 - CT1"],
        cmap=["gray", "gray", "gray", "gray", "RdBu_r"],
        vmin=[None, None, None, None, -v], vmax=[None, None, None, None, v],
        p=(1, 99), magnitude=False, panel_size=(2.7, 3.0),
        suptitle="NYUMets masks: lime = h5 `mask` (brain), orange = `support` (common FOV)"
                 + (f", cyan = mask eroded by {erode}px" if erode else ""),
        show=False)

    # subplot_images takes ONE overlay colour per call, and these are three different regions,
    # so the extra contours go on afterwards -- same axes, same pixel grid.
    #
    # They bypass plot_image, so they do NOT get its display rotation for free: orient them by
    # hand or every contour lands transposed on top of an already-rotated panel.
    orient = _vis._DEFAULT_ORIENT
    for i, (mask, sup, er) in enumerate(ovls):
        mask, sup, er = (None if m is None else orient_image(m, orient) for m in (mask, sup, er))
        for j in range(len(CONTRASTS) + 1):
            ax = axes[i, j]
            for m, col, lw in ((mask, "lime", 0.7), (sup, "orange", 0.7), (er, "cyan", 0.7)):
                if m is not None and np.nanmax(m) > 0:
                    ax.contour(m, levels=[0.5], colors=col, linewidths=lw)
    if show:
        import matplotlib.pyplot as plt
        plt.show()
    return fig, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="../datasets/NYUMets_h5_val")
    ap.add_argument("--n", type=int, default=4, help="sessions to sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--z", type=int, default=None, help="slice index (default: central)")
    ap.add_argument("--image-key", default="img_median_mad")
    ap.add_argument("--scale", type=float, default=3.0, help="divisor, as in the configs")
    ap.add_argument("--erode", type=int, default=0, help="also contour mask eroded by R pixels")
    ap.add_argument("--out", default="figs/nyumets_masks.png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    fig, rows = mask_panel(args.root, n=args.n, seed=args.seed, image_key=args.image_key,
                           scale=args.scale, erode=args.erode, z=args.z)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out}\n")
    hdr = f"{'session':>28s} {'z':>4s} {'mask':>6s} {'support':>8s} {'rim area':>9s} {'rim (T1-CT1)^2':>15s}"
    print(hdr)
    for r in rows:
        print(f"{r['session']:>28s} {r['slice']:>4d} {r['mask_frac']:6.3f} "
              f"{r['support_frac']:8.3f} {r['rim_area_share']:9.3f} "
              f"{r['rim_share_of_sq_diff']:15.3f}")
    print("\nrim = mask minus its erosion (--erode, default 3px): the orbital / scalp band. A rim "
          "share of (T1-CT1)^2 far above its area share means the fidelity budget is dominated by "
          "tissue E cannot predict.")


if __name__ == "__main__":
    main()
