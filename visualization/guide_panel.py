# -*- coding: utf-8 -*-
"""
`val/guides`: the guide planes a longitudinal (or otherwise guided) net was actually given, for one
validation sample, next to the current images -- so a run's wandb page shows whether its guides
carry anything (the same slice, the same anatomy, visible change) without opening a notebook.

Two pieces, used by train_i2sb and train_synthesis:

    guide_layout(loader)   what the loader's guide planes ARE (labels, in the order the loader
                           emits them) and where they ride: as their own `guide` entry, at the end
                           of `cond` (guide_as_cond), or at the end of the synthesis input X.
                           None when the loader has no guides, so unguided runs log nothing new.
    guide_panel(...)       one row of tiles in [0, 1] for `vutils.make_grid`, and its caption.

Intensity planes use the run's display window, as val/example does. An enhancement-map guide
(the loader's guide_enhancement, (CT1 - T1)_+) is drawn on [0, (hi - lo) / 2]: on the full window
it would be nearly black.
"""

import torch
from torch.utils.data import Subset

CONTRAST_NAMES = {0: "FLAIR", 1: "T1", 2: "CT1", 3: "T2"}
_MODE_TAG = {"other_study": "prior", "far_slice": "far", "central_slice": "central",
             "same_slice": "same slice"}
ENHANCEMENT = -1          # datasets.NYUMets.longitudinal_dataset.ENHANCEMENT, without the import


def _unwrap(ds):
    """The NYUMetsGuidedDataset under any Subset (fixed_val_subset) or synthesis view."""
    synth = False
    for _ in range(6):
        if isinstance(ds, Subset):
            ds = ds.dataset
        elif not hasattr(ds, "modes") and hasattr(ds, "ds"):      # datasets.NYUMets _SynthView
            ds, synth = ds.ds, True
        else:
            break
    return ds, synth


def guide_layout(loader):
    """-> None, or dict(labels, enh, as_cond, n_cond, t1_pos) describing the loader's guides.

    labels   one per guide plane, in the loader's order (`_guide_planes`: mode, then guide, then
             slice offset -w..+w, then contrast)
    enh      per plane: True for an enhancement-map plane
    as_cond  True when the planes are the TAIL of cond (bridge, guide_as_cond) or of the
             synthesis input X; False when they come as the batch's own `guide` entry
    n_cond   number of ordinary conditioning channels before them
    t1_pos   position of the current T1 among those channels, or None
    """
    ds, synth = _unwrap(getattr(loader, "dataset", None))
    modes = getattr(ds, "modes", None)
    if not modes:
        return None
    labels, enh = [], []
    for mode in modes:
        if mode == "same_session":
            for c in ds.guide_contrasts:
                labels.append("%s (this study)" % CONTRAST_NAMES.get(c, c))
                enh.append(False)
            continue
        w = int(getattr(ds, "guide_window", 0) or 0) if mode == "other_study" else 0
        for g in range(int(getattr(ds, "n_guides", 1))):
            for d in range(-w, w + 1):
                for c in ds.guide_idx:
                    e = int(c) == ENHANCEMENT
                    name = "(CT1-T1)+" if e else CONTRAST_NAMES.get(int(c), str(c))
                    lab = "%s %s" % (_MODE_TAG.get(mode, mode), name)
                    if w:
                        lab += " z%+d" % d
                    if int(getattr(ds, "n_guides", 1)) > 1:
                        lab += " #%d" % g
                    labels.append(lab)
                    enh.append(e)
    n_planes = getattr(ds, "n_guide_planes", len(labels))
    if n_planes != len(labels):
        return None                                   # a layout this module does not know
    cond_idx = list(getattr(ds, "cond_idx", []) or [])
    x1 = getattr(ds, "x1_idx", 1)
    return dict(labels=labels, enh=enh, as_cond=bool(synth or getattr(ds, "guide_as_cond", False)),
                n_cond=len(cond_idx), t1_pos=cond_idx.index(x1) if x1 in cond_idx else None)


def sample_guides(layout, cond=None, guide=None, X=None):
    """The guide planes of sample 0 as (G, H, W), from wherever this loader puts them, or None
    when the count does not match the layout."""
    if layout is None:
        return None
    if X is not None:
        g = X[0, layout["n_cond"]:]
    elif layout["as_cond"]:
        g = None if cond is None else cond[0, layout["n_cond"]:]
    else:
        g = None if guide is None else guide[0]
        if g is not None and g.dim() == 4:            # (G, 1, H, W) -> (G, H, W)
            g = g.flatten(0, 1)
    if g is None or g.shape[0] != len(layout["labels"]):
        return None
    return g


def guide_panel(current, guides, layout, window, mask=None):
    """-> (tiles (N, 1, H, W) in [0, 1], caption).

    current  list of (label, image (H, W) or (1, H, W)) drawn first, e.g. T1 now and the CT1 target
    guides   (G, H, W) from `sample_guides`
    window   (lo, hi) of the run's display window
    mask     (H, W) or (1, H, W), multiplied in after scaling (or None)
    """
    lo, hi = float(window[0]), float(window[1])
    span = max(hi - lo, 1e-8)
    tiles, caps = [], []
    for lab, img in current:
        img = img.reshape(img.shape[-2:]).float()
        tiles.append(((img - lo) / span).clamp(0, 1))
        caps.append(lab)
    for lab, e, g in zip(layout["labels"], layout["enh"], guides):
        g = g.float()
        if e:
            tiles.append((g / (span / 2)).clamp(0, 1))
            caps.append(lab + " [0, %.2g]" % (span / 2))
        else:
            tiles.append(((g - lo) / span).clamp(0, 1))
            caps.append(lab)
    grid = torch.stack(tiles)[:, None]
    if mask is not None:
        grid = grid * mask.reshape(mask.shape[-2:]).to(grid.dtype)
    return grid, " | ".join(caps)
