# -*- coding: utf-8 -*-
"""`nyumets_synth`: the NYUMets longitudinal data in the CONTRAST-SYNTHESIS batch layout.

`training.synthesis.train_synthesis` consumes (X, y, organ_mask, et) -- one input stack, one
target, a brain mask, an ET mask -- while NYUMetsGuidedDataset speaks the bridge layout
(x0, x1, cond, mask, guide). This is the adapter, so the end-to-end arm of the longitudinal
comparison runs the ORDINARY synthesis loop rather than a special case bolted into the bridge
trainer: same loss registry, same masking, same metrics, same backtracking, same panels.

    X = cat([cond contrasts of THIS session, the prior study's guide planes])
    y = this session's CT1
    organ_mask = the brain mask
    et = zeros -- NYUMets carries no enhancing-tumor segmentation

Channel order is `cond_idx` first, then the guides, which is what `guide_as_cond=True` produces in
the bridge layout too. So with the default cond_idx (FLAIR, T1, T2), `residual_src_idx: 1` is the
T1 anchor for residual mode, exactly as in the BraTS synthesis configs.

WHAT IS DELIBERATELY ABSENT is the bridge state x_t. An i2sb regressor takes
C = 1 + len(cond_idx) + n_guides because it also sees where it is on the bridge; a direct regressor
takes in_chans = len(cond_idx) + n_guides. That one channel IS the difference between the two arms,
so it is not padded away to make the numbers match.
"""
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, Dataset

from datasets.registry import register_loader
from datasets.NYUMets.longitudinal_dataset import NYUMetsGuidedDataset


class _SynthView(Dataset):
    """A NYUMetsGuidedDataset seen as (X, y, organ_mask, et).

    The wrapped dataset is built with guide_as_cond=False so the guides arrive as their own
    (G, 1, H, W) entry; they are flattened to (G, H, W) and appended to cond here. Letting the
    dataset fold them in itself would work equally well, but then a single-study patient and a
    guide-less config would take different code paths through `__getitem__`, and the flattening
    would be invisible.
    """

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        s = self.ds[i]
        if isinstance(s, dict):
            x0, cond, mask = s["x0"], s["cond"], s["mask"]
            guide = s.get("guide")
        else:                                   # (x0, x1, cond, mask): no guide to carry
            x0, _, cond, mask = s
            guide = None
        parts = [cond]
        if guide is not None:
            # (G, 1, H, W) -> (G, H, W). The singleton channel exists so models.guided_prox can
            # tell G planes of 1 channel from 1 plane of G channels; a plain stack does not care.
            parts.append(guide.flatten(0, 1) if guide.dim() == 4 else guide)
        X = torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]
        et = torch.zeros_like(mask)             # NYUMets has no ET segmentation
        return X, x0, mask, et


@register_loader("nyumets_synth")
def build_nyumets_synth_loader(root=None,
                               x0_idx=2,                 # CT1 target (stored: flair,t1,t1ce,t2)
                               x1_idx=1,                 # unused here; kept so one data block can
                                                         # be shared with the i2sb configs
                               cond_idx=(0, 1, 3),       # FLAIR, T1, T2 -> X channels 0..2
                               image_key="img_median_mad",
                               scales=None,
                               # --- guide (the prior study) ---
                               guide_mode="none",
                               guide_idx=None,
                               guide_window=0,
                               guide_contrasts=None,
                               n_guides=1,
                               min_slice_gap=5,
                               guide_slice="matched",
                               deterministic=False,
                               slice_range=None,
                               # --- mask (see the dataset docstring) ---
                               mask_source="h5",         # "h5" | "t1ct1" (support of T1 & CT1)
                               mask_apply=False,         # multiply the images by the mask
                               mask_guides="target",     # guides: "target" | "own" | "none"
                               mask_eps=0.0,             # nonzero means |value| > mask_eps
                               mask_idx=None,            # contrasts defining it; default [x1_idx, x0_idx]
                               # --- geometry ---
                               center_crop=None,
                               crop_size=None,
                               random_flips=False,
                               # --- loader ---
                               batch_size=16,
                               num_workers=8,
                               pin_memory=True,
                               shuffle=False,
                               drop_last=False,
                               **unused):
    ds_cfg = SimpleNamespace(
        root=root,
        # x1 is irrelevant to a direct regressor, but the dataset requires a valid index and
        # returns it; the adapter drops it.
        x0_idx=x0_idx, x1_idx=x1_idx, cond_idx=list(cond_idx),
        image_key=image_key, scales=scales,
        guide_mode=guide_mode,
        guide_idx=x0_idx if guide_idx is None else guide_idx,
        guide_contrasts=guide_contrasts,
        n_guides=n_guides, min_slice_gap=min_slice_gap, guide_slice=guide_slice,
        guide_window=guide_window,
        deterministic=deterministic,
        guide_as_cond=False,                   # the adapter concatenates; see _SynthView
        center_crop=center_crop, crop_size=crop_size, random_flips=random_flips,
        slice_range=slice_range,
        mask_source=mask_source, mask_apply=mask_apply, mask_guides=mask_guides,
        mask_eps=mask_eps, mask_idx=mask_idx,
        x1_source="contrast", x1_other_idx=None, y_idx=None,
    )
    dataset = _SynthView(NYUMetsGuidedDataset(ds_cfg))
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      pin_memory=pin_memory,
                      drop_last=drop_last)


def n_input_channels(cond_idx=(0, 1, 3), guide_mode="none", guide_idx=None, guide_window=0,
                     n_guides=1, **unused):
    """The `in_chans` a model needs for a given data block -- so a config cannot disagree with it.

    Mirrors NYUMetsGuidedDataset.n_guide_planes for the other_study case, which is the only guide
    mode the longitudinal experiments use.
    """
    modes = [guide_mode] if isinstance(guide_mode, str) else list(guide_mode)
    n = len(list(cond_idx))
    if "other_study" in modes:
        n += len(list(guide_idx or [])) * (2 * int(guide_window) + 1) * int(n_guides)
    return n
