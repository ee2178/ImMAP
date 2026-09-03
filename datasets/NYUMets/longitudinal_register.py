"""
Registry loader for the NYUMets guided-I2SB dataset.

Registers under "nyumets_guided", so a config data block like

    {"name": "nyumets_guided", "root": "../datasets/NYUMets_h5_train",
     "guide_mode": "other_study", "image_key": "img_median_mad", ...}

is dispatched here by build_loader(cfg["data"][split], ...).

Both NYUMets dataset types come from this one class:
  * plain contrast synthesis -> guide_mode "none"  (returns the same 4-tuple as the
    BraTS "i2sb" loader, which also still works against these h5 files)
  * guided synthesis         -> any other guide_mode (appends a 5th item, `guide`)

Every keyword the dataset understands is named explicitly below. Adding a dataset knob
without adding it here silently drops it: **unused swallows the leftovers.
"""

from types import SimpleNamespace
from torch.utils.data import DataLoader

from datasets.registry import register_loader
from datasets.NYUMets.longitudinal_dataset import NYUMetsGuidedDataset


@register_loader("nyumets_guided")
def build_nyumets_guided_loader(root=None,
                                x0_idx=2,                 # CT1  (stored: flair,t1,t1ce,t2)
                                x1_idx=1,                 # T1
                                cond_idx=(0, 1, 3),       # FLAIR, T1, T2
                                image_key="img_median_mad",
                                scales=None,
                                # --- guide ---
                                guide_mode="none",        # see GUIDE_MODES
                                guide_idx=None,           # default: same contrast as x0
                                n_guides=1,
                                min_slice_gap=5,          # far_slice: adjacent is too easy
                                guide_slice="central",    # other_study: "central" | "random"
                                deterministic=False,      # set True for val/test
                                # --- geometry ---
                                center_crop=None,
                                crop_size=None,
                                random_flips=False,
                                # --- loader ---
                                batch_size=16,
                                num_workers=8,
                                pin_memory=True,
                                # overrides injected by build_loader in train.py:
                                shuffle=False,
                                drop_last=False,
                                **unused):
    ds_cfg = SimpleNamespace(
        root=root,
        x0_idx=x0_idx, x1_idx=x1_idx, cond_idx=list(cond_idx),
        image_key=image_key, scales=scales,
        guide_mode=guide_mode,
        guide_idx=x0_idx if guide_idx is None else guide_idx,
        n_guides=n_guides, min_slice_gap=min_slice_gap, guide_slice=guide_slice,
        deterministic=deterministic,
        center_crop=center_crop, crop_size=crop_size, random_flips=random_flips,
    )
    dataset = NYUMetsGuidedDataset(ds_cfg)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      pin_memory=pin_memory,
                      drop_last=drop_last)
