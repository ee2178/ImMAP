"""
Registry loader for the NYUMets guided-I2SB dataset.

Registers under "nyumets_guided", so a config data block like

    {"name": "nyumets_guided", "root": "../datasets/NYUMets_h5_train",
     "guide_mode": "other_study", "image_key": "img_median_mad", ...}

is dispatched here by build_loader(cfg["data"][split], ...).

Both NYUMets dataset types come from this one class:
  * plain contrast synthesis -> guide_mode "none"  (returns the same 4-tuple as the
    BraTS "i2sb" loader, which also still works against these h5 files)
  * guided synthesis         -> any other guide_mode (returns a DICT with a `guide` key;
                                NOT a 5-tuple, which train_i2sb would read as the ET mask)

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
                                guide_mode="none",        # str or list; see GUIDE_MODES
                                guide_idx=None,           # other_study/*_slice contrast(s): int or list
                                                          # (e.g. [1, 2] = T1/CT1 pair); default x0
                                guide_window=0,           # other_study: +-k neighbouring guide slices
                                guide_contrasts=None,     # same_session contrasts; default T1,T2,FLAIR
                                n_guides=1,               # planes per non-session mode
                                min_slice_gap=5,          # far_slice: adjacent is too easy
                                guide_slice="matched",    # other_study: matched | index | central | random
                                deterministic=False,      # set True for val/test
                                guide_as_cond=False,      # append guides to cond (plain nets)
                                slice_range=None,         # [lo, hi) of ORIGINAL slice indices to keep
                                x1_source="contrast",     # "contrast" (x1_idx) | "other_study"
                                x1_other_idx=None,        # other_study: contrast of the start; default x0
                                y_idx=None,               # measurement returned as "y"; default x1_idx
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
        guide_contrasts=guide_contrasts,
        n_guides=n_guides, min_slice_gap=min_slice_gap, guide_slice=guide_slice,
        guide_window=guide_window,
        deterministic=deterministic, guide_as_cond=guide_as_cond,
        center_crop=center_crop, crop_size=crop_size, random_flips=random_flips,
        slice_range=slice_range,
        x1_source=x1_source, x1_other_idx=x1_other_idx, y_idx=y_idx,
    )
    dataset = NYUMetsGuidedDataset(ds_cfg)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      pin_memory=pin_memory,
                      drop_last=drop_last)
