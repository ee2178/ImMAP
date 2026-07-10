"""Registry loader for the I2SB Schrodinger-bridge task ("i2sb").

A config data block like
    {"name": "i2sb", "root": ".../BraTS2021_DataSet_train",
     "x0_idx": 2, "x1_idx": 1, "cond_idx": [0, 1, 3], "scales": [1, 1, 1, 1], ...}
is dispatched here by build_loader(cfg["data"][split], shuffle=..., drop_last=...).

Conditioning toggle: cond_idx=[] disables conditioning (and CDLNet must then be built with
C=1); cond_idx=[0,1,3] enables it (CDLNet C = 1 + len(cond_idx) = 4).
"""

from types import SimpleNamespace
from torch.utils.data import DataLoader

from datasets.registry import register_loader
from datasets.BraTS.i2sb_dataset import I2SBDataset


@register_loader("i2sb")
def build_i2sb_loader(root=None,
                      manifest=None,
                      x0_idx=2,                 # T1ce  (stored: flair,t1,t1ce,t2)
                      x1_idx=1,                 # T1
                      cond_idx=(0, 1, 3),       # FLAIR, T1, T2  ([] to disable conditioning)
                      scales=None,              # per-stored-channel multipliers; None = ones
                      image_key="img",          # "img" (normalized) or "img_raw" (unnormalized)
                      center_crop=None,
                      crop_size=None,
                      random_flips=False,
                      batch_size=16,
                      num_workers=8,
                      pin_memory=True,
                      # overrides injected by build_loader in train.py:
                      shuffle=False,
                      drop_last=False,
                      **unused):
    ds_cfg = SimpleNamespace(
        root=root, manifest=manifest,
        x0_idx=x0_idx, x1_idx=x1_idx, cond_idx=list(cond_idx),
        scales=scales, image_key=image_key,
        center_crop=center_crop, crop_size=crop_size, random_flips=random_flips,
    )
    dataset = I2SBDataset(ds_cfg)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
