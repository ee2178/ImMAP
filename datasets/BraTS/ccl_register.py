"""
Registry loader for CCL pretraining data.

Registers under "ccl" so a config data block like
    {"name": "ccl", "root": ".../BraTS2021_DataSet_train", "n_clusters": 20, ...}
is dispatched here by build_loader(cfg["data"][split], shuffle=..., drop_last=...).

Point `root` at the symlink train/val directories made by make_split_symlinks.py
(or pass `manifest` instead to use a manifest CSV). Place the dataset class at
datasets/ccl_dataset.py.
"""

from types import SimpleNamespace
from torch.utils.data import DataLoader

from datasets.registry import register_loader
from datasets.BraTS.ccl_dataset import CCLPretrainDataset


@register_loader("ccl")
def build_ccl_loader(root=None,
                     manifest=None,
                     n_clusters=None,
                     contrast_idx=(0, 1, 2, 3),
                     patch_size=4,
                     num_samples_loss_eval=100,
                     foreground_from_label=True,
                     batch_size=16,
                     num_workers=8,
                     pin_memory=True,
                     # overrides injected by build_loader in train.py:
                     shuffle=False,
                     drop_last=False,
                     **unused):
    """Build a DataLoader of (X, y_true) for CCL pretraining from a directory `root`
    (preferred) or a `manifest` CSV. patch_size MUST equal the constraint downsampling."""
    ds_cfg = SimpleNamespace(
        root=root,
        manifest=manifest,
        n_clusters=n_clusters,
        patch_size=patch_size,
        num_samples_loss_eval=num_samples_loss_eval,
        contrast_idx=list(contrast_idx),
        foreground_from_label=foreground_from_label,
    )
    dataset = CCLPretrainDataset(ds_cfg)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      pin_memory=pin_memory,
                      drop_last=drop_last)
