"""Registry loader for GroupCDL dictionary pretraining ("contrast_stack").

A config data block like
    {"name": "contrast_stack", "root": ".../BraTS2021_DataSet_train",
     "cond_idx": [0, 1, 3], "ch0": "bridge", ...}
is dispatched here by build_loader(cfg["data"][split], shuffle=..., drop_last=...).

Returns ONE tensor per item (train_denoiser's contract), not I2SBDataset's 4-tuple. Channel
width is 1 + len(cond_idx), which is what GroupCDL's C must be set to:
    step 1 (joint)  cond_idx: [0, 1, 3], ch0: "bridge"  ->  C = 4
    step 2 (T1ce)   cond_idx: [],        ch0: "x0"       ->  C = 1
"""

from types import SimpleNamespace
from torch.utils.data import DataLoader

from datasets.registry import register_loader
from datasets.BraTS.stack_dataset import ContrastStackDataset


@register_loader("contrast_stack")
def build_contrast_stack_loader(root=None,
                                manifest=None,
                                x0_idx=2,                 # T1ce  (stored: flair,t1,t1ce,t2)
                                x1_idx=1,                 # T1
                                cond_idx=(0, 1, 3),       # FLAIR, T1, T2  ([] -> 1-channel stack)
                                ch0="bridge",             # "bridge" | "x0" | "x1"
                                scales=None,              # per-stored-channel divisors; None = ones
                                image_key="img",          # "img" (zscore, signed) or "img_raw"
                                # bridge knobs, used only when ch0 == "bridge". No tau: it
                                # cancels out of the interpolant mean (see stack_dataset docs);
                                # set the noise level via the training block's noise_std.
                                bridge_type="brownian",
                                n_points=1000,
                                bridge_shape="constant",
                                beta_max=0.3,
                                tau=0.19,                 # absolute bridge-noise scale (bridge_sample only)
                                bridge_sample=False,      # True -> ch0 is a bridge SAMPLE, cond stays clean
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
        x0_idx=x0_idx, x1_idx=x1_idx, cond_idx=list(cond_idx), ch0=ch0,
        scales=scales, image_key=image_key,
        bridge_type=bridge_type, n_points=n_points, bridge_shape=bridge_shape,
        beta_max=beta_max, tau=tau, bridge_sample=bridge_sample,
        center_crop=center_crop, crop_size=crop_size, random_flips=random_flips,
    )
    dataset = ContrastStackDataset(ds_cfg)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
