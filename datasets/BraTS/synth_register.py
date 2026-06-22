"""Registry loader for the contrast-synthesis task ("synthesis")."""

from types import SimpleNamespace
from torch.utils.data import DataLoader

from datasets.registry import register_loader
from datasets.BraTS.synth_dataset import SynthesisDataset


@register_loader("synthesis")
def build_synthesis_loader(root=None,
                           manifest=None,
                           input_idx=(1, 3, 0),     # T1, T2, FLAIR  (stored: flair,t1,t1ce,t2)
                           target_idx=(2,),         # T1ce
                           random_flips=False,
                           batch_size=16,
                           num_workers=8,
                           pin_memory=True,
                           shuffle=False,
                           drop_last=False,
                           **unused):
    ds_cfg = SimpleNamespace(root=root, manifest=manifest,
                             input_idx=list(input_idx), target_idx=list(target_idx),
                             random_flips=random_flips)
    dataset = SynthesisDataset(ds_cfg)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
