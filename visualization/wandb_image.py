"""
`wandb_image` -- build a `wandb.Image` from a float tensor/array without it logging black.

WHY THIS EXISTS. Until wandb 0.29.0 (2026-08-26), `wandb.Image` rescaled float pixel data itself:
anything with a negative value was min-max normalised, and anything with max <= 1 was multiplied by
255. 0.29.0 dropped that ("wandb.Image no longer normalizes pixel values. Callers must now ensure
their data already falls within the range [0, 255]") and now casts straight to uint8. Every panel
this repo logs is a float in [0, 1] -- so each one became 0 (or 1/255) and rendered black: val
examples, residuals, and the dictionary filter grids alike.

`wandb_image` restores the old conversion here, so the result is the same on every wandb version:

    min < 0        -> (x - min) / (max - min)            (old behaviour)
    max <= 1       -> x * 255                            (old behaviour)
    then clip to [0, 255] and cast to uint8

with one deliberate change: NaN / inf are replaced by 0 BEFORE the min/max, and reported. The old
code used np.min / np.max, so a single NaN pixel made both NaN, skipped the scaling, and blacked out
the WHOLE panel -- the same symptom as the version change, for a different reason.

Accepts torch tensors or numpy arrays shaped (H, W), (C, H, W) [torch layout] or (H, W, C); anything
else (a matplotlib figure, a path, a PIL image) is passed to wandb.Image unchanged.
"""

import numpy as np
import torch


def to_uint8_image(data):
    """-> (H, W) or (H, W, C) uint8 numpy array, with the pre-0.29 wandb scaling."""
    if isinstance(data, torch.Tensor):
        data = data.detach().float().cpu()
        if data.dim() == 3 and data.shape[0] in (1, 3, 4):   # CHW -> HWC, as wandb did for tensors
            data = data.permute(1, 2, 0)
        data = data.numpy()
    data = np.asarray(data)
    if data.dtype == np.uint8:
        return np.squeeze(data)
    data = data.astype(np.float64)

    bad = ~np.isfinite(data)
    if bad.any():
        print(f"[wandb_image] {int(bad.sum())} non-finite pixel(s) set to 0 before logging")
        data = np.where(bad, 0.0, data)

    if data.size and data.min() < 0:
        rng = data.max() - data.min()
        data = (data - data.min()) / (rng if rng > 0 else 1.0)
    if data.size and data.max() <= 1.0:
        data = data * 255.0
    return np.squeeze(np.clip(data, 0, 255).astype(np.uint8))


def wandb_image(data, **kwargs):
    """wandb.Image(data, **kwargs), with float tensors/arrays converted by `to_uint8_image`."""
    import wandb

    if isinstance(data, (torch.Tensor, np.ndarray)):
        data = to_uint8_image(data)
    return wandb.Image(data, **kwargs)
