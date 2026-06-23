# -*- coding: utf-8 -*-
"""
Small factories for swappable normalization and activation layers.

Used by the configurable U-Net (models/cclnet.py) so the CCL / synthesis
architecture can be tuned from the experiment config instead of hard-coding a
norm/activation. String names map to torch modules:

    norm: "batch" | "group" | "instance" | "layer" | "none"
    act : "relu"  | "leaky_relu" | "tanh" | "gelu" | "elu" | "silu" | "none"

GroupNorm follows the TF reference convention of a fixed number of channels per
group (`group_size`, default 4 -> num_groups = channels // 4). Pass `num_groups`
to override with an explicit group count.
"""

import torch.nn as nn


def build_norm(norm, num_channels, group_size=4, num_groups=None, affine=True):
    """Return a normalization module for `num_channels` channels (2D feature maps)."""
    if norm is None:
        return nn.Identity()
    name = str(norm).lower()

    if name in ("none", "identity"):
        return nn.Identity()
    if name in ("batch", "bn", "batchnorm"):
        return nn.BatchNorm2d(num_channels, affine=affine)
    if name in ("instance", "in", "instancenorm"):
        return nn.InstanceNorm2d(num_channels, affine=affine)
    if name in ("layer", "ln", "layernorm"):
        # LayerNorm over the channel dim == GroupNorm with a single group.
        return nn.GroupNorm(1, num_channels, affine=affine)
    if name in ("group", "gn", "groupnorm"):
        g = num_groups if num_groups is not None else max(1, num_channels // group_size)
        # GroupNorm requires channels % groups == 0; back off to the nearest divisor.
        while num_channels % g != 0 and g > 1:
            g -= 1
        return nn.GroupNorm(g, num_channels, affine=affine)

    raise ValueError(f"Unknown norm '{norm}'")


def build_activation(act, inplace=True):
    """Return an activation module. Parameter-free activations keep state_dicts
    identical across choices, so swapping `act` never breaks weight transfer."""
    if act is None:
        return nn.Identity()
    name = str(act).lower()

    if name in ("none", "identity", "linear"):
        return nn.Identity()
    if name == "relu":
        return nn.ReLU(inplace=inplace)
    if name in ("leaky_relu", "lrelu", "leakyrelu"):
        return nn.LeakyReLU(0.2, inplace=inplace)
    if name == "tanh":
        return nn.Tanh()
    if name == "gelu":
        return nn.GELU()
    if name == "elu":
        return nn.ELU(inplace=inplace)
    if name in ("silu", "swish"):
        return nn.SiLU(inplace=inplace)
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "prelu":
        return nn.PReLU()

    raise ValueError(f"Unknown activation '{act}'")
