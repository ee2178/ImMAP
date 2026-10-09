# -*- coding: utf-8 -*-
"""The synthesis trainer's validation panel must never log black.

It used to normalise input | target | prediction by the input and target alone and not clamp.
A prediction that overshot them left values above 1 in the grid, and
`visualization.wandb_image.to_uint8_image` reads a float image whose maximum exceeds 1 as
already being on a 0..255 scale -- so one overshooting pixel turned the whole panel into 0s and
1s out of 255. `display_scale` clamps, and takes the fixed window the bridge trainer uses.
"""
import sys
import types

import pytest
import torch

sys.modules.setdefault("nibabel", types.ModuleType("nibabel"))

from training.synthesis import display_scale                  # noqa: E402
from visualization.wandb_image import to_uint8_image          # noqa: E402


def panels(overshoot=4.0):
    torch.manual_seed(0)
    inp, gt = torch.rand(1, 1, 16, 16) * 1.2, torch.rand(1, 1, 16, 16) * 1.5
    pred = gt.clone()
    pred[0, 0, 3, 3] = overshoot                              # ONE pixel above the input/target range
    return torch.cat([inp, gt, pred], dim=0)


def test_the_old_scaling_logged_black():
    """The failure being fixed, reproduced -- so the test below is known to be about it."""
    g = panels()
    old = g - g[0:2].min()
    old = old / old[0:2].max().clamp(min=1e-8)
    assert float(old.max()) > 1.0
    assert to_uint8_image(old[:, 0].reshape(48, 16)).mean() < 1.0


@pytest.mark.parametrize("window", [None, [0.0, 2.0]])
@pytest.mark.parametrize("overshoot", [4.0, 1.0e6, -3.0])
def test_display_scale_is_clamped_and_visible(window, overshoot):
    g = display_scale(panels(overshoot), n_ref=2, window=window)
    assert float(g.min()) >= 0.0 and float(g.max()) <= 1.0
    u8 = to_uint8_image(g[:, 0].reshape(48, 16))
    assert u8.max() > 100 and u8.mean() > 30, "the panel would log (near) black"


def test_fixed_window_does_not_depend_on_the_sample():
    """With a window, the same intensity maps to the same grey in every panel and every run."""
    a = display_scale(torch.full((3, 1, 4, 4), 0.5), window=[0.0, 2.0])
    b = display_scale(torch.cat([torch.full((2, 1, 4, 4), 0.5), torch.full((1, 1, 4, 4), 9.0)]),
                      window=[0.0, 2.0])
    assert float(a[0, 0, 0, 0]) == float(b[0, 0, 0, 0]) == 0.25
    assert float(b[2].min()) == 1.0                           # out of window: clamped, not rescaled


def test_constant_panel_does_not_divide_by_zero():
    g = display_scale(torch.zeros(3, 1, 4, 4))
    assert torch.isfinite(g).all() and float(g.max()) == 0.0
