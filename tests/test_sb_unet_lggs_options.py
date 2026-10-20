# -*- coding: utf-8 -*-
"""SBUnetLGGS: the measurement-noise injection and the UNet-only switch.

meas_noise   training only: LGGS gets u + sigma_n * eps and the level sqrt(sigma_t^2 + sigma_n^2);
             in eval mode the model is exactly the noise-free one with the same weights.
refine=False x0_hat is the UNet estimate itself, and nothing downstream of it runs.
"""
import sys
import types

import pytest
import torch

sys.modules.setdefault("nibabel", types.ModuleType("nibabel"))

import models.sb_unet_lggs as mod  # noqa: E402
from models.sb_unet_lggs import SBUnetLGGS  # noqa: E402

H = 32
UNET = dict(out_channels=1, model_channels=16, num_res_blocks=1, channel_mult=[1, 2],
            attention_resolutions=[], image_size=H, num_heads=1, num_head_channels=16,
            kind="brownian", tau=0.1, n_points=50, beta_max=0.3)
LGGS = dict(K=2, M=8, C=1, P=3, s=2, Mh=4, window=1, guide_window=3, joint_softmax=True,
            attn_backend="gather", dK=1, is_complex=False, preproc="image", degrees=1)


def net(**kw):
    torch.manual_seed(0)
    return SBUnetLGGS(C=4, n_guide_planes=3, lggs_guides=[0, 1, 2], unet=UNET, lggs=LGGS, **kw)


def inputs():
    g = torch.Generator().manual_seed(1)
    y = torch.rand(2, 4, H, H, generator=g)
    guide = torch.rand(2, 3, 1, H, H, generator=g)
    sigma = torch.tensor([0.05, 0.2]).view(2, 1, 1, 1)
    return y, sigma, guide


def test_unet_only_returns_u():
    m = net(refine=False).eval()
    y, sigma, guide = inputs()
    x, state = m(y, sigma=sigma, guide=guide)
    assert state is None and torch.equal(x, m.last_u)


def test_noise_is_training_only_and_eval_matches_the_noise_free_model():
    noisy, clean = net(meas_noise={"max": 0.1, "p_zero": 0.0}), net()
    # same seed and no extra parameters, so the two start identical (load_state_dict(assign=)
    # is torch 2.x only, and this runs on the local 1.12 too)
    assert all(torch.equal(a, b) for a, b in zip(noisy.state_dict().values(), clean.state_dict().values()))
    y, sigma, guide = inputs()
    noisy.eval(), clean.eval()
    with torch.no_grad():
        assert torch.equal(noisy(y, sigma=sigma, guide=guide)[0], clean(y, sigma=sigma, guide=guide)[0])
    noisy.train()
    with torch.no_grad():
        a = noisy(y, sigma=sigma, guide=guide)[0]
        b = noisy(y, sigma=sigma, guide=guide)[0]
    assert not torch.equal(a, b), "training-mode noise did not change the output"


def test_lggs_sees_the_noisy_measurement_and_the_combined_level(monkeypatch):
    m = net(meas_noise={"max": 0.1, "p_zero": 0.0}).train()
    seen = {}
    real = m.lggs.forward

    def spy(meas, guide=None, sigma=None, **kw):
        seen["meas"], seen["sigma"] = meas.detach(), sigma.detach()
        return real(meas, guide=guide, sigma=sigma, **kw)

    monkeypatch.setattr(m.lggs, "forward", spy)
    y, sigma, guide = inputs()
    m(y, sigma=sigma, guide=guide)
    dev = (seen["meas"] - m.last_u.detach()).flatten(1).std(dim=1)
    assert float(dev.max()) > 1e-3 and float(dev.max()) < 0.15
    s = seen["sigma"].flatten()
    assert torch.all(s >= sigma.flatten() - 1e-7) and torch.all(s <= (sigma.flatten() ** 2 + 0.01) ** 0.5 + 1e-6)


def test_bad_noise_settings_are_refused():
    with pytest.raises(ValueError, match="meas_noise"):
        net(meas_noise={"max": 0.0})
    with pytest.raises(ValueError, match="meas_noise"):
        net(meas_noise={"max": 0.1, "p_zero": 1.0})
