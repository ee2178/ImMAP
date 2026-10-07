# -*- coding: utf-8 -*-
"""Unpaired Neural Schrodinger Bridge (sb/unsb.py + models/unsb_nets.py).

Run with `python -m tests.test_unsb` or pytest.

What is pinned, and why each is the kind of thing that fails silently:

  * the UNSB bridge step IS the repo's existing `reverse_sample("ddpm")` posterior, and on the
    Brownian schedule it IS the paper's Eq. 11 with tau_paper = std_fwd[-1]**2. If either drifts,
    the sampler quietly becomes a different bridge than the one the net was trained on.
  * `unsb_sample` reproduces `reverse_sample`'s predictions step for step (deterministic AND with
    the same noise stream), and `unsb_rollout` reproduces the states the sampler visits -- the
    training chain and the inference chain must be the same chain.
  * gradient routing: d_loss never reaches G, g_loss never reaches D/E, the NCE key is detached.
  * a real SBUnet end to end, with PatchNCE reading its `input_blocks`.
  * architecture-agnosticism: SBCDLNet and SBGuidedGroupCDL (with a guide) run the same chain with
    the G-independent PatchEncoder, and the discriminator's conditioning is ONE toggle.

Runs on local torch 1.12: `models/__init__` needs torch.nn.attention (flex attention), which that
build lacks, so a stub is installed first when the import fails. SBUnet itself needs nothing newer.
"""
import math
import os
import sys
import types

import numpy as np
import torch
import torch.nn as nn

try:                                                    # local torch 1.12 has no flex attention
    import torch.nn.attention.flex_attention  # noqa: F401
except ImportError:
    _m = types.ModuleType("torch.nn.attention")
    _m.__path__ = []
    _f = types.ModuleType("torch.nn.attention.flex_attention")
    _f.flex_attention = _f.create_block_mask = lambda *a, **k: None
    _m.flex_attention = _f
    sys.modules["torch.nn.attention"] = _m
    sys.modules["torch.nn.attention.flex_attention"] = _f

from sb.base import brownian, space_indices, reverse_sample, forward_sample     # noqa: E402
from sb.unsb import (unsb_steps, bridge_step_coeffs, make_g_fn, unsb_rollout,     # noqa: E402
                     unsb_forward, unsb_sample, tau_sb, remaining_fraction,
                     set_requires_grad, d_loss, e_loss, g_loss, mine_et, make_nce_fn)
from models.unsb_nets import (PatchDiscriminator, FeatureTap, PatchNCE,          # noqa: E402
                              PatchEncoder, build_critics)

TAU, N, K = 0.05, 1000, 5
B, H = 3, 32


class Skipped(Exception):
    pass


def _skip(msg):
    if "PYTEST_CURRENT_TEST" in os.environ:                 # under pytest: a real skip
        import pytest
        pytest.skip(msg)
    raise Skipped(msg)                                      # under `python -m tests.test_unsb`


def sched():
    return brownian(TAU, N)


class Stub(nn.Module):
    """A deterministic regressor with the repo's (y, E, sigma) signature and a few parameters."""

    def __init__(self, c_in=3):
        super().__init__()
        torch.manual_seed(0)
        self.body = nn.Conv2d(c_in, 4, 3, padding=1)
        self.head = nn.Conv2d(4, 1, 1)

    def forward(self, y, E=None, sigma=None):
        s = sigma.reshape(-1, 1, 1, 1)
        return self.head(torch.relu(self.body(y)) * (1.0 + s)), None


def batch(seed=0, c_cond=2):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(B, 1, H, H, generator=g), torch.randn(B, 1, H, H, generator=g),
            torch.randn(B, c_cond, H, H, generator=g))


# ---------------------------------------------------------------------------------------------
# the clock and the step
# ---------------------------------------------------------------------------------------------
def test_uniform_grid_is_reverse_samples_checkpoints():
    assert unsb_steps(sched(), K) == space_indices(N, K + 1)


def test_official_grid_lands_on_the_paper_code_clock():
    steps = unsb_steps(sched(), K, grid="official")
    t_eff = (np.array(steps) + 1) / N
    want_t = 1.0 - np.array([1.0, 0.94, 0.86, 0.74, 0.5, 0.0])          # target end -> source end
    # 5 stages: s = [0, .5, .74, .86, .94] (+ the unevaluated target end s = 1)
    assert np.allclose(t_eff, want_t, atol=1.5 / N + 0.005), (t_eff, want_t)
    assert steps[0] == 0 and steps[-1] == N - 1


def test_step_is_paper_eq11_with_tau_paper_equal_total_variance():
    sc = sched()
    t_eff = (np.arange(N) + 1) / N
    assert abs(tau_sb(sc) - (2 * TAU) ** 2) < 1e-9                       # 0.01 for tau = 0.05
    rng = np.random.RandomState(0)
    for _ in range(20):
        n, n_prev = sorted(rng.choice(np.arange(1, N), 2, replace=False))[::-1]
        s_cur, s_next = 1 - t_eff[n], 1 - t_eff[n_prev]                    # the paper's clock
        a_paper = (s_next - s_cur) / (1 - s_cur)
        var_paper = a_paper * (1 - a_paper) * (1 - s_cur) * tau_sb(sc)
        a, b, std = bridge_step_coeffs(sc, int(n), int(n_prev))
        assert abs(float(a) - a_paper) < 1e-5 and abs(float(b) - (1 - a_paper)) < 1e-5
        assert abs(float(std) ** 2 - var_paper) < 1e-7


# ---------------------------------------------------------------------------------------------
# the chain
# ---------------------------------------------------------------------------------------------
def _reverse(net, x1, cond, deterministic, seed):
    sc = sched()
    g_fn = make_g_fn(net, sc, x1, cond=cond)
    torch.manual_seed(seed)
    return reverse_sample(sc, g_fn, x1, nfe=K, deterministic=deterministic, log_count=K,
                          verbose=False)


def test_sampler_reproduces_reverse_sample_predictions():
    net, (_, x1, cond) = Stub().eval(), batch()
    for deterministic in (True, False):
        _, _, ref = _reverse(net, x1, cond, deterministic, seed=7)
        torch.manual_seed(7)
        recon, _, hats = unsb_sample(net, x1, sched(), cond=cond, num_stages=K,
                                     deterministic=deterministic)
        assert torch.allclose(hats, ref, atol=1e-6), f"deterministic={deterministic}"
        assert torch.equal(recon, hats[:, 0].to(recon.device))            # output = LAST prediction


def test_rollout_visits_the_states_the_sampler_visits():
    net, (_, x1, cond) = Stub().eval(), batch()
    sc = sched()
    torch.manual_seed(3)
    _, xs, _ = unsb_sample(net, x1, sc, cond=cond, num_stages=K)           # xs: newest-first
    steps = unsb_steps(sc, K)
    g_fn = make_g_fn(net, sc, x1, cond=cond)
    for j in range(K):
        torch.manual_seed(3)
        x_t, n = unsb_rollout(sc, g_fn, x1, steps, j)
        assert n == steps[K - j]
        assert torch.allclose(x_t, xs[:, K - 1 - j], atol=1e-6), f"stage {j}"
    assert torch.equal(xs[:, -1], x1)                                      # stage 0 enters at x1


def test_nfe_subset_is_an_even_subset_of_the_trained_stages():
    net, (_, x1, cond) = Stub().eval(), batch()
    _, xs, hats = unsb_sample(net, x1, sched(), cond=cond, num_stages=K, nfe=2)
    assert xs.shape[1] == hats.shape[1] == 2 and torch.equal(xs[:, -1], x1)
    for bad in (0, K + 1):
        try:
            unsb_sample(net, x1, sched(), cond=cond, num_stages=K, nfe=bad)
            raise AssertionError("nfe out of range was accepted")
        except ValueError:
            pass


# ---------------------------------------------------------------------------------------------
# losses and gradient routing
# ---------------------------------------------------------------------------------------------
def _nets(n_cond=2, cond_d=True):
    torch.manual_seed(0)
    D = PatchDiscriminator(1 + (n_cond if cond_d else 0), ndf=8)
    E = PatchDiscriminator(2, ndf=8, pair=True)
    return D, E


def _state(net, x1, cond, stage=2):
    sc = sched()
    return unsb_forward(make_g_fn(net, sc, x1, cond=cond), sc, x1, unsb_steps(sc, K), stage), sc


def _has_grad(mod):
    return any(p.grad is not None and p.grad.abs().sum() > 0 for p in mod.parameters())


def test_d_loss_trains_D_only_and_e_loss_trains_E_only():
    net, (x0, x1, cond) = Stub(), batch()
    D, E = _nets()
    state, _ = _state(net, x1, cond)
    assert state.x_hat.requires_grad and not state.x_t.requires_grad
    d_loss(D, state, x0, cond=cond).backward()
    assert _has_grad(D) and not _has_grad(net)
    e_loss(E, state).backward()
    assert _has_grad(E) and not _has_grad(net)


def test_g_loss_trains_G_only_and_reports_its_terms():
    net, (x0, x1, cond) = Stub(), batch()
    D, E = _nets()
    state, sc = _state(net, x1, cond)
    set_requires_grad([D, E], False)
    total, terms = g_loss(D, E, state, sc, x1, cond=cond)
    total.backward()
    assert _has_grad(net) and not _has_grad(D) and not _has_grad(E)
    assert set(terms) == {"adv", "transport", "entropy", "sb", "nce"}
    assert all(torch.isfinite(v) for v in terms.values())
    assert float(terms["nce"]) == 0.0                                      # no nce fn given


def test_unconditional_D_and_no_entropy_paths():
    net, (x0, x1, cond) = Stub(), batch()
    D, _ = _nets(cond_d=False)
    state, sc = _state(net, x1, cond)
    assert torch.isfinite(d_loss(D, state, x0))
    total, terms = g_loss(D, None, state, sc, x1)                          # E = None, D unconditional
    total.backward()
    assert float(terms["entropy"]) == 0.0 and _has_grad(net)


def test_entropy_critic_needs_a_batch():
    net, (x0, x1, cond) = Stub(), batch()
    _, E = _nets()
    state, sc = _state(net, x1[:1], cond[:1])
    try:
        e_loss(E, state)
        raise AssertionError("batch of 1 was accepted")
    except ValueError:
        pass


def test_mine_et_is_joint_minus_log_mean_exp():
    a, b = torch.randn(4, 2, 8, 8), torch.randn(4, 2, 8, 8)
    E = lambda x, step, y: (x * y).mean(1, keepdim=True)                   # a fixed critic
    et, log_z, joint = mine_et(E, a, b, None)
    marg = E(a, None, b).flatten()
    assert abs(float(joint) - float(E(a, None, a).mean())) < 1e-6
    assert abs(float(log_z) - float(torch.log(torch.exp(marg).mean()))) < 1e-5
    assert abs(float(et) - (float(joint) - float(log_z))) < 1e-6


def test_remaining_fraction_is_t_on_the_brownian_schedule():
    sc = sched()
    for k in (0, 10, 500, N - 1):
        assert abs(remaining_fraction(sc, k) - (k + 1) / N) < 1e-6


# ---------------------------------------------------------------------------------------------
# PatchNCE and the tap
# ---------------------------------------------------------------------------------------------
class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Conv2d(3, 6, 3, padding=1)
        self.b = nn.Conv2d(6, 8, 3, padding=1)
        self.c = nn.Conv2d(8, 1, 3, padding=1)
        self.calls = 0

    def forward(self, y, E=None, sigma=None):
        self.calls += 1
        return self.c(torch.relu(self.b(torch.relu(self.a(y))))), None


def test_tap_is_silent_outside_its_own_calls_and_stops_early_inside_them():
    net, (_, x1, cond) = Tiny(), batch()
    tap = FeatureTap(net, ["a", "b"])
    sigma = torch.zeros(B, 1, 1, 1)
    net(torch.cat([x1, cond], 1), sigma=sigma)           # an ordinary forward: no exception, no buffer
    assert tap._buf == {}
    net.calls = 0
    f_early = tap(x1, cond, sigma)
    assert net.calls == 1 and tap._buf.keys() == {"a", "b"}
    assert [f.shape[1] for f in f_early] == [6, 8] == tap.channels(x1, cond, sigma)
    ref = net.b(torch.relu(net.a(torch.cat([x1, cond], 1))))
    assert torch.allclose(f_early[1], ref)
    try:
        FeatureTap(net, ["nope"])
        raise AssertionError("an unknown layer name was accepted")
    except ValueError:
        pass


def test_patchnce_pulls_the_query_toward_the_same_location_and_detaches_the_key():
    torch.manual_seed(0)
    k = torch.randn(2, 8, 12, 12)
    pnce = PatchNCE([8], nc=16, num_patches=64)
    same = pnce([k.clone().requires_grad_()], [k])
    shifted = pnce([torch.roll(k, 5, dims=3).clone().requires_grad_()], [k])    # anatomy moved
    assert float(same) < float(shifted) - 0.5, (float(same), float(shifted))

    q = k.clone().requires_grad_()
    kk = k.clone().requires_grad_()
    pnce([q], [kk]).backward()
    assert q.grad is not None and q.grad.abs().sum() > 0
    assert kk.grad is None                                                 # the key is detached


def test_patchnce_mask_confines_the_samples_to_the_brain():
    mask = torch.zeros(2, 1, 12, 12)
    mask[:, :, 3:9, 3:9] = 1
    ids = PatchNCE._ids(2, 12, 12, 30, mask, torch.device("cpu"))          # 36 in-mask locations
    rows, cols = torch.div(ids, 12, rounding_mode="floor"), ids % 12
    assert bool(((rows >= 3) & (rows < 9) & (cols >= 3) & (cols < 9)).all())


# ---------------------------------------------------------------------------------------------
# the real regressor, end to end
# ---------------------------------------------------------------------------------------------
def test_sbunet_end_to_end_with_patchnce_on_its_encoder():
    from models.sb_unet import SBUnet
    torch.manual_seed(0)
    sc = sched()
    net = SBUnet(C=3, model_channels=32, num_res_blocks=1, channel_mult=(1, 2),
                 attention_resolutions=(8,), image_size=16, num_head_channels=16,
                 kind="brownian", tau=TAU, n_points=N)
    net.assert_schedule_matches(sc)
    x0, x1, cond = batch()
    tap = FeatureTap(net, ["input_blocks.1", "input_blocks.3"])
    sigma = torch.zeros(B, 1, 1, 1)
    chans = tap.channels(x1, cond, sigma)
    assert chans == [32, 64], chans
    pnce = PatchNCE(chans, nc=32, num_patches=64)
    D, E = _nets()

    g_fn = make_g_fn(net, sc, x1, cond=cond)
    state = unsb_forward(g_fn, sc, x1, unsb_steps(sc, K), stage=3)
    d_loss(D, state, x0, cond=cond).backward()
    e_loss(E, state).backward()
    D.zero_grad(set_to_none=True)
    E.zero_grad(set_to_none=True)
    net.zero_grad(set_to_none=True)
    set_requires_grad([D, E], False)
    nce = make_nce_fn(tap, pnce, sc, cond=cond)
    total, terms = g_loss(D, E, state, sc, x1, cond=cond, nce=nce)
    total.backward()
    assert math.isfinite(float(total)) and float(terms["nce"]) > 0
    assert _has_grad(net) and _has_grad(pnce) and not _has_grad(D)

    recon, xs, hats = unsb_sample(net.eval(), x1, sc, cond=cond, num_stages=K)
    assert recon.shape == x1.shape and hats.shape == (B, K, 1, H, H)


# ---------------------------------------------------------------------------------------------
# architecture-agnostic G, and the discriminator
# ---------------------------------------------------------------------------------------------
def test_chain_state_carries_the_source_with_the_bridge_weight():
    """Why the SB-CDL nets (which subtract mu_1 * x_1) stay correct under UNSB: on the Brownian
    schedule the state is w * x_1 + (1 - w) * (mix of predictions), w = t = (n+1)/N, which is the
    bridge mean forward_sample builds up to the schedule's O(1/N) first-step convention."""
    sc, steps = sched(), unsb_steps(sched(), K)
    _, x1, _ = batch()
    zero = lambda x, n: torch.zeros_like(x)                                # every prediction = 0
    for j in range(K):
        x_t, n = unsb_rollout(sc, zero, x1, steps, j, noise_scale=0.0)
        assert torch.allclose(x_t, (n + 1) / N * x1, atol=1e-5), f"stage {j}"
        ref = forward_sample(sc, torch.full((B,), n, dtype=torch.long), torch.zeros_like(x1), x1,
                             deterministic=True)
        assert float((x_t - ref).abs().max()) < 2.5 / N * float(x1.abs().max())


def test_cond_toggle_is_one_switch_for_D_and_its_losses():
    net, (x0, x1, cond) = Stub(), batch()
    state, sc = _state(net, x1, cond)
    for cond_d, width in ((True, 3), (False, 1)):
        D, E = build_critics(target_channels=1, n_cond=2, cond_d=cond_d, ndf=8)
        assert D.in_channels == width and E.pair and E.in_channels == 2
        # the SAME call, cond always passed: the toggle alone decides whether D uses it
        assert torch.isfinite(d_loss(D, state, x0, cond=cond, cond_d=cond_d))
        total, _ = g_loss(D, E, state, sc, x1, cond=cond, cond_d=cond_d)
        assert torch.isfinite(total)
    D_cond, _ = build_critics(1, 2, cond_d=True, ndf=8)
    try:                                                                   # toggle disagrees with D
        d_loss(D_cond, state, x0, cond=cond, cond_d=False)
        raise AssertionError("a conditional D accepted an unconditional call")
    except ValueError as e:
        assert "build_critics" in str(e)
    assert build_critics(1, 2, entropy=False)[1] is None


def test_gan_only_configuration_is_just_the_adversarial_term():
    net, (x0, x1, cond) = Stub(), batch()
    D, _ = build_critics(1, 2, entropy=False, ndf=8)
    state, sc = _state(net, x1, cond)
    set_requires_grad(D, False)
    total, terms = g_loss(D, None, state, sc, x1, cond=cond, lambda_sb=0.0)
    assert torch.equal(total, terms["adv"]) and "entropy" not in terms
    total.backward()
    assert _has_grad(net) and not _has_grad(D)


def test_spectral_norm_discriminator_trains():
    net, (x0, x1, cond) = Stub(), batch()
    D, _ = build_critics(1, 2, ndf=8, spectral_norm=True, entropy=False)
    state, _ = _state(net, x1, cond)
    d_loss(D, state, x0, cond=cond).backward()
    assert _has_grad(D)


def test_patch_encoder_is_a_g_independent_extractor():
    enc = PatchEncoder(in_channels=1, ch=8, n_levels=3)
    _, x1, _ = batch()
    feats = enc(x1, None, None)
    assert [f.shape[1] for f in feats] == enc.channels() == [8, 16, 32]
    assert [f.shape[-1] for f in feats] == [H, H // 2, H // 4]
    pnce = PatchNCE(enc.channels(), nc=16, num_patches=64)
    sc = sched()
    x_hat = (x1 + 0.1 * torch.randn_like(x1)).requires_grad_()
    loss = make_nce_fn(enc, pnce, sc)(x1, x_hat)
    loss.backward()
    assert x_hat.grad is not None and _has_grad(enc) and _has_grad(pnce)


def _cycle(net, with_guide=None):
    """One D / E / G pass and a sample for a regressor whose cond is [source, other]: the shape the
    SB-CDL nets need (source at prior_idx 0). NCE runs on a PatchEncoder, never on `net`."""
    sc = sched()
    net.assert_schedule_matches(sc)
    x0, x1, _ = batch()
    cond = torch.cat([x1, torch.randn(B, 1, H, H)], dim=1)               # [source, T2-like]
    kw = {} if with_guide is None else {"guide": with_guide}
    g_fn = make_g_fn(net, sc, x1, cond=cond, **kw)
    D, E = build_critics(1, 2, ndf=8)
    enc = PatchEncoder(1, ch=8, n_levels=2)
    pnce = PatchNCE(enc.channels(), nc=16, num_patches=64)

    state = unsb_forward(g_fn, sc, x1, unsb_steps(sc, K), stage=3)
    d_loss(D, state, x0, cond=cond).backward()
    e_loss(E, state).backward()
    set_requires_grad([D, E], False)
    for m in (D, E):
        m.zero_grad(set_to_none=True)
    net.zero_grad(set_to_none=True)
    total, terms = g_loss(D, E, state, sc, x1, cond=cond, nce=make_nce_fn(enc, pnce, sc))
    total.backward()
    assert math.isfinite(float(total)) and float(terms["nce"]) > 0
    assert _has_grad(net) and not _has_grad(D)

    recon, xs, hats = unsb_sample(net.eval(), x1, sc, cond=cond, num_stages=K, **kw)
    assert recon.shape == x1.shape and hats.shape == (B, K, 1, H, H)


def test_sbcdlnet_runs_the_whole_unsb_cycle():
    from models.sb_cdlnet import SBCDLNet
    torch.manual_seed(0)
    _cycle(SBCDLNet(K=2, M=8, P=3, s=2, C=3, prior_idx=0, kind="brownian", tau=TAU, n_points=N))


def test_sb_guided_groupcdl_runs_the_whole_unsb_cycle_with_a_guide():
    from models.sb_guided_groupcdl import SBGuidedGroupCDL
    torch.manual_seed(0)
    try:
        net = SBGuidedGroupCDL(K=2, M=8, C=3, P=3, s=2, Mh=8, guide_window=5, prior_idx=0,
                               kind="brownian", tau=TAU, n_points=N)
    except RuntimeError as e:
        if "LAPACK" in str(e):
            _skip("local torch 1.12 has no LAPACK (the net's init needs it); run on the cluster")
        raise
    _cycle(net, with_guide=torch.randn(B, 1, H, H))


def main():
    tests = sorted((n, f) for n, f in globals().items() if n.startswith("test_") and callable(f))
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok    {name}")
        except Skipped as e:
            print(f"  SKIP  {name}: {e}")
        except Exception as e:                                             # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
