# -*- coding: utf-8 -*-
"""EMA semantics, and that gradient accumulation reproduces the full batch.

Both were added to match the NVlabs I2SB reference, which trains at an effective batch of 256 via
micro-batching and samples from an EMA at decay 0.99. The two things worth pinning are the ones
that fail silently:

  * accumulation must average the micro-batch gradients, not sum them -- summing scales the
    effective learning rate by accum_steps and nothing errors.
  * the EMA context manager must put the LIVE weights back even when the block raises, or training
    silently continues from the average.
"""
import sys
import types

import pytest
import torch

sys.modules.setdefault("nibabel", types.ModuleType("nibabel"))

from training.common import EMA                      # noqa: E402


def tiny(seed=0):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Conv2d(2, 4, 3, padding=1), torch.nn.ReLU(),
                               torch.nn.Conv2d(4, 1, 3, padding=1))


# ---------------------------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------------------------
def test_shadow_starts_at_the_current_weights():
    net = tiny()
    ema = EMA(net.parameters(), decay=0.99)
    for sh, p in zip(ema.shadow, net.parameters()):
        assert torch.equal(sh, p)


def test_update_is_the_exponential_average():
    """Hand-computed against the recursion, with the warmup ramp disabled."""
    net = tiny()
    d = 0.9
    ema = EMA(net.parameters(), decay=d, use_num_updates=False)
    p0 = [p.detach().clone() for p in net.parameters()]

    with torch.no_grad():                            # one arbitrary weight change
        for p in net.parameters():
            p.add_(1.0)
    ema.update()
    for sh, p in zip(ema.shadow, p0):
        assert torch.allclose(sh, d * p + (1 - d) * (p + 1.0)), "first update is wrong"

    with torch.no_grad():
        for p in net.parameters():
            p.add_(1.0)
    ema.update()
    for sh, p in zip(ema.shadow, p0):
        want = d * (d * p + (1 - d) * (p + 1.0)) + (1 - d) * (p + 2.0)
        assert torch.allclose(sh, want), "second update is wrong"


def test_warmup_ramp_matches_torch_ema():
    """decay_eff = min(decay, (1+n)/(10+n)) -- torch_ema's default, which upstream relies on."""
    net = tiny()
    ema = EMA(net.parameters(), decay=0.99, use_num_updates=True)
    seen = []
    for _ in range(5):
        ema.num_updates += 1
        seen.append(ema._effective_decay())
    assert seen == [min(0.99, (1 + n) / (10 + n)) for n in range(1, 6)]
    # and it reaches the nominal decay only after ~890 updates
    ema.num_updates = 889
    assert ema._effective_decay() < 0.99
    ema.num_updates = 891
    assert ema._effective_decay() == 0.99


def test_average_parameters_restores_the_live_weights():
    net = tiny()
    ema = EMA(net.parameters(), decay=0.5, use_num_updates=False)
    with torch.no_grad():
        for p in net.parameters():
            p.add_(2.0)
    ema.update()                                     # shadow is now halfway
    live = [p.detach().clone() for p in net.parameters()]

    with ema.average_parameters():
        for p, sh in zip(net.parameters(), ema.shadow):
            assert torch.equal(p, sh), "the averaged weights were not installed"
    for p, l in zip(net.parameters(), live):
        assert torch.equal(p, l), "the live weights were not restored"


def test_average_parameters_restores_even_on_exception():
    """The failure this guards: an exception in validation leaving the model on the average,
    and training silently continuing from it."""
    net = tiny()
    ema = EMA(net.parameters(), decay=0.5, use_num_updates=False)
    with torch.no_grad():
        for p in net.parameters():
            p.add_(2.0)
    ema.update()
    live = [p.detach().clone() for p in net.parameters()]

    with pytest.raises(RuntimeError, match="boom"):
        with ema.average_parameters():
            raise RuntimeError("boom")
    for p, l in zip(net.parameters(), live):
        assert torch.equal(p, l), "live weights lost after an exception inside the block"


def test_state_dict_round_trip():
    net = tiny()
    ema = EMA(net.parameters(), decay=0.99)
    for _ in range(7):
        with torch.no_grad():
            for p in net.parameters():
                p.add_(0.1)
        ema.update()
    sd = ema.state_dict()

    other = EMA(tiny(seed=1).parameters(), decay=0.5, use_num_updates=False)
    other.load_state_dict(sd)
    assert other.num_updates == 7
    assert other.decay == pytest.approx(0.99)
    assert other.use_num_updates is True
    for a, b in zip(other.shadow, ema.shadow):
        assert torch.equal(a, b)


def test_load_state_dict_rejects_a_mismatched_model():
    ema = EMA(tiny().parameters(), decay=0.99)
    sd = ema.state_dict()
    sd["shadow"] = sd["shadow"][:-1]
    with pytest.raises(ValueError, match="floating-point parameters"):
        ema.load_state_dict(sd)


def test_non_float_parameters_are_skipped():
    m = torch.nn.Module()
    m.register_parameter("w", torch.nn.Parameter(torch.randn(3)))
    ema = EMA(m.parameters(), decay=0.9)
    assert len(ema.shadow) == 1
    ema.update()                                     # must not raise


# ---------------------------------------------------------------------------------------------
# gradient accumulation
# ---------------------------------------------------------------------------------------------
def test_accumulated_gradient_equals_the_full_batch_gradient():
    """The reason each micro-batch loss is divided by accum_steps.

    A full batch of 8 and 4 micro-batches of 2 must give the SAME gradient, because both are the
    mean over the same 8 samples. Without the division the accumulated gradient is 4x too large --
    which nothing reports, it just quadruples the effective learning rate.
    """
    torch.manual_seed(0)
    x = torch.randn(8, 2, 8, 8)
    y = torch.randn(8, 1, 8, 8)

    net = tiny()
    net.zero_grad()
    torch.nn.functional.mse_loss(net(x), y).backward()
    full = [p.grad.detach().clone() for p in net.parameters()]

    accum = 4
    micro = x.shape[0] // accum
    net.zero_grad()
    for k in range(accum):
        sl = slice(k * micro, (k + 1) * micro)
        loss = torch.nn.functional.mse_loss(net(x[sl]), y[sl])
        (loss / accum).backward()
    got = [p.grad.detach().clone() for p in net.parameters()]

    for f, g in zip(full, got):
        assert torch.allclose(f, g, atol=1e-6), "accumulated gradient != full-batch gradient"

    # and show the un-divided version is wrong by exactly accum_steps, so the test is not vacuous
    net.zero_grad()
    for k in range(accum):
        sl = slice(k * micro, (k + 1) * micro)
        torch.nn.functional.mse_loss(net(x[sl]), y[sl]).backward()
    summed = [p.grad.detach().clone() for p in net.parameters()]
    # compared as NORMS, not element-wise: a per-element ratio explodes wherever the full-batch
    # gradient is near zero and says nothing about the scale.
    n_full = torch.sqrt(sum((f ** 2).sum() for f in full))
    n_sum = torch.sqrt(sum((s ** 2).sum() for s in summed))
    assert float(n_full) > 1e-8, "the fixture produced no gradient; the test would be vacuous"
    assert float(n_sum / n_full) == pytest.approx(accum, rel=1e-4), (
        "summing micro-batch losses should scale the gradient by %d, measured %.4f"
        % (accum, float(n_sum / n_full)))


def test_reported_step_loss_is_the_batch_mean():
    """running_loss accumulates loss/accum_steps per micro-batch, so avg_loss stays comparable
    with a non-accumulated run rather than shrinking by accum_steps."""
    torch.manual_seed(0)
    losses = [torch.tensor(v) for v in (1.0, 2.0, 3.0, 4.0)]
    accum = len(losses)
    step_loss = 0.0
    for l in losses:
        step_loss = step_loss + l / accum
    assert float(step_loss) == pytest.approx(sum(float(l) for l in losses) / accum)


# ---------------------------------------------------------------------------------------------
# sigma (1/sigma^2) loss weighting
# ---------------------------------------------------------------------------------------------
from training.common import snr_loss_weight            # noqa: E402


def _sigmas(b=64, seed=0):
    """Per-sample sigma_t drawn from the brownian tau=0.1 schedule, as train_i2sb would."""
    from sb.base import build_schedule, forward_std, n_steps
    s = build_schedule(kind="brownian", tau=0.1, n_points=1000, beta_max=0.3, device="cpu")
    g = torch.Generator().manual_seed(seed)
    step = torch.randint(0, n_steps(s), (b,), generator=g)
    return forward_std(s, step, xdim=(1, 4, 4)), s


def test_uniform_is_all_ones():
    sig, _ = _sigmas()
    w = snr_loss_weight(sig, "uniform")
    assert torch.allclose(w, torch.ones_like(w))


def test_snr_is_inverse_sigma_squared_up_to_the_batch_mean():
    """The weighting is 1/sigma^2, normalized so the batch mean is 1.

    Checked as a RATIO between samples, which the normalization cannot change -- comparing absolute
    values would just re-derive the normalizer.
    """
    sig, _ = _sigmas()
    w = snr_loss_weight(sig, "snr")
    assert float(w.mean()) == pytest.approx(1.0, rel=1e-5)
    s2 = sig.reshape(sig.shape[0], -1)[:, 0] ** 2
    ratio = w / (1.0 / s2)
    assert torch.allclose(ratio, ratio[0].expand_as(ratio), rtol=1e-5), (
        "snr weights are not proportional to 1/sigma^2")


def test_t1_is_the_other_direction():
    sig, _ = _sigmas()
    lo = int(torch.argmin(sig.reshape(sig.shape[0], -1)[:, 0]))
    hi = int(torch.argmax(sig.reshape(sig.shape[0], -1)[:, 0]))
    snr = snr_loss_weight(sig, "snr")
    t1 = snr_loss_weight(sig, "t1")
    assert snr[lo] > snr[hi], "snr should favour SMALL sigma"
    assert t1[hi] > t1[lo], "t1 should favour LARGE sigma"


def test_w_max_caps_the_raw_weight_and_narrows_the_range():
    """The point of the cap: 1/sigma^2 spans 1000x on this schedule, and the batch-mean
    normalization is then set by whichever sample drew the smallest t."""
    sig, sched = _sigmas(b=512, seed=3)
    s2 = sig.reshape(sig.shape[0], -1)[:, 0] ** 2
    raw = 1.0 / s2
    assert float(raw.max() / raw.min()) > 100, "fixture did not sample a wide sigma range"

    capped = snr_loss_weight(sig, "snr", w_max=1000.0)
    uncapped = snr_loss_weight(sig, "snr")
    assert float(capped.mean()) == pytest.approx(1.0, rel=1e-5)
    assert float(capped.max() / capped.min()) < float(uncapped.max() / uncapped.min()), (
        "the cap did not reduce the weight range")
    # samples BELOW the cap keep their relative weighting exactly
    below = raw < 1000.0
    assert int(below.sum()) > 10
    r = capped[below] / raw[below]
    assert torch.allclose(r, r[0].expand_as(r), rtol=1e-5), (
        "the cap changed the relative weighting of samples it should not touch")


def test_w_max_above_the_maximum_is_a_no_op():
    sig, _ = _sigmas(b=128)
    a = snr_loss_weight(sig, "snr")
    b = snr_loss_weight(sig, "snr", w_max=1e9)
    assert torch.allclose(a, b)


def test_w_max_is_ignored_for_uniform():
    sig, _ = _sigmas()
    assert torch.allclose(snr_loss_weight(sig, "uniform", w_max=10.0),
                          torch.ones(sig.shape[0]))


def test_a_nonpositive_w_max_is_refused():
    sig, _ = _sigmas()
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="must be positive"):
            snr_loss_weight(sig, "snr", w_max=bad)


def test_an_unknown_mode_is_refused():
    sig, _ = _sigmas()
    with pytest.raises(ValueError, match="loss_weight must be"):
        snr_loss_weight(sig, "sigma", w_max=None)
