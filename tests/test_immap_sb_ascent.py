# -*- coding: utf-8 -*-
"""ImMAP-SB annealed ascent on the bridge (sb/immap_sb_ascent.py).

Run with `python -m tests.test_immap_sb_ascent` or pytest.

Pinned:
  * I2SB's ddpm posterior IS one bridge_ascent_step (h = 1 - r^2, beta = 1/(1+r)), same eps;
  * the noise bookkeeping: from a bridge-marginal sample at t with a perfect denoiser, one step
    lands on the bridge marginal at t' (residual std == std_sb(t'));
  * no stall: started at x1 itself (init="x1"), an oracle run reaches x0, in every sigma_mode,
    real and complex;
  * the self-consistent t (quadratic) recovers the true t of a bridge sample;
  * the prox hook is called once per iteration plus the final denoise; the guard rails.
"""
import os
import sys
import types

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

try:
    import torch.nn.attention.flex_attention  # noqa: F401
except ImportError:                                       # torch 1.12 (local only)
    _m = types.ModuleType("torch.nn.attention")
    _m.__path__ = []
    _f = types.ModuleType("torch.nn.attention.flex_attention")
    _f.flex_attention = _f.create_block_mask = lambda *a, **k: None
    _m.flex_attention = _f
    sys.modules["torch.nn.attention"] = _m
    sys.modules["torch.nn.attention.flex_attention"] = _f
sys.modules.setdefault("nibabel", types.ModuleType("nibabel"))

from sb.base import build_schedule, bridge_coeffs                          # noqa: E402
from sb.immap_sb_ascent import (bridge_ascent_step, immap_sb_ascent,       # noqa: E402
                                _Bridge, _quadratic_t)

TAU, N = 0.1, 1000


def pair(B=2, H=48, complex_=False, seed=0):
    g = torch.Generator().manual_seed(seed)
    x0 = torch.rand(B, 1, H, H, generator=g)
    x1 = 0.6 * x0 + 0.4 * torch.rand(B, 1, H, H, generator=g)
    if complex_:
        x0 = x0 * torch.exp(1j * torch.tensor(0.3))
        x1 = x1 * torch.exp(1j * torch.tensor(-0.2))
    return x0, x1


def oracle(x0):
    """A perfect regressor: D(x_t, t) = x0 whatever it is asked."""
    return lambda y, E=None, sigma=None: (x0, None)


def raises(exc, fn, needle=""):
    try:
        fn()
    except exc as e:
        assert needle in str(e), f"{needle!r} not in {e}"
        return
    raise AssertionError(f"{exc.__name__} not raised")


# ---------------------------------------------------------------------------------------------
def test_ddpm_posterior_is_one_ascent_step():
    sched = build_schedule(kind="brownian", tau=TAU, n_points=N)
    br = _Bridge(sched)
    x0, x1 = pair()
    x_n = torch.randn_like(x0)
    x_hat = torch.randn_like(x0)
    eps = torch.randn_like(x0)
    for n, p in ((900, 700), (600, 500), (120, 119), (40, 0)):
        sn2, sp2 = float(sched.std_fwd[n]) ** 2, float(sched.std_fwd[p]) ** 2
        ref = (sn2 - sp2) / sn2 * x_hat + sp2 / sn2 * x_n \
            + (sp2 * (sn2 - sp2) / sn2) ** 0.5 * eps                    # sb.base.reverse_sample
        t_n = torch.full((2,), sn2 / br.V)
        t_p = torch.full((2,), sp2 / br.V)
        r = float(br.sigma_of_t(t_p)[0] / br.sigma_of_t(t_n)[0])
        got = bridge_ascent_step(x_n, x_hat, x1, t_n, t_p, 1 - r ** 2, 1 / (1 + r),
                                 br.std_sb(t_p), eps)
        err = float((got - ref).abs().max())
        assert err < 1e-5, (n, p, err)


def test_step_from_the_marginal_lands_on_the_next_marginal():
    sched = build_schedule(kind="brownian", tau=TAU, n_points=N)
    br = _Bridge(sched)
    x0, x1 = pair(B=1, H=256)
    torch.manual_seed(1)
    for t0, h, beta in ((0.9, 0.05, 0.5), (0.5, 0.3, 0.2), (0.2, 0.8, 0.05)):
        t = torch.tensor([t0])
        x = (1 - t0) * x0 + t0 * x1 + br.std_sb(t) * torch.randn_like(x0)
        t_next = br.t_of_sigma((1 - beta * h) * br.sigma_of_t(t))
        x_new = bridge_ascent_step(x, x0, x1, t, t_next, h, beta, br.std_sb(t_next),
                                   torch.randn_like(x0))
        tn = float(t_next)
        resid = x_new - ((1 - tn) * x0 + tn * x1)
        emp, want = float(resid.pow(2).mean().sqrt()), float(br.std_sb(t_next))
        assert abs(emp / want - 1) < 0.02, (t0, h, beta, emp, want)


def test_oracle_run_reaches_x0_from_x1_in_every_mode():
    sched = build_schedule(kind="brownian", tau=TAU, n_points=N)
    for complex_ in (False, True):
        x0, x1 = pair(complex_=complex_)
        for mode in ("open", "lagged", "quadratic"):
            for det in (True, False):
                torch.manual_seed(0)
                x0_hat, info = immap_sb_ascent(oracle(x0), x1, sched, init="x1",
                                               sigma_mode=mode, deterministic=det,
                                               beta=0.5, h_0=0.05, sigma_min=0.01,
                                               verbose=False)
                err = float((info["x"] - x0).abs().pow(2).mean().sqrt())
                assert info["iters"] < 1000, (mode, det, info["iters"])
                assert info["t"][-1] < info["t"][0], (mode, det, info["t"][:3])
                assert err < 0.02, (complex_, mode, det, err)


def test_open_loop_sigma_follows_the_eero_ladder():
    sched = build_schedule(kind="brownian", tau=TAU, n_points=N)
    x0, x1 = pair()
    _, info = immap_sb_ascent(oracle(x0), x1, sched, t_0=0.99, beta=0.5, h_0=0.05,
                              sigma_min=0.01, verbose=False)
    br = _Bridge(sched)
    s = float(br.sigma_of_t(torch.tensor(0.99)))
    for k, got in enumerate(info["sigma"], start=1):
        assert abs(got - s) < 1e-4 * max(s, 1), (k, got, s)
        h = 0.05 * k / (1 + 0.05 * (k - 1))
        s *= 1 - 0.5 * h
    assert info["iters"] == 26                       # the count quoted in the derivation note


def test_quadratic_recovers_the_true_t_of_a_bridge_sample():
    sched = build_schedule(kind="brownian", tau=TAU, n_points=N)
    br = _Bridge(sched)
    x0, x1 = pair(B=1, H=256)
    torch.manual_seed(2)
    for t0 in (0.9, 0.7, 0.3, 0.05):
        t = torch.tensor([t0])
        x = (1 - t0) * x0 + t0 * x1 + br.std_sb(t) * torch.randn_like(x0)
        t_hat = float(_quadratic_t(x, x0, x1, br.V, torch.tensor([min(t0 + 0.05, 0.99)])))
        assert abs(t_hat - t0) < 0.02, (t0, t_hat)


def test_prox_is_applied_every_iteration():
    sched = build_schedule(kind="brownian", tau=TAU, n_points=N)
    x0, x1 = pair()
    calls = []

    def prox(v, y, step, cond=None, mask=None):
        calls.append(int(step))
        return v, {}

    _, info = immap_sb_ascent(oracle(x0), x1, sched, prox=prox, verbose=False)
    assert len(calls) == info["iters"] + 2           # warm start + iterations + final denoise
    assert calls[1] > calls[-1]                      # the step walks down toward t = 0


def test_guards():
    sched = build_schedule(kind="brownian", tau=TAU, n_points=N)
    x0, x1 = pair()
    raises(ValueError, lambda: immap_sb_ascent(oracle(x0), x1, sched, t_0=1.0), "fixed point")
    raises(ValueError, lambda: immap_sb_ascent(oracle(x0), x1, sched, sigma_mode="x"), "sigma_mode")
    raises(ValueError, lambda: immap_sb_ascent(oracle(x0), x1, sched, beta=1.5), "beta")
    raises(ValueError, lambda: immap_sb_ascent(oracle(x0), x1, sched, init="noise"), "init")


def main():
    tests = sorted((n, f) for n, f in globals().items() if n.startswith("test_") and callable(f))
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok    {name}")
        except Exception as e:                                             # noqa: BLE001
            failed += 1
            import traceback
            traceback.print_exc(limit=6)
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
