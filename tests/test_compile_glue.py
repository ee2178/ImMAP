# -*- coding: utf-8 -*-
"""The compiled glue of LGGS (models/compile_glue.py) against the eager code it replaces.

On the CPU box the graphs are traced with backend 'aot_eager' -- dynamo + AOTAutograd, the
same capture inductor compiles, without code generation -- so what is checked here is that the
pieces trace WHOLE (fullgraph), are the same map forward and backward, are compiled once for
all layers rather than once per layer, and are taken exactly where they apply.
"""
import sys
import types

import pytest
import torch

sys.modules.setdefault("nibabel", types.ModuleType("nibabel"))

import models.guided_prox as gp                                    # noqa: E402
import models.prox as prox_mod                                     # noqa: E402
from models import compile_glue                                    # noqa: E402
from models.guided_lpds import LGGSNet                             # noqa: E402
from models.mg_lpds import MGLPDSNet                               # noqa: E402

D = torch.float64


@pytest.fixture
def setup(monkeypatch):
    """Planar state on, the triton constructors usable on the CPU (the materialised window runs
    its torch reference here), and the fused self-check marked done."""
    monkeypatch.setattr(MGLPDSNet, "PLANAR_STATE", True)
    monkeypatch.setattr(prox_mod, "HAVE_TRITON", True)
    monkeypatch.setattr(gp, "_FUSED_JOINT_CHECKED", _AlwaysChecked())
    old = compile_glue.MODE
    yield lambda mode: compile_glue.set_mode(mode)
    compile_glue.set_mode(old)
    compile_glue._CACHE.clear()


class _AlwaysChecked(set):
    def __contains__(self, item):
        return True


def lggs(K=7, heads=1, backend="triton", **kw):
    torch.manual_seed(0)
    return LGGSNet(K=K, M=12, C=1, P=5, s=2, Mh=4, nheads=heads, window=1, guide_window=5,
                   dK=3, lam0=0.05, theta0=0.3, preproc="image", attn_backend=backend,
                   triton_materialize=True, spectral_init=False, **kw).double()


def problem(n_guides=2):
    g = torch.Generator().manual_seed(1)
    c = lambda: torch.complex(torch.randn(2, 1, 16, 14, generator=g, dtype=D),   # noqa: E731
                              torch.randn(2, 1, 16, 14, generator=g, dtype=D))
    return c(), [c() for _ in range(n_guides)], torch.full((2, 1, 1, 1), 0.05, dtype=D)


def run(net, y, guides, sigma):
    net.zero_grad()
    xh, (_, z) = net(y, guide=guides, sigma=sigma)
    (xh.abs().pow(2).mean() + 0.1 * z.abs().pow(2).mean()).backward()
    return xh.detach(), z.detach(), {k: p.grad.clone() for k, p in net.named_parameters()
                                     if p.grad is not None}


def close(a, b, tol):
    return torch.allclose(a, b, rtol=tol, atol=tol * float(b.abs().max()))


@pytest.mark.parametrize("heads", [1, 2])
@pytest.mark.parametrize("n_guides", [1, 2])
def test_compiled_glue_is_the_eager_map(heads, n_guides, setup):
    y, guides, sigma = problem(n_guides)
    res = []
    for mode in (None, "aot_eager"):
        setup(mode)
        res.append(run(lggs(heads=heads), y, guides, sigma))
    (x0, z0, g0), (x1, z1, g1) = res
    assert close(x1, x0, 1e-12) and close(z1, z0, 1e-12)
    assert g0.keys() == g1.keys()
    for k in g0:
        assert close(g1[k], g0[k], 1e-10), k


def test_the_glue_is_taken_and_compiled_once_for_all_layers(setup, monkeypatch):
    # count calls OUTSIDE the compiled region (a counter inside it would be traced as a Python
    # side effect, and dynamo would specialise on its value and recompile every call)
    calls = {"_glue_energy_pre": 0, "_glue_envelope_post": 0, "_primal_step": 0}
    real = compile_glue.compiled

    def counting(fn):
        cf = real(fn)

        def wrapped(*a):
            calls[fn.__name__] += 1
            return cf(*a)
        return wrapped

    monkeypatch.setattr(compile_glue, "compiled", counting)
    import torch._dynamo
    torch._dynamo.reset()
    from torch._dynamo.utils import counters
    counters.clear()
    setup("aot_eager")
    K = 7
    run(lggs(K=K), *problem())
    # every sweep's prox (cold start included); the primal step on every sweep but the cold start
    assert calls == {"_glue_energy_pre": K, "_glue_envelope_post": K, "_primal_step": K - 1}
    # pre, post and the primal step: one graph each (a second for a grad-mode or first-call
    # variant at most) -- not one per layer
    assert counters["stats"]["unique_graphs"] <= 6, dict(counters["stats"])


def test_off_and_inapplicable_configurations_keep_the_eager_path(setup, monkeypatch):
    taken = []
    monkeypatch.setattr(gp, "_glue_energy_pre", lambda *a: taken.append(1))
    y, guides, sigma = problem()
    setup(None)                                            # off
    run(lggs(K=3), y, guides, sigma)
    setup("aot_eager")
    run(lggs(K=3, backend="gather"), y, guides, sigma)     # no WindowAdjacency
    monkeypatch.setattr(MGLPDSNet, "PLANAR_STATE", False)  # complex code
    run(lggs(K=3), y, guides, sigma)
    assert not taken
