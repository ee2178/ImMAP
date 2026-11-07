# -*- coding: utf-8 -*-
"""The planar state for LGGS: (x, z) carried as real [re; im] tensors between sweeps.

The same network -- same parameters, same map -- so every check is an equivalence against the
complex state: the guided prox alone (shrink and clip, guided and not, gather and the
materialised triton construction), then the net end to end (output, dual state and every
parameter gradient), with the identity operator and with SENSE under k-space preprocessing.
"""
import math
import sys
import types

import pytest
import torch

sys.modules.setdefault("nibabel", types.ModuleType("nibabel"))

import models.guided_prox as gp                                    # noqa: E402
from models.components import to_complex, to_planar                # noqa: E402
from models.guided_lpds import LGGSNet                             # noqa: E402
from models.guided_prox import GuidedGroupThreshold, ProjectedGuide  # noqa: E402
from models.mg_lpds import MGLPDSNet, planar_state_report          # noqa: E402
from operators import FFT2D, Mask, Sense                           # noqa: E402
from physics.mask import make_acc_mask                             # noqa: E402


@pytest.fixture
def planar(monkeypatch):
    def set_(on):
        monkeypatch.setattr(MGLPDSNet, "PLANAR_STATE", bool(on))
    return set_


def close(a, b, rtol, atol=0.0):
    a, b = a.detach(), b.detach()
    return torch.allclose(a, b, rtol=rtol, atol=atol + rtol * float(b.abs().max()))


def cplx(*shape, dtype=torch.float64, g=None):
    return torch.complex(torch.randn(*shape, dtype=dtype, generator=g),
                         torch.randn(*shape, dtype=dtype, generator=g))


# =============================================================================================
# the prox
# =============================================================================================
@pytest.mark.parametrize("dual", [False, True])
@pytest.mark.parametrize("n_guides", [0, 1, 2])
@pytest.mark.parametrize("heads", [1, 2])
@pytest.mark.parametrize("backend", ["gather", "triton-materialized"])
def test_planar_prox_is_the_complex_prox(dual, n_guides, heads, backend):
    if backend != "gather" and n_guides == 0:
        pytest.skip("no guide: the plain GroupThreshold runs the real triton kernel (CUDA)")
    torch.manual_seed(0)
    M = 8
    p = GuidedGroupThreshold(M, Mh=4, nheads=heads, window=1, guide_window=5, dK=1,
                             sim_fun="pidistance", attn_backend="gather", rho0=0.7,
                             triton_materialize=backend != "gather").double()
    if backend != "gather":
        p.attn_backend, p.joint_fused = "triton", True       # constructors only on the CPU box
    with torch.no_grad():                    # a per-channel rho, so its planar doubling counts
        p.rho.weight.add_(0.3 * torch.rand_like(p.rho.weight))
    z = cplx(2, M, 10, 9)
    gs = [cplx(2, M, 10, 9) for _ in range(n_guides)]
    want, _ = p(z, gs or None, None, {}, dual=dual)
    got, _ = p(to_planar(z), [to_planar(g) for g in gs] or None, None, {}, dual=dual)
    assert not got.is_complex() and got.shape == (2, 2 * M, 10, 9)
    assert close(to_complex(got), want, 1e-12)


def test_projected_guide_planar_is_the_planar_of_the_complex():
    torch.manual_seed(0)
    layer = LGGSNet(K=1, M=8, C=1, P=3, s=2, Mh=4, nheads=2, window=1, guide_window=3,
                    attn_backend="gather", preproc="image", spectral_init=False).double().layers[0]
    w = cplx(2, 1, 12, 10)
    for planar_first in ("alpha", "phi"):
        pc = ProjectedGuide(w, layer.analysis, layer.prox.prox)
        pp = ProjectedGuide(w, layer.analysis, layer.prox.prox, planar=True)
        for name in (planar_first, "phi" if planar_first == "alpha" else "alpha"):
            a, b = getattr(pp, name), getattr(pc, name)
            assert not a.is_complex()
            assert torch.allclose(a, to_planar(b), rtol=0, atol=0)


# =============================================================================================
# the net
# =============================================================================================
def _grads(net):
    return {n: p.grad.detach().clone() for n, p in net.named_parameters() if p.grad is not None}


def _run(net, planar_on, set_planar, *args, **kw):
    set_planar(planar_on)
    net.zero_grad()
    xh, (x, z) = net(*args, **kw)
    assert xh.is_complex() and x.is_complex() and z.is_complex()    # complex at the boundary
    (xh.abs().pow(2).mean() + 0.1 * z.abs().pow(2).mean()).backward()
    return xh.detach(), z.detach(), _grads(net)


@pytest.mark.parametrize("n_guides", [0, 1, 2])
@pytest.mark.parametrize("heads", [1, 2])
@pytest.mark.parametrize("fuse_guide", [True, False])
def test_lggs_planar_state_identity_operator(n_guides, heads, fuse_guide, planar, monkeypatch):
    monkeypatch.setattr(ProjectedGuide, "FUSE", fuse_guide)
    torch.manual_seed(0)
    net = LGGSNet(K=7, M=12, C=1, P=5, s=2, Mh=4, nheads=heads, window=1, guide_window=5,
                  dK=3, lam0=0.05, theta0=0.3, preproc="image", attn_backend="gather",
                  spectral_init=False).double()
    g = torch.Generator().manual_seed(1)
    y = cplx(2, 1, 16, 14, g=g)
    gd = [cplx(2, 1, 16, 14, g=g) for _ in range(n_guides)] or None
    sigma = torch.full((2, 1, 1, 1), 0.05, dtype=torch.float64)
    (x0, z0, g0), (x1, z1, g1) = (_run(net, on, planar, y, guide=gd, sigma=sigma)
                                  for on in (False, True))
    assert close(x1, x0, 1e-11) and close(z1, z0, 1e-11)
    assert g0.keys() == g1.keys()
    for k in g0:
        assert close(g1[k], g0[k], 1e-9), k


def _sense_problem(h=32, w=24, coils=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    yy = torch.arange(h)[:, None] - h / 2
    xx = torch.arange(w)[None, :] - w / 2
    sm = []
    for c in range(coils):
        a = 2 * math.pi * c / coils
        mag = torch.exp(-((yy - 0.4 * h * math.sin(a)) ** 2 + (xx - 0.4 * w * math.cos(a)) ** 2)
                        / (2 * (0.4 * h) ** 2))
        sm.append(mag * torch.exp(1j * (a + 0.02 * (yy + xx))))
    sm = torch.stack(sm)
    sm = (sm / sm.abs().pow(2).sum(0, keepdim=True).sqrt()).to(torch.complex64)[None]
    x = torch.complex(torch.randn(1, 1, h, w, generator=g), torch.randn(1, 1, h, w, generator=g))
    m = make_acc_mask((h, w), accel=4, acs_lines=6, dim=1, mode="uniform").float()
    E = Mask(m) @ FFT2D() @ Sense(sm)
    guide = torch.complex(torch.randn(1, 1, h, w, generator=g), torch.randn(1, 1, h, w, generator=g))
    return E.forward(x), E, guide


def test_lggs_planar_state_sense_kspace(planar):
    """The k-space preprocessing and the SENSE Gram (planar in, complex inside, planar out)."""
    torch.manual_seed(0)
    net = LGGSNet(K=6, M=12, C=1, P=5, s=2, Mh=4, window=1, guide_window=5, dK=5, lam0=0.05,
                  theta0=0.3, preproc="kspace", attn_backend="gather", spectral_init=False)
    y, E, guide = _sense_problem()
    sigma = torch.full((1, 1, 1, 1), 0.02)
    (x0, z0, g0), (x1, z1, g1) = (_run(net, on, planar, y, guide=[guide], E=E, sigma=sigma)
                                  for on in (False, True))
    assert close(x1, x0, 2e-5) and close(z1, z0, 2e-5)        # fp32: reassociation only
    for k in g0:
        assert close(g1[k], g0[k], 2e-4), k


def test_the_flag_is_reported_and_a_real_net_ignores_it(planar):
    planar(True)
    cnet = LGGSNet(K=2, M=8, C=1, P=3, s=2, Mh=4, guide_window=3, attn_backend="gather",
                   preproc="image", spectral_init=False)
    rnet = LGGSNet(K=2, M=8, C=1, P=3, s=2, Mh=4, guide_window=3, attn_backend="gather",
                   preproc="image", is_complex=False, spectral_init=False)
    assert cnet.planar_state_active() and not rnet.planar_state_active()
    assert planar_state_report(cnet).startswith("planar state: ON")
    y = torch.randn(1, 1, 8, 8)
    out, _ = rnet(y, guide=torch.randn(1, 1, 8, 8))           # real path, unchanged
    assert not out.is_complex()
    planar(False)
    assert planar_state_report(cnet) == "planar state: off"


def test_planar_state_takes_the_fused_scale(planar, monkeypatch):
    """The final clip of a planar code goes through GroupThreshold's shared fused-kernel entry
    (which declines on the CPU and falls back to the eager formula)."""
    calls = []
    orig = gp.GroupThreshold._scale_planar_code

    def spy(self, *a, **k):
        calls.append(1)
        return orig(self, *a, **k)

    monkeypatch.setattr(gp.GroupThreshold, "_scale_planar_code", spy)
    planar(True)
    torch.manual_seed(0)
    net = LGGSNet(K=3, M=8, C=1, P=3, s=2, Mh=4, guide_window=3, attn_backend="gather",
                  preproc="image", spectral_init=False)
    net(cplx(1, 1, 8, 8, dtype=torch.float32), guide=cplx(1, 1, 8, 8, dtype=torch.float32))
    assert len(calls) == 3                                     # one clip per sweep


def test_a_complex_net_fed_a_real_image_is_untouched_by_the_flag(planar):
    """A complex-weight net may be given a REAL image (a magnitude estimate). It has C channels,
    not 2C, so it must not be mistaken for a planar state."""
    torch.manual_seed(0)
    net = LGGSNet(K=4, M=8, C=1, P=3, s=2, Mh=4, guide_window=3, dK=2, lam0=0.05,
                  attn_backend="gather", preproc="image", spectral_init=False).double()
    y = torch.randn(2, 1, 10, 8, dtype=torch.float64)
    g = torch.randn(2, 1, 10, 8, dtype=torch.float64)
    sigma = torch.full((2, 1, 1, 1), 0.05, dtype=torch.float64)
    res = []
    for on in (False, True):
        planar(on)
        net.zero_grad()
        xh, (_, z) = net(y, guide=g, sigma=sigma)
        (xh.abs().pow(2).mean() + z.abs().pow(2).mean()).backward()
        res.append((xh.detach(), z.detach(), _grads(net)))
    (x0, z0, g0), (x1, z1, g1) = res
    assert torch.equal(x1, x0) and torch.equal(z1, z0)
    assert all(torch.equal(g1[k], g0[k]) for k in g0)
