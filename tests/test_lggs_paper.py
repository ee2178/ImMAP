# -*- coding: utf-8 -*-
"""The LGGS paper-alignment switches (notes/lggs_paper_alignment.md; arXiv 2610.22643).

Each switch is checked for what it claims, and the defaults for changing nothing:

  tie_transforms     one W_theta/phi/alpha/beta for all layers; the paper-size parameter count;
                     a checkpoint with untied W's refused
  z_init="zero"      layer 0 is a full sweep from (y~, 0)
  self_sim_fun       the self block's similarity, through the fused (pointwise) branch and gather
                     alike
  similarity_scale   "none" removes rho; equals the learned scale at its init value
  adj_blend=False    a rebuilt adjacency is not blended (gamma never consulted)
  guide_mean="joint" one mean over the guide slices
  guide_slices       the loader's N_g side slices, each coil-combined with its own offline maps,
                     through a training step of the paper configuration
  the generator's lggs_paper cell builds and differs from lggs only where intended
"""
import os
import sys
import tempfile
import types

import h5py
import numpy as np
import pytest
import torch

sys.modules.setdefault("nibabel", types.ModuleType("nibabel"))

import tests.test_guided_recon as tgr                              # noqa: E402  (its stubs first)
import models.guided_prox as gp                                    # noqa: E402
import models.prox as prox_mod                                     # noqa: E402
from models.guided_lpds import LGGSNet                             # noqa: E402
from models.guided_prox import GuidedGroupThreshold               # noqa: E402
from operators import Sense                                        # noqa: E402
from operators.fourier import fftc, ifftc                          # noqa: E402

D = torch.float64
PAPER = dict(tie_transforms=True, z_init="zero", guide_mean="joint", self_sim_fun="distance",
             adj_blend=False, similarity_scale="none", dK=1)


def cplx(*shape, g=None):
    return torch.complex(torch.randn(*shape, dtype=D, generator=g),
                         torch.randn(*shape, dtype=D, generator=g))


def small(**kw):
    torch.manual_seed(0)
    base = dict(K=4, M=12, C=1, P=5, s=2, Mh=4, window=1, guide_window=3, dK=2, lam0=0.05,
                theta0=0.3, preproc="image", attn_backend="gather", spectral_init=False)
    return LGGSNet(**dict(base, **kw)).double()


def close(a, b, tol):
    a, b = a.detach(), b.detach()
    return torch.allclose(a, b, rtol=tol, atol=tol * float(b.abs().max()))


# =============================================================================================
# defaults change nothing
# =============================================================================================
def test_defaults_are_the_port_with_tied_transforms():
    """Every switch defaults to the port, except tying, which is forced (Sljiva's runs tie the
    W's and gamma; so does the paper for the W's)."""
    net = small()
    gt = net.layers[0].prox.prox
    assert (net.tie_transforms, net.z_init, net.guide_mean) == (True, "prox", "per_guide")
    assert (gt.self_sim, gt.adj_blend, gt.similarity_scale) == (gt.sim_fun, True, "learned")
    assert gt.rho is not None
    assert net.layers[0].prox.prox.Wtheta is net.layers[1].prox.prox.Wtheta
    with pytest.raises(ValueError, match="no longer supported"):
        small(tie_transforms=False)


# =============================================================================================
# tied transforms
# =============================================================================================
def test_tied_transforms_are_one_object_and_train_as_one():
    net = small(tie_transforms=True)
    w0 = net.layers[0].prox.prox
    for layer in net.layers[1:]:
        t = layer.prox.prox
        assert all(getattr(t, n) is getattr(w0, n)
                   for n in ("Wtheta", "Wphi", "Walpha", "Wbeta", "gamma"))
        assert t.tau is not w0.tau and t.rho is not w0.rho             # tau, rho per layer
        assert layer.analysis is not net.layers[0].analysis            # A, B per layer
    for w in (".Walpha.", ".gamma."):
        assert len([n for n, _ in net.named_parameters() if w in n]) == 1   # counted once
    y, g = cplx(1, 1, 16, 14), [cplx(1, 1, 16, 14)]
    opt = torch.optim.SGD(net.parameters(), lr=1e-2)
    xh, _ = net(y, guide=g, sigma=torch.full((1, 1, 1, 1), 0.05, dtype=D))
    xh.abs().pow(2).mean().backward()
    opt.step()
    for layer in net.layers[1:]:
        assert torch.equal(layer.prox.prox.Walpha.weight, w0.Walpha.weight)


def test_paper_size_parameter_count():
    """Table S1: LGGS 1.08 M (a complex weight counts twice, a tied one once). With real tied
    W's the port lands at ~1.04 M (untied, as it was before 2026-11-09, 2.30 M)."""
    n = sum(p.numel() for p in LGGSNet(
        K=30, M=169, C=1, P=7, s=2, Mh=64, window=1, guide_window=3, attn_backend="gather",
        preproc="image", spectral_init=False).parameters())
    assert 1.03e6 < n < 1.09e6, n


@pytest.mark.parametrize("which", ["Walpha", "gamma"])
def test_tied_net_refuses_an_untied_checkpoint(which):
    """A checkpoint from before the tying (its layers hold different W's / gammas) is refused,
    rather than silently loaded into the one shared tensor, last layer wins."""
    net = small()
    sd = {k: v.clone() for k, v in net.state_dict().items()}
    key = next(k for k in sd if k.startswith("layers.1.") and f".{which}." in k)
    sd[key] += 1.0                                                   # layer 1 now differs
    with pytest.raises(ValueError, match="untied"):
        net.load_state_dict(sd)
    net.load_state_dict(small().state_dict())                        # a tied one loads


# =============================================================================================
# z(0) = 0
# =============================================================================================
@pytest.mark.parametrize("planar", [False, True])
def test_zero_init_is_a_full_sweep_from_zero(planar, monkeypatch):
    from models.mg_lpds import MGLPDSNet
    monkeypatch.setattr(MGLPDSNet, "PLANAR_STATE", planar)
    net = small(z_init="zero")
    layer = net.layers[0]
    y = cplx(1, 1, 16, 14)
    ref = y if not planar else __import__("models.components", fromlist=["x"]).to_planar(y)
    z0 = layer._zero_code(ref)
    assert z0.shape == layer.analysis(ref).shape and z0.dtype == layer.analysis(ref).dtype
    assert not z0.any()
    sig = torch.full((1, 1, 1, 1), 0.05, dtype=D)
    (x_a, z_a), _ = layer(None, ref, guide=None, sigma=sig, cache={})
    (x_b, z_b), _ = layer((ref, z0), ref, guide=None, sigma=sig, cache={})
    assert torch.equal(x_a, x_b) and torch.equal(z_a, z_b)


def test_zero_init_changes_the_net_and_prox_init_does_not():
    """The sweep from (y~, 0) moves x by -eta (E^H E y~ - y~). When E E^H is a projection and
    y~ = E^H(.) -- E = Identity, or single-coil masked FFT -- that is zero and the sweep's dual
    step IS the cold start: the two inits give the same net. With several coils (S S^H != I,
    the knee data's 15) it is a genuine gradient step and the nets differ."""
    from operators import FFT2D, Mask
    from physics.mask import make_acc_mask
    from tests.test_planar_lggs import _sense_problem
    g = [cplx(1, 1, 16, 14).to(torch.complex64)]
    sig = torch.full((1, 1, 1, 1), 0.05)
    mask = make_acc_mask((16, 14), accel=2, acs_lines=4, dim=1, mode="uniform").float()
    for E in (None, Mask(mask) @ FFT2D()):
        x = cplx(1, 1, 16, 14).to(torch.complex64)
        y, kw = (x, {}) if E is None else (E.forward(x), dict(preproc="kspace"))
        a, _ = small(**kw).float()(y, guide=g, E=E, sigma=sig)
        c, _ = small(z_init="zero", **kw).float()(y, guide=g, E=E, sigma=sig)
        assert torch.allclose(a, c, rtol=1e-5, atol=1e-6), E          # coincide
    y, E, guide = _sense_problem(h=16, w=14, coils=3)                 # 3 coils
    sig = torch.full((1, 1, 1, 1), 0.02)
    kw = dict(preproc="kspace")
    a, _ = small(**kw).float()(y, guide=[guide], E=E, sigma=sig)
    b, _ = small(z_init="prox", **kw).float()(y, guide=[guide], E=E, sigma=sig)
    c, _ = small(z_init="zero", **kw).float()(y, guide=[guide], E=E, sigma=sig)
    assert torch.equal(a, b) and not torch.allclose(a, c, rtol=1e-3, atol=1e-5)


# =============================================================================================
# the self similarity, the scale, the blend
# =============================================================================================
@pytest.mark.parametrize("heads", [1, 2])
def test_self_similarity_distance_fused_equals_gather(heads, monkeypatch):
    monkeypatch.setattr(prox_mod, "HAVE_TRITON", True)
    kw = dict(Mh=4, nheads=heads, window=1, guide_window=5, sim_fun="pidistance",
              self_sim_fun="distance", attn_backend="gather", dK=1)
    torch.manual_seed(0)
    gather = GuidedGroupThreshold(8, **kw).double()
    fused = GuidedGroupThreshold(8, triton_materialize=True, **kw).double()
    fused.load_state_dict(gather.state_dict())
    fused.attn_backend, fused.joint_fused = "triton", True        # constructors only, CPU box
    gp._FUSED_JOINT_CHECKED.add(("triton", "pidistance", heads, 1, 5, True, True, "distance"))
    z, gs = cplx(2, 8, 11, 11), [cplx(2, 8, 11, 11) for _ in range(2)]
    a, _ = gather(z, gs, None, {}, dual=True)
    b, _ = fused(z, gs, None, {}, dual=True)
    assert close(b, a, 1e-10)
    # at init W_phi = W_theta, so the self term's q = k and Re<q,q> = |<q,q>|: the two
    # similarities coincide there and differ as soon as the transforms do
    with torch.no_grad():
        gather.Wphi.weight.add_(0.3 * torch.randn_like(gather.Wphi.weight))
    other = GuidedGroupThreshold(8, **dict(kw, self_sim_fun=None)).double()
    other.load_state_dict(gather.state_dict())
    a, _ = gather(z, gs, None, {}, dual=True)
    c, _ = other(z, gs, None, {}, dual=True)
    assert not torch.allclose(a, c)                                  # the switch does something


def test_wider_self_window_with_a_different_self_similarity_is_refused_on_fused():
    with pytest.raises(ValueError, match="1 x 1 self window"):
        GuidedGroupThreshold(8, Mh=4, window=3, guide_window=5, attn_backend="flex",
                             sim_fun="distance", self_sim_fun="dot")


def test_no_similarity_scale_has_no_rho_and_matches_the_scale_at_init():
    torch.manual_seed(0)
    learned = GuidedGroupThreshold(8, Mh=4, window=1, guide_window=5, attn_backend="gather",
                                   sim_fun="pidistance", rho0=1.0, eps=0.0).double()
    none = GuidedGroupThreshold(8, Mh=4, window=1, guide_window=5, attn_backend="gather",
                                sim_fun="pidistance", similarity_scale="none", eps=0.0).double()
    none.load_state_dict({k: v for k, v in learned.state_dict().items() if ".rho." not in
                          f".{k}"}, strict=False)
    assert none.rho is None and not any("rho" in n for n, _ in none.named_parameters())
    none.project_()                                                 # no rho to clamp
    z, gs = cplx(2, 8, 9, 9), [cplx(2, 8, 9, 9)]
    a, _ = learned(z, gs, None, {})
    b, _ = none(z, gs, None, {})
    assert close(b, a, 1e-12)                                       # rho = 1, eps = 0: q / 1


def test_no_blend_never_consults_gamma(monkeypatch):
    p = GuidedGroupThreshold(8, Mh=4, window=1, guide_window=5, attn_backend="gather",
                             sim_fun="pidistance", dK=1, adj_blend=False).double()
    def no_gamma(*a, **k):
        raise AssertionError("the blend consulted gamma")
    monkeypatch.setattr(p.gamma, "forward", no_gamma)
    z, gs = cplx(1, 8, 9, 9), [cplx(1, 8, 9, 9)]
    cache = {}
    for _ in range(3):
        _, cache = p(z, gs, None, cache)
    Phi, _ = p._build_adjacencies(z, gs, None)
    assert torch.equal(cache["Phi"].values, Phi.values)             # the fresh build, as is


# =============================================================================================
# guide mean
# =============================================================================================
def test_joint_guide_mean():
    net = small(guide_mean="joint")
    gs = [cplx(1, 1, 16, 14) + 3.0 * k for k in range(3)]
    out = net._prep_guides(gs, gs[0])
    assert abs(complex(torch.stack(out).mean())) < 1e-12              # zero mean jointly ...
    assert abs(complex(out[0].mean())) > 1.0                         # ... not each on its own
    one = [cplx(1, 1, 16, 14)]
    assert torch.equal(small(guide_mean="joint")._prep_guides(one, one[0])[0],
                       small()._prep_guides(one, one[0])[0])         # N_g = 1: identical


# =============================================================================================
# the loader's guide slices
# =============================================================================================
def _fake_pairs(root, mroot, n=6, C=3, H=12, W=10):
    """One PD/PDFS pair whose offline maps differ PER SLICE, so using the wrong slice's maps
    shows. kspace (n, C, H, W); smaps (n, C, H, W); image (n, H, W)."""
    g = np.random.default_rng(0)
    os.makedirs(mroot, exist_ok=True)
    k = lambda: (g.standard_normal((n, C, H, W)) + 1j * g.standard_normal((n, C, H, W))  # noqa: E731
                 ).astype(np.complex64)
    for name, acq in (("pd0.h5", "CORPD_FBK"), ("pdfs0.h5", "CORPDFS_FBK")):
        with h5py.File(os.path.join(root, name), "w") as f:
            f.create_dataset("kspace", data=k())
            f.attrs["acquisition"] = acq
    sm = (g.standard_normal((n, C, H, W)) + 1j * g.standard_normal((n, C, H, W)))
    sm = (sm / np.sqrt((np.abs(sm) ** 2).sum(1, keepdims=True))).astype(np.complex64)
    with h5py.File(os.path.join(mroot, "pdfs0.h5"), "w") as f:
        f.create_dataset("smaps", data=sm)
        f.create_dataset("image", data=np.zeros((n, H, W), np.complex64))
    csv = os.path.join(root, "split.csv")
    with open(csv, "w") as f:
        f.write("pd,pdfs\npd0.h5,pdfs0.h5\n")
    return csv


def test_loader_guide_slices_and_their_maps():
    from datasets.fastmri.paired import PairedKneeDataset
    from training.recon import _guide, _unpack
    with tempfile.TemporaryDirectory() as root:
        mroot = os.path.join(root, "maps")
        csv = _fake_pairs(root, mroot)
        one = PairedKneeDataset(csv, root, scale_fac=1.0, target="sense", smap_root=mroot,
                                enumerate_slices=True)
        three = PairedKneeDataset(csv, root, scale_fac=1.0, target="sense", smap_root=mroot,
                                  enumerate_slices=True, guide_slices=3)
        assert len(one[2]) == 6 and len(three[2]) == 7            # 1 slice: the batch as before
        with h5py.File(os.path.join(root, "pd0.h5")) as f, \
                h5py.File(os.path.join(mroot, "pdfs0.h5")) as m:
            pdk, maps = f["kspace"][()], m["smaps"][()]
        for sl, js in ((2, [1, 2, 3]), (0, [0, 0, 1]), (5, [4, 5, 5])):   # edges repeat
            item = three[sl]
            assert torch.equal(item[5], torch.from_numpy(pdk[js]))
            assert torch.equal(item[6], torch.from_numpy(maps[js]))
        with pytest.raises(ValueError, match="target='sense'"):
            PairedKneeDataset(csv, root, guide_slices=3)
        with pytest.raises(ValueError, match="odd"):
            PairedKneeDataset(csv, root, target="sense", smap_root=mroot, guide_slices=2)

        # _guide: each slice coil-combined with its own maps -> (B, G, 1, H, W)
        batch = [t[None] for t in three[2]]
        *_, side = _unpack(batch)
        net = small()
        from operators.truncate import embed_operator               # noqa: F401  (import check)
        T = types.SimpleNamespace(is_identity=True)
        g = _guide(net, side, None, T, "cpu")
        assert g.shape == (1, 3, 1, 12, 10)
        for j, js in enumerate([1, 2, 3]):
            want = Sense(torch.from_numpy(maps[js])[None]).adjoint(
                ifftc(torch.from_numpy(pdk[js])[None]))
            assert torch.allclose(g[:, j], want, atol=1e-5)


def test_paper_configuration_trains_a_step_on_three_guide_slices():
    """The whole path: loader (guide_slices=3) -> _guide -> LGGS with every paper switch."""
    net = LGGSNet(K=3, M=8, C=1, P=3, s=2, Mh=4, window=1, guide_window=3, lam0=0.05,
                  preproc="kspace", attn_backend="gather", spectral_init=False, **PAPER)
    with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as d:
        csv, _ = tgr.make_data(root, bad_pair=False)
        mroot = tgr.make_maps(root, os.path.join(root, "maps"))
        seen = []
        net.register_forward_pre_hook(
            lambda mod, a, k: seen.append(tuple(k["guide"].shape)), with_kwargs=True)
        tgr.run_recon(net, csv, root, d, data=dict(target="sense", smap_root=mroot,
                                                   guide_slices=3),
                      online_smaps="espirit")
    assert seen and all(s[1] == 3 for s in seen), seen          # (B, 3, 1, H, W) guides


# =============================================================================================
# the generator's cell
# =============================================================================================
def test_generator_lggs_paper_cell():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "gen", os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts",
                            "make_guided_recon_configs.py"))
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    p = gen.MODELS["lggs_paper"]["model"]["params"]
    assert p["guide_window"] == 3 and p["dK"] == 1 and p["tie_transforms"] is True
    assert (p["z_init"], p["guide_mean"], p["self_sim_fun"], p["similarity_scale"]) == \
        ("zero", "joint", "distance", "none") and p["adj_blend"] is False
    assert gen.MODELS["lggs_paper"]["data"] == {"guide_slices": 3}
    # the recipe switch: the grid's now, Sljiva's (config/train.yaml) at the next sweep
    assert gen.TRAIN_RECIPE in ("grid", "sljiva")
    assert gen.SLJIVA_RECIPE == dict(lr=4e-4, eta_min=2e-6, clip_grad=1.0)
    # LGGS: Sljiva's noise-adaptive threshold, constant steps
    assert (gen.LGGS_PARAMS["degrees"], gen.LGGS_PARAMS["prox_degrees"]) == (0, 1)
    assert "lggs_paper" not in gen.DEFAULT_MODELS                   # array numbering unchanged
    assert gen.LGGS_PARAMS == __import__("models.guided_lpds", fromlist=["x"]).lggs_defaults(
        attn_backend="triton")[0]                                   # the port's point untouched


def _load_generator():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "gen_recipe", os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts",
                                   "make_guided_recon_configs.py"))
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    return gen


@pytest.mark.parametrize("recipe", ["grid", "sljiva"])
def test_one_recipe_for_every_cell(recipe, monkeypatch):
    """Whatever TRAIN_RECIPE says, it says it for EVERY cell: "grid" leaves the grid's values
    (as launched before 2026-11-09), "sljiva" puts 4e-4 -> 2e-6, clip 1 on all of them."""
    gen = _load_generator()
    monkeypatch.setattr(gen, "TRAIN_RECIPE", recipe)
    models = ["lpdsnet", "lggs", "lggs_paper", "tgvn_c8", "sbtgvn_c8_guide"]
    grid = gen.grid_cells(8, [gen.MODELS[m]["grid_base"] for m in models])
    cfgs = {m: gen.make_config(8, m, 1, grid) for m in models}
    clips = {m: c["training"]["clip_grad"] for m, c in cfgs.items()}
    etas = {m: c["scheduler"]["params"]["eta_min"] for m, c in cfgs.items()}
    lrs = {m: c["optimizer"]["params"]["lr"] for m, c in cfgs.items()}
    if recipe == "sljiva":
        assert set(clips.values()) == {1.0} and set(etas.values()) == {2e-6}
        assert set(lrs.values()) == {4e-4}
    else:
        assert set(clips.values()) == {0.05} and set(etas.values()) == {1e-6}
        for m in models:                                  # the grid's own per-model LR
            assert lrs[m] == grid[gen.MODELS[m]["grid_base"]]["optimizer"]["params"]["lr"], m
    assert lrs["lggs"] == lrs["lggs_paper"]               # the paper cell is no exception


# =============================================================================================
# the noise-adaptive threshold
# =============================================================================================
def test_prox_degrees_sets_the_threshold_alone():
    net = small(degrees=0, prox_degrees=1)
    layer = net.layers[0]
    gt = layer.prox.prox
    assert (layer.tau.degrees, layer.theta.degrees, gt.tau.degrees) == (0, 0, 1)
    assert small(degrees=1).layers[0].prox.prox.tau.degrees == 1   # None follows `degrees`
    assert small().layers[0].prox.prox.tau.degrees == 0
    with torch.no_grad():
        gt.tau.weight[1].fill_(0.5)                                  # a slope in sigma
    lo = gt.tau(torch.full((2, 1, 1, 1), 0.01, dtype=D))
    hi = gt.tau(torch.full((2, 1, 1, 1), 0.05, dtype=D))
    assert lo.shape == (2, 12, 1, 1) and torch.all(hi > lo)        # adapts per sample
    sig = torch.tensor([0.01, 0.05], dtype=D).view(2, 1, 1, 1)
    y = cplx(2, 1, 16, 14)
    out, _ = net(y, guide=[cplx(2, 1, 16, 14)], sigma=sig)
    out.abs().pow(2).mean().backward()
    assert gt.tau.weight.grad[1].abs().sum() > 0                    # the slope is trained


def test_lggs_defaults_carry_sljivas_split():
    from models.guided_lpds import lggs_defaults
    cfg, _ = lggs_defaults()
    assert (cfg["degrees"], cfg["prox_degrees"]) == (0, 1)
