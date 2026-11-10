# -*- coding: utf-8 -*-
"""I2SB for MRI reconstruction (sb/i2sb_recon.py, models/sb_recon_unet.py, training/i2sb_recon.py)
on synthetic multicoil data.

Run with `python -m tests.test_i2sb_recon` or pytest.

Pinned:
  * the complex bridge: x1 = E^H y, forward_sample at step 0 is ~x0, and the reverse sampler
    driven by an ORACLE regressor lands on x0 -- complex noise and all;
  * SBReconUnet's planar round trip (complex in, complex out) and its guard rails;
  * the training loop runs, updates the net, writes net.ckpt (latest, resumable) and
    net_bestval.ckpt ONLY at validation (and after the last epoch), samples net.ckpt through its
    EMA, and refuses a schedule mismatch and an embedding it cannot use;
  * the shipped configs: every key reaches train_i2sb_recon exactly once, and the model builds.

Runs locally on torch 1.12 with the same stubs as tests/test_train_unsb.py.
"""
import inspect
import json
import os
import sys
import tempfile
import types

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

if "weights_only" not in inspect.signature(torch.load).parameters:    # torch 1.12 (local only)
    _torch_load = torch.load
    torch.load = lambda *a, **k: _torch_load(*a, **{n: v for n, v in k.items() if n != "weights_only"})

try:
    import torch.nn.attention.flex_attention  # noqa: F401
except ImportError:
    _m = types.ModuleType("torch.nn.attention")
    _m.__path__ = []
    _f = types.ModuleType("torch.nn.attention.flex_attention")
    _f.flex_attention = _f.create_block_mask = lambda *a, **k: None
    _m.flex_attention = _f
    sys.modules["torch.nn.attention"] = _m
    sys.modules["torch.nn.attention.flex_attention"] = _f
sys.modules.setdefault("nibabel", types.ModuleType("nibabel"))

try:
    import torchvision.utils  # noqa: F401
except ImportError:
    _tv, _tvu = types.ModuleType("torchvision"), types.ModuleType("torchvision.utils")

    def _make_grid(t, nrow=8, **k):
        t = t if t.ndim == 4 else t[None]
        rows = [torch.cat(list(t[i:i + nrow]), dim=-1) for i in range(0, len(t), nrow)]
        return torch.cat(rows, dim=-2)

    _tvu.make_grid, _tvu.save_image, _tv.utils = _make_grid, (lambda *a, **k: None), _tvu
    sys.modules["torchvision"], sys.modules["torchvision.utils"] = _tv, _tvu

try:
    import wandb as _wandb_mod  # noqa: F401
    _have_wandb = hasattr(_wandb_mod, "Image")
except ImportError:
    _have_wandb = False
if not _have_wandb:
    _w = types.ModuleType("wandb")
    _w.Image = lambda data, **kw: ("image", tuple(np.shape(data)))
    sys.modules["wandb"] = _w
sys.modules.setdefault("lpips", types.ModuleType("lpips"))

try:                                    # `training/__init__` imports every trainer; skip it locally
    import training  # noqa: F401
except Exception:                       # noqa: BLE001
    for _k in [k for k in sys.modules if k == "training" or k.startswith("training.")]:
        del sys.modules[_k]
    _pkg = types.ModuleType("training")
    _pkg.__path__ = [os.path.join(REPO, "training")]
    sys.modules["training"] = _pkg

from operators import Mask, FFT2D, Sense                                   # noqa: E402
from physics.mask import get_mask_cached                                   # noqa: E402
from sb.base import build_schedule, forward_sample                         # noqa: E402
from sb.i2sb_recon import (acs_lines_mask, bridge_start, cg_data_consistency,  # noqa: E402
                           i2sb_recon_sample)
from models.sb_recon_unet import SBReconUnet, to_planar, from_planar       # noqa: E402
from training.i2sb_recon import train_i2sb_recon                           # noqa: E402

TAU, N, H, NC = 0.1, 1000, 32, 4
MRI = dict(R=4, acs_lines=4, kspace_type="simulated", mask_dist="uniform")


class FakeWandb:
    def __init__(self):
        self.calls = []

    def log(self, d, step=None):
        self.calls.append((step, d))

    def keys(self):
        return {k for _, d in self.calls for k in d}


def phantom(n, seed=0):
    """(kspace, smaps, image, mask, pad_hw) like the fastMRI recon loader, unit-RSS maps."""
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, H))
    disk = ((yy ** 2 + xx ** 2) < 0.6).float()
    mag = disk * (0.5 + 0.5 * torch.rand(n, 1, H, H, generator=g))
    phase = 0.3 * torch.randn(n, 1, 1, 1, generator=g)
    image = (mag * torch.exp(1j * phase)).to(torch.complex64)
    smaps = torch.randn(n, NC, H, H, generator=g) + 1j * torch.randn(n, NC, H, H, generator=g)
    smaps = (smaps / smaps.abs().pow(2).sum(1, keepdim=True).sqrt()).to(torch.complex64)
    kspace = torch.zeros(n, NC, H, H, dtype=torch.complex64)
    mask = torch.ones(n, 1, H, H, dtype=torch.bool)
    pad_hw = torch.tensor([[H, H]] * n)
    return kspace, smaps, image, mask, pad_hw


def loader(n=4, bs=2, seed=0):
    return DataLoader(TensorDataset(*phantom(n, seed)), batch_size=bs, shuffle=False)


def make_net(seed=0, tau=TAU, cond_x1=True):
    torch.manual_seed(seed)
    return SBReconUnet(cond_x1=cond_x1, model_channels=32, num_res_blocks=1, channel_mult=(1, 2),
                       attention_resolutions=(16,), image_size=H, num_head_channels=32,
                       kind="brownian", tau=tau, n_points=N)


def operator(smaps, image):
    mask = get_mask_cached(image, R=MRI["R"], acs_lines=MRI["acs_lines"], mode="uniform")
    return Mask(mask) @ FFT2D() @ Sense(smaps)


def raises(exc, fn, needle=""):
    try:
        fn()
    except exc as e:
        assert needle in str(e), f"{needle!r} not in {e}"
        return
    raise AssertionError(f"{exc.__name__} not raised")


def run(net, save_dir, num_epochs=2, start_epoch=0, wandb=None, **kw):
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    args = dict(num_epochs=num_epochs, steps_per_epoch=2, val_every_epochs=1, accum_steps=2,
                ema_decay=0.9, noise_std=(0.0, 0.01), val_noise_std=0.005, val_seed=0,
                kind="brownian", tau=TAU, n_points=N, val_mode="full_recon", val_nfe=2,
                save_dir=save_dir, **MRI)
    args.update(kw)
    return train_i2sb_recon(net, opt, None, "cpu", loader(), loader(n=2, seed=1),
                            wandb=wandb, start_epoch=start_epoch, **args)


# ---------------------------------------------------------------------------------------------
def test_planar_round_trip():
    x = torch.randn(2, 1, 8, 8, dtype=torch.complex64)
    p = to_planar(x)
    assert p.shape == (2, 2, 8, 8) and not p.is_complex()
    assert torch.equal(from_planar(p), x)


def test_adapter_is_complex_in_complex_out_and_guards():
    net = make_net()
    sched = build_schedule(kind="brownian", tau=TAU, n_points=N)
    x = torch.randn(2, 1, H, H, dtype=torch.complex64)
    out, _ = net(x, sigma=sched.std_fwd[[3, 500]].view(-1, 1, 1, 1), x1=x)
    assert out.shape == x.shape and out.is_complex()
    raises(TypeError, lambda: net(x.real, sigma=sched.std_fwd[:2], x1=x), "complex")
    raises(ValueError, lambda: net(x, sigma=sched.std_fwd[:2]), "needs x1")
    raises(ValueError, lambda: SBReconUnet(C=4), "sets the UNet's C")
    out, _ = make_net(cond_x1=False)(x, sigma=sched.std_fwd[:2])          # no x1 needed
    assert out.shape == x.shape


def test_bridge_start_is_the_adjoint_and_step0_is_x0():
    _, smaps, image, _, _ = phantom(2)
    E = operator(smaps, image)
    y = E(image)
    x1 = bridge_start(y, E)
    assert torch.allclose(x1, E.adjoint(y))
    raises(ValueError, lambda: bridge_start(y, E, "cg"), "not supported")
    sched = build_schedule(kind="brownian", tau=TAU, n_points=N)
    xt = forward_sample(sched, torch.zeros(2, dtype=torch.long), image, x1, deterministic=True)
    assert (xt - image).abs().max() < 1e-2 * image.abs().max()
    # complex bridge noise has total std std_sb: E|eps|^2 = 1
    xt = forward_sample(sched, torch.full((2,), N // 2, dtype=torch.long), image, image)
    from sb.base import bridge_coeffs
    std = float(bridge_coeffs(sched)[2][N // 2])
    emp = float((xt - image).abs().pow(2).mean().sqrt())
    assert abs(emp / std - 1) < 0.05, (emp, std)


def test_oracle_regressor_samples_to_x0():
    _, smaps, image, _, _ = phantom(2)
    E = operator(smaps, image)
    y = E(image)
    sched = build_schedule(kind="brownian", tau=TAU, n_points=N)

    def oracle(x_t, sigma=None, **_):
        return image, None

    for posterior in ("ddpm", "interpolant"):
        torch.manual_seed(0)
        rec, xs, preds = i2sb_recon_sample(oracle, y, E, sched, nfe=10, posterior=posterior,
                                           verbose=False)
        assert rec.is_complex() and rec.shape == image.shape
        err = float((rec - image).abs().max() / image.abs().max())
        assert err < 0.05, (posterior, err)


def test_training_loop_updates_and_checkpoints():
    with tempfile.TemporaryDirectory() as d:
        net = make_net()
        before = [p.detach().clone() for p in net.parameters()]
        wb = FakeWandb()
        run(net, d, wandb=wb)
        changed = sum(not torch.equal(a, b) for a, b in zip(before, net.parameters()))
        assert changed > 0
        assert sorted(f for f in os.listdir(d) if f.endswith(".ckpt")) == \
            ["net.ckpt", "net_bestval.ckpt"], os.listdir(d)          # no net_ema / net_last
        # net.ckpt: the LATEST validation's state, whatever its score -- the final step here --
        # resumable (optimizer, scheduler slot, EMA) and carrying that validation's -PSNR
        last = torch.load(os.path.join(d, "net.ckpt"), map_location="cpu", weights_only=False)
        assert last["step"] == 2 * 2, last["step"]                   # 2 epochs x 2 steps
        assert last["optimizer_state_dict"] is not None and last["ema_state_dict"] is not None
        psnrs = [c[1]["val/psnr"] for c in wb.calls if "val/psnr" in c[1]]
        assert abs(-last["best_val_loss"] - psnrs[-1]) < 1e-4
        # the LIVE weights (training continues from them), not the average
        live = dict(net.named_parameters())
        assert all(torch.equal(v, live[k].detach()) for k, v in last["model_state_dict"].items()
                   if k in live)
        keys = wb.keys()
        for k in ("train/loss", "val/psnr", "val/zf_psnr", "val/recon_example"):
            assert k in keys, (k, sorted(keys))


def test_checkpoints_only_at_validation():
    """Every write happens at a validation step, or after the last epoch: none on the epochs
    between, however much the train loss improves (bigpurple's scratch, 2026-11-10)."""
    from training.common import save_ckpt
    from training.i2sb_recon import load_sampling_weights
    writes = []

    def counting_save(path, **kw):
        writes.append((os.path.basename(path), kw["step"]))
        return save_ckpt(path, **kw)

    with tempfile.TemporaryDirectory() as d:
        net = make_net()
        # 5 epochs x 2 steps, validation every 2 epochs -> validations at steps 4 and 8, the
        # last epoch ends at step 10 (not a validation)
        run(net, d, num_epochs=5, val_every_epochs=2, save_ckpt_fn=counting_save)
        steps = {s for _, s in writes}
        assert steps <= {4, 8, 10}, writes
        assert [s for f, s in writes if f == "net.ckpt"] == [4, 8, 10], writes
        assert all(s in (4, 8) for f, s in writes if f == "net_bestval.ckpt"), writes

        # sampling from net.ckpt uses its EMA, net_bestval.ckpt its (already averaged) weights
        ck = torch.load(os.path.join(d, "net.ckpt"), map_location="cpu", weights_only=False)
        fresh = make_net()
        load_sampling_weights(fresh, os.path.join(d, "net.ckpt"))
        named = [(k, p) for k, p in fresh.named_parameters() if p.is_floating_point()]
        assert len(named) == len(ck["ema_state_dict"]["shadow"])
        assert all(torch.equal(p.detach(), sh)
                   for (_, p), sh in zip(named, ck["ema_state_dict"]["shadow"]))
        assert any(not torch.equal(p.detach(), ck["model_state_dict"][k])
                   for k, p in named)                                # i.e. not the live ones
        bv = torch.load(os.path.join(d, "net_bestval.ckpt"), map_location="cpu",
                        weights_only=False)
        load_sampling_weights(fresh, os.path.join(d, "net_bestval.ckpt"))
        assert all(torch.equal(v, fresh.state_dict()[k]) for k, v in bv["model_state_dict"].items())


def test_single_pass_validation_and_resume():
    with tempfile.TemporaryDirectory() as d:
        net = make_net()
        run(net, d, num_epochs=1, val_mode="single_pass")
        run(net, d, num_epochs=2, start_epoch=1, val_mode="single_pass",
            ckpt=os.path.join(d, "net.ckpt"))


def test_refuses_schedule_mismatch_and_embedding():
    with tempfile.TemporaryDirectory() as d:
        raises(ValueError, lambda: run(make_net(tau=0.2), d), "schedule mismatch")
        net = make_net()
        net.pad_stride = 64                                  # pretend it needs a bigger grid
        raises(ValueError, lambda: run(net, d), "pad_stride")


def test_bridge_target_follows_the_operator_maps():
    """Online maps in a different phase convention: x0 must be the image THEY produce from the
    full k-space, gt the stored image, and |x0| == |gt| (the metrics are unchanged)."""
    import training.i2sb_recon as tir
    from operators.fourier import fftc
    kspace, smaps, image, mask, pad_hw = phantom(2)
    kspace = fftc(smaps * image)                                     # consistent full k-space
    yy = torch.linspace(-2, 2, H)
    phase = torch.exp(1j * (yy[:, None] + 0.5 * yy[None, :])).to(torch.complex64)
    online = smaps * phase                                           # same maps, other reference

    real = tir.prepare_measurement
    def fake(**kw):
        y, s, extra = real(**{**kw, "kspace_type": "simulated", "online_smaps": None})
        return y, s, {**extra, "smaps": online, "smaps_data": smaps}
    tir.prepare_measurement = fake
    try:
        meas = dict(R=4, acs_lines=4, mask_dist="uniform", mask_offset=0, center_frac=None,
                    adjust_accel=False, noise_dist="uniform", online_smaps="espirit",
                    online_smaps_kws=None, x1_source="adjoint")
        batch = (kspace, smaps, image, mask, pad_hw)
        net = make_net()
        x0, x1, y, E, s, om, gt, guide = tir._measure(batch, net, "cpu",
                                                      kspace_type="measurement_awgn",
                                                      noise_std=0.0, x0_source="operator", **meas)
        assert guide is None                                          # 5-tuple batch: no side
        assert torch.allclose(gt, image)
        assert torch.allclose(x0, image * phase.conj(), atol=1e-5)
        assert torch.allclose(x0.abs(), gt.abs(), atol=1e-5)
        x0s = tir._measure(batch, net, "cpu", kspace_type="measurement_awgn", noise_std=0.0,
                           x0_source="stored", **meas)[0]
        assert torch.allclose(x0s, image)
        x0sim = tir._measure(batch, net, "cpu", kspace_type="simulated", noise_std=0.0,
                             x0_source="operator", **meas)[0]
        assert torch.allclose(x0sim, image)                          # simulated: image IS x0
    finally:
        tir.prepare_measurement = real


def test_shipped_configs_reach_the_loop_and_build():
    from models import build_model
    params = inspect.signature(train_i2sb_recon).parameters
    for anat in ("brain", "knee"):
        path = os.path.join(REPO, "config", anat, "i2sb", "unet_R8.json")
        import yaml
        with open(path) as f:
            cfg = yaml.safe_load(f)                 # how train.py reads it: 1e-06 is a STRING
        assert isinstance(cfg["scheduler"]["params"]["eta_min"], float),             f"{anat}: eta_min loads as {cfg['scheduler']['params']['eta_min']!r}"
        assert cfg["task"] == "i2sb_recon"
        blocks = [cfg["training"], cfg["mri"], cfg["i2sb"], cfg["paths"]]
        seen = set()
        for b in blocks:
            for k in b:
                assert k in params, f"{anat}: {k!r} is not a train_i2sb_recon argument"
                assert k not in seen, f"{anat}: {k!r} appears in two config blocks"
                seen.add(k)
        for k in ("kind", "tau", "n_points", "beta_max"):
            assert cfg["model"]["params"][k] == cfg["i2sb"][k], (anat, k)
        net = build_model(cfg)
        assert isinstance(net, SBReconUnet)
        net.assert_schedule_matches(build_schedule(
            **{k: cfg["i2sb"][k] for k in ("kind", "tau", "n_points", "beta_max")}))


def _rel_res(x, y, E):
    return float((y - E(x)).abs().pow(2).sum().sqrt() / y.abs().pow(2).sum().sqrt())


def test_cg_data_consistency_solves_the_tikhonov_problem():
    """x_bar = argmin ||y - E x||^2 + lam ||x - x_hat||^2: the normal equations hold, 0 iters is
    the identity, and a hallucinating x_hat is pulled toward the data."""
    _, smaps, image, _, _ = phantom(2)
    E = operator(smaps, image)
    y = E(image)
    g = torch.Generator().manual_seed(1)
    x_hat = image + 0.3 * torch.randn(image.shape, generator=g, dtype=torch.complex64)
    assert cg_data_consistency(x_hat, y, E, 1e-2, 0) is x_hat
    lam = 1e-2
    x = cg_data_consistency(x_hat, y, E, lam, 200)
    normal = E.adjoint(E(x) - y) + lam * (x - x_hat)
    assert float(normal.abs().max()) < 1e-4, float(normal.abs().max())
    assert _rel_res(x, y, E) < 0.1 * _rel_res(x_hat, y, E)
    # few iterations from the warm start already cut the data residual, per slice
    x7 = cg_data_consistency(x_hat, y, E, 7e-5, 7)
    for b in range(2):
        Eb = _slice_op(smaps, image, b)
        assert _rel_res(x7[b:b + 1], y[b:b + 1], Eb) < 0.5 * _rel_res(x_hat[b:b + 1],
                                                                      y[b:b + 1], Eb)


def _slice_op(smaps, image, b):
    return operator(smaps[b:b + 1], image[b:b + 1])


def test_dc_sampler_corrects_a_hallucinating_regressor():
    """A regressor that adds a structure the data rules out: the plain sampler keeps it, the
    CG-corrected sampler (I2SB-Recon) removes most of it. dc_iters=0 is the plain sampler."""
    _, smaps, image, _, _ = phantom(2)
    E = operator(smaps, image)
    y = E(image)
    sched = build_schedule(kind="brownian", tau=TAU, n_points=N)
    blob = torch.zeros_like(image)
    blob[..., 10:14, 10:14] = 0.5

    def hallucinator(x_t, sigma=None, **_):
        return image + blob, None

    out = {}
    for it in (0, 7):
        torch.manual_seed(0)
        out[it] = i2sb_recon_sample(hallucinator, y, E, sched, nfe=10, verbose=False,
                                    dc_iters=it, dc_lam=7e-5)[0]
    torch.manual_seed(0)
    plain = i2sb_recon_sample(hallucinator, y, E, sched, nfe=10, verbose=False)[0]
    assert torch.equal(plain, out[0])
    err = {k: float((v - image).abs().pow(2).sum().sqrt()) for k, v in out.items()}
    assert err[7] < 0.5 * err[0], err
    assert _rel_res(out[7], y, E) < 0.5 * _rel_res(out[0], y, E)


def test_support_and_trace():
    """`support`: every x0 prediction (so the recon) is 0 outside it, also with CG; with
    `support_state` the state is too. `trace`: one entry per evaluation, pre / post DC."""
    _, smaps, image, _, _ = phantom(2)
    E = operator(smaps, image)
    y = E(image)
    sched = build_schedule(kind="brownian", tau=TAU, n_points=N)
    bg = torch.zeros_like(image)
    bg[..., :4, :] = 0.3                                     # the regressor leaks background
    sup = torch.ones(2, 1, H, H, dtype=torch.bool)
    sup[..., :6, :] = False

    def leaky(x_t, sigma=None, **_):
        return image + bg, None

    for dc in (0, 3):
        trace = []
        torch.manual_seed(0)
        rec = i2sb_recon_sample(leaky, y, E, sched, nfe=6, verbose=False, dc_iters=dc,
                                dc_lam=1e-2, support=sup, trace=trace)[0]
        assert float(rec[~sup.expand_as(rec)].abs().max()) == 0.0
        assert len(trace) == 6 and all(set(t) == {"step", "x_t", "x0_pre", "x0_dc", "x0_post"}
                                       for t in trace)
        assert float(trace[0]["x0_pre"][..., :4, :].abs().max()) > 0.1   # before the support
        assert float(trace[0]["x0_post"][..., :4, :].abs().max()) == 0.0
        # the state keeps its background unless support_state
        assert float(trace[-1]["x_t"][..., :4, :].abs().max()) > 0
    trace = []
    i2sb_recon_sample(leaky, y, E, sched, nfe=6, verbose=False, support=sup, support_state=True,
                      trace=trace)
    assert all(float(t["x_t"][..., :6, :].abs().max()) == 0.0 for t in trace)


def test_online_support_thresholds_the_operators_eigenvalues():
    import physics.online_smaps as osm
    from sb.i2sb_recon import online_support
    lam = torch.linspace(0, 1, H * H).reshape(1, H, H).expand(2, H, H)
    seen = {}

    def fake(y, mask, acs_lines=None, thresh_eig=None, return_lam=False, **kw):
        seen.update(kw, thresh_eig=thresh_eig, return_lam=return_lam)
        return torch.ones(2, 4, H, H, dtype=torch.complex64), lam
    real, osm.online_smaps = osm.online_smaps, fake
    try:
        s = online_support(torch.zeros(2, 4, H, H, dtype=torch.complex64), None, 4, 0.9,
                           kernel_size=4, thresh_eig=0.0)
    finally:
        osm.online_smaps = real
    assert s.shape == (2, 1, H, H) and s.dtype == torch.bool
    assert torch.equal(s[:, 0], lam > 0.9)
    assert seen["return_lam"] and seen["thresh_eig"] == 0.0 and seen["kernel_size"] == 4


def test_dc_validation_logs_both_samplers():
    with tempfile.TemporaryDirectory() as d:
        wb = FakeWandb()
        run(make_net(), d, num_epochs=1, wandb=wb, dc_iters=3, dc_lam=1e-3)
        keys = wb.keys()
        for k in ("val/psnr", "val/nodc_psnr", "val/nodc_ssim", "val/nodc_nrmse"):
            assert k in keys, (k, sorted(keys))
        raises(ValueError, lambda: run(make_net(), d, dc_iters=-1), "dc_iters")
    with tempfile.TemporaryDirectory() as d:
        wb = FakeWandb()
        run(make_net(), d, num_epochs=1, wandb=wb)
        assert "val/nodc_psnr" not in wb.keys()


def test_acs_bridge_start_is_the_centre_block_low_pass():
    """x1_source="acs": E^H of y restricted to the contiguous sampled block around the centre
    line -- exactly the coil-combined low-pass of the full k-space there, so no aliasing."""
    from operators.fourier import ifftc as _ifftc
    _, smaps, image, _, _ = phantom(2)
    E = operator(smaps, image)
    y = E(image)
    acs = acs_lines_mask(y, E)
    lines = acs.reshape(-1, H)[0]
    idx = torch.nonzero(lines).flatten()
    assert bool(lines[H // 2]) and int(idx[-1] - idx[0] + 1) == int(lines.sum())   # contiguous
    sampled = (E.ops[0].mask.abs() > 0).reshape(-1, H).any(0) if torch.is_tensor(
        E.ops[0].mask) else None
    if sampled is not None:
        assert int(lines.sum()) < int(sampled.sum())                      # outer lines dropped
        assert not bool(sampled[idx[0] - 1]) and not bool(sampled[idx[-1] + 1])  # maximal run
    x1 = bridge_start(y, E, "acs")
    full = Sense(smaps).forward(image)                                    # coil images
    k = __import__("operators.fourier", fromlist=["fftc"]).fftc(full) * lines.view(1, 1, 1, H)
    want = (smaps.conj() * _ifftc(k)).sum(1, keepdim=True)
    assert torch.allclose(x1, want, atol=1e-5), float((x1 - want).abs().max())
    # same from y's support alone (no Mask op to read)
    class Bare:
        pass
    from_y = acs_lines_mask(y, Bare()).reshape(-1, H)                     # one row per slice
    assert from_y.shape[0] == 2 and all(torch.equal(r, lines) for r in from_y)
    raises(ValueError, lambda: acs_lines_mask(y * 0, Bare()), "centre line")


def test_training_with_acs_bridge_start():
    with tempfile.TemporaryDirectory() as d:
        wb = FakeWandb()
        run(make_net(), d, num_epochs=1, wandb=wb, x1_source="acs")
        assert "val/psnr" in wb.keys()


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
