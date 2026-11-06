# -*- coding: utf-8 -*-
"""Guided reconstruction plumbing: the paired knee loader, the TGVN port, and train_recon's guide
path (datasets/fastmri/paired.py, models/tgvn/, training/recon.py, scripts/
make_guided_recon_configs.py), on synthetic PD/PDFS h5 pairs.

Run with `python -m tests.test_guided_recon` or pytest.

Pinned:
  * the loader pairs by index, drops shape-mismatched pairs, checks acquisitions (pd = side),
    returns the RSS target and the side k-space;
  * the vendored TGVN differs from upstream only in its import block;
  * TGVN gets the side K-SPACE, LGGS the side IMAGE under the operator's maps, an unguided net
    on paired data gets nothing -- and train_recon runs end to end with each;
  * the refusals: simulated k-space or operator-map models without online maps on paired data,
    a guided model without side data;
  * the generator's LGGS point equals lggs_defaults(), and every config key reaches train_recon.

Runs locally on torch 1.12 with stubs (no flex attention / torchvision / wandb, and no LAPACK --
TGVN's Cholesky normalization of its (B, 2, 2) covariances then uses the closed-form 2x2 factor,
which is DIFFERENTIABLE like the real one: a numpy stand-in hid an autograd bug once, see
test_cond_plane_backward).
"""
import inspect
import json
import math
import os
import sys
import tempfile
import types

import h5py
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

if "weights_only" not in inspect.signature(torch.load).parameters:
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
    _tvu.make_grid = lambda t, nrow=8, **k: t[0] if t.ndim == 4 else t
    _tvu.save_image, _tv.utils = (lambda *a, **k: None), _tvu
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
try:
    import training  # noqa: F401
except Exception:                       # noqa: BLE001
    for _k in [k for k in sys.modules if k == "training" or k.startswith("training.")]:
        del sys.modules[_k]
    _pkg = types.ModuleType("training")
    _pkg.__path__ = [os.path.join(REPO, "training")]
    sys.modules["training"] = _pkg

try:                                    # `datasets/__init__` imports every loader; skip it locally
    import datasets  # noqa: F401
except Exception:                       # noqa: BLE001
    for _k in [k for k in sys.modules if k == "datasets" or k.startswith("datasets.")]:
        del sys.modules[_k]
    _dpkg = types.ModuleType("datasets")
    _dpkg.__path__ = [os.path.join(REPO, "datasets")]
    sys.modules["datasets"] = _dpkg

try:                                    # local torch builds without LAPACK
    torch.linalg.cholesky(torch.eye(2).expand(1, 2, 2))
except RuntimeError:
    def _chol(A):
        if A.shape[-2:] != (2, 2):
            return torch.from_numpy(np.linalg.cholesky(A.detach().cpu().numpy())).to(A)
        l11 = A[..., 0, 0].sqrt()                 # closed form, differentiable, as LAPACK's is
        l21 = A[..., 1, 0] / l11
        l22 = (A[..., 1, 1] - l21 ** 2).sqrt()
        z = torch.zeros_like(l11)
        return torch.stack([torch.stack([l11, z], -1), torch.stack([l21, l22], -1)], -2)

    def _inv(A):
        if A.shape[-2:] != (2, 2):
            return torch.from_numpy(np.linalg.inv(A.detach().cpu().numpy())).to(A)
        a, b, c, d = A[..., 0, 0], A[..., 0, 1], A[..., 1, 0], A[..., 1, 1]
        det = a * d - b * c
        return torch.stack([torch.stack([d, -b], -1), torch.stack([-c, a], -1)], -2)             / det[..., None, None]
    torch.linalg.cholesky, torch.linalg.inv = _chol, _inv
    def _qr(A, mode="reduced"):
        q, r = np.linalg.qr(A.detach().cpu().numpy(), mode=mode)
        return torch.from_numpy(q).to(A), torch.from_numpy(r).to(A)
    torch.linalg.qr = _qr

from datasets.fastmri.paired import PairedKneeDataset, get_paired_knee_loader  # noqa: E402
from models.tgvn import TGVN                                                   # noqa: E402
from operators import FFT2D, Mask, Sense                                       # noqa: E402
from operators.fourier import fftc, ifftc                                      # noqa: E402
from physics.mask import get_mask_cached                                       # noqa: E402
import training.recon as tr                                                    # noqa: E402

S, C, H, W = 4, 4, 32, 32
TINY_TGVN = dict(num_cascades=2, sens_chans=4, sens_pools=2, chans=4, pools=2)


def _smooth_maps(seed=7):
    """Smooth unit-RSS coil maps: shared by every synthetic volume, so the tests can hand the
    TRUE maps to code paths that would estimate them online."""
    g = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing="ij")
    sm = []
    for c in range(C):
        a = 2 * np.pi * c / C
        sm.append(np.exp(-((yy - 0.7 * np.sin(a)) ** 2 + (xx - 0.7 * np.cos(a)) ** 2))
                  * np.exp(1j * (g.random() * 2 * np.pi + yy * c)))
    sm = np.stack(sm)
    return (sm / np.sqrt((np.abs(sm) ** 2).sum(0, keepdims=True))).astype(np.complex64)


MAPS = _smooth_maps()


def _vol(g):
    yy, xx = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing="ij")
    disk = ((yy ** 2 + xx ** 2) < 0.6).astype(np.float32)
    img = disk[None] * (0.5 + 0.5 * g.random((S, H, W)))
    sm = MAPS
    coils = sm[None] * img[:, None]
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(coils, axes=(-2, -1)), norm="ortho"),
                           axes=(-2, -1)).astype(np.complex64), sm.astype(np.complex64)


def make_data(root, n_pairs=3, bad_pair=True, swap=False):
    """n_pairs PD/PDFS volumes (+ one shape-mismatched pair) and a TGVN-style CSV."""
    g = np.random.default_rng(0)
    rows, maps = [], {}
    for i in range(n_pairs + int(bad_pair)):
        pd_name, fs_name = f"pd{i}.h5", f"pdfs{i}.h5"
        for name, acq in ((pd_name, "CORPD_FBK"), (fs_name, "CORPDFS_FBK")):
            k, sm = _vol(g)
            if bad_pair and i == n_pairs and acq == "CORPD_FBK":
                k = k[:, :, :, :-2]                       # different matrix size
            with h5py.File(os.path.join(root, name), "w") as f:
                f.create_dataset("kspace", data=k)
                f.attrs["acquisition"] = ("CORPDFS_FBK" if acq == "CORPD_FBK" else
                                          "CORPD_FBK") if swap else acq
            maps[name] = sm
        rows.append((pd_name, fs_name))
    csv = os.path.join(root, "split.csv")
    with open(csv, "w") as f:
        f.write("pd,pdfs\n" + "".join(f"{a},{b}\n" for a, b in rows))
    return csv, maps


def make_maps(root, out):
    """Offline maps + SENSE images for every PDFS file in `root`, laid out as
    scripts/make_espirit_smaps.py writes them (smaps (S, C, H, W), image (S, H, W))."""
    os.makedirs(out, exist_ok=True)
    for name in sorted(os.listdir(root)):
        if not name.startswith("pdfs"):
            continue
        with h5py.File(os.path.join(root, name), "r") as f:
            k = torch.from_numpy(f["kspace"][()])
        sm = torch.from_numpy(MAPS)[None].expand(k.shape[0], -1, -1, -1)
        img = (sm.conj() * ifftc(k)).sum(dim=1)
        with h5py.File(os.path.join(out, name), "w") as f:
            f.create_dataset("smaps", data=sm.numpy())
            f.create_dataset("image", data=img.numpy())
    return out


def raises(exc, fn, needle=""):
    try:
        fn()
    except exc as e:
        assert needle in str(e), f"{needle!r} not in {e}"
        return
    raise AssertionError(f"{exc.__name__} not raised")


class Spy(torch.nn.Module):
    """Wraps a net and records the guide each forward received."""
    def __init__(self, net):
        super().__init__()
        self.net, self.guides = net, []
        for a in ("guided", "guide_domain", "uses_operator_smaps", "pad_stride"):
            if hasattr(net, a):
                setattr(self, a, getattr(net, a))

    def forward(self, y, E=None, sigma=None, guide=None):
        self.guides.append(None if guide is None else tuple(guide.shape))
        return self.net(y, E=E, sigma=sigma) if guide is None else \
            self.net(y, E=E, sigma=sigma, guide=guide)


class FakeWandb:
    """Records every wandb.log call; images arrive through the module stub (or real wandb)."""
    def __init__(self):
        self.calls = []

    def log(self, d, step=None):
        self.calls.append(d)

    def last(self, key):
        return [c[key] for c in self.calls if key in c][-1]


def _raw_image(data, caption=None, **_):
    """Stands in for visualization.wandb_image.wandb_image: keeps the panel and caption."""
    return data, caption


tr.wandb_image = _raw_image             # both loops log panels through their own import
import training.i2sb_recon as _tir      # noqa: E402
_tir.wandb_image = _raw_image


def panel_tiles(img):
    """(tiles, caption) of a logged recon_panel strip: W-wide tiles, 2-px separators."""
    panel, caption = img
    return (panel.shape[-1] + 2) // (W + 2), caption


def run_recon(net, csv, root, d, data=None, **mri):
    loader = lambda: get_paired_knee_loader(csv, [root], batch_size=1, shuffle=False,
                                            drop_last=False, scale_fac=1.0, start_slice=1,
                                            end_slice=3, num_workers=0, **(data or {}))
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    kw = dict(R=4, acs_lines=4, kspace_type="measurement_awgn", noise_std=(0.0, 0.01),
              val_noise_std=0.005, val_seed=0, loss_type="magnitude-nl1-nl2", num_epochs=1,
              steps_per_epoch=2, val_every_epochs=1, save_dir=d)
    kw.update(mri)
    wb = FakeWandb()
    tr.train_recon(net, opt, None, loader(), loader(), "cpu", wandb=wb, **kw)
    return wb


# ---------------------------------------------------------------------------------------------
def test_vendored_tgvn_differs_from_upstream_only_in_imports():
    up = os.path.join(os.path.dirname(REPO), "TGVN", "src", "tgvn", "models.py")
    if not os.path.exists(up):
        print("    (skipped: no TGVN checkout beside the repo)")
        return
    ours = open(os.path.join(REPO, "models", "tgvn", "_upstream.py")).read().splitlines()
    ours = [l for l in ours if not l.startswith("#")]
    theirs = open(up).read().splitlines()
    assert len(ours) == len(theirs)
    diff = [(a, b) for a, b in zip(ours, theirs) if a != b]
    assert len(diff) == 2 and all(b.startswith("from fastmri") for _, b in diff), diff


def test_loader_pairs_drops_and_checks():
    with tempfile.TemporaryDirectory() as root:
        csv, _ = make_data(root)
        ds = PairedKneeDataset(csv, root, start_slice=1, end_slice=3)
        assert len(ds) == 3 and len(ds.dropped) == 1
        k, sm, img, organ, pad_hw, side = ds[0]
        assert k.shape == side.shape == (C, H, W) and img.shape == (1, H, W)
        assert torch.allclose(img.real, ifftc(k).abs().pow(2).sum(0, keepdim=True).sqrt())
        assert torch.allclose(sm.abs().pow(2).sum(0), torch.ones(H, W))   # flat placeholder
        assert organ.all() and pad_hw.tolist() == [H, W]
        en = PairedKneeDataset(csv, root, start_slice=1, end_slice=3, enumerate_slices=True)
        assert len(en) == 6 and en.item_id(1) == ("pdfs0", 2)
    with tempfile.TemporaryDirectory() as root:
        csv, _ = make_data(root, bad_pair=False, swap=True)
        raises(ValueError, lambda: PairedKneeDataset(csv, root), "pd column = side")


def test_loader_scales_each_contrast_by_its_own_factor():
    """side_scale_fac scales the guide (PD) alone; None keeps the shared factor."""
    with tempfile.TemporaryDirectory() as root:
        csv, _ = make_data(root)
        kw = dict(start_slice=1, end_slice=3, enumerate_slices=True)
        base = PairedKneeDataset(csv, root, scale_fac=1.0, **kw)[1]
        shared = PairedKneeDataset(csv, root, scale_fac=3.0, **kw)[1]
        split = PairedKneeDataset(csv, root, scale_fac=3.0, side_scale_fac=0.5, **kw)[1]
        close = lambda a, b: torch.allclose(a, b, atol=1e-5)               # noqa: E731
        assert close(shared[0], 3 * base[0]) and close(shared[5], 3 * base[5])
        assert close(split[0], 3 * base[0]) and close(split[2], 3 * base[2])
        assert close(split[5], 0.5 * base[5])


def test_loader_offline_sense_target():
    with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as mroot:
        csv, _ = make_data(root)
        make_maps(root, mroot)
        ds = PairedKneeDataset(csv, root, scale_fac=2.0, start_slice=1, end_slice=3,
                               enumerate_slices=True, target="sense", smap_root=mroot,
                               organ_mask_source="smaps")
        k, sm, img, organ, _, _ = ds[1]                                   # pdfs0, slice 2
        with h5py.File(os.path.join(mroot, "pdfs0.h5"), "r") as f:
            want = torch.from_numpy(f["image"][2])[None] * 2.0
        assert torch.allclose(img, want)                                  # stored, scaled
        assert torch.allclose(sm, torch.from_numpy(MAPS))                 # real maps
        assert torch.equal(organ, sm.abs().sum(0, keepdim=True) > 0)      # map support
        rss = ifftc(k).abs().pow(2).sum(0, keepdim=True).sqrt()
        assert not torch.allclose(img.abs(), rss)                         # not the RSS
        raises(ValueError, lambda: PairedKneeDataset(csv, root, target="sense"), "smap_root")
        raises(ValueError, lambda: PairedKneeDataset(csv, root, organ_mask_source="smaps"),
               "offline maps")
        os.remove(os.path.join(mroot, "pdfs1.h5"))
        raises(FileNotFoundError, lambda: PairedKneeDataset(csv, root, target="sense",
                                                            smap_root=mroot), "pdfs1.h5")


def test_train_recon_on_the_sense_target():
    with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as mroot,             tempfile.TemporaryDirectory() as d:
        csv, _ = make_data(root)
        make_maps(root, mroot)
        spy = Spy(TGVN(**TINY_TGVN))
        wb = run_recon(spy, csv, root, d, data=dict(target="sense", smap_root=mroot))
        assert spy.guides and os.path.exists(os.path.join(d, "net.ckpt"))
        assert any("val/psnr" in c for c in wb.calls)


def test_tgvn_on_operator_maps_uses_online_estimate():
    """The shipped TGVN cells (use_smaps + cond_sigma) train on the operator's ONLINE maps: the
    loop must estimate them, and refuse without online_smaps."""
    import physics.online_smaps as osm
    with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as mroot,             tempfile.TemporaryDirectory() as d:
        csv, _ = make_data(root)
        make_maps(root, mroot)
        net = TGVN(use_smaps=True, cond_sigma=True, **TINY_TGVN)
        assert net.uses_operator_smaps and net.net.sens_net is None
        data = dict(target="sense", smap_root=mroot)
        raises(ValueError, lambda: run_recon(net, csv, root, d, data=data), "online")
        calls = []
        real = osm.online_smaps
        osm.online_smaps = lambda y, mask, **kw: (calls.append(1),
                                                  torch.from_numpy(MAPS)[None].expand_as(y))[1]
        try:
            wb = run_recon(net, csv, root, d, data=data, online_smaps="espirit")
        finally:
            osm.online_smaps = real
        assert calls and any("val/psnr" in c for c in wb.calls)


def test_tgvn_adapter_guided_and_baseline():
    with tempfile.TemporaryDirectory() as root:
        csv, _ = make_data(root)
        k, _, _, _, _, side = PairedKneeDataset(csv, root)[0]
    k, side = k[None], side[None]
    mask = get_mask_cached(k[:, :1], R=4, acs_lines=4, mode="uniform")
    E = Mask(mask) @ FFT2D() @ Sense(torch.ones(1, C, H, W, dtype=torch.complex64))
    y = mask * k
    out, _ = TGVN(**TINY_TGVN)(y, E=E, guide=side)
    assert out.shape == (1, 1, H, W) and not out.is_complex() and (out >= 0).all()
    raises(ValueError, lambda: TGVN(**TINY_TGVN)(y, E=E), "side k-space")
    base = TGVN(guided=False, **TINY_TGVN)
    assert base(y, E=E)[0].shape == (1, 1, H, W)
    raises(ValueError, lambda: base(y, E=E, guide=side), "no-side-information")


def test_guide_routing():
    k = torch.randn(1, C, H, W, dtype=torch.complex64)
    sm = torch.randn(1, C, H, W, dtype=torch.complex64)
    from operators.truncate import Truncate
    T = Truncate((H, W), (H, W))
    assert tr._guide(TGVN(**TINY_TGVN), k, sm, T, "cpu") is k                    # k-space
    assert tr._guide(TGVN(guided=False, **TINY_TGVN), k, sm, T, "cpu") is None   # baseline
    img_net = types.SimpleNamespace(forward=lambda y, guide=None, E=None, sigma=None: None)
    g = tr._guide(img_net, k, sm, T, "cpu")
    assert torch.allclose(g, Sense(sm).adjoint(ifftc(k)))                        # side image
    raises(ValueError, lambda: tr._guide(img_net, None, sm, T, "cpu"), "no side information")
    unguided = types.SimpleNamespace(forward=lambda y, E=None, sigma=None: None)
    assert tr._guide(unguided, k, sm, T, "cpu") is None
    raises(ValueError, lambda: tr._check_paired(unguided, "simulated", "espirit"), "simulated")
    raises(ValueError, lambda: tr._check_paired(unguided, "measurement_awgn", None), "online")
    tr._check_paired(TGVN(**TINY_TGVN), "measurement_awgn", None)       # own maps: fine


def test_train_recon_end_to_end_tgvn_and_baseline():
    with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as d:
        csv, _ = make_data(root)
        spy = Spy(TGVN(**TINY_TGVN))
        run_recon(spy, csv, root, d)
        assert spy.guides and all(g == (1, C, H, W) for g in spy.guides), spy.guides
        assert os.path.exists(os.path.join(d, "net.ckpt"))
        spy0 = Spy(TGVN(guided=False, **TINY_TGVN))
        run_recon(spy0, csv, root, d)
        assert spy0.guides and all(g is None for g in spy0.guides)


def test_train_recon_end_to_end_lggs_gets_the_side_image():
    from models.guided_lpds import LGGSNet
    import physics.online_smaps as osm
    with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as d:
        csv, maps = make_data(root)
        real = osm.online_smaps
        osm.online_smaps = lambda y, mask, **kw: torch.full_like(y, C ** -0.5)  # no LAPACK here
        try:
            net = LGGSNet(K=2, M=8, P=3, s=2, Mh=4, guide_window=3, attn_backend="gather")
            spy = Spy(net)
            run_recon(spy, csv, root, d, online_smaps="espirit")
        finally:
            osm.online_smaps = real
        assert spy.guides and all(g == (1, 1, H, W) for g in spy.guides), spy.guides


def test_gt_max_metrics_are_per_slice_and_scale_free():
    """metric_range "gt_max": PSNR against each slice's own peak, averaged over slices, so a
    global intensity scale (scale_fac) cannot move it; the fixed range-1 PSNR moves by
    -20 log10(scale)."""
    from training.metrics import compute_metrics
    g = torch.Generator().manual_seed(0)
    gt = torch.rand(3, 1, 32, 32, generator=g) * torch.tensor([0.2, 0.5, 1.0]).view(3, 1, 1, 1)
    pred = gt + 0.01 * torch.randn(gt.shape, generator=g)
    m = compute_metrics(gt, pred, data_range="gt_max")
    want = sum(float(10 * torch.log10(gt[b].max() ** 2 / (gt[b] - pred[b]).pow(2).mean()))
               for b in range(3)) / 3
    assert abs(float(m["psnr"]) - want) < 1e-3, (float(m["psnr"]), want)
    m3 = compute_metrics(3 * gt, 3 * pred, data_range="gt_max")
    for k in ("psnr", "ssim", "nrmse"):
        assert abs(float(m[k]) - float(m3[k])) < 1e-4, k
    f1, f3 = compute_metrics(gt, pred)["psnr"], compute_metrics(3 * gt, 3 * pred)["psnr"]
    assert abs(float(f1 - f3) - 20 * math.log10(3)) < 1e-3
    mask = torch.zeros(3, 1, 32, 32, dtype=torch.bool)
    mask[..., 4:28, 4:28] = True
    assert torch.isfinite(compute_metrics(gt, pred, data_range="gt_max", mask=mask)["psnr"])
    try:
        compute_metrics(gt, pred, data_range="volume_max")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown data_range accepted")


def test_generator_cells_and_config_keys():
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    import make_guided_recon_configs as gen
    from models.guided_lpds import lggs_defaults
    from training.i2sb_recon import train_i2sb_recon
    assert gen.LGGS_PARAMS == lggs_defaults(attn_backend="flex")[0]
    names = [gen.cell_name(*c) for c in gen.cells()]
    assert names == ["tgvn_R8_g1", "tgvn_R8_g2", "sbtgvn_R8_g1", "sbtgvn_R8_g2", "lpdsnet_R8"]
    root = os.path.join(REPO, "config", "knee", "guided")
    if not os.path.exists(os.path.join(root, "tgvn_R8_g1.json")):
        print("    (skipped key check: run scripts/make_guided_recon_configs.py first)")
        return
    loops = {"recon": (("training", "mri", "paths"), tr.train_recon),
             "i2sb_recon": (("training", "mri", "i2sb", "paths"), train_i2sb_recon)}
    import yaml
    for name in names:
        with open(os.path.join(root, f"{name}.json")) as f:
            cfg = yaml.safe_load(f)                 # how train.py reads it: 1e-06 is a STRING
        assert isinstance(cfg["scheduler"]["params"]["eta_min"], float),             f"{name}: eta_min loads as {cfg['scheduler']['params']['eta_min']!r}"
        blocks, fn = loops[cfg["task"]]
        params = inspect.signature(fn).parameters
        seen = set()
        for block in blocks:
            for k in cfg[block]:
                assert k in params, f"{name}: {block}.{k} is not a {fn.__name__} argument"
                assert k not in seen, f"{name}: {k} in two blocks"
                seen.add(k)
        assert cfg["data"]["train"]["name"] == "fastmri_paired"
        assert set(inspect.signature(get_paired_knee_loader).parameters) >=             set(cfg["data"]["train"]) - {"name"}
        if cfg["task"] == "i2sb_recon":
            for k in ("kind", "tau", "n_points", "beta_max"):
                assert cfg["model"]["params"][k] == cfg["i2sb"][k], (name, k)
            assert isinstance(cfg["i2sb"]["dc_lam"], float), (name, cfg["i2sb"]["dc_lam"])
            assert cfg["i2sb"]["dc_iters"] > 0, name
        assert cfg["mri"]["guide_R"] == (2 if name.endswith("_g2") else 1)
        assert cfg["training"]["metric_range"] == "gt_max", name
    for n in gen.TGVN_CASCADE_SWEEP:
        p = gen.MODELS[f"tgvn_c{n}"]["model"]["params"]
        assert p["num_cascades"] == n and p["chans"] == 18 and p["sens_chans"] == 8


def test_guide_undersampling():
    side = torch.randn(1, C, H, W, dtype=torch.complex64) + 1
    assert tr.undersample_guide(side, 1) is side
    g = tr.undersample_guide(side, 2, guide_offset=0)
    cols = (g[0, 0] != 0).any(dim=0)
    assert cols.tolist() == [i % 2 == 0 for i in range(W)]            # every 2nd line, no ACS
    phases = {tuple((tr.undersample_guide(side, 2, guide_offset="random")[0, 0] != 0)
                    .any(dim=0).tolist()) for _ in range(20)}
    assert len(phases) == 2                                          # redrawn per call
    sm = torch.randn(1, C, H, W, dtype=torch.complex64)
    from operators.truncate import Truncate
    k = tr._guide(TGVN(**TINY_TGVN), side, sm, Truncate((H, W), (H, W)), "cpu", 2, 0,
                  "uniform", 0)
    assert torch.equal(k, g)


def test_cond_plane_backward():
    """Backward through a noise-conditioned plane, with the scale DEPENDING on the input (as it
    does through a real Cholesky). An in-place rescale of the plane failed exactly here on the
    cluster: "modified by an inplace operation ... AsStridedBackward0"."""
    from models.tgvn import CondNormUnet
    torch.manual_seed(0)
    u = CondNormUnet(4, 2, n_cond=2, noise_channels=(1,))
    x = torch.randn(1, 1, H, W, 2, requires_grad=True)
    u.cond = torch.tensor([[0.3, 0.05]])
    u(x).pow(2).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is not None for p in u.parameters() if p.requires_grad)


def test_cond_plane_is_scale_equivariant():
    from models.tgvn import CondNormUnet
    torch.manual_seed(0)
    u = CondNormUnet(4, 2, n_cond=2, noise_channels=(1,)).eval()
    x = torch.randn(2, 1, H, W, 2)
    cond = torch.tensor([[0.3, 0.05], [-0.2, 0.02]])
    with torch.no_grad():
        u.cond = cond
        out = u(x)
        u.cond = cond * torch.tensor([1.0, 7.0])                      # sigma x 7, position kept
        out7 = u(7.0 * x)
        u.cond = cond * torch.tensor([1.0, 3.0])                      # sigma moves: output moves
        out3 = u(x)
    assert torch.allclose(out7, 7.0 * out, rtol=1e-3, atol=1e-4)
    assert not torch.allclose(out3, out, atol=1e-5)
    u.cond = None
    raises(RuntimeError, lambda: u(x), "cond")


def _tiny_batch():
    with tempfile.TemporaryDirectory() as root:
        csv, _ = make_data(root)
        k, _, _, _, _, side = PairedKneeDataset(csv, root, scale_fac=1.0)[0]
    k, side = k[None], side[None]
    mask = get_mask_cached(k[:, :1], R=4, acs_lines=4, mode="uniform")
    return mask * k, side, Mask(mask) @ FFT2D() @ Sense(torch.from_numpy(MAPS)[None])


def test_tgvn_cond_and_known_maps():
    y, side, E = _tiny_batch()
    net = TGVN(cond_sigma=True, **TINY_TGVN)
    raises(ValueError, lambda: net(y, E=E, guide=side), "sigma")
    assert net(y, E=E, sigma=torch.tensor([0.01]), guide=side)[0].shape == (1, 1, H, W)
    km = TGVN(use_smaps=True, output="complex", **TINY_TGVN)
    assert km.net.sens_net is None and km.uses_operator_smaps
    out = km(y, E=E, guide=side)[0]
    assert out.is_complex() and out.shape == (1, 1, H, W)
    raises(ValueError, lambda: TGVN(output="complex", **TINY_TGVN), "use_smaps")


def test_bridge_tgvn_forward():
    from models.tgvn import BridgeTGVN
    from sb.base import build_schedule
    y, side, E = _tiny_batch()
    net = BridgeTGVN(tau=0.1, **TINY_TGVN)
    net.assert_schedule_matches(build_schedule(kind="brownian", tau=0.1, n_points=1000))
    x_t = E.adjoint(y)
    out, _ = net(x_t, sigma=build_schedule(tau=0.1).std_fwd[[500]], y=y, E=E,
                 sigma_n=torch.tensor([0.01]), guide=side)
    assert out.is_complex() and out.shape == (1, 1, H, W)
    raises(ValueError, lambda: net(x_t, sigma=None, y=y, E=E, sigma_n=0.01, guide=side),
           "bridge position")


class BridgeSpy(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net, self.guides = net, []
        for a in ("guided", "guide_domain", "uses_operator_smaps", "pad_stride"):
            setattr(self, a, getattr(net, a))

    def assert_schedule_matches(self, sched):
        self.net.assert_schedule_matches(sched)

    def forward(self, x_t, sigma=None, x1=None, y=None, E=None, sigma_n=None, guide=None):
        self.guides.append(guide.detach().clone())
        return self.net(x_t, sigma=sigma, x1=x1, y=y, E=E, sigma_n=sigma_n, guide=guide)


def test_val_panel_shows_the_guides():
    """val/recon_example = zf | recon | gt | residual, + the guide as seen, + the fully-sampled
    guide when it was accelerated; nothing extra for an unguided net."""
    with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as d:
        csv, _ = make_data(root)
        for net, extra, tiles in ((TGVN(**TINY_TGVN), {}, 5),
                                  (TGVN(**TINY_TGVN), {"guide_R": 2}, 6),
                                  (TGVN(guided=False, **TINY_TGVN), {}, 4)):
            wb = run_recon(net, csv, root, d, **extra)
            n, cap = panel_tiles(wb.last("val/recon_example"))
            assert n == tiles, (extra, n, tiles)
            assert ("guide" in cap) == (tiles > 4), cap
    seen = torch.randn(1, C, H, W, dtype=torch.complex64)
    tl = tr.guide_display(TGVN(**TINY_TGVN), seen, seen, 1)
    assert [l for l, _ in tl] == ["guide"] and tl[0][1].shape == (1, 1, H, W)
    assert torch.allclose(tl[0][1], ifftc(seen).abs().pow(2).sum(1, keepdim=True).sqrt())
    assert tr.guide_display(TGVN(**TINY_TGVN), None, seen, 1) == []


def test_intensity_relative_bridge_noise():
    """kappa scales the injected noise and nothing else, forward and reverse; the slice scale is
    scale-equivariant; a relative bridge without tau_ref is refused."""
    from sb.base import build_schedule, bridge_coeffs, forward_sample, reverse_sample
    from sb.i2sb_recon import bridge_noise_scale, slice_scale
    sched = build_schedule(kind="brownian", tau=0.03, n_points=1000)
    x0 = torch.randn(2, 1, 64, 64, dtype=torch.complex64)
    step = torch.full((2,), 500, dtype=torch.long)
    kappa = torch.tensor([0.5, 3.0])
    torch.manual_seed(0)
    xt = forward_sample(sched, step, x0, x0, noise_scale=kappa)
    std = float(bridge_coeffs(sched)[2][500])
    emp = (xt - x0).abs().pow(2).flatten(1).mean(1).sqrt()
    assert torch.allclose(emp, kappa * std, rtol=0.03), (emp, kappa * std)
    det = forward_sample(sched, step, x0, 2 * x0, deterministic=True, noise_scale=kappa)
    assert torch.allclose(det, forward_sample(sched, step, x0, 2 * x0, deterministic=True))
    # reverse: with a regressor that ignores x_t the state is linear in the injected noise, so
    # (noisy - noise-free) scales by exactly kappa under the same seed
    oracle = lambda x_t, n: x0                                              # noqa: E731
    run = lambda **kw: reverse_sample(sched, oracle, 2 * x0, nfe=8, verbose=False, **kw)[0]
    det = run(deterministic=True)
    assert torch.allclose(det, run(deterministic=True, noise_scale=kappa))
    torch.manual_seed(1)
    noisy1 = run()
    torch.manual_seed(1)
    noisyk = run(noise_scale=kappa)
    assert torch.allclose(noisyk - det, kappa.view(2, 1, 1, 1) * (noisy1 - det), atol=1e-5)
    # slice scale
    img = torch.zeros(2, 1, H, W, dtype=torch.complex64)
    img[:, :, 8:24, 8:24] = 1.0 + 0.1 * torch.randn(2, 1, 16, 16)
    s = slice_scale(img)
    assert torch.allclose(slice_scale(3.0 * img), 3.0 * s, rtol=1e-4)
    assert abs(float(s[0]) - 1.0) < 0.05                      # the foreground's RMS, not the frame's
    assert bridge_noise_scale(img) is None
    assert torch.allclose(bridge_noise_scale(img, "x1_rms", 2.0), s / 2.0)
    raises(ValueError, lambda: bridge_noise_scale(img, "x1_rms", None), "tau_ref")


def test_i2sb_recon_end_to_end_bridge_tgvn_with_2x_guide():
    from models.tgvn import BridgeTGVN
    from training.i2sb_recon import train_i2sb_recon
    import physics.online_smaps as osm
    with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as d:
        csv, _ = make_data(root)
        loader = lambda: get_paired_knee_loader(csv, [root], batch_size=1, shuffle=False,
                                                drop_last=False, scale_fac=1.0, start_slice=1,
                                                end_slice=3, num_workers=0)
        spy = BridgeSpy(BridgeTGVN(tau=0.1, **TINY_TGVN))
        wb = FakeWandb()
        real = osm.online_smaps
        osm.online_smaps = lambda y, mask, **kw: torch.from_numpy(MAPS)[None].expand_as(y)
        try:
            train_i2sb_recon(spy, torch.optim.Adam(spy.parameters(), lr=1e-4), None, "cpu",
                             loader(), loader(), num_epochs=1, steps_per_epoch=2,
                             val_every_epochs=1, R=4, acs_lines=4,
                             kspace_type="measurement_awgn", online_smaps="espirit",
                             noise_std=(0.0, 0.01), val_noise_std=0.005, val_seed=0,
                             loss_type="complex-mse", tau=0.1, val_mode="full_recon",
                             val_nfe=2, guide_R=2, save_dir=d, wandb=wb,
                             tau_scale="x1_rms", tau_ref=0.5)
        finally:
            osm.online_smaps = real
        assert len(spy.guides) >= 4                       # 2 train steps + val sampler calls
        for g in spy.guides:
            cols = (g[0, 0] != 0).any(dim=0)
            assert int(cols.sum()) == W // 2, int(cols.sum())          # half the lines
        assert os.path.exists(os.path.join(d, "net_bestval.ckpt"))
        n, cap = panel_tiles(wb.last("val/recon_example"))
        assert n == 6 and "guide 2x zero-filled" in cap, (n, cap)   # + guide 2x, + full guide
        raises(ValueError, lambda: train_i2sb_recon(
            spy, torch.optim.Adam(spy.parameters()), None, "cpu", loader(), None, tau=0.1,
            tau_scale="x1_rms", save_dir=d), "tau_ref")


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
            traceback.print_exc(limit=8)
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
