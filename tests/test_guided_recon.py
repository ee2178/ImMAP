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
TGVN's Cholesky normalization then goes through numpy, forward-exact but not differentiated).
"""
import inspect
import json
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
    torch.linalg.cholesky = lambda A: torch.from_numpy(np.linalg.cholesky(A.detach().cpu().numpy())).to(A)
    torch.linalg.inv = lambda A: torch.from_numpy(np.linalg.inv(A.detach().cpu().numpy())).to(A)
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


def run_recon(net, csv, root, d, **mri):
    loader = lambda: get_paired_knee_loader(csv, [root], batch_size=1, shuffle=False,
                                            drop_last=False, scale_fac=1.0, start_slice=1,
                                            end_slice=3, num_workers=0)
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    kw = dict(R=4, acs_lines=4, kspace_type="measurement_awgn", noise_std=(0.0, 0.01),
              val_noise_std=0.005, val_seed=0, loss_type="magnitude-nl1-nl2", num_epochs=1,
              steps_per_epoch=2, val_every_epochs=1, save_dir=d)
    kw.update(mri)
    return tr.train_recon(net, opt, None, loader(), loader(), "cpu", **kw)


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
    for name in names:
        with open(os.path.join(root, f"{name}.json")) as f:
            cfg = json.load(f)
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
        assert cfg["mri"]["guide_R"] == (2 if name.endswith("_g2") else 1)


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
        real = osm.online_smaps
        osm.online_smaps = lambda y, mask, **kw: torch.from_numpy(MAPS)[None].expand_as(y)
        try:
            train_i2sb_recon(spy, torch.optim.Adam(spy.parameters(), lr=1e-4), None, "cpu",
                             loader(), loader(), num_epochs=1, steps_per_epoch=2,
                             val_every_epochs=1, R=4, acs_lines=4,
                             kspace_type="measurement_awgn", online_smaps="espirit",
                             noise_std=(0.0, 0.01), val_noise_std=0.005, val_seed=0,
                             loss_type="complex-mse", tau=0.1, val_mode="full_recon",
                             val_nfe=2, guide_R=2, save_dir=d)
        finally:
            osm.online_smaps = real
        assert len(spy.guides) >= 4                       # 2 train steps + val sampler calls
        for g in spy.guides:
            cols = (g[0, 0] != 0).any(dim=0)
            assert int(cols.sum()) == W // 2, int(cols.sum())          # half the lines
        assert os.path.exists(os.path.join(d, "net_bestval.ckpt"))


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
