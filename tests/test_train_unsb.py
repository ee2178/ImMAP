# -*- coding: utf-8 -*-
"""The UNSB training loop (training/unsb.py), run for a few steps on synthetic data.

Run with `python -m tests.test_train_unsb` or pytest.

Pinned, because each fails silently in a long GAN run:
  * the loop actually updates G, D, E and the NCE head, and writes the three checkpoints it
    promises (net.ckpt = resume, unsb_nets.ckpt = critics, net_best.ckpt = best val PSNR);
  * a resume restores the critics (losing them restarts the adversarial game);
  * the cond-swap panel and the image panels are logged, and the gan/ scalars are NOT unless asked;
  * the guard rails: prior_idx pointing at the wrong cond channel, entropy at batch 1,
    ReduceLROnPlateau, lambda_gan = 0, and a critics checkpoint from a different config.

Runs locally on torch 1.12 with the stubs below (no torchvision / wandb / flex attention there);
on the cluster the real packages are used and the stubs never install.
"""
import contextlib
import io
import os
import sys
import tempfile
import types

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import inspect  # noqa: E402

if "weights_only" not in inspect.signature(torch.load).parameters:    # torch 1.12 (local only)
    _torch_load = torch.load      # the repo's own load_ckpt passes weights_only=False to torch 2.x
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

from sb.base import build_schedule                                         # noqa: E402
from training.common import load_ckpt                                      # noqa: E402
from training.unsb import train_unsb                                       # noqa: E402

TAU, N, H = 0.05, 1000, 32
STEPS = 3


class FakeWandb:
    def __init__(self):
        self.calls = []

    def log(self, d, step=None):
        self.calls.append((step, d))

    def keys(self):
        return {k for _, d in self.calls for k in d}


def loader(n=8, bs=4, seed=0, shuffle=False):
    g = torch.Generator().manual_seed(seed)
    x0, x1 = torch.rand(n, 1, H, H, generator=g), torch.rand(n, 1, H, H, generator=g)
    flair, t2 = torch.rand(n, 1, H, H, generator=g), torch.rand(n, 1, H, H, generator=g)
    cond = torch.cat([flair, x1, t2], dim=1)            # source T1 at index 1, like cond_idx [0,1,3]
    ds = TensorDataset(x0, x1, cond, torch.ones(n, 1, H, H))
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=shuffle)


def make_unet(seed=0):
    from models.sb_unet import SBUnet
    torch.manual_seed(seed)
    return SBUnet(C=4, model_channels=32, num_res_blocks=1, channel_mult=(1, 2),
                  attention_resolutions=(8,), image_size=16, num_head_channels=16,
                  kind="brownian", tau=TAU, n_points=N)


def make_cdl(prior_idx=1, seed=0):
    from models.sb_cdlnet import SBCDLNet
    torch.manual_seed(seed)
    return SBCDLNet(K=2, M=8, P=3, s=2, C=4, prior_idx=prior_idx, kind="brownian", tau=TAU,
                    n_points=N)


def opt_sched(net):
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, betas=(0.5, 0.999))
    return opt, torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)


def run(net, save_dir, num_epochs=2, start_epoch=0, wb=None, opt=None, sched=None, bs=4, **kw):
    if opt is None:
        opt, sched = opt_sched(net)
    base = dict(num_stages=3, kind="brownian", tau=TAU, n_points=N, ndf=8, nce_patches=32,
                steps_per_epoch=STEPS, val_every_epochs=1, save_every_epochs=1, val_seed=0,
                data_range=1.0)
    base.update(kw)
    return train_unsb(net, opt, sched, torch.device("cpu"),
                      loader(bs=bs, shuffle=True), loader(), wandb=wb, num_epochs=num_epochs,
                      start_epoch=start_epoch, save_dir=save_dir, **base), opt, sched


def snapshot(net):
    return [p.detach().clone() for p in net.parameters()]


def changed(net, snap):
    return any(not torch.equal(a, b) for a, b in zip(net.parameters(), snap))


def raises(exc, fn, needle=""):
    try:
        fn()
    except exc as e:
        assert needle in str(e), f"{needle!r} not in {e}"
        return
    raise AssertionError(f"{exc.__name__} not raised")


# ---------------------------------------------------------------------------------------------
def test_sbunet_loop_updates_everything_writes_checkpoints_and_resumes():
    with tempfile.TemporaryDirectory() as d:
        net, wb = make_unet(), FakeWandb()
        before = snapshot(net)
        net, opt, sched = run(net, d, wb=wb, ema_decay=0.9, nce=True, nce_extractor="tap",
                              nce_layers=["input_blocks.1", "input_blocks.3"], entropy=True,
                              cond_d=True)
        assert changed(net, before), "G did not train"
        for f in ("net.ckpt", "unsb_nets.ckpt", "net_best.ckpt"):
            assert os.path.exists(os.path.join(d, f)), f
        st = torch.load(os.path.join(d, "unsb_nets.ckpt"), weights_only=False)
        assert st["E"] is not None and st["pnce"] is not None and st["encoder"] is None
        assert st["meta"]["cond_d"] and st["meta"]["nce_extractor"] == "tap"
        # every adversarial/NCE optimizer actually stepped once per training step ...
        for key in ("opt_d", "opt_e", "opt_aux"):
            steps = {int(float(v["step"])) for v in st[key]["state"].values()}
            assert steps == {2 * STEPS}, (key, steps)
        # ... and the critics' lr FOLLOWS the regressor's cosine schedule (same ratio to its base)
        assert opt.param_groups[0]["lr"] < 1e-3
        for key in ("opt_d", "opt_e", "opt_aux"):
            assert abs(st[key]["param_groups"][0]["lr"] - opt.param_groups[0]["lr"]) < 1e-12, key
        keys = wb.keys()
        assert {"val/psnr", "val/ssim", "val/nrmse", "val/example", "val/residual",
                "val/cond_swap", "train/loss", "train/psnr"} <= keys, keys
        assert not any(k.startswith("gan/") for k in keys), "gan/ scalars logged unasked"

        # resume exactly as train.py does: weights + optimizer + scheduler from net.ckpt first
        net2 = make_unet(seed=1)
        opt2, sched2 = opt_sched(net2)
        _, _, _, start_step = load_ckpt(os.path.join(d, "net.ckpt"), model=net2, optimizer=opt2,
                                        scheduler=sched2)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            run(net2, d, num_epochs=3, start_epoch=start_step // STEPS, wb=FakeWandb(), opt=opt2,
                sched=sched2, ema_decay=0.9, nce=True, nce_extractor="tap",
                nce_layers=["input_blocks.1", "input_blocks.3"], entropy=True, cond_d=True)
        assert "resumed D/E/NCE" in buf.getvalue(), buf.getvalue()[-400:]


def test_sbcdlnet_with_own_encoder_unconditional_D_and_shuffled_pairing():
    with tempfile.TemporaryDirectory() as d:
        net, wb = make_cdl(), FakeWandb()
        before = snapshot(net)
        run(net, d, num_epochs=1, wb=wb, nce=True, nce_extractor="encoder", entropy=False,
            cond_d=False, pairing="shuffle", log_gan_terms=True, lambda_sb=0.5)
        assert changed(net, before)
        st = torch.load(os.path.join(d, "unsb_nets.ckpt"), weights_only=False)
        assert st["E"] is None and st["encoder"] is not None and not st["meta"]["cond_d"]
        assert any(k.startswith("gan/") for k in wb.keys()), "log_gan_terms had no effect"


def test_gan_only_runs_with_nothing_but_the_discriminator():
    with tempfile.TemporaryDirectory() as d:
        net = make_unet()
        run(net, d, num_epochs=1, nce=False, entropy=False, lambda_sb=0.0, cond_d=True)
        st = torch.load(os.path.join(d, "unsb_nets.ckpt"), weights_only=False)
        assert st["pnce"] is None and st["E"] is None and st["opt_aux"] is None


def test_guard_rails():
    with tempfile.TemporaryDirectory() as d:
        # prior_idx must index the source INSIDE cond (here T1 is cond[1], not cond[0])
        raises(ValueError, lambda: run(make_cdl(prior_idx=0), d, num_epochs=1), "prior_idx")
        # the entropy critic needs negatives from the rest of the batch
        raises(ValueError, lambda: run(make_unet(), d, num_epochs=1, bs=1, entropy=True),
               "batch_size >= 2")
        raises(ValueError, lambda: run(make_unet(), d, num_epochs=1, lambda_gan=0.0), "lambda_gan")
        net = make_unet()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(opt)
        raises(ValueError, lambda: run(net, d, num_epochs=1, opt=opt, sched=plateau),
               "ReduceLROnPlateau")
        raises(ValueError, lambda: run(make_unet(), d, num_epochs=1, nce=True,
                                       nce_extractor="tap"), "nce_layers")


def test_resume_refuses_critics_from_a_different_config():
    with tempfile.TemporaryDirectory() as d:
        run(make_unet(), d, num_epochs=1, nce=False, entropy=True, cond_d=True)
        net2 = make_unet()
        opt2, sched2 = opt_sched(net2)
        load_ckpt(os.path.join(d, "net.ckpt"), model=net2, optimizer=opt2, scheduler=sched2)
        raises(ValueError, lambda: run(net2, d, num_epochs=2, start_epoch=1, opt=opt2,
                                       sched=sched2, nce=False, entropy=True, cond_d=False),
               "would not fit")


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
