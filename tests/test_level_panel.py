"""
`training.log_levels`: the val/levels panel of training/recon.py.

Run with `python -m tests.test_level_panel`. CPU, a few seconds.

1. `level_images` (models/level_trace.py): one image per grid level, finest
   first, halving; the finest + DC IS the output; every level matches
   scripts/level_figure.py::mg_levels (the notebook's figures); the planar
   state gives the same images; a flat net has none;
2. `upsample_repeat` / `levels_panel` (visualization/image.py): pixel
   repetition, coarsest -> finest -> ground truth, on the ground truth's scale;
3. `train_recon(log_levels=True)` end to end on a tiny MGLPDS net with a fake
   wandb: val/levels is logged at each validation, its finest tile is the
   recon tile of val/recon_example (same crop, same mask), the caption names
   every level; off by default; a net without levels logs one notice, no panel.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
import types

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

FAIL = []


def check(name, ok, detail=""):
    print(f"[{'ok ' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def operator(N=32, coils=4, seed=0):
    from operators import FFT2D, Mask, Sense
    g = torch.Generator().manual_seed(seed)
    smaps = torch.randn(1, coils, N, N, dtype=torch.complex64, generator=g) + 1.0
    smaps = smaps / smaps.abs().pow(2).sum(1, keepdim=True).sqrt()
    mask = torch.zeros(1, 1, N, N)
    mask[..., ::2] = 1.0
    mask[..., N // 2 - 4:N // 2 + 4] = 1.0
    return Mask(mask) @ FFT2D() @ Sense(smaps)


def mg_net(K=(2, [2, 2, 2]), **kw):
    from models.mg_lpds import MGLPDSNet
    torch.manual_seed(0)
    return MGLPDSNet(K=list(K) if not isinstance(K, int) else K, M=4, C=1, P=3, s=2,
                     is_complex=True, preproc="kspace", lam0=1e-3, tau0=0.5, theta0=0.0,
                     alpha0=0.3, **kw)


def test_level_images():
    import level_figure as lf
    from models.level_trace import level_images, trace_forward
    from models.mg_lpds import MGLPDSNet
    E = operator()
    img = torch.randn(1, 1, 32, 32, dtype=torch.complex64,
                      generator=torch.Generator().manual_seed(1)) + 2.0
    y, sig = E(img), torch.full((1, 1, 1, 1), 0.01)
    net = mg_net().eval()
    out, imgs = net.forward_level_images(y, E=E, sigma=sig)
    x_hat = out[0]
    check("one image per grid level, finest first, halving",
          imgs is not None and [tuple(x.shape[-2:]) for x in imgs] == [(32, 32), (16, 16), (8, 8)],
          str(None if imgs is None else [tuple(x.shape[-2:]) for x in imgs]))
    check("the finest level + DC is the output",
          float((imgs[0] - x_hat).abs().max() / x_hat.abs().max()) < 1e-5)
    _, tr = trace_forward(net, y, E=E, sigma=sig)
    npz, _, _ = lf.mg_levels(tr, x_hat, x_hat, torch.ones(1, 1, 32, 32), lf._transfer_kws(net))
    o = tr.outers[-1]
    gaps = [float(np.abs(npz[f"mg/x/o{o}/l{l}"] - imgs[l][0, 0].numpy()).max()
                  / np.abs(npz[f"mg/x/o{o}/l{l}"]).max()) for l in range(3)]
    check("every level matches the notebook's level figure (mg_levels)", max(gaps) < 1e-5,
          str([f"{g:.1e}" for g in gaps]))
    prev = MGLPDSNet.PLANAR_STATE
    try:
        MGLPDSNet.PLANAR_STATE = True
        planar = mg_net().eval()
        active = planar.planar_state_active()
        _, imgs_p = planar.forward_level_images(y, E=E, sigma=sig)
    finally:
        MGLPDSNet.PLANAR_STATE = prev
    gp = max(float((a - b).abs().max() / b.abs().max()) for a, b in zip(imgs_p, imgs))
    check("planar state: the same images (complex out)", active and gp < 1e-4
          and all(torch.is_complex(a) for a in imgs_p), f"active={active} gap {gp:.1e}")

    from training.recon import _levels_log
    flat = mg_net(K=6).eval()
    check("a flat net has no level images", _levels_log(
        flat, y, E, sig, _Identity(), None, None, img) is None)


class _Identity:
    @staticmethod
    def forward(x):
        return x


def test_panel():
    from visualization.image import levels_panel, upsample_repeat
    x = torch.arange(6.0).reshape(1, 1, 2, 3)
    u = upsample_repeat(x, (4, 6))
    check("upsample_repeat repeats each pixel 2x2",
          u.shape == (1, 1, 4, 6) and torch.equal(u[0, 0, ::2, ::2], x[0, 0])
          and torch.equal(u[0, 0, 1::2, 1::2], x[0, 0]))
    try:
        upsample_repeat(x, (5, 6))
        check("upsample_repeat refuses a non-whole factor", False)
    except ValueError:
        check("upsample_repeat refuses a non-whole factor", True)
    g = torch.rand(1, 1, 8, 8) + 0.5
    lv = [g * 0.9, g * 0.5, g * 0.25]                      # finest first
    p = levels_panel(lv, g)
    W = 8
    tiles = [p[0, :, k * (W + 2):k * (W + 2) + W] for k in range(4)]
    ref = g.max()
    check("levels_panel: coarsest -> finest -> ground truth, on the gt's scale",
          p.shape == (1, 8, 4 * W + 3 * 2) and torch.allclose(tiles[0], (g[0, 0] * 0.25) / ref)
          and torch.allclose(tiles[2], (g[0, 0] * 0.9) / ref)
          and torch.allclose(tiles[3], g[0, 0] / ref) and float(p.max()) <= 1.0,
          str(tuple(p.shape)))


def _fake_wandb():
    logs = []
    mod = types.ModuleType("wandb")
    mod.Image = lambda data, caption=None, **k: ("image", np.asarray(data), caption)
    sys.modules["wandb"] = mod

    class Run:
        def log(self, d, step=None):
            logs.append((step, d))
    return Run(), logs


def test_train_recon():
    from tests.test_skip_nonfinite_step import STEPS, Toy
    from tests.test_skip_nonfinite_step import batches as _batches
    from training.recon import train_recon

    def batches(n):
        # 16 x 12 slices solved on a 16 x 16 grid (pad_stride 8 for three
        # levels): the panel's crop back to 16 x 12 is exercised too
        return [b[:4] + (torch.tensor([[16, 16]]),) for b in _batches(n)]

    import training.recon as rm

    def run(net, **kw):
        run_, logs = _fake_wandb()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        buf = io.StringIO()
        # the filter grids are not under test (and need torchvision's make_grid)
        grids, rm.get_filter_grids = rm.get_filter_grids, lambda net: {}
        try:
            with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(buf):
                train_recon(net, opt, None, batches(4), batches(2), torch.device("cpu"),
                            R=2, acs_lines=4, kspace_type="simulated", noise_std=(0.0, 0.01),
                            val_noise_std=0.005, val_seed=0, num_epochs=2,
                            steps_per_epoch=STEPS, clip_grad=1.0, val_every_epochs=1,
                            save_dir=d, wandb=run_, use_organ_mask=True, **kw)
        finally:
            rm.get_filter_grids = grids
        return logs, buf.getvalue()

    logs, _ = run(mg_net(), log_levels=True)
    lv = [(s, d["val/levels"]) for s, d in logs if "val/levels" in d]
    ex = [(s, d["val/recon_example"]) for s, d in logs if "val/recon_example" in d]
    check("log_levels: val/levels at every validation", len(lv) == 2 and len(ex) == 2,
          f"{len(lv)} level panels, {len(ex)} recon panels")
    arr, cap = lv[-1][1][1], lv[-1][1][2]
    rec = ex[-1][1][1]
    # recon_example = [zero-filled | recon | gt | residual]; val/levels = [L2 | L1 | L0 | gt]
    H, Wt = arr.shape[0], (arr.shape[1] - 3 * 2) // 4
    l0 = arr[:, 2 * (Wt + 2):2 * (Wt + 2) + Wt]
    r_tile = rec[:, Wt + 2:Wt + 2 + Wt]
    check("the finest tile IS the recon tile (same crop, mask and greyscale)",
          arr.shape[1] == 4 * Wt + 6 and l0.shape == r_tile.shape
          and int(np.abs(l0.astype(int) - r_tile.astype(int)).max()) <= 1,
          f"panel {arr.shape}, max diff {int(np.abs(l0.astype(int) - r_tile.astype(int)).max())}")
    check("the caption names each level and its grid", all(f"L{l}" in cap for l in (0, 1, 2))
          and "ground truth" in cap, cap.splitlines()[0])
    logs, _ = run(mg_net())
    check("off by default", not any("val/levels" in d for _, d in logs))
    logs, out = run(Toy(), log_levels=True)
    check("a net without levels: one notice, no panel, no error",
          not any("val/levels" in d for _, d in logs) and out.count("log_levels:") == 1)


def main():
    for t in (test_level_images, test_panel, test_train_recon):
        print(f"\n== {t.__name__}")
        t()
    print(f"\n{'FAILED: ' + ', '.join(FAIL) if FAIL else 'all checks passed'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
