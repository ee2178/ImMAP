"""
Per-level supervision (`training.level_loss_weight`, training/recon.py).

Run with `python -m tests.test_level_supervision`. CPU, a few seconds.

1. `collect_level_outputs` (models/level_trace.py) keeps each COARSE level's
   final iterate with its autograd graph: levels 1..L, halving grids, one per
   V-cycle under "all" and the last only under "last"; they ARE the traced
   `x_out` of those levels; the forward is unchanged; the hooks are gone after.
2. `supervised_level_images` adds the DC back exactly as `level_images` (the
   val/levels panel) does; planar state gives the same.
3. `_level_loss`: the documented formula (restricted gt, restricted crop
   support, per-level weights); zero for a level holding exactly R^l gt; its
   gradient reaches the coarsest level's weights, which the output loss alone
   barely moves.
4. `_level_loss_weights`: off at 0, one weight per coarse level, refuses a
   flat net, a wrong count, a negative weight, an unknown `outers`.
5. `train_recon(level_loss_weight=...)` end to end; the validation loss is
   still the output's alone.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from tests.test_level_panel import _fake_wandb, mg_net, operator  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(f"[{'ok ' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def _rel(a, b):
    return float((a - b).abs().max() / (b.abs().max() + 1e-12))


def _inputs():
    E = operator()
    img = torch.randn(1, 1, 32, 32, dtype=torch.complex64,
                      generator=torch.Generator().manual_seed(1)) + 2.0
    return E, img, E(img), torch.full((1, 1, 1, 1), 0.01)


def test_collect():
    from models.level_trace import collect_level_outputs, trace_forward
    E, img, y, sig = _inputs()
    net = mg_net(K=(3, [2, 2, 2]))
    with torch.no_grad():
        ref = net(y, E=E, sigma=sig)[0]
    with collect_level_outputs(net, "last") as rec:
        out, (x0, _) = net(y, E=E, sigma=sig)
    lv = rec.levels
    check("collects levels 1..L, not 0", sorted(lv) == [1, 2], str(sorted(lv)))
    check("one iterate per level under 'last', grids halving",
          [len(lv[l]) for l in (1, 2)] == [1, 1]
          and [tuple(lv[l][0].shape[-2:]) for l in (1, 2)] == [(16, 16), (8, 8)])
    check("they carry autograd", all(lv[l][0].requires_grad for l in (1, 2)))
    check("the forward is unchanged", _rel(out.detach(), ref) < 1e-6)
    _, tr = trace_forward(net, y, E=E, sigma=sig)
    gaps = [_rel(lv[l][0].detach(), tr.get(-1, l, "x_out")) for l in (1, 2)]
    check("they ARE the traced x_out of the last V-cycle", max(gaps) < 1e-6,
          str([f"{g:.1e}" for g in gaps]))
    with collect_level_outputs(net, "all") as rec_all:
        net(y, E=E, sigma=sig)
    check("'all': one per V-cycle", [len(rec_all.levels[l]) for l in (1, 2)] == [3, 3])
    n_before = len(rec.levels[1])
    net(y, E=E, sigma=sig)
    check("hooks removed on exit", len(rec.levels[1]) == n_before)


def test_dc():
    from models.level_trace import collect_level_outputs, supervised_level_images
    from models.mg_lpds import MGLPDSNet
    E, img, y, sig = _inputs()
    for planar in (False, True):
        prev = MGLPDSNet.PLANAR_STATE
        try:
            MGLPDSNet.PLANAR_STATE = planar
            net = mg_net()
            with collect_level_outputs(net) as rec:
                out, (x0, _) = net(y, E=E, sigma=sig)
            sup = supervised_level_images(net, rec.levels, out, x0)
            with torch.no_grad():
                _, imgs = net.forward_level_images(y, E=E, sigma=sig)
        finally:
            MGLPDSNet.PLANAR_STATE = prev
        gaps = [_rel(sup[l][0].detach(), imgs[l]) for l in (1, 2)]
        check(f"planar={planar}: image units == the val/levels panel's (DC added back)",
              max(gaps) < 1e-4 and all(torch.is_complex(sup[l][0]) for l in (1, 2)),
              str([f"{g:.1e}" for g in gaps]))


class _Crop:
    """Truncate-like: the 32 x 32 net grid is the embedding of a 32 x 28 image."""
    def __init__(self, H=32, W=32, w=28):
        self.H, self.W, self.w = H, W, w
        self.c0 = (W - w) // 2

    def forward(self, x):
        return x[..., self.c0:self.c0 + self.w]

    def adjoint(self, x):
        out = torch.zeros(*x.shape[:-1], self.W, dtype=x.dtype)
        out[..., self.c0:self.c0 + self.w] = x
        return out


def test_loss():
    from models.level_trace import _transfer_kws, collect_level_outputs, supervised_level_images
    from operators.resample import restrict
    from training.losses import LOSS_REGISTRY
    from training.recon import _level_loss
    E, img, y, sig = _inputs()
    T = _Crop()
    image = T.forward(img)
    organ = (image.abs() > 1.5).float()
    loss_fn = LOSS_REGISTRY["magnitude-nl1-nl2"]
    net = mg_net()
    with collect_level_outputs(net) as rec:
        out, (x0, _) = net(y, E=E, sigma=sig)
    got = _level_loss(net, rec.levels, out, x0, image, organ, T, None, True, [0.3, 0.7],
                      loss_fn, sig)
    kws = _transfer_kws(net)
    sup = supervised_level_images(net, rec.levels, out, x0)
    g, w = T.adjoint(image), T.adjoint(torch.ones_like(image)).real
    w = w * T.adjoint(organ.to(image.dtype)).real
    want = 0.0
    for l, wt in ((1, 0.3), (2, 0.7)):
        g, w = restrict(g, **kws), restrict(w, **kws)
        wl = w.clamp(0, 1)
        want = want + wt * loss_fn(g * wl, sup[l][0] * wl, sig)
    check("the documented formula", abs(float(got) - float(want)) < 1e-6 * abs(float(want)),
          f"{float(got):.6e} vs {float(want):.6e}")

    # a level holding exactly R^l gt scores zero
    g1 = restrict(T.adjoint(image), **kws)
    exact = {1: [g1 - (out - x0).detach().mean(dim=(-2, -1), keepdim=True)]}
    zero = _level_loss(net, exact, out.detach(), x0.detach(), image, organ, T, None, False,
                       [1.0, 1.0], loss_fn, sig)
    check("zero for a level that holds R^l gt", float(zero) < 1e-6, f"{float(zero):.1e}")

    # its gradient reaches the coarsest level's own weights
    net.zero_grad()
    with collect_level_outputs(net) as rec:
        out, (x0, _) = net(y, E=E, sigma=sig)
    _level_loss(net, rec.levels, out, x0, image, organ, T, None, True, [0.0, 1.0],
                loss_fn, sig).backward()
    from models.level_trace import _pd_chains
    leaf = max(_pd_chains(net).items(), key=lambda kv: (kv[0][0], kv[0][1]))[1]
    gl = sum(float(p.grad.abs().sum()) for p in leaf.parameters() if p.grad is not None)
    check("the level loss trains the coarsest level", gl > 0, f"|grad| {gl:.2e}")


def test_weights():
    from training.recon import _level_loss_weights
    net = mg_net()
    check("0 -> off", _level_loss_weights(net, 0.0, "last") == [])
    check("a number -> that weight on every coarse level",
          _level_loss_weights(net, 0.1, "last") == [0.1, 0.1])
    check("a list -> one per coarse level", _level_loss_weights(net, [0.2, 0.5], "all") == [0.2, 0.5])
    for bad, kw in (("flat net", dict(net=mg_net(K=6), weight=0.1, outers="last")),
                    ("wrong count", dict(net=net, weight=[0.1, 0.1, 0.1], outers="last")),
                    ("negative", dict(net=net, weight=-0.1, outers="last")),
                    ("unknown outers", dict(net=net, weight=0.1, outers="first"))):
        try:
            _level_loss_weights(**kw)
            check(f"refused: {bad}", False)
        except ValueError:
            check(f"refused: {bad}", True)


def test_train_recon():
    from tests.test_skip_nonfinite_step import STEPS
    from tests.test_skip_nonfinite_step import batches as _batches
    import training.recon as rm

    def batches(n):
        return [b[:4] + (torch.tensor([[16, 16]]),) for b in _batches(n)]

    def run(net, **kw):
        run_, logs = _fake_wandb()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        buf = io.StringIO()
        grids, rm.get_filter_grids = rm.get_filter_grids, lambda net: {}
        try:
            with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(buf):
                rm.train_recon(net, opt, None, batches(4), batches(2), torch.device("cpu"),
                               R=2, acs_lines=4, kspace_type="simulated",
                               noise_std=(0.0, 0.01), val_noise_std=0.005, val_seed=0,
                               num_epochs=2, steps_per_epoch=STEPS, clip_grad=1.0,
                               val_every_epochs=1, save_dir=d, wandb=run_,
                               use_organ_mask=True, **kw)
        finally:
            rm.get_filter_grids = grids
        return logs, buf.getvalue()

    def val_psnr(logs):
        return [float(d["val/psnr"]) for _, d in logs if "val/psnr" in d]

    logs_off, _ = run(mg_net())
    logs_on, out = run(mg_net(), level_loss_weight=0.1)
    tl_on = [float(d["train/loss"]) for _, d in logs_on if "train/loss" in d]
    tl_off = [float(d["train/loss"]) for _, d in logs_off if "train/loss" in d]
    check("trains end to end with per-level supervision",
          "per-level supervision: weights [0.1, 0.1]" in out
          and len(val_psnr(logs_on)) == 2 and all(map(lambda v: v == v, val_psnr(logs_on))),
          f"val psnr {val_psnr(logs_on)}")
    check("the training loss includes the level terms (larger than without)",
          bool(tl_on) and bool(tl_off) and tl_on[0] > tl_off[0],
          f"first epoch {tl_on[:1]} vs {tl_off[:1]}")
    try:
        run(mg_net(K=6), level_loss_weight=0.1)
        check("a flat net refuses level_loss_weight at launch", False)
    except ValueError:
        check("a flat net refuses level_loss_weight at launch", True)


def main():
    for t in (test_collect, test_dc, test_loss, test_weights, test_train_recon):
        print(f"\n== {t.__name__}")
        t()
    print(f"\n{'FAILED: ' + ', '.join(FAIL) if FAIL else 'all checks passed'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
