"""Tests of the k-space training losses (training/ssdu.py, train_recon's kspace_loss).

    python tests/test_ssdu.py

Reuses tests/test_guided_recon.py's stubs and synthetic paired knee data.
"""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_guided_recon as tg            # noqa: E402

import torch                              # noqa: E402

from physics.mask import get_mask_cached  # noqa: E402
from training.ssdu import (center_run, kspace_nl1nl2, sampled_lines,  # noqa: E402
                           ssdu_split)


def _mask(W=64, R=4, acs=6):
    return get_mask_cached(torch.zeros(1, 1, 16, W, dtype=torch.complex64), R=R,
                           acs_lines=acs, mode="uniform", offset=0)


def test_split_is_a_disjoint_cover_with_the_acs_in_the_input():
    m = _mask()
    lines = sampled_lines(m)
    acs = center_run(lines)
    assert int(acs.sum()) >= 6
    torch.manual_seed(0)
    theta, lam = ssdu_split(m, 4, rho=0.4)
    th, la = theta[:, 0, 0] > 0, lam[:, 0, 0] > 0
    n_cand = int((lines & ~acs).sum())
    for b in range(4):
        assert not (th[b] & la[b]).any()
        assert ((th[b] | la[b]) == lines).all()
        assert th[b][acs].all() and not la[b][acs].any()
        assert int(la[b].sum()) == round(0.4 * n_cand)
    assert not all((la[0] == la[b]).all() for b in range(1, 4)), "same split for every sample"


def test_split_refuses_a_non_line_mask():
    m = _mask().clone()
    m[..., 3, 1] = 1
    tg.raises(ValueError, lambda: ssdu_split(m, 1), "line mask")


def test_loss_is_normalized_and_ignores_unscored_lines():
    torch.manual_seed(0)
    k = torch.randn(2, 4, 16, 32, dtype=torch.complex64)
    m = torch.zeros(2, 1, 1, 32)
    m[..., ::3] = 1
    assert float(kspace_nl1nl2(k, k, m)) == 0.0
    p = k + 0.1 * torch.randn_like(k)
    a = kspace_nl1nl2(p, k, m)
    assert torch.allclose(kspace_nl1nl2(5 * p, 5 * k, m), a)           # scale-free
    junk = (1 - m) * torch.randn_like(k)                                # off the loss lines
    assert torch.allclose(kspace_nl1nl2(p + junk, k + junk * 3, m), a)


class Recorder(torch.nn.Module):
    """Records the sampled columns of every y the wrapped net is called with."""
    def __init__(self, net):
        super().__init__()
        self.net, self.cols = net, []
        for a in ("guide_domain", "uses_operator_smaps", "pad_stride"):
            if hasattr(net, a):
                setattr(self, a, getattr(net, a))
        self.guided = tg.tr._wants_guide(net)          # this wrapper's own signature takes one

    def forward(self, y, E=None, sigma=None, guide=None):
        self.cols.append((self.training, int((y.abs().sum(dim=(0, 1, 2)) > 0).sum())))
        kw = {} if guide is None else {"guide": guide}
        return self.net(y, E=E, sigma=sigma, **kw)


def _train(net, **kw):
    import physics.online_smaps as osm
    with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as d:
        csv, _ = tg.make_data(root)
        real = osm.online_smaps
        osm.online_smaps = lambda y, mask, **k: torch.full_like(y, tg.C ** -0.5)
        try:
            rec = Recorder(net)
            wb = tg.run_recon(rec, csv, root, d, online_smaps="espirit", **kw)
        finally:
            osm.online_smaps = real
    return rec, wb


def test_ssdu_trains_on_fewer_lines_and_validates_on_all():
    from models.guided_lpds import LGGSNet
    torch.manual_seed(0)
    net = LGGSNet(K=2, M=8, P=3, s=2, Mh=4, guide_window=3, attn_backend="gather",
                  separate_guide_w=True)
    rec, wb = _train(net, kspace_loss="ssdu", ssdu_rho=0.4)
    train_cols = {c for t, c in rec.cols if t}
    val_cols = {c for t, c in rec.cols if not t}
    assert len(val_cols) == 1 and max(train_cols) < min(val_cols), (train_cols, val_cols)
    losses = [c["train/loss"] for c in wb.calls if "train/loss" in c]
    assert losses and all(torch.isfinite(torch.as_tensor(l)) for l in losses)
    assert torch.isfinite(torch.as_tensor(wb.last("val/psnr")))


def test_unsampled_mode_and_the_other_nets_train():
    from models.tgvn import TGVN
    from models import build_model
    torch.manual_seed(0)
    for net in (TGVN(guided=True, cond_sigma=True, use_smaps=True, output="complex",
                     **tg.TINY_TGVN),
                build_model({"model": {"type": "MGLPDSNet", "params": dict(
                    K=2, M=8, C=1, P=3, s=2, lam0=1e-3, tau0=0.5, theta0=0.0, alpha0=1.0,
                    is_complex=True, preproc="kspace")}})):
        for mode in ("ssdu", "unsampled"):
            rec, wb = _train(net, kspace_loss=mode)
            assert torch.isfinite(torch.as_tensor(wb.last("val/psnr"))), (type(net), mode)
            if mode == "unsampled":                          # the input is all acquired lines
                assert len({c for _, c in rec.cols}) == 1, rec.cols


def test_magnitude_output_is_refused():
    from models.tgvn import TGVN
    net = TGVN(guided=True, cond_sigma=True, use_smaps=True, **tg.TINY_TGVN)
    tg.raises(ValueError, lambda: _train(net, kspace_loss="ssdu"), "output='complex'")


def test_refusals():
    from models.guided_lpds import LGGSNet
    net = LGGSNet(K=2, M=8, P=3, s=2, Mh=4, guide_window=3, attn_backend="gather")
    tg.raises(ValueError, lambda: _train(net, kspace_loss="kspace"), "kspace_loss must be")
    tg.raises(ValueError, lambda: _train(net, kspace_loss="ssdu", whiten_kspace=True),
              "whiten")


def test_generator_cells():
    sys.path.insert(0, os.path.join(tg.REPO, "scripts"))
    import make_guided_recon_configs as g
    for base in ("lpdsnet", "tgvn_c6", "lggs_guide_w"):
        spec = g.MODELS[f"{base}_ssdu"]
        assert spec["training"]["kspace_loss"] == "ssdu"
        assert spec["guided"] == g.MODELS[base]["guided"]
        if base != "tgvn_c6":                                  # its output changes (below)
            assert spec["model"] == g.MODELS[base]["model"]
    assert g.MODELS["lggs_guide_w_ssdu"]["model"]["params"]["separate_guide_w"]
    assert g.MODELS["tgvn_c6_ssdu"]["model"]["params"]["output"] == "complex"
    assert "output" not in g.MODELS["tgvn_c6"]["model"]["params"]      # the baseline untouched


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
