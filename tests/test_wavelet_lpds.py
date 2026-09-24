"""
Sanity tests for the DT-CWT-initialised LPDS net (models/wavelet_lpds.py).

The load-bearing test is `test_matches_dtcwt`: with untruncated filters
(P = 11) the grouped cascade followed by Q reproduces the `dtcwt` package's
oriented coefficients at every level, band and orientation (interior only --
dtcwt extends symmetrically, the convs zero-pad).  Skipped without `dtcwt`.

Run with:  python -m tests.test_wavelet_lpds
"""

import sys

import numpy as np
import torch
import torch.nn.functional as F

from models import build_model
from models.wavelet_lpds import WaveletLPDSLayer, WaveletLPDSNet
from models.wavelets import _qshift_taps, dtcwt_Q, dtcwt_weights

torch.manual_seed(0)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"[{'ok ' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")


def _unshuffle_inv(m):
    """(4^k, H, W) phases of one band -> (H 2^k, W 2^k) map."""
    while m.shape[0] > 1:
        m = torch.complex(F.pixel_shuffle(m.real[None], 2)[0],
                          F.pixel_shuffle(m.imag[None], 2)[0])
    return m[0]


def test_matches_dtcwt(P=11, N=128, margin=6):
    try:
        import dtcwt
    except ImportError:
        print("[skip] test_matches_dtcwt -- dtcwt not installed")
        return
    weights, tags = dtcwt_weights(P)
    Q = dtcwt_Q(tags).to(torch.complex128)
    X = np.random.default_rng(0).standard_normal((N, N))
    h = torch.tensor(X)[None, None]
    for l, w in enumerate(weights):
        h = F.conv2d(h, w.double(), stride=2, padding=P // 2,
                     groups=1 if l == 0 else 4)
    B, _, H, W = h.shape
    z = torch.einsum("cij,bjchw->bichw", Q,
                     h.to(torch.complex128).view(B, 4, len(tags), H, W))[0]

    pyr = dtcwt.Transform2d(biort="legall", qshift="qshift_06").forward(X, nlevels=3)
    slots = {1: (2, 3), 2: (0, 5), 3: (1, 4)}     # our band -> dtcwt (p-q, p+q)
    err = 0.0
    for level in (1, 2, 3):
        for b in (1, 2, 3):
            ch = [c for c, t in enumerate(tags) if t == (level, b)]
            for row, k in zip((0, 3), slots[b]):
                ours = _unshuffle_inv(z[row, ch]).numpy() * np.sqrt(2)
                ref = pyr.highpasses[level - 1][:, :, k]
                s = slice(margin, ref.shape[0] - margin)
                err = max(err, np.abs(ours[s, s] - ref[s, s]).max() / np.abs(ref).max())
    check("P=11 cascade + Q reproduces dtcwt (3 levels, 6 orientations)",
          err < 1e-6, f"max rel err {err:.1e}")


def test_truncation():
    """P = 7 drops exactly the +-0.0352 end tap of each q-shift filter."""
    ok = True
    for hi in (False, True):
        for lin in (0, 1):
            full, short = _qshift_taps(hi, lin, 11), _qshift_taps(hi, lin, 7)
            dropped = np.concatenate([full[:2], full[-2:]])
            ok &= np.allclose(full[2:-2], short)
            ok &= np.allclose(np.sort(np.abs(dropped)), [0, 0, 0, 0.0351638366])
    check("P=7 drops only the 0.0352 end tap", bool(ok))


def test_Q():
    _, tags = dtcwt_weights(7)
    Q = dtcwt_Q(tags).to(torch.complex128)
    eye = torch.eye(4, dtype=torch.complex128)
    QQh = torch.einsum("cij,ckj->cik", Q, Q.conj())
    check("Q unitary per channel", bool(torch.allclose(QQh, eye.expand_as(QQh))))
    check("Q is the identity on LL^3", bool(torch.allclose(Q[0], eye)))


def test_layer_init():
    lay = WaveletLPDSLayer(P=7, lam0=1e-3, tau0=0.5, degrees=1)
    check("grouped by tree at levels 2-3",
          [a.groups for a in lay.analysis] == [1, 4, 4])

    x = torch.randn(2, 1, 32, 32, dtype=torch.complex64)
    z = torch.randn(2, 256, 4, 4, dtype=torch.complex64)
    lhs = (lay.analyse(x).conj() * z).sum()
    rhs = (x.conj() * lay.adjoint(z)).sum()
    check("B = A^H: <Kx, z> = <x, K^H z>",
          abs(lhs - rhs).item() < 1e-4 * abs(lhs).item(),
          f"{lhs.item():.4f} vs {rhs.item():.4f}")

    n2 = lay.op_norm2()
    check("||K|| = 1 after normalize", abs(n2 - 1) < 1e-3, f"||K||^2 = {n2:.5f}")
    check("tau0 inside the Condat-Vu bound", 0.5 * (0.5 + n2) <= 1.0)

    before = [c.weight.clone() for c in list(lay.analysis) + list(lay.synthesis)]
    lay.project_()
    after = [c.weight for c in list(lay.analysis) + list(lay.synthesis)]
    check("project() is a no-op at init",
          all(torch.allclose(a, b, atol=1e-6) for a, b in zip(before, after)))

    t = lay.prox.prox.tau.weight
    ll = lay.ll_channels
    rest = [c for c in range(256) if c not in ll]
    check("LL^3 thresholds start at 0", bool((t[:, ll] == 0).all()))
    check("other thresholds start at lam0 (sigma-slope 0)",
          bool((t[0, rest] == 1e-3).all() and (t[1, rest] == 0).all()))


def test_net_forward_backward():
    net = build_model({"model": {"type": "WaveletLPDSNet", "params": dict(
        K=4, M=16, P=7, s=2, degrees=1, preproc="identity")}})
    check("build_model registers WaveletLPDSNet", isinstance(net, WaveletLPDSNet))
    y = torch.randn(2, 1, 32, 32, dtype=torch.complex64)
    sigma = torch.full((2, 1, 1, 1), 0.05)
    x_hat, (x, z) = net(y, sigma=sigma)
    check("output shape and finite",
          x_hat.shape == y.shape and bool(torch.isfinite(x_hat.abs()).all()))
    check("dual lives on the depth-3 grid", tuple(z.shape) == (2, 256, 4, 4))

    x_hat.abs().pow(2).sum().backward()
    g = net.layer(1).prox.prox.tau.weight.grad
    ll = net.layer(1).ll_channels
    check("LL^3 thresholds get gradient at 0",
          g is not None and g[:, ll].abs().sum().item() > 0)


if __name__ == "__main__":
    test_matches_dtcwt()
    test_truncation()
    test_Q()
    test_layer_init()
    test_net_forward_backward()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    sys.exit(1 if FAIL else 0)
