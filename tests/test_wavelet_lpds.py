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
from models.wavelets import (FAMILIES, NEAR_SYM_A_H0, NEAR_SYM_A_H1, _biort_level1,
                             _qshift_taps, dtcwt_Q, dtcwt_weights)

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


def test_matches_dtcwt(P=11, N=128, margin=6, family="dtcwt", biort="legall"):
    try:
        import dtcwt
    except ImportError:
        print("[skip] test_matches_dtcwt -- dtcwt not installed")
        return
    weights, tags = FAMILIES[family][0](P)
    Q = dtcwt_Q(tags).to(torch.complex128)
    X = np.random.default_rng(0).standard_normal((N, N))
    h = torch.tensor(X)[None, None]
    for l, w in enumerate(weights):
        h = F.conv2d(h, w.double(), stride=2, padding=P // 2,
                     groups=1 if l == 0 else 4)
    B, _, H, W = h.shape
    z = torch.einsum("cij,bjchw->bichw", Q,
                     h.to(torch.complex128).view(B, 4, len(tags), H, W))[0]

    pyr = dtcwt.Transform2d(biort=biort, qshift="qshift_06").forward(X, nlevels=3)
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
    check(f"{family}: P=11 cascade + Q reproduces dtcwt({biort}) (3 levels, 6 orientations)",
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


def test_dtcwt57_crop():
    """P = 7 crops only the +-0.0107 outer highpass tap, in the parity-1 trees."""
    taps = _biort_level1(NEAR_SYM_A_H0, NEAR_SYM_A_H1)
    ok = True
    for hi in (False, True):
        for parity in (0, 1):
            full, short = taps(hi, parity, 9), taps(hi, parity, 7)
            dropped = np.abs(np.concatenate([full[:1], full[-1:]]))
            ok &= np.allclose(full[1:-1], short)
            lost = 0.0107142857 if (hi and parity == 1) else 0.0
            ok &= np.allclose(np.sort(dropped), [0.0, lost])
    check("dtcwt57: P=7 drops only the 0.0107 tap of the parity-1 highpass", bool(ok))


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


def test_carry_unshuffle_matches_conv():
    """At init the fixed-unshuffle carries give exactly the dense-conv K, K^H."""
    fast = WaveletLPDSLayer(P=7, spectral_init=False, carry="unshuffle")
    dense = WaveletLPDSLayer(P=7, spectral_init=False, carry="conv")
    check("unshuffle carry: only the LL split is a conv at levels 2-3",
          [tuple(a.weight.shape) for a in fast.analysis]
          == [(16, 1, 7, 7)] * 3)
    x = torch.randn(2, 1, 32, 32, dtype=torch.complex64)
    z = torch.randn(2, 256, 4, 4, dtype=torch.complex64)
    ea = (fast.analyse(x) - dense.analyse(x)).abs().max().item()
    eh = (fast.adjoint(z) - dense.adjoint(z)).abs().max().item()
    check("unshuffle carry: K matches the dense cascade", ea < 1e-5, f"{ea:.1e}")
    check("unshuffle carry: K^H matches the dense cascade", eh < 1e-5, f"{eh:.1e}")


def _complexify(lay):
    """Random complex weights, with B = conj(A) so K^H is still the adjoint."""
    from models.base import set_weight
    for a, b in zip(lay.analysis, lay.synthesis):
        w = a.weight + 0.1 * torch.randn_like(a.weight)
        set_weight(a, w)
        set_weight(b, w.conj())


def test_interleaved_matches_gauss():
    """The one-real-conv interleaved path == the Gauss-trick complex modules."""
    lay = WaveletLPDSLayer(P=7, spectral_init=False, carry="conv")
    _complexify(lay)
    x = torch.randn(2, 1, 32, 32, dtype=torch.complex64)
    z = torch.randn(2, 256, 4, 4, dtype=torch.complex64)

    def mix(v, Q):
        B, _, h, w = v.shape
        v = v.reshape(B, 4, lay.Cg, h, w)
        return torch.einsum("cij,bjchw->bichw", Q, v).reshape(B, -1, h, w)

    ref_a = x
    for a in lay.analysis:
        ref_a = a(ref_a)
    ref_a = mix(ref_a, lay.Q)
    ref_h = mix(z, lay.QH)
    for b in reversed(lay.synthesis):
        ref_h = b(ref_h)
    ea = ((lay.analyse(x) - ref_a).abs().max() / ref_a.abs().max()).item()
    eh = ((lay.adjoint(z) - ref_h).abs().max() / ref_h.abs().max()).item()
    check("interleaved K == Gauss-trick K (complex weights)", ea < 1e-5, f"{ea:.1e}")
    check("interleaved K^H == Gauss-trick K^H (complex weights)", eh < 1e-5, f"{eh:.1e}")
    check("real input goes through K", lay.analyse(x.real).is_complex())


def test_adjoint_complex_weights():
    for carry in ("unshuffle", "conv"):
        lay = WaveletLPDSLayer(P=7, spectral_init=False, carry=carry)
        _complexify(lay)
        x = torch.randn(2, 1, 32, 32, dtype=torch.complex128)
        z = torch.randn(2, 256, 4, 4, dtype=torch.complex128)
        lay.double()
        lhs = (lay.analyse(x).conj() * z).sum()
        rhs = (x.conj() * lay.adjoint(z)).sum()
        err = (abs(lhs - rhs) / abs(lhs)).item()
        check(f"<Kx, z> = <x, K^H z> with complex weights ({carry})",
              err < 1e-10, f"rel err {err:.1e}")


def test_haar():
    """Each tree is an exact orthonormal Haar DWT of its cycle-spun image."""
    from models.wavelets import haar_weights
    hw, htags = haar_weights(7)
    dw, dtags = dtcwt_weights(7)
    check("haar: same shapes and tags as dtcwt",
          [w.shape for w in hw] == [w.shape for w in dw] and htags == dtags)

    N = 64
    x = torch.zeros(1, 1, N, N, dtype=torch.complex128)
    x[..., 8:-8, 8:-8] = torch.randn(1, 1, N - 16, N - 16, dtype=torch.complex128)
    for carry in ("unshuffle", "conv"):
        lay = WaveletLPDSLayer(P=7, spectral_init=False, carry=carry, family="haar").double()
        z = lay.analyse(x)                                  # Q = I: tree t at t * Cg
        Cg = lay.Cg
        ll = z[:, [t * Cg for t in range(4)]]
        ref = F.avg_pool2d(x.real, 8) * 8 + 1j * F.avg_pool2d(x.imag, 8) * 8
        e_ll = (ll[:, 0:1] - ref).abs().max().item()
        e_gram = (lay.adjoint(z) - 4 * x).abs().max().item()
        check(f"haar ({carry}): LL^3 of tree 0 = 8x8 block sum / 8", e_ll < 1e-10, f"{e_ll:.1e}")
        check(f"haar ({carry}): K^H K = 4 I (interior image, 4 orthonormal trees)",
              e_gram < 1e-10, f"{e_gram:.1e}")

    # tree (1, 1) reads pairs (2j+1, 2j+2): tree (0, 0) of the image moved up-left
    xs = torch.roll(x, shifts=(-1, -1), dims=(-2, -1))
    z0, zs = lay.analyse(x), lay.analyse(xs)
    e_sh = (z0[:, 3 * Cg:4 * Cg] - zs[:, 0:Cg]).abs().max().item()
    check("haar: tree (1,1) = tree (0,0) of the shifted image", e_sh < 1e-10, f"{e_sh:.1e}")

    # level-3 details of tree 0 are the Haar differences of LL^2 (4x4 sums / 4)
    ll2 = (F.avg_pool2d(x.real, 4) + 1j * F.avg_pool2d(x.imag, 4)) * 4
    a, b = ll2[..., 0::2, 0::2], ll2[..., 0::2, 1::2]
    c, d = ll2[..., 1::2, 0::2], ll2[..., 1::2, 1::2]
    ref = {1: (a - b + c - d) / 2, 2: (a + b - c - d) / 2, 3: (a - b - c + d) / 2}
    e_d = max((z0[:, htags.index((3, band))] - ref[band][:, 0]).abs().max().item()
              for band in (1, 2, 3))
    check("haar: level-3 bands = Haar differences of LL^2 (B1 lo-H/hi-W, B2, B3)",
          e_d < 1e-10, f"{e_d:.1e}")

    lay = WaveletLPDSLayer(P=7, lam0=1e-3, tau0=0.5, degrees=1, family="haar")
    n2 = lay.op_norm2()
    check("haar: ||K|| = 1 after normalize", abs(n2 - 1) < 1e-3, f"||K||^2 = {n2:.5f}")
    before = [c.weight.clone() for c in list(lay.analysis) + list(lay.synthesis)]
    lay.project_()
    check("haar: project() is a no-op at init",
          all(torch.allclose(b_, c.weight, atol=1e-6) for b_, c in
              zip(before, list(lay.analysis) + list(lay.synthesis))))
    try:
        WaveletLPDSLayer(family="db4")
        check("unknown family is refused", False)
    except ValueError:
        check("unknown family is refused", True)


def test_band_norm():
    """`band_norm="equal"`: one noise gain for every deep channel, ||K|| = 1,
    and still inside the unit ball, so `project_` cannot undo it."""
    base = WaveletLPDSLayer(P=7, family="dtcwt")
    nb = base.band_norms()
    check("band_norm='none' (default) keeps the DT-CWT spread",
          float(nb.max() / nb.min()) > 2.0, f"spread {float(nb.max() / nb.min()):.2f}")
    for carry in ("unshuffle", "conv"):
        lay = WaveletLPDSLayer(P=7, family="dtcwt", carry=carry, band_norm="equal")
        nb = lay.band_norms()
        spread = float(nb.max() / nb.min())
        check(f"equal ({carry}): every deep channel has one noise gain",
              spread < 1.001, f"spread {spread:.4f}")
        n2 = lay.op_norm2()
        check(f"equal ({carry}): ||K|| = 1", abs(n2 - 1) < 1e-2, f"||K||^2 = {n2:.4f}")
        convs = list(lay.analysis) + list(lay.synthesis)
        before = [c.weight.clone() for c in convs]
        lay.project_()
        check(f"equal ({carry}): project() is a no-op at init (slices <= 1)",
              all(torch.allclose(b_, c.weight, atol=1e-5) for b_, c in zip(before, convs)))
        x = torch.randn(1, 1, 32, 32, dtype=torch.complex64)
        z = torch.randn(1, 256, 4, 4, dtype=torch.complex64)
        lhs, rhs = (lay.analyse(x).conj() * z).sum(), (x.conj() * lay.adjoint(z)).sum()
        check(f"equal ({carry}): B = A^H still holds",
              abs(lhs - rhs).item() < 1e-4 * abs(lhs).item())
    a = WaveletLPDSLayer(P=7, family="haar", band_norm="none")
    b = WaveletLPDSLayer(P=7, family="haar", band_norm="equal")
    check("equal is a no-op on Haar (already orthonormal)",
          all(torch.allclose(x.weight, y.weight, atol=1e-6)
              for x, y in zip(a.analysis, b.analysis)))


def test_union():
    """family=["dtcwt", "haar"]: the two transforms stacked, 32/128/512."""
    x = torch.randn(2, 1, 32, 32, dtype=torch.complex128)
    z = torch.randn(2, 512, 4, 4, dtype=torch.complex128)
    for carry in ("unshuffle", "conv"):
        u = WaveletLPDSLayer(P=7, spectral_init=False, carry=carry,
                             family=["dtcwt", "haar"]).double()
        d = WaveletLPDSLayer(P=7, spectral_init=False, carry=carry, family="dtcwt").double()
        h = WaveletLPDSLayer(P=7, spectral_init=False, carry=carry, family="haar").double()
        if carry == "unshuffle":
            check("union: level-1 conv is 1 -> 32, 8 trees after",
                  tuple(u.analysis[0].weight.shape[:2]) == (32, 1)
                  and [a.groups for a in u.analysis] == [1, 8, 8])
        ku = u.analyse(x)
        e = max((ku[:, :256] - d.analyse(x)).abs().max().item(),
                (ku[:, 256:] - h.analyse(x)).abs().max().item())
        check(f"union ({carry}): K = [K_dtcwt; K_haar] exactly", e < 1e-12, f"{e:.1e}")
        e = (u.adjoint(z) - d.adjoint(z[:, :256]) - h.adjoint(z[:, 256:])).abs().max().item()
        check(f"union ({carry}): K^H = K_dtcwt^H + K_haar^H", e < 1e-12, f"{e:.1e}")

    u = WaveletLPDSLayer(P=7, family=["dtcwt", "haar"])
    nf = [u.op_norm2(family=f) for f in (0, 1)]
    check("union: the two families carry equal weight (||K_f|| equal)",
          abs(nf[0] / nf[1] - 1) < 2e-2, ", ".join(f"{v:.4f}" for v in nf))
    n2 = u.op_norm2()
    check("union: ||K|| = 1", abs(n2 - 1) < 1e-2, f"||K||^2 = {n2:.4f}")
    check("union: 8 unpenalised LL^3 channels, one per tree",
          u.ll_channels == [t * 64 for t in range(8)]
          and bool((u.prox.prox.tau.weight[:, u.ll_channels] == 0).all()))
    convs = list(u.analysis) + list(u.synthesis)
    before = [c.weight.clone() for c in convs]
    u.project_()
    check("union: project() is a no-op at init",
          all(torch.allclose(b_, c.weight, atol=1e-5) for b_, c in zip(before, convs)))

    ue = WaveletLPDSLayer(P=7, family=["dtcwt", "haar"], band_norm="equal")
    nb = ue.band_norms()
    check("union + band_norm='equal': one noise gain over all 512 channels",
          float(nb.max() / nb.min()) < 1.001, f"spread {float(nb.max() / nb.min()):.4f}")
    convs = list(ue.analysis) + list(ue.synthesis)
    before = [c.weight.clone() for c in convs]
    ue.project_()
    check("union + equal: project() is a no-op at init",
          all(torch.allclose(b_, c.weight, atol=1e-5) for b_, c in zip(before, convs)))

    net = build_model({"model": {"type": "WaveletLPDSNet", "params": dict(
        K=3, M=32, P=7, s=2, degrees=1, preproc="identity", family=["dtcwt", "haar"])}})
    y = torch.randn(1, 1, 32, 32, dtype=torch.complex64)
    x_hat, (_, zz) = net(y, sigma=torch.full((1, 1, 1, 1), 0.05))
    check("union net: dual is 512 channels on the depth-3 grid",
          tuple(zz.shape) == (1, 512, 4, 4) and x_hat.shape == y.shape)
    try:
        WaveletLPDSNet(K=2, M=16, family=["dtcwt", "haar"])
        check("union net: M must be 16 per family", False)
    except ValueError:
        check("union net: M must be 16 per family", True)
    for bad in (["dtcwt", "dtcwt"], [], ["dtcwt", "db4"]):
        try:
            WaveletLPDSLayer(family=bad)
            check(f"family={bad!r} is refused", False)
        except ValueError:
            check(f"family={bad!r} is refused", True)


def test_union_three():
    """Any number of families: ["dtcwt", "haar", "dtcwt57"] is 12 trees,
    48/192/768, and K is exactly the three transforms stacked."""
    fams = ["dtcwt", "haar", "dtcwt57"]
    x = torch.randn(2, 1, 32, 32, dtype=torch.complex128)
    z = torch.randn(2, 768, 4, 4, dtype=torch.complex128)
    for carry in ("unshuffle", "conv"):
        u = WaveletLPDSLayer(P=7, spectral_init=False, carry=carry, family=fams).double()
        parts = [WaveletLPDSLayer(P=7, spectral_init=False, carry=carry,
                                  family=f).double() for f in fams]
        ku = u.analyse(x)
        e = max((ku[:, 256 * i:256 * (i + 1)] - p.analyse(x)).abs().max().item()
                for i, p in enumerate(parts))
        check(f"3 families ({carry}): K = [K_dtcwt; K_haar; K_dtcwt57] exactly",
              e < 1e-12, f"{e:.1e}")
        ref = sum(p.adjoint(z[:, 256 * i:256 * (i + 1)]) for i, p in enumerate(parts))
        e = (u.adjoint(z) - ref).abs().max().item()
        check(f"3 families ({carry}): K^H = sum of the three adjoints", e < 1e-12, f"{e:.1e}")

    u = WaveletLPDSLayer(P=7, family=fams)
    check("3 families: grouped into 12 trees", [a.groups for a in u.analysis] == [1, 12, 12])
    nf = [u.op_norm2(family=f) for f in range(3)]
    check("3 families: equal ||K_f||", max(nf) / min(nf) - 1 < 2e-2,
          ", ".join(f"{v:.4f}" for v in nf))
    n2 = u.op_norm2()
    check("3 families: ||K|| = 1", abs(n2 - 1) < 1e-2, f"||K||^2 = {n2:.4f}")
    check("3 families: 12 unpenalised LL^3 channels",
          u.ll_channels == [t * 64 for t in range(12)]
          and bool((u.prox.prox.tau.weight[:, u.ll_channels] == 0).all()))
    convs = list(u.analysis) + list(u.synthesis)
    before = [c.weight.clone() for c in convs]
    u.project_()
    check("3 families: project() is a no-op at init",
          all(torch.allclose(b_, c.weight, atol=1e-5) for b_, c in zip(before, convs)))

    net = build_model({"model": {"type": "WaveletLPDSNet", "params": dict(
        K=3, M=48, P=7, s=2, degrees=1, preproc="identity", family=fams)}})
    y = torch.randn(1, 1, 32, 32, dtype=torch.complex64)
    x_hat, (_, zz) = net(y, sigma=torch.full((1, 1, 1, 1), 0.05))
    check("3-family net: dual is 768 channels on the depth-3 grid",
          tuple(zz.shape) == (1, 768, 4, 4) and x_hat.shape == y.shape)


def test_random_family():
    """"random": the DT-CWT layout (groups, one-hot carries, tags, Q) with every
    filter drawn at random -- so the trees no longer start as shifts."""
    rw, rtags = FAMILIES["random"][0](7)
    dw, dtags = dtcwt_weights(7)
    check("random: same shapes and tags as dtcwt",
          [w.shape for w in rw] == [w.shape for w in dw] and rtags == dtags)
    carries_same = all(torch.equal(r.view(4, -1, *r.shape[1:])[:, 4:],
                                   d.view(4, -1, *d.shape[1:])[:, 4:])
                       for r, d in zip(rw[1:], dw[1:]))
    check("random: carries are dtcwt's one-hot kernels", carries_same)
    rows = [rw[0][:, 0]] + [w.view(4, -1, *w.shape[1:])[:, :4, 0].reshape(16, 7, 7)
                            for w in rw[1:]]
    unit = all(torch.allclose(r.flatten(1).norm(dim=1), torch.ones(16)) for r in rows)
    check("random: every filter row at unit norm", unit)
    flat = torch.cat(rows).flatten(1)
    flat = flat / flat.norm(dim=1, keepdim=True)
    off = (flat @ flat.T - torch.eye(len(flat))).abs().max().item()
    check("random: all 48 filters distinct (no tree is a copy)", off < 0.9, f"max |cos| {off:.2f}")
    again, _ = FAMILIES["random"][0](7)
    check("random: deterministic in its seed",
          all(torch.equal(a_, b_) for a_, b_ in zip(rw, again)))

    for carry in ("unshuffle", "conv"):
        lay = WaveletLPDSLayer(P=7, lam0=1e-3, degrees=1, family="random", carry=carry)
        n2 = lay.op_norm2()
        check(f"random ({carry}): ||K|| = 1", abs(n2 - 1) < 1e-2, f"||K||^2 = {n2:.4f}")
        convs = list(lay.analysis) + list(lay.synthesis)
        before = [c.weight.clone() for c in convs]
        lay.project_()
        check(f"random ({carry}): project() is a no-op at init",
              all(torch.allclose(b_, c.weight, atol=1e-5) for b_, c in zip(before, convs)))
        x = torch.randn(1, 1, 32, 32, dtype=torch.complex64)
        z = torch.randn(1, 256, 4, 4, dtype=torch.complex64)
        lhs, rhs = (lay.analyse(x).conj() * z).sum(), (x.conj() * lay.adjoint(z)).sum()
        check(f"random ({carry}): B = A^H", abs(lhs - rhs).item() < 1e-4 * abs(lhs).item())
    check("random: LL^3 thresholds start at 0",
          bool((lay.prox.prox.tau.weight[:, lay.ll_channels] == 0).all()))

    u = WaveletLPDSLayer(P=7, family=["dtcwt", "random"])
    nf = [u.op_norm2(family=f) for f in (0, 1)]
    check("random in a union: equal ||K_f||", max(nf) / min(nf) - 1 < 2e-2,
          ", ".join(f"{v:.4f}" for v in nf))


def test_random_full():
    """"random_full": the tree grouping and Q only -- every weight of every
    level random, no one-hot carries.  Needs carry="conv"."""
    fw, ftags = FAMILIES["random_full"][0](7)
    dw, dtags = dtcwt_weights(7)
    check("random_full: same shapes and tags as dtcwt",
          [w.shape for w in fw] == [w.shape for w in dw] and ftags == dtags)
    dense = all(bool((w != 0).all()) for w in fw)
    check("random_full: every weight nonzero (no one-hot carries, no zero blocks)", dense)
    rows = all(torch.allclose(w.flatten(1).norm(dim=1), torch.ones(w.shape[0]), atol=1e-5)
               for w in fw[1:])
    slices = max(w.flatten(2).norm(dim=2).max().item() for w in fw)
    check("random_full: deep output rows at unit norm, every slice inside the ball",
          rows and slices <= 1.0 + 1e-6, f"max slice norm {slices:.3f}")
    again, _ = FAMILIES["random_full"][0](7)
    check("random_full: deterministic in its seed",
          all(torch.equal(a_, b_) for a_, b_ in zip(fw, again)))
    for fam in ("random_full", ["dtcwt", "random_full"]):
        try:
            WaveletLPDSLayer(P=7, family=fam, carry="unshuffle")
            check(f"{fam!r} with carry='unshuffle' is refused", False)
        except ValueError:
            check(f"{fam!r} with carry='unshuffle' is refused", True)

    lay = WaveletLPDSLayer(P=7, lam0=1e-3, degrees=1, family="random_full", carry="conv")
    n2 = lay.op_norm2()
    check("random_full (conv): ||K|| = 1", abs(n2 - 1) < 1e-2, f"||K||^2 = {n2:.4f}")
    convs = list(lay.analysis) + list(lay.synthesis)
    before = [c.weight.clone() for c in convs]
    lay.project_()
    check("random_full (conv): project() is a no-op at init",
          all(torch.allclose(b_, c.weight, atol=1e-5) for b_, c in zip(before, convs)))
    x = torch.randn(1, 1, 32, 32, dtype=torch.complex64)
    z = torch.randn(1, 256, 4, 4, dtype=torch.complex64)
    lhs, rhs = (lay.analyse(x).conj() * z).sum(), (x.conj() * lay.adjoint(z)).sum()
    check("random_full (conv): B = A^H", abs(lhs - rhs).item() < 1e-4 * abs(lhs).item())

    net = build_model({"model": {"type": "WaveletLPDSNet", "params": dict(
        K=3, M=16, P=7, s=2, degrees=1, preproc="identity", family="random_full",
        carry="conv")}})
    y = torch.randn(1, 1, 32, 32, dtype=torch.complex64)
    x_hat, _ = net(y, sigma=torch.full((1, 1, 1, 1), 0.05))
    x_hat.abs().pow(2).sum().backward()
    g = net.layer(1).analysis[2].conv_real.weight.grad.view(4, 64, 16, -1).abs().sum(-1)
    # row 0 of each tree feeds LL^3, whose threshold starts at 0: its dual is
    # clipped to 0, so that row learns only once the threshold leaves 0
    check("random_full net: every level-3 block but the LL^3 rows gets gradient",
          bool((g[:, 1:] > 0).all()) and bool((g[:, 0] == 0).all()))


def test_dual_step():
    """dual_step="learned": filters at wavelet normalisation, the redundancy in
    sigma_d -- and the SAME algorithm as "absorbed" at init, exactly."""
    # Seeded here, not by test order: the power-iteration error is ~1e-3 of the
    # move and depends on y, so a y drawn after other tests can cross the bound.
    torch.manual_seed(0)
    y = torch.randn(2, 1, 32, 32, dtype=torch.complex64)
    sig = torch.full((2, 1, 1, 1), 0.015)
    cases = [("dtcwt", dict(M=16)), ("haar", dict(M=16, family="haar")),
             ("dtcwt+eq", dict(M=16, band_norm="equal")),
             ("dtcwt+haar", dict(M=32, family=["dtcwt", "haar"]))]
    # lam0 large enough that the dual path does real work (at the config's
    # 1e-3 the net moves y by ~0.1% and any error hides under that), and the
    # error is measured against how far the net moves y.
    def moved(net):
        with torch.no_grad():
            return net(y, sigma=sig)[0] - y
    for name, kw in cases:
        a = WaveletLPDSNet(K=6, degrees=1, lam0=0.3, preproc="identity", **kw)
        b = WaveletLPDSNet(K=6, degrees=1, lam0=0.3, preproc="identity",
                           dual_step="learned", **kw)
        da, db = moved(a), moved(b)
        err = ((da - db).norm() / da.norm()).item()
        check(f"learned == absorbed at init, whole net ({name})", err < 1e-3,
              f"rel err {err:.1e} of a {(da.norm() / y.norm()).item():.0%} move")
        if name == "dtcwt":                    # the check must be able to fail
            with torch.no_grad():              # undo the threshold rescale
                for lay in b.net.layers:
                    lay.prox.prox.tau.weight.mul_(float(lay.sigma_d.weight[0, 0]) ** -0.5)
            bad = ((da - moved(b)).norm() / da.norm()).item()
            check("... and it catches unscaled thresholds", bad > 1e-2, f"rel err {bad:.1e}")

        lay = b.layer(0)
        convs = list(lay.analysis) + list(lay.synthesis)
        before = [c.weight.clone() for c in convs]
        lay.project_()
        check(f"learned ({name}): project() is a no-op at init",
              all(torch.allclose(b_, c.weight, atol=1e-5) for b_, c in zip(before, convs)))
        sb_a, sb_b = a.layer(0).step_bound(), lay.step_bound()
        check(f"learned ({name}): same Condat-Vu step bound",
              abs(sb_a - sb_b) < 1e-2 * sb_a, f"{sb_a:.3f} vs {sb_b:.3f}")

    b = WaveletLPDSNet(K=2, M=16, family="haar", dual_step="learned", preproc="identity")
    n1 = b.layer(0).analysis[0].weight.abs().pow(2).sum((2, 3)).sqrt()
    check("learned (haar): level-1 filters stay at unit norm (not 0.5)",
          bool(torch.allclose(n1, torch.ones_like(n1), atol=1e-6)))
    sd = float(b.layer(0).sigma_d.weight[0, 0])
    check("learned (haar): sigma_d = 1/||K||^2 = 1/4", abs(sd - 0.25) < 1e-3, f"{sd:.4f}")
    try:
        WaveletLPDSLayer(dual_step="free")
        check("unknown dual_step is refused", False)
    except ValueError:
        check("unknown dual_step is refused", True)


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
    test_matches_dtcwt(family="dtcwt57", biort="near_sym_a")
    test_truncation()
    test_dtcwt57_crop()
    test_Q()
    test_layer_init()
    test_carry_unshuffle_matches_conv()
    test_interleaved_matches_gauss()
    test_adjoint_complex_weights()
    test_haar()
    test_band_norm()
    test_union()
    test_union_three()
    test_random_family()
    test_random_full()
    test_dual_step()
    test_net_forward_backward()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    sys.exit(1 if FAIL else 0)
