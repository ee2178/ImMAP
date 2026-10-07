"""
`_GaussConvNd.COMPLEX_MODE`: Gauss's 3-multiply trick vs one real conv on a
planar `[re; im]` channel stacking.  And `SoftThreshold.FUSED`, the Fenchel
clip as one Triton kernel.

Run with `python -m tests.test_planar_conv`.

The two conv formulations are the same linear map, so everything here is an
equivalence check -- including the places it would be easy to get subtly wrong:

1. the BLOCK PATTERN differs between conv and conv_transpose, because
   conv_transpose2d's weight is (in, out, ...) rather than (out, in, ...).
   Getting the two confused transposes the imaginary sign and is invisible on a
   real-valued test input, so every check here is complex;
2. GRADIENTS must reach conv_real and conv_imag and agree, since the point of
   the change is to use it in training too;
3. `groups > 1` (wavelet_lpds) and a non-None bias are NOT expressible and must
   fall back to "gauss" rather than quietly returning something else.  The bias
   case is checked against the TRUE complex conv, which exposes a pre-existing
   inconsistency in the gauss branch -- see `_planar_bias`;
4. the whole network, end to end.

Each equivalence check has a NEGATIVE CONTROL (the block pattern built wrong),
so a pass means the check can see the error it is looking for.

CPU only.  The fused prox needs CUDA + triton and declines everywhere else, so
what is checked here is that it declines SAFELY and that its emulated
arithmetic matches the eager chain it replaces.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.clip_triton import HAVE_TRITON, clip_modulus
from models.components import Conv2d, ConvTranspose2d, _GaussConvNd, to_complex, to_planar
from models.mg_lpds import MGLPDSNet
from models.prox import SoftThreshold
from operators import FFT2D, Mask, Sense
from physics.mask import make_acc_mask

FAIL = []
TOL = 1e-5                 # fp32: planar reassociates every complex multiply


def check(name, ok, detail=""):
    print(f"[{'ok ' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def rel(a, b):
    a, b = a.detach(), b.detach()
    return float((a - b).abs().max() / (b.abs().max() + 1e-30))


def under(mode, fn, *a, **k):
    orig = _GaussConvNd.COMPLEX_MODE
    _GaussConvNd.COMPLEX_MODE = mode
    try:
        return fn(*a, **k)
    finally:
        _GaussConvNd.COMPLEX_MODE = orig


# ===========================================================================
def test_conv_equivalence():
    for cls, args, tag in ((Conv2d, (2, 5, 7), "Conv2d"),
                           (ConvTranspose2d, (5, 2, 7), "ConvTranspose2d")):
        for s in (1, 2):
            torch.manual_seed(0)
            m = cls(*args, stride=s)
            x = torch.randn(1, args[0], 16, 16, dtype=torch.complex64)
            g = under("gauss", m, x)
            p = under("planar", m, x)
            check(f"{tag} stride={s}: planar == gauss",
                  g.shape == p.shape and p.is_complex() and rel(p, g) < TOL,
                  f"rel {rel(p, g):.2e}, shape {tuple(p.shape)}")


def test_negative_control():
    """A wrong block pattern must FAIL the same comparison."""
    torch.manual_seed(0)
    m = Conv2d(2, 5, 7, stride=2)
    x = torch.randn(1, 2, 16, 16, dtype=torch.complex64)
    g = under("gauss", m, x)

    orig = _GaussConvNd._planar_weight

    def flipped(self):
        wr, wi = self.conv_real.weight, self.conv_imag.weight
        # the sign error: [[wr, wi], [wi, wr]] is not a complex multiply
        return torch.cat((torch.cat((wr, wi), 1), torch.cat((wi, wr), 1)), 0)

    _GaussConvNd._planar_weight = flipped
    try:
        bad = under("planar", m, x)
    finally:
        _GaussConvNd._planar_weight = orig
    check("negative control: a wrong block pattern is detected",
          rel(bad, g) > 1e-2, f"rel {rel(bad, g):.2e} (must be large)")

    def swapped(self):
        """The conv pattern used for conv_transpose -- i.e. forgetting that its
        weight is (in, out, ...)."""
        wr, wi = self.conv_real.weight, self.conv_imag.weight
        return torch.cat((torch.cat((wr, -wi), 1), torch.cat((wi, wr), 1)), 0)

    torch.manual_seed(0)
    mt = ConvTranspose2d(5, 2, 7, stride=2)
    z = torch.randn(1, 5, 8, 8, dtype=torch.complex64)
    gt = under("gauss", mt, z)
    _GaussConvNd._planar_weight = swapped
    try:
        badt = under("planar", mt, z)
    finally:
        _GaussConvNd._planar_weight = orig
    check("negative control: conv's block pattern is wrong for conv_transpose",
          rel(badt, gt) > 1e-2, f"rel {rel(badt, gt):.2e} (must be large)")


def test_gradients():
    for cls, args, tag in ((Conv2d, (2, 5, 7), "Conv2d"),
                           (ConvTranspose2d, (5, 2, 7), "ConvTranspose2d")):
        torch.manual_seed(0)
        m = cls(*args, stride=2)
        x = torch.randn(1, args[0], 16, 16, dtype=torch.complex64)

        def grads(mode):
            for p in m.parameters():
                p.grad = None
            under(mode, lambda: m(x).abs().pow(2).sum().backward())
            return [p.grad.clone() for p in m.parameters()]

        g, p = grads("gauss"), grads("planar")
        worst = max(rel(b, a) for a, b in zip(g, p))
        check(f"{tag}: gradients agree and reach conv_real + conv_imag",
              len(p) == 2 and all(float(v.abs().max()) > 0 for v in p)
              and worst < TOL, f"worst rel {worst:.2e} over {len(p)} tensors")


def test_fallbacks():
    """groups > 1 and a bias are not expressible; both must stay on gauss."""
    torch.manual_seed(0)
    mg = Conv2d(4, 8, 3, stride=1, groups=2)
    x = torch.randn(1, 4, 8, 8, dtype=torch.complex64)
    check("groups=2: planar declines and matches gauss bit-for-bit",
          not under("planar", mg._planar_ok)
          and rel(under("planar", mg, x), under("gauss", mg, x)) == 0.0)

    torch.manual_seed(0)
    mb = Conv2d(2, 4, 3, stride=1, bias=True)
    xb = torch.randn(1, 2, 8, 8, dtype=torch.complex64)
    check("bias: planar declines",
          not under("planar", mb._planar_ok))

    # ... because gauss's bias is inconsistent with a complex bias, so letting
    # planar handle it WOULD change the answer. Pinned so the day someone fixes
    # the gauss side, this test says what to do about it.
    wr, wi = mb.conv_real.weight, mb.conv_imag.weight
    br, bi = mb.conv_real.bias, mb.conv_imag.bias
    truth = F.conv2d(xb, torch.complex(wr, wi), bias=torch.complex(br, bi),
                     padding=1)
    got = under("gauss", mb, xb)
    check("gauss's complex bias is wrong (real gets br - bi, imag loses bi)",
          rel(got, truth) > 1e-2,
          f"rel {rel(got, truth):.2e} vs the true complex conv")


def test_planar_roundtrip():
    x = torch.randn(2, 3, 5, 5, dtype=torch.complex64)
    xp = to_planar(x)
    check("to_planar / to_complex round-trip",
          xp.shape == (2, 6, 5, 5) and not xp.is_complex()
          and rel(to_complex(xp), x) == 0.0)
    r = torch.randn(2, 3, 5, 5)
    check("to_planar passes a real tensor through", to_planar(r) is r)


def test_network_end_to_end():
    H, W, C = 32, 32, 4
    torch.manual_seed(0)
    sm = torch.randn(1, C, H, W, dtype=torch.complex64)
    sm = sm / sm.abs().pow(2).sum(1, keepdim=True).sqrt()
    m = make_acc_mask((H, W), accel=4, acs_lines=8, dim=1, mode="uniform").float()
    E = Mask(m.reshape(1, 1, H, W)) @ FFT2D() @ Sense(sm)
    x = torch.randn(1, 1, H, W, dtype=torch.complex64)
    y = E.forward(x)

    def run(mode):
        torch.manual_seed(0)
        net = MGLPDSNet(K=[2, [2, 2, 2]], M=8, C=1, P=3, s=2, lam0=1e-3,
                        tau0=0.5, theta0=0.0, alpha0=1.0, is_complex=True,
                        degrees=1, preproc="kspace", resize_noise=True).eval()
        with torch.no_grad():
            return under(mode, net, y, E=E, sigma=0.005)[0]

    g, p = run("gauss"), run("planar")
    check("MGLPDSNet forward: planar == gauss", rel(p, g) < TOL,
          f"rel {rel(p, g):.2e}")


class CatCount:
    """Count `torch.cat` calls in a block (the block weight is three of them)."""

    def __enter__(self):
        self.n, self._cat = 0, torch.cat

        def cat(*a, **k):
            self.n += 1
            return self._cat(*a, **k)

        torch.cat = cat
        return self

    def __exit__(self, *exc):
        torch.cat = self._cat


def test_weight_cache():
    """The planar block weight is reused between NO-GRAD calls and never goes
    stale: every way the weights change in this repo must reach the next call.
    Each case compares against a freshly built module with the same weights
    (`fresh`), so a stale cache is a numerical mismatch, not just a flag."""
    for cls, args, tag in ((Conv2d, (2, 5, 7), "Conv2d"),
                           (ConvTranspose2d, (5, 2, 7), "ConvTranspose2d")):
        torch.manual_seed(0)
        m = cls(*args, stride=2).eval()
        x = torch.randn(1, args[0], 16, 16, dtype=torch.complex64)

        def fresh():
            f = cls(*args, stride=2).to(m.conv_real.weight.dtype).eval()
            f.load_state_dict(m.state_dict())
            with torch.no_grad():
                return f(x)

        def run():
            with torch.no_grad():
                return m(x)

        def same(a, b):
            return a.dtype == b.dtype and torch.equal(a, b)

        with CatCount() as c1:
            a = under("planar", run)
        with CatCount() as c2:
            b = under("planar", run)
        check(f"{tag}: the first no-grad call builds the block weight, the second reuses it",
              (c1.n, c2.n) == (4, 1) and torch.equal(a, b), f"cat calls {c1.n} then {c2.n}")
        W1 = under("planar", lambda: torch.no_grad()(m._planar_weight)())
        check(f"{tag}: ...as the SAME tensor, bit-identical to a rebuild",
              W1 is under("planar", lambda: torch.no_grad()(m._planar_weight)())
              and torch.equal(W1, m._build_planar_weight()))

        # -- under autograd: never cached, gradients reach both halves --------
        with CatCount() as c3:
            out = under("planar", m, x)
        out.abs().pow(2).sum().backward()
        gr, gi = m.conv_real.weight.grad, m.conv_imag.weight.grad
        check(f"{tag}: with grad the weight is rebuilt and gradients reach real and imag",
              c3.n == 4 and gr is not None and gi is not None
              and float(gr.abs().sum()) > 0 and float(gi.abs().sum()) > 0)

        # -- every writer must invalidate -------------------------------------
        m.weight = m.weight * 0.5                               # the property setter
        check(f"{tag}: the weight setter (set_weight / project_ / init) invalidates",
              same(under("planar", run), under("planar", fresh)))
        with torch.no_grad():                                   # what an optimizer does
            m.conv_real.weight.add_(0.1)
            m.conv_imag.weight.mul_(-1.0)
        check(f"{tag}: an in-place optimizer-style update invalidates",
              same(under("planar", run), under("planar", fresh)))
        sd = {k: torch.randn_like(v) for k, v in m.state_dict().items()}
        m.load_state_dict(sd)
        check(f"{tag}: load_state_dict invalidates",
              same(under("planar", run), under("planar", fresh)))
        m.double()
        x = x.to(torch.complex128)
        check(f"{tag}: a dtype change (.double()) invalidates",
              under("planar", run).dtype == torch.complex128
              and same(under("planar", run), under("planar", fresh)))
        m.float()
        x = x.to(torch.complex64)
        check(f"{tag}: ...and so does changing back",
              same(under("planar", run), under("planar", fresh)))

        # -- the documented hole, and what closes it ---------------------------
        under("planar", run)                                    # cache is warm
        m.conv_real.weight.data.mul_(3.0)                       # raw .data: no version bump
        stale = under("planar", run)
        check(f"{tag}: control -- a raw .data write is NOT seen (so the checks above "
              f"can fail), and train()/eval() clears it",
              not same(stale, under("planar", fresh))
              and same(under("planar", lambda: (m.train(), m.eval(), run())[2]),
                       under("planar", fresh)))

        # -- kill switch --------------------------------------------------------
        _GaussConvNd.PLANAR_WEIGHT_CACHE = False
        try:
            with CatCount() as c4:
                off = under("planar", run)
        finally:
            _GaussConvNd.PLANAR_WEIGHT_CACHE = True
        check(f"{tag}: PLANAR_WEIGHT_CACHE=False rebuilds every call, same answer",
              c4.n == 4 and torch.equal(off, under("planar", run)))


def test_weight_cache_network():
    """End to end: a V-cycle net's no-grad forward is bit-identical with the
    cache on and off, and the second forward builds no block weight at all."""
    H, W, C = 32, 32, 4
    torch.manual_seed(0)
    sm = torch.randn(1, C, H, W, dtype=torch.complex64)
    sm = sm / sm.abs().pow(2).sum(1, keepdim=True).sqrt()
    m = make_acc_mask((H, W), accel=4, acs_lines=8, dim=1, mode="uniform").float()
    E = Mask(m.reshape(1, 1, H, W)) @ FFT2D() @ Sense(sm)
    y = E.forward(torch.randn(1, 1, H, W, dtype=torch.complex64))
    net = MGLPDSNet(K=[2, [2, 2, 2]], M=8, C=1, P=3, s=2, lam0=1e-3, tau0=0.5,
                    theta0=0.0, alpha0=1.0, is_complex=True, degrees=1,
                    preproc="kspace", resize_noise=True).eval()

    def run():
        with torch.no_grad():
            return net(y, E=E, sigma=0.005)[0]

    with CatCount() as c1:
        a = under("planar", run)
    with CatCount() as c2:
        b = under("planar", run)
    _GaussConvNd.PLANAR_WEIGHT_CACHE = False
    try:
        with CatCount() as c3:
            off = under("planar", run)
    finally:
        _GaussConvNd.PLANAR_WEIGHT_CACHE = True
    convs = sum(1 for mod in net.modules() if isinstance(mod, _GaussConvNd))
    check("MGLPDSNet: cached forward is bit-identical to the uncached one",
          torch.equal(a, b) and torch.equal(a, off))
    check("MGLPDSNet: the second forward saves three `cat`s per conv call",
          c1.n == c3.n and c3.n - c2.n == 3 * (c3.n // 4) and c2.n == c3.n // 4,
          f"{c3.n} uncached, {c2.n} cached; {convs} conv modules")
    net.project()                                               # what training does
    check("MGLPDSNet: project() between forwards is picked up",
          torch.equal(under("planar", run), _uncached(run)))


def _uncached(run):
    _GaussConvNd.PLANAR_WEIGHT_CACHE = False
    try:
        return under("planar", run)
    finally:
        _GaussConvNd.PLANAR_WEIGHT_CACHE = True


def test_fused_prox():
    torch.manual_seed(0)
    z = torch.randn(2, 6, 8, 8, dtype=torch.complex64)
    z.view(-1)[::7] = 0                                   # exercise z = 0
    st = SoftThreshold(6, tau0=0.2, degrees=1)
    t = st.threshold(z, 0.01)

    check("clip_modulus declines on CPU rather than raising",
          clip_modulus(z, t, st.fenchel_eps) is None,
          f"HAVE_TRITON={HAVE_TRITON}, cuda={z.is_cuda}")
    check("clip_modulus declines a full-resolution threshold map",
          clip_modulus(z, torch.rand_like(z.real), st.fenchel_eps) is None)
    check("clip_modulus declines a non-contiguous input",
          clip_modulus(z.transpose(-1, -2), t, st.fenchel_eps) is None)

    # FUSED must be a no-op wherever the kernel declines
    st.FUSED = True
    with torch.no_grad():
        a = st.fenchel(z, 0.01)[0]
    st.FUSED = False
    with torch.no_grad():
        b = st.fenchel(z, 0.01)[0]
    check("FUSED is bit-identical where the kernel declines", rel(a, b) == 0.0)

    # the arithmetic the kernel performs, emulated in torch index math: this is
    # what fails if `c = (i // HW) % M` addresses the wrong channel
    n, HW, M = z.numel(), z.shape[2] * z.shape[3], z.shape[1]
    zr = torch.view_as_real(z.contiguous()).reshape(-1)
    i = torch.arange(n)
    re, im = zr[2 * i], zr[2 * i + 1]
    tf = t.reshape(-1)[(i // HW) % M]
    s = torch.clamp(tf / torch.clamp((re * re + im * im).sqrt(),
                                     min=st.fenchel_eps), max=1.0)
    out = torch.empty(2 * n)
    out[2 * i], out[2 * i + 1] = re * s, im * s
    emu = torch.view_as_complex(out.reshape(-1, 2)).reshape(z.shape)
    check("kernel arithmetic (emulated) == the eager Fenchel chain",
          rel(emu, b) < 1e-6, f"rel {rel(emu, b):.2e}")

    # channel addressing: with t nonzero on ONE channel, only that one survives
    t1 = torch.zeros(1, M, 1, 1)
    t1[0, 2] = 1e9
    tf1 = t1.reshape(-1)[(i // HW) % M]
    s1 = torch.clamp(tf1 / torch.clamp((re * re + im * im).sqrt(),
                                       min=st.fenchel_eps), max=1.0)
    o1 = torch.empty(2 * n)
    o1[2 * i], o1[2 * i + 1] = re * s1, im * s1
    e1 = torch.view_as_complex(o1.reshape(-1, 2)).reshape(z.shape)
    kept = [float(e1[:, c].abs().max()) for c in range(M)]
    check("kernel channel addressing picks the right channel",
          kept[2] > 0 and max(kept[:2] + kept[3:]) == 0.0,
          f"per-channel max {[round(v, 3) for v in kept]}")

    # phase is preserved, modulus is clipped at t -- the defining property
    mag = b.abs()
    live = mag > 1e-6
    check("clip preserves phase and bounds |z| by t",
          bool((mag <= t.reshape(1, M, 1, 1) + 1e-5).all())
          and torch.allclose(torch.angle(b[live]), torch.angle(z[live]), atol=1e-4))


def main():
    for fn in (test_conv_equivalence, test_negative_control, test_gradients,
               test_fallbacks, test_planar_roundtrip, test_network_end_to_end,
               test_weight_cache, test_weight_cache_network, test_fused_prox):
        print(f"\n--- {fn.__name__}")
        fn()
    print(f"\n{'FAILED: ' + ', '.join(FAIL) if FAIL else 'all checks passed'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
