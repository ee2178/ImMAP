"""
`MGLPDSNet.PLANAR_STATE`: the iterates carried as real `[re; im]` tensors.

Run with `python -m tests.test_planar_state`.

The planar state is the SAME network -- same parameters, same map -- with the
complex pair (x, z) held as real tensors of twice the channels between layers.
So everything here is an equivalence check against the complex state, piece by
piece and then end to end:

1. a conv given a planar tensor returns the planar form of what it returns for
   the complex one (both conv kinds, both strides, under either COMPLEX_MODE);
2. the prox pairs channel m with channel m + M: the planar clip and the planar
   shrink equal the complex ones, per-channel and noise-adaptive thresholds
   included. NEGATIVE CONTROL: thresholding the two halves independently -- the
   mistake this layout invites -- must fail the same comparison;
3. the fused clip kernel's planar index arithmetic, emulated on the CPU (the
   kernel itself needs CUDA), and that the kernel declines safely here;
4. the network: outputs, latent and PARAMETER GRADIENTS agree for a flat stack
   and a V-cycle, Galerkin and rediscretized, on a size that needs the
   image-domain embedding too;
5. the flag is ignored where the planar state is not implemented (group prox,
   widened V-cycle), so those nets are untouched;
6. it does what it is for: the M-channel code is no longer converted around
   every conv.

CPU only, small problems, no LAPACK needed (the group case is skipped without).
"""

from __future__ import annotations

import math

import torch

from models.clip_triton import clip_modulus_planar
from models.components import Conv2d, ConvTranspose2d, _GaussConvNd, to_complex, to_planar
from models.mg_lpds import MGLPDSNet
from models.prox import FenchelProx, SoftThreshold
from operators import FFT2D, Mask, Sense
from operators.truncate import embed_operator
from physics.mask import make_acc_mask

FAIL = []
TOL = 2e-5                 # fp32: a different association of the same arithmetic
NET = dict(M=8, C=1, P=3, s=2, lam0=5e-2, tau0=0.5, theta0=0.5, alpha0=1.0,
           is_complex=True, degrees=1, preproc="kspace", resize_noise=True)


def check(name, ok, detail=""):
    print(f"[{'ok ' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def rel(a, b):
    a, b = a.detach(), b.detach()
    return float((a - b).abs().max() / (b.abs().max() + 1e-30))


def fdiv(a, b):
    return torch.div(a, b, rounding_mode="floor")


class planar_state:
    def __init__(self, on=True):
        self.on = on

    def __enter__(self):
        self.keep = MGLPDSNet.PLANAR_STATE
        MGLPDSNet.PLANAR_STATE = self.on

    def __exit__(self, *exc):
        MGLPDSNet.PLANAR_STATE = self.keep


def problem(h=64, w=48, coils=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    yy = torch.arange(h)[:, None] - h / 2
    xx = torch.arange(w)[None, :] - w / 2
    sm = []
    for c in range(coils):
        a = 2 * math.pi * c / coils
        mag = torch.exp(-((yy - 0.4 * h * math.sin(a)) ** 2 + (xx - 0.4 * w * math.cos(a)) ** 2)
                        / (2 * (0.4 * h) ** 2))
        sm.append(mag * torch.exp(1j * (a + 0.02 * (yy + xx))))
    sm = torch.stack(sm)
    sm = (sm / sm.abs().pow(2).sum(0, keepdim=True).sqrt()).to(torch.complex64)[None]
    x = torch.complex(torch.randn(1, 1, h, w, generator=g), torch.randn(1, 1, h, w, generator=g))
    m = make_acc_mask((h, w), accel=4, acs_lines=8, dim=1, mode="uniform").float()
    E = Mask(m) @ FFT2D() @ Sense(sm)
    return E.forward(x), E


# ---------------------------------------------------------------------------
def test_convs():
    for cls, args, tag in ((Conv2d, (2, 5, 7), "Conv2d"),
                           (ConvTranspose2d, (5, 2, 7), "ConvTranspose2d")):
        for s in (1, 2):
            torch.manual_seed(0)
            m = cls(*args, stride=s)
            x = torch.randn(2, args[0], 16, 16, dtype=torch.complex64)
            for mode in ("gauss", "planar"):
                keep = _GaussConvNd.COMPLEX_MODE
                _GaussConvNd.COMPLEX_MODE = mode
                try:
                    ref = to_planar(m(x))
                    got = m(to_planar(x))
                finally:
                    _GaussConvNd.COMPLEX_MODE = keep
                check(f"{tag} stride={s} [{mode}]: planar in -> planar out == complex path",
                      got.shape == ref.shape and not got.is_complex() and rel(got, ref) < TOL,
                      f"rel {rel(got, ref):.1e}")
    torch.manual_seed(0)
    m = Conv2d(2, 5, 3, stride=1)
    r = torch.randn(1, 2, 8, 8)                       # a genuinely REAL input: C channels
    check("a real input with C channels is still the real-input path (complex out)",
          m(r).is_complex())
    g = Conv2d(4, 4, 3, groups=2)
    try:
        g(torch.randn(1, 8, 8, 8))
        check("a planar input to a grouped conv is refused", False, "no error")
    except ValueError:
        check("a planar input to a grouped conv is refused", True)


def test_prox():
    torch.manual_seed(0)
    M = 6
    z = torch.randn(2, M, 8, 8, dtype=torch.complex64)
    z.view(-1)[::7] = 0                                   # exercise z = 0
    st = SoftThreshold(M, tau0=0.4, degrees=1)
    with torch.no_grad():
        st.tau.weight[0] = torch.linspace(0.1, 1.2, M)   # a different threshold per channel
        st.tau.weight[1] = 3.0                            # and noise-adaptive
    sig = torch.tensor([0.02, 0.05]).view(2, 1, 1, 1)
    zp = to_planar(z)

    clip_c = st.fenchel(z, sig)[0]
    clip_p = st.fenchel(zp, sig)[0]
    check("planar clip == complex clip (per-channel, noise-adaptive threshold)",
          rel(clip_p, to_planar(clip_c)) < TOL, f"rel {rel(clip_p, to_planar(clip_c)):.1e}")
    shr_c, shr_p = st(z, sig)[0], st(zp, sig)[0]
    check("planar shrink == complex shrink",
          rel(shr_p, to_planar(shr_c)) < TOL, f"rel {rel(shr_p, to_planar(shr_c)):.1e}")
    check("clip + shrink == identity in planar form too (Moreau)",
          rel(clip_p + shr_p, zp) < TOL)
    fp = FenchelProx(st)
    check("FenchelProx routes a planar code to the planar clip",
          torch.equal(fp(zp, sig)[0], clip_p))
    with torch.no_grad():                                 # the no-grad (hypot) branch
        check("...and the no-grad branch gives the same clip",
              rel(st.fenchel(zp, sig)[0], clip_p) < TOL)

    # exact zeros (a zero-padded border has them in every embedded batch) must
    # not poison the gradient: hypot / sqrt give 0/0 there
    zz = torch.zeros(1, 2 * M, 4, 4, requires_grad=True)
    out = st.fenchel(zz, 0.03)[0] + st(zz, 0.03)[0]
    out.sum().backward()
    check("the gradient at z = 0 is finite (clip and shrink)",
          bool(torch.isfinite(zz.grad).all())
          and all(bool(torch.isfinite(p.grad).all()) for p in st.parameters()
                  if p.grad is not None))

    # NEGATIVE CONTROL: the halves thresholded independently (|re| and |im|
    # each clipped at t) is a different map, and this comparison must see it
    t = st.threshold(z, sig)
    t2 = torch.cat((t, t), dim=1)
    wrong = zp * (t2 / zp.abs().clamp_min(1e-12)).clamp_max(1.0)
    check("  control: clipping re and im independently is caught",
          rel(wrong, to_planar(clip_c)) > 1e-2, f"rel {rel(wrong, to_planar(clip_c)):.2f}")

    # the fused kernel: declines here, and its index arithmetic emulated
    check("clip_modulus_planar declines on CPU rather than raising",
          clip_modulus_planar(zp, st.threshold(zp, 0.03), st.fenchel_eps) is None)
    t1 = st.threshold(zp, 0.03)                           # (1, M, 1, 1)
    B, HW = zp.shape[0], zp.shape[2] * zp.shape[3]
    n, MHW = zp.numel() // 2, M * HW
    flat = zp.contiguous().reshape(-1)
    i = torch.arange(n)
    off = i + fdiv(i, MHW) * MHW
    re, im = flat[off], flat[off + MHW]
    tf = t1.reshape(-1)[fdiv(i, HW) % M]
    s = torch.clamp(tf / torch.clamp((re * re + im * im).sqrt(), min=st.fenchel_eps), max=1.0)
    out = torch.empty_like(flat)
    out[off], out[off + MHW] = re * s, im * s
    check("planar kernel arithmetic (emulated) == the eager planar clip",
          rel(out.reshape(zp.shape), st.fenchel(zp, 0.03)[0]) < 1e-6
          and sorted(torch.cat((off, off + MHW)).tolist()) == list(range(zp.numel())),
          "every float written exactly once")
    bad = i + fdiv(i, MHW) * HW                           # control: wrong batch stride
    check("  control: a wrong batch offset would not cover the tensor",
          B > 1 and sorted(torch.cat((bad, bad + MHW)).tolist()) != list(range(zp.numel())))


def run(net, y, E, planar, grad=False):
    sig = torch.full((1, 1, 1, 1), 0.02)
    with planar_state(planar):
        if not grad:
            with torch.no_grad():
                xh, (x, z) = net.eval()(y, E=E, sigma=sig)
            return xh, z, None
        net.train().zero_grad(set_to_none=True)
        xh, (x, z) = net(y, E=E, sigma=sig)
        xh.abs().pow(2).mean().backward()
        return xh.detach(), z.detach(), {n: p.grad.clone() for n, p in net.named_parameters()
                                         if p.grad is not None}


def test_network():
    y, E = problem()
    y_e, E0 = problem(64, 44)                             # 44 columns -> embedded to 48
    E_e, T = embed_operator(E0, (64, 44), 8)
    cases = (("flat K=4", dict(K=4), y, E),
             ("V-cycle, galerkin", dict(K=[2, [2, 2, 2]]), y, E),
             ("V-cycle, rediscretize", dict(K=[2, [2, 2, 2]], coarse_op="rediscretize"), y, E),
             ("V-cycle, rediscretize, embedded 64x44", dict(K=[1, [2, 2, 2]],
                                                           coarse_op="rediscretize"), y_e, E_e))
    for tag, kws, yy, EE in cases:
        torch.manual_seed(0)
        net = MGLPDSNet(**dict(NET, **kws))
        with planar_state(True):
            active = net.planar_state_active()
        a, za, _ = run(net, yy, EE, False)
        b, zb, _ = run(net, yy, EE, True)
        check(f"{tag}: planar state == complex state (output and code)",
              active and b.is_complex() and zb.is_complex() and a.shape == b.shape
              and rel(b, a) < TOL and rel(zb, za) < TOL,
              f"x rel {rel(b, a):.1e}, z rel {rel(zb, za):.1e}")
        _, _, ga = run(net, yy, EE, False, grad=True)
        _, _, gb = run(net, yy, EE, True, grad=True)
        worst = max(rel(gb[k], ga[k]) for k in ga if float(ga[k].abs().max()) > 0)
        check(f"{tag}: every parameter gradient agrees",
              ga.keys() == gb.keys() and worst < 1e-3, f"{len(ga)} tensors, worst rel {worst:.1e}")

    # warm start: a complex state handed back in is accepted and matches
    torch.manual_seed(0)
    net = MGLPDSNet(**dict(NET, K=[1, [2, 2, 2]])).eval()
    sig = torch.full((1, 1, 1, 1), 0.02)
    with torch.no_grad():
        _, st0 = net(y, E=E, sigma=sig)
        ref = net(y, E=E, sigma=sig, state=st0)[0]
        with planar_state(True):
            got = net(y, E=E, sigma=sig, state=st0)[0]
    check("a complex warm-start state is accepted in planar mode", rel(got, ref) < TOL,
          f"rel {rel(got, ref):.1e}")


def test_ignored_where_unsupported():
    y, E = problem()
    for tag, kws in (("widened V-cycle", dict(K=[1, [2, 2, 2]], widen=2)),
                     ("group prox", dict(K=[1, [2, 2]], window=5, Mh=4, attn_backend="gather"))):
        try:
            torch.manual_seed(0)
            net = MGLPDSNet(**dict(NET, **kws))
        except Exception as e:                                    # noqa: BLE001
            print(f"[skip] {tag}: could not build here ({type(e).__name__}: {e})")
            continue
        with planar_state(True):
            active = net.planar_state_active()
        a, _, _ = run(net, y, E, False)
        b, _, _ = run(net, y, E, True)
        check(f"{tag}: the flag is ignored, output bit-identical",
              not active and torch.equal(a, b))


class Conversions:
    """Count the complex<->planar conversions of big tensors in a block."""

    def __enter__(self):
        import models.components as comp
        import models.lpds as lpds_mod
        import models.mg_lpds as mg_mod
        self.mods = (comp, lpds_mod, mg_mod)
        self.keep = [(m, n, getattr(m, n)) for m in self.mods
                     for n in ("to_planar", "to_complex") if hasattr(m, n)]
        self.code = self.image = 0
        outer = self

        def wrap(fn):
            def inner(x):
                changed = (fn.__name__ == "to_planar") == torch.is_complex(x)
                if changed:
                    if x.shape[1] > 2:
                        outer.code += 1
                    else:
                        outer.image += 1
                return fn(x)
            return inner

        for m, n, f in self.keep:
            setattr(m, n, wrap(f))
        return self

    def __exit__(self, *exc):
        for m, n, f in self.keep:
            setattr(m, n, f)


def test_conversions():
    y, E = problem()
    torch.manual_seed(0)
    net = MGLPDSNet(**dict(NET, K=[2, [2, 2, 2]], coarse_op="rediscretize")).eval()
    sig = torch.full((1, 1, 1, 1), 0.02)
    keep = _GaussConvNd.COMPLEX_MODE
    _GaussConvNd.COMPLEX_MODE = "planar"
    try:
        counts = {}
        for planar in (False, True):
            with planar_state(planar), torch.no_grad(), Conversions() as c:
                net(y, E=E, sigma=sig)
            counts[planar] = (c.code, c.image)
    finally:
        _GaussConvNd.COMPLEX_MODE = keep
    (code_c, img_c), (code_p, img_p) = counts[False], counts[True]
    check("planar state: the M-channel code is converted ONCE (on the way out), "
          "not around every conv",
          code_p == 1 and code_c > 20, f"code conversions {code_c} -> {code_p}; "
          f"image-sized {img_c} -> {img_p}")


def main():
    for fn in (test_convs, test_prox, test_network, test_ignored_where_unsupported,
               test_conversions):
        print(f"\n--- {fn.__name__}")
        fn()
    print(f"\n{'FAILED: ' + ', '.join(FAIL) if FAIL else 'all checks passed'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
