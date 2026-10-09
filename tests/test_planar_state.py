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
5. the flag is ignored where the planar state is not implemented (a widened
   V-cycle), so that net is untouched. The GROUP prox is supported and has its
   own file, tests/test_planar_group.py;
6. it does what it is for: the M-channel code is no longer converted around
   every conv;
7. the fused prox WITH ITS BACKWARD (`SoftThreshold.FUSED_GRAD`): the
   hand-written derivative against autograd through the eager chain, clip and
   shrink, every threshold layout; the two kernels' arithmetic and addressing
   emulated program by program; and the network trained through the autograd
   Function (`clip_triton.EMULATE`) against the eager planar chain. NEGATIVE
   CONTROL: a derivative without its radial term must fail.

CPU only, small problems, no LAPACK needed (the group case is skipped without).
"""

from __future__ import annotations

import math

import torch

import models.clip_triton as clip_mod
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


class emulate:
    """Run the planar prox through `clip_triton`'s autograd Function on its
    torch formulas -- the kernels' arithmetic, without the kernels."""

    def __enter__(self):
        self.keep = clip_mod.EMULATE
        clip_mod.EMULATE = True

    def __exit__(self, *exc):
        clip_mod.EMULATE = self.keep


def prox_grads(st, zp, sig, g, dual):
    """`(out, dL/dz, dL/dtau.weight)` for L = <g, prox(z)>."""
    z = zp.clone().requires_grad_(True)
    st.zero_grad(set_to_none=True)
    out = (st.fenchel if dual else st)(z, sig)[0]
    (out * g).sum().backward()
    return out, z.grad.clone(), st.tau.weight.grad.clone()


def kernel_forward(zp, tf, TN, eps, dual, BLOCK=24):
    """`_prox_kernel_planar`, program by program. -> (out, floats written)."""
    B, M, HW = zp.shape[0], zp.shape[1] // 2, zp.shape[2] * zp.shape[3]
    n, MHW = B * M * HW, M * HW
    flat, out, hits = zp.reshape(-1), torch.zeros(zp.numel()), torch.zeros(zp.numel())
    for pid in range(-(-n // BLOCK)):
        i = pid * BLOCK + torch.arange(BLOCK)
        i = i[i < n]
        off = i + fdiv(i, MHW) * MHW
        re, im = flat[off], flat[off + MHW]
        q = tf[fdiv(i, HW) % TN] / torch.clamp((re * re + im * im).sqrt(), min=eps)
        s = torch.clamp(q, max=1.0) if dual else torch.clamp(1.0 - q, min=0.0)
        out[off], out[off + MHW] = re * s, im * s
        hits[off] += 1
        hits[off + MHW] += 1
    return out.reshape(zp.shape), hits


def kernel_backward(zp, tf, TN, g, eps, dual, BLOCK=24):
    """`_prox_backward_planar`, program by program.
    -> (dz, dt per (batch, channel), floats written, slots written)."""
    B, M, HW = zp.shape[0], zp.shape[1] // 2, zp.shape[2] * zp.shape[3]
    MHW, NB = M * HW, -(-HW // BLOCK)
    Z, G = zp.reshape(-1), g.reshape(-1)
    GZ, hits = torch.zeros(zp.numel()), torch.zeros(zp.numel())
    PART, slots = torch.zeros(B * M * NB), torch.zeros(B * M * NB)
    for r in range(B * M):
        for pb in range(NB):
            j = pb * BLOCK + torch.arange(BLOCK)
            j = j[j < HW]
            off = (r + (r // M) * M) * HW + j
            re, im, gr, gi = Z[off], Z[off + MHW], G[off], G[off + MHW]
            t = tf[r % TN]
            a = (re * re + im * im).sqrt()
            ac = torch.clamp(a, min=eps)
            q = t / ac
            d = (gr * re + gi * im) / ac
            zero = torch.zeros(())
            if dual:
                act = q <= 1.0
                s, w = torch.where(act, q, torch.ones(())), torch.where(act, d, zero)
            else:
                act = q < 1.0
                s, w = torch.where(act, 1.0 - q, zero), torch.where(act, -d, zero)
            k = torch.where(a >= eps, w * q / ac, zero)
            GZ[off], GZ[off + MHW] = gr * s - k * re, gi * s - k * im
            hits[off] += 1
            hits[off + MHW] += 1
            PART[r * NB + pb] = w.sum()
            slots[r * NB + pb] += 1
    return (GZ.reshape(zp.shape), PART.view(B, M, NB).sum(2).view(B, M, 1, 1), hits, slots)


def test_fused_grad():
    """The planar prox as one kernel forward and one backward: the hand-written
    derivative, the kernels' index arithmetic, and the Function wiring."""
    torch.manual_seed(0)
    B, M = 2, 6
    zp = to_planar(torch.randn(B, M, 8, 8, dtype=torch.complex64))
    zp.view(-1)[::7] = 0                                  # zero halves, and ...
    zp[:, :, :2] = 0                                      # ... exactly-zero pairs (a padded border)
    g = torch.randn_like(zp)
    eps = SoftThreshold.fenchel_eps

    def make(degrees, m=M):
        st = SoftThreshold(m, tau0=0.4, degrees=degrees)
        with torch.no_grad():
            st.tau.weight[0] = torch.linspace(0.1, 1.2, m)
            if degrees:
                st.tau.weight[1] = 3.0
        return st

    cases = (("per channel (1, M)", make(0), None, M),
             ("noise-adaptive, one sigma (1, M)", make(1), 0.03, M),
             ("noise-adaptive, per batch (B, M)", make(1),
              torch.tensor([0.02, 0.05]).view(B, 1, 1, 1), B * M))
    for dual in (True, False):
        kind = "clip" if dual else "shrink"
        for tag, st, sig, TN in cases:
            ref_out, ref_dz, ref_dt = prox_grads(st, zp, sig, g, dual)   # eager chain + autograd
            used = "PlanarProx" in type(ref_out.grad_fn).__name__
            with emulate():
                out, dz, dt = prox_grads(st, zp, sig, g, dual)
                fn = type(out.grad_fn).__name__
                with torch.no_grad():
                    out_ng = (st.fenchel if dual else st)(zp, sig)[0]
            check(f"{kind}, {tag}: the Function is taken, value == the eager chain",
                  "PlanarProx" in fn and not used and rel(out, ref_out) < 1e-6
                  and rel(out_ng, ref_out) < 1e-6, f"grad_fn {fn}")
            check(f"{kind}, {tag}: hand-written dL/dz and dL/dtau == autograd",
                  rel(dz, ref_dz) < 1e-5 and rel(dt, ref_dt) < 1e-5
                  and bool(torch.isfinite(dz).all()),
                  f"dz rel {rel(dz, ref_dz):.1e}, dtau rel {rel(dt, ref_dt):.1e}")

            # the kernels' own arithmetic and addressing, program by program
            t4 = st.threshold(zp, sig).detach()
            tf = (t4.expand(B, M, 1, 1) if t4.shape[0] > 1 else t4).reshape(-1)
            k_out, hits = kernel_forward(zp, tf, tf.numel(), eps, dual)
            k_dz, k_dt, bhits, slots = kernel_backward(zp, tf, tf.numel(), g, eps, dual)
            e_dz, e_dt = clip_mod._eager_backward(zp, t4, g, eps, dual)
            check(f"{kind}, {tag}: kernel arithmetic (emulated), forward and backward",
                  tf.numel() == TN and rel(k_out, ref_out) < 1e-6 and rel(k_dz, ref_dz) < 1e-5
                  and rel(k_dt, e_dt) < 1e-5
                  and bool((hits == 1).all()) and bool((bhits == 1).all())
                  and bool((slots == 1).all()), "every float and every slot written once")

    # NEGATIVE CONTROLS: the comparison must see a wrong derivative ...
    st, sig = cases[1][1], cases[1][2]
    _, ref_dz, _ = prox_grads(st, zp, sig, g, True)
    t4 = st.threshold(zp, sig).detach()
    out = clip_mod._eager_forward(zp, t4, eps, True)
    pairs = zp.reshape(B, 2, M, 8, 8)
    s = (t4 / torch.hypot(pairs[:, 0], pairs[:, 1]).clamp_min(eps)).clamp_max(1.0)
    no_radial = (g.reshape(B, 2, M, 8, 8) * s.unsqueeze(1)).reshape(zp.shape)
    check("  control: dropping the radial term (dz = s g) is caught",
          rel(no_radial, ref_dz) > 1e-2, f"rel {rel(no_radial, ref_dz):.2f}")
    # ... and a wrong row offset in the backward kernel
    tf = t4.reshape(-1)
    k_dz = kernel_backward(zp, tf, M, g, eps, True)[0]
    check("  control: the (emulated) kernel differs for another incoming gradient",
          rel(kernel_backward(zp, tf, M, g.flip(0), eps, True)[0], k_dz) > 1e-2
          and rel(out, prox_grads(st, zp, sig, g, True)[0]) < 1e-6)

    # a scalar threshold (TN = 1), and the gradient at exact zeros
    st1 = SoftThreshold(1, tau0=0.7)
    z1 = to_planar(torch.randn(2, 1, 6, 6, dtype=torch.complex64))
    g1 = torch.randn_like(z1)
    ref = prox_grads(st1, z1, None, g1, True)
    with emulate():
        got = prox_grads(st1, z1, None, g1, True)
    check("a single threshold (TN = 1): value and both gradients",
          all(rel(a, b) < 1e-5 for a, b in zip(got, ref)))
    with emulate():
        zz = torch.zeros(1, 2 * M, 4, 4, requires_grad=True)
        stz = make(0)
        (stz.fenchel(zz, None)[0] + stz(zz, None)[0]).sum().backward()
    check("the gradient at z = 0 is finite and passes g through (clip + shrink = id)",
          bool(torch.isfinite(zz.grad).all()) and rel(zz.grad, torch.ones_like(zz)) < 1e-6
          and bool(torch.isfinite(stz.tau.weight.grad).all()))

    # where it must NOT be taken
    st = make(1)
    z = zp.clone().requires_grad_(True)
    check("without CUDA (and without EMULATE) the eager chain runs",
          clip_mod.prox_planar_grad(z, st.threshold(z, 0.03), eps) is None
          and clip_mod.prox_planar(zp, st.threshold(zp, 0.03), eps, dual=False) is None)
    with emulate():
        check("a full-resolution threshold map is declined",
              clip_mod.prox_planar_grad(z, torch.rand(B, M, 8, 8), eps) is None)
        check("a non-contiguous code is declined",
              clip_mod.prox_planar_grad(z.transpose(-1, -2), st.threshold(z, 0.03), eps) is None)
        keep = SoftThreshold.FUSED_GRAD
        SoftThreshold.FUSED_GRAD = False
        try:
            off = st.fenchel(z, 0.03)[0]
        finally:
            SoftThreshold.FUSED_GRAD = keep
        check("FUSED_GRAD = False keeps the eager chain under autograd",
              "PlanarProx" not in type(off.grad_fn).__name__)
        keep = SoftThreshold.FUSED
        SoftThreshold.FUSED = False
        try:
            off = st.fenchel(z, 0.03)[0]
        finally:
            SoftThreshold.FUSED = keep
        check("FUSED = False switches it off too (the one kill switch)",
              "PlanarProx" not in type(off.grad_fn).__name__)
    check("nothing was marked as a kernel on the CPU",
          clip_mod.planar_kernel_report() == "fwd not used, bwd not used",
          clip_mod.planar_kernel_report())


def test_network_fused_grad():
    """End to end: the network trained through the Function gives the same
    gradients as through the eager planar chain (and the complex state)."""
    import models.prox as prox_mod

    y, E = problem()
    y_e, E0 = problem(64, 44)
    E_e, T = embed_operator(E0, (64, 44), 8)
    for tag, kws, yy, EE in (("flat K=4", dict(K=4), y, E),
                             ("V-cycle, rediscretize, embedded 64x44",
                              dict(K=[1, [2, 2, 2]], coarse_op="rediscretize"), y_e, E_e)):
        torch.manual_seed(0)
        net = MGLPDSNet(**dict(NET, **kws))
        _, _, g_cplx = run(net, yy, EE, False, grad=True)
        a, za, g_eager = run(net, yy, EE, True, grad=True)

        calls = dict(clip=0, shrink=0)
        real = prox_mod.prox_planar_grad

        def counted(z, t, eps, dual=True):
            out = real(z, t, eps, dual)
            if out is not None:
                calls["clip" if dual else "shrink"] += 1
            return out

        prox_mod.prox_planar_grad = counted
        try:
            with emulate():
                b, zb, g_fused = run(net, yy, EE, True, grad=True)
        finally:
            prox_mod.prox_planar_grad = real
        worst = max(rel(g_fused[k], g_eager[k]) for k in g_eager
                    if float(g_eager[k].abs().max()) > 0)
        worst_c = max(rel(g_fused[k], g_cplx[k]) for k in g_cplx
                      if float(g_cplx[k].abs().max()) > 0)
        n_layers = sum(1 for m in net.modules() if isinstance(m, FenchelProx))
        check(f"{tag}: trained through the Function == the eager planar chain",
              g_fused.keys() == g_eager.keys() and rel(b, a) < TOL and rel(zb, za) < TOL
              and worst < 1e-4 and worst_c < 1e-3 and calls["clip"] >= n_layers > 0,
              f"{calls['clip']} clips through it; worst gradient rel {worst:.1e} "
              f"(vs complex {worst_c:.1e})")
        # The primal-dual V-cycle's FAS correction has no prox subgradient
        # (models/mg_lpds.py), so an LPDS net only ever CLIPS; the shrink is
        # covered by test_fused_grad.
        check(f"{tag}: an LPDS net never calls the shrink", calls["shrink"] == 0)


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
    for tag, kws in (("widened V-cycle", dict(K=[1, [2, 2, 2]], widen=2)),):
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
    for fn in (test_convs, test_prox, test_fused_grad, test_network,
               test_network_fused_grad, test_ignored_where_unsupported, test_conversions):
        print(f"\n--- {fn.__name__}")
        fn()
    print(f"\n{'FAILED: ' + ', '.join(FAIL) if FAIL else 'all checks passed'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
