"""
The planar state and the fused kernel for the GROUP prox.

Run with `python -m tests.test_planar_group`.

`GroupThreshold` on a planar code (real, `[re; im]` on the channel axis) is the
same map as on the complex one, and its last step -- both halves scaled by
`s(tau / (xi + eps))` -- runs as one kernel each way. Checked here, against
the complex state throughout:

1. the pieces: the real 1x1 transforms on a planar map (one conv for both
   halves), the per-head layout flex wants, |.|^2 pairing the halves;
2. the prox: planar clip and shrink == complex, grouped and ungrouped, one
   head and two, noise-adaptive thresholds; Moreau; the closed-form
   subgradients. What reaches the attention is the SAME (q, k) on every
   backend, and flex still refuses a phase-invariant similarity. NEGATIVE
   CONTROL: an envelope built from the halves separately must fail;
3. the fused step: the hand-written backward (to the code, to the envelope and
   to the threshold) against autograd through the eager chain; both kernels'
   arithmetic and addressing emulated program by program; and the prox run
   through the autograd Function (`clip_triton.EMULATE`). NEGATIVE CONTROL: a
   backward without the envelope's gradient must fail;
4. the network: output, code and every PARAMETER GRADIENT agree between the
   complex state, the planar state on the eager chain and the planar state
   through the Function -- flat, V-cycle, Galerkin and rediscretized;
5. the gate: a prox SUBCLASS (a guided prox, say) keeps the net on the complex
   state.

CPU only; the attention runs on the `gather` backend (flex needs CUDA), which
is why (2) checks flex's inputs rather than its output.
"""

from __future__ import annotations

import torch

import models.clip_triton as clip_mod
import models.prox as prox_mod
from models.components import to_complex, to_planar
from models.mg_lpds import MGLPDSNet
from models.prox import FenchelProx, GroupThreshold, PixelConv, SoftThreshold
from tests.test_planar_state import NET, emulate, fdiv, planar_state, problem, rel, run

FAIL = []
TOL = 2e-5


def check(name, ok, detail=""):
    print(f"[{'ok ' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def make_gt(M=6, Mh=4, nheads=1, **kw):
    torch.manual_seed(0)
    gt = GroupThreshold(M, Mh=Mh, nheads=nheads, window=5, tau0=0.3, degrees=1,
                        attn_backend="gather", **kw)
    with torch.no_grad():
        gt.tau.weight[0] = torch.linspace(0.05, 0.6, M)
        gt.tau.weight[1] = 2.0
    return gt


def code(M=6, B=2, hw=(8, 8)):
    z = torch.randn(B, M, *hw, dtype=torch.complex64)
    z[:, :, :2] = 0                                    # a padded border: exact zeros
    return z


# ---------------------------------------------------------------------------
def test_pieces():
    torch.manual_seed(0)
    for groups in (1, 2):
        pc = PixelConv(6, 4, groups=groups)
        x = torch.randn(2, 6, 5, 7, dtype=torch.complex64)
        u = torch.randn(2, 4, 5, 7, dtype=torch.complex64)
        a, b = pc(to_planar(x)), to_planar(pc(x))
        at, bt = pc.transpose_apply(to_planar(u)), to_planar(pc.transpose_apply(u))
        check(f"PixelConv groups={groups}: planar in -> planar out == complex, W and W^T",
              a.shape == b.shape and not a.is_complex() and rel(a, b) < TOL
              and at.shape == bt.shape and rel(at, bt) < TOL,
              f"rel {rel(a, b):.1e}, {rel(at, bt):.1e}")
        r = torch.randn(2, 6, 5, 7)
        check(f"PixelConv groups={groups}: a real input of cin channels is unchanged",
              torch.equal(pc(r), pc.conv(r)))
    for h in (1, 2):
        gt = make_gt(M=8, Mh=4, nheads=h)
        x = torch.randn(2, 4, 5, 6, dtype=torch.complex64)
        check(f"nheads={h}: the planar map in flex's per-head layout == _stack_ri(complex)",
              torch.equal(gt._heads_ri(to_planar(x)), gt._stack_ri(x)))
    x = torch.randn(2, 4, 5, 6, dtype=torch.complex64)
    check("|.|^2 of a planar map pairs channel m with m + C",
          rel(GroupThreshold._abs2_planar(to_planar(x)), x.abs() ** 2) < TOL)


def test_prox():
    sig = torch.tensor([0.02, 0.05]).view(2, 1, 1, 1)
    for tag, kw in (("grouped, 1 head", dict()),
                    ("grouped, 2 heads", dict(M=8, Mh=4, nheads=2)),
                    ("ungrouped (Mh=None)", dict(Mh=None)),
                    ("pidistance", dict(sim_fun="pidistance"))):
        gt = make_gt(**kw)
        z = code(gt.M)
        zp = to_planar(z)
        with torch.no_grad():
            shr_c, shr_p = gt(z, sig, {})[0], gt(zp, sig, {})[0]
            clip_c, clip_p = FenchelProx(gt)(z, sig, {})[0], FenchelProx(gt)(zp, sig, {})[0]
        check(f"{tag}: planar shrink and clip == complex",
              not shr_p.is_complex() and rel(shr_p, to_planar(shr_c)) < TOL
              and rel(clip_p, to_planar(clip_c)) < TOL and rel(clip_p + shr_p, zp) < TOL,
              f"rel {rel(shr_p, to_planar(shr_c)):.1e}, {rel(clip_p, to_planar(clip_c)):.1e}")
        if gt.grouped:
            for mode in ("simple", "rigorous"):
                with torch.no_grad():
                    gc, gp = gt.subgradient(z, sig, {}, mode)[0], gt.subgradient(zp, sig, {}, mode)[0]
                check(f"{tag}: the {mode} subgradient of a planar code == complex",
                      rel(gp, to_planar(gc)) < TOL)

    # the adjacency cache across layers (dK): three calls sharing one cache
    gt = make_gt(dK=2)
    z = code()
    with torch.no_grad():
        cc, cp, oc, op = {}, {}, [], []
        for i in range(3):
            zi = z * (1 + 0.1 * i)
            o, cc = gt(zi, sig, cc)
            p, cp = gt(to_planar(zi), sig, cp)
            oc.append(o)
            op.append(p)
    check("the adjacency cache (dK=2, blended on gather) follows the same schedule",
          all(rel(p, to_planar(o)) < TOL for o, p in zip(oc, op))
          and cc["dupdate"] == cp["dupdate"])

    # NEGATIVE CONTROL: the envelope from each half separately (what treating
    # the planar code as 2M real channels would compute) is a different map
    gt = make_gt()
    z = code()
    with torch.no_grad():
        ref = to_planar(gt(z, sig, {})[0])
        halves = torch.cat((gt(torch.complex(z.real, torch.zeros_like(z.real)), sig, {})[0].real,
                            gt(torch.complex(z.imag, torch.zeros_like(z.imag)), sig, {})[0].real), 1)
    check("  control: thresholding the halves separately is caught",
          rel(halves, ref) > 1e-2, f"rel {rel(halves, ref):.2f}")

    # what reaches the attention, per backend
    seen = []

    class FakeFlex:
        def __init__(self, q, k, *a, **kw):
            seen.append((q.detach().clone(), k.detach().clone()))

    keep = prox_mod.FlexAdjacency
    prox_mod.FlexAdjacency = FakeFlex
    try:
        for h in (1, 2):
            gt = make_gt(M=8, Mh=4, nheads=h)
            gt.attn_backend = "flex"
            gt._flex_block_mask = lambda ref, win=None: None
            z = code(8)
            seen.clear()
            gt._build_gamma(z, sig)
            gt._build_gamma(to_planar(z), sig)
            (qc, kc), (qp, kp) = seen
            check(f"flex, {h} head(s): the planar state hands it the same (q, k)",
                  not qp.is_complex() and qc.shape == qp.shape
                  and rel(qp, qc) < TOL and rel(kp, kc) < TOL)
        gt = make_gt(sim_fun="pidistance")
        gt.attn_backend = "flex"
        refused = 0
        for zz in (code(), to_planar(code())):
            try:
                gt._build_gamma(zz, sig)
            except ValueError:
                refused += 1
        check("flex still refuses a phase-invariant similarity, complex or planar",
              refused == 2, f"{refused}/2")
    finally:
        prox_mod.FlexAdjacency = keep


# ---------------------------------------------------------------------------
def eager_chain(z, xi, t, eps, dual):
    """The chain `GroupThreshold._planar_prox` runs when the kernel declines."""
    B, C2, H, W = z.shape
    q = t / (xi + eps)
    s = q.clamp_max(1.0) if dual else torch.relu(1.0 - q)
    return (z.reshape(B, 2, C2 // 2, H, W) * s.unsqueeze(1)).reshape(z.shape)


def kernel_forward(zp, xi, tf, TN, eps, dual, BLOCK=24):
    """`_scale_kernel_planar`, program by program."""
    B, M, HW = zp.shape[0], zp.shape[1] // 2, zp.shape[2] * zp.shape[3]
    n, MHW = B * M * HW, M * HW
    flat, X = zp.reshape(-1), xi.reshape(-1)
    out, hits = torch.zeros(zp.numel()), torch.zeros(zp.numel())
    for pid in range(-(-n // BLOCK)):
        i = pid * BLOCK + torch.arange(BLOCK)
        i = i[i < n]
        off = i + fdiv(i, MHW) * MHW
        q = tf[fdiv(i, HW) % TN] / (X[i] + eps)
        s = torch.clamp(q, max=1.0) if dual else torch.clamp(1.0 - q, min=0.0)
        out[off], out[off + MHW] = flat[off] * s, flat[off + MHW] * s
        hits[off] += 1
        hits[off + MHW] += 1
    return out.reshape(zp.shape), hits


def kernel_backward(zp, xi, tf, TN, g, eps, dual, BLOCK=24):
    """`_scale_backward_planar`, program by program."""
    B, M, HW = zp.shape[0], zp.shape[1] // 2, zp.shape[2] * zp.shape[3]
    MHW, NB = M * HW, -(-HW // BLOCK)
    Z, G, X = zp.reshape(-1), g.reshape(-1), xi.reshape(-1)
    GZ, hits = torch.zeros(zp.numel()), torch.zeros(zp.numel())
    GXI, xhits = torch.zeros(xi.numel()), torch.zeros(xi.numel())
    PART, slots = torch.zeros(B * M * NB), torch.zeros(B * M * NB)
    zero = torch.zeros(())
    for r in range(B * M):
        for pb in range(NB):
            j = pb * BLOCK + torch.arange(BLOCK)
            j = j[j < HW]
            off = (r + (r // M) * M) * HW + j
            xo = r * HW + j
            re, im, gr, gi = Z[off], Z[off + MHW], G[off], G[off + MHW]
            den = X[xo] + eps
            q = tf[r % TN] / den
            d = (gr * re + gi * im) / den
            if dual:
                act = q <= 1.0
                s, w = torch.where(act, q, torch.ones(())), torch.where(act, d, zero)
            else:
                act = q < 1.0
                s, w = torch.where(act, 1.0 - q, zero), torch.where(act, -d, zero)
            GZ[off], GZ[off + MHW] = gr * s, gi * s
            GXI[xo] = -w * q
            PART[r * NB + pb] = w.sum()
            hits[off] += 1
            hits[off + MHW] += 1
            xhits[xo] += 1
            slots[r * NB + pb] += 1
    return (GZ.reshape(zp.shape), GXI.reshape(xi.shape),
            PART.view(B, M, NB).sum(2).view(B, M, 1, 1), hits, xhits, slots)


def test_fused_step():
    torch.manual_seed(0)
    B, M, eps = 2, 6, 1e-8
    zp = to_planar(code(M))
    xi = torch.rand(B, M, 8, 8) * 2.0                   # straddles the thresholds below
    xi[:, :, :1] = 0                                    # an envelope of exactly zero
    g = torch.randn_like(zp)
    for dual in (True, False):
        kind = "group clip" if dual else "group shrink"
        for tag, t in (("per channel (1, M)", torch.linspace(0.2, 1.6, M).view(1, M, 1, 1)),
                       ("per batch (B, M)", (torch.linspace(0.2, 1.6, B * M)).view(B, M, 1, 1)),
                       ("one threshold", torch.tensor(0.9).view(1, 1, 1, 1))):
            z_, x_, t_ = (v.clone().requires_grad_(True) for v in (zp, xi, t))
            ref = eager_chain(z_, x_, t_, eps, dual)
            (ref * g).sum().backward()                           # autograd, eager chain
            with emulate():
                z2, x2, t2 = (v.clone().requires_grad_(True) for v in (zp, xi, t))
                out = clip_mod.scale_planar_grad(z2, x2, t2, eps, dual)
                (out * g).sum().backward()
                with torch.no_grad():
                    out_ng = clip_mod.scale_planar(zp, xi, t, eps, dual)
            fn = type(out.grad_fn).__name__
            check(f"{kind}, {tag}: the Function's value and its three gradients == autograd",
                  "PlanarScale" in fn and rel(out, ref) < 1e-6 and rel(out_ng, ref) < 1e-6
                  and rel(z2.grad, z_.grad) < 1e-5 and rel(x2.grad, x_.grad) < 1e-5
                  and rel(t2.grad, t_.grad) < 1e-5 and bool(torch.isfinite(x2.grad).all()),
                  f"dz {rel(z2.grad, z_.grad):.1e}, dxi {rel(x2.grad, x_.grad):.1e}, "
                  f"dt {rel(t2.grad, t_.grad):.1e}")

            tf = (t.expand(B, M, 1, 1) if t.numel() == B else t).reshape(-1)
            k_out, hits = kernel_forward(zp, xi, tf, tf.numel(), eps, dual)
            k_dz, k_dx, k_dt, bh, xh, slots = kernel_backward(zp, xi, tf, tf.numel(), g, eps, dual)
            e_dz, e_dx, e_dt = clip_mod._eager_scale_backward(zp, xi, t, g, eps, dual)
            check(f"{kind}, {tag}: kernel arithmetic (emulated), forward and backward",
                  rel(k_out, ref) < 1e-6 and rel(k_dz, z_.grad) < 1e-5
                  and rel(k_dx, x_.grad) < 1e-5 and rel(k_dt, e_dt) < 1e-5
                  and all(bool((v == 1).all()) for v in (hits, bh, xh, slots)),
                  "every float of dz, every envelope gradient and every slot written once")

    # NEGATIVE CONTROL: the envelope carries the whole dependence of the scale
    # on the code, so a backward that returned no gradient for it is wrong
    t = torch.linspace(0.2, 1.6, M).view(1, M, 1, 1)
    x_ = xi.clone().requires_grad_(True)
    (eager_chain(zp, x_, t, eps, True) * g).sum().backward()
    check("  control: the envelope's gradient is not negligible",
          float(x_.grad.abs().max()) > 1e-3)

    check("without CUDA (and without EMULATE) the eager chain runs",
          clip_mod.scale_planar(zp, xi, t, eps) is None
          and clip_mod.scale_planar_grad(zp, xi, t, eps) is None)
    with emulate():
        check("an envelope of the wrong shape is declined",
              clip_mod.scale_planar(zp, xi[:, :3], t, eps) is None
              and clip_mod.scale_planar_grad(zp, torch.rand(B, 2 * M, 8, 8), t, eps) is None)

    # the prox through the Function, parameters and all
    sig = torch.tensor([0.02, 0.05]).view(2, 1, 1, 1)
    for tag, kw in (("grouped", dict()), ("2 heads", dict(M=8, Mh=4, nheads=2))):
        gt = make_gt(**kw)
        z0 = to_planar(code(gt.M))
        gg = torch.randn_like(z0)

        def grads(dual):
            z = z0.clone().requires_grad_(True)
            gt.zero_grad(set_to_none=True)
            out = (gt.fenchel(z, sig, {}) if dual else gt(z, sig, {}))[0]
            (out * gg).sum().backward()
            return out, z.grad.clone(), {n: p.grad.clone() for n, p in gt.named_parameters()
                                         if p.grad is not None}

        for dual in (True, False):
            ro, rz, rp = grads(dual)
            with emulate():
                o, dz, dp = grads(dual)
            worst = max(rel(dp[k], rp[k]) for k in rp if float(rp[k].abs().max()) > 0)
            check(f"GroupThreshold ({tag}, {'clip' if dual else 'shrink'}) through the "
                  f"Function == the eager chain",
                  "PlanarScale" in type(o.grad_fn).__name__
                  and "PlanarScale" not in type(ro.grad_fn).__name__
                  and rel(o, ro) < 1e-6 and rel(dz, rz) < 1e-5 and worst < 1e-4
                  and rp.keys() == dp.keys(),
                  f"{len(rp)} parameter gradients, worst rel {worst:.1e}")
    with emulate():
        z = z0.clone().requires_grad_(True)
        keep = SoftThreshold.FUSED_GRAD
        SoftThreshold.FUSED_GRAD = False
        try:
            off = gt.fenchel(z, sig, {})[0]
        finally:
            SoftThreshold.FUSED_GRAD = keep
    check("FUSED_GRAD = False keeps the group prox on the eager chain",
          "PlanarScale" not in type(off.grad_fn).__name__)


# ---------------------------------------------------------------------------
GROUP = dict(window=5, Mh=4, dK=2, attn_backend="gather")


def test_network():
    y, E = problem()
    for tag, kws in (("flat K=3", dict(K=3)),
                     ("V-cycle, galerkin", dict(K=[1, [2, 2, 2]])),
                     ("V-cycle, rediscretize", dict(K=[1, [2, 2, 2]], coarse_op="rediscretize")),
                     ("V-cycle, 2 heads", dict(K=[1, [2, 2]], nheads=2))):
        torch.manual_seed(0)
        net = MGLPDSNet(**dict(NET, **GROUP, **kws))
        with planar_state(True):
            active = net.planar_state_active()
        a, za, _ = run(net, y, E, False)
        b, zb, _ = run(net, y, E, True)
        check(f"group {tag}: planar state == complex state (output and code)",
              active and b.is_complex() and rel(b, a) < TOL and rel(zb, za) < TOL,
              f"x rel {rel(b, a):.1e}, z rel {rel(zb, za):.1e}")
        _, _, ga = run(net, y, E, False, grad=True)
        _, _, gb = run(net, y, E, True, grad=True)

        calls = [0]
        real = prox_mod.scale_planar_grad

        def counted(*a_, **k_):
            out = real(*a_, **k_)
            calls[0] += out is not None
            return out

        prox_mod.scale_planar_grad = counted
        try:
            with emulate():
                c, zc, gc = run(net, y, E, True, grad=True)
        finally:
            prox_mod.scale_planar_grad = real
        nz = [k for k in ga if float(ga[k].abs().max()) > 0]
        w_eager = max(rel(gb[k], ga[k]) for k in nz)
        w_fused = max(rel(gc[k], ga[k]) for k in nz)
        n_prox = sum(1 for m in net.modules() if isinstance(m, FenchelProx))
        check(f"group {tag}: every parameter gradient agrees -- eager chain and Function",
              ga.keys() == gb.keys() == gc.keys() and w_eager < 1e-3 and w_fused < 1e-3
              and rel(c, a) < TOL and calls[0] >= n_prox > 0,
              f"{len(ga)} tensors, worst rel {w_eager:.1e} / {w_fused:.1e}; "
              f"{calls[0]} group clips through the Function")


def test_gate():
    class Guided(GroupThreshold):
        pass

    torch.manual_seed(0)
    net = MGLPDSNet(**dict(NET, **GROUP, K=[1, [2, 2]]))
    with planar_state(True):
        on = net.planar_state_active()
        slot = next(m for m in net.modules() if isinstance(m, FenchelProx))
        slot.prox.__class__ = Guided
        net._planar_proxes_ok = None
        off = net.planar_state_active()
        slot.prox.__class__ = GroupThreshold
    check("a prox SUBCLASS in any slot keeps the net on the complex state", on and not off)
    y, E = problem()
    torch.manual_seed(0)
    wide = MGLPDSNet(**dict(NET, **GROUP, K=[1, [2, 2]], widen=2))
    with planar_state(True):
        active = wide.planar_state_active()
    a, _, _ = run(wide, y, E, False)
    b, _, _ = run(wide, y, E, True)
    check("a widened group V-cycle still ignores the flag, output bit-identical",
          not active and torch.equal(a, b))


def main():
    for fn in (test_pieces, test_prox, test_fused_step, test_network, test_gate):
        print(f"\n--- {fn.__name__}")
        fn()
    print(f"\n{'FAILED: ' + ', '.join(FAIL) if FAIL else 'all checks passed'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
