"""
The two V-cycle correctness fixes of exp15 (2026-10-22).

LATTICE (`transfer_filter="sinc_vertex"`, operators/resample.py)
  1. the vertex-centred sinc is an exact transfer pair: P = 4 R^T, R(1) = 1,
     P(1) = 1, and P interpolates (a coarse sample lands unchanged on its fine
     sample);
  2. it has the cell-centred sinc's |H| to within 0.02;
  3. the IMAGE path (restrict x, then the stride-2 P=7 conv) and the LATENT path
     (the conv, then restrict z) land on the same lattice at every level --
     the cell-centred sinc is off by 0.5 / 1.5 / 3.5 fine px;
  4. `use_filter_length` and `GridTransfer` handle the odd length.

FAS (`fas_dual`, models/mg_lpds.py::FAS_DUAL)
  5. COHERENCE AT THE SWEEP: with the fine residual zero, one coarse sweep
     must leave (x_c, z_c) where it is. "fixed_point" (A) always does;
     "gradient" (B) does while z_c is in the coarse dual set and not past it;
     "legacy" does not. Real and complex codes. (tests/test_mg_lpds.py's
     `test_fas_consistency` checks the residual identity only, which every
     mode satisfies -- that is how the legacy placement went unnoticed.)
  6. the same end to end through `PDObjectiveDownsample`, from a constructed
     FINE fixed point (z = t sgn(Ax), y = x + Dz);
  7. `dF` forms pi_z by the right formula for each mode;
  8. MGLPDSNet: legacy is bit-identical to the default, every coarse layer
     carries the mode, `set_fas_dual` swaps and restores, the gradient form is
     refused for the group prox, and a vertex-lattice rediscretized net runs
     end to end with a gradient.

Run with `python -m tests.test_fas_lattice`.
"""

from __future__ import annotations

import math
import sys

import torch
import torch.nn.functional as F

from models.lpds import LPDSLayer, make_lpds_layer
from models.mg_lpds import MGLPDSNet, PDVCycle, first_pd_layer, set_fas_dual
from operators import FFT2D, Identity, Mask, Sense
from operators.resample import (GridTransfer, filter_length, prolong, restrict,
                                transfer_kernel, use_filter_length)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"[{'ok ' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")


def rel(a, b, eps=1e-12):
    return float((a - b).abs().max() / (b.abs().max() + eps))


# ---------------------------------------------------------------------------
#  LATTICE
# ---------------------------------------------------------------------------
def test_vertex_pair():
    d = torch.float64
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 3, 32, 48, dtype=d, generator=g)
    w = torch.randn(2, 3, 16, 24, dtype=d, generator=g)
    f = "sinc_vertex"
    lhs = (restrict(x, filter=f) * w).sum()
    rhs = (x * prolong(w, filter=f)).sum() / 4
    check("sinc_vertex: P = 4 R^T", abs(float(lhs - rhs)) < 1e-10, f"{float(lhs - rhs):.1e}")
    one = torch.ones(1, 1, 32, 48, dtype=d)
    check("sinc_vertex: R(1) = 1 and P(1) = 1",
          float((restrict(one, filter=f) - 1).abs().max()) < 1e-12
          and float((prolong(torch.ones(1, 1, 16, 24, dtype=d), filter=f) - 1).abs().max()) < 1e-12)
    Pw = prolong(w, filter=f)
    check("sinc_vertex: P interpolates (coarse sample = its fine sample)",
          rel(Pw[..., ::2, ::2], w) < 1e-12, f"rel={rel(Pw[..., ::2, ::2], w):.1e}")
    x32 = torch.randn(1, 2, 32, 32, dtype=torch.complex128, generator=g)
    check("sinc_vertex: complex input, shape halves",
          restrict(x32, filter=f).shape == (1, 2, 16, 16)
          and prolong(restrict(x32, filter=f), filter=f).shape == x32.shape)
    check("sinc_vertex is 7 taps at factor 2", filter_length(f) == 7)


def test_vertex_response():
    def resp(k, wfreq):
        h = k[0, 0].sum(0)                         # separable: one axis
        h = h / h.sum()
        t = torch.arange(len(h), dtype=torch.float64) - (len(h) - 1) / 2
        return abs(complex((h * torch.cos(wfreq * t)).sum(), -(h * torch.sin(wfreq * t)).sum()))
    kc = transfer_kernel("sinc", dtype=torch.float64)
    kv = transfer_kernel("sinc_vertex", dtype=torch.float64)
    gap = max(abs(resp(kc, f * math.pi) - resp(kv, f * math.pi))
              for f in (0.25, 0.4, 0.5, 0.6, 0.75, 1.0))
    check("sinc_vertex |H| within 0.02 of the cell-centred sinc", gap < 0.02, f"max gap {gap:.3f}")


def _offsets(filt, levels=3, N=128, P=7):
    d = torch.float64
    wd = torch.zeros(1, 1, P, P, dtype=d)
    wd[0, 0, P // 2, P // 2] = 1
    A = lambda t: F.conv2d(t, wd, stride=2, padding=(P - 1) // 2)  # noqa: E731
    ramp = torch.arange(N, dtype=d)[:, None].expand(N, N)[None, None].contiguous()
    x, z, out = ramp, A(ramp), []
    for _ in range(levels):
        x, z = restrict(x, filter=filt), restrict(z, filter=filt)
        zi = A(x)
        r, c = 3, zi.shape[-1] // 2              # interior row, away from the wrap
        out.append(float(z[0, 0, r, c] - zi[0, 0, r, c]))
    return out


def test_lattice_alignment():
    off_c, off_v = _offsets("sinc"), _offsets("sinc_vertex")
    check("cell-centred sinc: image/latent lattices off by 0.5, 1.5, 3.5 fine px",
          all(abs(a - b) < 1e-9 for a, b in zip(off_c, (0.5, 1.5, 3.5))), str(off_c))
    check("sinc_vertex: lattices coincide at every level",
          all(abs(a) < 1e-9 for a in off_v), str(off_v))


def test_vertex_length_and_learned():
    with use_filter_length("sinc_vertex", 11) as taps:
        ok = taps == 11 and filter_length("sinc_vertex") == 11
        one = torch.ones(1, 1, 32, 32, dtype=torch.float64)
        ok = ok and float((prolong(restrict(one, filter="sinc_vertex"),
                                   filter="sinc_vertex") - 1).abs().max()) < 1e-12
    check("use_filter_length: 11-tap sinc_vertex, still exact on constants", ok)
    check("use_filter_length restores 7 taps", filter_length("sinc_vertex") == 7)
    try:
        with use_filter_length("sinc_vertex", 8):
            pass
        check("use_filter_length refuses an even vertex length", False)
    except ValueError:
        check("use_filter_length refuses an even vertex length", True)

    # The GridTransfer kernel is built in float32 (a buffer), so agreement is to
    # float32 precision -- the cell-centred sinc differs by the same ~1e-7.
    gt = GridTransfer(channels=1, filter="sinc_vertex", learn=False).double()
    x = torch.randn(1, 3, 16, 16, dtype=torch.float64)
    a = rel(gt.restrict(x), restrict(x, filter="sinc_vertex"))
    b = rel(gt.prolong(x), prolong(x, filter="sinc_vertex"))
    check("GridTransfer(sinc_vertex) == module-level restrict / prolong",
          a < 1e-6 and b < 1e-6 and gt.prolong(gt.restrict(x)).shape == x.shape,
          f"R {a:.1e}  P {b:.1e}")


# ---------------------------------------------------------------------------
#  FAS
# ---------------------------------------------------------------------------
def _fixed_dual(layer, z, sigma):
    """`z` projected into the layer's dual set -- feasible, with some entries
    on the boundary and some inside."""
    return layer.prox(z, sigma, {})[0]


def _sweep_change(layer, mode, x_c, z_c, y, sigma):
    """One coarse sweep from (x_c, z_c) with the fine residual taken as ZERO:
    pi = the coarse residual itself (`R r_fine = 0`)."""
    with torch.no_grad():
        pi_x = x_c - y + layer.synthesis(z_c)                         # E = I
        Ax = layer.analysis(x_c)
        if mode == "gradient":
            pi_z = Ax - layer.prox.normal_cone(z_c, Ax, sigma)
        else:
            pi_z = z_c - layer.prox(z_c + Ax, sigma, {})[0]
        layer.fas_dual = mode
        (xn, zn), _ = layer((x_c, z_c), y, E=Identity(), sigma=sigma, pi=(pi_x, pi_z))
        layer.fas_dual = "legacy"
    return rel(xn, x_c), rel(zn, z_c)


def test_sweep_coherence():
    sigma = torch.tensor(0.015)
    for cplx in (False, True):
        torch.manual_seed(1)
        L = make_lpds_layer(1, 6, P=5, stride=2, is_complex=cplx, lam0=0.05)
        dt = torch.complex64 if cplx else torch.float32
        x_c = torch.randn(1, 1, 16, 16, dtype=dt)
        y = torch.randn(1, 1, 16, 16, dtype=dt)
        feas = _fixed_dual(L, 0.08 * torch.randn(1, 6, 8, 8, dtype=dt), sigma)
        t = L.prox.prox.threshold(feas, sigma)
        on = float(((feas.abs() >= 0.999 * t)).float().mean())
        past = feas * 1.5 / (feas.abs() / t).clamp_min(1e-6).clamp_max(1.0)   # |z| = 1.5 t
        tag = "complex" if cplx else "real"
        for mode, z_c, want, label in (
                ("fixed_point", feas, True, "feasible z_c"),
                ("fixed_point", past, True, "z_c past the boundary"),
                ("gradient", feas, True, "feasible z_c"),
                ("gradient", past, False, "z_c past the boundary"),
                ("legacy", feas, False, "feasible z_c")):
            dx, dz = _sweep_change(L, mode, x_c, z_c, y, sigma)
            stays = dx < 1e-5 and dz < 1e-5
            check(f"{tag}: {mode:11s} {label:22s} -> coarse sweep "
                  f"{'stays put' if want else 'MOVES'}", stays == want,
                  f"dx={dx:.1e} dz={dz:.1e}" + (f" ({on:.0%} of z_c on the boundary)"
                                                 if z_c is feas and mode == "gradient" else ""))


def test_coherence_through_dF():
    """From a FINE fixed point (z = t sgn(Ax), so z = clip(z + Ax); y = x + Dz),
    the coarse sweep fed by the real dF must stay put. The box transfer keeps
    |Rz| <= t, so option B is inside its coherence region too."""
    sigma = torch.tensor(0.015)
    for mode, filt, want in (("fixed_point", "sinc", True), ("fixed_point", "sinc_vertex", True),
                             ("gradient", "box", True), ("legacy", "box", False)):
        torch.manual_seed(2)
        V = PDVCycle([2, 2], C=1, M=4, P=5, stride=2, is_complex=False, lam0=0.05,
                     transfer_filter=filt, fas_dual=mode)
        fine = V.lpdsA.first
        with torch.no_grad():
            x = torch.randn(1, 1, 16, 16)
            Ax = fine.analysis(x)
            z = fine.prox.prox.threshold(Ax, sigma) * torch.sign(Ax)
            y = x + fine.synthesis(z)
            (xf, zf), _ = fine((x, z), y, E=Identity(), sigma=sigma)
            fine_fixed = rel(xf, x) < 1e-5 and rel(zf, z) < 1e-5
            (x_c, z_c, pi_x, pi_z, y_c, E_c, sigma_c, _) = V.dF(x, z, y, Identity(), sigma, {}, {})
            coarse = first_pd_layer(V.mglayer)
            (xn, zn), _ = coarse((x_c, z_c), y_c, E=E_c, sigma=sigma_c, pi=(pi_x, pi_z))
            dx, dz = rel(xn, x_c), rel(zn, z_c)
        check(f"constructed fine fixed point is one ({mode}, {filt})", fine_fixed)
        check(f"dF -> coarse sweep, {mode:11s} {filt:11s}: "
              f"{'stays put' if want else 'MOVES'}",
              (dx < 1e-5 and dz < 1e-5) == want, f"dx={dx:.1e} dz={dz:.1e}")


def test_dF_formula():
    sigma = torch.tensor(0.015)
    for mode in ("legacy", "fixed_point", "gradient"):
        torch.manual_seed(3)
        V = PDVCycle([2, 2], C=1, M=4, P=5, stride=2, is_complex=False, fas_dual=mode)
        dF = V.dF
        with torch.no_grad():
            x, z, y = torch.randn(1, 1, 16, 16), 0.05 * torch.randn(1, 4, 8, 8), torch.randn(1, 1, 16, 16)
            x_c, z_c, _, pi_z, _, _, sigma_c, _ = dF(x, z, y, Identity(), sigma, {}, {})
            Af, Ac = dF.analysis_fine(x), dF.analysis_coarse(x_c)
            if mode == "gradient":
                want = (Ac - dF.prox_coarse.normal_cone(z_c, Ac, sigma_c)) \
                    - restrict(Af - dF.prox_fine.normal_cone(z, Af, sigma))
            else:
                want = (z_c - dF.prox_coarse(z_c + Ac, sigma_c, {})[0]) \
                    - restrict(z - dF.prox_fine(z + Af, sigma, {})[0])
        check(f"dF pi_z formula ({mode})", rel(pi_z, want) < 1e-5, f"rel={rel(pi_z, want):.1e}")


def _net(**kw):
    torch.manual_seed(4)
    return MGLPDSNet(K=[2, [2, 2, 2]], M=4, C=1, P=5, s=2, is_complex=False,
                     preproc="identity", lam0=0.05, **kw)


def test_net():
    y = torch.randn(1, 1, 32, 32)
    sig = torch.tensor(0.015)
    with torch.no_grad():
        base = _net()(y, sigma=sig)[0]
        leg = _net(fas_dual="legacy")(y, sigma=sig)[0]
        netA = _net(fas_dual="fixed_point")
        A = netA(y, sigma=sig)[0]
        B = _net(fas_dual="gradient")(y, sigma=sig)[0]
    check("fas_dual='legacy' is bit-identical to the default", torch.equal(base, leg))
    check("fixed_point and gradient change the output",
          rel(A, base) > 1e-6 and rel(B, base) > 1e-6, f"A {rel(A, base):.1e}  B {rel(B, base):.1e}")

    coarse_modes = set()
    for V in (m for m in netA.modules() if isinstance(m, PDVCycle)):
        coarse_modes |= {l.fas_dual for l in V.mglayer.modules() if isinstance(l, LPDSLayer)}
        coarse_modes.add(V.dF.fas_dual)
    check("every coarse layer and dF carries the mode", coarse_modes == {"fixed_point"},
          str(coarse_modes))

    net = _net()
    with torch.no_grad():
        prev = set_fas_dual(net, "fixed_point")
        A2 = net(y, sigma=sig)[0]
        set_fas_dual(net, prev)
        back = net(y, sigma=sig)[0]
    check("set_fas_dual on a built net == built with the mode; restores",
          prev == "legacy" and torch.equal(A2, A) and torch.equal(back, base))

    try:
        MGLPDSNet(K=[1, [2, 2]], M=4, C=1, P=5, s=2, is_complex=False, preproc="identity",
                  window=3, Mh=4, attn_backend="gather", fas_dual="gradient")
        check("gradient form refused for the group prox", False)
    except ValueError:
        check("gradient form refused for the group prox", True)
    try:
        _net(fas_dual="outside")
        check("unknown fas_dual refused", False)
    except ValueError:
        check("unknown fas_dual refused", True)


def test_vertex_net_end_to_end():
    """A complex, k-space, rediscretized net on the vertex lattice: runs, is
    finite, and back-propagates (coarse coil maps restricted with the odd kernel)."""
    N = 32
    g = torch.Generator().manual_seed(0)
    smaps = torch.randn(1, 4, N, N, dtype=torch.complex64, generator=g) + 1.0
    smaps = smaps / smaps.abs().pow(2).sum(1, keepdim=True).sqrt()
    mask = torch.zeros(1, 1, N, N)
    mask[..., ::2] = 1.0
    mask[..., N // 2 - 4:N // 2 + 4] = 1.0
    E = Mask(mask) @ FFT2D() @ Sense(smaps)
    img = torch.randn(1, 1, N, N, dtype=torch.complex64, generator=g)
    for mode in ("fixed_point", "gradient"):
        torch.manual_seed(5)
        net = MGLPDSNet(K=[1, [2, 2, 2]], M=4, C=1, P=5, s=2, is_complex=True,
                        preproc="kspace", coarse_op="rediscretize",
                        transfer_filter="sinc_vertex", fas_dual=mode)
        x_hat, _ = net(E(img), E=E, sigma=torch.full((1, 1, 1, 1), 0.015))
        loss = (x_hat - img).abs().pow(2).mean()
        loss.backward()
        grads = [p.grad for p in net.parameters() if p.grad is not None]
        check(f"vertex lattice + rediscretize + {mode}: forward/backward",
              x_hat.shape == img.shape and bool(torch.isfinite(loss))
              and len(grads) > 0 and all(bool(torch.isfinite(g_).all()) for g_ in grads))


_COPIES = ("analysis", "synthesis", "prox")


def _drift_(layer, scale=0.3, seed=0):
    """Move a layer's convs the way training moves them after construction."""
    from models.base import set_weight
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for conv in (layer.analysis, layer.synthesis):
            w = conv.weight
            set_weight(conv, w + scale * w.abs().mean() * torch.randn(w.shape, generator=g).to(w.dtype))


def test_tied_copies():
    """`fas_copies="tied"`: the downsampler uses F_1's / C_1's own modules, so
    the FAS correction stays coherent however far training moves them."""
    from models.mg_lpds import PDObjectiveDownsample
    tied, untied = _net(fas_copies="tied"), _net(fas_copies="untied")
    same = []
    for V in (m for m in tied.modules() if isinstance(m, PDVCycle)):
        F1, C1 = V.lpdsA.first, first_pd_layer(V.mglayer)
        same += [getattr(V.dF, f"{n}_fine") is getattr(F1, n) for n in _COPIES]
        same += [getattr(V.dF, f"{n}_coarse") is getattr(C1, n) for n in _COPIES]
    check("tied: every downsampler holds F_1's / C_1's own modules, in every outer V-cycle",
          len(same) == 2 * 3 * 2 * 2 and all(same), f"{sum(same)}/{len(same)}")
    copies = sum(p.numel() for d in untied.modules() if isinstance(d, PDObjectiveDownsample)
                 for n in _COPIES for s in ("fine", "coarse")
                 for p in getattr(d, f"{n}_{s}").parameters())
    n_t, n_u = (sum(p.numel() for p in m.parameters()) for m in (tied, untied))
    check("tied drops exactly the copies' parameters", n_u - n_t == copies,
          f"{n_u:,} -> {n_t:,} ({copies:,} in copies)")
    y = torch.randn(1, 1, 32, 32)
    with torch.no_grad():
        check("tied and untied nets agree before any training (copies == layers)",
              torch.equal(tied(y, sigma=torch.tensor(0.015))[0],
                          untied(y, sigma=torch.tensor(0.015))[0]))

    # after "training": F_1 and C_1 move, an untied dF's copies do not
    sigma = torch.tensor(0.015)
    for copies_mode, mode, filt, want in (("untied", "fixed_point", "sinc_vertex", False),
                                          ("tied", "fixed_point", "sinc_vertex", True),
                                          ("tied", "gradient", "box", True)):
        torch.manual_seed(2)
        V = PDVCycle([2, 2], C=1, M=4, P=5, stride=2, is_complex=False, lam0=0.05,
                     transfer_filter=filt, fas_dual=mode, fas_copies=copies_mode)
        F1, C1 = V.lpdsA.first, first_pd_layer(V.mglayer)
        _drift_(F1, seed=1)
        _drift_(C1, seed=2)
        with torch.no_grad():
            x = torch.randn(1, 1, 16, 16)
            Ax = F1.analysis(x)
            z = F1.prox.prox.threshold(Ax, sigma) * torch.sign(Ax)     # z = clip(z + Ax)
            y = x + F1.synthesis(z)                                     # fine residual 0
            (x_c, z_c, pi_x, pi_z, y_c, E_c, sigma_c, _) = V.dF(x, z, y, Identity(), sigma, {}, {})
            (xn, zn), _ = C1((x_c, z_c), y_c, E=E_c, sigma=sigma_c, pi=(pi_x, pi_z))
            dx, dz = rel(xn, x_c), rel(zn, z_c)
        check(f"after drift, {copies_mode:6s} copies + {mode:11s}: coarse sweep "
              f"{'stays put' if want else 'MOVES'}", (dx < 1e-5 and dz < 1e-5) == want,
              f"dx={dx:.1e} dz={dz:.1e}")

    # checkpoints: untied (drifted) -> tied refused; tied -> tied and tied -> untied load
    drifted = _net(fas_copies="untied")
    for V in (m for m in drifted.modules() if isinstance(m, PDVCycle)):
        _drift_(V.lpdsA.first, seed=3)
    try:
        _net(fas_copies="tied").load_state_dict(drifted.state_dict())
        check("an untied checkpoint with drifted copies is refused by a tied net", False)
    except ValueError:
        check("an untied checkpoint with drifted copies is refused by a tied net", True)
    a = _net(fas_copies="tied")
    for V in (m for m in a.modules() if isinstance(m, PDVCycle)):
        _drift_(V.lpdsA.first, seed=4)
    b, c = _net(fas_copies="tied"), _net(fas_copies="untied")
    b.load_state_dict(a.state_dict())
    c.load_state_dict(a.state_dict())
    with torch.no_grad():
        oa, ob, oc = (m(y, sigma=sigma)[0] for m in (a, b, c))
    check("tied checkpoint round-trips (tied -> tied bit-identical, -> untied equal)",
          torch.equal(oa, ob) and rel(oc, oa) < 1e-6, f"untied rel {rel(oc, oa):.1e}")

    # training mechanics on the tied net
    t = _net(fas_copies="tied")
    out = t(y, sigma=sigma)[0]
    out.pow(2).mean().backward()
    C1 = first_pd_layer(next(m for m in t.modules() if isinstance(m, PDVCycle)).mglayer)
    g = C1.synthesis.conv_real.weight.grad
    t.project()
    check("tied net: gradients reach the shared modules; project() runs",
          g is not None and bool(torch.isfinite(g).all()) and float(g.abs().sum()) > 0)
    try:
        _net(fas_copies="shared")
        check("unknown fas_copies refused", False)
    except ValueError:
        check("unknown fas_copies refused", True)


def main():
    for t in (test_vertex_pair, test_vertex_response, test_lattice_alignment,
              test_vertex_length_and_learned, test_sweep_coherence,
              test_coherence_through_dF, test_dF_formula, test_net,
              test_vertex_net_end_to_end, test_tied_copies):
        print(f"\n== {t.__name__}")
        t()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED:", *FAIL, sep="\n  ")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
