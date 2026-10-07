"""
How well does a REDISCRETIZED coarse SENSE operator match the Galerkin one?

Run with `python -m tests.test_coarse_operator`.

Two coarse operators for one V-cycle level (see operators/coarse.py):

    Galerkin        G_gal = R E^H E P                 (what the models use)
    rediscretized   G_c   = S_c^H F_c^H M_c F_c S_c   (every FFT on the coarse grid)

CORRECTNESS is pinned where the two SHOULD agree -- full sampling, where both
reduce to (nearly) the identity on content the coarse grid can represent:
scaling, k-space centring, the crop, the map transfer and the adjoint. Each of
those checks has a NEGATIVE CONTROL (a deliberately wrong variant that must
fail), so a pass means the check can see the error it is looking for.

AGREEMENT under undersampling is REPORTED, not asserted: the rediscretized
operator drops every sampled line outside the central band, so the two coarse
problems legitimately differ, and by how much is the question this answers.
Four numbers per acceleration:

    Gram (smooth / phantom / noise)   ||G_c x - G_gal x|| / ||G_gal x|| for a
                                      smooth image, a restricted phantom, and
                                      white noise (which stresses the band edge)
    spectral                          ||G_c - G_gal||_2 / ||G_gal||_2 by power
                                      iteration -- the worst case over inputs
    rhs                               ||E_c^H y_c - R E^H y|| / ||R E^H y||, the
                                      data term a coarse level would start from

CPU only, no LAPACK, synthetic data (320 x 320, 8 complex coils): the Julia
mask at 320 PE lines is the grid's (13 ACS lines at center_frac 0.04).
"""

import math

import torch

from operators import FFT2D, Mask, Sense
from operators.coarse import (coarse_data, coarse_sense, coarse_truncate, crop_center,
                              rediscretize)
from operators.resample import galerkin, restrict
from operators.truncate import Truncate, embed_operator
from physics.mask import effective_accel, make_acc_mask

FAIL = []
N, C = 320, 8
ACCELS = (4, 8, 12, 16)
CENTER_FRAC = 0.04


def check(name, ok, detail=""):
    print(f"[{'ok ' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def rel(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def inner(a, b):
    return torch.sum(a.conj() * b)


# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------
def fixtures(seed=0):
    g = torch.Generator().manual_seed(seed)
    yy = (torch.arange(N)[:, None] - N / 2).double()
    xx = (torch.arange(N)[None, :] - N / 2).double()

    # brain-like: an ellipse with internal structure and a smooth phase
    head = (((yy / 0.85) ** 2 + (xx / 0.7) ** 2) < (0.42 * N) ** 2).double()
    inner_ = (((yy + 20) / 0.6) ** 2 + ((xx - 15) / 0.5) ** 2 < (0.15 * N) ** 2).double()
    ring = ((yy ** 2 + xx ** 2) < (0.3 * N) ** 2).double() - ((yy ** 2 + xx ** 2) < (0.27 * N) ** 2).double()
    x_fine = (head * (1 + 0.3 * torch.cos(xx / 9) * torch.sin(yy / 13))
              + 0.5 * inner_ - 0.3 * ring)
    x_fine = x_fine * torch.exp(1j * (0.004 * xx + 0.003 * yy))

    # complex coil maps: localised magnitudes, random offsets AND phase ramps
    sm = []
    for c in range(C):
        a = 2 * math.pi * c / C
        cy, cx = 0.55 * N * math.sin(a), 0.55 * N * math.cos(a)
        mag = torch.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (0.35 * N) ** 2))
        ph = (2 * math.pi * torch.rand(1, generator=g, dtype=torch.float64)
              + 0.01 * (yy * math.cos(a) - xx * math.sin(a)))
        sm.append(mag * torch.exp(1j * ph))
    sm = torch.stack(sm)
    sm = sm / sm.abs().pow(2).sum(0, keepdim=True).sqrt()

    cdt = torch.complex128
    x_fine = x_fine.to(cdt)[None, None]
    sm = sm.to(cdt)[None]

    # coarse test inputs
    nc = N // 2
    yc = (torch.arange(nc)[:, None] - nc / 2).double()
    xc = (torch.arange(nc)[None, :] - nc / 2).double()
    smooth = torch.exp(-(yc ** 2 + xc ** 2) / (2 * (0.2 * nc) ** 2)) * torch.exp(1j * 0.02 * xc)
    noise = torch.complex(torch.randn(nc, nc, generator=g, dtype=torch.float64),
                          torch.randn(nc, nc, generator=g, dtype=torch.float64))
    inputs = {
        "smooth": smooth.to(cdt)[None, None],
        "phantom": restrict(x_fine),
        "noise": noise.to(cdt)[None, None],
    }
    return x_fine, sm, inputs


def operators_for(mask, sm):
    E = Mask(mask) @ FFT2D() @ Sense(sm)
    Ec, mask_c, sm_c = coarse_sense(mask, sm)
    return E, galerkin(E), Ec, mask_c, sm_c


def spectral_rel(G_a, G_b, shape, dtype, iters=30, seed=1):
    """||G_a - G_b||_2 / ||G_b||_2, both by power iteration (both Hermitian PSD-ish)."""
    g = torch.Generator().manual_seed(seed)

    def power(fn):
        v = torch.complex(torch.randn(shape, generator=g, dtype=torch.float64),
                          torch.randn(shape, generator=g, dtype=torch.float64)).to(dtype)
        v = v / v.norm()
        lam = 0.0
        for _ in range(iters):
            w = fn(v)
            lam = float(w.norm())
            v = w / max(lam, 1e-30)
        return lam

    return power(lambda v: G_a(v) - G_b(v)) / power(G_b)


# ---------------------------------------------------------------------------
#  Correctness, where the two must agree: full sampling
# ---------------------------------------------------------------------------
def test_full_sampling():
    x_fine, sm, inputs = fixtures()
    full = torch.ones(1, 1, N, N, dtype=torch.float64)
    E, Egal, Ec, mask_c, sm_c = operators_for(full, sm)

    # 1. the Grams agree on content the coarse grid represents
    for name in ("smooth", "phantom"):
        x = inputs[name]
        err = rel(Ec.gram(x), Egal.gram(x))
        check(f"full sampling: coarse Gram matches Galerkin on {name} input",
              err < 0.05, f"rel err {err:.2e}")

    # 2. the coarse k-space IS the (scaled) centre of the fine k-space --
    #    scaling and centring. y_c = crop(y)/2 against E_c applied to R x.
    y = E.forward(x_fine)
    err = rel(Ec.forward(restrict(x_fine)), coarse_data(y))
    check("full sampling: E_c R x matches crop(E x)/2 (scale + centring)",
          err < 0.1, f"rel err {err:.2e}")
    # negative controls: without the /2, and a crop off by one column
    err_scale = rel(Ec.forward(restrict(x_fine)), crop_center(y))
    check("  control: omitting the /2 is caught", err_scale > 0.4,
          f"rel err {err_scale:.2f}")
    shifted = torch.roll(y, 1, dims=-1)
    err_shift = rel(Ec.forward(restrict(x_fine)), coarse_data(shifted))
    check("  control: a one-sample k-space shift is caught", err_shift > 0.2,
          f"rel err {err_shift:.2f}")

    # 3. the coarse data term matches the restricted fine one
    err = rel(Ec.adjoint(coarse_data(y)), restrict(E.adjoint(y)))
    check("full sampling: E_c^H y_c matches R E^H y (the coarse data term)",
          err < 0.1, f"rel err {err:.2e}")

    # 4. both Grams are Hermitian, and E_c's adjoint is its adjoint
    g = torch.Generator().manual_seed(3)
    a = torch.complex(torch.randn(1, 1, N // 2, N // 2, generator=g, dtype=torch.float64),
                      torch.randn(1, 1, N // 2, N // 2, generator=g, dtype=torch.float64))
    b = torch.complex(torch.randn(1, 1, N // 2, N // 2, generator=g, dtype=torch.float64),
                      torch.randn(1, 1, N // 2, N // 2, generator=g, dtype=torch.float64))
    for name, op in (("rediscretized", Ec), ("Galerkin", Egal)):
        h = abs(complex(inner(op.gram(a), b) - inner(a, op.gram(b)))) / abs(complex(inner(a, op.gram(b))))
        check(f"{name} coarse Gram is Hermitian", h < 1e-10, f"{h:.1e}")
    kb = torch.complex(torch.randn(1, C, N // 2, N // 2, generator=g, dtype=torch.float64),
                       torch.randn(1, C, N // 2, N // 2, generator=g, dtype=torch.float64))
    adj = abs(complex(inner(Ec.forward(a), kb) - inner(a, Ec.adjoint(kb)))) / abs(complex(inner(Ec.forward(a), kb)))
    check("E_c forward/adjoint are a true adjoint pair", adj < 1e-10, f"{adj:.1e}")

    # 5. conjugation control. Checked on the FORWARD operator: a fully sampled
    #    Gram is sum_c |s_c|^2 x, identical for conj(s), so it cannot see this.
    Ec_bad = Mask(mask_c) @ FFT2D() @ Sense(sm_c.conj())
    err_conj = rel(Ec_bad.forward(restrict(x_fine)), coarse_data(y))
    err_ok = rel(Ec.forward(restrict(x_fine)), coarse_data(y))
    check("  control: conjugated coarse maps are caught (forward vs data)",
          err_conj > 5 * err_ok, f"{err_conj:.2e} vs {err_ok:.2e}")

    # 6. the maps: restricted maps stay unit RSS after renorm
    rss = sm_c.abs().pow(2).sum(1).sqrt()
    check("coarse maps are unit RSS", float((rss - 1).abs().max()) < 1e-6,
          f"max |rss-1| {float((rss - 1).abs().max()):.1e}")

    # band edge, reported: white noise has content the Kaiser-sinc transfer
    # attenuates near the coarse Nyquist, which the rediscretized FFT keeps
    err = rel(Ec.gram(inputs["noise"]), Egal.gram(inputs["noise"]))
    print(f"       (full sampling, white-noise input: rel err {err:.2e} -- "
          f"the transfer filter's roll-off near coarse Nyquist; informational)")


# ---------------------------------------------------------------------------
#  Agreement under the grid's undersampling: reported
# ---------------------------------------------------------------------------
def test_undersampled_report():
    x_fine, sm, inputs = fixtures()
    print(f"\n  Julia masks, N={N} PE lines, center_frac={CENTER_FRAC}, {C} coils\n")
    print(f"  {'R':>3} {'eff':>6} {'lines':>7} {'in band':>8} |"
          f" {'Gram smooth':>11} {'phantom':>8} {'noise':>7} | {'spectral':>8} | {'rhs':>6}")
    for R in ACCELS:
        m = make_acc_mask((N, N), accel=R, center_frac=CENTER_FRAC, dim=1,
                          mode="uniform", adjust_accel=True).to(torch.float64)
        E, Egal, Ec, mask_c, _ = operators_for(m, sm)
        lines = int(m[0, 0, 0].sum())
        band = int(mask_c[0, 0, 0].sum())
        errs = [rel(Ec.gram(inputs[k]), Egal.gram(inputs[k]))
                for k in ("smooth", "phantom", "noise")]
        spec = spectral_rel(Ec.gram, Egal.gram, inputs["noise"].shape,
                            inputs["noise"].dtype)
        y = E.forward(x_fine)
        rhs = rel(Ec.adjoint(coarse_data(y)), restrict(E.adjoint(y)))
        print(f"  {R:>3} {float(effective_accel(m)):>6.2f} {lines:>7} {band:>8} |"
              f" {errs[0]:>11.3f} {errs[1]:>8.3f} {errs[2]:>7.3f} | {spec:>8.3f} | {rhs:>6.3f}")
        check(f"R={R}: every number is finite",
              all(math.isfinite(v) for v in errs + [spec, rhs]))

    # cost, for scale: FFT pixels per Gram
    print(f"\n  per Gram: Galerkin runs 2*{C} FFTs of {N}x{N}; rediscretized "
          f"2*{C} of {N // 2}x{N // 2} (1/4 the pixels) plus no resampling")


# ---------------------------------------------------------------------------
#  The image-domain embedding: E @ Truncate
# ---------------------------------------------------------------------------
def rect_fixtures(h, w, ramp=0.01, fill=0.30, seed=0):
    """A phantom well inside an h x w FOV and C unit-RSS complex coil maps."""
    g = torch.Generator().manual_seed(seed)
    yy = (torch.arange(h)[:, None] - h / 2).double()
    xx = (torch.arange(w)[None, :] - w / 2).double()
    head = (((yy / (fill * h)) ** 2 + (xx / (fill * w)) ** 2) < 1).double()
    blob = ((((yy + 0.06 * h) / (0.2 * h)) ** 2 + ((xx - 0.05 * w) / (0.15 * w)) ** 2) < 1).double()
    x = head * (1 + 0.3 * torch.cos(xx / 9) * torch.sin(yy / 13)) + 0.5 * blob
    x = (x * torch.exp(1j * (0.004 * xx + 0.003 * yy))).to(torch.complex128)[None, None]
    sm = []
    for c in range(C):
        a = 2 * math.pi * c / C
        cy, cx = 0.55 * h * math.sin(a), 0.55 * w * math.cos(a)
        mag = torch.exp(-((yy - cy) ** 2 / (2 * (0.35 * h) ** 2)
                          + (xx - cx) ** 2 / (2 * (0.35 * w) ** 2)))
        ph = (2 * math.pi * torch.rand(1, generator=g, dtype=torch.float64)
              + ramp * (yy * math.cos(a) - xx * math.sin(a)))
        sm.append(mag * torch.exp(1j * ph))
    sm = torch.stack(sm)
    sm = (sm / sm.abs().pow(2).sum(0, keepdim=True).sqrt()).to(torch.complex128)[None]
    return x, sm


def bump(shape, width=0.10):
    """A complex Gaussian that is ~0 at the grid's edges (4+ sigma)."""
    hc, wc = shape
    yc = (torch.arange(hc)[:, None] - hc / 2).double()
    xc = (torch.arange(wc)[None, :] - wc / 2).double()
    g = torch.exp(-(yc ** 2 / (2 * (width * hc) ** 2) + xc ** 2 / (2 * (width * wc) ** 2)))
    return (g * torch.exp(1j * 0.02 * xc)).to(torch.complex128)[None, None]


def sampling(h, w, R):
    if R == 1:
        return torch.ones(1, 1, h, w, dtype=torch.float64)
    return make_acc_mask((h, w), accel=R, center_frac=CENTER_FRAC, dim=1, mode="uniform",
                         adjust_accel=True).to(torch.float64)


def test_embedded_bookkeeping():
    """Sizes, offsets and the lattice shift -- exact, no physics."""
    # the knee case: 372 columns embedded to 376, three levels
    T0 = Truncate((640, 376), (640, 372))
    T1, p1 = coarse_truncate(T0)
    T2, p2 = coarse_truncate(T1)
    check("640x372 in 640x376: level 1 is 320x186 in 320x188 at column 1, no shift",
          (T1.big, T1.small, T1.top, T1.left, p1) == ((320, 188), (320, 186), 0, 1, (0, 0)),
          f"{T1!r} phase {p1}")
    check("...level 2 is 160x93 in 160x94 at column 1, maps shifted one column",
          (T2.big, T2.small, T2.top, T2.left, p2) == ((160, 94), (160, 93), 0, 1, (0, 1)),
          f"{T2!r} phase {p2}")
    check("...and an odd measured grid (93) has no further coarse form",
          coarse_truncate(T2) is None)

    # R T^H w == T_c^H R(roll(w, -phase)) EXACTLY, for w that vanishes at the
    # window's edge: the claim the shifted lattice rests on.
    T = Truncate((80, 76), (78, 74), offset=(1, 1))               # odd offsets
    Tc, phase = coarse_truncate(T)
    w = bump(T.small, width=0.06)
    lhs = restrict(T.adjoint(w))
    rhs = Tc.adjoint(restrict(torch.roll(w, shifts=(-phase[0], -phase[1]), dims=(-2, -1))))
    check("odd offset: restrict(T^H w) == T_c^H restrict(shifted w), to roundoff",
          rel(rhs, lhs) < 1e-10, f"phase {phase}, rel err {rel(rhs, lhs):.1e}")
    bad = Tc.adjoint(restrict(w))
    check("  control: without the shift it is off by half a coarse pixel",
          rel(bad, lhs) > 1e-2, f"rel err {rel(bad, lhs):.2e}")
    centred = Truncate(Tc.big, Tc.small)
    check("  control: the centred coarse window is the wrong one here",
          (centred.top, centred.left) != (Tc.top, Tc.left)
          and rel(centred.adjoint(restrict(torch.roll(w, (-phase[0], -phase[1]), (-2, -1)))),
                  lhs) > 1e-2,
          f"centred {(centred.top, centred.left)} vs {(Tc.top, Tc.left)}")


def test_embedded_agreement():
    """E @ Truncate: the rediscretized Gram against Galerkin, two levels down.

    316 x 300 is embedded to 320 x 304 (both axes: multiples of 4, not of 8),
    so level 1 has even offsets and level 2 odd ones. The no-embedding problem
    at 320 x 304 is the reference: the embedding should cost nothing extra on
    content that stays inside the FOV.
    """
    h, w = 316, 300
    x, sm = rect_fixtures(h, w)
    xr, smr = rect_fixtures(320, 304)
    print(f"\n  measured {h}x{w} embedded to 320x304, {C} coils; reference: plain 320x304\n")
    print(f"  {'R':>3} {'level':>6} | {'smooth':>8} {'phantom':>8} | {'ref smooth':>10} "
          f"{'ref phantom':>11} | {'edge':>7}")
    for R in (1, 8):
        E_t, T = embed_operator(Mask(sampling(h, w, R)) @ FFT2D() @ Sense(sm), (h, w), 8)
        E_r = Mask(sampling(320, 304, R)) @ FFT2D() @ Sense(smr)
        x_c, xr_c = restrict(T.adjoint(x)), restrict(xr)
        for level in (1, 2):
            red, gal = rediscretize(E_t), galerkin(E_t)
            red_r, gal_r = rediscretize(E_r), galerkin(E_r)
            Tc = red.ops[3]
            want = ((2, 2), (1, 1)) if level == 1 else ((1, 1), (1, 1))
            check(f"R={R} level {level}: {Tc!r}",
                  (T.top, T.left) == want[0] and (Tc.top, Tc.left) == want[1]
                  and tuple(red.ops[2].smaps.shape[-2:]) == Tc.small)
            grid = Tc.big
            errs = {k: rel(red.gram(v), gal.gram(v))
                    for k, v in (("smooth", bump(grid)), ("phantom", x_c))}
            refs = {k: rel(red_r.gram(v), gal_r.gram(v))
                    for k, v in (("smooth", bump(grid)), ("phantom", xr_c))}
            edge = rel(red.gram(bump(grid, 0.25)), gal.gram(bump(grid, 0.25)))
            print(f"  {R:>3} {level:>6} | {errs['smooth']:>8.4f} {errs['phantom']:>8.4f} | "
                  f"{refs['smooth']:>10.4f} {refs['phantom']:>11.4f} | {edge:>7.4f}")
            for k in ("smooth", "phantom"):
                check(f"R={R} level {level}: embedded Gram matches Galerkin on {k} input "
                      f"as well as the plain operator does",
                      errs[k] < 0.05 and errs[k] < 1.5 * refs[k] + 2e-3,
                      f"{errs[k]:.2e} (plain {refs[k]:.2e})")

            if level == 2:
                g = torch.Generator().manual_seed(7)
                rnd = lambda *s: torch.complex(                    # noqa: E731
                    torch.randn(*s, generator=g, dtype=torch.float64),
                    torch.randn(*s, generator=g, dtype=torch.float64))
                a, b = rnd(1, 1, *grid), rnd(1, 1, *grid)
                hh = (abs(complex(inner(red.gram(a), b) - inner(a, red.gram(b))))
                      / abs(complex(inner(a, red.gram(b)))))
                check(f"R={R}: the embedded coarse Gram is Hermitian on an odd "
                      f"{Tc.small[0]}x{Tc.small[1]} measured grid", hh < 1e-10, f"{hh:.1e}")
                kb = rnd(1, C, *Tc.small)
                adj = (abs(complex(inner(red.forward(a), kb) - inner(a, red.adjoint(kb))))
                       / abs(complex(inner(red.forward(a), kb))))
                check(f"R={R}: ...and its forward/adjoint are a true adjoint pair",
                      adj < 1e-10, f"{adj:.1e}")

            # one level down, each side from ITS OWN level-1 rediscretized operator
            E_t, T, E_r = red, Tc, red_r
            x_c, xr_c = restrict(x_c), restrict(xr_c)
    print("\n  `edge`: a wide input that is NOT ~0 at the measured window's border. There\n"
          "  the two coarse problems differ (Galerkin sees a zero-padded edge, the\n"
          "  rediscretized FOV is periodic); informational.")


def main():
    for fn in (test_full_sampling, test_undersampled_report, test_embedded_bookkeeping,
               test_embedded_agreement):
        print(f"\n--- {fn.__name__}")
        fn()
    print(f"\n{'FAILED: ' + ', '.join(FAIL) if FAIL else 'all checks passed'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
