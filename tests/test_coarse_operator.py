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
from operators.coarse import coarse_data, coarse_sense, crop_center
from operators.resample import galerkin, restrict
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


def main():
    for fn in (test_full_sampling, test_undersampled_report):
        print(f"\n--- {fn.__name__}")
        fn()
    print(f"\n{'FAILED: ' + ', '.join(FAIL) if FAIL else 'all checks passed'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
