"""
Maps estimated from the measurement (`physics/online_smaps.py`).

Run with `python -m tests.test_online_smaps`.

The properties that matter, in the order they bite:

1. THE ACS BLOCK. `Sljiva/src/closures/mrireco.jl:244` takes the full readout
   extent and the ACS line count, each clamped to 32 -- NOT a square. A square
   block starves the calibration for no reason, because the ACS lines are fully
   sampled along readout.
2. NOTHING OUTSIDE THE ACS REACHES THE MAPS. This is the whole point of
   estimating online: a map that saw unmeasured k-space is a map the scanner
   could not produce at inference.
3. The estimators run and produce unit-RSS maps.

ESPIRiT needs LAPACK, which the local anaconda torch lacks, so those cases skip
here and run on the cluster. The ACS-geometry cases need no linear algebra and
run everywhere -- and they are the ones that encode the bug this fixes.
"""

import torch

from operators.fourier import fftc, ifftc
from physics.mask import make_acc_mask
from physics.online_smaps import (
    acs_block, acs_line_count, acs_taper, center_mask, estimate_cost_gb,
    hamming_window, online_smaps, resolve_lines,
)

FAIL = []


def check(name, ok, detail=""):
    print(f"[{'ok ' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def have_lapack():
    try:
        torch.linalg.svd(torch.randn(4, 4, dtype=torch.complex64))
        return True
    except Exception:
        return False


def phantom(C=6, H=64, W=48, seed=0):
    g = torch.Generator().manual_seed(seed)
    y = torch.arange(H)[:, None] - H / 2
    x = torch.arange(W)[None, :] - W / 2
    obj = ((y ** 2 + x ** 2) < (0.3 * H) ** 2).to(torch.complex64)
    sm = torch.stack([torch.exp(-((y - 7 * c) ** 2 + (x - 5 * c) ** 2) / (2 * 40.0 ** 2))
                      for c in range(C)]).to(torch.complex64)
    sm = sm / sm.abs().pow(2).sum(0, keepdim=True).sqrt()
    k = fftc((sm * obj)[None])
    return k, sm[None], obj


# ---------------------------------------------------------------------------
def test_acs_line_count():
    # The counted run can EXCEED the requested block when an accelerated line
    # abuts it -- that is why `resolve_lines` prefers the config's number.
    for lines in (8, 20, 24):
        m = make_acc_mask((64, 48), accel=4, acs_lines=lines, dim=1, mode="uniform")
        got, res = acs_line_count(m), resolve_lines(m, lines)
        check(f"counts at least the {lines} requested ACS lines", got >= lines,
              f"counted {got}")
        check(f"resolve_lines({lines}) returns the config's number", res == lines,
              f"got {res}")
    m = make_acc_mask((64, 48), accel=4, acs_lines=8, dim=1, mode="uniform")
    check("resolve_lines clamps a request larger than the sampled run",
          resolve_lines(m, 999) == acs_line_count(m),
          f"{resolve_lines(m, 999)} vs counted {acs_line_count(m)}")

    # outer sampled lines must not be counted: the calibration region is the
    # CONTIGUOUS centre, and a gapped one is not a calibration region at all
    m = make_acc_mask((64, 48), accel=2, acs_lines=6, dim=1, mode="uniform")
    got = acs_line_count(m)
    check("R=2 (dense outer sampling) still counts only the contiguous centre",
          got >= 6, f"got {got} for 6 ACS lines at R=2")


def test_acs_block_is_asymmetric():
    """The bug this module exists for: 20x20 square vs full readout x lines."""
    k, _, _ = phantom(H=640, W=320)
    m = make_acc_mask((640, 320), accel=16, acs_lines=20, dim=1, mode="uniform")
    ax, ay = acs_block(m, k.shape)
    check("readout extent is clamped to 32, not the line count", ax == 32, f"ax={ax}")
    check("phase-encode extent is the ACS line count", ay == 20, f"ay={ay}")

    # what that buys: rows of the calibration matrix, which must exceed
    # ks^2 * C for a null space to exist at all
    for ks, C in ((6, 20), (8, 20)):
        rows_sq = max(20 - ks + 1, 0) ** 2
        rows_rect = max(ax - ks + 1, 0) * max(ay - ks + 1, 0)
        cols = ks * ks * C
        print(f"       ks={ks} C={C}: 20x20 -> {rows_sq} rows, "
              f"{ax}x{ay} -> {rows_rect} rows, cols={cols}")
        check(f"the rectangular block gives more rows (ks={ks})",
              rows_rect > rows_sq, f"{rows_rect} vs {rows_sq}")


def test_only_the_acs_reaches_the_estimator():
    k, _, _ = phantom()
    m = make_acc_mask((64, 48), accel=4, acs_lines=8, dim=1, mode="uniform")
    lines = acs_line_count(m)
    cm = center_mask(m, lines)
    kc = k * cm.to(k.dtype)

    outside = kc[..., cm == 0]
    check("every k-space column outside the ACS is zeroed",
          float(outside.abs().max()) == 0.0,
          f"max |k| outside = {float(outside.abs().max()):.2e}")
    check("the ACS columns are untouched",
          torch.equal(kc[..., cm > 0], k[..., cm > 0]))

    w = hamming_window(kc)
    check("the window keeps the centre and tapers the edge",
          float(w[0, 0, 32, 24].abs()) > float(w[0, 0, 0, 0].abs()),
          "centre is not the brightest sample after windowing")

    # the ACS-width taper: same support as the centre mask, tapered ACROSS it
    t = acs_taper(m, lines)
    check("the ACS taper has exactly the centre mask's support",
          torch.equal(t > 0, cm > 0))
    tl = t[cm > 0]
    check("the ACS taper actually tapers the ACS (edge << centre)",
          float(tl[0]) < 0.1 and float(tl.max()) > 0.99,
          f"edge {float(tl[0]):.3f}, peak {float(tl.max()):.3f}")


def test_walsh_window_default():
    """Walsh defaults to the ACS taper, ESPIRiT to the box.

    Checked through what reaches the estimator, by stubbing it out.
    """
    import physics.online_smaps as osm
    k, _, _ = phantom()
    m = make_acc_mask((64, 48), accel=4, acs_lines=8, dim=1, mode="uniform")
    lines = acs_line_count(m)
    seen = {}
    real_walsh, real_espirit, real_ifftc = osm.walsh, osm.espirit, osm.ifftc
    try:
        osm.ifftc = lambda x: x                   # hand walsh k-space as-is
        osm.walsh = lambda x, **kw: seen.__setitem__("walsh", x) or x
        osm.espirit = lambda x, **kw: seen.__setitem__("espirit", x) or x
        osm.online_smaps(k, m, method="walsh", acs_lines=lines)
        osm.online_smaps(k, m, method="espirit", acs_lines=lines)
        osm.online_smaps(k, m, method="walsh", acs_lines=lines, window="box")
        box_walsh = seen["walsh"]
        osm.online_smaps(k, m, method="walsh", acs_lines=lines)
    finally:
        osm.walsh, osm.espirit, osm.ifftc = real_walsh, real_espirit, real_ifftc
    cm = center_mask(m, lines)
    kc = k * cm.to(k.dtype)
    t = acs_taper(m, lines)
    check("walsh gets the ACS-tapered calibration data by default",
          torch.allclose(seen["walsh"], k * t.to(k.dtype)))
    check("espirit gets the untapered ACS by default",
          torch.equal(seen["espirit"], kc))
    check("window='box' still reaches walsh untapered",
          torch.equal(box_walsh, kc))
    try:
        online_smaps(k, m, window="nope")
        check("an unknown window is rejected", False, "no error")
    except ValueError:
        check("an unknown window is rejected", True)


def test_cost_bound():
    gb = estimate_cost_gb((1, 20, 640, 320), n_kernels=325)
    check("the cost bound is reported in GB", 9.0 < gb < 11.0, f"{gb:.1f} GB")
    print(f"       (1, 20, 640, 320) with 325 kernels -> {gb:.1f} GB peak")


def test_estimators_run():
    if not have_lapack():
        print("[skip] estimator cases need LAPACK (svd); ACS geometry above is "
              "what runs locally")
        return
    k, sm_true, obj = phantom()
    m = make_acc_mask((64, 48), accel=4, acs_lines=16, dim=1, mode="uniform")

    for method in ("espirit", "walsh"):
        sm = online_smaps(k, m, method=method, kernel_size=4)
        check(f"{method}: shape matches the k-space", tuple(sm.shape) == tuple(k.shape),
              f"{tuple(sm.shape)}")
        nrm = sm.abs().pow(2).sum(1).sqrt()
        lit = nrm > 1e-6
        err = float((nrm[lit] - 1).abs().max()) if bool(lit.any()) else float("nan")
        check(f"{method}: unit-RSS where the maps are nonzero", err < 1e-3,
              f"max |‖s‖-1| = {err:.2e}")

        # the maps must explain the coil data inside the object
        coil = ifftc(k)
        x = (sm.conj() * coil).sum(1, keepdim=True)
        o = obj.abs() > 0
        res = float(((coil - sm * x)[..., o]).norm() / (coil[..., o]).norm())
        check(f"{method}: explains the coil data over the object", res < 0.25,
              f"residual {res:.3f}")

    sm = online_smaps(k, m, method="espirit", kernel_size=4)
    check("espirit online has NO hard support (thresh_eig=0, as in Julia)",
          float((sm.abs().sum(1) > 0).float().mean()) > 0.99,
          f"{float((sm.abs().sum(1) > 0).float().mean()):.1%} of the FOV is nonzero")


def test_walsh_phase_reference():
    """Localised coils: the strongest coil is dark over part of the object.

    Referencing each patch's eigenvector to that coil (Sljiva's `walsh_smaps`,
    phase_ref="strongest") takes the phase of noise there, and neighbouring
    patches blended by the upsampling come out with unrelated phases -- jumps
    of radians in the combined image. The virtual-coil reference has signal
    everywhere the object does, so the phase stays smooth.
    """
    if not have_lapack():
        print("[skip] walsh needs LAPACK (svd)")
        return
    import math
    from physics.smaps import walsh
    N, C = 96, 8
    g = torch.Generator().manual_seed(0)
    yy = (torch.arange(N)[:, None] - N / 2).double()
    xx = (torch.arange(N)[None, :] - N / 2).double()
    obj = (((yy / 0.42) ** 2 + (xx / 0.35) ** 2) < N ** 2 / 4).double()
    obj = obj * (1 + 0.3 * torch.cos(xx / 7.0))
    sm = []
    for c in range(C):
        a = 2 * math.pi * c / C
        cy, cx = 0.45 * N * math.sin(a), 0.45 * N * math.cos(a)
        mag = torch.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 8.0 ** 2))
        ph = (2 * math.pi * torch.rand(1, generator=g, dtype=torch.float64)
              + 0.04 * (yy * math.sin(a) + xx * math.cos(a)))
        sm.append(mag * torch.exp(1j * ph))
    sm = torch.stack(sm)
    sm = sm / sm.abs().pow(2).sum(0, keepdim=True).sqrt()
    y = sm * obj + 1e-3 * (torch.randn((C, N, N), generator=g, dtype=torch.float64)
                           + 1j * torch.randn((C, N, N), generator=g,
                                              dtype=torch.float64))
    y = y[None].to(torch.complex64)
    inner = obj > 0                      # away from the edge, where bilinear
    for _ in range(4):                   # upsampling smears across it
        inner = (inner & torch.roll(inner, 1, 0) & torch.roll(inner, -1, 0)
                 & torch.roll(inner, 1, 1) & torch.roll(inner, -1, 1))
    pair = inner[1:, :] & inner[:-1, :]

    jump = {}
    for ref in ("strongest", "virtual"):
        s = walsh(y, phase_ref=ref)[0]
        x = (s.conj() * y[0]).sum(0)
        jump[ref] = float(torch.angle(x[1:, :] * x[:-1, :].conj())[pair].abs().max())
    check("walsh strongest-coil reference breaks on localised coils "
          "(the fixture is not vacuous)", jump["strongest"] > 1.0,
          f"max phase jump {jump['strongest']:.2f} rad")
    check("walsh virtual-coil reference keeps the phase smooth",
          jump["virtual"] < 0.2, f"max phase jump {jump['virtual']:.2f} rad")
    try:
        walsh(y, phase_ref="nope")
        check("an unknown walsh phase_ref is rejected", False, "no error")
    except ValueError:
        check("an unknown walsh phase_ref is rejected", True)


def test_bad_inputs():
    k, _, _ = phantom()
    m = make_acc_mask((64, 48), accel=4, acs_lines=8, dim=1, mode="uniform")
    try:
        online_smaps(k, m, method="nope")
        check("an unknown method is rejected", False, "no error")
    except ValueError:
        check("an unknown method is rejected", True)
    try:
        online_smaps(k.real, m)
        check("real k-space is rejected", False, "no error")
    except ValueError:
        check("real k-space is rejected", True)
    try:
        acs_block(torch.zeros(48), k.shape)
        check("a mask with no sampled centre is rejected", False, "no error")
    except ValueError:
        check("a mask with no sampled centre is rejected", True)


# ---------------------------------------------------------------------------
def test_phase_correct():
    """`phase_correct`: the maps' per-pixel phase convention cancels EXACTLY.

    Whatever phase e^{i phi(r)} an estimator leaves in the maps, the phase-
    corrected maps must come out identical -- that is what makes the choice of
    reference (strongest coil, virtual coil, noise) irrelevant to the image the
    net reconstructs.  Magnitude, unit-RSS and a thresholded support survive.
    """
    import math
    from physics.online_smaps import phase_correct_maps
    k, sm, obj = phantom()
    H, W = obj.shape
    yy = torch.arange(H)[:, None].float() / H
    xx = torch.arange(W)[None, :].float() / W
    x = obj * torch.exp(1j * (2.0 * yy + 1.5 * xx ** 2)).to(torch.complex64)
    coils = sm * x
    sm = torch.where(obj.abs()[None, None] > 0, sm, torch.zeros_like(sm))   # a hard support
    calib = ifftc(center_mask(torch.ones(1, 1, 1, W), 9).to(torch.complex64) * fftc(coils))

    g = torch.Generator().manual_seed(1)
    phi = 2 * math.pi * torch.rand(1, 1, H, W, generator=g)
    a = phase_correct_maps(sm, calib)
    b = phase_correct_maps(sm * torch.exp(1j * phi), calib)
    check("any per-pixel map phase cancels exactly", torch.allclose(a, b, atol=1e-5),
          f"max diff {float((a - b).abs().max()):.1e}")
    check("magnitudes (so unit-RSS) unchanged", torch.allclose(a.abs(), sm.abs(), atol=1e-6))
    check("a zeroed (thresholded) support stays zero",
          bool((a[:, :, obj.abs() == 0] == 0).all()))

    implied = (a.conj() * coils).sum(1)[0]                  # the image these maps imply
    ph = torch.angle(implied)[obj.abs() > 0].abs()
    check("the implied image is near-real (its low-res phase removed)",
          float(ph.quantile(0.95)) < 0.3, f"95th pct |phase| {float(ph.quantile(0.95)):.3f} rad")

    if not have_lapack():
        print("[skip] online_smaps(phase_correct=True) needs LAPACK (svd)")
        return
    m = make_acc_mask((H, W), 4, acs_lines=9).reshape(1, 1, H, W)
    y = m * fftc(coils)
    for method in ("walsh", "espirit"):
        kw = dict(kernel_size=4) if method == "espirit" else {}
        s0 = online_smaps(y, m, method=method, acs_lines=9, **kw)
        s1 = online_smaps(y, m, method=method, acs_lines=9, phase_correct=True, **kw)
        check(f"{method}: phase_correct changes phase only",
              torch.allclose(s0.abs(), s1.abs(), atol=1e-5))


def main():
    for fn in (test_acs_line_count, test_acs_block_is_asymmetric,
               test_only_the_acs_reaches_the_estimator,
               test_walsh_window_default, test_cost_bound,
               test_estimators_run, test_walsh_phase_reference,
               test_phase_correct, test_bad_inputs):
        print(f"\n--- {fn.__name__}")
        fn()
    print(f"\n{'FAILED: ' + ', '.join(FAIL) if FAIL else 'all checks passed'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
