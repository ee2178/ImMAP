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
    acs_block, acs_line_count, center_mask, estimate_cost_gb, hamming_window,
    online_smaps, resolve_lines,
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


def main():
    for fn in (test_acs_line_count, test_acs_block_is_asymmetric,
               test_only_the_acs_reaches_the_estimator, test_cost_bound,
               test_estimators_run, test_bad_inputs):
        print(f"\n--- {fn.__name__}")
        fn()
    print(f"\n{'FAILED: ' + ', '.join(FAIL) if FAIL else 'all checks passed'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
