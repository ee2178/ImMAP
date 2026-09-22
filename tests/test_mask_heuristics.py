"""
Sampling-mask heuristics (`physics/mask.py`), against `Sljiva/src/mask.jl`.

Run with `python -m tests.test_mask_heuristics`.

THE FINDING THIS PINS. With `adjust_accel=False` -- every run before this
test existed -- the ACS block is ADDED on top of a grid laid every `accel`
lines, so R is nominal. On a 320-line axis with 20 ACS lines a "R=16" mask
samples 39 lines: an effective R of 8.2. `adjust_accel=True` reproduces
`generate_uniform_mask`, which charges the ACS to the budget.

Everything here is integer geometry; no linear algebra, so it runs locally.
"""

import torch

from physics.mask import (
    effective_accel, get_mask_cached, make_acc_mask, resolve_acs_lines,
)
from physics.online_smaps import acs_block

FAIL = []
N = 320
SHAPE = (640, N)


def check(name, ok, detail=""):
    print(f"[{'ok ' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def lines(m):
    return int(m[0, 0, 0].sum())


def julia_uniform(accel, center_frac=0.04, adjust=True, offset=0):
    """A literal transcription of generate_uniform_mask, 0-indexed."""
    nc = int(round(N * center_frac))
    if nc % 2 == 0:
        nc -= 1
    adj = int(round((N - nc) / (N / accel - nc))) if adjust else accel
    m = torch.zeros(N, dtype=torch.bool)
    m[offset::adj] = True
    pad = (N - nc + 1) // 2
    m[pad:pad + nc] = True
    return m


# ---------------------------------------------------------------------------
def test_default_is_unchanged():
    """Old configs must reproduce old masks exactly."""
    for R, want in ((4, 95), (8, 57), (16, 39)):
        m = make_acc_mask(SHAPE, accel=R, acs_lines=20, dim=1, mode="uniform")
        check(f"default R={R} still samples {want} lines", lines(m) == want,
              f"got {lines(m)} (effective R {effective_accel(m):.2f})")


def test_nominal_r_is_not_real_without_adjust():
    m = make_acc_mask(SHAPE, accel=16, acs_lines=20, dim=1, mode="uniform")
    eff = effective_accel(m)
    check("a nominal R=16 with 20 ACS lines runs at about R=8", 7.5 < eff < 9.0,
          f"effective R = {eff:.2f}")


def test_matches_julia_uniform():
    for R in (4, 8, 16):
        ours = make_acc_mask(SHAPE, accel=R, center_frac=0.04, dim=1,
                             mode="uniform", adjust_accel=True)[0, 0, 0] > 0
        ref = julia_uniform(R)
        check(f"R={R} uniform matches generate_uniform_mask line for line",
              torch.equal(ours, ref),
              f"{int((ours != ref).sum())} lines differ; ours {int(ours.sum())}, "
              f"julia {int(ref.sum())}")


def test_adjust_hits_nominal_rate():
    for R in (4, 8, 16):
        u = make_acc_mask(SHAPE, accel=R, center_frac=0.04, dim=1,
                          mode="uniform", adjust_accel=True)
        r = make_acc_mask(SHAPE, accel=R, center_frac=0.04, dim=1,
                          mode="random", adjust_accel=True, seed=0)
        eu, er = effective_accel(u), effective_accel(r)
        # uniform rounds the SPACING to an integer, so it lands near R, not on it
        check(f"R={R} uniform lands within 10% of nominal", abs(eu - R) < 0.1 * R,
              f"effective {eu:.2f}")
        check(f"R={R} random samples exactly floor(N/R) lines",
              lines(r) == N // R, f"{lines(r)} lines, want {N // R}")


def test_center_frac_geometry():
    nc = resolve_acs_lines(N, 20, 0.04)
    check("center_frac=0.04 at N=320 gives 13 lines (forced odd)", nc == 13, f"{nc}")
    check("without center_frac the config's count is used",
          resolve_acs_lines(N, 20, None) == 20)

    m = make_acc_mask(SHAPE, accel=8, center_frac=0.04, dim=1, mode="uniform",
                      adjust_accel=True)[0, 0, 0] > 0
    pad = (N - nc + 1) // 2
    check("the ACS block sits at Julia's pad = (N - Nc + 1) // 2",
          bool(m[pad:pad + nc].all()), f"start {pad}")

    # the online estimator must calibrate from THAT block, not the config's 20
    ax, ay = acs_block(m[None, None, None].float(), (1, 20, 640, N),
                       acs_lines=resolve_acs_lines(N, 20, 0.04))
    check("online maps calibrate from the 13-line ACS, not acs_lines=20",
          ay == 13, f"acs block {ax}x{ay}")


def test_guard_and_cache():
    try:
        make_acc_mask(SHAPE, accel=16, acs_lines=20, dim=1, mode="uniform",
                      adjust_accel=True)
        check("20 ACS lines at R=16 (the whole budget) is refused", False, "no error")
    except ValueError:
        check("20 ACS lines at R=16 (the whole budget) is refused", True)

    img = torch.zeros(1, 1, *SHAPE)
    a = get_mask_cached(img, 16, 20, "uniform")
    b = get_mask_cached(img, 16, 20, "uniform", center_frac=0.04, adjust_accel=True)
    check("the cache keeps adjusted and unadjusted masks apart",
          lines(a) != lines(b), f"{lines(a)} vs {lines(b)} lines")


def main():
    for fn in (test_default_is_unchanged, test_nominal_r_is_not_real_without_adjust,
               test_matches_julia_uniform, test_adjust_hits_nominal_rate,
               test_center_frac_geometry, test_guard_and_cache):
        print(f"\n--- {fn.__name__}")
        fn()
    print(f"\n{'FAILED: ' + ', '.join(FAIL) if FAIL else 'all checks passed'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
