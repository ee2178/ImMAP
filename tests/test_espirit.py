"""
ESPIRiT map estimation (`physics/smaps.py`).

Run with `python -m tests.test_espirit`.

THE REGRESSION THIS EXISTS FOR. The row-space truncation keeps the singular
vectors above `thresh_rowspace`, and that count differs from slice to slice.
It used to be applied as `V[:, :, :counts.max()]` -- the widest retention in
the batch, given to every slice in it. Retaining extra basis vectors can only
raise the eigenvalue map (with the complete basis lam == 1 everywhere), so a
slice batched with a noisier one came out with a DILATED support, and the maps
depended on `scripts/make_espirit_smaps.py --chunk`, documented there as a pure
memory knob. The fix zeroes the tail per sample instead.

So the property under test is: a slice's maps must not depend on what it was
batched with.

`torch.linalg.svd` needs LAPACK, which the local anaconda torch does not have;
the whole module skips rather than fails there, and runs on the cluster.
"""

import torch

from operators.fourier import fftc
from physics.smaps import espirit, espirit_soft

FAIL = []


def check(name, ok, detail=""):
    print(f"[{'ok ' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def have_lapack():
    try:
        torch.linalg.svd(torch.randn(4, 4, dtype=torch.complex64))
        return True
    except Exception as e:                       # RuntimeError on a LAPACK-less build
        print(f"SKIPPED: torch.linalg.svd is unavailable here ({type(e).__name__}: "
              f"{str(e).splitlines()[0][:70]})")
        return False


# ---------------------------------------------------------------------------
def coil_phantom(C=4, H=48, W=48, smooth=True, noise=0.0, seed=0):
    """`(1, C, H, W)` k-space of a disc seen through smooth coil maps."""
    g = torch.Generator().manual_seed(seed)
    y = torch.arange(H)[:, None] - H / 2
    x = torch.arange(W)[None, :] - W / 2
    obj = ((y ** 2 + x ** 2) < (0.30 * H) ** 2).to(torch.complex64)

    maps = []
    for c in range(C):
        cy, cx = H * (0.5 + 0.6 * torch.cos(torch.tensor(2 * 3.14159 * c / C))), \
                 W * (0.5 + 0.6 * torch.sin(torch.tensor(2 * 3.14159 * c / C)))
        d2 = (y - (cy - H / 2)) ** 2 + (x - (cx - W / 2)) ** 2
        s = torch.exp(-d2 / (2 * (0.9 * H) ** 2))
        if not smooth:                            # structured, higher-rank maps
            s = s * (1 + 0.5 * torch.cos(6.28 * (y / H + x / W)))
        maps.append(s.to(torch.complex64))
    sm = torch.stack(maps)[None]
    sm = sm / sm.abs().pow(2).sum(1, keepdim=True).sqrt().clamp_min(1e-8)

    coils = sm * obj[None, None]
    k = fftc(coils)
    if noise > 0:
        k = k + noise * torch.complex(
            torch.randn(k.shape, generator=g), torch.randn(k.shape, generator=g))
    return k


def maps_equal(a, b, tol=1e-4):
    """Per-pixel agreement up to nothing: `espirit` fixes the per-pixel phase
    by referencing it to the strongest coil, so maps from two runs on the same
    data are comparable element by element."""
    return float((a - b).abs().max()), float((a - b).abs().max()) < tol


# ---------------------------------------------------------------------------
def test_batch_does_not_change_a_slice():
    """The regression: same slice, alone and batched with a noisier one."""
    kw = dict(acs_size=(16, 16), kernel_size=4, thresh_rowspace=0.05,
              thresh_eig=0.9, maxit=200)
    clean = coil_phantom(noise=0.0, seed=1)
    noisy = coil_phantom(noise=0.05, smooth=False, seed=2)

    torch.manual_seed(0)
    alone = espirit(clean, **kw)
    torch.manual_seed(0)
    batched = espirit(torch.cat([clean, noisy]), **kw)[:1]

    d, ok = maps_equal(alone, batched, tol=1e-3)
    check("maps for a slice are the same alone and in a batch", ok,
          f"max |delta| = {d:.2e}")

    sup_a = (alone.abs().sum(1) > 0)
    sup_b = (batched.abs().sum(1) > 0)
    agree = float((sup_a == sup_b).float().mean())
    check("the SUPPORT is the same alone and in a batch", agree == 1.0,
          f"{agree:.4%} of pixels agree; batched keeps "
          f"{float(sup_b.float().mean()):.1%} vs {float(sup_a.float().mean()):.1%} alone")


def test_batch_order_does_not_matter():
    """Same two slices, swapped: each one's maps must be unchanged."""
    kw = dict(acs_size=(16, 16), kernel_size=4, thresh_eig=0.9, maxit=200)
    a = coil_phantom(noise=0.0, seed=3)
    b = coil_phantom(noise=0.08, smooth=False, seed=4)

    torch.manual_seed(0)
    ab = espirit(torch.cat([a, b]), **kw)
    torch.manual_seed(0)
    ba = espirit(torch.cat([b, a]), **kw)

    d, ok = maps_equal(ab[:1], ba[1:], tol=1e-3)
    check("swapping the batch order leaves each slice's maps alone", ok,
          f"max |delta| = {d:.2e}")


def test_soft_maps_are_batch_independent():
    kw = dict(acs_size=(16, 16), kernel_size=4, thresh_eig=0.9, num_maps=2)
    clean = coil_phantom(noise=0.0, seed=5)
    noisy = coil_phantom(noise=0.05, smooth=False, seed=6)

    torch.manual_seed(0)
    alone = espirit_soft(clean, **kw)
    torch.manual_seed(0)
    batched = espirit_soft(torch.cat([clean, noisy]), **kw)[:1]
    d, ok = maps_equal(alone, batched, tol=1e-3)
    check("espirit_soft is batch-independent too", ok, f"max |delta| = {d:.2e}")


def test_phase_reference_is_the_strongest_coil():
    """Sljiva references the phase to the coil with the most ACS energy.

    Coil 0 was a porting slip (`src/solver.jl` picks `Cref = argmax(...)`, and
    `physics/smaps.py::walsh` does the same): where the reference coil is dark
    its phase is noise, and the ground truth `x = S^H c` inherits it.
    """
    k = coil_phantom(C=4, noise=0.0, seed=8)
    # Make coil 0 the WEAKEST, so referencing to it and to the strongest coil
    # cannot coincide.
    k = k.clone()
    k[:, 0] *= 0.02
    k[:, 2] *= 3.0

    sm = espirit(k, acs_size=(16, 16), kernel_size=4, thresh_eig=0.9, maxit=200)
    sup = (sm.abs().sum(1, keepdim=True) > 0)

    acs = k[:, :, 24 - 8:24 + 8, 24 - 8:24 + 8]
    cref = int(acs.abs().pow(2).sum(dim=(-2, -1)).argmax(dim=1))
    check("the reference coil is the strongest one, not coil 0",
          cref == 2, f"argmax energy = coil {cref}")

    ref_map = sm[:, cref:cref + 1][sup]
    imag = float(ref_map.imag.abs().max()) if ref_map.numel() else float("nan")
    real_min = float(ref_map.real.min()) if ref_map.numel() else float("nan")
    check("the reference coil's map is real and non-negative inside the support",
          imag < 1e-5 and real_min >= -1e-6,
          f"max |imag| = {imag:.2e}, min real = {real_min:.2e}")

    coil0 = sm[:, 0:1][sup]
    check("coil 0 is NOT the one made real (it would be, before the fix)",
          float(coil0.imag.abs().max()) > 1e-4,
          f"max |imag| on coil 0 = {float(coil0.imag.abs().max()):.2e}")


def test_maps_explain_the_data():
    """A sanity floor: on noiseless rank-1 data the SENSE residual is tiny."""
    from operators.fourier import ifftc
    k = coil_phantom(noise=0.0, seed=7)
    sm = espirit(k, acs_size=(16, 16), kernel_size=4, thresh_eig=0.9, maxit=200)
    coil = ifftc(k)
    x = (sm.conj() * coil).sum(1, keepdim=True)
    sup = (sm.abs().sum(1, keepdim=True) > 0)
    num = ((coil - sm * x) * sup).norm()
    den = (coil * sup).norm().clamp_min(1e-12)
    res = float(num / den)
    check("the maps explain the coil data inside their support", res < 0.05,
          f"residual {res:.4f}")

    rss = sm.abs().pow(2).sum(1).sqrt()
    inside = rss[sup[:, 0]]
    err = float((inside - 1).abs().max()) if inside.numel() else float("nan")
    check("maps are unit-RSS inside the support (mri_awgn assumes it)",
          err < 1e-4, f"max |‖s‖-1| = {err:.2e}")


def main():
    if not have_lapack():
        return 0
    for fn in (test_batch_does_not_change_a_slice, test_batch_order_does_not_matter,
               test_soft_maps_are_batch_independent,
               test_phase_reference_is_the_strongest_coil, test_maps_explain_the_data):
        print(f"\n--- {fn.__name__}")
        fn()
    print(f"\n{'FAILED: ' + ', '.join(FAIL) if FAIL else 'all checks passed'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
