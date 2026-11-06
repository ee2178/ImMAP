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
from physics.smaps import _eigh_batched, espirit, espirit_soft

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
def coil_phantom(C=4, H=48, W=48, smooth=True, noise=0.0, seed=0, phase=False):
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
        s = s.to(torch.complex64)
        if phase:
            # A DIFFERENT smooth phase per coil. Without it every map shares one
            # phase, referencing to any coil makes them all real, and a test of
            # WHICH coil is the reference cannot tell the answers apart.
            s = s * torch.exp(1j * (0.8 * (c + 1)) * (y / H + 0.6 * x / W))
        maps.append(s)
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
    k = coil_phantom(C=4, noise=0.0, seed=8, phase=True)
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


def test_eigh_sub_batches_are_exact():
    """_eigh_batched (sub-batches, for cuSOLVER's batched-solver limits) returns the same
    eigenpairs as one eigh call, and names non-finite input instead of a solver error."""
    g = torch.Generator().manual_seed(0)
    A = torch.randn(3, 37, 5, 6, generator=g, dtype=torch.complex64)
    G = A @ A.conj().transpose(-2, -1)                                  # (3, 37, 5, 5) Hermitian
    w0, v0 = torch.linalg.eigh(G)
    w1, v1 = _eigh_batched(G, max_batch=8)                              # 111 matrices / 8
    check("sub-batched eigenvalues match one call", bool(torch.allclose(w0, w1, atol=1e-4)),
          f"max diff {float((w0 - w1).abs().max()):.2e}")
    # eigenvectors are defined up to phase: compare the projectors v v^H
    P0 = v0[..., -1:] @ v0[..., -1:].conj().transpose(-2, -1)
    P1 = v1[..., -1:] @ v1[..., -1:].conj().transpose(-2, -1)
    check("sub-batched top eigenvectors match", bool(torch.allclose(P0, P1, atol=1e-4)))
    bad = G.clone()
    bad[1, 2, 0, 0] = float("nan")
    try:
        _eigh_batched(bad)
        check("non-finite input is named", False, "no error")
    except ValueError as e:
        check("non-finite input is named", "NaN/Inf" in str(e))


def test_lam_rethresholds_the_support_and_is_scale_free():
    """return_lam: the maps are unchanged, their support is exactly lam > thresh_eig, a lower
    threshold only grows it, and scaling the k-space moves neither lam nor the support
    (scripts/check_espirit_support.py relies on all three)."""
    k = coil_phantom(noise=0.02, seed=3)
    kw = dict(acs_size=(16, 16), kernel_size=4, thresh_eig=0.9, maxit=200)
    plain = espirit(k, **kw)
    sm, lam = espirit(k, return_lam=True, **kw)
    check("return_lam leaves the maps unchanged", maps_equal(plain, sm))
    sup = sm.abs().sum(1) > 0
    check("support == lam > thresh_eig", bool(torch.equal(sup, lam > 0.9)))
    check("lam in [0, 1]", bool((lam >= -1e-4).all() and (lam <= 1 + 1e-4).all()),
          f"range [{float(lam.min()):.4f}, {float(lam.max()):.4f}]")
    check("a lower threshold only grows the support", bool(((lam > 0.95) <= (lam > 0.8)).all()))
    _, lam_s = espirit(1e-4 * k, return_lam=True, **kw)
    check("lam is scale-free", bool(torch.allclose(lam, lam_s, atol=1e-3)),
          f"max diff {float((lam - lam_s).abs().max()):.2e}")


def main():
    if not have_lapack():
        return 0
    for fn in (test_eigh_sub_batches_are_exact,
               test_batch_does_not_change_a_slice, test_batch_order_does_not_matter,
               test_soft_maps_are_batch_independent,
               test_phase_reference_is_the_strongest_coil, test_maps_explain_the_data,
               test_lam_rethresholds_the_support_and_is_scale_free):
        print(f"\n--- {fn.__name__}")
        fn()
    print(f"\n{'FAILED: ' + ', '.join(FAIL) if FAIL else 'all checks passed'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
