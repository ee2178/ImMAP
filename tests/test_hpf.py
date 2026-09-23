"""
`operators/hpf.py::HighPassFilter` -- that it high-passes, and stays self-adjoint.

Run with `python -m tests.test_hpf`.

THE BUG THIS PINS. `get_window` built its Gaussian from `torch.fft.fftfreq`, which is in
UNSHIFTED order (DC at index 0), and `forward` applies that window to `fftc(x)`, which is
CENTERED (DC at index n // 2). The `1 - gaussian` notch therefore sat at the array corner
instead of at DC, and the operator low-passed: a constant image came through at 0.998 of its
amplitude and a Nyquist checkerboard at 0.000 -- precisely inverted.

`models/ipalmnet.py` and `models/diffipalmnet.py` both build one of these with
`sigma_hpf=0.2` (config/knee/ipalm.json), so any iPALM checkpoint trained before the fix saw a
LOW-pass where the architecture intends a high-pass.

Pure FFTs, no linear algebra, so it runs locally.
"""

import torch

from operators.hpf import HighPassFilter

FAIL = []
H = W = 32
SIGMA = 0.2


def check(name, got, want, tol):
    ok = abs(got - want) <= tol
    print(f"  {name:<34s} {got:7.4f}  (want {want:.2f} +- {tol:g})  {'OK' if ok else 'FAIL'}")
    if not ok:
        FAIL.append(name)


def main():
    psi = HighPassFilter(SIGMA)

    # DC must be annihilated, Nyquist must pass.
    dc = torch.ones(1, 1, H, W)
    y, x = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float(), indexing="ij")
    nyq = ((-1.0) ** (x + y)).view(1, 1, H, W)

    print("direction:")
    check("constant (DC) -> ~0", float(psi.forward(dc).abs().mean()), 0.0, 5e-3)
    check("Nyquist checkerboard -> ~1",
          float(psi.forward(nyq).abs().mean() / nyq.abs().mean()), 1.0, 5e-3)

    # Monotone in frequency: a low-frequency cosine must be attenuated more than a high one.
    print("monotonicity (gain must grow with frequency):")
    gains = []
    for k in (1, 2, 4, 8, 16):
        wave = torch.cos(2 * torch.pi * k * x / W).view(1, 1, H, W)
        gains.append(float(psi.forward(wave).abs().mean() / wave.abs().mean()))
        print(f"  k={k:<3d} gain {gains[-1]:.4f}")
    if any(b < a - 1e-6 for a, b in zip(gains, gains[1:])):
        print("  FAIL: gain is not non-decreasing in frequency")
        FAIL.append("monotone gain")
    else:
        print("  OK: non-decreasing")

    # The window is real and symmetric, so the operator is self-adjoint: <psi(a), b> = <a, psi(b)>
    print("self-adjointness:")
    a = torch.randn(1, 1, H, W)
    b = torch.randn(1, 1, H, W)
    lhs = float((psi.forward(a).conj() * b).real.sum())
    rhs = float((a.conj() * psi.forward(b)).real.sum())
    check("<psi a, b> - <a, psi b>", lhs - rhs, 0.0, 1e-3 * max(abs(lhs), 1.0))

    # The cache must not hand back a window of the wrong size.
    print("window cache:")
    small = torch.ones(1, 1, 16, 16)
    psi.forward(small)
    w = psi.get_window(small)
    check("reshapes for a new spatial size", float(w.shape[-1]), 16.0, 0)
    w2 = psi.get_window(torch.ones(1, 1, H, W))
    check("and back again", float(w2.shape[-1]), float(W), 0)

    print("\n" + ("ALL PASS" if not FAIL else f"{len(FAIL)} FAILED: {FAIL}"))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
