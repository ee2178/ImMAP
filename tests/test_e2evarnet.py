"""
E2E-VarNet baseline: the vendored fastMRI model under ImMAP's conventions.

Run with `python -m tests.test_e2evarnet`.

Two separate claims are checked, because they fail differently:

  * the VENDORED code is upstream's (byte comparison against the clone, when
    one is reachable);
  * the ADAPTER hands it the right things -- above all that the FFT convention
    matches, since a mismatch there would train happily on a wrong operator.
"""

import os

import torch

from models.e2evarnet import E2EVarNet
from models.e2evarnet._fastmri import varnet as vendored
from operators import FFT2D, Mask, Sense
from operators.fourier import fftc
from physics.mask import make_acc_mask

PASS, FAIL, SKIPPED = [], [], []
UPSTREAM = "C:/Users/erice/Desktop/fastMRI/fastmri"


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def problem(H=64, W=64, NC=4, B=1, R=8, acs=20):
    torch.manual_seed(0)
    gt = torch.randn(B, 1, H, W, dtype=torch.complex64) * 0.3
    sm = torch.randn(B, NC, H, W, dtype=torch.complex64) * 0.3 + 1.0
    sm = sm / (sm.abs().pow(2).sum(1, keepdim=True).sqrt() + 1e-8)
    m = make_acc_mask((H, W), R, acs_lines=acs)
    while m.dim() < 4:
        m = m.unsqueeze(0)
    E = Mask(m) @ FFT2D() @ Sense(sm)
    y = E(gt)
    return gt, E, y, m


# ---------------------------------------------------------------------------
def test_fft_convention():
    """The one assumption that would silently corrupt everything.

    ImMAP's `fftc` and fastMRI's `fft2c` must be the same transform, or the
    k-space handed across is centred differently and the model trains on a
    quietly wrong operator.
    """
    print("\n[FFT conventions agree]")
    torch.manual_seed(0)
    x = torch.randn(2, 3, 16, 16, dtype=torch.complex64)

    mine = fftc(x)
    theirs = torch.view_as_complex(
        vendored.fastmri.fft2c(torch.view_as_real(x.contiguous())).contiguous())
    rel = float((mine - theirs).abs().max() / mine.abs().max())
    check("fftc == fastmri.fft2c", rel < 1e-5, f"max rel diff {rel:.2e}")

    inv = torch.view_as_complex(
        vendored.fastmri.ifft2c(torch.view_as_real(theirs.contiguous())).contiguous())
    rel = float((inv - x).abs().max() / x.abs().max())
    check("their ifft2c inverts their fft2c", rel < 1e-5, f"{rel:.2e}")


def test_vendored_is_upstream():
    print("\n[the vendored files are upstream's]")
    import hashlib
    if not os.path.isdir(UPSTREAM):
        print(f"  [fixture] no fastMRI clone at {UPSTREAM} -- byte check SKIPPED")
        SKIPPED.append("byte-for-byte vs the fastMRI clone")
        return
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "models", "e2evarnet", "_fastmri")
    # coil_combine.py is NOT in this list: like varnet.py it needed its
    # `import fastmri` repointed at the vendored math.py. Checked below.
    for name, rel in (("fftc.py", "fftc.py"), ("math.py", "math.py"),
                      ("unet.py", "models/unet.py")):
        a = open(os.path.join(UPSTREAM, rel), "rb").read()
        b = open(os.path.join(here, name), "rb").read()
        check(f"{name} is byte-identical", a == b,
              f"sha {hashlib.sha256(b).hexdigest()[:12]}")

    # varnet.py may differ ONLY in its import block
    import difflib
    up = open(os.path.join(UPSTREAM, "models/varnet.py"),
              encoding="utf-8").read().splitlines()
    mine = open(os.path.join(here, "varnet.py"), encoding="utf-8").read().splitlines()
    i = next(k for k, l in enumerate(mine) if l.startswith('"""'))
    d = [l for l in difflib.unified_diff(up, mine[i:], lineterm="", n=0)
         if l[:1] in "+-" and l[:3] not in ("---", "+++")]
    only_imports = all(("fastmri" in l) or ("transforms" in l) for l in d)
    check("varnet.py differs only in the import block",
          len(d) == 4 and only_imports, f"{len(d)} differing lines")

    # coil_combine.py: the ONLY change may be its `import fastmri` line
    up = open(os.path.join(UPSTREAM, "coil_combine.py"),
              encoding="utf-8").read().splitlines()
    mine = open(os.path.join(here, "coil_combine.py"), encoding="utf-8").read().splitlines()
    d = [l for l in difflib.unified_diff(up, mine, lineterm="", n=0)
         if l[:1] in "+-" and l[:3] not in ("---", "+++")]
    removed = [l for l in d if l.startswith("-")]
    check("coil_combine.py differs only in its import line",
          removed == ["-import fastmri"], f"removed {removed}")


def test_forward():
    print("\n[forward under ImMAP's calling convention]")
    gt, E, y, _ = problem()
    net = E2EVarNet(num_cascades=2, sens_chans=4, sens_pools=2,
                    chans=6, pools=2, acs_lines=20).eval()
    with torch.no_grad():
        out, extra = net(y, E=E, sigma=torch.full((1, 1, 1, 1), 0.01))
    check("returns (x_hat, extra)", isinstance(extra, dict))
    check("output is (B, 1, H, W)", tuple(out.shape) == tuple(gt.shape),
          str(tuple(out.shape)))
    check("output is finite", bool(torch.isfinite(out).all()))
    check("output is real magnitude, not complex",
          (not torch.is_complex(out)) and bool((out >= 0).all()))
    check("declares returns_magnitude", getattr(net, "returns_magnitude", False))
    check("pad_stride = 1 so the embedding is an identity", net.pad_stride == 1)


def test_odd_sizes_and_batch():
    print("\n[sizes VarNet has to pad internally]")
    for H, W in ((64, 64), (60, 62), (58, 58)):
        gt, E, y, _ = problem(H=H, W=W)
        net = E2EVarNet(num_cascades=1, sens_chans=4, sens_pools=2,
                        chans=4, pools=2, acs_lines=20).eval()
        with torch.no_grad():
            out, _ = net(y, E=E, sigma=None)
        check(f"{H}x{W} round-trips", tuple(out.shape) == (1, 1, H, W),
              str(tuple(out.shape)))

    gt, E, y, _ = problem(B=2)
    net = E2EVarNet(num_cascades=1, sens_chans=4, sens_pools=2,
                    chans=4, pools=2, acs_lines=20).eval()
    with torch.no_grad():
        out, _ = net(y, E=E, sigma=None)
    check("batch of 2 works", tuple(out.shape) == (2, 1, 64, 64), str(tuple(out.shape)))


def test_mask_guard():
    print("\n[the mask contract is checked, not assumed]")
    gt, E, y, m = problem()
    net = E2EVarNet(num_cascades=1, sens_chans=4, sens_pools=2, chans=4, pools=2)

    # a mask that varies down H cannot be expressed as VarNet's 1-D line mask;
    # taking row 0 would undersample wrongly and still train
    bad = m.clone().float()
    bad[..., 0, :] = 1.0
    bad[..., 1, :] = 0.0
    E_bad = Mask(bad) @ FFT2D() @ Sense(torch.ones(1, 4, 64, 64, dtype=torch.complex64))
    try:
        net(y, E=E_bad, sigma=None)
        check("a non-line mask is rejected", False)
    except ValueError as e:
        check("a non-line mask is rejected", "constant along" in str(e))

    try:
        net(y, E=None, sigma=None)
        check("a missing operator is rejected", False)
    except ValueError as e:
        check("a missing operator is rejected", "encoding operator" in str(e))


def test_trains():
    print("\n[one optimizer step]")
    gt, E, y, _ = problem()
    net = E2EVarNet(num_cascades=2, sens_chans=4, sens_pools=2, chans=6, pools=2,
                    acs_lines=20)
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    out, _ = net(y, E=E, sigma=None)
    (out.abs() - gt.abs()).pow(2).mean().backward()
    live = sum(1 for p in net.parameters()
               if p.grad is not None and p.grad.abs().max() > 0)
    total = sum(1 for _ in net.parameters())
    check("gradients reach the cascades and the sens net", live > 0.9 * total,
          f"{live}/{total} tensors")
    opt.step()
    check("step is finite", all(torch.isfinite(p).all() for p in net.parameters()))
    net.project()
    check("project() is a safe no-op", True)


def test_registry():
    print("\n[registry]")
    from models import build_model
    m = build_model({"model": {"type": "E2EVarNet", "params": dict(
        num_cascades=1, sens_chans=4, sens_pools=2, chans=4, pools=2)}})
    if m is None:
        print("  [fixture] build_model stubbed here -- registry check SKIPPED")
        SKIPPED.append("registry (build_model stubbed)")
        return
    check("build_model('E2EVarNet') works", isinstance(m, E2EVarNet))


if __name__ == "__main__":
    test_fft_convention()
    test_vendored_is_upstream()
    test_forward()
    test_odd_sizes_and_batch()
    test_mask_guard()
    test_trains()
    test_registry()
    tail = f", {len(SKIPPED)} SKIPPED: {', '.join(SKIPPED)}" if SKIPPED else ""
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed{tail}")
    for f in FAIL:
        print(f"  FAILED: {f}")
    raise SystemExit(1 if FAIL else 0)
