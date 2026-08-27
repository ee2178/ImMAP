"""
`FenchelProx` direct form: `z * min(1, t/|z|)` instead of `z - prox_g(z)`.

Run with `python -m tests.test_fenchel_direct`.

The rewrite must be a pure speed change, so the bar is EXACTNESS, not
closeness: same values, same gradients, and the same network output to fp32
roundoff. `FenchelProx.DIRECT` is the kill switch the A/B toggles.
"""

import torch

from models.prox import FenchelProx, SoftThreshold, build_prox

PASS, FAIL, SKIPPED = [], [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def both_ways(fn):
    """Run `fn()` with the direct form on and off."""
    FenchelProx.DIRECT = True
    a = fn()
    FenchelProx.DIRECT = False
    b = fn()
    FenchelProx.DIRECT = True
    return a, b


def test_values_and_grads():
    print("\n[the two forms agree]")
    for dtype in (torch.complex64, torch.float32):
        torch.manual_seed(0)
        prox = FenchelProx(SoftThreshold(8, tau0=0.3, degrees=1))
        z = torch.randn(4, 8, 16, 16, dtype=dtype) * 2
        z = torch.cat([z, torch.zeros_like(z[:1])], 0)      # include exact zeros
        sig = torch.full((z.shape[0], 1, 1, 1), 0.02)

        a, b = both_ways(lambda: prox(z, sig, {})[0])
        d = float((a - b).abs().max())
        check(f"values match ({str(dtype).split('.')[-1]})", d < 1e-5, f"max|diff| {d:.2e}")
        check(f"finite at z=0 ({str(dtype).split('.')[-1]})",
              bool(torch.isfinite(a.abs()).all()))

    torch.manual_seed(0)
    prox = FenchelProx(SoftThreshold(4, tau0=0.3, degrees=1))
    z0 = torch.randn(2, 4, 8, 8, dtype=torch.complex64)
    sig = torch.full((2, 1, 1, 1), 0.02)

    def grad():
        z = z0.clone().requires_grad_(True)
        out = prox(z, sig, {})[0]
        return torch.autograd.grad(out.abs().pow(2).sum(), z)[0]

    ga, gb = both_ways(grad)
    d = float((ga - gb).abs().max())
    check("gradients match", d < 1e-5, f"max|diff| {d:.2e}")

    # the threshold's own gradient has to survive too
    def tau_grad():
        pr = FenchelProx(SoftThreshold(4, tau0=0.3, degrees=1))
        torch.manual_seed(5)
        with torch.no_grad():
            pr.prox.tau.weight.add_(0.1 * torch.randn_like(pr.prox.tau.weight))
        pr(z0, sig, {})[0].abs().pow(2).sum().backward()
        return pr.prox.tau.weight.grad.clone()

    ta, tb = both_ways(tau_grad)
    d = float((ta - tb).abs().max())
    check("gradient w.r.t. the threshold matches", d < 1e-5, f"max|diff| {d:.2e}")


def test_dispatch():
    print("\n[dispatch]")
    p_soft = build_prox(8, window=1, dual=True, tau0=1e-3, degrees=1)
    check("dual soft prox is a FenchelProx", isinstance(p_soft, FenchelProx))
    check("it wraps a SoftThreshold", isinstance(p_soft.prox, SoftThreshold))
    check("SoftThreshold exposes fenchel", hasattr(p_soft.prox, "fenchel"))

    # anything without a `fenchel` must still work through the subtraction
    class Plain(torch.nn.Module):
        def forward(self, z, sigma=None, cache=None):
            return 0.25 * z, cache

    fp = FenchelProx(Plain())
    z = torch.randn(1, 3, 4, 4)
    out, _ = fp(z, None, {})
    check("a prox without `fenchel` falls back to z - prox(z)",
          torch.allclose(out, z - 0.25 * z))


def test_network_output_unchanged():
    print("\n[the whole model is unchanged]")
    from models.mg_lpds import MGLPDSNet

    base = dict(M=16, C=1, P=3, s=2, widen=1, degrees=1, lam0=1e-3, tau0=0.5,
                theta0=0.0, alpha0=1.0, is_complex=True, preproc="identity",
                resize_noise=True)
    torch.manual_seed(1)
    net = MGLPDSNet(K=[2, [2, 2, 2]], **base).eval()
    torch.manual_seed(2)
    y = torch.randn(1, 1, 32, 32, dtype=torch.complex64) * 0.3
    sig = torch.full((1, 1, 1, 1), 0.02)

    with torch.no_grad():
        a, b = both_ways(lambda: net(y, E=None, sigma=sig)[0])
    rel = float((a - b).abs().norm() / b.abs().norm())
    check("MGLPDSNet output identical", rel < 1e-6, f"relative {rel:.2e}")

    def grad_norm():
        net.zero_grad(set_to_none=True)
        net(y, E=None, sigma=sig)[0].abs().pow(2).sum().backward()
        return sum(float(p.grad.abs().sum()) for p in net.parameters()
                   if p.grad is not None)

    ga, gb = both_ways(grad_norm)
    rel = abs(ga - gb) / abs(gb)
    check("MGLPDSNet gradients identical", rel < 1e-5, f"relative {rel:.2e}")


def test_real_fastmri():
    """The rewrite must be exact on real magnitudes, not just on randn.

    Synthetic z is unit-scale and dense; a real latent after a real forward pass
    is neither. `t / |z|` is where the two forms could diverge -- on a genuinely
    sparse code many entries sit near zero, which is exactly the branch the
    `clamp_min` guard covers.
    """
    print("\n[real fastMRI data]")
    from tests.fastmri_fixture import load_slice, banner
    from models.mg_lpds import MGLPDSNet
    from operators import FFT2D, Mask, Sense
    from operators.noise import mri_awgn
    from physics.mask import make_acc_mask

    sample = load_slice(anatomy="brain", pad_multiple=8)
    print(banner(sample))
    if sample is None:
        SKIPPED.append("real fastMRI data")
        return

    image, smaps = sample["image"], sample["smaps"]
    H, W = image.shape[-2:]
    mask = make_acc_mask((H, W), 8, acs_lines=20)
    while mask.dim() < 4:
        mask = mask.unsqueeze(0)
    E = Mask(mask) @ FFT2D() @ Sense(smaps)
    y, _, _ = mri_awgn(image, mask, smaps, 0.005, "uniform")
    sig = torch.full((1, 1, 1, 1), 0.005)

    from operators.truncate import embed_operator
    base = dict(M=16, C=1, P=3, s=2, widen=1, degrees=1, lam0=1e-3, tau0=0.5,
                theta0=0.0, alpha0=1.0, is_complex=True, preproc="kspace",
                resize_noise=True)
    torch.manual_seed(1)
    net = MGLPDSNet(K=[1, [2, 2, 2]], **base).eval()
    E_emb, T = embed_operator(E, (H, W), net.pad_stride)

    with torch.no_grad():
        a, b = both_ways(lambda: T.forward(net(y, E=E_emb, sigma=sig)[0]))
    rel = float((a - b).abs().norm() / b.abs().norm().clamp_min(1e-20))
    check("output identical on a real slice", rel < 1e-6, f"relative {rel:.2e}")

    # Report how sparse the real code is, but do NOT assert on it: at
    # initialisation the dictionary is untrained and the code is dense, so
    # nothing sits near the eps guard. That is a fact about an untrained
    # network, not a property of the rewrite. The z = 0 branch is covered
    # exactly, and deterministically, by the exact-zeros row in
    # test_values_and_grads above.
    with torch.no_grad():
        _, latent = net(y, E=E_emb, sigma=sig)
    z = latent[1] if isinstance(latent, tuple) else latent
    a_z = z.abs()
    tiny = float((a_z < 1e-6 * a_z.max()).float().mean())
    print(f"        real code: {100 * tiny:.2f}% of entries below 1e-6 of max, "
          f"min|z|/max|z| = {float(a_z.min() / a_z.max().clamp_min(1e-30)):.2e}")
    print("        (dense at init, as expected; the z=0 branch is covered "
          "synthetically)")

    def grad_sum():
        net.zero_grad(set_to_none=True)
        out = T.forward(net(y, E=E_emb, sigma=sig)[0])
        (out.abs() - image.abs()).pow(2).mean().backward()
        return sum(float(p.grad.abs().sum()) for p in net.parameters()
                   if p.grad is not None)

    ga, gb = both_ways(grad_sum)
    rel = abs(ga - gb) / max(abs(gb), 1e-20)
    check("gradients identical on a real slice", rel < 1e-5, f"relative {rel:.2e}")


if __name__ == "__main__":
    test_values_and_grads()
    test_dispatch()
    test_network_output_unchanged()
    test_real_fastmri()
    tail = f", {len(SKIPPED)} section(s) SKIPPED: {', '.join(SKIPPED)}" if SKIPPED else ""
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed{tail}")
    for f in FAIL:
        print(f"  FAILED: {f}")
    raise SystemExit(1 if FAIL else 0)
