"""
Sanity tests for the LADMM port (models/ladmm.py, solvers/cg.py, the operator
accessors and the Mask adjoint fix).

Run with:  python -m tests.test_ladmm
"""

import sys

import torch

from models.ladmm import (AltSplitCDLNet, LADMMLayer, LearnedHighpass,
                          LSmapUpdate, build_denoiser)
from operators import FFT2D, Identity, Mask, Sense
from operators.accessors import get_mask, get_sensitivity_map, set_sensitivity_map
from solvers.cg import batched_cg, tcg

torch.manual_seed(0)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"[{'ok ' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")


def rel(a, b):
    return (a - b).abs().max().item() / (b.abs().max().item() + 1e-12)


def make_mri(B=2, C=4, N=16, dtype=torch.complex64):
    smaps = torch.randn(B, C, N, N, dtype=dtype)
    smaps = smaps / smaps.abs().pow(2).sum(1, keepdim=True).sqrt()
    mask = (torch.rand(B, 1, N, N) > 0.4).to(dtype)
    E = Mask(mask) @ FFT2D() @ Sense(smaps)
    x = torch.randn(B, 1, N, N, dtype=dtype)
    return E, x, E(x), smaps, mask


# ---------------------------------------------------------------------------
def test_cg():
    E, x, y, _, _ = make_mri()
    rho = torch.tensor(0.5)
    b = E.adjoint(y)
    xs = tcg(E.gram, rho, b, max_iter=200, tol=1e-10)
    res = (E.gram(xs) + rho * xs - b).abs().max() / b.abs().max()
    check("tcg solves (E^H E + rho I) x = b", res.item() < 1e-4,
          f"rel residual = {res.item():.2e}")

    # per-image step sizes: badly scaled batch elements must both converge
    A = torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    scales = torch.tensor([1.0, 1e4]).view(2, 1)
    bb = torch.randn(2, 4) * scales

    def op(v):
        return v @ A.t()

    xb = batched_cg(op, bb, max_iter=50, tol=1e-12)
    err = (op(xb) - bb).abs() / bb.abs().clamp_min(1e-12)
    check("batched CG converges on a badly scaled batch",
          err.max().item() < 1e-5, f"max rel err = {err.max().item():.2e}")


def test_implicit_gradient():
    """The implicit backward must reproduce backprop-through-CG exactly."""
    torch.manual_seed(1)
    B, N = 2, 8
    smaps = torch.randn(B, 3, N, N, dtype=torch.complex64)
    mask = (torch.rand(B, 1, N, N) > 0.3).to(torch.complex64)
    b = torch.randn(B, 1, N, N, dtype=torch.complex64)

    def run(implicit):
        s = smaps.clone().requires_grad_(True)
        lam = torch.tensor([0.7]).requires_grad_(True)
        bb = b.clone().requires_grad_(True)
        E = Mask(mask) @ FFT2D() @ Sense(s)
        x = tcg(E.gram, lam.view(1, 1, 1, 1), bb, params=(s,),
                implicit=implicit, max_iter=400, tol=1e-12)
        (x.abs() ** 2).sum().backward()
        return x.detach(), s.grad, lam.grad, bb.grad

    x0, gs0, gl0, gb0 = run(False)
    x1, gs1, gl1, gb1 = run(True)
    check("implicit and unrolled solves agree", rel(x1, x0) < 1e-5)
    check("implicit d/db matches", rel(gb1, gb0) < 1e-3, f"rel={rel(gb1, gb0):.2e}")
    check("implicit d/dlam matches", rel(gl1, gl0) < 1e-2, f"rel={rel(gl1, gl0):.2e}")
    check("implicit d/dsmaps matches", rel(gs1, gs0) < 1e-2, f"rel={rel(gs1, gs0):.2e}")


def test_accessors_and_mask():
    E, x, y, smaps, mask = make_mri()
    check("get_mask", torch.equal(get_mask(E), mask))
    check("get_sensitivity_map", torch.equal(get_sensitivity_map(E), smaps))
    new = torch.randn_like(smaps)
    E2 = set_sensitivity_map(E, new)
    check("set_sensitivity_map is functional",
          torch.equal(get_sensitivity_map(E2), new)
          and torch.equal(get_sensitivity_map(E), smaps))

    # complex diag adjoint:  <Mx a, b> == <a, (Mx)^H b>
    xm = torch.randn(1, 1, 8, 8, dtype=torch.complex64)
    a = torch.randn(1, 3, 8, 8, dtype=torch.complex64)
    bb = torch.randn(1, 3, 8, 8, dtype=torch.complex64)
    M = Mask(xm)
    lhs = (M(a).conj() * bb).sum()
    rhs = (a.conj() * M.adjoint(bb)).sum()
    check("Mask adjoint is the true adjoint for a complex mask",
          rel(lhs, rhs) < 1e-5, f"rel={rel(lhs, rhs):.2e}")


def test_highpass():
    hp = LearnedHighpass(sigma_g=1.5, sigma_max=3.0)
    check("kernel size follows the 5-sigma rule and is odd",
          hp.ks == 15 and hp.ks % 2 == 1, str(hp.ks))
    k = hp.kernel()
    check("W kernel sums to ~0 (it kills DC)", abs(k.sum().item()) < 1e-5,
          f"sum={k.sum().item():.2e}")

    s = torch.randn(2, 4, 16, 16, dtype=torch.complex64)
    Ws = hp(s)
    check("W preserves shape and handles complex", Ws.shape == s.shape and Ws.is_complex())
    quad = (s.conj() * Ws).sum().real
    check("W is PSD: <s, W s> >= 0", quad.item() > 0, f"{quad.item():.3f}")
    # (I - G)^2 kills a constant wherever the full kernel fits inside the image
    const = torch.ones(1, 1, 64, 64)
    m = hp.ks           # stay clear of the zero-padded border
    check("W annihilates a constant field (interior)",
          hp(const)[..., m:-m, m:-m].abs().max().item() < 1e-5,
          f"max={hp(const)[..., m:-m, m:-m].abs().max().item():.2e}")


def test_highpass_separability():
    """The separable path must be EXACTLY the 2-D kernel, not an approximation."""
    import torch.nn.functional as Fn
    hp = LearnedHighpass(sigma_g=1.5, sigma_max=3.0)

    def two_d(t):
        return Fn.conv2d(t, hp.kernel(), padding=(hp.ks - 1) // 2)

    x = torch.randn(8, 1, 48, 48)
    with torch.no_grad():
        check("separable == explicit 2-D kernel", rel(hp(x), two_d(x)) < 1e-5,
              f"rel={rel(hp(x), two_d(x)):.2e}")

    # the complex path the coil solve actually takes
    xc = torch.randn(1, 4, 48, 48, dtype=torch.complex64)
    with torch.no_grad():
        ref = torch.complex(two_d(xc.real.reshape(-1, 1, 48, 48)),
                            two_d(xc.imag.reshape(-1, 1, 48, 48))).reshape(xc.shape)
        check("separable == 2-D on complex input", rel(hp(xc), ref) < 1e-5)

    # sigma_g must still get the same gradient through the separable path
    grads = []
    for fn in (hp, two_d):
        hp.zero_grad(); hp._k_key = None
        fn(x).pow(2).sum().backward()
        grads.append(hp.sigma_raw.grad.item())
    check("d/dsigma_g matches the 2-D path",
          abs(grads[0] - grads[1]) / abs(grads[1]) < 1e-5,
          f"{grads[0]:.4f} vs {grads[1]:.4f}")


def test_highpass_kernel_cache():
    """Cached across a CG solve, invalidated on a step and on entering grad."""
    hp = LearnedHighpass()
    x = torch.randn(2, 1, 32, 32)
    hp._k_key = None
    with torch.no_grad():
        hp(x); first = hp._k_val
        hp(x)
        check("kernel reused across applies", hp._k_val is first)
        hp.sigma_raw.add_(0.1)              # what optimizer.step() does
        hp(x)
        check("kernel rebuilt after an in-place param update",
              hp._k_val is not first)
        hp(x); nograd = hp._k_val
    withgrad = hp.kernels_1d()
    # a kernel built under no_grad carries no graph -- reusing it inside an
    # enable_grad region would silently drop sigma_g's gradient
    check("kernel rebuilt when entering grad mode",
          withgrad is not nograd and withgrad[0].requires_grad)


def test_smap_update():
    E, x, y, smaps, mask = make_mri(B=1, C=4, N=16)
    upd = LSmapUpdate(cg_maxit=300, cg_tol=1e-12)
    s_prev = smaps + 0.3 * torch.randn_like(smaps)      # perturbed initial maps
    s_new = upd(x, y, E, s_prev)
    check("smap update keeps the map shape", s_new.shape == smaps.shape)
    check("smap update is finite", torch.isfinite(s_new.abs()).all().item())

    # it must solve its own normal equations:
    #   (Ex^H Ex + mu W + gamma I) s = Ex^H y + gamma s_prev
    import torch.nn.functional as Fn
    from models.ladmm import SmapNormalOp, clamp_mu, clamp_stepsize
    mu = clamp_mu(upd.mu_raw).view(1, 1, 1, 1)
    gamma = clamp_stepsize(upd.gamma_raw).view(1, 1, 1, 1)
    Ex = Mask(get_mask(E)) @ FFT2D() @ Mask(x)
    G = SmapNormalOp(Ex, upd.W, mu)
    rhs = Ex.adjoint(y) + gamma * s_prev
    resid = (G(s_new) + gamma * s_new - rhs).abs().max() / rhs.abs().max()
    check("smap solve satisfies its normal equations", resid.item() < 1e-3,
          f"rel residual = {resid.item():.2e}")

    # and it must pull the perturbed maps back toward the data
    d0 = (Mask(get_mask(E))(FFT2D()(x * s_prev)) - y).abs().pow(2).sum()
    d1 = (Mask(get_mask(E))(FFT2D()(x * s_new)) - y).abs().pow(2).sum()
    check("refined maps reduce the data residual", d1 < d0,
          f"{d0.item():.3e} -> {d1.item():.3e}")


def test_ladmm_forward_backward():
    dk = dict(K=[1, [2, 2]], M=4, C=1, P=5, s=1, is_complex=True)
    net = AltSplitCDLNet(admm_iters=2, denoiser_type="mgcdlnet",
                         denoiser_kws=dk, cg_maxit=4, reuse_latent=True)
    E, x, y, smaps, mask = make_mri(B=1, C=4, N=16)
    xh, extras = net(y, E=E, sigma=torch.tensor([0.02]))
    check("LADMM output shape", xh.shape == x.shape, str(tuple(xh.shape)))
    check("LADMM returns the latent + iterates",
          extras["z"] is not None and "x" in extras and "u" in extras)
    xh.abs().sum().backward()
    grads = [p.grad is not None for p in net.parameters()]
    check("LADMM backward reaches most parameters",
          sum(grads) > 0.8 * len(grads), f"{sum(grads)}/{len(grads)}")
    net.project()
    check("LADMM project() runs", True)

    # latent reuse must actually be wired through
    layer = net.net.layers[1]
    check("reuse_latent flag is honoured", layer.reuse_latent)


def test_ladmm_joint_smaps():
    dk = dict(K=2, M=4, C=1, P=5, s=1, is_complex=True)
    net = AltSplitCDLNet(admm_iters=2, denoiser_type="mgcdlnet",
                         denoiser_kws=dk, cg_maxit=4, smap_update=True)
    E, x, y, smaps, mask = make_mri(B=1, C=4, N=16)
    xh, extras = net(y, E=E, sigma=torch.tensor([0.02]))
    check("joint-smap LADMM output shape", xh.shape == x.shape)
    check("joint-smap LADMM returns refined maps",
          extras["smaps"].shape == smaps.shape)
    check("maps actually changed", not torch.allclose(extras["smaps"], smaps))
    xh.abs().sum().backward()
    hp = net.net.layers[0].smap.W.sigma_raw
    check("the learned high-pass width gets a gradient",
          hp.grad is not None and torch.isfinite(hp.grad).all().item())


def test_denoiser_factory():
    d = build_denoiser("mgcdlnet", K=[1, [2, 2]], M=4, C=1, P=5, s=1,
                       is_complex=False)
    out, z = d(torch.randn(1, 1, 16, 16), E=Identity(), sigma=0.1)
    check("factory-built denoiser runs", out.shape == (1, 1, 16, 16))
    out2, z2 = d(torch.randn(1, 1, 16, 16), E=Identity(), sigma=0.1, z0=z)
    check("warm-started denoiser runs", out2.shape == (1, 1, 16, 16))


if __name__ == "__main__":
    for fn in (test_cg, test_implicit_gradient, test_accessors_and_mask,
               test_highpass, test_highpass_separability,
               test_highpass_kernel_cache, test_smap_update, test_ladmm_forward_backward,
               test_ladmm_joint_smaps, test_denoiser_factory):
        print(f"\n--- {fn.__name__} ---")
        fn()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("failed:", FAIL)
    sys.exit(1 if FAIL else 0)
