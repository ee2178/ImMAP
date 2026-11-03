"""
exp21: `cg_init`, RAM's realigned input as the start of MGLPDSNet and VarNet
(models/cg_init.py, solvers/cg.py).

Run with `python -m tests.test_cg_init`. CPU, a few seconds. Checks 1-3 need deepinv
(the RAM baseline's dependency) and are skipped without it.

1. `batched_cg(stop="rhs")` is deepinv 0.4.2's `conjugate_gradient`, iterate for
   iterate, at several iteration caps and at its default tolerance.
2. `CGInit` matches RAM's own `realign_input` on RAM's own physics wrappers (Pad,
   MultiScaleLinearPhysics, the adapter's E), batch of two at different noise levels.
3. The gradient with respect to `fact` matches deepinv's implicit backward.
4. The default stopping rule is unchanged: the old LADMM solve is bit-identical.
5. MGLPDSNet(cg_init=True): starts from that x0 (centred), forms z from it in layer 0,
   keeps the layer count, runs planar, and trains `cg_init.fact`; cg_init=False is the
   net it always was.
6. E2EVarNet(cg_init=True) starts its cascades from F S x0 and trains `fact`; it refuses
   use_smaps=False.
7. The generated exp21 cells build, with names of their own, and change only cg_init.
"""

from __future__ import annotations

import math
import os
import sys

import torch

from models.cg_init import CGInit, ram_gamma
from operators import FFT2D, Mask, Sense
from solvers.cg import batched_cg, tcg

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import make_mg_recon_configs as gen                        # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(f"[{'ok ' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def problem(B=2, C=4, H=48, W=56, seed=0, sigma=(0.01, 0.02)):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, 1, H, W, dtype=torch.complex64, generator=g)
    s = torch.randn(B, C, H, W, dtype=torch.complex64, generator=g)
    s = s / s.abs().pow(2).sum(1, keepdim=True).sqrt()
    cols = (torch.rand(1, 1, 1, W, generator=g) > 0.75).float()
    cols[..., W // 2 - 3: W // 2 + 3] = 1.0
    E = Mask(cols.expand(B, 1, H, W)) @ FFT2D() @ Sense(s)
    sig = torch.tensor(sigma[:B])
    n = torch.randn(B, C, H, W, dtype=torch.complex64, generator=g) * sig.view(-1, 1, 1, 1)
    return E, E.forward(x) + E.ops[0].mask * n, sig


def rel(a, b):
    return float((a - b).abs().pow(2).sum().sqrt() / b.abs().pow(2).sum().sqrt())


def deepinv_checks():
    try:
        from deepinv.optim.linear import conjugate_gradient
        from models.ram import adapter as ad
        from models.ram._upstream import ram as up
    except ModuleNotFoundError as e:
        print(f"[skip] 1-3: {e.name} not installed")
        return

    E, y, sig = problem()
    z = E.adjoint(y)
    mu = 1.0 / ram_gamma(y, sig).view(-1, 1, 1, 1)
    H = lambda v: E.gram(v) + mu * v                          # noqa: E731
    b = z + mu * z
    for it in (1, 3, 7, 50):
        ref = conjugate_gradient(H, b, max_iter=it, tol=1e-4, parallel_dim=[0], init=z)
        got = batched_cg(H, b, x0=z, max_iter=it, tol=1e-4, stop="rhs")
        check(f"1. rhs stop = deepinv CG at max_iter={it}", rel(got, ref) < 1e-6, f"rel {rel(got, ref):.1e}")

    # RAM's own realignment, on the wrappers its forward builds
    s = sig / math.sqrt(2.0)
    phys = up.Pad(ad._make_physics(E, s), (0, 0))
    y_r = torch.stack((y.real, y.imag), dim=1).contiguous()
    x_in = phys.A_adjoint(y_r)
    ms = up.MultiScaleLinearPhysics(phys, x_in.shape[-3:], device=x_in.device)

    class _Fact(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fact_realign = torch.nn.Parameter(torch.tensor([1.3]))

    holder = _Fact()
    ref = up.RAM.realign_input(holder, x_in, ms, y_r)
    ref_c = torch.complex(ref[:, 0:1], ref[:, 1:2])
    init = CGInit()
    with torch.no_grad():
        init.fact.fill_(1.3)
    got = init(y, E, sig)
    check("2. CGInit = RAM.realign_input (batch of 2, two sigmas)", rel(got, ref_c) < 1e-5,
          f"rel {rel(got, ref_c):.1e}")
    check("2. the realignment moved the image (not a no-op)", rel(z, ref_c) > 1e-2,
          f"|x0 - E^H y| / |x0| = {rel(z, ref_c):.2f}")

    w = torch.randn(ref_c.shape, dtype=torch.complex64, generator=torch.Generator().manual_seed(3))
    (ref_c.conj() * w).real.sum().backward()
    (got.conj() * w).real.sum().backward()
    gr, gg = holder.fact_realign.grad.item(), init.fact.grad.item()
    check("3. d/d fact = deepinv's implicit gradient", abs(gg - gr) <= 1e-3 * abs(gr) + 1e-8,
          f"ours {gg:.5e}, deepinv {gr:.5e}")


def ratio_rule_unchanged():
    E, y, _ = problem(seed=5)
    b = E.adjoint(y)
    torch.manual_seed(0)
    a = tcg(E.gram, 0.3, b, max_iter=20, tol=1e-5)
    c = batched_cg(lambda v: 0.3 * v + E.gram(v), b, max_iter=20, tol=1e-5)
    check("4. default stop: tcg = batched_cg as before", torch.equal(a, c))
    lam = torch.tensor(0.3, requires_grad=True)
    bi = b.clone().requires_grad_(True)
    out = tcg(E.gram, lam, bi, implicit=True, max_iter=30, tol=1e-6)
    out.abs().pow(2).sum().backward()
    check("4. implicit tcg still differentiates lam and b",
          lam.grad is not None and bi.grad is not None and bool(torch.isfinite(lam.grad)))


def mglpds_checks():
    from models.mg_lpds import MGLPDSNet
    from preprocessing.kspace import kspace_pre_process

    E, y, sig = problem(B=1, H=48, W=48, sigma=(0.015,))
    kw = dict(K=[2, [2, 2, 2]], M=4, C=1, P=5, s=2, is_complex=True, preproc="kspace",
              lam0=1e-3, degrees=1, alpha0=1.0)
    torch.manual_seed(1)
    plain = MGLPDSNet(**kw)
    torch.manual_seed(1)
    cg = MGLPDSNet(cg_init=True, **kw)
    names = {n for n, _ in cg.named_parameters()} - {n for n, _ in plain.named_parameters()}
    check("5. cg_init adds exactly one parameter", names == {"cg_init.fact"}, str(names))
    sd = {k: v for k, v in cg.state_dict().items() if not k.startswith("cg_init")}
    check("5. the rest of the net is unchanged",
          all(torch.equal(v, plain.state_dict()[k]) for k, v in sd.items()))

    first = cg.net.layers[0].lpdsA.layers[0]
    seen = {}
    h = first.register_forward_hook(lambda m, a, o: seen.update(x=o[0][0].detach()))
    out, _ = cg(y, E=E, sigma=sig)
    h.remove()
    with torch.no_grad():
        x0 = cg.cg_init(y, E, sig)
        _, _, (mu, _) = kspace_pre_process(y, E, cg.pad_stride)
    check("5. layer 0 starts from x0 - mu (cold start keeps x, forms z)",
          torch.allclose(seen["x"], x0 - mu, atol=1e-5))
    with torch.no_grad():
        a = plain(y, E=E, sigma=sig)[0]
    check("5. and the output differs from E^H y's start", rel(out.detach(), a) > 1e-4)

    out.abs().pow(2).sum().backward()
    g = cg.cg_init.fact.grad
    check("5. cg_init.fact gets a gradient", g is not None and bool(torch.isfinite(g)) and g.abs() > 0)

    MGLPDSNet.PLANAR_STATE = True
    try:
        with torch.no_grad():
            p = cg(y, E=E, sigma=sig)[0]
        check("5. planar state gives the same output", torch.allclose(p, out.detach(), atol=1e-4),
              f"max diff {float((p - out.detach()).abs().max()):.1e}")
    finally:
        MGLPDSNet.PLANAR_STATE = False

    torch.manual_seed(1)
    flat = MGLPDSNet(cg_init=True, **dict(kw, K=3))
    with torch.no_grad():
        f = flat(y, E=E, sigma=sig)[0]
    check("5. flat LPDS with cg_init runs", bool(torch.isfinite(f).all()))
    torch.manual_seed(1)
    f23 = MGLPDSNet(cg_init=True, cg_fact_init=23.0, **kw)
    with torch.no_grad():
        x23 = f23.cg_init(y, E, sig)
        init23 = CGInit()
        init23.fact.fill_(23.0)
    check("5. cg_fact_init=23 is CGInit with fact 23", float(f23.cg_init.fact) == 23.0
          and torch.allclose(x23, init23(y, E, sig)))
    try:
        MGLPDSNet(cg_init=True, **dict(kw, preproc="image"))
        check("5. preproc='image' refused", False)
    except ValueError:
        check("5. preproc='image' refused", True)


def varnet_checks():
    from models.e2evarnet.adapter import E2EVarNet
    E, y, sig = problem(B=1, H=48, W=48, sigma=(0.015,))
    torch.manual_seed(2)
    net = E2EVarNet(num_cascades=2, chans=4, pools=2, use_smaps=True, output="sense", cg_init=True)
    seen = {}
    h = net.net.cascades[0].register_forward_pre_hook(lambda m, a: seen.update(k=a[0].detach()))
    out, _ = net(y, E=E, sigma=sig)
    h.remove()
    with torch.no_grad():
        x0 = net.cg_init(y, E, sig)
        k0 = torch.view_as_complex(seen["k"].contiguous())
        back = (E.ops[2].smaps.conj() * FFT2D().adjoint(k0)).sum(1, keepdim=True)
    check("6. VarNet's first cascade sees F S x0", torch.allclose(back, x0, atol=1e-4),
          f"rel {rel(back, x0):.1e}")
    out.abs().pow(2).sum().backward()
    g = net.cg_init.fact.grad
    check("6. VarNet's cg_init.fact gets a gradient", g is not None and bool(torch.isfinite(g)))
    v = E2EVarNet(num_cascades=2, chans=4, pools=2, use_smaps=True, output="sense", cg_init=True,
                  cg_fact_init=23.0)
    check("6. cg_fact_init sets fact's start", float(v.cg_init.fact) == 23.0)
    try:
        CGInit(fact_init=0.0)
        check("6. fact_init <= 0 refused", False)
    except ValueError:
        check("6. fact_init <= 0 refused", True)
    try:
        E2EVarNet(num_cascades=2, chans=4, pools=2, cg_init=True)
        check("6. use_smaps=False refused", False)
    except ValueError:
        check("6. use_smaps=False refused", True)


def generator_checks():
    from models import build_model
    names = set()
    for cell in gen.EXP21_CELLS:
        src = cell.replace("cgcf32", "cf32")
        a, b = gen.MODELS[cell], gen.MODELS[src]
        diff = {k for k in set(a["params"]) | set(b["params"]) if a["params"].get(k) != b["params"].get(k)}
        same_spec = {k: v for k, v in a.items() if k != "params"} == {k: v for k, v in b.items() if k != "params"}
        check(f"7. {cell} = {src} + cg_init", diff == {"cg_init"} and same_spec, str(diff))
        net = build_model({"model": {"type": a["type"], "params": a["params"]}})
        check(f"7. {cell} starts fact at 1, as RAM does", float(net.cg_init.fact) == 1.0)
        check(f"7. {cell} is OPT_IN", cell in gen.OPT_IN)
        names.add(gen._display_name(a["type"], a["params"]) + "_cg")
    check("7. exp21 run names are distinct", len(names) == len(gen.EXP21_CELLS), str(sorted(names)))


def main():
    torch.set_num_threads(min(4, torch.get_num_threads()))
    deepinv_checks()
    ratio_rule_unchanged()
    mglpds_checks()
    varnet_checks()
    generator_checks()
    print("\nALL OK" if not FAIL else f"\n{len(FAIL)} FAILED: {FAIL}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
