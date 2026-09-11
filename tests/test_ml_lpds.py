"""
Sanity tests for Multilevel Learned Primal-Dual Splitting (models/ml_lpds.py).

The load-bearing tests:

  * `test_matches_algorithm` re-implements the unrolled net straight from
    Algorithm 2, with the network's OWN (untied, perturbed) weights.  It forms
    both stacked operators the naive O(L^2) way, so it also checks the two
    cascade factorisations the net relies on.
  * `test_L1_is_lpds` copies an `MGLPDSNet(K=int)` into `MLLPDSNet(L=1)` and
    checks they compute the same thing -- the reduction the record claims.
  * `test_cascade_adjoint` checks that at init the up cascade is the exact
    adjoint of the down cascade (Horner), i.e. that `B = A^H` survives the
    strides.
  * `test_condat_vu_converges` iterates one tied layer with tau under
    `step_bound()` and checks it reaches a KKT point of the analysis problem,
    i.e. that before training this really is Condat-Vu.

Everything else is shapes, liveness, constraints and plumbing.

Run with:  python -m tests.test_ml_lpds
"""

import sys

import torch

from models import build_model
from models.base import set_weight
from models.components import CLIP
from models.mg_lpds import MGLPDSNet
from models.ml_lpds import MLLPDSLayer, MLLPDSNet
from operators import FFT2D, Identity, Mask, Sense

torch.manual_seed(0)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"[{'ok ' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")


def rel(a, b, eps=1e-12):
    return ((a - b).abs().max() / (b.abs().max() + eps)).item()


def mri_operator(B=1, C=4, N=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    smaps = torch.randn(B, C, N, N, dtype=torch.complex64, generator=g) + 1.0
    smaps = smaps / smaps.abs().pow(2).sum(1, keepdim=True).sqrt()
    mask = torch.zeros(1, 1, N, N)
    mask[..., ::2] = 1.0
    mask[..., N // 2 - 4:N // 2 + 4] = 1.0
    return Mask(mask) @ FFT2D() @ Sense(smaps)


@torch.no_grad()
def untie_filters(net, scale=0.05):
    """Perturb every A_l and B_l independently, so B_l != A_l^H and every layer
    differs -- otherwise a reference that confused A with B^H, or layer k with
    layer j, would still agree."""
    for m in net.modules():
        if hasattr(m, "conv_real"):
            w = m.weight
            set_weight(m, w + scale * torch.randn_like(w))


def inner(u, v):
    return (u.conj() * v).sum() if torch.is_complex(u) else (u * v).sum()


# ---------------------------------------------------------------------------
def test_shapes():
    y = torch.randn(2, 1, 32, 32, dtype=torch.complex64)
    sigma = torch.full((2, 1, 1, 1), 0.05)
    for L in (1, 2, 3):
        for s in (1, 2):
            net = MLLPDSNet(K=3, L=L, M=6, C=1, P=3, s=s, widen=2,
                            preproc="image")
            x, (xp, z) = net(y, E=Identity(), sigma=sigma)
            check(f"L={L} s={s} round-trips the image grid",
                  x.shape == y.shape, str(tuple(x.shape)))
            want = [(2, net.Mch[l], 32 // (s * 2 ** (l - 1)),
                     32 // (s * 2 ** (l - 1))) for l in range(1, L + 1)]
            got = [tuple(z[l].shape) for l in range(1, L + 1)]
            check(f"L={L} s={s} dual l lives on level l's grid, M_l channels",
                  got == want and z[0] is None, str(got))
            check(f"L={L} s={s} pad_stride = s 2^(L-1)",
                  net.pad_stride == s * 2 ** (L - 1), str(net.pad_stride))

    net = MLLPDSNet(K=3, L=2, M=6, C=1, P=3, s=1, is_complex=False,
                    preproc="image")
    xr, _ = net(torch.randn(2, 1, 32, 32), E=Identity(), sigma=sigma)
    check("real-valued path runs",
          xr.shape == (2, 1, 32, 32) and not torch.is_complex(xr))

    # warm start from a returned pair
    net = MLLPDSNet(K=3, L=2, M=6, C=1, P=3, s=1, preproc="identity")
    x1, state = net(y, E=Identity(), sigma=sigma)
    x2, _ = net(y, E=Identity(), sigma=sigma, state=state)
    check("warm start from (x, z) runs and moves the output",
          x2.shape == x1.shape and rel(x2, x1) > 1e-6)


# ---------------------------------------------------------------------------
def test_cascade_adjoint():
    """<A x, z> = <x, A^H z> for the STACKED operator, at init."""
    for is_complex in (True, False):
        torch.manual_seed(1)
        lay = MLLPDSLayer(C=1, M=4, L=3, widen=2, P=3, s=2,
                          is_complex=is_complex)
        dt = torch.complex64 if is_complex else torch.float32
        x = torch.randn(2, 1, 32, 32, dtype=dt)
        with torch.no_grad():
            t = lay.analyse(x)
            z = [None] + [torch.randn_like(t[l]) for l in range(1, lay.L + 1)]
            lhs = sum(inner(t[l], z[l]) for l in range(1, lay.L + 1))
            rhs = inner(x, lay.adjoint(z))
        err = (abs(lhs - rhs) / abs(lhs)).item()
        check(f"up cascade is the adjoint of the down cascade "
              f"({'complex' if is_complex else 'real'})", err < 1e-4,
              f"rel={err:.2e}")

    # ||A||^2 is at least ||A_1||^2 = 1 and at most sum_l ||A_(1,l)||^2 <= L
    lay = MLLPDSLayer(C=1, M=4, L=3, widen=1, P=3, s=1, is_complex=False)
    n2 = lay.op_norm2(num_iter=300)
    check("1 <= ||A||^2 <= L at init", 0.95 <= n2 <= lay.L + 0.05,
          f"||A||^2={n2:.3f}, L={lay.L}")
    check("step_bound = 1/(1/2 + ||A||^2)",
          abs(lay.step_bound(num_iter=300) - 1 / (0.5 + n2)) < 2e-2)


# ---------------------------------------------------------------------------
def test_matches_algorithm():
    """The whole unrolled net, re-implemented from Algorithm 2."""
    n, L, K = 32, 3, 4
    E = mri_operator(N=n)
    y = E(torch.randn(1, 1, n, n, dtype=torch.complex64))
    sigma = torch.full((1, 1, 1, 1), 0.03)
    # preproc='identity' so y~ = E^H y with no mean/pad bookkeeping between the
    # reference and the net; window=1 so the prox is stateless.
    torch.manual_seed(4)
    net = MLLPDSNet(K=K, L=L, M=4, C=1, P=3, s=1, widen=2, degrees=1,
                    tau0=0.2, theta0=0.6, preproc="identity")
    untie_filters(net)

    with torch.no_grad():
        y_t = E.adjoint(y)

        def A_1l(lay, x, l):                 # A_l ... A_1 x, from scratch
            for j in range(1, l + 1):
                x = lay.levels[j - 1].analysis(x)
            return x

        def B_1l(lay, z, l):                 # B_1 ... B_l z, from scratch
            for j in range(l, 0, -1):
                z = lay.levels[j - 1].synthesis(z)
            return z

        def clip(lay, l, v):
            return lay.levels[l - 1].prox(v, sigma, None)[0]

        # cold start: x = y~, z_l = clip(A_(1,l) y~)
        lay = net.net.layers[0]
        x = y_t
        z = [None] + [clip(lay, l, A_1l(lay, x, l)) for l in range(1, L + 1)]
        for k in range(1, K):
            lay = net.net.layers[k]
            AHz = sum(B_1l(lay, z[l], l) for l in range(1, L + 1))
            tau, theta = lay.tau(sigma, ref=x), lay.theta(sigma, ref=x)
            x_plus = x - tau * (E.gram(x) - y_t + AHz)
            x_bar = x_plus + theta * (x_plus - x)
            z = [None] + [clip(lay, l, z[l] + A_1l(lay, x_bar, l))
                          for l in range(1, L + 1)]
            x = x_plus
        x_ref, z_ref = x, z

        x_net, (_, z_net) = net(y, E=E, sigma=sigma)

    check("net matches Algorithm 2 (primal)", rel(x_net, x_ref) < 1e-5,
          f"rel={rel(x_net, x_ref):.2e}")
    worst = max(rel(z_net[l], z_ref[l]) for l in range(1, L + 1))
    check("net matches Algorithm 2 (every dual)", worst < 1e-5,
          f"worst rel={worst:.2e}")


# ---------------------------------------------------------------------------
def test_L1_is_lpds():
    """`MLLPDSNet(K, L=1)` is `MGLPDSNet(K)` with the level index inserted."""
    kws = dict(M=6, C=1, P=5, s=2, lam0=2e-2, tau0=0.3, theta0=0.5,
               degrees=1, is_complex=True, preproc="kspace")
    K = 5
    torch.manual_seed(5)
    ref = MGLPDSNet(K=K, **kws)
    untie_filters(ref)
    ml = MLLPDSNet(K=K, L=1, **kws)

    mapped = {}
    for key, v in ref.state_dict().items():
        p = key.split(".")                         # net.layers.k.<field>...
        if p[3] in ("analysis", "synthesis", "prox"):
            p = p[:3] + ["levels", "0"] + p[3:]
        mapped[".".join(p)] = v
    try:
        ml.load_state_dict(mapped, strict=True)
        check("L=1 has exactly LPDS's parameters", True)
    except RuntimeError as e:
        check("L=1 has exactly LPDS's parameters", False, str(e)[:200])
        return

    E = mri_operator(N=32)
    y = E(torch.randn(1, 1, 32, 32, dtype=torch.complex64))
    sigma = torch.full((1, 1, 1, 1), 0.015)
    with torch.no_grad():
        xa, (_, za) = ref(y, E=E, sigma=sigma)
        xb, (_, zb) = ml(y, E=E, sigma=sigma)
    check("MLLPDSNet(L=1) == MGLPDSNet(K=int), primal", rel(xb, xa) < 1e-6,
          f"rel={rel(xb, xa):.2e}")
    check("MLLPDSNet(L=1) == MGLPDSNet(K=int), dual", rel(zb[1], za) < 1e-6,
          f"rel={rel(zb[1], za):.2e}")


# ---------------------------------------------------------------------------
def objective(lay, x, y, lam):
    t = lay.analyse(x)
    return 0.5 * (x - y).pow(2).sum() + lam * sum(
        t[l].abs().sum() for l in range(1, lay.L + 1))


def test_condat_vu_converges():
    """Before training, a layer iterated with tau < step_bound is Condat-Vu:
    it must reach a KKT point of  1/2||x - y||^2 + lam sum_l ||A_(1,l) x||_1."""
    torch.manual_seed(2)
    lam = 0.05
    lay = MLLPDSLayer(C=1, M=4, L=3, widen=1, P=3, s=1, lam0=lam,
                      theta0=1.0, is_complex=False)
    tau = 0.9 * lay.step_bound(num_iter=300)
    with torch.no_grad():
        lay.tau.weight.fill_(tau)
        y = torch.randn(1, 1, 32, 32)
        state, cache = lay(None, y, E=Identity())
        for _ in range(3000):
            state, cache = lay(state, y, E=Identity(), cache=cache)
        x, z = state

        # KKT: primal stationarity  x - y + A^H z = 0
        r_x = ((x - y + lay.adjoint(z)).abs().max() / y.abs().max()).item()
        # KKT: z_l in d(lam ||.||_1)(A_(1,l) x)  <=>  z_l = clip(z_l + A_(1,l) x)
        t = lay.analyse(x)
        r_z = max(rel(CLIP(z[l] + t[l], lam), z[l]) for l in range(1, 4))
        F_x, F_y = objective(lay, x, y, lam), objective(lay, y, y, lam)
        # the minimiser beats small perturbations of itself (convex, so global)
        F_pert = min(objective(lay, x + 1e-2 * torch.randn_like(x), y, lam)
                     for _ in range(8))

    check("tau set from step_bound", 0 < tau < 1 / 1.5, f"tau={tau:.3f}")
    check("primal stationarity  x - y + A^H z = 0", r_x < 1e-3,
          f"rel={r_x:.2e}")
    check("dual fixed point  z = clip(z + A x)", r_z < 1e-3, f"rel={r_z:.2e}")
    check("objective decreased from x = y", float(F_x) < float(F_y),
          f"{float(F_y):.3f} -> {float(F_x):.3f}")
    check("no nearby point does better", float(F_x) <= float(F_pert) + 1e-4,
          f"F={float(F_x):.5f}, best perturbed={float(F_pert):.5f}")


# ---------------------------------------------------------------------------
def dead_params(K, L):
    """Parameters that legitimately receive no gradient -- the same two sets
    `LPDSStack` has, one per level:

      * layer 0 is the cold start, which makes no primal step, so its B_l, tau
        and theta are never used;
      * the last layer's dual update feeds nothing the output reads, so its
        A_l and prox are dead -- and so is its theta, which only forms the
        x_bar that update consumes.
    """
    dead = {f"net.layers.0.levels.{i}.synthesis.conv_{p}.weight"
            for i in range(L) for p in ("real", "imag")}
    dead |= {"net.layers.0.tau.weight", "net.layers.0.theta.weight",
             f"net.layers.{K - 1}.theta.weight"}
    dead |= {f"net.layers.{K - 1}.levels.{i}.analysis.conv_{p}.weight"
             for i in range(L) for p in ("real", "imag")}
    dead |= {f"net.layers.{K - 1}.levels.{i}.prox.prox.tau.weight"
             for i in range(L)}
    return dead


def test_every_level_is_live():
    """The record's third consequence: every level's parameters are live in
    every layer.  Contrast MLCDLNet(readout='level1'), whose final sweep has
    a dead tail above level 1."""
    y = torch.randn(2, 1, 32, 32, dtype=torch.complex64)
    sigma = torch.full((2, 1, 1, 1), 0.05)
    for K, L in ((2, 1), (4, 3)):
        net = MLLPDSNet(K=K, L=L, M=6, C=1, P=3, s=1, widen=2,
                        preproc="image")
        x, _ = net(y, E=Identity(), sigma=sigma)
        x.abs().sum().backward()
        missing = {n for n, p in net.named_parameters()
                   if p.requires_grad and p.grad is None}
        check(f"K={K} L={L}: gradients reach exactly the live parameters",
              missing == dead_params(K, L),
              f"delta={sorted(missing ^ dead_params(K, L))[:3]}")


# ---------------------------------------------------------------------------
def test_mri():
    E = mri_operator(N=32)
    image = torch.randn(1, 1, 32, 32, dtype=torch.complex64)
    y = E(image)
    sigma = torch.full((1, 1, 1, 1), 0.015)
    net = MLLPDSNet(K=4, L=3, M=6, C=1, P=5, s=2, widen=2, degrees=1,
                    preproc="kspace")
    x_hat, _ = net(y, E=E, sigma=sigma)
    loss = (x_hat - image).abs().pow(2).mean()
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    check("preproc='kspace' reconstructs on the image grid",
          x_hat.shape == image.shape, str(tuple(x_hat.shape)))
    check("kspace output and gradients are finite",
          bool(torch.isfinite(x_hat).all()) and bool(grads)
          and all(bool(torch.isfinite(g).all()) for g in grads))

    # preproc='image' pads y~ but not the operator -- must refuse.
    Ebad = mri_operator(N=20)
    ybad = Ebad(torch.randn(1, 1, 20, 20, dtype=torch.complex64))
    bad = MLLPDSNet(K=2, L=3, M=6, C=1, P=3, s=2, preproc="image")
    try:
        bad(ybad, E=Ebad, sigma=sigma)
        check("preproc='image' + operator + pad raises", False)
    except ValueError as e:
        check("preproc='image' + operator + pad raises",
              "preproc='kspace'" in str(e), str(e)[:60])


# ---------------------------------------------------------------------------
def test_preconditions():
    y = torch.randn(1, 1, 20, 20, dtype=torch.complex64)
    net = MLLPDSNet(K=2, L=3, M=6, C=1, P=3, s=2, preproc="identity")
    try:
        net(y, E=Identity(), sigma=0.05)
        check("indivisible grid raises", False)
    except ValueError as e:
        check("indivisible grid raises", "pad_stride" in str(e))

    net2 = MLLPDSNet(K=2, L=2, M=6, C=1, P=3, s=1, preproc="image")
    y2 = torch.randn(2, 1, 32, 32, dtype=torch.complex64)
    try:
        net2(y2, E=Identity(), sigma=torch.rand(2, 1, 32, 32))
        check("spatial noise map raises", False)
    except ValueError as e:
        check("spatial noise map raises", "noise map" in str(e))

    for bad in (dict(K=0, L=2), dict(K=2, L=0)):
        try:
            MLLPDSNet(M=4, C=1, P=3, **bad)
            check(f"{bad} raises", False)
        except ValueError:
            check(f"{bad} raises", True)

    lay = MLLPDSLayer(C=1, M=4, L=2, P=3, s=1, spectral_init=False)
    x = torch.randn(1, 1, 16, 16, dtype=torch.complex64)
    state, _ = lay(None, x)
    try:
        lay(state, x, pi=(x, x))
        check("a FAS correction is refused", False)
    except ValueError:
        check("a FAS correction is refused", True)


# ---------------------------------------------------------------------------
def test_constraints():
    net = MLLPDSNet(K=3, L=3, M=6, C=1, P=3, s=1, widen=2, spectral_init=False)
    with torch.no_grad():
        for p in net.parameters():
            p.mul_(20.0).sub_(5.0)
    net.project()
    lays = list(net.net.layers)
    check("project() clamps theta into [0, 1]",
          all(float(l.theta.weight.min()) >= 0 and
              float(l.theta.weight.max()) <= 1 + 1e-6 for l in lays))
    check("project() clamps tau >= 0",
          all(float(l.tau.weight.min()) >= 0 for l in lays))
    check("project() clamps every level's clip threshold >= 0",
          all(float(lev.prox.prox.tau.weight.min()) >= 0
              for l in lays for lev in l.levels))
    norms = [float(m.weight.norm(dim=(2, 3)).max())
             for l in lays for lev in l.levels
             for m in (lev.analysis, lev.synthesis)]
    check("project() keeps every A_l, B_l in the unit ball",
          max(norms) <= 1 + 1e-5, f"max={max(norms):.4f}")


# ---------------------------------------------------------------------------
def test_group_prox():
    y = torch.randn(1, 1, 32, 32, dtype=torch.complex64)
    sigma = torch.full((1, 1, 1, 1), 0.05)
    net = MLLPDSNet(K=3, L=2, M=8, C=1, P=3, s=1, widen=2, window=3, Mh=4,
                    preproc="image")
    x, _ = net(y, E=Identity(), sigma=sigma)
    check("group prox runs at every level", x.shape == y.shape)
    check("attention width widens with the level",
          [lev.prox.prox.Mh for lev in net.net.layers[0].levels] == [4, 8])
    _, cache = net.net(None, y, E=Identity(), sigma=sigma, cache={})
    G1, G2 = cache["level1"].get("Gamma"), cache["level2"].get("Gamma")
    check("each level keeps its own adjacency",
          G1 is not None and G2 is not None and G1 is not G2)


# ---------------------------------------------------------------------------
def test_registry_and_diagnostics():
    base = dict(K=2, L=2, M=4, C=1, P=3, s=1, preproc="image")
    net = build_model({"model": {"type": "MLLPDSNet", "params": base}})
    check("build_model knows MLLPDSNet", isinstance(net, MLLPDSNet))
    for mtype, extra in (("MLGroupLPDS", {}),
                         ("MLLPDSNet", dict(window=3, Mh=4))):
        try:
            build_model({"model": {"type": mtype,
                                   "params": dict(base, **extra)}})
            check(f"{mtype} with {extra or 'no group keys'} is rejected", False)
        except ValueError:
            check(f"{mtype} with {extra or 'no group keys'} is rejected", True)
    g = build_model({"model": {"type": "MLGroupLPDS",
                               "params": dict(base, window=3, Mh=4)}})
    check("build_model knows MLGroupLPDS", isinstance(g, MLLPDSNet))

    y = torch.randn(1, 1, 32, 32, dtype=torch.complex64)
    with torch.no_grad():
        _, (_, z) = net(y, E=Identity(), sigma=0.05)
        parts = net.level_contributions(z)
        total = net.layer(-1).adjoint(z)
    check("level_contributions sum to A^H z",
          len(parts) == net.L and rel(sum(parts), total) < 1e-5,
          f"rel={rel(sum(parts), total):.2e}")


if __name__ == "__main__":
    for fn in (test_shapes, test_cascade_adjoint, test_matches_algorithm,
               test_L1_is_lpds, test_condat_vu_converges,
               test_every_level_is_live, test_mri, test_preconditions,
               test_constraints, test_group_prox,
               test_registry_and_diagnostics):
        print(f"\n--- {fn.__name__} ---")
        try:
            fn()
        except Exception as exc:                                  # noqa: BLE001
            check(fn.__name__, False, f"{type(exc).__name__}: {exc}")
            import traceback
            traceback.print_exc()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("failed: " + ", ".join(FAIL))
    sys.exit(1 if FAIL else 0)
