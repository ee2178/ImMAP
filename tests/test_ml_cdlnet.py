"""
Sanity tests for the multilevel port (models/ml_cdlnet.py).

The two load-bearing tests are `test_ml_ista_matches_algorithm` and
`test_split_matches_algorithm`: they re-implement one outer iteration straight
from the algorithm blocks, using the network's OWN weights, and compare.  That
is what actually pins down the indexing -- which B synthesises which level,
which code is the target, where E goes, which rho multiplies which arm.
Everything else here is shapes, degeneracies and constraints.

Run with:  python -m tests.test_ml_cdlnet
"""

import sys

import torch

from models.lista import gram
from models.ml_cdlnet import (MLCDLNet, MLSplitCDLNet, level_channels,
                              level_strides)
from operators import FFT2D, Identity, Mask, Sense

torch.manual_seed(0)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"[{'ok ' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")


def rel(a, b):
    return (a - b).abs().max().item() / (b.abs().max().item() + 1e-12)


def dead_params(cls, K, L, warm):
    """Parameters that legitimately receive no gradient, for `readout='level1'`.

    Two separate causes, both structural rather than bugs:

    1. COLD START.  `z0 = None` means g_L = 0, so the whole first sweep takes
       `LISTALayer`'s cold-start shortcut and never calls a synthesis.  Same
       property as `LISTA` / `MGCDLNet`, whose first layer cold-starts
       identically.  A warm start removes it.  `MLSplitCDLNet` materialises its
       state at zero instead of using `None`, so it never has this gap.

    2. THE readout='level1' TAIL.  ML-ISTA updates level 1 FIRST in its
       ascending sweep, so under `D g_1` everything the final sweep does above
       level 1 is downstream of the output -- the analysis and prox of levels
       2..L in sweep K-1 are unreachable no matter how the forward is written.
       `MLSweep`'s `stop=1` avoids paying for them; it cannot make them live.
       `MLSplitCDLNet` has no such tail: its descending half updates level 1
       LAST, so every level feeds g_1 within the same sweep.

    Both vanish under `readout='cascade'`, where g_L is the output path.
    """
    dead = set()
    if cls is not MLSplitCDLNet and not warm:
        dead |= {f"sweeps.0.levels.{i}.synthesis.conv_{p}.weight"
                 for i in range(L) for p in ("real", "imag")}
    if cls is not MLSplitCDLNet and L > 1 and K > 1:
        dead |= {f"sweeps.{K - 1}.levels.{i}.analysis.conv_{p}.weight"
                 for i in range(1, L) for p in ("real", "imag")}
        dead |= {f"sweeps.{K - 1}.levels.{i}.prox.tau.weight"
                 for i in range(1, L)}
    return dead


def mri_operator(B=1, coils=4, n=32):
    smaps = torch.randn(B, coils, n, n, dtype=torch.complex64)
    mask = (torch.rand(B, 1, n, n) > 0.5).to(torch.complex64)
    return Mask(mask) @ FFT2D() @ Sense(smaps)


# ---------------------------------------------------------------------------
def test_schedule():
    check("M_0 = C, M_l = M widen^(l-1)",
          level_channels(2, 8, 3, 2) == [2, 8, 16, 32],
          str(level_channels(2, 8, 3, 2)))
    check("widen=1 keeps the width flat",
          level_channels(1, 8, 3, 1) == [1, 8, 8, 8])
    check("level 1 carries s, deeper levels halve",
          level_strides(2, 3) == [None, 2, 2, 2] and
          level_strides(1, 3) == [None, 1, 2, 2],
          str(level_strides(1, 3)))


# ---------------------------------------------------------------------------
def test_shapes_and_grads():
    y = torch.randn(2, 1, 32, 32, dtype=torch.complex64)
    sigma = torch.full((2, 1, 1, 1), 0.05)

    for cls in (MLCDLNet, MLSplitCDLNet):
        nm = cls.__name__
        for L in (1, 2, 3):
            net = cls(K=2, L=L, M=8, C=1, P=3, s=1, widen=2)
            x, z = net(y, E=Identity(), sigma=sigma)
            check(f"{nm} L={L} round-trips the grid", x.shape == y.shape,
                  str(tuple(x.shape)))
            x.abs().sum().backward()
            missing = [n for n, p in net.named_parameters()
                       if p.requires_grad and p.grad is None]
            check(f"{nm} L={L} gradients reach exactly the live parameters",
                  set(missing) == dead_params(cls, K=2, L=L, warm=False),
                  f"delta={sorted(set(missing) ^ dead_params(cls, 2, L, False))[:3]}")

        # A warm start removes the cold-start gap; the readout='level1' tail
        # stays, because it is structural rather than an artefact of g_L = 0.
        net = cls(K=2, L=2, M=8, C=1, P=3, s=1, widen=2)
        z0 = torch.randn(2, net.Mch[2], 16, 16, dtype=torch.complex64)
        x, _ = net(y, E=Identity(), sigma=sigma, z0=z0)
        x.abs().sum().backward()
        missing = {n for n, p in net.named_parameters()
                   if p.requires_grad and p.grad is None}
        check(f"{nm} warm start closes the cold-start gap",
              missing == dead_params(cls, K=2, L=2, warm=True),
              f"delta={sorted(missing ^ dead_params(cls, 2, 2, True))[:3]}")

        # readout='cascade' puts every level on the output path: no dead tail.
        net = cls(K=2, L=3, M=6, C=1, P=3, s=1, widen=2, readout="cascade")
        z0 = torch.randn(2, net.Mch[3], 8, 8, dtype=torch.complex64)
        x, _ = net(y, E=Identity(), sigma=sigma, z0=z0)
        x.abs().sum().backward()
        missing = [n for n, p in net.named_parameters()
                   if p.requires_grad and p.grad is None]
        check(f"{nm} readout='cascade' + warm start reaches EVERY parameter",
              not missing, str(missing[:4]))

        # strided latent grid: pad_stride = s * 2^(L-1) must still divide
        net = cls(K=1, L=3, M=6, C=1, P=3, s=2, widen=2)
        x, _ = net(y, E=Identity(), sigma=sigma)
        check(f"{nm} s=2, L=3 round-trips the grid", x.shape == y.shape,
              str(tuple(x.shape)))

        # real-valued path
        net = cls(K=2, L=2, M=8, C=1, P=3, s=1, is_complex=False)
        xr, _ = net(torch.randn(2, 1, 32, 32), E=Identity(), sigma=sigma)
        check(f"{nm} real-valued path runs", xr.shape == (2, 1, 32, 32) and
              not torch.is_complex(xr))

        # group prox, widened attention channels per level
        net = cls(K=1, L=2, M=8, C=1, P=3, s=1, widen=2, W=3, Mh=4)
        xg, _ = net(y, E=Identity(), sigma=sigma)
        check(f"{nm} group prox runs at every level", xg.shape == y.shape)


# ---------------------------------------------------------------------------
def test_mri():
    n = 32
    E = mri_operator(B=1, n=n)
    x = torch.randn(1, 1, n, n, dtype=torch.complex64)
    y = E(x)
    sigma = torch.full((1, 1, 1, 1), 0.02)

    for cls in (MLCDLNet, MLSplitCDLNet):
        nm = cls.__name__
        net = cls(K=2, L=2, M=8, C=1, P=3, s=1, widen=2, preproc="kspace")
        xh, _ = net(y, E=E, sigma=sigma)
        check(f"{nm} preproc='kspace' reconstructs on the image grid",
              xh.shape == x.shape, str(tuple(xh.shape)))
        check(f"{nm} kspace output is finite", torch.isfinite(xh).all())

        # preproc='image' pads y~ but not the operator -- must refuse rather
        # than silently multiplying tensors of different extent.  20 is NOT a
        # multiple of pad_stride = s 2^(L-1) = 8, so the pad is non-zero and the
        # 20x20 mask/maps inside E would no longer line up.
        Ebad = mri_operator(B=1, n=20)
        ybad = Ebad(torch.randn(1, 1, 20, 20, dtype=torch.complex64))
        bad = cls(K=1, L=3, M=6, C=1, P=3, s=2, preproc="image")
        try:
            bad(ybad, E=Ebad, sigma=sigma)
            check(f"{nm} preproc='image' + operator + pad raises", False)
        except ValueError as e:
            check(f"{nm} preproc='image' + operator + pad raises",
                  "preproc='kspace'" in str(e), str(e)[:60])


# ---------------------------------------------------------------------------
def test_preconditions():
    y = torch.randn(1, 1, 20, 20, dtype=torch.complex64)
    for cls in (MLCDLNet, MLSplitCDLNet):
        nm = cls.__name__
        net = cls(K=1, L=3, M=6, C=1, P=3, s=2, preproc="identity")
        check(f"{nm} pad_stride = s 2^(L-1)", net.pad_stride == 8,
              str(net.pad_stride))
        try:
            net(y, E=Identity(), sigma=0.05)
            check(f"{nm} indivisible grid raises", False)
        except ValueError as e:
            check(f"{nm} indivisible grid raises", "pad_stride" in str(e))

        net2 = cls(K=1, L=2, M=8, C=1, P=3, s=1)
        try:
            net2(torch.randn(2, 1, 32, 32, dtype=torch.complex64),
                 E=Identity(), sigma=torch.rand(2, 1, 32, 32))
            check(f"{nm} spatial noise map raises", False)
        except ValueError as e:
            check(f"{nm} spatial noise map raises",
                  "noise map" in str(e) and "propagation" in str(e))

        # a (B,1,1,1) per-image level is the supported form and must pass
        net2(torch.randn(2, 1, 32, 32, dtype=torch.complex64),
             E=Identity(), sigma=torch.full((2, 1, 1, 1), 0.05))
        check(f"{nm} (B,1,1,1) per-image sigma is accepted", True)

        try:
            cls(K=0, L=2, M=8, C=1)
            check(f"{nm} K=0 raises", False)
        except ValueError:
            check(f"{nm} K=0 raises", True)


# ---------------------------------------------------------------------------
def test_ml_ista_matches_algorithm():
    """One outer iteration, re-implemented from the ML-ISTA algorithm block."""
    n, L, K = 32, 3, 2
    y = torch.randn(1, 1, n, n, dtype=torch.complex64)
    sigma = torch.full((1, 1, 1, 1), 0.05)
    E = Identity()
    # preproc='identity' so y~ == y and no mean/pad bookkeeping sits between
    # the reference and the network; W=1 so the prox is stateless and the
    # reference does not have to reproduce the group-prox cache.
    net = MLCDLNet(K=K, L=L, M=6, C=1, P=3, s=1, widen=2, W=1,
                   preproc="identity", readout="level1")

    with torch.no_grad():
        g_L, ref = None, None
        for k in range(K):
            lev = net.sweeps[k].levels                    # lev[i] is level i+1
            # synthesis sweep (down):  ghat_l = B_{l+1} ghat_{l+1}
            ghat = [None] * (L + 1)
            ghat[L] = g_L
            for l in range(L - 1, 0, -1):
                ghat[l] = None if ghat[l + 1] is None else \
                    lev[l].synthesis(ghat[l + 1])
            # analysis sweep (up):  g_l = prox(ghat_l - A_l(G_l(B_l ghat_l) - g_{l-1}))
            prev = y
            for l in range(1, L + 1):
                A, B = lev[l - 1].analysis, lev[l - 1].synthesis
                if ghat[l] is None:                       # cold start, g_L = 0
                    u = A(prev)
                else:
                    E_l = E if l == 1 else None
                    u = ghat[l] - A(gram(E_l, B(ghat[l])) - prev)
                prev = lev[l - 1].prox(u, sigma, None)[0]
                if l == 1:
                    g1 = prev
            g_L, ref = prev, g1
        x_ref = net.D(ref)

        x, _ = net(y, E=E, sigma=sigma)

    check("ML-ISTA sweep matches Algorithm 2", rel(x, x_ref) < 1e-5,
          f"rel={rel(x, x_ref):.2e}")


def test_split_matches_algorithm():
    """One outer iteration, re-implemented from the ML-ADMM algorithm block."""
    n, L, K = 32, 3, 2
    y = torch.randn(1, 1, n, n, dtype=torch.complex64)
    sigma = torch.full((1, 1, 1, 1), 0.05)
    E = Identity()
    net = MLSplitCDLNet(K=K, L=L, M=6, C=1, P=3, s=1, widen=2, W=1,
                        preproc="identity", readout="level1")

    with torch.no_grad():
        g, u = net._zeros(y)

        def block(lev, l, g, u):
            layer = lev[l - 1]
            # encoder arm:  A_l ( rho_{l-1} . ( G_l(B_l g_l) - g_{l-1} - u_{l-1} ) )
            E_l = E if l == 1 else None
            resid = gram(E_l, layer.synthesis(g[l])) - (y if l == 1 else g[l - 1])
            if l > 1:
                resid = (resid - u[l - 1]) * lev[l - 2].rho(sigma, ref=None)
            step = layer.analysis(resid)
            # decoder arm:  rho_l . ( g_l - B_{l+1} g_{l+1} + u_l )
            if l < L:
                step = step + layer.rho(sigma, ref=g[l]) * \
                    (g[l] - lev[l].synthesis(g[l + 1]) + u[l])
            return layer.prox(g[l] - layer.mu(sigma, ref=g[l]) * step,
                              sigma, None)[0]

        for k in range(K):
            lev = net.sweeps[k].levels
            for l in range(1, L + 1):             # encoder half-sweep
                g[l] = block(lev, l, g, u)
            for l in range(L - 1, 0, -1):         # decoder half-sweep
                g[l] = block(lev, l, g, u)
            for l in range(1, L):                 # dual ascent
                u[l] = u[l] + (g[l] - lev[l].synthesis(g[l + 1]))
        x_ref = net.D(g[1])

        x, _ = net(y, E=E, sigma=sigma)

    check("ML-ADMM sweep matches the algorithm block", rel(x, x_ref) < 1e-5,
          f"rel={rel(x, x_ref):.2e}")


# ---------------------------------------------------------------------------
def test_degeneracies():
    n = 32
    y = torch.randn(1, 1, n, n, dtype=torch.complex64)
    sigma = torch.full((1, 1, 1, 1), 0.05)
    E = Identity()

    # -- L=1 is plain CDLNet: an unrolled ISTA stack, nothing multilevel ------
    net = MLCDLNet(K=3, L=1, M=8, C=1, P=3, s=1, W=1, preproc="identity")
    with torch.no_grad():
        z = None
        for k in range(3):
            layer = net.sweeps[k].levels[0]
            if z is None:
                z = layer.prox(layer.analysis(y), sigma, None)[0]
            else:
                z = layer.prox(
                    z - layer.analysis(gram(E, layer.synthesis(z)) - y),
                    sigma, None)[0]
        x_ref = net.D(z)
        x, _ = net(y, E=E, sigma=sigma)
    check("MLCDLNet(L=1) is plain CDLNet", rel(x, x_ref) < 1e-5,
          f"rel={rel(x, x_ref):.2e}")

    # -- K=1, cascade read-out is a feed-forward strided CNN (paper Eq. 14) ---
    L = 3
    net = MLCDLNet(K=1, L=L, M=6, C=1, P=3, s=1, widen=2, W=1,
                   preproc="identity", readout="cascade")
    with torch.no_grad():
        lev = net.sweeps[0].levels
        h = y
        for l in range(1, L + 1):                       # encoder
            h = lev[l - 1].prox(lev[l - 1].analysis(h), sigma, None)[0]
        for l in range(L - 1, 0, -1):                   # decoder
            h = lev[l].synthesis(h)
        x_ref = net.D(h)
        x, _ = net(y, E=E, sigma=sigma)
    check("MLCDLNet(K=1, cascade) is a feed-forward CNN", rel(x, x_ref) < 1e-5,
          f"rel={rel(x, x_ref):.2e}")

    # -- rho = 0 decouples the levels ----------------------------------------
    net = MLSplitCDLNet(K=2, L=3, M=6, C=1, P=3, s=1, widen=2, W=1,
                        preproc="identity", readout="level1")
    with torch.no_grad():
        for sweep in net.sweeps:
            for layer in sweep.levels:
                if layer.rho is not None:
                    layer.rho.weight.zero_()
        x0, _ = net(y, E=E, sigma=sigma)
        for sweep in net.sweeps:                       # perturb levels 2..L
            for layer in list(sweep.levels)[1:]:
                for p in layer.parameters():
                    p.add_(torch.randn_like(p))
        x1, _ = net(y, E=E, sigma=sigma)
    check("MLSplitCDLNet with rho=0 depends on level 1 alone",
          rel(x1, x0) < 1e-6, f"rel={rel(x1, x0):.2e}")


# ---------------------------------------------------------------------------
def test_constraints():
    net = MLSplitCDLNet(K=2, L=3, M=6, C=1, P=3, s=1, widen=2)
    with torch.no_grad():
        for sweep in net.sweeps:
            for layer in sweep.levels:
                layer.mu.weight.fill_(10.0)
                if layer.rho is not None:
                    layer.rho.weight.uniform_(-1.0, 3.0)
    net.project()

    ok_rho = all((layer.rho.weight >= 0).all()
                 for sweep in net.sweeps for layer in sweep.levels
                 if layer.rho is not None)
    check("project() clamps rho >= 0", bool(ok_rho))

    worst, bound_ok = 0.0, True
    for sweep in net.sweeps:
        levels = list(sweep.levels)
        for i, layer in enumerate(levels):
            rp = 1.0 if i == 0 else float(levels[i - 1].rho.weight.max())
            rl = 0.0 if layer.rho is None else float(layer.rho.weight.max())
            bound = 1.0 / max(rp + rl, 1e-8)
            worst = max(worst, float(layer.mu.weight.max()) - bound)
            bound_ok &= bool((layer.mu.weight <= bound + 1e-6).all())
    check("project() clamps mu into 1/(max rho_{l-1} + max rho_l)", bound_ok,
          f"worst overshoot={worst:.2e}")

    # filters stay in the unit ball, same contract as every other unrolled net
    norms = [float(layer.analysis.weight.norm(dim=(2, 3)).max())
             for sweep in net.sweeps for layer in sweep.levels]
    check("project() keeps analysis filters in the unit ball",
          max(norms) <= 1.0 + 1e-5, f"max={max(norms):.4f}")


# ---------------------------------------------------------------------------
def test_readout_and_diagnostics():
    n = 32
    y = torch.randn(1, 1, n, n, dtype=torch.complex64)
    sigma = torch.full((1, 1, 1, 1), 0.05)

    for cls in (MLCDLNet, MLSplitCDLNet):
        nm = cls.__name__
        a = cls(K=2, L=2, M=8, C=1, P=3, s=1, widen=2, W=1, readout="level1")
        b = cls(K=2, L=2, M=8, C=1, P=3, s=1, widen=2, W=1, readout="cascade")
        b.load_state_dict(a.state_dict())
        xa, _ = a(y, E=Identity(), sigma=sigma)
        xb, _ = b(y, E=Identity(), sigma=sigma)
        check(f"{nm} the two read-outs differ at finite K", rel(xa, xb) > 1e-3,
              f"rel={rel(xa, xb):.2e}")

        # at L=1 the model has no intermediate code, so they must coincide
        a1 = cls(K=2, L=1, M=8, C=1, P=3, s=1, W=1, readout="level1")
        b1 = cls(K=2, L=1, M=8, C=1, P=3, s=1, W=1, readout="cascade")
        b1.load_state_dict(a1.state_dict())
        x1a, _ = a1(y, E=Identity(), sigma=sigma)
        x1b, _ = b1(y, E=Identity(), sigma=sigma)
        check(f"{nm} read-outs coincide at L=1", rel(x1a, x1b) < 1e-6)

        _, state = a.forward_codes(y, E=Identity(), sigma=sigma)
        r = a.residuals(state)
        check(f"{nm} residuals() gives one entry per constraint", len(r) == 1)
        check(f"{nm} residuals() lives on the level it indexes",
              r[0].shape[1] == a.Mch[1], str(tuple(r[0].shape)))


# ---------------------------------------------------------------------------
def test_tie_outer():
    y = torch.randn(1, 1, 32, 32, dtype=torch.complex64)
    sigma = torch.full((1, 1, 1, 1), 0.05)
    for cls in (MLCDLNet, MLSplitCDLNet):
        nm = cls.__name__
        untied = cls(K=4, L=2, M=8, C=1, P=3, s=1, widen=2, tie_outer=False)
        tied = cls(K=4, L=2, M=8, C=1, P=3, s=1, widen=2, tie_outer=True)
        check(f"{nm} tie_outer keeps one sweep", len(tied.sweeps) == 1 and
              len(untied.sweeps) == 4)
        n_t = sum(p.numel() for p in tied.parameters())
        n_u = sum(p.numel() for p in untied.parameters())
        check(f"{nm} tie_outer is the cheaper arm", n_t < n_u, f"{n_t} < {n_u}")
        x, _ = tied(y, E=Identity(), sigma=sigma)
        check(f"{nm} tie_outer forward runs", x.shape == y.shape)


if __name__ == "__main__":
    for fn in (test_schedule, test_shapes_and_grads, test_mri,
               test_preconditions, test_ml_ista_matches_algorithm,
               test_split_matches_algorithm, test_degeneracies,
               test_constraints, test_readout_and_diagnostics, test_tie_outer):
        print(f"\n--- {fn.__name__} ---")
        fn()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("failed:", FAIL)
    sys.exit(1 if FAIL else 0)
