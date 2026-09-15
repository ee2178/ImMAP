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
from models.ml_cdlnet import (MLCDLNet, MLSplitCDLNet, check_iters,
                              level_channels, level_strides, visit_order)
from operators import FFT2D, Identity, Mask, Sense

torch.manual_seed(0)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"[{'ok ' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")


def rel(a, b):
    return (a - b).abs().max().item() / (b.abs().max().item() + 1e-12)


def cold_start_dead(cls, L):
    """`z0=None` means g_L = 0, so `MLCDLNet`'s whole first sweep takes
    `LISTALayer`'s cold-start shortcut and never calls a synthesis -- those B
    filters get no gradient.  Same property as `LISTA` / `MGCDLNet`, whose first
    layer cold-starts identically, and it is what DDP's unused-parameter check
    trips on.  `MLSplitCDLNet` materialises its state at zero, so it has no gap.
    """
    if cls is MLSplitCDLNet:
        return set()
    return {f"sweeps.0.levels.{i}.synthesis.conv_{p}.weight"
            for i in range(L) for p in ("real", "imag")}


def in_last_sweep_tail(cls, name, K, L):
    """Is `name` a parameter of levels 2..L of the FINAL sweep?

    `MLCDLNet` ends its analysis sweep at level L while `readout='level1'` reads
    g_1, so the last sweep's work above level 1 is downstream of the output.
    `MLSplitCDLNet` pairs its sweep with its read-out (see `_check_reachable`)
    and so has NO tail at all -- any dead parameter there is a real regression.

    Asserting the dead set is a SUBSET of this, rather than pinning it exactly,
    keeps the test robust to which sub-parameters happen to be reached while
    still failing loudly if a live level ever goes dead.
    """
    if cls is MLSplitCDLNet:
        return False
    return any(name.startswith(f"sweeps.{K - 1}.levels.{i}.")
               for i in range(1, L))


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
        K = 3
        for L in (1, 2, 3):
            net = cls(K=K, L=L, M=8, C=1, P=3, s=1, widen=2)
            x, z = net(y, E=Identity(), sigma=sigma)
            check(f"{nm} L={L} round-trips the grid", x.shape == y.shape,
                  str(tuple(x.shape)))
            x.abs().sum().backward()
            missing = {n for n, p in net.named_parameters()
                       if p.requires_grad and p.grad is None}
            unexplained = missing - cold_start_dead(cls, L)
            unexplained = {m for m in unexplained
                           if not in_last_sweep_tail(cls, m, K, L)}
            check(f"{nm} L={L} no parameter dies outside the known causes",
                  not unexplained, str(sorted(unexplained)[:3]))

        # A warm start removes the cold-start gap; the readout='level1' tail
        # stays, because it is structural rather than an artefact of g_L = 0.
        net = cls(K=3, L=2, M=8, C=1, P=3, s=1, widen=2)
        z0 = torch.randn(2, net.Mch[2], 16, 16, dtype=torch.complex64)
        x, _ = net(y, E=Identity(), sigma=sigma, z0=z0)
        x.abs().sum().backward()
        missing = {n for n, p in net.named_parameters()
                   if p.requires_grad and p.grad is None}
        check(f"{nm} warm start closes the cold-start gap",
              all(in_last_sweep_tail(cls, m, 3, 2) for m in missing),
              str(sorted(missing)[:3]))

        # readout='cascade' puts every level on the output path: no dead tail --
        # paired with the sweep that ENDS at level L (see _check_reachable).
        kw = {} if cls is MLCDLNet else dict(sweep="ascending")
        net = cls(K=2, L=3, M=6, C=1, P=3, s=1, widen=2, readout="cascade", **kw)
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


def test_split_matches_algorithm(sweep="ascending", iters=None):
    """One outer iteration, re-implemented from the algorithm block.

    Recomputes every synthesis from scratch, so this doubles as the regression
    test for the `Pi` memo in `MLSplitSweep.forward`: if the cache ever hands
    back a stale tensor, this diverges.
    """
    n, L, K = 32, 3, 2
    y = torch.randn(1, 1, n, n, dtype=torch.complex64)
    sigma = torch.full((1, 1, 1, 1), 0.05)
    E = Identity()
    readout = "cascade" if sweep == "ascending" else "level1"
    net = MLSplitCDLNet(K=K, L=L, M=6, C=1, P=3, s=1, widen=2, W=1,
                        preproc="identity", readout=readout, sweep=sweep,
                        iters=iters)

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
            for l, reps in net.sweeps[k].visits:   # primal Gauss-Seidel pass
                for _ in range(reps):
                    g[l] = block(lev, l, g, u)
            for l in range(1, L):                  # dual ascent
                u[l] = u[l] + (g[l] - lev[l].synthesis(g[l + 1]))
        code = g[1] if readout == "level1" else \
            net.sweeps[K - 1].synthesize(g[L], stop=1)
        x_ref = net.D(code)

        x, _ = net(y, E=E, sigma=sigma)

    check(f"split sweep={sweep} iters={iters} matches the algorithm block",
          rel(x, x_ref) < 1e-5, f"rel={rel(x, x_ref):.2e}")


def test_split_all_schedules():
    """Every visit order and depth, against the same from-scratch reference."""
    for sweep, iters in (("ascending", None), ("descending", None),
                         ("symmetric", 2), ("ascending", [3, 2, 1]),
                         ("symmetric", [4, 2, 3])):
        test_split_matches_algorithm(sweep=sweep, iters=iters)


def test_schedule_helpers():
    check("iters=None is one step per level", check_iters(None, 3) == [1, 1, 1])
    check("an int broadcasts", check_iters(2, 3) == [2, 2, 2])
    check("ascending visits 1..L once",
          visit_order("ascending", [1, 1, 1]) == [(1, 1), (2, 1), (3, 1)])
    check("descending visits L..1 once",
          visit_order("descending", [1, 1, 1]) == [(3, 1), (2, 1), (1, 1)])
    check("symmetric is a V-cycle traversal, coarsest visited once",
          visit_order("symmetric", [2, 2, 3]) ==
          [(1, 1), (2, 1), (3, 3), (2, 1), (1, 1)],
          str(visit_order("symmetric", [2, 2, 3])))
    for bad, why in (((None, 3, "symmetric"), "odd iters under symmetric"),
                     (([1, 1], 3, "ascending"), "wrong length"),
                     (((0, 1, 1), 3, "ascending"), "zero iters")):
        try:
            check_iters(*bad)
            check(f"check_iters rejects {why}", False)
        except ValueError:
            check(f"check_iters rejects {why}", True)
    try:
        visit_order("sideways", [1, 1])
        check("visit_order rejects an unknown sweep", False)
    except ValueError:
        check("visit_order rejects an unknown sweep", True)


# ---------------------------------------------------------------------------
def test_ml_ista_iters():
    """nu_l = 1 is Algorithm 2; nu_l > 1 is a warm-started inner solve."""
    y = torch.randn(1, 1, 32, 32, dtype=torch.complex64)
    sigma = torch.full((1, 1, 1, 1), 0.05)
    a = MLCDLNet(K=2, L=3, M=6, C=1, P=3, s=1, widen=2, W=1, preproc="identity")
    b = MLCDLNet(K=2, L=3, M=6, C=1, P=3, s=1, widen=2, W=1, preproc="identity",
                 iters=[2, 2, 2])
    b.load_state_dict(a.state_dict())
    with torch.no_grad():
        xa, _ = a(y, E=Identity(), sigma=sigma)
        xb, _ = b(y, E=Identity(), sigma=sigma)
    check("MLCDLNet iters>1 changes the map", rel(xa, xb) > 1e-4,
          f"rel={rel(xa, xb):.2e}")
    check("MLCDLNet iters>1 adds no parameters",
          sum(p.numel() for p in a.parameters()) ==
          sum(p.numel() for p in b.parameters()))
    with torch.no_grad():
        _, codes = b.forward_codes(y, E=Identity(), sigma=sigma)
    check("MLCDLNet iters>1 still fills every level",
          all(codes[l] is not None for l in range(1, 4)))


def test_reachability_guard():
    """The sweep must end at the level the read-out reads."""
    try:
        MLSplitCDLNet(K=2, L=3, M=6, C=1, P=3, s=1, sweep="ascending")
        check("split refuses ascending + level1", False)
    except ValueError as e:
        check("split refuses ascending + level1",
              "away from the read-out" in str(e), str(e)[:60])
    try:
        MLSplitCDLNet(K=2, L=3, M=6, C=1, P=3, s=1, sweep="descending",
                      readout="cascade")
        check("split refuses descending + cascade (the mirror case)", False)
    except ValueError as e:
        check("split refuses descending + cascade (the mirror case)",
              "away from the read-out" in str(e), str(e)[:60])
    for kw in (dict(K=2, L=3),                                  # descending
               dict(K=2, L=3, sweep="ascending", readout="cascade"),
               dict(K=2, L=3, sweep="symmetric", iters=2),
               dict(K=2, L=3, sweep="symmetric", iters=2, readout="cascade"),
               dict(K=1, L=4, sweep="descending"),
               dict(K=1, L=1, sweep="ascending")):              # L=1 is exempt
        MLSplitCDLNet(M=6, C=1, P=3, s=1, **kw)
        check(f"split accepts {kw}", True)
    # MLCDLNet is exempt: its decode reaches level 1 in one hop
    MLCDLNet(K=2, L=4, M=6, C=1, P=3, s=1)
    check("MLCDLNet is exempt (decode is a one-hop shortcut)", True)

    # and the paired configs really do leave nothing dead
    y = torch.randn(1, 1, 32, 32, dtype=torch.complex64)
    sigma = torch.full((1, 1, 1, 1), 0.05)
    for kw in (dict(sweep="descending"), dict(sweep="symmetric", iters=2),
               dict(sweep="ascending", readout="cascade")):
        net = MLSplitCDLNet(K=3, L=3, M=6, C=1, P=3, s=1, widen=2, **kw)
        x, _ = net(y, E=Identity(), sigma=sigma)
        x.abs().sum().backward()
        missing = [n for n, p in net.named_parameters()
                   if p.requires_grad and p.grad is None]
        check(f"split {kw} leaves no dead parameter", not missing,
              str(missing[:3]))


def test_memo_counts():
    """Pin the synthesis-application counts the `Pi` memo claims to achieve.

    `test_split_all_schedules` already proves the memo returns the RIGHT values
    (it compares against a reference that recomputes everything); this proves it
    actually avoids the work, so the docstring's arithmetic cannot drift.
    """
    y = torch.randn(1, 1, 32, 32, dtype=torch.complex64)
    sigma = torch.full((1, 1, 1, 1), 0.05)
    for L in (2, 3, 4):
        for sweep, iters, want in (("descending", None, 2 * L - 1),
                                   ("ascending", None, 2 * L - 1),
                                   ("symmetric", 2, 3 * L - 1)):
            readout = "cascade" if sweep == "ascending" else "level1"
            net = MLSplitCDLNet(K=1, L=L, M=4, C=1, P=3, s=1, sweep=sweep,
                                iters=iters, readout=readout,
                                preproc="identity")
            n = [0]
            hs = [lev.synthesis.register_forward_hook(
                      lambda *a: n.__setitem__(0, n[0] + 1))
                  for lev in net.sweeps[0].levels]
            with torch.no_grad():
                net(y, E=Identity(), sigma=sigma)
            for h in hs:
                h.remove()
            # the read-out's own cascade synthesis is outside the sweep
            obs = n[0] - (L - 1 if readout == "cascade" else 0)
            check(f"L={L} {sweep}: B applied {want}x per sweep", obs == want,
                  f"observed {obs}")


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
        # symmetric is the only order legal with BOTH read-outs, so it is what
        # lets this compare them with everything else held fixed.
        kw = {} if cls is MLCDLNet else dict(sweep="symmetric", iters=2)
        a = cls(K=2, L=2, M=8, C=1, P=3, s=1, widen=2, W=1, readout="level1", **kw)
        b = cls(K=2, L=2, M=8, C=1, P=3, s=1, widen=2, W=1, readout="cascade", **kw)
        b.load_state_dict(a.state_dict())
        xa, _ = a(y, E=Identity(), sigma=sigma)
        xb, _ = b(y, E=Identity(), sigma=sigma)
        check(f"{nm} the two read-outs differ at finite K", rel(xa, xb) > 1e-3,
              f"rel={rel(xa, xb):.2e}")

        # at L=1 the model has no intermediate code, so they must coincide
        a1 = cls(K=2, L=1, M=8, C=1, P=3, s=1, W=1, readout="level1", **kw)
        b1 = cls(K=2, L=1, M=8, C=1, P=3, s=1, W=1, readout="cascade", **kw)
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
               test_preconditions, test_schedule_helpers,
               test_ml_ista_matches_algorithm, test_ml_ista_iters,
               test_split_matches_algorithm, test_split_all_schedules,
               test_reachability_guard, test_memo_counts,
               test_degeneracies,
               test_constraints, test_readout_and_diagnostics, test_tie_outer):
        print(f"\n--- {fn.__name__} ---")
        fn()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("failed:", FAIL)
    sys.exit(1 if FAIL else 0)
