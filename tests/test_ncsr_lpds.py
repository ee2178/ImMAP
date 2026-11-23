"""Tests of the NCSR-LPDS prototype (models/ncsr_lpds.py).

    python tests/test_ncsr_lpds.py

Reuses tests/test_guided_recon.py's stubs (old local torch, no LAPACK) and its synthetic paired
knee data for the end-to-end check.
"""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_guided_recon as tg            # noqa: E402  (stubs + paired fixtures)

import torch                              # noqa: E402

from models.ncsr_lpds import (NCSRLPDSLayer, NCSRLPDSNet, NCSRAttention,  # noqa: E402
                              joint_adjacency, window_apply, window_offsets)
from operators import FFT2D, Mask, Sense  # noqa: E402

torch.manual_seed(0)
CD = torch.complex128


def crandn(*s, dtype=CD):
    return torch.randn(*s, dtype=dtype)


def rand_G(B=2, h=1, K=9, H=6, W=7):
    P = torch.softmax(torch.randn(B, h, K, H, W, dtype=torch.float64), dim=2)
    return P * torch.sgn(crandn(B, h, K, H, W))


# ---------------------------------------------------------------------------------------------
def test_window_apply_adjoint():
    """<Gamma x, y> = <x, Gamma^H y> for a random complex windowed Gamma."""
    offs = window_offsets(3)
    G = rand_G(K=len(offs))
    x, y = crandn(2, 1, 4, 6, 7), crandn(2, 1, 4, 6, 7)
    lhs = (window_apply(G, x, offs).conj() * y).sum()
    rhs = (x.conj() * window_apply(G, y, offs, adjoint=True)).sum()
    assert torch.allclose(lhs, rhs), (lhs, rhs)


def test_window_apply_matches_dense():
    """Against the dense matrix of the periodic window, on a 1-D-flattened grid."""
    offs = window_offsets(3, include_center=False)
    B, h, c, H, W = 1, 1, 2, 4, 5
    G = rand_G(B, h, len(offs), H, W)
    x = crandn(B, h, c, H, W)
    D = torch.zeros(H * W, H * W, dtype=CD)
    for j, (dy, dx) in enumerate(offs):
        for r in range(H * W):
            i = ((r // W + dy) % H) * W + (r % W + dx) % W
            D[r, i] += G[0, 0, j].reshape(-1)[r]
    want = (D @ x[0, 0].reshape(c, -1).T).T.reshape(c, H, W)
    assert torch.allclose(window_apply(G, x, offs)[0, 0], want)
    wantH = (D.conj().T @ x[0, 0].reshape(c, -1).T).T.reshape(c, H, W)
    assert torch.allclose(window_apply(G, x, offs, adjoint=True)[0, 0], wantH)


def test_window_apply_gradcheck():
    offs = window_offsets(3)
    for adj in (False, True):
        G = rand_G(1, 1, len(offs), 4, 5).requires_grad_()
        x = crandn(1, 1, 2, 4, 5).requires_grad_()
        assert torch.autograd.gradcheck(lambda G, x: window_apply(G, x, offs, adj), (G, x),
                                        eps=1e-6, atol=1e-5), adj


def test_joint_adjacency_is_a_complex_row_stochastic_simplex():
    """|Gamma| rows sum to 1 over the self AND guide windows together; every weight is a
    nonnegative weight times a unit phasor."""
    so, go = window_offsets(3, include_center=False), window_offsets(5)
    q, k, kg = crandn(2, 1, 4, 6, 6), crandn(2, 1, 4, 6, 6), crandn(2, 1, 4, 6, 6)
    Gs, Gg = joint_adjacency(q, k, [kg], so, go)
    tot = Gs.abs().sum(2) + Gg.abs().sum(2)
    assert torch.allclose(tot, torch.ones_like(tot)), tot


def test_transport_makes_the_guide_phase_irrelevant():
    """A phase on the guide's keys AND values (here: one per image) leaves Gamma_zw v unchanged:
    the matched value is turned into the query's phase before it is averaged."""
    so, go = window_offsets(3, include_center=False), window_offsets(3)
    q, k, kg, v = (crandn(1, 1, 4, 5, 5) for _ in range(4))
    ph = torch.exp(1j * torch.tensor(1.234, dtype=torch.float64))
    _, G1 = joint_adjacency(q, k, [kg], so, go)
    _, G2 = joint_adjacency(q, k, [kg * ph], so, go)
    a, b = window_apply(G1, v, go), window_apply(G2, v * ph, go)
    assert torch.allclose(a, b), (a - b).abs().max()


def test_window_dot_gradcheck():
    import models.ncsr_lpds as m
    x = torch.randn(1, 20, 3, dtype=torch.float64, requires_grad=True)
    y = torch.randn(1, 20, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x, y: m._WindowDot.apply(x, y, 4, 5, 3), (x, y))


def test_leave_one_out_centre_has_zero_weight():
    net = _small_net()
    u = crandn(1, 8, 4, 4, dtype=torch.complex64)
    Gs, Gg = net.attn(u, [crandn(1, 8, 4, 4, dtype=torch.complex64)])
    c = len(net.attn.self_offsets) // 2
    assert net.attn.self_offsets[c] == (0, 0) and float(Gs[:, :, c].abs().max()) == 0.0
    tot = Gs.abs().sum(2) + Gg.abs().sum(2)
    assert torch.allclose(tot, torch.ones_like(tot), atol=1e-5)


def test_fused_path_matches_roll_path():
    """FUSED=True routes Gamma's build and every apply through models/circulant_window.py (its
    torch reference on the CPU: the same code path the triton kernels replace on CUDA); outputs
    and every gradient must match the roll path. 9 code channels -> 18 real: crosses no chunk;
    the chunking is checked separately with _CHUNK = 4."""
    import models.ncsr_lpds as m
    y = crandn(2, 1, 12, 12, dtype=torch.complex64)
    w = crandn(2, 1, 12, 12, dtype=torch.complex64)
    res = {}
    for fused, chunk in ((False, 128), (True, 128), (True, 4)):
        m.FUSED, m._CHUNK = fused, chunk
        try:
            net = _small_net(M=9, Mh=4, window=3, guide_window=5, separate_guide_w=True)
            x, _ = net(y, guide=w)
            x.abs().pow(2).sum().backward()
            res[(fused, chunk)] = (x.detach(), {n: p.grad.clone() for n, p in
                                                net.named_parameters() if p.grad is not None})
        finally:
            m.FUSED, m._CHUNK = "auto", 128
    x0, g0 = res[(False, 128)]
    for key in ((True, 128), (True, 4)):
        x1, g1 = res[key]
        assert torch.allclose(x0, x1, atol=1e-5), (key, (x0 - x1).abs().max())
        assert g0.keys() == g1.keys()
        for n in g0:
            assert torch.allclose(g0[n], g1[n], rtol=1e-4, atol=1e-5), \
                (key, n, (g0[n] - g1[n]).abs().max())


def _small_net(**kw):
    p = dict(K=3, M=8, C=1, P=3, s=2, Mh=4, window=3, guide_window=3, dK=2,
             spectral_init=False)
    p.update(kw)
    torch.manual_seed(0)
    return NCSRLPDSNet(**p)


def test_beta_zero_is_plain_lpds():
    """beta = 0: no centring, no guide -- the sweep is LPDSLayer's with the elementwise clip."""
    from models.lpds import LPDSLayer
    net = _small_net(beta0=0.0)
    y = crandn(1, 1, 8, 8, dtype=torch.complex64)
    w = crandn(1, 1, 8, 8, dtype=torch.complex64)
    state, cache = None, {}
    ref_state, ref_cache = None, {}
    for layer in net.layers:
        ref = LPDSLayer(1, 8, P=3, stride=2, lam0=1e-3, tau0=0.5, theta0=0.0, is_complex=True)
        ref.load_state_dict({k: v for k, v in layer.state_dict().items()
                             if k.startswith(("analysis", "synthesis", "tau", "theta"))},
                            strict=False)
        ref.prox.prox.tau.weight.data.copy_(layer.lam.weight.data)
        state, cache = layer(state, y, [w], net.attn, cache=cache, dK=net.dK)
        ref_state, ref_cache = ref(ref_state, y, cache=ref_cache)
        for a, b in zip(state, ref_state):
            assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max()


def test_guide_changes_the_output_and_gradients_reach_everything():
    net = _small_net(separate_guide_w=True)
    y = crandn(2, 1, 8, 8, dtype=torch.complex64)
    w = crandn(2, 1, 8, 8, dtype=torch.complex64)
    x1, _ = net(y, guide=w)
    x0, _ = net(y, guide=torch.zeros_like(w) + 1e-3 * crandn(2, 1, 8, 8, dtype=torch.complex64))
    assert (x1 - x0).abs().max() > 1e-4
    x1.abs().pow(2).sum().backward()
    a = net.attn
    for name, p in [("Wtheta", a.Wtheta.conv_imag.weight), ("Wphi", a.Wphi.conv_imag.weight),
                    ("Wphi_g", a.Wphi_g.conv_imag.weight), ("rho", a.rho.weight),
                    ("beta", net.layers[1].beta.weight), ("lam", net.layers[1].lam.weight),
                    ("A_g", net.layers[1].guide_analysis.conv_real.weight)]:
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, \
            name


def test_complex_w_semi_orthogonal_and_untied_guide_starts_tied():
    """The W's are complex with W W^H = I; separate_guide_w adds W_phi^g only, equal to W_phi at
    init, so the untied net's first forward is the tied net's exactly."""
    tied, untied = _small_net(), _small_net(separate_guide_w=True)
    W = tied.attn.Wtheta.weight.reshape(tied.attn.Mh, tied.M)
    assert W.is_complex() and W.imag.abs().max() > 0
    assert torch.allclose(W @ W.conj().T, torch.eye(tied.attn.Mh, dtype=W.dtype), atol=1e-5)
    assert tied.attn.Wphi_g is None and tied.attn.guide_Wphi is tied.attn.Wphi
    extra = {k for k, _ in untied.named_parameters()} - {k for k, _ in tied.named_parameters()}
    assert extra and all(k.startswith("attn.Wphi_g.") for k in extra), extra
    untied.load_state_dict(tied.state_dict(), strict=False)
    untied.attn.Wphi_g.weight = tied.attn.Wphi.weight      # property setter (set_weight)
    y = crandn(1, 1, 8, 8, dtype=torch.complex64)
    w = crandn(1, 1, 8, 8, dtype=torch.complex64)
    with torch.no_grad():
        assert torch.allclose(tied(y, guide=w)[0], untied(y, guide=w)[0], atol=1e-6)


def test_checkpoint_changes_nothing():
    y = crandn(1, 1, 8, 8, dtype=torch.complex64)
    w = crandn(1, 1, 8, 8, dtype=torch.complex64)
    grads = []
    for ck in (True, False):
        net = _small_net(checkpoint=ck)
        x, _ = net(y, guide=w)
        x.abs().sum().backward()
        grads.append([p.grad.clone() for p in net.parameters() if p.grad is not None])
    for a, b in zip(*grads):
        assert torch.allclose(a, b, atol=1e-6)


def test_constant_gamma_fixed_point_minimises_the_ncsr_objective():
    """With Gamma held fixed (dK > K), tied operators and steps inside the primal-dual condition,
    the sweep is Condat-Vu on F(x) = 1/2||x - y||^2 + sum lam |K x - b|, K = (I - beta Gamma_zz) A,
    b = beta Gamma_zw A_g w (Prop. ncsr): its fixed point is F's minimiser, so no direction lowers
    F there. (Convergence alone is not enough: with Gamma in place of Gamma^H in the primal step
    the iteration still settles, at a point some direction improves -- checked.)"""
    net = _small_net(K=1500, dK=10 ** 6, beta0=0.8, tau0=0.2, theta0=1.0, lam0=0.3,
                     spectral_init=True)
    for layer in net.layers[1:]:
        layer.load_state_dict(net.layers[0].state_dict())
    torch.manual_seed(1)
    y = 3 * crandn(1, 1, 8, 8, dtype=torch.complex64)
    w = 3 * crandn(1, 1, 8, 8, dtype=torch.complex64)
    L, A = net.layers[0], net.attn
    with torch.no_grad():
        guides = net._prep_guides(w, y)
        state, cache = None, {}
        for layer in net.layers:
            state, cache = layer(state, y, guides, A, cache=cache, dK=net.dK)
        Gs, Gg = cache["G"][0], list(cache["G"][1:])
        v = [L.guide_analysis(g) for g in guides]
        beta, lam = L.beta(), L.lam()

        def F(x):
            r = L._K(L.analysis(x), Gs, A, beta) - L._offset(Gg, v, A, beta)
            return 0.5 * (x - y).abs().pow(2).sum() + (lam * r.abs()).sum()

        x = state[0]
        f0 = F(x)
        worst = min(float(F(x + 1e-2 * torch.randn_like(x)) - f0) for _ in range(200))
    assert worst > -1e-4, worst


def test_train_recon_end_to_end_on_paired_knee():
    """training/recon.py hands the net its image guide; one epoch trains and validates."""
    import physics.online_smaps as osm
    with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as d:
        csv, _ = tg.make_data(root)
        real = osm.online_smaps
        osm.online_smaps = lambda y, mask, **kw: torch.full_like(y, tg.C ** -0.5)
        try:
            net = _small_net(K=3, M=8)
            spy = tg.Spy(net)
            wb = tg.run_recon(spy, csv, root, d, online_smaps="espirit")
        finally:
            osm.online_smaps = real
        assert spy.guides and all(g == (1, 1, tg.H, tg.W) for g in spy.guides), spy.guides
        assert torch.isfinite(torch.as_tensor(wb.last("val/psnr")))


def test_build_model_and_generator_cell():
    from models import build_model
    sys.path.insert(0, os.path.join(tg.REPO, "scripts"))
    import make_guided_recon_configs as g
    spec = g.MODELS["ncsr_lpds"]["model"]
    net = build_model({"model": dict(spec, params=dict(spec["params"], K=2, M=8, Mh=4,
                                                       spectral_init=False))})
    assert isinstance(net, NCSRLPDSNet) and net.guided and net.attn.separate_guide_w


def main():
    tests = sorted((n, f) for n, f in globals().items() if n.startswith("test_") and callable(f))
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok    {name}")
        except Exception as e:                                             # noqa: BLE001
            failed += 1
            import traceback
            traceback.print_exc(limit=6)
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
