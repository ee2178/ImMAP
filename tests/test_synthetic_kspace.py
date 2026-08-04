"""
Checks for the synthetic k-space reconstruction path.

`operators/noise.py::mri_awgn` (the port of `genobs(clo::SyntheticMRIReco, ...)`
from Sljiva's `src/closures/mrireco.jl`), the `prepare_measurement` wiring, the
mask-offset knob, and one train step per model in the multigrid fastMRI grid.

Run with:  python -m tests.test_synthetic_kspace
"""

import sys

import torch

from models.circulant_similarity import circulant_similarity_window
from models.ladmm import AltSplitCDLNet
from models.multigrid import MGCDLNet
from operators import FFT2D, Identity, Mask, Sense
from operators.accessors import get_mask, get_sensitivity_map
from operators.noise import awgn, mri_awgn, rss, sample_sigma
from physics.mask import get_mask_cached, make_acc_mask
from preprocessing.kspace import kspace_post_process, kspace_pre_process
from training.common import prepare_measurement

torch.manual_seed(0)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"[{'ok ' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")


def phantom(B=2, C=4, N=32, seed=0, dc=0.0):
    """A coil-combined image plus UNIT-RSS coil maps.

    Unit RSS is what this repo's map generation produces, and `mri_awgn` assumes
    it rather than renormalizing -- so the fixture has to satisfy the assumption
    for the noise-level identity to mean anything.
    """
    g = torch.Generator().manual_seed(seed)
    image = torch.randn(B, 1, N, N, dtype=torch.complex64, generator=g) + dc
    smaps = torch.randn(B, C, N, N, dtype=torch.complex64, generator=g) + 1.0
    smaps = smaps / smaps.abs().pow(2).sum(1, keepdim=True).sqrt()
    return image, smaps


# ---------------------------------------------------------------------------
def test_complex_randn_convention():
    """`sigma * randn` must have TOTAL variance sigma^2 for a complex tensor.

    Everything downstream -- the meaning of sigma in a config, the noise-adaptive
    thresholds, the sigma sweep in scripts/eval_mg_recon.py -- rests on this, and
    it is a convention rather than something the code can enforce.
    """
    z = torch.randn(400000, dtype=torch.complex64)
    check("complex randn has unit total variance",
          abs((z.abs() ** 2).mean().item() - 1.0) < 0.02,
          f"E|z|^2={(z.abs() ** 2).mean().item():.4f}")


def test_sample_sigma():
    for dist in ("uniform", "log", "cosine"):
        s = sample_sigma(8, (0.0, 0.05), dist=dist)
        ok = (s.shape == (8, 1, 1, 1)
              and float(s.min()) >= 0.0
              and float(s.max()) <= 0.05 + 1e-6)
        check(f"sample_sigma({dist}) shape + range", ok,
              f"{tuple(s.shape)} in [{float(s.min()):.4f}, {float(s.max()):.4f}]")

    check("sample_sigma passes a bare number through",
          sample_sigma(4, 0.03) == 0.03)

    s = sample_sigma(16, (0.0, 0.05))
    check("sample_sigma is per-batch-element", s.unique().numel() == 16)

    s3 = sample_sigma(4, (0.0, 0.05), ndim=3)
    check("sample_sigma respects ndim", s3.shape == (4, 1, 1))

    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    check("sample_sigma is reproducible under a generator",
          torch.equal(sample_sigma(4, (0.0, 0.05), generator=g1),
                      sample_sigma(4, (0.0, 0.05), generator=g2)))


def test_awgn_still_works():
    """awgn now routes through sample_sigma; its contract must be unchanged."""
    x = torch.randn(3, 1, 16, 16)
    y, s = awgn(x, (0.1, 0.1))
    check("awgn returns (B,1,1,1) sigma", s.shape == (3, 1, 1, 1))
    check("awgn noise std tracks sigma", abs((y - x).std().item() - 0.1) < 0.02,
          f"std={(y - x).std().item():.4f}")

    y3, s3 = awgn(torch.randn(3, 16, 16), (0.1, 0.1))
    check("awgn handles a 3-D input", s3.shape == (3, 1, 1))


# ---------------------------------------------------------------------------
def test_mri_awgn_is_the_forward_model():
    image, smaps = phantom()
    mask = torch.ones(1, 1, 32, 32)

    y, sigma, smaps_out = mri_awgn(image, mask, smaps, 0.0)

    check("mri_awgn returns multicoil k-space", y.shape == (2, 4, 32, 32),
          str(tuple(y.shape)))
    check("mri_awgn passes the maps through unchanged",
          torch.equal(smaps_out, smaps))

    # At sigma = 0 the simulation IS the encoding operator applied to the image.
    E = Mask(mask) @ FFT2D() @ Sense(smaps)
    err = (y - E(image)).abs().max().item() / E(image).abs().max().item()
    check("mri_awgn(sigma=0) == E(image)", err < 1e-5, f"rel={err:.2e}")

    # The maps arrive already unit-RSS, so E is a genuine adjoint pair and the
    # noise-level identity below holds without any renormalization.
    a = torch.randn(2, 1, 32, 32, dtype=torch.complex64)
    b = torch.randn(2, 4, 32, 32, dtype=torch.complex64)
    gap = ((E(a).conj() * b).sum() - (a.conj() * E.adjoint(b)).sum()).abs().item()
    check("E is adjoint-consistent with the given maps", gap < 1e-3,
          f"|<Ea,b> - <a,E^Hb>| = {gap:.2e}")
    check("the given maps are unit-RSS",
          (rss(smaps, eps=0.0) - 1).abs().max().item() < 1e-5)


def test_mri_awgn_noise_level():
    """sigma is the noise std of the coil-combined adjoint, E^H y.

    With unit-RSS maps and a fully sampled mask,
        E^H y = sum_c conj(s_c) (s_c x + n_c) = x + sum_c conj(s_c) n_c,
    and the residual has variance sigma^2 sum_c |s_c|^2 = sigma^2. That identity
    is the reason mri_awgn renormalizes the maps.
    """
    image, smaps = phantom(B=1, C=8, N=64)
    mask = torch.ones(1, 1, 64, 64)

    for want in (0.0, 0.01, 0.05):
        y, sigma, _ = mri_awgn(image, mask, smaps, want)
        E = Mask(mask) @ FFT2D() @ Sense(smaps)
        got = (E.adjoint(y) - image).std().item()
        check(f"adjoint noise std tracks sigma={want}", abs(got - want) < 0.004,
              f"got={got:.5f}")


def test_mri_awgn_determinism():
    image, smaps = phantom()
    mask = torch.ones(1, 1, 32, 32)
    g1 = torch.Generator().manual_seed(11)
    g2 = torch.Generator().manual_seed(11)
    y1, s1, _ = mri_awgn(image, mask, smaps, (0.0, 0.05), generator=g1)
    y2, s2, _ = mri_awgn(image, mask, smaps, (0.0, 0.05), generator=g2)
    check("mri_awgn is reproducible under a generator",
          torch.equal(y1, y2) and torch.equal(s1, s2))


def test_prepare_measurement():
    image, smaps = phantom()
    mask = make_acc_mask((32, 32), accel=4, acs_lines=8)
    kspace = torch.zeros(2, 4, 32, 32, dtype=torch.complex64)   # unused when simulated

    y, sigma, extra = prepare_measurement(
        image=image, kspace=kspace, mask=mask, smaps=smaps,
        kspace_type="simulated", noise_std=[0.0, 0.05],
        noise_dist="uniform", whiten_kspace=False,
    )

    check("prepare_measurement(simulated) shapes",
          y.shape == (2, 4, 32, 32) and sigma.shape == (2, 1, 1, 1),
          f"y={tuple(y.shape)} sigma={tuple(sigma.shape)}")
    check("prepare_measurement sigma in [0, 0.05]",
          float(sigma.min()) >= 0.0 and float(sigma.max()) <= 0.05 + 1e-6)
    check("prepare_measurement hands back the maps y is consistent with",
          torch.equal(extra["smaps"], smaps))
    check("prepare_measurement respects the mask",
          torch.equal(y, mask * y))

    # A scalar sigma from the measurement branch must still arrive as (B,1,1,1),
    # which is the only shape MGCDLNet(resize_noise=True) recognises as flat.
    _, sigma_m, extra_m = prepare_measurement(
        image=image, kspace=torch.randn_like(image).expand(-1, 4, -1, -1).contiguous(),
        mask=mask, smaps=smaps, kspace_type="measurement",
        noise_std=[0.0, 0.05], noise_dist="uniform", whiten_kspace=False,
    )
    check("prepare_measurement(measurement) sigma is (B,1,1,1)",
          sigma_m.shape == (2, 1, 1, 1))
    check("prepare_measurement(measurement) still returns smaps",
          "smaps" in extra_m)


# ---------------------------------------------------------------------------
def test_kspace_preprocess_dc():
    """`y~ = E^H y - mu E^H E 1` must reproduce the true gradient exactly.

    The claim being pinned: writing x = v + mu 1 in the data-fidelity term gives
    grad_v = E^H E v - (E^H y - mu E^H E 1), so that bracket -- not
    `E^H y - mean(E^H y)` -- is what a LISTA layer must be handed.
    """
    image, smaps = phantom(B=1, C=6, N=32, seed=5, dc=0.7 - 0.3j)
    mask = make_acc_mask((32, 32), accel=4, acs_lines=8)
    E = Mask(mask) @ FFT2D() @ Sense(smaps)
    y = E(image)

    y_tilde, E_p, (mu, pad) = kspace_pre_process(y, E, stride=8)
    ones = torch.ones_like(image)

    # -- mu is the exact minimizer of ||y - E(mu 1)||^2 ----------------------
    def resid(m):
        return (y - E(m * ones)).abs().pow(2).sum().item()

    m0 = mu.reshape(())
    around = [resid(m0 * (1 + d)) for d in (-0.05, -0.01, 0.0, 0.01, 0.05)]
    check("mu minimizes ||y - E(mu 1)||^2", around[2] == min(around),
          f"mu={complex(m0):.4f}")

    # -- the reformulation is exact ------------------------------------------
    v = torch.randn(1, 1, 32, 32, dtype=torch.complex64)
    g_x = E.gram(v + mu * ones) - E.adjoint(y)     # gradient in x-coordinates
    g_v = E.gram(v) - y_tilde                      # gradient in v-coordinates
    rel = (g_x - g_v).abs().max().item() / g_x.abs().max().item()
    check("E^HE-corrected y~ gives the true gradient", rel < 1e-5,
          f"rel={rel:.2e}")

    # -- and plain mean subtraction does not ---------------------------------
    old = E.adjoint(y) - E.adjoint(y).mean()
    rel_old = ((g_x - (E.gram(v) - old)).abs().max().item()
               / g_x.abs().max().item())
    check("plain mean subtraction gets it wrong", rel_old > 1e-2,
          f"rel={rel_old:.2e} -- the error this replaces")

    # -- mean(E^H y) is not mean(x) once coil maps are involved --------------
    naive = E.adjoint(y).mean()
    check("mu beats mean(E^H y) as an estimate of mean(x)",
          (mu.reshape(()) - image.mean()).abs() < (naive - image.mean()).abs(),
          f"mu err={(mu.reshape(()) - image.mean()).abs():.4f} vs "
          f"mean(E^Hy) err={(naive - image.mean()).abs():.4f}")

    # -- round trip -----------------------------------------------------------
    # A network that perfectly recovered v must post-process back to x.
    back = kspace_post_process(image - mu * ones, (mu, pad))
    check("post(pre(.)) restores the DC",
          (back - image).abs().max().item() < 1e-5)


def test_kspace_preprocess_degenerates():
    """At E = I the correction must collapse to the old mean subtraction."""
    image, _ = phantom(B=2, C=4, N=32, seed=6, dc=0.5)
    y_tilde, E, (mu, pad) = kspace_pre_process(image, Identity(), stride=8)
    check("E=I: mu == mean(x)",
          (mu.reshape(-1) - image.mean(dim=(-2, -1)).reshape(-1)).abs().max() < 1e-6)
    check("E=I: y~ == x - mean(x)",
          (y_tilde - (image - image.mean(dim=(-2, -1), keepdim=True))
           ).abs().max().item() < 1e-6)


def test_kspace_preprocess_pads_the_operator():
    """A non-aligned grid must pad y~ AND E, so the two still agree."""
    image, smaps = phantom(B=1, C=4, N=30, seed=7)
    mask = make_acc_mask((30, 30), accel=3, acs_lines=6)
    E = Mask(mask) @ FFT2D() @ Sense(smaps)

    y_tilde, E_p, (mu, pad) = kspace_pre_process(E(image), E, stride=8)

    check("kspace preproc pads y~ up to the stride",
          y_tilde.shape[-2:] == (32, 32), str(tuple(y_tilde.shape)))
    check("kspace preproc pads the coil maps too",
          get_sensitivity_map(E_p).shape[-2:] == (32, 32),
          str(tuple(get_sensitivity_map(E_p).shape)))
    check("kspace preproc pads the mask too",
          get_mask(E_p).shape[-2:] == (32, 32),
          str(tuple(get_mask(E_p).shape)))
    # The padded operator must accept the padded image -- the failure this
    # replaces was a broadcast error deep inside gram(E, B z).
    check("the padded operator applies to the padded grid",
          E_p.gram(y_tilde).shape == y_tilde.shape)
    check("post-processing unpads back to the input size",
          kspace_post_process(y_tilde, (mu, pad)).shape[-2:] == (30, 30))


# ---------------------------------------------------------------------------
def test_mask_offset():
    m0 = make_acc_mask((32, 32), accel=4, acs_lines=0, offset=0)
    m1 = make_acc_mask((32, 32), accel=4, acs_lines=0, offset=1)
    check("mask offset shifts the sampled lines", not torch.equal(m0, m1))
    check("mask offset preserves the line count",
          float(m0.sum()) == float(m1.sum()),
          f"{float(m0.sum())} vs {float(m1.sum())}")
    check("mask offset wraps modulo accel",
          torch.equal(make_acc_mask((32, 32), 4, acs_lines=0, offset=5), m1))

    smaps = torch.randn(1, 2, 32, 32, dtype=torch.complex64)
    a = get_mask_cached(smaps, R=4, acs_lines=8, mode="uniform")
    b = get_mask_cached(smaps, R=4, acs_lines=8, mode="uniform")
    check("uniform masks are cached (same object)", a is b)

    c = get_mask_cached(smaps, R=4, acs_lines=8, mode="uniform", offset="random")
    d = get_mask_cached(smaps, R=4, acs_lines=8, mode="uniform", offset="random")
    check("offset='random' bypasses the cache", c is not d)


def test_grid_guards():
    """A bad input size must name the problem, not surface as a broadcast error."""
    net = MGCDLNet(K=[1, [2, 2]], M=4, C=1, P=5, s=2, preproc="identity",
                   is_complex=True)
    check("pad_stride is s * 2^(levels-1)", net.pad_stride == 4, str(net.pad_stride))

    try:
        net(torch.randn(1, 1, 30, 30, dtype=torch.complex64), E=Identity(),
            sigma=0.01)
        check("preproc='identity' rejects an unaligned grid", False)
    except ValueError as e:
        check("preproc='identity' rejects an unaligned grid", "pad_stride" in str(e))

    # preproc='image' + a real E: the pad would leave E's mask/maps behind.
    net_i = MGCDLNet(K=[1, [2, 2]], M=4, C=1, P=5, s=2, preproc="image",
                     is_complex=True)
    image, smaps = phantom(B=1, C=2, N=30)
    mask = torch.ones(1, 1, 30, 30)
    E = Mask(mask) @ FFT2D() @ Sense(smaps)
    try:
        net_i(E(image), E=E, sigma=0.01)
        check("preproc='image' rejects a padding E", False)
    except ValueError as e:
        check("preproc='image' rejects a padding E", "kspace" in str(e))

    # ... and is silent for denoising, where there is no operator to fall behind.
    out, _ = net_i(torch.randn(1, 1, 30, 30, dtype=torch.complex64),
                   E=Identity(), sigma=0.01)
    check("preproc='image' still pads for denoising", out.shape == (1, 1, 30, 30))

    # preproc='kspace' handles the same case: it pads the operator too.
    net_k = MGCDLNet(K=[1, [2, 2]], M=4, C=1, P=5, s=2, preproc="kspace",
                     is_complex=True)
    out_k, _ = net_k(E(image), E=E, sigma=0.01)
    check("preproc='kspace' accepts an unaligned grid with a real E",
          out_k.shape == image.shape, str(tuple(out_k.shape)))

    try:
        MGCDLNet(K=[1, [2, 2]], M=4, C=1, P=5, s=2, preproc="nonsense")
        check("an unknown preproc is rejected", False)
    except ValueError:
        check("an unknown preproc is rejected", True)


def test_pidistance_score_mod_reduction():
    """`softmax(pidistance)` must equal `softmax(|q.k| - 1/2||k||^2)`.

    That reduction is what lets FlexAttention carry the phase-invariant
    similarity: a score_mod sees one number per pair, so a similarity is fusible
    iff it is a pointwise function of `q.k` plus per-query / per-key terms. This
    checks the algebra densely, which is possible without a GPU (torch has no
    CPU backward for FlexAttention, so the kernel itself cannot be exercised
    here -- tests/test_init.py compares the two backends numerically on GPU).
    """
    torch.manual_seed(0)
    B, C, H, W, win = 2, 8, 8, 8, 5

    def rowsm(v):
        return torch.softmax(v, dim=2)

    def window_bias(per_key, col, B):
        """Gather a per-key (B, N) vector into window space (B, N, K)."""
        return torch.gather(per_key.unsqueeze(1).expand(-1, col.shape[0], -1), 2,
                            col.unsqueeze(0).expand(B, -1, -1))

    # -- real features: the fusion is exact ----------------------------------
    q, k = torch.randn(B, C, H, W), torch.randn(B, C, H, W)
    ref, col, _ = circulant_similarity_window("pidistance", q, k, win)
    dot, _, _ = circulant_similarity_window("realdot", q, k, win)
    ksq = window_bias((k * k).sum(1).reshape(B, -1), col, B)

    err = (rowsm(ref) - rowsm(dot.abs() - 0.5 * ksq)).abs().max().item()
    check("real: softmax(pidistance) == softmax(|q.k| - ||k||^2/2)",
          err < 1e-5, f"max|diff|={err:.2e}")

    ref_pd, _, _ = circulant_similarity_window("pidot", q, k, win)
    err = (rowsm(ref_pd) - rowsm(dot.abs())).abs().max().item()
    check("real: softmax(pidot) == softmax(|q.k|)", err < 1e-5,
          f"max|diff|={err:.2e}")

    # -- complex features: it is NOT, and the guard must say so --------------
    qc = torch.randn(B, C, H, W, dtype=torch.complex64)
    kc = torch.randn(B, C, H, W, dtype=torch.complex64)
    qs, ks = torch.cat([qc.real, qc.imag], 1), torch.cat([kc.real, kc.imag], 1)

    ref_c, col_c, _ = circulant_similarity_window("pidistance", qc, kc, win)
    dot_s, _, _ = circulant_similarity_window("realdot", qs, ks, win)  # = Re<q,k>
    ksq_c = window_bias(kc.abs().pow(2).sum(1).reshape(B, -1), col_c, B)

    gap = (rowsm(ref_c) - rowsm(dot_s.abs() - 0.5 * ksq_c)).abs().max().item()
    check("complex: |Re<q,k>| does NOT reproduce pidistance", gap > 1e-2,
          f"max|diff|={gap:.2e} -- the modulus needs Im<q,k> as well")

    # ... while the stacking really does preserve Re<q,k>, which is why
    # distance and realdot survive it unchanged.
    d0, _, _ = circulant_similarity_window("realdot", qc, kc, win)
    err = (dot_s - d0).abs().max().item()
    check("complex: [Re;Im] stacking preserves Re<q,k>", err < 1e-4,
          f"max|diff|={err:.2e}")


def test_pidistance_guard():
    """The impossible combination must raise, not silently use Re<q,k>."""
    net = MGCDLNet(K=[1, [2, 2]], M=4, Mh=2, C=1, P=5, s=2, W=5,
                   sim_fun="pidistance", attn_backend="flex",
                   preproc="identity", is_complex=True)
    try:
        net(torch.randn(1, 1, 32, 32, dtype=torch.complex64), E=Identity(),
            sigma=0.01)
        check("complex pidistance + flex is rejected", False)
    except ValueError as e:
        check("complex pidistance + flex is rejected",
              "gather" in str(e) and "distance" in str(e))

    # ... and a REAL model takes the same similarity happily.
    net_r = MGCDLNet(K=[1, [2, 2]], M=4, Mh=2, C=1, P=5, s=2, W=5,
                     sim_fun="pidistance", attn_backend="gather",
                     preproc="identity", is_complex=False)
    out, _ = net_r(torch.randn(1, 1, 32, 32), E=Identity(), sigma=0.01)
    check("real pidistance model runs", out.shape == (1, 1, 32, 32))


def test_attn_backend_attr():
    """train.py compiles the flex kernel off this attribute; GroupCDL has it."""
    net = MGCDLNet(K=[1, [2, 2]], M=4, Mh=2, C=1, P=5, s=2, W=5,
                   attn_backend="gather", preproc="identity")
    check("MGCDLNet exposes attn_backend",
          getattr(net, "attn_backend", None) == "gather")


# ---------------------------------------------------------------------------
def grid_models():
    """A tiny stand-in per model type, shaped like the generated configs.

    preproc='kspace', widen=1 with alpha_conv=False, pidistance + gather, at
    sizes a CPU runner can afford. The two `dual` entries are NOT in the
    generated grid (see scripts/make_mg_recon_configs.py::MODELS) but are kept
    here because the FenchelProx / `y~ - Dz` read-out is live code either way.
    """
    base = dict(M=4, C=1, P=5, s=2, widen=1, degrees=1, alpha_conv=False,
                is_complex=True, preproc="kspace", resize_noise=True)
    group = dict(Mh=2, W=5, dK=2, sim_fun="pidistance",
                 init_strategy="semi_orthogonal", attn_backend="gather")
    vk = [1, [2, 2]]
    return {
        "mgcdlnet":    lambda: MGCDLNet(K=vk, **base),
        "mggroupcdl":  lambda: MGCDLNet(K=vk, **base, **group),
        "mglpds":      lambda: MGCDLNet(K=vk, dual=True, **base),
        "mggrouplpds": lambda: MGCDLNet(K=vk, dual=True, **base, **group),
        "cdlnet":      lambda: MGCDLNet(K=4, **base),
        "groupcdl":    lambda: MGCDLNet(K=4, **base, **group),
        # smap_update=True as in altsplit.yaml -- that path keeps raw k-space
        # and re-forms E^H y per layer, so preproc stays 'identity'.
        "ladmm":       lambda: AltSplitCDLNet(
            admm_iters=2, cg_maxit=3, implicit_cg=False, preproc="identity",
            smap_update=True, denoiser_type="mgcdlnet",
            denoiser_kws=dict(K=[1, [2, 2]], M=4, C=1, P=5, s=2, degrees=1,
                              widen=1, alpha0=1.0, alpha_conv=False,
                              is_complex=True, resize_noise=True)),
    }


def test_grid_models_take_a_step():
    """Each grid model must survive one real forward+backward on synthetic data.

    This is the check that would have caught the dead `mri_awgn` and the
    single-coil shape mismatch: it runs the actual measurement path, not a
    hand-built y.
    """
    image, smaps = phantom(B=1, C=4, N=32, seed=3)
    mask = make_acc_mask((32, 32), accel=4, acs_lines=8)

    y, sigma, extra = prepare_measurement(
        image=image, kspace=torch.zeros_like(smaps), mask=mask, smaps=smaps,
        kspace_type="simulated", noise_std=[0.0, 0.05], noise_dist="uniform",
        whiten_kspace=False,
    )
    E = Mask(mask) @ FFT2D() @ Sense(extra["smaps"])

    for tag, build in grid_models().items():
        try:
            torch.manual_seed(2)
            net = build()
            recon, _ = net(y, E=E, sigma=sigma)

            loss = (recon - image).abs().pow(2).mean()
            loss.backward()

            grads = [p.grad for p in net.parameters() if p.requires_grad]
            got_grad = any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
                           for g in grads)
            net.project()

            ok = (recon.shape == image.shape
                  and torch.isfinite(recon.abs()).all().item()
                  and torch.isfinite(loss).item()
                  and got_grad)
            check(f"grid model '{tag}' trains a step", ok,
                  f"loss={float(loss):.4e} shape={tuple(recon.shape)}")
        except Exception as exc:                                  # noqa: BLE001
            check(f"grid model '{tag}' trains a step", False,
                  f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for fn in (test_complex_randn_convention, test_sample_sigma,
               test_awgn_still_works, test_mri_awgn_is_the_forward_model,
               test_mri_awgn_noise_level, test_mri_awgn_determinism,
               test_prepare_measurement, test_kspace_preprocess_dc,
               test_kspace_preprocess_degenerates,
               test_kspace_preprocess_pads_the_operator,
               test_mask_offset, test_grid_guards,
               test_pidistance_score_mod_reduction, test_pidistance_guard,
               test_attn_backend_attr, test_grid_models_take_a_step):
        print(f"\n--- {fn.__name__} ---")
        fn()

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("failed: " + ", ".join(FAIL))
    sys.exit(1 if FAIL else 0)
