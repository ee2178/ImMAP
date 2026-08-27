"""
Image-domain embedding: `E @ Truncate` instead of padding the operator.

Run with `python -m tests.test_embed_truncate`.

The property under test is that the embedding changes the GRID the network
unrolls on without changing the PROBLEM: `E`, the mask, the coil maps and `y`
are all untouched, and `E'^H E' = T^H E^H E T` exactly. Contrast with
`pad_operator`, which resamples the sampling mask onto the larger grid and so
claims phase-encode lines the data was never measured at.
"""

import math

import torch

from operators import FFT2D, Mask, Sense
from operators.truncate import Truncate, embed_operator, embedded_size
from operators.noise import mri_awgn
from physics.mask import make_acc_mask
from preprocessing.kspace import kspace_pre_process

PASS, FAIL, SKIPPED = [], [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def problem(H, W, NC=4, seed=0, R=8, sigma=0.005, device="cpu"):
    torch.manual_seed(seed)
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W),
                            indexing="ij")
    mag = 0.25 + 0.9 * torch.exp(-(xx ** 2 + yy ** 2) * 2.5)
    gt = (mag * torch.exp(1j * 1.5 * xx * yy)).to(torch.complex64)[None, None]
    sm = torch.randn(1, NC, H, W, dtype=torch.complex64) * 0.3 + 1.0
    sm = sm / (sm.abs().pow(2).sum(1, keepdim=True).sqrt() + 1e-8)
    m = make_acc_mask((H, W), R, acs_lines=20)
    while m.dim() < 4:
        m = m.unsqueeze(0)
    E = Mask(m) @ FFT2D() @ Sense(sm)
    y, _, _ = mri_awgn(gt, m, sm, sigma, "uniform")
    return (gt.to(device), E, y.to(device),
            torch.full((1, 1, 1, 1), sigma, device=device))


# ---------------------------------------------------------------------------
def test_sizes():
    print("\n[embedded_size]")
    check("already divisible is a no-op", embedded_size((320, 320), 8) == (320, 320))
    check("rounds up to the next multiple", embedded_size((322, 318), 8) == (328, 320),
          str(embedded_size((322, 318), 8)))
    check("multiple=1 never grows", embedded_size((313, 317), 1) == (313, 317))
    T = Truncate((320, 320), (320, 320))
    check("equal sizes -> identity Truncate", T.is_identity)
    x = torch.randn(1, 1, 320, 320)
    check("identity Truncate is exactly the identity",
          torch.equal(T.forward(x), x) and torch.equal(T.adjoint(x), x))
    try:
        Truncate((60, 60), (64, 64))
        check("shrinking is rejected", False)
    except ValueError:
        check("shrinking is rejected", True)


def test_adjoint_and_gram():
    print("\n[Truncate is an exact adjoint pair]")
    gt, E, y, _ = problem(60, 60)
    E_emb, T = embed_operator(E, (60, 60), 8)
    check("grew to the next multiple of 8", T.big == (64, 64), repr(T))
    check("fused SENSE gram still matched", E_emb._fused_gram is not None)

    torch.manual_seed(3)
    xb = torch.randn(1, 1, 64, 64, dtype=torch.complex64)
    yr = torch.randn_like(y)
    lhs = torch.vdot(E_emb(xb).flatten(), yr.flatten())
    rhs = torch.vdot(xb.flatten(), E_emb.adjoint(yr).flatten())
    rel = float(abs(lhs - rhs) / abs(lhs))
    check("<E'x, y> == <x, E'^H y>", rel < 1e-5, f"relative error {rel:.2e}")

    ref = T.adjoint(E.gram(T.forward(xb)))
    diff = float((E_emb.gram(xb) - ref).abs().max())
    check("gram == T^H (E^H E) T", diff < 1e-4, f"max|diff| {diff:.2e}")

    # crop . zero-pad is the identity; the other order is a projection
    small = torch.randn(1, 1, 60, 60, dtype=torch.complex64)
    check("T . T^H == I on the measured grid",
          torch.allclose(T.forward(T.adjoint(small)), small))


def test_no_operator_padding():
    print("\n[the operator is never resampled]")
    gt, E, y, _ = problem(60, 60)
    E_emb, T = embed_operator(E, (60, 60), 8)

    yt_ref, _, par_ref = kspace_pre_process(y, E, 1)       # stride 1: no padding
    yt_emb, _, par_emb = kspace_pre_process(y, E_emb, 8)

    check("kspace_pre_process applies no pad", not any(par_emb[1]), str(par_emb[1]))
    check("surrogate is on the embedded grid", tuple(yt_emb.shape[-2:]) == (64, 64))
    check("mu is the unpadded problem's mu",
          torch.allclose(par_ref[0], par_emb[0], atol=1e-6))
    check("y~_embedded == zeropad(y~_reference)",
          torch.allclose(yt_emb, T.adjoint(yt_ref), atol=1e-5))
    outside = float(yt_emb.abs().sum() - T.forward(yt_emb).abs().sum())
    check("surrogate is exactly zero outside the window", abs(outside) < 1e-5,
          f"{outside:.2e}")


def test_network_is_flat_across_sizes():
    print("\n[MGLPDSNet stops caring about divisibility]")
    from models.mg_lpds import MGLPDSNet

    base = dict(M=16, C=1, P=3, s=2, widen=1, degrees=1, lam0=1e-3, tau0=0.5,
                theta0=0.0, alpha0=1.0, is_complex=True, preproc="kspace",
                resize_noise=True)
    torch.manual_seed(1)
    net = MGLPDSNet(K=[2, [2, 2, 2]], **base).eval()
    check("pad_stride is s * 2^(levels-1)", net.pad_stride == 8, str(net.pad_stride))

    def psnr(a, b):
        e = float(((a.abs() - b.abs()) ** 2).mean())
        return 10 * math.log10(float(b.abs().max()) ** 2 / max(e, 1e-20))

    padded, embedded = [], []
    for H in (56, 58, 60, 62, 64):
        gt, E, y, sig = problem(H, H)
        with torch.no_grad():
            padded.append(psnr(net(y, E=E, sigma=sig)[0], gt))
            E_emb, T = embed_operator(E, (H, H), net.pad_stride)
            embedded.append(psnr(T.forward(net(y, E=E_emb, sigma=sig)[0]), gt))
        print(f"        {H}x{H}  %8={H % 8}   operator-padded {padded[-1]:6.2f} dB"
              f"   embedded {embedded[-1]:6.2f} dB")

    sp_pad = max(padded) - min(padded)
    sp_emb = max(embedded) - min(embedded)
    check("operator padding is size-dependent", sp_pad > 5.0, f"spread {sp_pad:.2f} dB")
    check("embedding is flat across sizes", sp_emb < 1.5, f"spread {sp_emb:.2f} dB")
    check("embedding never loses to operator padding",
          all(e >= p - 0.01 for e, p in zip(embedded, padded)))


def test_train_step_end_to_end():
    print("\n[one optimizer step through the real code path]")
    from models.mg_lpds import MGLPDSNet
    from training.recon import _embed

    base = dict(M=16, C=1, P=3, s=2, widen=1, degrees=1, lam0=1e-3, tau0=0.5,
                theta0=0.0, alpha0=1.0, is_complex=True, preproc="kspace",
                resize_noise=True)
    torch.manual_seed(1)
    net = MGLPDSNet(K=[1, [2, 2, 2]], **base)
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)

    gt, E, y, sig = problem(60, 60)
    pad_hw = torch.tensor(embedded_size((60, 60), net.pad_stride)).reshape(1, 2)

    E2, T = _embed(net, E, gt, pad_hw)
    recon, _ = net(y, E=E2, sigma=sig)
    check("network output is on the embedded grid", tuple(recon.shape[-2:]) == (64, 64))
    recon = T.forward(recon)
    check("cropped output matches the ground-truth grid", recon.shape == gt.shape)

    loss = (recon.abs() - gt.abs()).pow(2).mean()
    loss.backward()
    live = sum(1 for q in net.parameters()
               if q.grad is not None and q.grad.abs().max() > 0)

    # Compare against the SAME network on an already-divisible grid rather than
    # against the parameter count: some tensors are structurally dead whatever
    # the grid (the cold-start layer never uses its synthesis / tau / theta,
    # and the last post-smoother's dual path has no route to x_hat). The
    # question is whether the embedding costs any reach, not whether every
    # tensor is live.
    torch.manual_seed(1)
    ref = MGLPDSNet(K=[1, [2, 2, 2]], **base)
    gt2, E3, y3, sig3 = problem(64, 64)
    r2, _ = ref(y3, E=E3, sigma=sig3)
    (r2.abs() - gt2.abs()).pow(2).mean().backward()
    live_ref = sum(1 for q in ref.parameters()
                   if q.grad is not None and q.grad.abs().max() > 0)
    check("embedding costs no gradient reach", live >= live_ref,
          f"embedded {live} vs unembedded {live_ref} of "
          f"{sum(1 for _ in net.parameters())} tensors")
    opt.step()
    check("optimizer step is finite",
          all(torch.isfinite(p).all() for p in net.parameters()))

    # the contract check must fire rather than silently re-pad the operator
    try:
        _embed(net, E, gt, torch.tensor([[62, 62]]))
        check("a pad_multiple below pad_stride is rejected", False)
    except ValueError as e:
        check("a pad_multiple below pad_stride is rejected", "pad_stride" in str(e))


def test_real_fastmri():
    """The same properties, on a real slice -- plus what only real data shows."""
    print("\n[real fastMRI data]")
    from tests.fastmri_fixture import (adjoint_tol, banner, load_slice,
                                       survey_sizes)
    from models.mg_lpds import MGLPDSNet
    from operators.noise import mri_awgn
    from physics.mask import make_acc_mask
    from training.recon import _embed

    sample = load_slice(anatomy="brain", pad_multiple=8)
    print(banner(sample))
    if sample is None:
        SKIPPED.append("real fastMRI data")
        return

    image, smaps = sample["image"], sample["smaps"]
    H, W = image.shape[-2:]
    pad_hw = sample["pad_hw"]
    big = (int(pad_hw[0, 0]), int(pad_hw[0, 1]))

    check("dataset's pad_hw matches embedded_size",
          big == embedded_size((H, W), 8), f"{big} vs {embedded_size((H, W), 8)}")
    check("embedded grid divides by 8", big[0] % 8 == 0 and big[1] % 8 == 0, str(big))
    if big == (H, W):
        print("        NOTE: this volume already divides by 8, so Truncate is the "
              "identity here.\n              The padded path is covered by the "
              "synthetic tests above.")

    # mri_awgn's sigma only means "noise std of the coil-combined adjoint" if
    # the maps are unit-RSS. Synthetic maps are normalised by construction;
    # these came off the preprocessing pipeline.
    rss = smaps.abs().pow(2).sum(1).sqrt()
    sup = rss[rss > 1e-3]
    check("real coil maps are unit-RSS on their support",
          bool((sup - 1.0).abs().max() < 5e-2),
          f"max deviation {float((sup - 1.0).abs().max()):.3e}")

    # THE PREMISE. Zero-padding the image domain is only reasonable because the
    # anatomy sits inside the FOV. That is a claim about the DATA, so it can
    # only be checked here. Measure the energy in the ring the pad will occupy.
    T_probe = Truncate(big, (H, W))
    if not T_probe.is_identity:
        padded_gt = T_probe.adjoint(image)
        ring = 1.0 - float(T_probe.forward(padded_gt).abs().pow(2).sum()
                           / padded_gt.abs().pow(2).sum().clamp_min(1e-20))
        check("the pad ring is exactly zero by construction", abs(ring) < 1e-12,
              f"{ring:.2e}")
    # ... and how much signal already sits in the outermost rows/cols, which is
    # what would get *cropped* if this were the other direction.
    edge = max(big[0] - H, big[1] - W, 1)
    border = image.clone()
    border[..., edge:-edge, edge:-edge] = 0
    frac = float(border.abs().pow(2).sum() / image.abs().pow(2).sum().clamp_min(1e-20))
    check(f"anatomy is inside the FOV (outer {edge}px holds <1% of energy)",
          frac < 0.01, f"{100 * frac:.3f}%")

    # the operator built from real maps, embedded
    mask = make_acc_mask((H, W), 8, acs_lines=20)
    while mask.dim() < 4:
        mask = mask.unsqueeze(0)
    E = Mask(mask) @ FFT2D() @ Sense(smaps)
    y, _, _ = mri_awgn(image, mask, smaps, 0.005, "uniform")
    E_emb, T = embed_operator(E, (H, W), 8)

    torch.manual_seed(3)
    xb = torch.randn(1, 1, *big, dtype=torch.complex64)
    lhs = torch.vdot(E_emb(xb).flatten(), y.flatten())
    rhs = torch.vdot(xb.flatten(), E_emb.adjoint(y).flatten())
    rel = float(abs(lhs - rhs) / abs(lhs).clamp_min(1e-20))
    # tolerance scales with the accumulation length; see adjoint_tol
    tol = adjoint_tol(y.numel())
    check("adjointness holds on real maps", rel < tol,
          f"relative {rel:.2e} vs tol {tol:.2e} ({y.numel():,} terms)")

    ref = T.adjoint(E.gram(T.forward(xb)))
    d = float((E_emb.gram(xb) - ref).abs().max())
    check("gram == T^H (E^H E) T on real maps", d < 1e-3, f"max|diff| {d:.2e}")

    # a real batch through the training path (small dictionary: the point here
    # is the operator and the sizes, not the width of the filter bank)
    base = dict(M=16, C=1, P=3, s=2, widen=1, degrees=1, lam0=1e-3, tau0=0.5,
                theta0=0.0, alpha0=1.0, is_complex=True, preproc="kspace",
                resize_noise=True)
    torch.manual_seed(1)
    net = MGLPDSNet(K=[1, [2, 2, 2]], **base).eval()
    E2, T2 = _embed(net, E, image, pad_hw)
    with torch.no_grad():
        recon, _ = net(y, E=E2, sigma=torch.full((1, 1, 1, 1), 0.005))
    check("model runs on the real embedded grid",
          tuple(recon.shape[-2:]) == big, str(tuple(recon.shape[-2:])))
    recon = T2.forward(recon)
    check("cropped output matches the real image grid", recon.shape == image.shape)
    check("output is finite", bool(torch.isfinite(recon.abs()).all()))

    # does the embedding fire on this dataset at all?
    sizes = survey_sizes(anatomy="brain", pad_multiple=8)
    if sizes:
        tot = sum(sizes.values())
        ok8 = sum(v for (h, w), v in sizes.items() if h % 8 == 0 and w % 8 == 0)
        print(f"        size histogram over {tot} volumes: "
              f"{dict(sorted(sizes.items(), key=lambda kv: -kv[1]))}")
        print(f"        {tot - ok8}/{tot} volumes need the embedding "
              f"({100 * (tot - ok8) / max(tot, 1):.1f}%)")
        check("survey read every volume", tot > 0, f"{tot} volumes")


def test_loader_contract():
    print("\n[dataset reports the embedded size]")
    import inspect
    from datasets.fastmri.loader import FastMRIDataset, get_fastmri_loader
    for fn, name in ((FastMRIDataset.__init__, "FastMRIDataset"),
                     (get_fastmri_loader, "get_fastmri_loader")):
        check(f"{name} accepts pad_multiple",
              "pad_multiple" in inspect.signature(fn).parameters)


if __name__ == "__main__":
    test_sizes()
    test_adjoint_and_gram()
    test_no_operator_padding()
    test_network_is_flat_across_sizes()
    test_train_step_end_to_end()
    test_loader_contract()
    test_real_fastmri()
    tail = f", {len(SKIPPED)} section(s) SKIPPED: {', '.join(SKIPPED)}" if SKIPPED else ""
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed{tail}")
    for f in FAIL:
        print(f"  FAILED: {f}")
    raise SystemExit(1 if FAIL else 0)
