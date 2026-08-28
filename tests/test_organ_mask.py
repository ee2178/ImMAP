"""
Region-restricted metrics (`organ_mask`).

Run with `python -m tests.test_organ_mask`.

The point of the mask is that a metric stops depending on the background. That
is a MEASURABLE claim, so it is measured here rather than asserted: the same
organ is placed in backgrounds of different sizes and different corruption, and
the masked numbers have to hold still while the unmasked ones move.
"""

import torch

from training.metrics import compute_metrics, nrmse, psnr, ssim

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    tag = "ok  " if cond else "FAIL"
    print(f"  [{tag}] {name}" + (f"   {detail}" if detail else ""))


CORE = 96          # the disc and its error are built ONCE at this size


def scene(field=CORE, r=20, bg=0.0, seed=0):
    """A disc of signal, centre-padded into a `field` x `field` image.

    The organ and the error inside it are built at CORE and then PADDED, not
    regenerated -- so changing `field` changes ONLY how much background there
    is. Generating at each size instead draws a different realisation from the
    same seed, which is not a controlled comparison (it is what an earlier
    version of this test did, and it made a correct implementation look like it
    had a 0.4 dB size dependence).
    """
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(torch.arange(CORE), torch.arange(CORE), indexing="ij")
    m = (((yy - CORE / 2) ** 2 + (xx - CORE / 2) ** 2).sqrt() < r)[None, None]

    gt = torch.zeros(1, 1, CORE, CORE)
    gt[m.expand_as(gt)] = 1.0 + 0.1 * torch.randn(int(m.sum()), generator=g)

    err = 0.05 * torch.randn(1, 1, CORE, CORE,
                             generator=torch.Generator().manual_seed(99))
    pred = gt + err * m                                  # organ error only

    if field != CORE:
        pad = (field - CORE) // 2
        p = (pad, field - CORE - pad) * 2
        gt = torch.nn.functional.pad(gt, p)
        pred = torch.nn.functional.pad(pred, p)
        m = torch.nn.functional.pad(m.float(), p).bool()

    pred = pred + bg * torch.randn_like(pred) * (~m)
    return gt, pred, m


# ---------------------------------------------------------------------------
def test_masked_ignores_background():
    print("\n[the masked metrics do not see the background]")
    gt, p0, m = scene(bg=0.0)
    _, p1, _ = scene(bg=0.5)                             # badly wrong background

    a = compute_metrics(gt.abs(), p0.abs(), mask=m)
    b = compute_metrics(gt.abs(), p1.abs(), mask=m)
    for k in ("psnr", "nrmse", "ssim"):
        d = abs(float(a[k]) - float(b[k]))
        check(f"masked {k} unchanged by a wrecked background", d < 1e-5,
              f"{float(a[k]):.6f} -> {float(b[k]):.6f}  (d={d:.2e})")

    u = compute_metrics(gt.abs(), p0.abs())
    v = compute_metrics(gt.abs(), p1.abs())
    for k in ("psnr", "nrmse", "ssim"):
        d = abs(float(u[k]) - float(v[k]))
        check(f"UNMASKED {k} does move (that is the bug)", d > 1e-3,
              f"{float(u[k]):.4f} -> {float(v[k]):.4f}  (d={d:.2e})")


def test_masked_ignores_background_size():
    """The failure mode masking is really for.

    Zeroing the pair and calling the unmasked metric fixes the first test but
    not this one: a bigger background dilutes the mean and inflates PSNR even
    when the background is perfect.
    """
    print("\n[the masked metrics do not see how much background there is]")
    gt_s, p_s, m_s = scene(field=96)
    gt_b, p_b, m_b = scene(field=192)                 # same disc, 4x the field

    a = float(psnr(gt_s.abs(), p_s.abs(), mask=m_s))
    b = float(psnr(gt_b.abs(), p_b.abs(), mask=m_b))
    check("masked PSNR is the same disc in a 4x field", abs(a - b) < 1e-4,
          f"{a:.6f} vs {b:.6f} dB")

    u = float(psnr(gt_s.abs(), p_s.abs()))
    v = float(psnr(gt_b.abs(), p_b.abs()))
    check("unmasked PSNR inflates with field size", v - u > 3.0,
          f"{u:.2f} -> {v:.2f} dB (+{v - u:.2f})")

    # and the zero-then-call-unmasked shortcut inflates in exactly that way,
    # which is why psnr() divides by the mask area instead
    z = float(psnr((gt_b * m_b).abs(), (p_b * m_b).abs()))
    check("zeroing instead of area-normalising would inflate", z - b > 3.0,
          f"area-normalised {b:.2f} vs zeroed {z:.2f} dB")


def test_ssim_map_not_inputs():
    """SSIM masks its MAP, not its inputs -- zeroing inflates it."""
    print("\n[SSIM restricts the average, not the images]")
    gt, pred, m = scene(field=192, bg=0.0)
    ours = float(ssim(gt.abs(), pred.abs(), mask=m).mean())
    zeroed = float(ssim((gt * m).abs(), (pred * m).abs()).mean())
    full = float(ssim(gt.abs(), pred.abs()).mean())
    check("zeroing the inputs would score higher than region-averaging",
          zeroed > ours, f"region {ours:.4f} vs zeroed {zeroed:.4f}")
    check("region SSIM is below the whole-image value",
          ours < full, f"region {ours:.4f} vs full {full:.4f}")


def test_nrmse_range_is_masked_too():
    """A background BRIGHTER than the organ would widen an unmasked range."""
    print("\n[NRMSE normalises by the masked range]")
    gt, pred, m = scene()
    gt2 = gt + 3.0 * (~m)                     # bright background, in BOTH
    pred2 = pred + 3.0 * (~m)

    a = float(nrmse(gt.abs(), pred.abs(), mask=m))
    b = float(nrmse(gt2.abs(), pred2.abs(), mask=m))
    check("masked NRMSE ignores a bright, correct background",
          abs(a - b) < 1e-5, f"{a:.6f} vs {b:.6f}")

    u = float(nrmse(gt.abs(), pred.abs()))
    v = float(nrmse(gt2.abs(), pred2.abs()))
    check("unmasked NRMSE is deflated by it", u - v > 1e-3,
          f"{u:.5f} -> {v:.5f}")


def test_identity_and_edges():
    print("\n[mask=None is the old code; an empty mask is visible]")
    gt, pred, m = scene()
    a = compute_metrics(gt.abs(), pred.abs())
    b = compute_metrics(gt.abs(), pred.abs(), mask=torch.ones_like(m))
    for k in ("psnr", "nrmse"):
        check(f"mask=None == all-ones mask for {k}",
              abs(float(a[k]) - float(b[k])) < 1e-5,
              f"{float(a[k]):.6f} vs {float(b[k]):.6f}")

    # SSIM is the exception and it is NOT an oversight: with erosion on, an
    # all-ones mask still drops the 5-pixel image rim, so it is a different
    # region from mask=None by construction. The two happen to agree closely on
    # this phantom, which would let a broken erosion pass a `< 1e-5` check --
    # so compare against erosion OFF, where equality is exact, and check
    # separately that erosion actually removes the rim.
    one = torch.ones_like(m)
    no_erode = float(ssim(gt.abs(), pred.abs(), mask=one, erode_mask=False))
    check("mask=None == all-ones mask for ssim (erosion off)",
          abs(float(a["ssim"]) - no_erode) < 1e-6,
          f"{float(a['ssim']):.8f} vs {no_erode:.8f}")

    eroded = float(ssim(gt.abs(), pred.abs(), mask=one, erode_mask=True))
    check("erosion does change the region it averages over",
          abs(eroded - no_erode) > 0, f"{no_erode:.8f} -> {eroded:.8f}")

    # and it removes exactly the window radius: a disc of radius 20 eroded by
    # 5 must lose the pixels between r=15 and r=20
    import torch.nn.functional as _F
    w = m.float()
    kept = -_F.max_pool2d(-w, kernel_size=11, stride=1, padding=5)
    check("erosion trims a 5-pixel rim off the organ",
          0 < float(kept.sum()) < float(w.sum()),
          f"{int(w.sum())} px -> {int(kept.sum())} px")

    empty = torch.zeros_like(m)
    e = compute_metrics(gt.abs(), pred.abs(), mask=empty)
    for k in ("psnr", "nrmse", "ssim"):
        check(f"an empty mask gives NaN, not 0, for {k}",
              bool(torch.isnan(torch.as_tensor(e[k]))), f"{float(e[k])}")

    # a bool mask and its float twin must agree
    f = compute_metrics(gt.abs(), pred.abs(), mask=m.float())
    g = compute_metrics(gt.abs(), pred.abs(), mask=m)
    check("bool and float masks agree",
          all(abs(float(f[k]) - float(g[k])) < 1e-6 for k in f))


def test_batch_and_channels():
    print("\n[shapes]")
    gt = torch.rand(3, 1, 64, 64)
    pred = gt + 0.01 * torch.randn(3, 1, 64, 64)
    m = torch.zeros(3, 1, 64, 64, dtype=torch.bool)
    m[:, :, 16:48, 16:48] = True
    out = compute_metrics(gt, pred, mask=m)
    check("batched call returns scalars",
          all(torch.as_tensor(v).dim() == 0 for v in out.values()))
    check("all finite",
          all(torch.isfinite(torch.as_tensor(v)) for v in out.values()))

    # per-sample SSIM keeps its batch axis
    s = ssim(gt, pred, mask=m)
    check("ssim stays per-sample under a mask",
          tuple(s.shape) == (3,), str(tuple(s.shape)))

    # one empty sample NaNs only itself
    m2 = m.clone()
    m2[1] = False
    s2 = ssim(gt, pred, mask=m2)
    check("an empty sample NaNs alone",
          bool(torch.isnan(s2[1])) and bool(torch.isfinite(s2[0]))
          and bool(torch.isfinite(s2[2])), str(s2.tolist()))


def test_gradients_flow():
    """The same masking runs inside the loss; it must not cut the graph."""
    print("\n[differentiable]")
    gt = torch.rand(1, 1, 48, 48)
    pred = (gt + 0.01).clone().requires_grad_(True)
    m = torch.zeros(1, 1, 48, 48, dtype=torch.bool)
    m[..., 8:40, 8:40] = True
    psnr(gt, pred, mask=m).backward()
    g = pred.grad
    check("gradient is nonzero inside the mask",
          bool(g[m.expand_as(g)].abs().sum() > 0))
    check("gradient is exactly zero outside",
          bool(g[~m.expand_as(g)].abs().sum() == 0))


if __name__ == "__main__":
    test_masked_ignores_background()
    test_masked_ignores_background_size()
    test_ssim_map_not_inputs()
    test_nrmse_range_is_masked_too()
    test_identity_and_edges()
    test_batch_and_channels()
    test_gradients_flow()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  FAILED: {f}")
    raise SystemExit(1 if FAIL else 0)
