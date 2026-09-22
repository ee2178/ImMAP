"""
The RSS organ mask (`physics/object_mask.py`).

Run with `python -m tests.test_object_mask`.

Every case is a phantom with a KNOWN answer, so each assertion is a measurement
rather than a snapshot: a disc of tissue in a noise background, with the
failure modes the pipeline exists to remove (speckle, a bright corner blob, a
dark interior hole, an object touching the image edge).

The property that matters most is the LAST one: the mask must be tight against
the object's edge, which the coil-map support it replaces is not.
"""

import torch

from physics.object_mask import (
    closing, component_of_peak, dilate, erode, fill_holes, opening,
    otsu_threshold, propagate, rss_object_mask,
)

FAIL = []


def check(name, ok, detail=""):
    print(f"[{'ok ' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def disc(H, W, cy, cx, r, device="cpu"):
    y = torch.arange(H, device=device)[:, None] - cy
    x = torch.arange(W, device=device)[None, :] - cx
    return (y ** 2 + x ** 2) <= r ** 2


def phantom(H=128, W=96, r=30, sigma=0.02, seed=0):
    """A bright disc on a noise floor, plus the RSS-like positive bias."""
    g = torch.Generator().manual_seed(seed)
    obj = disc(H, W, H // 2, W // 2, r).float()
    noise = torch.randn(H, W, generator=g).abs() * sigma
    return obj + noise, obj.bool()


# ---------------------------------------------------------------------------
def test_shapes_round_trip():
    x, _ = phantom()
    for shape in ((128, 96), (1, 128, 96), (2, 1, 128, 96)):
        v = x.expand(shape) if len(shape) > 2 else x
        m = rss_object_mask(v)
        check(f"rss_object_mask keeps shape {shape}",
              tuple(m.shape) == tuple(shape) and m.dtype == torch.bool,
              f"got {tuple(m.shape)} {m.dtype}")

    bad = torch.zeros(2, 3, 16, 16)
    try:
        rss_object_mask(bad)
        check("a coil axis is rejected", False, "no error raised")
    except ValueError:
        check("a coil axis is rejected", True)


def test_otsu_splits_background():
    x, obj = phantom(sigma=0.02)
    t = float(otsu_threshold(x))
    bg_max = float(x[~obj].max())
    fg_min = float(x[obj].min())
    check("Otsu lands between the noise floor and the object",
          bg_max < t < fg_min, f"bg<{bg_max:.3f}  t={t:.3f}  fg>{fg_min:.3f}")

    # 10x brighter data, same picture: the threshold must scale with it, or a
    # fixed level would silently mean something different per volume.
    t10 = float(otsu_threshold(x * 10))
    check("Otsu is scale-equivariant", abs(t10 - 10 * t) < 0.05 * 10 * t,
          f"{t:.4f} -> {t10:.4f}")


def test_otsu_survives_a_hot_pixel():
    """A single bright artifact must not drag the threshold into the noise."""
    x, obj = phantom(sigma=0.02)
    t = float(otsu_threshold(x))
    x2 = x.clone()
    x2[3, 3] = 500.0
    t2 = float(otsu_threshold(x2))
    check("a 500x hot pixel barely moves the threshold",
          abs(t2 - t) < 0.25 * t, f"{t:.4f} -> {t2:.4f}")


def test_morphology_primitives():
    m = disc(64, 64, 32, 32, 10)
    check("dilate grows monotonically with r",
          int(m.sum()) < int(dilate(m, 1).sum()) < int(dilate(m, 3).sum()),
          f"{int(m.sum())} < {int(dilate(m, 1).sum())} < {int(dilate(m, 3).sum())}")
    check("erode is the inverse on a smooth shape",
          int(erode(dilate(m, 2), 2).sum()) == int(m.sum()) or
          abs(int(erode(dilate(m, 2), 2).sum()) - int(m.sum())) < 0.02 * int(m.sum()),
          f"{int(m.sum())} -> {int(erode(dilate(m, 2), 2).sum())}")

    # the border trap: erosion must trim an object that runs off the edge
    edge = torch.zeros(32, 32, dtype=torch.bool)
    edge[:, :8] = True
    er = erode(edge, 2)
    check("erode treats outside the image as background",
          not bool(er[:, 0].any()) and not bool(er[0, 4]),
          f"{int(edge.sum())} -> {int(er.sum())} px")

    thin = torch.zeros(48, 48, dtype=torch.bool)
    thin[24, 4:44] = True                      # 1-px line
    check("opening removes a 1-px line", int(opening(thin, 1).sum()) == 0)

    gap = disc(64, 64, 32, 32, 12) & ~disc(64, 64, 32, 32, 8)
    gap = gap.clone()
    gap[28:36, 30:34] = False                  # cut a notch in the annulus
    check("closing bridges a 4-px notch",
          int(closing(gap, 3).sum()) > int(gap.sum()))


def test_propagate_and_fill():
    allowed = disc(64, 64, 32, 32, 20)
    seed = torch.zeros(64, 64, dtype=torch.bool)
    seed[32, 32] = True
    grown = propagate(seed, allowed)
    check("propagate fills a connected region", int(grown.sum()) == int(allowed.sum()),
          f"{int(grown.sum())} of {int(allowed.sum())}")

    far = disc(64, 64, 8, 8, 4)
    check("propagate does not jump a gap",
          int(propagate(seed, allowed | far).sum()) == int(allowed.sum()))

    ring = disc(64, 64, 32, 32, 20) & ~disc(64, 64, 32, 32, 6)
    filled = fill_holes(ring)
    check("fill_holes takes in an interior hole",
          int(filled.sum()) == int(disc(64, 64, 32, 32, 20).sum()),
          f"{int(ring.sum())} -> {int(filled.sum())}")

    open_c = disc(64, 64, 32, 32, 20).clone()
    open_c[32:, :] = False                     # a cup, open to the border
    check("fill_holes leaves a region connected to the border alone",
          int(fill_holes(open_c).sum()) == int(open_c.sum()))


def test_component_of_peak():
    big = disc(64, 64, 20, 20, 10)
    small = disc(64, 64, 55, 55, 4)
    w = torch.zeros(64, 64)
    w[big] = 1.0
    w[small] = 0.5
    kept = component_of_peak(big | small, w)
    check("component_of_peak keeps the peak's blob and drops the other",
          bool((kept == big).all()))


def test_mask_on_a_phantom():
    x, obj = phantom(H=160, W=128, r=40, sigma=0.03)

    # speckle in the background, a dark hole inside, a blob in a corner
    g = torch.Generator().manual_seed(3)
    x = x.clone()
    spike = torch.rand(160, 128, generator=g) > 0.997
    x[spike] += 0.9
    hole = disc(160, 128, 80, 64, 7)
    x[hole] = 0.02
    corner = disc(160, 128, 12, 12, 5)
    x[corner] += 1.0

    m = rss_object_mask(x)
    inside = float((m & obj).sum()) / float(obj.sum())
    leak = float((m & ~obj).sum()) / float((~obj).sum())
    check("the mask covers the object", inside > 0.99, f"{inside:.3%} of it")
    check("speckle and the corner blob are excluded", leak < 0.01,
          f"{leak:.3%} of the background kept")
    check("the interior hole is filled", bool(m[hole].all()))
    check("the corner blob is gone", not bool(m[corner].any()))

    # Tightness: how far the mask boundary sits from the object boundary, which
    # is the whole reason this replaced the coil-support mask.
    grew = int((m & ~obj).sum())
    ring = int((dilate(obj, 2) & ~obj).sum())
    check("the boundary is tight (within ~2 px of the object)", grew <= ring,
          f"{grew} px outside vs {ring} px in a 2-px rim")


def test_thresh_scale_and_empty():
    # A SOFT-EDGED phantom, so the threshold has somewhere to move: on the
    # binary disc every level between the noise and 1.0 gives the same mask and
    # the check below would pass without measuring anything.
    H, W = 128, 128
    y = torch.arange(H)[:, None] - H / 2
    x_ = torch.arange(W)[None, :] - W / 2
    rad = (y ** 2 + x_ ** 2).sqrt()
    x = torch.sigmoid((30.0 - rad) / 6.0)            # edge spread over ~12 px
    loose = rss_object_mask(x, thresh_scale=0.5)
    # 3.0 would exceed the phantom's peak and return an EMPTY mask, which is
    # correct but degenerate -- 1.5 keeps both sides of the comparison alive.
    tight = rss_object_mask(x, thresh_scale=1.5)
    check("thresh_scale tightens and loosens the mask",
          int(tight.sum()) <= int(rss_object_mask(x).sum()) <= int(loose.sum()),
          f"{int(tight.sum())} <= {int(rss_object_mask(x).sum())} <= {int(loose.sum())}")

    noise = torch.rand(64, 64) * 1e-6
    m = rss_object_mask(noise, thresh=1.0)
    check("an all-noise slice gives an EMPTY mask, not a full one",
          int(m.sum()) == 0, f"{int(m.sum())} px kept")


def test_batch_is_independent():
    """Per-sample thresholds: a bright slice must not set a dim one's level."""
    a, obj_a = phantom(H=96, W=96, r=20, sigma=0.02, seed=1)
    b, obj_b = phantom(H=96, W=96, r=20, sigma=0.02, seed=2)
    batch = torch.stack([a, b * 20.0])[:, None]

    m = rss_object_mask(batch)
    alone = torch.cat([rss_object_mask(a[None, None]),
                       rss_object_mask((b * 20.0)[None, None])])
    check("a batched mask equals the per-sample masks",
          bool((m == alone).all()),
          f"{int((m != alone).sum())} px differ")


def main():
    for fn in (test_shapes_round_trip, test_otsu_splits_background,
               test_otsu_survives_a_hot_pixel, test_morphology_primitives,
               test_propagate_and_fill, test_component_of_peak,
               test_mask_on_a_phantom, test_thresh_scale_and_empty,
               test_batch_is_independent):
        print(f"\n--- {fn.__name__}")
        fn()
    print(f"\n{'FAILED: ' + ', '.join(FAIL) if FAIL else 'all checks passed'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
