"""
Where the object is, from the data rather than from the sensitivity maps.

`datasets/fastmri/loader.py` used to take the organ mask to be the support of
the coil maps, `smaps.abs().sum(0) > 0`. That support is DILATED, by design and
not by accident: ESPIRiT's kernels are `ks x ks` samples of k-space padded to
the full grid, so the eigenvalue map that defines the support is band-limited to
`ks` samples and cannot vary faster than about `N / ks` pixels -- 80 px in
readout and 40 px in phase on a 640x320 brain grid, halved by the squaring.
A boundary that soft, hard-thresholded, lands tens of pixels outside the skull,
which is why residual panels showed error in the air around the head.

The object's boundary IS sharp in the data, so this takes the mask from the
root-sum-of-squares coil image instead. RSS needs no maps at all, which is the
point: the metric region stops depending on ESPIRiT's hyperparameters, and a
map regeneration no longer silently redefines what "inside the brain" means.

    threshold (Otsu)  ->  open  ->  component of the peak  ->  close  ->  fill

each step removing one failure mode of the one before it:

  Otsu           splits the strongly bimodal RSS histogram (air vs tissue)
                 without a hand-tuned level, so one threshold works across
                 volumes whose scaling differs.
  open           drops the speckle Otsu leaves in the noise floor.
  component      keeps the connected piece containing the brightest pixel, so a
                 surviving blob of coil noise in a corner is dropped even if it
                 is larger than the opening radius.
  close          bridges the dark skull, which otherwise splits scalp from
                 brain and would make the fill below leak.
  fill           takes in the ventricles and any other interior dark region:
                 they are inside the object and a metric that skips them
                 measures a different region on every slice.

EVERYTHING IS PURE TORCH AND DETERMINISTIC. scipy.ndimage would be shorter, but
a mask that exists only where scipy is installed is a mask that differs between
the cluster and a laptop, and masked numbers from the two would not be
comparable while looking as if they were.

All functions take `(B, 1, H, W)`, `(1, H, W)` or `(H, W)` and return the same
shape, so the loader can hand them one slice and a test can hand them a batch.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# shape plumbing
# ---------------------------------------------------------------------------
def _as_b1hw(x):
    """`(B, 1, H, W)` plus the inverse reshape, for any of the accepted ranks."""
    if x.dim() == 2:
        return x[None, None], lambda y: y[0, 0]
    if x.dim() == 3:
        return x[None], lambda y: y[0]
    if x.dim() == 4:
        if x.shape[1] != 1:
            raise ValueError(
                f"expected a single channel, got {tuple(x.shape)}. RSS and the "
                f"mask are scalar fields; a coil axis here means the caller "
                f"forgot to combine.")
        return x, lambda y: y
    raise ValueError(f"expected (H, W), (1, H, W) or (B, 1, H, W), got "
                     f"{tuple(x.shape)}")


# ---------------------------------------------------------------------------
# thresholding
# ---------------------------------------------------------------------------
def otsu_threshold(x, bins=256, clip_q=0.995):
    """Otsu's threshold per sample, as `(B, 1, 1, 1)`.

    The histogram is taken up to the `clip_q` quantile rather than the maximum:
    a handful of bright fat or an inflow artifact otherwise stretches the range
    so far that the whole head falls in the first bin and the threshold lands
    in the noise. Everything above the clip counts in the top bin, so no signal
    is lost -- only the bin spacing changes.
    """
    x, _ = _as_b1hw(x)
    B = x.shape[0]
    flat = x.reshape(B, -1).float()
    out = torch.empty(B, 1, 1, 1, device=x.device, dtype=torch.float32)

    for b in range(B):
        v = flat[b]
        hi = torch.quantile(v, clip_q) if v.numel() < 2 ** 24 else v.max()
        hi = float(hi.clamp_min(torch.finfo(torch.float32).tiny))
        hist = torch.histc(v.clamp(max=hi), bins=bins, min=0.0, max=hi)
        p = hist / hist.sum().clamp_min(1)
        centers = (torch.arange(bins, device=x.device, dtype=torch.float32)
                   + 0.5) * (hi / bins)

        omega = torch.cumsum(p, 0)                      # class-0 weight
        mu = torch.cumsum(p * centers, 0)               # class-0 first moment
        mu_t = mu[-1]
        denom = (omega * (1 - omega)).clamp_min(1e-12)
        between = (mu_t * omega - mu) ** 2 / denom      # between-class variance
        out[b] = centers[int(torch.argmax(between))]
    return out


# ---------------------------------------------------------------------------
# morphology
# ---------------------------------------------------------------------------
def dilate(m, r):
    """Binary dilation by a `(2r+1)` square."""
    if r <= 0:
        return m
    m4, back = _as_b1hw(m)
    return back(F.max_pool2d(m4.float(), 2 * r + 1, stride=1, padding=r) > 0)


def erode(m, r):
    """Binary erosion by a `(2r+1)` square, treating OUTSIDE the image as background.

    Erosion is dilation of the complement, and the padding is the whole trap:
    `max_pool2d(padding=r)` pads with zeros, so the complement's border pads
    with "not background" and an object touching the image edge is never
    trimmed there. Padding the complement explicitly with 1 fixes it.
    """
    if r <= 0:
        return m
    m4, back = _as_b1hw(m)
    comp = F.pad((~m4.bool()).float(), (r,) * 4, value=1.0)
    return back(~(F.max_pool2d(comp, 2 * r + 1, stride=1) > 0))


def closing(m, r):
    """Dilate then erode: bridges gaps narrower than `2r`."""
    return erode(dilate(m, r), r)


def opening(m, r):
    """Erode then dilate: drops features thinner than `2r`."""
    return dilate(erode(m, r), r)


def propagate(seed, allowed, max_iter=1024, check_every=8):
    """Grow `seed` through `allowed` under 4/8-connectivity until it stops.

    A flood fill written as repeated dilation, which keeps the whole thing on
    the device and out of scipy. The iteration count bounds the reachable
    distance, so a shape longer than `max_iter` pixels would be filled only
    partway -- 1024 covers any fastMRI grid, and the convergence test exits
    long before that on real data.

    `check_every` trades syncs for iterations: the fixed-point test is a
    device-to-host read, so it runs every few steps rather than every one.
    """
    seed4, back = _as_b1hw(seed)
    allowed4, _ = _as_b1hw(allowed)
    cur = (seed4.bool() & allowed4.bool()).float()

    for i in range(max_iter):
        nxt = (F.max_pool2d(cur, 3, stride=1, padding=1) > 0).float() * allowed4.float()
        if (i + 1) % check_every == 0:
            if bool((nxt == cur).all()):
                cur = nxt
                break
        cur = nxt
    return back(cur > 0)


def fill_holes(m):
    """Fill background regions not connected to the image border."""
    m4, back = _as_b1hw(m)
    bg = ~m4.bool()
    border = torch.zeros_like(bg)
    border[..., 0, :] = True
    border[..., -1, :] = True
    border[..., :, 0] = True
    border[..., :, -1] = True
    outside = propagate(border & bg, bg)
    return back(m4.bool() | (bg & ~outside))


def component_of_peak(m, weight):
    """The connected component of `m` containing the largest value of `weight`.

    Not "the largest component": the brightest pixel is inside the anatomy by
    construction, while the largest component would be whatever survived the
    opening -- usually the same thing, but not on a slice where the object
    leaves the FOV.
    """
    m4, back = _as_b1hw(m)
    w4, _ = _as_b1hw(weight)
    B = m4.shape[0]
    seed = torch.zeros_like(m4, dtype=torch.bool)
    flat = (w4 * m4.float()).reshape(B, -1)
    idx = flat.argmax(dim=1)
    seed.reshape(B, -1)[torch.arange(B, device=m4.device), idx] = True
    seed = seed & m4.bool()
    if not bool(seed.any()):                 # empty mask: nothing to grow
        return back(m4.bool())
    return back(propagate(seed, m4.bool()))


# ---------------------------------------------------------------------------
# the mask
# ---------------------------------------------------------------------------
def rss_object_mask(rss, thresh=None, thresh_scale=1.0, open_px=1, close_px=3,
                    largest=True, fill=True, pad_px=0, bins=256):
    """Anatomy mask from a root-sum-of-squares coil image.

    Parameters
    ----------
    rss : (B, 1, H, W) | (1, H, W) | (H, W) real, non-negative
    thresh : float or tensor, optional
        Absolute level. Default is Otsu's, per sample.
    thresh_scale : float
        Multiplies the threshold. > 1 tightens the mask, < 1 loosens it. Left
        at 1.0 the level comes entirely from the data.
    open_px, close_px, pad_px : int
        Radii, in pixels, of the opening, the closing and a final dilation.
        `pad_px` is there for the case where you would rather keep a rim of
        background than risk clipping the cortex.
    largest, fill : bool
        Keep only the peak's component; fill interior holes.

    Returns
    -------
    bool tensor, the shape of `rss`.

    An all-false result is possible on a slice that is entirely noise, and is
    returned as such rather than falling back to all-true: the metrics report
    NaN on an empty mask, which reads as "no measurement" instead of quietly
    scoring the whole FOV.
    """
    r4, back = _as_b1hw(rss)
    if torch.is_complex(r4):
        raise ValueError("rss must be real; pass |coil| combined, not the "
                         "complex image.")
    r4 = r4.float()

    t = otsu_threshold(r4, bins=bins) if thresh is None else torch.as_tensor(
        thresh, device=r4.device, dtype=torch.float32).reshape(-1, 1, 1, 1)
    m = r4 > (t * float(thresh_scale))

    m = opening(m, open_px)
    if largest:
        m = component_of_peak(m, r4)
    m = closing(m, close_px)
    if fill:
        m = fill_holes(m)
    m = dilate(m, pad_px)
    return back(m)
