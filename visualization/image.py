"""
Image display helpers.

The notebooks kept re-deriving the same four steps: get a tensor onto the CPU as a
2-D real array, pick a display window, draw a grid of panels with the ticks off, and
(sometimes) put a residual next to them on its own scale. Everything below is that,
factored out.

The one rule worth knowing: **panels that should be compared must share a window.**
`imshow` normalises to its own min/max by default, so an over- or under-shooting
reconstruction gets silently rescaled into looking correct. `disp_window` computes one
`(vmin, vmax)` from whichever images define the scale (usually prior + ground truth,
inside the brain mask), and `subplot_images` applies it to every panel that has not
been given an explicit window of its own.

Quick reference
---------------
    to_numpy(x)                              any tensor/array -> 2-D float32
    disp_window(a, b, mask=m, p=(1, 99))     one shared (vmin, vmax)
    plot_image(x, vmin=.., vmax=.., cmap=..) one panel
    subplot_images([a, b, c], titles=[...])  a row / grid of panels
    save_image(x, path)                      same, to disk
"""

import numpy as np
import torch
import matplotlib.pyplot as plt


# ===========================================================================
#  Tensor -> displayable array
# ===========================================================================
def to_numpy(x, magnitude=True, index=0):
    """
    Any image-ish object -> a 2-D float32 numpy array that `imshow` accepts.

    Handles torch/numpy, complex input, and the leading batch/channel dimensions:
    singleton dims are squeezed away, and any remaining leading dims are indexed
    (so a (B, C, H, W) batch displays sample/channel `index`).

    Parameters
    ----------
    x : torch.Tensor or np.ndarray
        Input image, real or complex, any leading dims.
    magnitude : bool
        For complex input, take `abs()` rather than the real part.
    index : int
        Which slice to take along each surviving leading dimension.

    Returns
    -------
    np.ndarray
        Shape (H, W), dtype float32.
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        if x.is_complex():
            x = x.abs() if magnitude else x.real
        x = x.float().numpy()
    else:
        x = np.asarray(x)
        if np.iscomplexobj(x):
            x = np.abs(x) if magnitude else x.real

    x = np.squeeze(np.asarray(x, dtype=np.float32))

    while x.ndim > 2:
        x = x[min(index, x.shape[0] - 1)]

    if x.ndim != 2:
        raise ValueError(f"expected a 2-D image after squeezing, got shape {x.shape}")

    return x


def _as_float(x, magnitude=True):
    """Like `to_numpy` but keeps every dimension -- for statistics, not for display."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        if x.is_complex():
            x = x.abs() if magnitude else x.real
        return x.float().numpy()

    x = np.asarray(x)
    if np.iscomplexobj(x):
        x = np.abs(x) if magnitude else x.real

    return np.asarray(x, dtype=np.float32)


def _broadcast_mask(m, shape):
    """Squeezed boolean mask -> index for an array of `shape`, or None if it cannot."""
    if m.shape == shape:
        return m

    if m.ndim == 2 and len(shape) >= 2 and m.shape == shape[-2:]:
        return np.broadcast_to(m, shape)

    if m.size == int(np.prod(shape)):
        return m.reshape(shape)

    return None


def _pool(imgs, mask=None, magnitude=True, max_samples=1_000_000, seed=0):
    """
    Flatten a set of images into one 1-D float array for computing statistics.

    `mask` (if given) restricts to the pixels where it is nonzero -- the usual case
    being a brain mask, so background zeros do not dominate the percentiles. Arrays
    larger than `max_samples` are randomly subsampled, because `np.percentile` over a
    few 10^7 voxels is slow enough to be noticeable in a notebook and the estimate is
    already converged well before that.
    """
    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]

    m = None
    if mask is not None:
        m = np.asarray(np.squeeze(_as_float(mask, magnitude=False)), dtype=bool)

    vals = []
    for im in imgs:
        if im is None:
            continue
        a = _as_float(im, magnitude=magnitude)
        mm = _broadcast_mask(m, a.shape) if m is not None else None
        vals.append(a[mm].ravel() if mm is not None else a.ravel())

    if not vals:
        raise ValueError("nothing to pool")

    v = np.concatenate(vals)
    v = v[np.isfinite(v)]

    if v.size == 0:
        raise ValueError("no finite values to pool")

    if v.size > max_samples:
        rng = np.random.default_rng(seed)
        v = v[rng.integers(0, v.size, max_samples)]

    return v


def percentile_range(v, p=None, symmetric=False):
    """
    `(lo, hi)` from a pooled 1-D array.

    `p` is a `(lo, hi)` percentile pair, a scalar `q` meaning `(100 - q, q)`, or
    None for the full min/max. `symmetric` widens the interval to be symmetric
    about zero -- what a signed quantity wants so that zero sits at the middle of a
    diverging colormap, or at the middle of a difference histogram.
    """
    if p is None:
        lo, hi = float(np.min(v)), float(np.max(v))
    else:
        lo_p, hi_p = ((100.0 - float(p), float(p)) if np.isscalar(p)
                      else (float(p[0]), float(p[1])))
        lo, hi = (float(q) for q in np.percentile(v, [lo_p, hi_p]))

    if symmetric:
        r = max(abs(lo), abs(hi))
        lo, hi = -r, r

    if not hi > lo:
        hi = lo + 1e-8

    return lo, hi


# ===========================================================================
#  Display window
# ===========================================================================
def disp_window(*imgs, mask=None, p=None, symmetric=False, magnitude=True,
                max_samples=1_000_000, seed=0):
    """
    One `(vmin, vmax)` shared by a set of panels, so they are directly comparable.

    Pass the images that DEFINE the scale -- typically the prior and the ground
    truth, not the reconstruction, so a reconstruction that overshoots reads as
    overshooting instead of being renormalised into agreement.

    Parameters
    ----------
    *imgs : tensors / arrays
        Images to pool. Any shape; complex is reduced by magnitude.
    mask : tensor / array or None
        Restrict the statistics to nonzero mask pixels (e.g. the brain mask).
    p : tuple(float, float), float, or None
        Percentile stretch. `(1, 99)` clips both tails; a scalar `q` means
        `(100 - q, q)`. None uses the full min/max of the pooled data.
    symmetric : bool
        Force the window symmetric about zero -- what a signed residual wants, so
        that `bwr` / `coolwarm` puts white at exactly zero.
    magnitude : bool
        Reduce complex input by magnitude rather than real part.
    max_samples, seed : int, int
        Subsampling guard for very large inputs.

    Returns
    -------
    (float, float)
        `vmin, vmax`, guaranteed to be a nondegenerate interval.
    """
    v = _pool(list(imgs), mask=mask, magnitude=magnitude,
              max_samples=max_samples, seed=seed)

    return percentile_range(v, p=p, symmetric=symmetric)


# ===========================================================================
#  Legacy contrast clipping
# ===========================================================================
def contrast_enhance(x, thresh=1.0):
    """
    Clamp image intensities for visualization.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    thresh : float
        Maximum intensity value.

    Returns
    -------
    torch.Tensor
        Contrast-enhanced tensor.
    """
    return torch.clamp(x, max=thresh)


def prepare_image(x, magnitude=True, contrast=False, thresh=1.0):
    """
    Prepare a tensor for visualization.

    Kept for the older call sites; `to_numpy` is the general entry point.

    Parameters
    ----------
    x : torch.Tensor
        Input image tensor.
    magnitude : bool
        Take magnitude if tensor is complex.
    contrast : bool
        Apply contrast clipping.
    thresh : float
        Clipping threshold.

    Returns
    -------
    np.ndarray
        2-D float32 array.
    """
    x = to_numpy(x, magnitude=magnitude)

    if contrast:
        x = np.clip(x, None, thresh)

    return x


# ===========================================================================
#  Single panel
# ===========================================================================
def plot_image(
    x,
    vmin=None,
    vmax=None,
    cmap="gray",
    p=None,
    mask=None,
    apply_mask=False,
    symmetric=False,
    title=None,
    xlabel=None,
    colorbar=False,
    figsize=None,
    ax=None,
    overlay=None,
    overlay_color="lime",
    overlay_lw=0.6,
    overlay_levels=(0.5,),
    magnitude=True,
    index=0,
    axis_off=True,
    show=None,
    contrast=False,
    thresh=1.0,
):
    """
    Display one image.

    Parameters
    ----------
    x : torch.Tensor or np.ndarray
        Image, any leading dims, real or complex.
    vmin, vmax : float or None
        Display window. Whichever is None is filled in from the data (using `p`,
        `mask` and `symmetric`), so `imshow` never picks its own scale behind
        your back.
    cmap : str
        Matplotlib colormap. "gray" for images, "magma" for a magnitude residual,
        "bwr" / "coolwarm" with `symmetric=True` for a signed one.
    p : tuple(float, float), float, or None
        Percentile stretch used to fill in a missing vmin/vmax. See `disp_window`.
    mask : tensor / array or None
        Restricts the auto window to nonzero mask pixels.
    apply_mask : bool
        Also multiply the displayed image by the mask (background -> 0). Off by
        default: a mask should not silently change the pixels you are looking at.
    symmetric : bool
        Make the auto window symmetric about zero.
    title, xlabel : str or None
        Panel title / caption. `xlabel` is the natural place for a metric.
    colorbar : bool
        Attach a colorbar to this panel.
    figsize : tuple or None
        Figure size, when creating a new figure.
    ax : matplotlib Axes or None
        Draw into an existing axis instead of a new figure.
    overlay : tensor / array or None
        Binary map to outline on top (e.g. the ET segmentation).
    overlay_color, overlay_lw, overlay_levels
        Contour styling for `overlay`.
    magnitude : bool
        Take magnitude if complex-valued.
    index : int
        Slice to take along the leading dims of a batch.
    axis_off : bool
        Hide the frame entirely. Ignored when an `xlabel` is set, since the frame
        is what the label hangs off; ticks are removed either way.
    show : bool or None
        Call `plt.show()`. Defaults to True when this call created the figure.
    contrast, thresh : bool, float
        Legacy clipping (`clip(x, max=thresh)`) applied before display.

    Returns
    -------
    matplotlib Axes
    """
    img = to_numpy(x, magnitude=magnitude, index=index)

    if contrast:
        img = np.clip(img, None, thresh)

    if apply_mask and mask is not None:
        img = img * to_numpy(mask, magnitude=False, index=index)

    if vmin is None or vmax is None:
        lo, hi = disp_window(img, mask=mask, p=p, symmetric=symmetric)
        vmin = lo if vmin is None else vmin
        vmax = hi if vmax is None else vmax

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)

    if overlay is not None:
        _draw_overlay(ax, overlay, overlay_color, overlay_lw, overlay_levels, index)

    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)

    _strip_ticks(ax, axis_off and xlabel is None)

    if colorbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if show is None:
        show = created

    if show:
        plt.tight_layout()
        plt.show()

    return ax


def _strip_ticks(ax, axis_off):
    """Ticks always go; the frame goes only when nothing needs to hang off it."""
    if axis_off:
        ax.axis("off")
    else:
        ax.set_xticks([])
        ax.set_yticks([])


def _draw_overlay(ax, overlay, color, lw, levels, index=0):
    """Outline a binary map (ET segmentation, sampling mask, ...) over the panel."""
    e = to_numpy(overlay, magnitude=False, index=index)

    if np.nanmax(e) > 0:
        ax.contour(e, levels=list(levels), colors=color, linewidths=lw)


# ===========================================================================
#  Grid of panels
# ===========================================================================
def subplot_images(
    images,
    titles=None,
    ncols=None,
    vmin=None,
    vmax=None,
    cmap="gray",
    p=None,
    mask=None,
    apply_mask=False,
    symmetric=False,
    share_window=True,
    window_from=None,
    row_labels=None,
    col_titles=None,
    xlabels=None,
    overlay=None,
    overlay_color="lime",
    overlay_lw=0.6,
    overlay_levels=(0.5,),
    panel_size=(2.4, 2.7),
    figsize=None,
    suptitle=None,
    colorbar=False,
    cbar_label=None,
    magnitude=True,
    index=0,
    axis_off=True,
    show=True,
):
    """
    Draw a row or grid of images that share one display window.

    `images` may be

      * a flat sequence            -> one row (or wrapped to `ncols`),
      * a nested sequence of rows  -> a grid, one inner sequence per row,
      * a batched tensor/array with ndim >= 3 -> split along the leading axis,

    and `None` in any slot leaves that cell blank.

    Per-panel overrides
    -------------------
    `cmap`, `vmin`, `vmax`, `p` and `overlay` each accept either one value for the
    whole grid or a sequence matching it (a per-column list is repeated down the
    rows). That is what a residual column needs: grayscale everywhere on the shared
    window, then `cmap=[..., "magma"]` with `vmin=[..., 0], vmax=[..., 0.3]` on the
    last panel, so the residual keeps its own FIXED scale and its brightness means
    the same fraction of signal at every epoch.

    Parameters
    ----------
    images : sequence, nested sequence, or array
        Panels to draw.
    titles : sequence, nested sequence, or None
        Per-panel titles, matching `images`.
    ncols : int or None
        Wrap a flat `images` into this many columns.
    vmin, vmax : float, sequence, or None
        Display window; see `share_window` for how None is resolved.
    cmap : str or sequence
        Colormap(s).
    p : tuple, float, sequence, or None
        Percentile stretch for the auto window. See `disp_window`.
    mask : tensor / array or None
        Restricts auto-window statistics to nonzero mask pixels.
    apply_mask : bool
        Multiply every panel by the mask before display.
    symmetric : bool
        Make the auto window symmetric about zero.
    share_window : bool
        Compute ONE window across all panels lacking an explicit vmin/vmax, so the
        panels are comparable. With False each panel gets its own window, which is
        right only when the panels are on genuinely different scales.
    window_from : sequence or None
        Compute the shared window from these images instead of from the panels --
        e.g. `[x1, x0]`, so the prior and ground truth set the scale and the
        reconstruction is judged against it.
    row_labels : sequence or None
        Label down the left edge, one per row.
    col_titles : sequence or None
        Titles on the FIRST row only, one per column -- the compact way to label a
        grid whose rows are different slices of the same set of methods.
    xlabels : sequence, nested sequence, or None
        Per-panel caption under the panel (metrics go here).
    overlay : tensor / array / sequence / None
        Binary map(s) to outline on the panels.
    overlay_color, overlay_lw, overlay_levels
        Contour styling.
    panel_size : tuple
        (width, height) in inches per panel; ignored if `figsize` is given.
    figsize : tuple or None
        Explicit figure size.
    suptitle : str or None
        Figure title.
    colorbar : bool or "each"
        True attaches one colorbar to the whole figure (from the last panel drawn);
        "each" gives every panel its own.
    cbar_label : str or None
        Label for the figure-level colorbar.
    magnitude, index, axis_off, show
        As in `plot_image`.

    Returns
    -------
    (fig, axes)
        `axes` is always a 2-D array, so `axes[i, j]` works even for one row.
    """
    grid = _as_grid(images, ncols)
    nrows = len(grid)
    ncols_ = max(len(r) for r in grid)
    grid = [list(r) + [None] * (ncols_ - len(r)) for r in grid]

    titles_g = _match(titles, nrows, ncols_, "titles")
    xlabels_g = _match(xlabels, nrows, ncols_, "xlabels")
    cmap_g = _match(cmap, nrows, ncols_, "cmap")
    vmin_g = _match(vmin, nrows, ncols_, "vmin")
    vmax_g = _match(vmax, nrows, ncols_, "vmax")
    p_g = _match(p, nrows, ncols_, "p", tuple_is_scalar=True)
    ovl_g = _match(overlay, nrows, ncols_, "overlay", array_is_scalar=True)

    # One window for every panel that did not ask for its own. Panels WITH an
    # explicit vmin/vmax (the residual column) are excluded from both the pooling
    # and the fill-in, so their fixed scale survives.
    shared = None
    if share_window:
        if window_from is not None:
            pool = list(window_from) if isinstance(window_from, (list, tuple)) else [window_from]
            p_ref = p if not isinstance(p, list) else None
        else:
            pool, p_ref = [], None
            for i in range(nrows):
                for j in range(ncols_):
                    if grid[i][j] is None:
                        continue
                    if vmin_g[i][j] is None and vmax_g[i][j] is None:
                        pool.append(grid[i][j])
                        if p_ref is None:
                            p_ref = p_g[i][j]
        if pool:
            shared = disp_window(*pool, mask=mask, p=p_ref,
                                 symmetric=symmetric, magnitude=magnitude)

    if figsize is None:
        figsize = (panel_size[0] * ncols_, panel_size[1] * nrows)

    fig, axes = plt.subplots(nrows, ncols_, figsize=figsize, squeeze=False)

    im = None
    for i in range(nrows):
        for j in range(ncols_):
            ax = axes[i, j]
            x = grid[i][j]

            if x is None:
                ax.axis("off")
                continue

            lo, hi = vmin_g[i][j], vmax_g[i][j]
            if shared is not None and lo is None and hi is None:
                lo, hi = shared

            ttl = titles_g[i][j]
            if ttl is None and col_titles is not None and i == 0 and j < len(col_titles):
                ttl = col_titles[j]

            plot_image(
                x,
                vmin=lo,
                vmax=hi,
                cmap=cmap_g[i][j] or "gray",
                p=p_g[i][j],
                mask=mask,
                apply_mask=apply_mask,
                symmetric=symmetric,
                title=ttl,
                xlabel=xlabels_g[i][j],
                colorbar=(colorbar == "each"),
                ax=ax,
                overlay=ovl_g[i][j],
                overlay_color=overlay_color,
                overlay_lw=overlay_lw,
                overlay_levels=overlay_levels,
                magnitude=magnitude,
                index=index,
                axis_off=axis_off,
                show=False,
            )
            im = ax.images[-1]

            if row_labels is not None and j == 0 and i < len(row_labels):
                # axis("off") would hide this, so a row label forces the frame back on.
                ax.axis("on")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_ylabel(row_labels[i], fontsize=10)

    if suptitle is not None:
        fig.suptitle(suptitle, y=1.0)

    if colorbar is True and im is not None:
        # A figure-level colorbar steals space from the axes, so tight_layout is
        # skipped here -- running both fights over the same room and clips labels.
        fig.colorbar(im, ax=axes, shrink=0.6, label=cbar_label)
    else:
        fig.tight_layout()

    if show:
        plt.show()

    return fig, axes


def _as_grid(images, ncols=None):
    """Normalise the shapes `images` can arrive in into a list of rows."""
    if isinstance(images, (torch.Tensor, np.ndarray)):
        images = [images[k] for k in range(images.shape[0])] if images.ndim >= 3 else [images]

    if not isinstance(images, (list, tuple)):
        images = [images]

    if any(isinstance(r, (list, tuple)) for r in images):
        return [list(r) if isinstance(r, (list, tuple)) else [r] for r in images]

    images = list(images)

    if ncols is None or ncols >= len(images):
        return [images]

    return [images[k:k + ncols] for k in range(0, len(images), ncols)]


def _match(value, nrows, ncols, name, tuple_is_scalar=False, array_is_scalar=False):
    """
    Broadcast a per-grid option to a nested list shaped like the grid.

    A scalar (or None, or a string) applies to every panel. A list is taken
    per-panel: nested lists are rows, a flat list of `nrows * ncols` entries is
    reshaped, and any other flat list is treated as one entry per COLUMN and
    repeated down the rows -- which is what a per-method cmap or a residual
    column's fixed window wants.
    """
    def fill(v):
        return [[v] * ncols for _ in range(nrows)]

    if value is None or isinstance(value, str):
        return fill(value)

    if tuple_is_scalar and isinstance(value, tuple):
        return fill(value)

    if array_is_scalar and isinstance(value, (torch.Tensor, np.ndarray)):
        return fill(value)

    if not isinstance(value, (list, tuple)):
        return fill(value)

    if any(isinstance(r, (list, tuple)) for r in value):
        rows = list(value)
    else:
        flat = list(value)
        if nrows == 1:
            rows = [flat]
        elif len(flat) == nrows * ncols:
            rows = [flat[i * ncols:(i + 1) * ncols] for i in range(nrows)]
        else:
            rows = [flat for _ in range(nrows)]

    out = []
    for i in range(nrows):
        r = rows[i] if i < len(rows) else None
        r = list(r) if isinstance(r, (list, tuple)) else [r]

        if len(r) > ncols:
            raise ValueError(
                f"{name}: row {i} has {len(r)} entries, grid has {ncols} columns"
            )

        out.append(r + [None] * (ncols - len(r)))

    return out


# ===========================================================================
#  Saving, k-space
# ===========================================================================
def save_image(
    x,
    path,
    vmin=None,
    vmax=None,
    cmap="gray",
    p=None,
    mask=None,
    magnitude=True,
    index=0,
    dpi=300,
    contrast=False,
    thresh=1.0,
    verbose=True,
):
    """
    Save an image to disk, cropped to the pixels and nothing else.

    Parameters
    ----------
    x : torch.Tensor or np.ndarray
        Input image.
    path : str
        Output filepath.
    vmin, vmax, cmap, p, mask, magnitude, index
        As in `plot_image`.
    dpi : int
        Output resolution.
    contrast, thresh : bool, float
        Legacy clipping applied before saving.
    verbose : bool
        Print the destination.
    """
    img = to_numpy(x, magnitude=magnitude, index=index)

    if contrast:
        img = np.clip(img, None, thresh)

    if vmin is None or vmax is None:
        lo, hi = disp_window(img, mask=mask, p=p)
        vmin = lo if vmin is None else vmin
        vmax = hi if vmax is None else vmax

    fig = plt.figure()
    plt.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.axis("off")

    plt.savefig(path, bbox_inches="tight", pad_inches=0, dpi=dpi)
    plt.close(fig)

    if verbose:
        print(f"Saved image to {path}.")


def show_kspace(kspace, log=True, cmap="gray", figsize=None, ax=None,
                title=None, show=None):
    """
    Visualize k-space magnitude.

    Parameters
    ----------
    kspace : torch.Tensor
        Complex k-space tensor.
    log : bool
        Apply a log1p transform -- without it the DC spike is the only visible pixel.
    cmap : str
        Matplotlib colormap.
    figsize : tuple or None
        Figure size.
    ax : matplotlib Axes or None
        Draw into an existing axis.
    title : str or None
        Panel title.
    show : bool or None
        Call `plt.show()`; defaults to True when this call created the figure.

    Returns
    -------
    matplotlib Axes
    """
    x = kspace.abs() if isinstance(kspace, torch.Tensor) else np.abs(kspace)

    if log:
        x = torch.log1p(x) if isinstance(x, torch.Tensor) else np.log1p(x)

    return plot_image(x, cmap=cmap, figsize=figsize, ax=ax, title=title,
                      magnitude=False, show=show)


# ===========================================================================
#  Reconstruction diagnostic panels
# ===========================================================================
def _to_mag(x):
    """(B,C,H,W) possibly-complex -> (1,1,H,W) real magnitude of the first slice."""
    x = x[:1]
    if x.shape[1] > 1:
        x = x[:, :1]
    return x.abs().detach().float().cpu()


def recon_panel(gt, recon, zero_filled=None, gain=4.0, eps=1e-12):
    """Side-by-side [zero-filled | recon | ground truth | |residual| x gain].

    The three IMAGE panels share one intensity scale (the ground truth's max),
    so they are directly comparable and an over/under-shooting reconstruction
    reads as such instead of being silently renormalised away. The residual is
    scaled by the SAME reference times `gain`, so its brightness means a fixed
    fraction of signal at every epoch -- an artifact that grows over training
    looks like it is growing.

    Returns `(panel, stats)`; `stats` carries the numbers worth putting in a
    caption (`res_max`, `res_rms`, both relative to the ground-truth max).
    """
    gt_m, rec_m = _to_mag(gt), _to_mag(recon)
    ref = gt_m.max().clamp_min(eps)

    res = (rec_m - gt_m).abs()
    stats = {"res_max": (res.max() / ref).item(),
             "res_rms": (res.pow(2).mean().sqrt() / ref).item(),
             "gain": float(gain)}

    panels = []
    if zero_filled is not None:
        panels.append((_to_mag(zero_filled) / ref).clamp(0, 1))
    panels += [(rec_m / ref).clamp(0, 1),
               (gt_m / ref).clamp(0, 1),
               (res * gain / ref).clamp(0, 1)]

    # Built with plain torch rather than make_grid: one strip, explicit
    # separators, and no renormalisation behind our back (make_grid's
    # `normalize` defaults differ across versions, and a silent rescale is
    # exactly what makes a residual panel lie).
    pad = torch.ones(1, 1, panels[0].shape[-2], 2)
    strip = [panels[0]]
    for q in panels[1:]:
        strip += [pad, q]
    return torch.cat(strip, dim=-1)[0], stats


def residual_kspace(gt, recon, eps=1e-12):
    """log-magnitude spectrum of the residual -- what kind of artifact is this?

    Reading it:
      * bright horizontal/vertical LINES at regular spacing -> undersampling
        fold-over. The spacing is the acceleration: replicas sit at multiples
        of N/R along the phase-encode axis. This is a reconstruction-quality
        problem (more data consistency / more iterations / stronger prior).
      * bright spots at the CORNERS and at the half/quarter band edges ->
        grid-locked structure at the Nyquist of a coarse level, i.e. the
        multigrid transfer operators or a strided ConvTranspose checkerboard.
        More depth will NOT remove this one.
      * a broadband floor -> plain noise.
    """
    d = _to_mag(recon) - _to_mag(gt)
    D = torch.fft.fftshift(torch.fft.fft2(d)).abs()
    D = torch.log10(D + eps)
    D = D - D.min()
    return (D / D.max().clamp_min(eps))[0]          # (1, H, W)
