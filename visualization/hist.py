"""
Intensity-histogram helpers.

Every histogram in the notebooks is one of three things, and all three want the same
four decisions made consistently:

  1. **Which voxels.** Almost always the brain, not the image -- background zeros are a
     spike at 0 that flattens everything else. Pass `mask=`.
  2. **Which range.** Raw BraTS intensities have a tail that runs to 10x the tissue
     range, so a full-range histogram is one bar. Pass `p=(0.5, 99.5)` (or a scalar
     `p=99.5`, meaning `(0.5, 99.5)`) and the tails are clipped out of the RANGE, not
     out of the data.
  3. **Shared bins.** Two distributions drawn with their own bin edges cannot be
     compared. `plot_hist` given several series computes one set of edges from the
     pooled data and uses it for all of them.
  4. **Density vs count.** Comparing populations of different size means `density=True`,
     which is the default as soon as there is more than one series.

Quick reference
---------------
    plot_hist(x, mask=m, p=99.5)                        one distribution
    plot_hist({"T1": a, "T1ce": b}, mask=m, log=True)   several, shared bins
    plot_hist(d, symmetric=True, vlines={"median": mu}) a signed difference
"""

import numpy as np
import matplotlib.pyplot as plt

from .image import _pool, percentile_range


def _as_series(data):
    """Normalise the shapes `data` arrives in into a list of `(label, values)`."""
    if isinstance(data, dict):
        return list(data.items())

    if isinstance(data, (list, tuple)):
        if all(isinstance(d, (list, tuple)) and len(d) == 2 and isinstance(d[0], str)
               for d in data):
            return [(str(k), v) for k, v in data]
        return [(None, d) for d in data]

    return [(None, data)]


def plot_hist(
    data,
    bins=100,
    mask=None,
    range=None,
    p=None,
    symmetric=False,
    density=None,
    log=False,
    histtype=None,
    alpha=None,
    color=None,
    label=None,
    vlines=None,
    vline_color="crimson",
    title=None,
    xlabel="value",
    ylabel=None,
    legend=None,
    grid=True,
    ax=None,
    figsize=(6, 4),
    show=None,
    max_samples=1_000_000,
    seed=0,
    return_values=False,
):
    """
    Plot one or several intensity distributions on shared bins.

    Parameters
    ----------
    data : tensor / array, dict, or sequence
        One series, `{label: values}`, a sequence of `(label, values)` pairs, or a
        plain sequence of arrays. Any shape; it is flattened. NaN/inf are dropped.
    bins : int or array
        Bin count, or explicit edges (which override `range` / `p`).
    mask : tensor / array or None
        Keep only the voxels where the mask is nonzero -- the brain, normally.
        Applied to every series, so they must share the mask's shape.
    range : tuple(float, float) or None
        Explicit `(lo, hi)` bin range.
    p : tuple(float, float), float, or None
        Percentile clip defining the range instead: `(0.5, 99.5)`, or a scalar `q`
        meaning `(100 - q, q)`. Ignored if `range` is given.
    symmetric : bool
        Widen the range to be symmetric about zero -- for a signed quantity
        (T1ce - T1, a residual), so that zero sits in the middle.
    density : bool or None
        Normalise to a density. Defaults to True with more than one series (the
        only way to compare populations of different size), False with one.
    log : bool
        Log-scale the y axis. Worth it whenever a tail matters, which for these
        distributions is most of the time.
    histtype : str or None
        Defaults to "step" for several series (fills occlude each other) and
        "bar" for one.
    alpha, color, label : float / str / str, or sequences
        Per-series styling. `label` overrides the labels carried by `data`.
    vlines : dict, sequence, or float or None
        Vertical markers: `{label: x}`, a sequence of `x` or `(x, label)`, or one
        `x`. Drawn dashed, and labelled ones join the legend.
    vline_color : str
        Colour for the markers.
    title, xlabel, ylabel : str or None
        Labels. `ylabel` defaults to "density" or "count".
    legend : bool or None
        Defaults to True when anything is labelled.
    grid : bool
        Light grid behind the bars.
    ax : matplotlib Axes or None
        Draw into an existing axis instead of a new figure.
    figsize : tuple
        Figure size, when creating a new figure.
    show : bool or None
        Call `plt.show()`; defaults to True when this call created the figure.
    max_samples, seed : int, int
        Subsampling guard, per series.
    return_values : bool
        Also return the pooled 1-D arrays actually plotted, keyed by label
        (or by position, for unlabelled series).

    Returns
    -------
    matplotlib Axes, or (Axes, dict) when `return_values` is True.
    """
    series = _as_series(data)
    n = len(series)

    values = [
        _pool(v, mask=mask, max_samples=max_samples, seed=seed)
        for _, v in series
    ]

    if density is None:
        density = n > 1
    if histtype is None:
        histtype = "step" if n > 1 else "bar"
    if alpha is None:
        alpha = 0.55 if (n > 1 and histtype == "bar") else None

    # One set of edges from the pooled data, so the series are comparable. Without
    # this each series gets its own edges and the overlay means nothing.
    if np.isscalar(bins):
        if range is None:
            lo, hi = percentile_range(np.concatenate(values), p=p, symmetric=symmetric)
        else:
            lo, hi = float(range[0]), float(range[1])
        edges = np.linspace(lo, hi, int(bins) + 1)
    else:
        edges = np.asarray(bins)

    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=figsize)

    labels = _per_series(label, n) if label is not None else [lb for lb, _ in series]
    colors = _per_series(color, n)
    alphas = _per_series(alpha, n)

    for k, v in enumerate(values):
        # `edgecolor` must not be passed at all for "step": there the edge IS the visible line,
        # and an explicit edgecolor -- None included -- overrides `color` and resets it to the
        # rcParam default (black). Passing it unconditionally made every step series black, so
        # the per-series `color` was silently dead on the multi-series default path.
        style = {} if histtype == "step" else {"edgecolor": "black"}
        ax.hist(
            v,
            bins=edges,
            density=density,
            histtype=histtype,
            label=labels[k],
            color=colors[k],
            alpha=alphas[k],
            linewidth=0.8 if histtype == "step" else 0.3,
            **style,
        )

    for x, lb in _as_vlines(vlines):
        ax.axvline(x, color=vline_color, ls="--", lw=1.2, label=lb)

    if log:
        ax.set_yscale("log")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel if ylabel is not None else ("density" if density else "count"))

    if title is not None:
        ax.set_title(title)

    if grid:
        ax.grid(alpha=0.3)

    if legend is None:
        legend = any(lb is not None for lb in labels) or bool(_as_vlines(vlines))
    if legend:
        ax.legend(fontsize=8)

    if show is None:
        show = created
    if show:
        plt.tight_layout()
        plt.show()

    if return_values:
        out = {}
        for k, v in enumerate(values):
            out[labels[k] if labels[k] is not None else k] = v
        return ax, out

    return ax


def subplot_hists(
    panels,
    ncols=None,
    titles=None,
    share_bins=False,
    suptitle=None,
    panel_size=(5.0, 3.6),
    figsize=None,
    show=True,
    **kwargs,
):
    """
    A row or grid of histograms, one `plot_hist` call per panel.

    Parameters
    ----------
    panels : sequence or dict
        One entry per panel, each of them anything `plot_hist` accepts as `data`.
        A dict `{title: data}` supplies the titles too.
    ncols : int or None
        Wrap into this many columns.
    titles : sequence or None
        Per-panel titles (overrides dict keys).
    share_bins : bool
        Compute one set of bin edges across every panel, so the panels are
        directly comparable. Off by default, since panels usually hold different
        quantities (S_add vs S_mul) that should NOT share an axis.
    suptitle : str or None
        Figure title.
    panel_size : tuple
        (width, height) in inches per panel; ignored if `figsize` is given.
    figsize : tuple or None
        Explicit figure size.
    show : bool
        Call `plt.show()`.
    **kwargs
        Forwarded to every `plot_hist` call. Any value given as a list whose
        length matches the panel count is spread ACROSS the panels instead.

    Returns
    -------
    (fig, axes)
        `axes` is a flat array, one entry per panel.
    """
    if isinstance(panels, dict):
        titles = titles if titles is not None else list(panels.keys())
        panels = list(panels.values())

    panels = list(panels)
    n = len(panels)

    ncols = ncols or n
    nrows = int(np.ceil(n / ncols))

    if figsize is None:
        figsize = (panel_size[0] * ncols, panel_size[1] * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.ravel()

    if share_bins and np.isscalar(kwargs.get("bins", 100)):
        pooled = []
        for pnl in panels:
            _, vals = plot_hist(pnl, ax=axes[0], show=False, return_values=True,
                                **{k: v for k, v in kwargs.items() if k != "bins"})
            pooled.extend(vals.values())
        axes[0].clear()
        lo, hi = percentile_range(
            np.concatenate(pooled),
            p=kwargs.get("p"),
            symmetric=kwargs.get("symmetric", False),
        )
        kwargs["bins"] = np.linspace(lo, hi, int(kwargs.get("bins", 100)) + 1)

    for k, pnl in enumerate(panels):
        kw = {key: (v[k] if isinstance(v, list) and len(v) == n else v)
              for key, v in kwargs.items()}
        plot_hist(
            pnl,
            ax=axes[k],
            title=titles[k] if titles is not None and k < len(titles) else None,
            show=False,
            **kw,
        )

    for ax in axes[n:]:
        ax.axis("off")

    if suptitle is not None:
        fig.suptitle(suptitle, y=1.0)

    fig.tight_layout()

    if show:
        plt.show()

    return fig, axes


def _per_series(value, n):
    """A scalar style applies to every series; a matching list is taken per series."""
    if isinstance(value, (list, tuple)) and len(value) == n:
        return list(value)
    return [value] * n


def _as_vlines(vlines):
    """Normalise the marker spec into a list of `(x, label)`."""
    if vlines is None:
        return []

    if isinstance(vlines, dict):
        return [(float(x), str(lb)) for lb, x in vlines.items()]

    if np.isscalar(vlines):
        return [(float(vlines), None)]

    out = []
    for v in vlines:
        if isinstance(v, (list, tuple)) and len(v) == 2:
            out.append((float(v[0]), str(v[1])))
        else:
            out.append((float(v), None))

    return out
