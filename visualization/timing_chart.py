"""
The figure for `scripts/time_net.py`: one upright column per network, stacked.

    bottom   forward     the train-mode forward pass
    top      backward    loss + backward pass

The column's height is forward + backward: the network's share of one training
step, which is what Sljiva's `loss_time` wraps (its `fwdtime` is the forward
alone). Nothing else is drawn -- the inference forward, the optimizer and the
projection stay in the printed table and the JSON. Columns keep the order the
configs were given in.

Colours are the two leading categorical slots of the dataviz reference palette,
validated for both themes: blue = forward, orange = backward. Text is never
coloured; identity comes from the legend swatches. Each segment carries its
value where it fits, and the total sits on top of the column.

matplotlib only, imported lazily by the caller. Columns are drawn in pixel
space so the 4 px rounded top and the 2 px gap between the two segments are
exact at any figure size.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.patches import PathPatch, Rectangle               # noqa: E402
from matplotlib.path import Path                                  # noqa: E402
from matplotlib.ticker import MaxNLocator                         # noqa: E402
from matplotlib.transforms import IdentityTransform               # noqa: E402

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7",
                  forward="#2a78d6", backward="#eb6834"),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835",
                 forward="#3987e5", backward="#d95926"),
}
FONT = ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"]


def _font():
    """The first of FONT that is installed (asking for a missing one logs a
    findfont warning per text object)."""
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    return next((f for f in FONT if f in have), "DejaVu Sans")


DPI = 200
PX = DPI / 96.0                      # one CSS px in figure pixels


def _lum(hex_):
    c = [int(hex_[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    c = [v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4 for v in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def _on(fill):
    """White or ink for text set INSIDE a fill, whichever contrasts more."""
    lf = _lum(fill)
    return "#ffffff" if (1.05) / (lf + 0.05) >= (lf + 0.05) / 0.05 else "#0b0b0b"


def _column(x0, x1, y0, y1, round_top):
    """Column outline in pixels: square at the bottom, rounded at the top."""
    r = min(4 * PX, (x1 - x0) / 2, (y1 - y0) / 2) if round_top else 0.0
    if r <= 0:
        return Path([(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)],
                    [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY])
    v = [(x0, y0), (x1, y0), (x1, y1 - r), (x1, y1), (x1 - r, y1), (x0 + r, y1),
         (x0, y1), (x0, y1 - r), (x0, y0)]
    c = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO,
         Path.CURVE3, Path.CURVE3, Path.CLOSEPOLY]
    return Path(v, c)


def _arch_lines(r):
    """Two muted lines under a network's name: its depth, then width and size."""
    K = r.get("K")
    if isinstance(K, (list, tuple)) and len(K) == 2 and isinstance(K[1], (list, tuple)):
        k = f"{K[0]} V-cycle{'s' if K[0] != 1 else ''} × [{', '.join(str(i) for i in K[1])}]"
    elif isinstance(K, int):
        k = f"K = {K}"
    elif isinstance(K, str) and K.endswith("casc"):
        k = f"{K[:-4]} cascades"
    else:
        k = str(K) if K is not None else ""
    size = f"{r['params'] / 1e6:.2f}M params"
    return k, (f"M = {r['M']}  ·  {size}" if r.get("M") else size)


def _split_common_suffix(names):
    """`(short names, "R = 12")` when every name ends in the same `_R<n>`."""
    import re
    tails = {m.group(1) for m in (re.search(r"_R(\d+)$", nm) for nm in names) if m}
    if len(tails) == 1 and all(re.search(r"_R\d+$", nm) for nm in names):
        r = tails.pop()
        return [nm[: -len(r) - 2] for nm in names], f"R = {r}"
    return list(names), None


def save_timing_figure(rows, meta, path, theme="light"):
    """Render `rows` (time_net.py records) to `path` (.png / .pdf / .svg)."""
    t = THEMES[theme]
    plt.rcParams.update({"font.family": _font(), "font.size": 9})
    n = len(rows)
    names, r_note = _split_common_suffix([r["name"] for r in rows])

    med = lambda r, k: float(r[k]["median"])                       # noqa: E731
    segs = [[("forward", med(r, "forward")), ("backward", med(r, "backward"))]
            for r in rows]
    totals = [sum(v for _, v in s) for s in segs]

    # ---- layout, in inches -------------------------------------------------
    col_w, left, right = 1.50, 0.80, 0.35
    top, plot_h, bottom = 1.35, 3.60, 1.05
    W = max(left + n * col_w + right, 7.5)                         # room for the title
    fig_h = top + plot_h + bottom
    fig = plt.figure(figsize=(W, fig_h), dpi=DPI, facecolor=t["surface"])
    ax = fig.add_axes([left / W, bottom / fig_h, n * col_w / W, plot_h / fig_h])

    ax.set_facecolor(t["surface"])
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0, max(totals) * 1.10)
    ax.set_xticks([])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
    ax.tick_params(axis="y", length=0, pad=6, labelsize=8, labelcolor=t["muted"])
    ax.grid(axis="y", color=t["grid"], linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.spines["bottom"].set_linewidth(0.9)
    ax.set_ylabel("milliseconds", fontsize=8, color=t["muted"], labelpad=8)

    fig.canvas.draw()                                              # freeze transforms
    renderer = fig.canvas.get_renderer()
    bar_w, gap = 58 * PX, 2 * PX
    fw, fh = fig.bbox.width, fig.bbox.height

    def text_h(size):
        tx = fig.text(0, 0, "0", fontsize=size)
        h = tx.get_window_extent(renderer).height
        tx.remove()
        return h

    inner_h = text_h(8)

    for i, r in enumerate(rows):
        xc = ax.transData.transform((i, 0))[0]
        y_px = lambda v: ax.transData.transform((i, v))[1]         # noqa: E731
        acc = 0.0
        for j, (key, v) in enumerate(segs[i]):
            a, b = y_px(acc), y_px(acc + v)
            last = j == len(segs[i]) - 1
            a2 = a + (gap / 2 if j else 0.0)
            b2 = b - (0.0 if last else gap / 2)
            if b2 > a2:
                fig.add_artist(PathPatch(_column(xc - bar_w / 2, xc + bar_w / 2, a2, b2, last),
                                         transform=IdentityTransform(), facecolor=t[key],
                                         edgecolor="none", zorder=3))
                # the segment's own value, only where it fits with padding
                if inner_h + 8 * PX <= b2 - a2:
                    fig.text(xc / fw, (a2 + b2) / 2 / fh, f"{v:.1f}", ha="center",
                             va="center_baseline", fontsize=8, color=_on(t[key]), zorder=4)
            acc += v
        fig.text(xc / fw, (y_px(acc) + 5 * PX) / fh, f"{totals[i]:.1f}", ha="center",
                 va="bottom", fontsize=9, color=t["ink"], fontweight="semibold")

        # column label: name in primary ink, architecture in muted ink
        y0 = bottom / fig_h
        k_line, size_line = _arch_lines(r)
        fig.text(xc / fw, y0 - 0.24 / fig_h, names[i], ha="center", va="center",
                 fontsize=10, color=t["ink"], fontweight="semibold")
        fig.text(xc / fw, y0 - 0.46 / fig_h, k_line, ha="center", va="center",
                 fontsize=7.5, color=t["muted"])
        fig.text(xc / fw, y0 - 0.64 / fig_h, size_line, ha="center", va="center",
                 fontsize=7.5, color=t["muted"])

    # ---- title, legend -------------------------------------------------------
    size = meta.get("size") or [0, 0]
    title = (f"Network speed  ·  {size[0]}×{size[1]}, {meta.get('coils', '?')} coils"
             + (f", {r_note}" if r_note else ""))
    sub = "  ·  ".join(str(s) for s in (
        meta.get("device_name") or meta.get("device"),
        f"torch {meta['torch']}" if meta.get("torch") else None,
        f"median of {meta['reps']} runs after {meta['warmup']} warm-up" if meta.get("reps") else None,
        ("GPU-synchronised around the network only"
         if str(meta.get("device", "cuda")).startswith("cuda")
         else "timed around the network only")) if s)
    x_left = 0.30 / W
    fig.text(x_left, 1 - 0.36 / fig_h, title, ha="left", va="center", fontsize=14,
             color=t["ink"], fontweight="bold")
    fig.text(x_left, 1 - 0.66 / fig_h, sub, ha="left", va="center", fontsize=8.5,
             color=t["ink2"])

    # legend: swatches + secondary ink, listed top segment first, like the stack
    x = x_left
    y = 1 - 1.02 / fig_h
    sw = 9 * PX / fw
    for key, lab in (("backward", "backward (loss + gradient)"), ("forward", "forward")):
        fig.add_artist(Rectangle((x, y - sw * W / fig_h / 2), sw, sw * W / fig_h,
                                 transform=fig.transFigure, facecolor=t[key],
                                 edgecolor="none"))
        x += sw + 5 * PX / fw
        tx = fig.text(x, y, lab, ha="left", va="center", fontsize=8.5, color=t["ink2"])
        x += tx.get_window_extent(renderer).width / fw + 16 * PX / fw

    fig.savefig(path, dpi=DPI, facecolor=t["surface"])
    plt.close(fig)
    return path
