"""
The figure for `scripts/time_net.py`: one row per network, two panels.

    left    inference forward (eval mode, no autograd)      one hue, one bar
    right   one training step, stacked                      forward | backward
                                                            | optimizer + projection

Both panels are milliseconds but on SEPARATE axes -- an inference forward and a
training step are different quantities, and one shared axis would squash the
smaller. Rows keep the order the configs were given in (it is usually a grid),
so a network is found by position, not by rank.

Colours are the three leading categorical slots of the dataviz reference
palette, validated for both themes (lightness band, chroma floor, colour-vision
separation, contrast): blue = a forward pass in either panel, orange = backward,
aqua = optimizer + projection. Text is never coloured; identity comes from the
legend swatches and the value labels.

matplotlib only, imported lazily by the caller. Bars are drawn in pixel space
so the 4 px rounded data-end and the 2 px gap between stacked segments are
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
                  forward="#2a78d6", backward="#eb6834", step="#1baf7a"),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835",
                 forward="#3987e5", backward="#d95926", step="#199e70"),
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


def _bar(x0, x1, y0, y1, round_end):
    """Bar outline in pixels: square at the baseline, rounded at the data end."""
    r = min(4 * PX, (x1 - x0) / 2, (y1 - y0) / 2) if round_end else 0.0
    if r <= 0:
        return Path([(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)],
                    [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY])
    v = [(x0, y0), (x1 - r, y0), (x1, y0), (x1, y0 + r), (x1, y1 - r), (x1, y1),
         (x1 - r, y1), (x0, y1), (x0, y0)]
    c = [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CURVE3,
         Path.CURVE3, Path.LINETO, Path.CLOSEPOLY]
    return Path(v, c)


def _k_label(r):
    K = r.get("K")
    if isinstance(K, (list, tuple)) and len(K) == 2 and isinstance(K[1], (list, tuple)):
        k = f"{K[0]} V-cycle{'s' if K[0] != 1 else ''} × [{', '.join(str(i) for i in K[1])}]"
    elif isinstance(K, int):
        k = f"K = {K}"
    elif isinstance(K, str) and K.endswith("casc"):
        k = f"{K[:-4]} cascades"
    else:
        k = str(K) if K is not None else ""
    parts = [k] if k else []
    if r.get("M"):
        parts.append(f"M = {r['M']}")
    parts.append(f"{r['params'] / 1e6:.2f}M params")
    return "  ·  ".join(parts)


def _split_common_suffix(names):
    """`(short names, "R = 12")` when every name ends in the same `_R<n>`."""
    import re
    tails = {m.group(1) for m in (re.search(r"_R(\d+)$", nm) for nm in names) if m}
    if len(tails) == 1 and all(re.search(r"_R\d+$", nm) for nm in names):
        r = tails.pop()
        return [nm[: -len(r) - 2] for nm in names], f"R = {r}"
    return list(names), None


def save_timing_figure(rows, meta, path, theme="light", segment_values=False):
    """Render `rows` (time_net.py records) to `path` (.png / .pdf / .svg).

    `segment_values` also prints each stacked segment's value inside it, where
    it fits. Off by default: the total at the bar's tip is the comparison, and
    the script's printed table carries every number.
    """
    t = THEMES[theme]
    plt.rcParams.update({"font.family": _font(), "font.size": 9})
    n = len(rows)
    names, r_note = _split_common_suffix([r["name"] for r in rows])
    with_step = all("opt" in r and "project" in r for r in rows)

    med = lambda r, k: float(r[k]["median"])                       # noqa: E731
    infer = [med(r, "infer") for r in rows]
    segs = [[("forward", med(r, "forward")), ("backward", med(r, "backward"))]
            + ([("step", med(r, "opt") + med(r, "project"))] if with_step else [])
            for r in rows]
    totals = [sum(v for _, v in s) for s in segs]

    # ---- layout, in inches -------------------------------------------------
    row_h, top, bottom = 0.50, 1.55, 0.62
    W = 12.0
    fig_h = top + n * row_h + bottom
    fig = plt.figure(figsize=(W, fig_h), dpi=DPI, facecolor=t["surface"])
    label_w, gutter, right = 3.35, 0.55, 0.35
    plot_w = W - label_w - gutter - right
    wa, wb = 0.36 * plot_w, 0.64 * plot_w
    y0 = bottom / fig_h
    h = n * row_h / fig_h
    axa = fig.add_axes([label_w / W, y0, wa / W, h])
    axb = fig.add_axes([(label_w + wa + gutter) / W, y0, wb / W, h])

    for ax, vals in ((axa, infer), (axb, totals)):
        ax.set_facecolor(t["surface"])
        ax.set_xlim(0, max(vals) * 1.17)
        ax.set_ylim(n - 0.5, -0.5)                                 # first config on top
        ax.set_yticks([])
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
        ax.tick_params(axis="x", length=0, pad=6, labelsize=8, labelcolor=t["muted"])
        ax.grid(axis="x", color=t["grid"], linewidth=0.8, linestyle="-")
        ax.set_axisbelow(True)
        for s in ("top", "right", "bottom"):
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_color(t["axis"])
        ax.spines["left"].set_linewidth(0.9)
        ax.set_xlabel("milliseconds", fontsize=8, color=t["muted"], labelpad=6)

    fig.canvas.draw()                                              # freeze transforms
    renderer = fig.canvas.get_renderer()
    bar_h, gap = 19 * PX, 2 * PX

    def text_w(s, size, weight="normal"):
        tx = fig.text(0, 0, s, fontsize=size, fontweight=weight)
        w = tx.get_window_extent(renderer).width
        tx.remove()
        return w

    def draw_row(ax, i, parts):
        """parts: [(colour, value)] stacked from the baseline; returns end px."""
        x_px = lambda v: ax.transData.transform((v, i))[0]         # noqa: E731
        yc = ax.transData.transform((0, i))[1]
        acc = 0.0
        for j, (col, v) in enumerate(parts):
            a, b = x_px(acc), x_px(acc + v)
            last = j == len(parts) - 1
            a2 = a + (gap / 2 if j else 0.0)
            b2 = b - (0.0 if last else gap / 2)
            if b2 > a2:
                fig.add_artist(PathPatch(_bar(a2, b2, yc - bar_h / 2, yc + bar_h / 2, last),
                                         transform=IdentityTransform(), facecolor=col,
                                         edgecolor="none", zorder=3))
                # a value INSIDE a segment only where it fits with padding
                if segment_values and len(parts) > 1:
                    lab = f"{v:.0f}"
                    if text_w(lab, 7.5) + 12 * PX <= b2 - a2:
                        fig.text((a2 + b2) / 2 / fig.bbox.width, yc / fig.bbox.height, lab,
                                 ha="center", va="center_baseline", fontsize=7.5,
                                 color=_on(col), zorder=4)
            acc += v
        return x_px(acc), yc

    for i, r in enumerate(rows):
        # row label: name in primary ink, architecture in muted ink
        yc = axa.transData.transform((0, i))[1] / fig.bbox.height
        x_lab = 0.30 / W
        fig.text(x_lab, yc + 0.085 / fig_h, names[i], ha="left", va="center",
                 fontsize=10, color=t["ink"], fontweight="semibold")
        fig.text(x_lab, yc - 0.105 / fig_h, _k_label(r), ha="left", va="center",
                 fontsize=7.5, color=t["muted"])

        xe, ypx = draw_row(axa, i, [(t["forward"], infer[i])])
        fig.text((xe + 6 * PX) / fig.bbox.width, ypx / fig.bbox.height, f"{infer[i]:.1f}",
                 ha="left", va="center_baseline", fontsize=8.5, color=t["ink"])
        xe, ypx = draw_row(axb, i, [(t[k], v) for k, v in segs[i]])
        fig.text((xe + 6 * PX) / fig.bbox.width, ypx / fig.bbox.height, f"{totals[i]:.1f}",
                 ha="left", va="center_baseline", fontsize=8.5, color=t["ink"],
                 fontweight="semibold")

    # ---- titles, legend ------------------------------------------------------
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
    fig.text(0.30 / W, 1 - 0.36 / fig_h, title, ha="left", va="center", fontsize=14,
             color=t["ink"], fontweight="bold")
    fig.text(0.30 / W, 1 - 0.66 / fig_h, sub, ha="left", va="center", fontsize=8.5,
             color=t["ink2"])

    def panel_head(ax, head, note):
        x = ax.get_position().x0
        fig.text(x, 1 - 1.08 / fig_h, head, ha="left", va="center", fontsize=10.5,
                 color=t["ink"], fontweight="semibold")
        fig.text(x, 1 - 1.30 / fig_h, note, ha="left", va="center", fontsize=8,
                 color=t["muted"])

    panel_head(axa, "Inference forward", "eval mode, no gradients")
    panel_head(axb, "Training step", "the network's share: no data loading, no map estimate")

    # legend for the stacked panel (>= 2 series): swatches + secondary ink
    keys = [("forward", "forward"), ("backward", "loss + backward")]
    if with_step:
        keys.append(("step", "optimizer + projection"))
    x = axb.get_position().x1
    y = 1 - 1.08 / fig_h
    sw = 9 * PX / fig.bbox.width
    for key, lab in reversed(keys):
        tx = fig.text(x, y, lab, ha="right", va="center", fontsize=8.5, color=t["ink2"])
        x -= tx.get_window_extent(renderer).width / fig.bbox.width + 5 * PX / fig.bbox.width + sw
        fig.add_artist(Rectangle((x, y - sw * W / fig_h / 2), sw, sw * W / fig_h,
                                 transform=fig.transFigure, facecolor=t[key],
                                 edgecolor="none"))
        x -= 14 * PX / fig.bbox.width

    fig.savefig(path, dpi=DPI, facecolor=t["surface"])
    plt.close(fig)
    return path
