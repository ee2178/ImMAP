"""
Shared engine for the figure tooling: config, dump loading, display.

Everything here reads HDF5 written by `scripts/dump_eval.py` and nothing else.
**No torch, no models, no datasets.** That is deliberate and load-bearing: the
viewer has to start in under a second and run on a laptop against an rsync'd
copy of the cluster's output, and importing `models` would drag in CUDA, flex
attention and a 30-second import for no benefit.

The other rule worth knowing is the one `visualization/image.py` already
states: panels that are compared must share a display window. Every function
here that computes a window takes the REFERENCE as its argument and the caller
applies the result to all columns, so a reconstruction that over- or
under-shoots cannot be silently rescaled into looking correct.

Config
------
A figure variant is a Python file under `configs/` setting any of the names in
`CONFIGURABLE`. The filename is the VARIANT, which is what the hand-picked
rows and zooms are keyed on:

    rows/<VARIANT>.json    which volume/slice each row shows   (viewer.py)
    zooms/<VARIANT>.json   where each row's zoom box sits      (viewer.py)

Two files, not one, so re-picking rows never discards zoom work. An unknown
name in a config is an error rather than a silent no-op -- a typo'd option that
did nothing would be indistinguishable from an option that did not help.
"""

from __future__ import annotations

import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


# ---------------------------------------------------------------------------
# DEFAULTS -- every name in CONFIGURABLE may be overridden by a config file
# ---------------------------------------------------------------------------

# Where the dumps live. A column spec is resolved against this; see column_dir.
DUMP_ROOT = os.path.join(ROOT, "trained_nets", "mg_recon", "brain")

# (column label, dump spec). `None` = the ground truth, read from whichever
# column comes first -- every dump stores the identical reference, so which one
# supplies it matters only if they came from different code.
COLUMNS = [
    ("Zero-filled",  "@zero_filled"),
    ("LPDSNet",      "lpdsnet_R8"),
    ("MG-LPDS",      "mglpds_R8"),
    ("MG-GroupLPDS", "mggrouplpds_R8"),
    ("Ground truth", None),
]

# (volume, slice). `slice` is the index into the DUMPED stack's `slice_index`,
# i.e. the real slice number in the source volume. Hand-picked rows in
# rows/<VARIANT>.json override this.
ROWS = []

ZOOM_W, ZOOM_H = 48, 48   # zoom window, in source pixels
ZOOM_UP = 8               # nearest-neighbour upsample for the inset
WINDOW_PCT = 99.9         # display window ceiling: percentile of the reference
WINDOW_SCALE = 0.85       # ... scaled by this
BRIGHTNESS = 1.0          # overall gain; >1 brighter. Linear, so it eventually
                          # clips highlights -- reach for GAMMA first if only
                          # the dark tissue is too dim.
GAMMA = 0.85              # display gamma, <1 lifts midtones without clipping
RESID_GAIN = 5.0          # residual panels are windowed at vmax / RESID_GAIN
RESID_CMAP = "inferno"    # residual colour scheme; None = grayscale
CROP_FOV = True           # crop every column to one identical square FOV box
CROP_PAD = 6              # padding around the foreground bounding box, px
MASK_THRESH = 0.0         # anatomy mask is reference > THRESH * peak
STATUS_METRIC = "psnr"    # which stored per-slice metric the viewer reports
NAME = "default"          # output identity
VARIANT = None            # input identity for rows/ and zooms/; None follows NAME

CONFIGURABLE = (
    "NAME", "DUMP_ROOT", "COLUMNS", "ROWS", "ZOOM_W", "ZOOM_H", "ZOOM_UP",
    "WINDOW_PCT", "WINDOW_SCALE", "BRIGHTNESS", "GAMMA", "RESID_GAIN",
    "RESID_CMAP", "CROP_FOV", "CROP_PAD", "MASK_THRESH", "STATUS_METRIC",
)


def apply_config(path):
    """Execute a config file and copy its settings over the defaults."""
    ns = {}
    with open(path) as f:
        exec(compile(f.read(), path, "exec"), ns)
    unknown = sorted(k for k in ns
                     if k.isupper() and not k.startswith("_")
                     and k not in CONFIGURABLE)
    if unknown:
        raise SystemExit(f"{path}: unknown option(s) {unknown}\n"
                         f"  known: {', '.join(sorted(CONFIGURABLE))}")
    for k in CONFIGURABLE:
        if k in ns:
            globals()[k] = ns[k]
    stem = os.path.splitext(os.path.basename(path))[0]
    if "NAME" not in ns:
        globals()["NAME"] = stem
    # rows/ and zooms/ key off the CONFIG, never off a --name override, so a
    # renamed rebuild still uses that config's hand-picked selection.
    globals()["VARIANT"] = ns.get("NAME", stem)


def variant():
    return VARIANT or NAME


def outdir():
    return os.path.join(HERE, "out", NAME)


def rows_file():
    return os.path.join(HERE, "rows", f"{variant()}.json")


def zooms_file():
    return os.path.join(HERE, "zooms", f"{variant()}.json")


def row_tag(row):
    return f"{row['volume']}_s{row['slice']}"


def load_zooms():
    if not os.path.exists(zooms_file()):
        return {}
    with open(zooms_file()) as f:
        return {k: tuple(v) for k, v in json.load(f).items()}


def save_zooms(zooms):
    os.makedirs(os.path.dirname(zooms_file()), exist_ok=True)
    with open(zooms_file(), "w") as f:
        json.dump({k: list(v) for k, v in zooms.items()}, f, indent=2)


def active_rows():
    """The variant's rows: a hand-picked selection if one exists, else the config's."""
    if os.path.exists(rows_file()):
        with open(rows_file()) as f:
            return json.load(f)
    return list(ROWS)


def save_rows(rows):
    os.makedirs(os.path.dirname(rows_file()), exist_ok=True)
    with open(rows_file(), "w") as f:
        json.dump(rows, f, indent=2)


# ---------------------------------------------------------------------------
# Dump discovery
# ---------------------------------------------------------------------------
# A column spec names one of three things:
#
#   "mglpds_R8"        a run under DUMP_ROOT; its dump is <run>/eval_dump/
#   "mglpds_R8:recon"  ... showing a named dataset instead of `recon`
#   "@zero_filled"     a dataset every dump carries, taken from the FIRST named
#                      column. The zero-filled adjoint is not any one method's
#                      output, so attaching it to a particular run would be
#                      arbitrary and would break when that run is dropped.
#   None               the reference, likewise taken from the first named column.

def col_spec(spec):
    """`"mglpds_R8:sense"` -> `("mglpds_R8", "sense")`; default dataset is `recon`."""
    if spec is None:
        return None, "reference"
    if spec.startswith("@"):
        return None, spec[1:]
    run, _, ds = spec.partition(":")
    return run, (ds or "recon")


def _first_named():
    for _, spec in COLUMNS:
        run, _ = col_spec(spec)
        if run is not None:
            return run
    raise SystemExit("COLUMNS names no run -- every column is a shared dataset")


def column_dir(spec):
    """Directory of HDF5s for a column spec.

    Accepts both dump layouts: the default `<run>/eval_dump/` and the mirrored
    tree that `dump_eval.py --out` writes, where the run directory IS the dump
    directory.
    """
    run, _ = col_spec(spec)
    if run is None:
        run = _first_named()
    for cand in (os.path.join(DUMP_ROOT, run, "eval_dump"),
                 os.path.join(DUMP_ROOT, run)):
        if os.path.isdir(cand) and any(f.endswith(".h5") for f in os.listdir(cand)):
            return cand
    raise SystemExit(
        f"no dump for column {spec!r}: looked in\n"
        f"  {os.path.join(DUMP_ROOT, run, 'eval_dump')}\n"
        f"  {os.path.join(DUMP_ROOT, run)}\n"
        f"Run scripts/dump_eval.py for that run first.")


def volumes():
    """Volume names present in EVERY column, sorted.

    The intersection, not the union: a row must be drawable in every column or
    the figure would have a hole in it.
    """
    common = None
    for _, spec in COLUMNS:
        if col_spec(spec)[0] is None:
            continue
        here = {f[:-3] for f in os.listdir(column_dir(spec)) if f.endswith(".h5")}
        common = here if common is None else common & here
    return sorted(common or [])


class Volume:
    """One dumped volume for one column: the stacks, the stored metrics, the attrs."""

    def __init__(self, path, dataset):
        import h5py

        with h5py.File(path, "r") as f:
            if dataset not in f:
                raise SystemExit(
                    f"{os.path.basename(path)} has no dataset {dataset!r}.\n"
                    f"  available: {', '.join(sorted(f))}\n"
                    f"  Re-run scripts/dump_eval.py if this dump predates it.")
            self.attrs = {k: _scalar(v) for k, v in f.attrs.items()}
            self.img = np.abs(f[dataset][:]).astype(np.float32)
            self.ref = np.abs(f["reference"][:]).astype(np.float32)
            self.organ = (f["organ_mask"][:].astype(bool)
                          if "organ_mask" in f else None)
            self.slice_index = (f["slice_index"][:].astype(int)
                                if "slice_index" in f else
                                np.arange(self.img.shape[0]))
            self.metrics = {k[:-6]: f[k][:].astype(float)
                            for k in f if k.endswith("_slice")}

    def __len__(self):
        return self.img.shape[0]

    def at(self, i):
        """Position of source slice number `i` in the stack, or None."""
        hit = np.where(self.slice_index == i)[0]
        return int(hit[0]) if len(hit) else None


def _scalar(v):
    """HDF5 attributes come back as 0-d arrays and bytes; make them Python."""
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, np.ndarray):
        return v.tolist() if v.ndim else v.item()
    if isinstance(v, np.generic):
        return v.item()
    return v


_CACHE, _CACHE_MAX = {}, 12


def load(spec, volume):
    """`Volume` for one column and one volume name, memoised."""
    key = (spec, volume)
    if key not in _CACHE:
        if len(_CACHE) >= _CACHE_MAX:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = Volume(os.path.join(column_dir(spec), f"{volume}.h5"),
                             col_spec(spec)[1])
    return _CACHE[key]


# Attributes that must agree across the columns of one figure. Sigma, R and the
# seed change what the network was asked to do; anatomy changes who is in the
# picture. A figure whose columns disagree on any of them is not a comparison.
COMPARABLE = ("sigma", "R", "seed", "anatomy", "acs_lines", "mask_dist")


def check_comparable(volume):
    """Warnings (as strings) for columns that are not comparable to the first."""
    named = [(lab, spec) for lab, spec in COLUMNS if col_spec(spec)[0] is not None]
    if not named:
        return []
    base = load(named[0][1], volume).attrs
    out = []
    for lab, spec in named[1:]:
        a = load(spec, volume).attrs
        bad = [k for k in COMPARABLE if k in base and k in a and base[k] != a[k]]
        if bad:
            out.append(f"{lab}: " + ", ".join(
                f"{k}={a[k]} vs {base[k]}" for k in bad))
    # The organ-mask policy does not invalidate the IMAGES, only the numbers,
    # so it is reported separately rather than as a mismatch.
    if len({load(s, volume).attrs.get("use_organ_mask") for _, s in named}) > 1:
        out.append("columns disagree on use_organ_mask -- their metrics are "
                   "averaged over different regions and cannot share a table")
    return out


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
def window(ref, mask=None):
    """The one display ceiling every column of a row shares."""
    v = ref[mask] if mask is not None and mask.any() else ref
    return float(np.percentile(v, WINDOW_PCT)) * WINDOW_SCALE / max(BRIGHTNESS, 1e-6)


def to8(a, vmax, gamma=None):
    """Magnitude -> uint8, windowed at `vmax` and gamma-corrected."""
    gamma = GAMMA if gamma is None else gamma
    return (np.clip(a / max(vmax, 1e-12), 0, 1) ** gamma * 255 + 0.5).astype(np.uint8)


_LUTS = {}


def apply_cmap(a, name):
    """Map [0,1] to RGB uint8 through a matplotlib colormap, as a 256-entry LUT."""
    if name not in _LUTS:
        import matplotlib
        # `matplotlib.colormaps` since 3.5; `cm.get_cmap` was removed in 3.9,
        # so reaching for it first would break on a current matplotlib and
        # work on the older one -- exactly backwards.
        try:
            cmap = matplotlib.colormaps[name]
        except AttributeError:                       # matplotlib < 3.5
            from matplotlib import cm
            cmap = cm.get_cmap(name)
        _LUTS[name] = (cmap(np.linspace(0, 1, 256))[:, :3] * 255
                       + 0.5).astype(np.uint8)
    lut = _LUTS[name]
    return lut[np.clip(a * 255, 0, 255).astype(np.uint8)]


def anatomy_mask(ref):
    """Support of the reference -- what a mask-averaged metric averages over."""
    return ref > MASK_THRESH * ref.max()


def fov_box(ref, pad=None):
    """Square bounding box of the anatomy, as (c0, r0, size).

    One box per row, applied identically to every column, so cropping cannot
    introduce a misalignment between panels.
    """
    pad = CROP_PAD if pad is None else pad
    m = ref > 0.05 * ref.max()
    if not m.any():
        return 0, 0, min(ref.shape)
    rows, cols = np.where(m)
    r0, r1, c0, c1 = rows.min(), rows.max(), cols.min(), cols.max()
    n = min(ref.shape)
    size = min(n, int(max(r1 - r0, c1 - c0)) + 1 + 2 * pad)
    rc, cc = (r0 + r1) // 2, (c0 + c1) // 2
    return (int(np.clip(cc - size // 2, 0, ref.shape[1] - size)),
            int(np.clip(rc - size // 2, 0, ref.shape[0] - size)),
            size)


def disagreement(ref, per_col):
    """Spread of the absolute error across methods -- where the columns differ.

    max - min rather than a variance: what a picked slice needs to show is that
    SOME method fails where another does not, and the spread says exactly that,
    while a variance is dominated by however many near-identical columns happen
    to be present.
    """
    if not per_col:
        return np.zeros_like(ref)
    errs = np.stack([np.abs(ref - x) for x in per_col.values()])
    return errs.max(0) - errs.min(0)


def _box_sum(a, h, w):
    """Sliding-window sums over every h x w window, via an integral image."""
    ii = np.pad(a, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    return ii[h:, w:] - ii[:-h, w:] - ii[h:, :-w] + ii[:-h, :-w]


def _erode(mask, k):
    """Binary erosion by a k x k box, at full size."""
    if k < 2:
        return mask
    s = _box_sum(mask.astype(np.float64), k, k) >= k * k - 0.5
    out = np.zeros(mask.shape, bool)
    o = k // 2
    out[o:o + s.shape[0], o:o + s.shape[1]] = s
    return out


def interior(ref, W, frac):
    """Anatomy silhouette eroded inward by `frac` of the frame.

    The skull rim is the brightest structure and dominates any absolute-error
    map, but it is not what the reconstruction is judged on. Eroding it keeps
    the automatic zoom over parenchyma. The threshold is deliberately low: this
    is the silhouette, not a tissue mask, and requiring bright tissue would
    reject the dark interior and leave nothing admissible.
    """
    return _erode(ref > 0.06 * ref.max(), max(2, int(frac * W)))


def auto_zoom(ref, energy, W, margin=8):
    """Window maximising `energy`, restricted to windows sitting on the anatomy.

    Attempts run strictest to loosest: interior, then plain foreground, then
    anywhere. Small cross-sections can exclude everything under the strict
    rules, hence the fallbacks.
    """
    zh, zw = min(ZOOM_H, W), min(ZOOM_W, W)
    base = _box_sum(energy, zh, zw)
    area = zh * zw
    fg = ref > 0.12 * ref.max()
    attempts = ((interior(ref, W, 0.05), 0.98), (fg, 0.85), (fg, 0.5),
                (np.ones_like(ref, bool), 0.0))

    for mask, cover in attempts:
        e = base.copy()
        e[_box_sum(mask.astype(np.float64), zh, zw) < cover * area] = -np.inf
        m = min(margin, max(0, (min(e.shape) - 1) // 2))
        if m:
            e[:m, :] = e[-m:, :] = -np.inf
            e[:, :m] = e[:, -m:] = -np.inf
        if np.isfinite(e).any():
            r, c = np.unravel_index(np.argmax(e), e.shape)
            return int(c), int(r)
    return 0, 0


def nrmse(ref, x, mask=None):
    """Live NRMSE, for arrays the dump has no stored metric for (e.g. zero-filled).

    Stored `*_slice` values are preferred wherever they exist: they came from
    `evaluation/metrics.py` on the GPU, so they match the campaign tables by
    construction, and this does not.
    """
    if mask is not None:
        d = (((ref - x) ** 2) * mask).sum()
        n = ((ref ** 2) * mask).sum()
    else:
        d, n = ((ref - x) ** 2).sum(), (ref ** 2).sum()
    return float(np.sqrt(d / max(n, 1e-30)))


def pretty(name):
    """`file_brain_AXT2_201_2010294` -> something that fits a status line."""
    return re.sub(r"^file_", "", name)
