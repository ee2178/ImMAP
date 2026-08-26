"""
Resampling operators and the multigrid restriction / prolongation pair.

Port of the `Resample` / `galerkin` / `meanpool` / `upsample` section of
Sljiva's `src/operators.jl`, in PyTorch (B, C, H, W) layout.

Two things live here:

  * `restrict` / `prolong` -- the grid-transfer pair used by the V-cycle.
    Both are built from ONE kernel `h`:

        R z = decimate_f( h * z )                (filter, then subsample)
        P w = f^2 . h *^T zero_insert_f( w )     (zero-insert, then filter)

    so that `P == f^2 . R^T` holds EXACTLY, as a matter of layout rather than
    of numerical luck -- `F.conv_transpose2d` is defined as the transpose of
    `F.conv2d`, and with `groups=C` both want the same `(C, 1, L, L)` weight.
    The `f^2` makes `P` preserve constants; `R` preserves them because `h`
    sums to one.  This holds for EVERY filter below, so switching filters
    varies the anti-aliasing and nothing else -- which is what makes the
    ablation clean.

  * `Resample` / `galerkin` -- the *operator-level* coarsening.  The coarse
    forward model is  E_c = E . P  (P = prolongation), so the coarse Gram is
    R E^H E P = (1/f^2) P^H E^H E P: a coarse latent is lifted to the fine
    grid, sees the true physics, and comes back.  Because R is the exact
    transpose of P, that Gram is Hermitian PSD -- i.e. the coarse level really
    is minimising `1/2 ||E P z_c - y||^2`, which is what the FAS correction
    assumes.  No coarse measurement model is ever needed.

    `galerkin(Identity()) -> Identity()` is the one optimisation that matters
    most in practice: for denoising the coarse Gram is the identity, so the
    whole V-cycle runs without a single resampling round-trip.

Choosing a filter
-----------------
The restriction is an anti-aliasing filter: everything it leaves above the
coarse Nyquist (pi/2) folds onto the coarse grid, generates a spurious FAS
correction `pi`, and gets prolonged back into the fine iterate as artifact.
But it must not eat the passband either -- whatever it removes below pi/2 is
detail the coarse grid could have represented and the coarse level now cannot
see.  Measured `|H(w)|` for the 1-D kernel of each option, at factor 2:

    filter       L    0.25pi  0.40pi  0.50pi  0.60pi  0.75pi   sigma_c/sigma
    none         2     1.000   1.000   1.000   1.000   1.000      1.0000
    box          2     0.924   0.809   0.707   0.588   0.383      0.5000
    bilinear     4     0.789   0.530   0.354   0.203   0.056      0.3125
    bicubic      8     0.936   0.709   0.486   0.272   0.067      0.4044
    gaussian     8     0.735   0.454   0.291   0.169   0.062      0.2821
    sinc         8     0.957   0.732   0.498   0.267   0.046      0.4162

Read the middle columns as passband (want ~1) and the right ones as stopband
(want ~0).  `none` is pure decimation -- the no-anti-aliasing extreme.  `box`
is the 2x2 mean this module used to use, and it cannot be improved on at its
length: minimising `|H(pi/2)| / |H(0)|` over ALL `[a, b]` gives `1/sqrt(2)`,
attained at `a = b = 1/2`.  The box is the *optimal* 2-tap filter and is still
only -3 dB at the coarse Nyquist; overlapping taps are the only way to buy
anything.  `bilinear` and `gaussian` buy the stopband by giving up half the
passband.  `sinc` and `bicubic` are the two that are flat where the coarse
grid can represent the signal and small where it cannot -- `sinc` is what
`deepinv.physics.Downsampling` uses and what the Reconstruct Anything Model
composes with its forward operator, and it is the default here.

`legacy` is the odd one out and is NOT a kernel: it restores this module's
pre-existing behaviour, a 2x2 mean-pool restriction paired with bilinear
interpolation for prolongation.  Those two are not a transpose pair -- bilinear
upsampling by 2 (`align_corners=False`) is exactly zero-insertion followed by
the separable `[1,3,3,1]/8` filter, i.e. the `bilinear` row above, so its
transpose is that filter and not the box.  They differ by 22%, and the
resulting coarse Gram comes out 14.5% non-Hermitian, which makes `dF_coarse`
the subgradient of no objective at all.  Keep it only as the "what we had
before" row of an ablation table; `Resample.is_adjoint` is False for it.

Selecting one
-------------
Three ways, in increasing order of scope:

    restrict(x, filter="box")                  # one call
    with use_filter("box"): ...                # a block
    MGCDLNet(..., transfer_filter="box")       # a model, and its config JSON

A raw `torch.Tensor` also works anywhere a name does, for a custom kernel.

Padding
-------
Circular, with the matching fold on the transpose, so `R` and `P` reproduce
constants EXACTLY (no darkened border) and stay exact transposes.  Circular is
also the honest choice for the image-grid path: the encoding operator is an
FFT, so the fine problem is already circulant.

Notes on fidelity to the Julia source
-------------------------------------
`julia_compat` used to force a `(1,0,1,0)` pre-pad before a 2x2 mean-pool, to
reproduce a half-cell shift in `mg_lista.jl` that only ever fired on odd
inputs.  Every level is even by construction (`MGCDLNet.pad_stride`), so it
never fired here either.  It is honoured under `filter="legacy"` and ignored
otherwise, so existing configs keep loading.
"""

from __future__ import annotations

import contextlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from operators.base import Operator
from operators.identity import Identity

DEFAULT_FACTOR = 2
DEFAULT_FILTER = "sinc"

#: Every selectable filter.  `legacy` is a code path, not a kernel.
TRANSFER_FILTERS = ("sinc", "sinc_unwindowed", "bicubic", "bilinear",
                    "gaussian", "box", "none", "legacy")

_default_filter = DEFAULT_FILTER


def default_filter():
    """The filter used when a call passes `filter=None`."""
    return _default_filter


def set_default_filter(name):
    """Set the process-wide default.  Prefer `use_filter` or a model kwarg."""
    global _default_filter
    _default_filter = _check_filter(name)
    return _default_filter


@contextlib.contextmanager
def use_filter(name):
    """Scope the default filter to a block -- the ablation-loop entry point.

        for f in TRANSFER_FILTERS:
            with use_filter(f):
                run(build_model())

    Note this affects models *built* inside the block only if they read the
    default at construction time (they do -- see `MGCDLNet.transfer_filter`),
    and calls made inside it that pass `filter=None`.  A model built with an
    explicit `transfer_filter` ignores this.
    """
    global _default_filter
    prev = _default_filter
    _default_filter = _check_filter(name)
    try:
        yield _default_filter
    finally:
        _default_filter = prev


#: Suffix selecting the symmetric normalisation, e.g. "box-unit".
UNIT_SUFFIX = "-unit"


def split_normalisation(name):
    """`("box-unit", ...)` -> `("box", "unit")`; a bare name -> `(name, "dc")`.

    Two ways to split the `factor^d` between `R` and `P`, both of which keep
    the round trip `P R` preserving constants:

      "dc"    sum(h) = 1,            c = factor^d   (= 4 at factor 2, 2-D)
              Each operator preserves constants on its own: `R(1) = P(1) = 1`.
              A restricted iterate therefore sits at the same POINTWISE
              amplitude as the fine one, which is what `preload_with_widening`
              assumes when it copies the fine prox thresholds down.

      "unit"  sum(h) = factor^(d/2), c = 1
              `R = P^T` exactly, and `||P||_2 ~ 1`. The `factor^d` is split
              evenly instead of being carried entirely by `P`, so `R(1) = 2`
              and `P(1) = 0.5` at factor 2. Cleaner operator algebra --
              `Resample.adjoint` becomes a true adjoint rather than
              `1/factor^d` of one -- at the cost of putting the coarse iterate
              on a different scale from the fine one.

    The coarse Gram `R E^H E P` is IDENTICAL under both (the gains cancel in
    the product), so this changes the coarse problem's variables, not the
    coarse problem. What it does reach is anything nonlinear in the coarse
    iterate -- the prox thresholds -- and `sigma_c = sigma . ||h||_2`, which
    tracks the kernel automatically.
    """
    if torch.is_tensor(name) or name is None:
        return name, "dc"
    if name.endswith(UNIT_SUFFIX):
        return name[:-len(UNIT_SUFFIX)], "unit"
    return name, "dc"


def _norm_gains(norm, factor):
    """`(gamma, c)`: the kernel is scaled by `gamma`, and `P = c . R^T`.

    `gamma^2 . c == factor^d` always, which is what makes `P R` preserve
    constants either way. At `d = 2`, `gamma = factor` gives `c = 1`.
    """
    if norm == "unit":
        return float(factor), 1.0
    return 1.0, float(factor) ** 2


def adjoint_scale(filter=None, factor=DEFAULT_FACTOR):
    """The `c` in `prolong == c . restrict^T`."""
    filter = _check_filter(filter)
    _, norm = split_normalisation(filter)
    return _norm_gains(norm, factor)[1]


def _check_filter(name):
    if torch.is_tensor(name):
        return name
    if name is None:
        return _default_filter
    base, norm = split_normalisation(name)
    if base not in TRANSFER_FILTERS:
        raise ValueError(
            f"unknown transfer filter {name!r}; expected one of "
            f"{', '.join(TRANSFER_FILTERS)} (optionally suffixed "
            f"{UNIT_SUFFIX!r} for the symmetric normalisation), or a "
            f"torch.Tensor kernel.")
    if norm == "unit" and base == "legacy":
        raise ValueError(
            f"'legacy{UNIT_SUFFIX}' is not meaningful: 'legacy' pairs a "
            f"mean-pool restriction with bilinear interpolation, which are "
            f"not transposes at ANY normalisation, so there is no c to set "
            f"to 1. Use 'box{UNIT_SUFFIX}' for the mean-pool as a genuine "
            f"adjoint pair.")
    return name


# ---------------------------------------------------------------------------
# complex-safe wrapper (torch's conv / pad / pool are real-only)
# ---------------------------------------------------------------------------
def _apply_complex(fn, x):
    if torch.is_complex(x):
        return torch.complex(fn(x.real), fn(x.imag))
    return fn(x)


# ---------------------------------------------------------------------------
# the filters
# ---------------------------------------------------------------------------
def _taps(length, dtype, device):
    """Tap positions centred BETWEEN samples -- the cell-centred convention.

    An even `length` is what puts the kernel's centre of mass on the coarse
    cell centre.  An odd one sits it on a fine sample instead, half a cell off,
    which is why every filter here has even length for an even factor.
    """
    return torch.arange(length, dtype=dtype, device=device) - (length - 1) / 2


def kaiser_sinc1d(factor=2, length=None, dtype=torch.float32, device=None,
                  windowed=True):
    """Anti-aliasing sinc, optionally Kaiser-windowed, normalised to sum one.

    `deepinv.physics.functional.sinc_filter`, which is what RAM's
    `MultiScalePhysics` composes with its forward operator.  The window
    parameter follows Kaiser's design rule for a transition width
    `df = 2 (2 - sqrt 2) / factor`.

    `windowed=False` truncates the sinc with a rectangular window instead.
    That is the textbook Gibbs situation: the sidelobes of `|H(w)|` stop
    falling with length (the first stays near -21 dB no matter how long the
    kernel), the passband ripples, and the kernel's spatial tails ring.  It is
    exposed so the window itself can be ablated -- see `notebooks/
    transfer_window_ablation.ipynb` for whether it earns its keep here.
    """
    length = 4 * int(factor) if length is None else int(length)
    f = torch.sinc(_taps(length, dtype, device) / factor)
    if not windowed:
        return f / f.sum()

    A = 2.285 * (length - 1) * math.pi * (2 * (2 - math.sqrt(2)) / factor) + 7.95
    if A <= 21:
        beta = 0.0
    elif A <= 50:
        beta = 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21)
    else:
        beta = 0.1102 * (A - 8.7)
    f = f * torch.kaiser_window(length, periodic=False, beta=beta,
                                dtype=dtype, device=device)
    return f / f.sum()


def _sinc_rect1d(factor, length, dtype, device):
    """`kaiser_sinc1d` with the window switched off -- the ablation's other arm."""
    return kaiser_sinc1d(factor, length, dtype, device, windowed=False)


def _bilinear1d(factor, length, dtype, device):
    """`deepinv.physics.functional.bilinear_filter`: `[1,3,3,1]/8` at factor 2.

    Also the exact transpose of `F.interpolate(mode='bilinear')` by `factor`
    with `align_corners=False` -- so this, not the box, is the restriction that
    the old `prolong` actually belonged with.
    """
    w = (1 - (_taps(length, dtype, device) / factor).abs()).clamp(min=0)
    return w / w.sum()


def _bicubic1d(factor, length, dtype, device):
    """`deepinv.physics.functional.bicubic_filter` (Keys, a = -1/2)."""
    x = (_taps(length, dtype, device) / factor).abs()
    a = -0.5
    w = ((a + 2) * x.pow(3) - (a + 3) * x.pow(2) + 1) * (x <= 1)
    w = w + (a * x.pow(3) - 5 * a * x.pow(2) + 8 * a * x - 4 * a) * (x > 1) * (x < 2)
    return w / w.sum()


def _gaussian1d(factor, length, dtype, device):
    """Gaussian of width `factor / 2`, on the same half-sample grid.

    Not `deepinv`'s `gaussian_blur(sigma=(factor, factor))`: that one is
    odd-sized, which puts it half a cell off the coarse grid, and twice as wide,
    which costs most of the passband.
    """
    sigma = factor / 2
    w = torch.exp(-_taps(length, dtype, device) ** 2 / (2 * sigma * sigma))
    return w / w.sum()


def _box1d(factor, length, dtype, device):
    """Uniform average over one cell -- the 2x2 mean, as a kernel."""
    return torch.ones(length, dtype=dtype, device=device) / length


def _delta1d(factor, length, dtype, device):
    """Pure decimation: keep one sample per cell, filter nothing.

    `R` still preserves constants; `P` does NOT -- its transpose scatters
    `factor^2 . w` onto one fine sample per cell and zero elsewhere.  That is
    what "no prolongation filter" means, and it is the point of this option.
    """
    w = torch.zeros(length, dtype=dtype, device=device)
    w[0] = 1
    return w


_FILTER_SPECS = {
    #  name      -> (builder, length as a multiple of factor)
    "sinc":            (kaiser_sinc1d, 4),
    "sinc_unwindowed": (_sinc_rect1d,  4),
    "bicubic":  (_bicubic1d,    4),
    "gaussian": (_gaussian1d,   4),
    "bilinear": (_bilinear1d,   2),
    "box":      (_box1d,        1),
    "none":     (_delta1d,      1),
}


def filter_length(filter=None, factor=DEFAULT_FACTOR, length=None):
    """Kernel length a given filter uses at this factor."""
    filter, _ = split_normalisation(_check_filter(filter))
    if torch.is_tensor(filter):
        return filter.shape[-1]
    if filter == "legacy":
        return int(factor)
    if length is not None:
        return int(length)
    return _FILTER_SPECS[filter][1] * int(factor)


def transfer_kernel(filter=None, factor=DEFAULT_FACTOR, length=None,
                    dtype=torch.float32, device=None):
    """The separable 2-D transfer kernel, shape `(1, 1, L, L)`, summing to one.

    `filter` may be a name from `TRANSFER_FILTERS` or a raw kernel tensor
    (1-D and taken as separable, or 2-D / 4-D and taken as-is).
    """
    filter, _norm = split_normalisation(_check_filter(filter))
    gamma = _norm_gains(_norm, factor)[0]
    if torch.is_tensor(filter):
        k = filter.to(dtype=dtype, device=device)
        if k.dim() == 1:
            k = torch.outer(k, k)
        if k.dim() == 2:
            k = k[None, None]
        if k.dim() != 4 or k.shape[0] != 1 or k.shape[1] != 1:
            raise ValueError(
                f"a custom transfer kernel must be 1-D (separable), 2-D, or "
                f"(1, 1, L, L); got {tuple(filter.shape)}.")
        return gamma * k / k.sum()
    if filter == "legacy":
        raise ValueError(
            "filter='legacy' is a code path, not a kernel -- it pairs a 2x2 "
            "mean-pool restriction with bilinear interpolation, which are not "
            "transposes of one another. Use 'box' for the mean-pool as a "
            "genuine kernel, or 'bilinear' for the filter that IS the "
            "transpose of bilinear interpolation.")

    build, mult = _FILTER_SPECS[filter]
    length = mult * int(factor) if length is None else int(length)
    h = build(factor, length, dtype, device)
    return gamma * torch.outer(h, h)[None, None]


def sinc_kernel2d(factor=2, length=None, dtype=torch.float32, device=None):
    """Back-compat alias for `transfer_kernel('sinc', ...)`."""
    return transfer_kernel("sinc", factor, length, dtype, device)


def transfer_padding(filter=None, factor=DEFAULT_FACTOR, length=None):
    """Left / right halo so `R` divides the grid exactly and `P` inverts it."""
    length = filter_length(filter, factor, length)
    span = length - int(factor)
    if span < 0:
        raise ValueError(
            f"a transfer kernel must be at least `factor` long (got "
            f"length={length}, factor={factor}).")
    if span % 2:
        raise ValueError(
            f"length - factor must be even (got length={length}, "
            f"factor={factor}). An odd difference puts the kernel's centre of "
            f"mass half a cell off the coarse grid, which shifts the whole "
            f"hierarchy; use an even-length kernel for an even factor.")
    return span // 2, span - span // 2


_KERNEL_CACHE = {}


def _cached_kernel(filter, factor, length, device, dtype):
    """Kernels are tiny and rebuilt ~100x per unrolled forward -- memoise.

    Tensor filters bypass the cache: they are already materialised, and keying
    them by identity would pin them alive for the process.
    """
    if torch.is_tensor(filter):
        return transfer_kernel(filter, factor, length, dtype, device)
    key = (filter, int(factor), length, device, dtype)
    k = _KERNEL_CACHE.get(key)
    if k is None:
        k = transfer_kernel(filter, factor, length, dtype, device)
        _KERNEL_CACHE[key] = k
    return k


# ---------------------------------------------------------------------------
# the transfer pair
# ---------------------------------------------------------------------------
def _fold_circular2d(g, a, b):
    """Transpose of `F.pad(..., mode='circular')`: wrap the halo back in.

    `pad` copies `x[N-a:]` in front and `x[:b]` behind, so its transpose adds
    the gradient of those copies back onto the rows / columns they came from.
    """
    for dim in (-2, -1):
        n = g.shape[dim] - a - b
        core = g.narrow(dim, a, n).clone()
        if a:
            core.narrow(dim, n - a, a).add_(g.narrow(dim, 0, a))
        if b:
            core.narrow(dim, 0, b).add_(g.narrow(dim, n + a, b))
        g = core
    return g


def _weight_for(x, filter, factor, length, groups):
    dtype = x.real.dtype if torch.is_complex(x) else x.dtype
    w = _cached_kernel(filter, factor, length, x.device, dtype)
    return w.expand(groups, 1, w.shape[-2], w.shape[-1])


def _legacy_restrict(x, factor, julia_compat):
    """The pre-existing 2x2 mean-pool, odd-size pre-pad and all."""
    H, W = x.shape[-2], x.shape[-1]
    kh = factor if H > 1 else 1
    kw = factor if W > 1 else 1
    if kh == 1 and kw == 1:
        return x
    ph = 1 if (kh > 1 and (julia_compat or H % factor)) else 0
    pw = 1 if (kw > 1 and (julia_compat or W % factor)) else 0

    def fn(t):
        if ph or pw:
            t = F.pad(t, (pw, 0, ph, 0))
        return F.avg_pool2d(t, (kh, kw), stride=(kh, kw))

    return _apply_complex(fn, x)


def _legacy_prolong(x, scale):
    """The pre-existing bilinear interpolation -- NOT `_legacy_restrict`^T."""
    def fn(t):
        return F.interpolate(t, scale_factor=float(scale), mode="bilinear",
                             align_corners=False)
    return _apply_complex(fn, x)


def restrict(x, factor=DEFAULT_FACTOR, length=None, julia_compat=False,
             filter=None):
    """Restriction `R`: anti-alias filter, then decimate by `factor`.

    Output is `H // factor` x `W // factor`.  Singleton spatial dims are passed
    through untouched, so this is safe to call on a broadcast noise map of
    shape `(B, 1, 1, 1)`.

    `julia_compat` applies only to `filter='legacy'`; see the module docstring.
    """
    filter = _check_filter(filter)
    if not torch.is_tensor(filter) and filter == "legacy":
        return _legacy_restrict(x, factor, julia_compat)

    H, W = x.shape[-2], x.shape[-1]
    if H <= 1 and W <= 1:
        return x
    if H % factor or W % factor:
        raise ValueError(
            f"restrict got a {H}x{W} input, which is not a multiple of "
            f"factor={factor}. Every multigrid level is padded to a multiple "
            f"of s * 2^(levels-1) precisely so this cannot happen -- see "
            f"MGCDLNet.pad_stride.")

    a, b = transfer_padding(filter, factor, length)
    groups = x.shape[1]
    w = _weight_for(x, filter, factor, length, groups)

    def fn(t):
        if a or b:
            t = F.pad(t, (a, b, a, b), mode="circular")
        return F.conv2d(t, w, stride=factor, groups=groups)

    return _apply_complex(fn, x)


def prolong(x, scale=DEFAULT_FACTOR, length=None, filter=None):
    """Prolongation `P`: zero-insert by `scale`, then the same filter.

    Exactly `adjoint_scale(filter) . restrict^T` for every filter but
    `'legacy'` -- `scale^2` under the default normalisation, `1` under
    `'-unit'`.  See `split_normalisation`.
    """
    filter = _check_filter(filter)
    scale = int(scale)
    if not torch.is_tensor(filter) and filter == "legacy":
        return _legacy_prolong(x, scale)

    a, b = transfer_padding(filter, scale, length)
    groups = x.shape[1]
    w = _weight_for(x, filter, scale, length, groups) * adjoint_scale(filter, scale)

    def fn(t):
        t = F.conv_transpose2d(t, w, stride=scale, groups=groups)
        return _fold_circular2d(t, a, b) if (a or b) else t

    return _apply_complex(fn, x)


def noise_scale(factor=DEFAULT_FACTOR, length=None, filter=None):
    """`sigma_c / sigma` for i.i.d. noise pushed through `restrict`.

    Filtering by `h` scales the standard deviation by `||h||_2` (decimation
    does not change it), so this is `||h||_2` -- 1 for no filter, 0.5 for the
    2x2 mean, 0.4162 for the default kaiser-sinc.  Hardcoding one of those
    alongside a different kernel mis-scales every coarse prox threshold, which
    is exactly the kind of thing an ablation would otherwise blame on the
    filter.
    """
    filter = _check_filter(filter)
    if not torch.is_tensor(filter) and filter == "legacy":
        return 1.0 / float(factor)          # ||box||_2 in 2-D
    return float(transfer_kernel(filter, factor, length,
                                 dtype=torch.float64).norm())


def restrict_noise(sigma, factor=DEFAULT_FACTOR, length=None,
                   julia_compat=False, filter=None):
    """Coarse-grid noise level.  Scalar, per-batch scalar, or a full map."""
    if sigma is None:
        return None
    if torch.is_tensor(sigma) and sigma.dim() == 4:
        sigma = restrict(sigma, factor=factor, length=length,
                         julia_compat=julia_compat, filter=filter)
    return sigma * noise_scale(factor, length, filter)


def is_adjoint_pair(filter=None):
    """Whether `prolong == adjoint_scale(filter) . restrict^T` for this filter.

    True for every kernel; False only for `'legacy'`, whose two halves come
    from different families.
    """
    filter = _check_filter(filter)
    return torch.is_tensor(filter) or filter != "legacy"


# ---------------------------------------------------------------------------
# learnable variant
# ---------------------------------------------------------------------------
class GridTransfer(nn.Module):
    """A tied restriction / prolongation pair, optionally learned.

    `learn=False` (the default) reproduces the module-level `restrict` /
    `prolong` exactly.  `learn=True` makes the kernel a parameter -- and
    because both directions read the SAME weight, `P == factor^2 . R^T` still
    holds at every training step, with no penalty term and no reprojection.

    `filter` picks the kernel it starts from; `'legacy'` is rejected, since a
    non-adjoint pair is not something to hand a gradient to.

    Notes on using `learn=True`
    ---------------------------
    * The kernel is initialised AT a real interpolant, never randomly:
      `preload_with_widening` bootstraps the whole hierarchy on "the coarse
      level starts as an exact replica of the fine one", which is meaningless
      if `P` is not already a sane interpolant.
    * `channels=1` broadcasts one kernel over every channel; `channels=M`
      gives each subband its own (depthwise).  Do not make it channel-MIXING:
      `widen_z` and `alpha` already do that, at the right place.
    * `filter='box'` gives a learnable 2x2, which is what a U-Net's strided
      conv is -- but a 2-tap filter cannot beat the mean it starts from (see
      the module docstring), and its transpose is piecewise-constant block
      replication.  `'bilinear'` is the shortest kernel with room to move.
    * A freely-learned `R` will alias on purpose if that lowers training loss,
      which couples the coarse level to the training noise / undersampling
      distribution.  Log `noise_scale` per level and watch it drift.

    Only the LATENT-grid transfers should ever be learned.  The image-grid
    ones define `E_c` and `y_c = E_c^H y`, so learning those means learning the
    coarse forward model, which is the one thing that makes this a solver
    rather than a U-Net.  RAM draws exactly this line: learned strided convs
    between feature stages, fixed windowed sinc inside the physics.
    """

    def __init__(self, channels=1, factor=DEFAULT_FACTOR, length=None,
                 learn=False, filter=None):
        super().__init__()
        filter = _check_filter(filter)
        if not torch.is_tensor(filter) and filter == "legacy":
            raise ValueError(
                "GridTransfer cannot use filter='legacy': its restriction and "
                "prolongation are not transposes, so there is no single kernel "
                "to hold. Use the module-level restrict / prolong for that "
                "ablation row.")
        self.factor = int(factor)
        self.length = filter_length(filter, self.factor, length)
        self.filter = filter if not torch.is_tensor(filter) else "custom"
        # gamma is the kernel's target sum, adjoint_scale the c in P = c . R^T.
        # Both are pinned by the normalisation; see `split_normalisation`.
        _, self.norm = split_normalisation(filter)
        self.gamma, self.adjoint_scale = _norm_gains(self.norm, self.factor)
        self.pad = transfer_padding(filter, self.factor, self.length)

        k = transfer_kernel(filter, self.factor, self.length)
        k = k.repeat(int(channels), 1, 1, 1)
        if learn:
            self.weight = nn.Parameter(k)
        else:
            self.register_buffer("weight", k)
        self.learn = bool(learn)

    # -- the normalised kernel ---------------------------------------------
    def kernel(self, device=None, dtype=None):
        """Renormalised to `gamma`, so the R/P scale survives training.

        NOT to 1: under the "-unit" normalisation the kernel is supposed to
        sum to `factor^(d/2)`, and forcing it to 1 here would quietly undo the
        normalisation the caller asked for while leaving `adjoint_scale` at 1.
        """
        w = self.weight
        if device is not None or dtype is not None:
            w = w.to(device=device, dtype=dtype)
        denom = w.sum(dim=(-2, -1), keepdim=True)
        return self.gamma * w / denom.clamp(min=1e-3)

    @property
    def noise_scale(self):
        """`sigma_c / sigma`, recomputed -- the kernel moves when it is learned.

        Reduced across channels as an RMS, not a mean: `sigma` broadcasts over
        the channel axis, so the pooled standard deviation is the root-mean-
        square of the per-channel `||h_c||_2`, not their average.
        """
        w = self.kernel()
        return w.flatten(1).norm(dim=1).pow(2).mean().sqrt()

    # -- the pair -----------------------------------------------------------
    def _w(self, x):
        dtype = x.real.dtype if torch.is_complex(x) else x.dtype
        w = self.kernel(device=x.device, dtype=dtype)
        groups = x.shape[1]
        if w.shape[0] == 1:
            w = w.expand(groups, 1, self.length, self.length)
        elif w.shape[0] != groups:
            raise ValueError(
                f"{type(self).__name__}(channels={w.shape[0]}) got a "
                f"{groups}-channel input.")
        return w, groups

    def restrict(self, x):
        H, W = x.shape[-2], x.shape[-1]
        if H <= 1 and W <= 1:
            return x
        a, b = self.pad
        w, groups = self._w(x)

        def fn(t):
            if a or b:
                t = F.pad(t, (a, b, a, b), mode="circular")
            return F.conv2d(t, w, stride=self.factor, groups=groups)

        return _apply_complex(fn, x)

    def prolong(self, z):
        a, b = self.pad
        w, groups = self._w(z)
        w = w * self.adjoint_scale

        def fn(t):
            t = F.conv_transpose2d(t, w, stride=self.factor, groups=groups)
            return _fold_circular2d(t, a, b) if (a or b) else t

        return _apply_complex(fn, z)

    def restrict_noise(self, sigma):
        if sigma is None:
            return None
        if torch.is_tensor(sigma) and sigma.dim() == 4:
            sigma = self.restrict(sigma)
        return sigma * self.noise_scale

    def extra_repr(self):
        return (f"filter={self.filter!r}, factor={self.factor}, "
                f"length={self.length}, norm={self.norm!r}, "
                f"c={self.adjoint_scale:g}, channels={self.weight.shape[0]}, "
                f"learn={self.learn}")


# ---------------------------------------------------------------------------
# operator form
# ---------------------------------------------------------------------------
class Resample(Operator):
    """The grid-transfer pair as a linear operator.

    forward  : `prolong`  (coarse -> fine)
    adjoint  : `restrict` (fine -> coarse)

    For every filter but `'legacy'` the adjoint is the EXACT transpose of the
    forward, up to the `scale^2` normalisation that both carry.  That is what
    makes `galerkin(E).gram(.)` Hermitian, and hence what makes the coarse
    level a real optimisation problem for the FAS correction to correct.
    `is_adjoint` reports which case you are in.
    """

    def __init__(self, scale=DEFAULT_FACTOR, length=None, filter=None):
        self.scale = int(scale)
        self.length = length
        self.filter = _check_filter(filter)

    @property
    def is_adjoint(self):
        return is_adjoint_pair(self.filter)

    def forward(self, x):
        return prolong(x, self.scale, self.length, filter=self.filter)

    def adjoint(self, x):
        return restrict(x, self.scale, self.length, filter=self.filter)

    def __repr__(self):
        name = self.filter if not torch.is_tensor(self.filter) else "custom"
        return f"Resample(scale={self.scale}, filter={name!r})"


def make_resample(scale, length=None, filter=None):
    """`Resample(1)` is the identity -- return the cheap operator instead."""
    return Identity() if int(scale) == 1 else Resample(scale, length, filter)


def galerkin(E, scale=DEFAULT_FACTOR, length=None, filter=None):
    """Coarse-grid forward operator  E_c = E . P.

    Dispatches on the identity so that denoising V-cycles never resample:
    `galerkin(Identity()) is Identity()` and the coarse Gram stays free.
    """
    if E is None:
        return Identity()
    if isinstance(E, Identity):
        return E
    return E @ Resample(scale, length, filter)
