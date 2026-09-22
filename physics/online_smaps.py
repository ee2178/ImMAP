"""
Coil maps estimated from the measurement, the way `Sljiva/src/closures/mrireco.jl`
does it in `genobs`:

    smaps = walsh_smaps(F'(hamming_window(k) .* cm .* k))                 # :walsh
    smaps = espirit(hamming_window(k) .* cm .* k; acs_size, thresh_eig=0) # :espirit

Three things come with that, and each is a departure from the precomputed maps
in `fastmri_preprocessed/`:

1. THE CALIBRATION DATA IS WHAT THE SCAN MEASURED. `cm .* k` is the sampled
   centre of the noisy, undersampled k-space -- not the fully sampled volume.
   Maps therefore degrade with sigma exactly as they would in deployment, and
   nothing the scan did not measure reaches the forward model.
2. THE ACS BLOCK IS ASYMMETRIC. `mrireco.jl:244` takes the FULL extent along
   readout and the ACS LINE COUNT along phase-encode, each clamped to 32:

       num_lines = sum(cm)
       acs_size  = (N, num_lines)   # readout_dir = :v
       acs_size  = clamp.(acs_size, 1, 32)

   The ACS lines are fully sampled along readout, so a square block throws
   away calibration data for free. A 20x20 block on a 20-coil 8x8 kernel gives
   169 rows against 1280 columns -- underdetermined, no estimated null space.
   32x20 gives 325.
3. NO EIGENVALUE THRESHOLD. The Julia call passes `thresh_eig=0`, so online
   maps have no hard support and `use_organ_mask` cannot be fed from them. That
   is the default here too; `physics/object_mask.py` is where the mask comes
   from now.

COST. ESPIRiT's kernel images are `coils x retained kernels x the full grid`,
complex, which on brain's 640x320 with 20 coils is GIGABYTES PER SLICE and is
recomputed every training step. `estimate_cost_gb` is exported so a caller can
print the bound before committing a run, and `walsh` is the cheap alternative
that Sljiva's own multigrid experiments actually used
(`makeconfigs_mglpds.jl`: `:online_smaps=>[true,]`, i.e. Walsh).
"""

from __future__ import annotations

import torch

from operators.fourier import ifftc
from physics.smaps import espirit, walsh

METHODS = ("espirit", "walsh")
ACS_CLAMP = 32          # mrireco.jl:246, `clamp.(acs_size, 1, 32)`


# ---------------------------------------------------------------------------
def acs_line_count(mask, dim=-1):
    """Number of contiguous centre lines in `mask`, along `dim`.

    The sampling mask is a set of phase-encode lines; its centre block is the
    ACS. Counting the CONTIGUOUS run through the centre, rather than every
    sampled line, is what makes this the calibration region: outer lines are
    sampled too, and including them would claim a calibration region with gaps
    in it.
    """
    m = mask
    while m.dim() > 1:
        m = m.amax(dim=0) if m.shape[0] > 1 else m[0]
    m = m.to(torch.bool)
    n = m.numel()
    c = n // 2
    if not bool(m[c]):
        return 0
    lo = c
    while lo > 0 and bool(m[lo - 1]):
        lo -= 1
    hi = c
    while hi < n - 1 and bool(m[hi + 1]):
        hi += 1
    return hi - lo + 1


def resolve_lines(mask, acs_lines=None):
    """How many centre lines to calibrate from.

    `acs_lines` is the config's own number and is preferred: Sljiva takes
    `num_lines = sum(cm)` from the centre mask it built, and ImMAP's `get_mask`
    returns only the UNION of centre and accelerated lines. Counting the
    contiguous run through the centre recovers it, but overcounts whenever an
    accelerated line happens to abut the block -- measured: 9 for an
    `acs_lines=8` mask at R=4, 25 for 24. Those extra lines are genuinely
    sampled, so using them would not be wrong, but the calibration region would
    then depend on the acceleration by accident rather than by design.

    The counted run is still the CEILING: asking for more lines than the mask
    contiguously samples would pull unmeasured columns into the ACS.
    """
    counted = acs_line_count(mask)
    if acs_lines is None:
        return counted
    return max(0, min(int(acs_lines), counted))


def acs_block(mask, shape, clamp=ACS_CLAMP, acs_lines=None):
    """`(ax, ay)` for `espirit`, following `mrireco.jl:244`.

    Full extent along readout, the ACS line count along phase-encode, each
    clamped to `clamp`. ImMAP's k-space is `(B, C, readout, phase)`, so the
    phase-encode axis is the last one.
    """
    nx, ny = int(shape[-2]), int(shape[-1])
    lines = resolve_lines(mask, acs_lines)
    if lines <= 0:
        raise ValueError(
            "the sampling mask has no sampled line at the centre of k-space, "
            "so there is no ACS to calibrate from. Online maps need a "
            "centre-sampled mask (physics/mask.py::make_acc_mask with "
            "acs_lines > 0).")
    return (min(nx, clamp), min(lines, clamp))


def center_mask(mask, lines, like=None):
    """A mask keeping only the `lines` centre phase-encode columns."""
    n = mask.shape[-1]
    lo = n // 2 - lines // 2
    cm = torch.zeros(n, device=mask.device, dtype=mask.dtype if
                     mask.dtype.is_floating_point else torch.float32)
    cm[lo:lo + lines] = 1
    return cm


def hamming_window(k, dims=(-2, -1)):
    """Separable Hamming window over `dims`, centred on k-space DC.

    `Sljiva.hamming_window` applies it to the calibration data before the
    transform: the ACS block is a hard truncation of k-space, and windowing is
    what keeps its ringing out of the estimated maps.
    """
    w = None
    for d in dims:
        n = k.shape[d]
        h = torch.hamming_window(n, periodic=False, device=k.device,
                                 dtype=torch.float32)
        shape = [1] * k.dim()
        shape[d] = n
        w = h.reshape(shape) if w is None else w * h.reshape(shape)
    return k * w


# ---------------------------------------------------------------------------
def estimate_cost_gb(shape, n_kernels, dtype_bytes=8):
    """Upper bound on `espirit`'s kernel-image tensor, in GB.

    `coils x kernels x grid`, complex -- the peak allocation of the whole
    routine and the reason online ESPIRiT is expensive at brain's grid size.
    `n_kernels` is bounded by `min(rows, ks^2 * C)`; the retained count is
    usually well below it, so this is a bound and not a prediction.
    """
    b, c, nx, ny = shape
    return b * c * int(n_kernels) * nx * ny * dtype_bytes / 2 ** 30


def online_smaps(kspace, mask, method="espirit", acs_lines=None, kernel_size=6,
                 thresh_rowspace=0.05, thresh_eig=0.0, maxit=100,
                 walsh_ks=5, walsh_stride=2, window=True, clamp=ACS_CLAMP):
    """Coil maps from the measured centre of `kspace`.

    Parameters
    ----------
    kspace : (B, C, H, W) complex   the MASKED measurement, noise included
    mask   : the sampling mask, any shape broadcasting to (..., W)
    method : "espirit" (default) or "walsh"
    acs_lines : the config's `mri.acs_lines`. Pass it -- see `resolve_lines`.
    thresh_eig : 0.0 by default, as in `mrireco.jl` -- no hard support.

    Returns
    -------
    (B, C, H, W) complex, unit-RSS wherever the maps are nonzero.

    `kernel_size` defaults to 6 rather than `espirit`'s 8: the ACS is at most
    32 x lines here, and an 8x8 kernel leaves too few patches to estimate a
    null space from (at 32x20: 325 rows for ks=6 against 169 for ks=8, with
    ks^2 * C columns either way).
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if not torch.is_complex(kspace):
        raise ValueError(f"expected complex k-space, got {kspace.dtype}")

    lines = resolve_lines(mask, acs_lines)
    ax, ay = acs_block(mask, kspace.shape, clamp=clamp, acs_lines=acs_lines)

    # cm .* k, then the window: only the sampled centre reaches the estimator,
    # so nothing outside the ACS -- measured or not -- can leak into the maps.
    cm = center_mask(mask, min(lines, ay), like=kspace)
    kc = kspace * cm.to(kspace.dtype)
    if window:
        kc = hamming_window(kc)

    if method == "walsh":
        return walsh(ifftc(kc), ks=walsh_ks, stride=walsh_stride)

    return espirit(kc, acs_size=(ax, ay), kernel_size=kernel_size,
                   thresh_rowspace=thresh_rowspace, thresh_eig=thresh_eig,
                   maxit=maxit)
