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
WINDOWS = ("box", "hamming", "hamming-acs")
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


def acs_taper(mask, lines, like=None):
    """`center_mask` with a Hamming taper ACROSS the `lines` centre columns.

    The ACS is a hard cut of k-space along phase-encode, so the calibration
    image rings (sinc, ~9% sidelobes, period W/lines). Near an object edge that
    ringing differs coil to coil -- it is each coil's (s_c * rho) that is
    truncated, not rho -- so it does not divide out of the maps: Walsh shows it
    as hatching around the edges and streaks inside (confirmed in
    espirit_acs_check: finer ripple at 31 lines than 13, gone with this
    taper). A taper over the ACS WIDTH is what removes it; the full-grid
    `hamming_window` is ~1 across a 13-line ACS and does not.
    """
    n = mask.shape[-1]
    lo = n // 2 - lines // 2
    w = torch.zeros(n, device=mask.device, dtype=torch.float32)
    w[lo:lo + lines] = torch.hamming_window(lines, periodic=False,
                                            device=mask.device)
    return w


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


def online_smaps(kspace, mask, method="espirit", acs_lines=None, kernel_size=4,
                 thresh_rowspace=0.05, thresh_eig=0.0, maxit=100,
                 walsh_ks=5, walsh_stride=2, walsh_phase_ref="virtual",
                 window=None, clamp=ACS_CLAMP, phase_correct=False):
    """Coil maps from the measured centre of `kspace`.

    Parameters
    ----------
    kspace : (B, C, H, W) complex   the MASKED measurement, noise included
    mask   : the sampling mask, any shape broadcasting to (..., W)
    method : "espirit" (default) or "walsh"
    acs_lines : the config's `mri.acs_lines`. Pass it -- see `resolve_lines`.
    thresh_eig : 0.0 by default, as in `mrireco.jl` -- no hard support.
    walsh_phase_ref : "virtual" (default) or "strongest" (Sljiva's
             `walsh_smaps`); see `physics.smaps.walsh`. The strongest-coil
             reference is noise wherever that coil is dark.
    window : None (default) -- "hamming-acs" for walsh, "box" for espirit.
             "hamming-acs" tapers across the ACS lines themselves
             (`acs_taper`): Walsh reads the calibration IMAGE, so the ACS
             truncation's Gibbs ringing lands directly in its maps. ESPIRiT
             keeps "box": its kernels assume the calibration data IS k-space
             of s_c * rho, and a taper (an image-domain convolution) breaks
             that relation.
             "box" -- the ACS lines exactly as measured, no taper --
             or "hamming", the full-grid window `mrireco.jl` applies
             (`Sljiva.hamming_window`, identical to
             `torch.hamming_window(N, periodic=False)`). True/False are read
             as "hamming"/"box" for older callers.

             What the full-grid Hamming actually does: along PHASE-ENCODE it
             is still 0.996 at the edge of a 13-line ACS, so it barely tapers
             the ACS truncation; along the fully sampled READOUT axis it is a
             real low-pass (the calibration image changes by ~57%). On a
             phantom with known maps the two gave the SAME map quality (the
             maps are smooth, so a readout blur costs nothing).
             Everything outside the ACS is zeroed by the centre mask in
             either case (verified: exactly 0 energy outside the ACS
             columns), so neither choice lets accelerated lines leak in.

    phase_correct : False (default) or True -- `Sljiva`'s `phase_correct`.
             Rotate the maps by the phase of the coil-combined calibration
             image, `sgn(sum_c conj(s_c) acs_c)`, computed with the SAME maps
             (see `phase_correct_maps`).  Removes whatever per-pixel phase
             the estimator's reference convention put in the maps -- the
             strongest coil's, noise where that coil is dark -- and with it
             the object's low-resolution phase, so the image the net must
             reconstruct is near-real.  Invisible to SENSE and to any
             magnitude image; it changes only what a complex-valued prior
             sees.

    Returns
    -------
    (B, C, H, W) complex, unit-RSS wherever the maps are nonzero.

    `kernel_size` defaults to 4 (since 2026-09-29; was 6) rather than
    `espirit`'s 8: the ACS is at most 32 x lines here, and the Hankel matrix
    has (32-ks+1)(lines-ks+1) rows against ks^2 * C columns. At the grid's
    32 x 13 block and 20 coils that is 290 rows vs 320 columns for ks=4,
    216 vs 720 for ks=6, and 150 vs 1280 for ks=8.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if not torch.is_complex(kspace):
        raise ValueError(f"expected complex k-space, got {kspace.dtype}")

    lines = resolve_lines(mask, acs_lines)
    ax, ay = acs_block(mask, kspace.shape, clamp=clamp, acs_lines=acs_lines)

    # cm .* k, then the window: only the sampled centre reaches the estimator,
    # so nothing outside the ACS -- measured or not -- can leak into the maps.
    if window is None:
        window = "hamming-acs" if method == "walsh" else "box"
    elif window is True:
        window = "hamming"
    elif window is False:
        window = "box"
    if window not in WINDOWS:
        raise ValueError(f"window must be one of {WINDOWS}, got {window!r}")

    if window == "hamming-acs":
        cm = acs_taper(mask, min(lines, ay), like=kspace)
    else:
        cm = center_mask(mask, min(lines, ay), like=kspace)
    kc = kspace * cm.to(kspace.dtype)
    if window == "hamming":
        kc = hamming_window(kc)

    if method == "walsh":
        smaps = walsh(ifftc(kc), ks=walsh_ks, stride=walsh_stride,
                      phase_ref=walsh_phase_ref)
    else:
        smaps = espirit(kc, acs_size=(ax, ay), kernel_size=kernel_size,
                        thresh_rowspace=thresh_rowspace, thresh_eig=thresh_eig,
                        maxit=maxit)
    if phase_correct:
        smaps = phase_correct_maps(smaps, ifftc(kc))
    return smaps


def phase_correct_maps(smaps, calib):
    """`s_c <- s_c * sgn(sum_c' conj(s_c') calib_c')` -- `Sljiva`'s phase_correct.

    Maps are defined only up to a per-pixel phase shared by all coils, and each
    estimator fixes it by some convention (ESPIRiT here: the strongest coil is
    real, per pixel).  Whatever that phase e^{i phi} is, the calibration image
    combined with the SAME maps carries e^{-i phi} too, so multiplying by its
    phase cancels phi exactly.  The implied image becomes
    `x conj(sgn(x_lowres))`: the convention -- and any noise in it -- is gone,
    and so is the object's smooth low-resolution phase.

    Magnitude, unit-RSS and support are unchanged.  Where the combination is
    exactly 0 (outside a thresholded support) the maps are left as they are.
    """
    t = (smaps.conj() * calib).sum(dim=1, keepdim=True)
    a = t.abs()
    pf = torch.where(a > 0, t / a.clamp_min(1e-30), torch.ones_like(t))
    return smaps * pf
