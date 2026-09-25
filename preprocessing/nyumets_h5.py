#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build one HDF5 per (patient, session) from NYUMets, for sessions carrying the full
FLAIR / T1 / CT1 / T2 quartet. Sessions missing any of the four are skipped.

Layout mirrors preprocessing/cmap.py, so the existing BraTS loaders read these unchanged:

    <out>/<patient>_<session>/<patient>_<session>_img.h5
        img_raw          (n, H, W, C) float32   unnormalized intensities
        img_median_mad   (n, H, W, C) float32   per-contrast median/MAD, within brain
        img              -> soft link to img_median_mad (an alias, zero extra bytes)
        mask             (n, H, W, 1) uint8     brain mask
        support          (n, H, W, 1) uint8     common acquisition support (see below)
        slice_index      (n,)         int32     maps n back to the original RAS z

COMMON SUPPORT -- COMPUTED AND STORED, NOT APPLIED (`--support store`, the default). Contrasts
of one session do not always cover the same anatomy: a T1 whose FOV reaches the eyes paired
with a CT1 whose FOV does not is the usual case. The INTERSECTION of the per-contrast
acquisition supports is written to `support` and summarised in the `support_lost_<C>` attrs
(a large value on one contrast IS the FOV mismatch), but the pixels are left alone.

`--support apply` restores the older behaviour of multiplying it into every channel. That was
the default until 2026-09-22 and was dropped because zeroing the pixels bakes a ragged FOV
boundary into the data, which the regressors then have to reproduce -- the visible boundary
artefacts in the reconstructions. The trade it makes is real in the other direction too: with
the support merely stored, a T1 -> CT1 bridge does see input anatomy (an eye) that its target
never acquired, and must learn to drop it. Apply `support` in the loss instead if that shows up.

CROSS-STUDY REGISTRATION -- dz FROM THE IMAGES, IN-PLANE FROM THE BRAIN CENTROID
(`--reg-method centroid`, the default since 2026-09-25). A patient's sessions are built together
and every one is put onto the index axis of the deepest: a through-plane shift dz found by the
masked correlation of the normalised T1 with the reference, then ONE in-plane shift (dh, dw) that
centres the study's brain mask on the reference's. Integer translations only; rotation is not
undone. `reg_offset` is the (dz, dh, dw) applied, `reg_slice_shift` the per-stored-slice (dh, dw)
(uniform under centroid), and `reg_ncc` the masked correlation with the reference afterwards -- the
quality gate. `--reg-method search` finds (dh, dw) by a correlation search instead, optionally per
slice: slightly better scores (median masked CT1 0.485 vs 0.465 for one searched shift, below),
at ~8 s per study against ~1-2 s.

Why not the alternatives (2026-09-25 comparison on the source NIfTIs, median masked CT1
correlation over 27 study pairs, which the search never sees): no registration 0.06, 3-D FFT
cross-correlation (`--reg-method xcorr`) 0.35, search with one in-plane shift 0.47, search with
per-slice shifts 0.49; dz with a brain-centroid shift landed close to the searched shift on visual
comparison, for a fraction of the cost. `--reg-method affine` -- resampling with inv(A_other) @ A_ref from the NIfTI
headers -- scored ~0: scanner coordinates are not anatomical across sessions (the patient lies
differently each visit), so the header transform re-poses a study rather than aligning it. The
NYUMets release's own tooling (github.com/nyumets/nyumets) registers across timepoints with a
translation-only SimpleITK fit, and also writes voxel spacing with `set_spacing([1, 1, 1])`
without resampling, so the "1 mm isotropic" in these headers is not a measurement.

Under `affine` the pixels are resampled onto the reference's grid and the `affine` / `native_size`
attrs describe THAT grid (the session's own are kept as `affine_source` / `native_size_source`);
`reg_matrix` is the transform applied. Under `search` and `xcorr` the pixels are only shifted.

ORIENTATION. Stored axes stay canonical RAS: H runs to the patient's RIGHT, W ANTERIOR, so a
slice drawn as-is has the eyes on the image's RIGHT. That is a DISPLAY concern and is fixed at
display time -- `visualization.image.set_display_orient("radiological")`, or the `orient=`
argument of plot_image / subplot_images -- NOT by rewriting the pixels, which would invalidate
every checkpoint trained on the old axes for no modelling gain.

Channel order is [FLAIR, T1, T1ce, T2] -- the BraTS order, so `contrast_idx` and every
existing config keep their meaning. NYUMets writes the enhanced T1 as CT1; the stored
LABEL stays T1ce.

Normalization is median/MAD (robust centre AND scale), computed per contrast over the
IN-BRAIN voxels of the whole volume via cmap.normalize_masked -- same statistics and
same background-zero convention as the BraTS h5s. Following cmap, the stats come from
the percentile-CLIPPED volume while `img_raw` stores the UNCLIPPED data, so inverting
img_median_mad -> raw is exact only away from the clipped tails.

    python -m preprocessing.nyumets_h5 --root ../datasets/NYUMets/data/imaging/patientId \
                                       --out ~/scratch/datasets/NYUMets_h5 --crop 224
"""

import os
import re
import csv
import glob
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import h5py
import nibabel as nib

from operators.fourier import fftc, ifftc
from preprocessing.cmap import (
    normalize_masked, channelwise_percentile_clip, center_crop_spatial, norm_key,
)

CONTRASTS = ("FLAIR", "T1", "T1ce", "T2")          # stored channel order (BraTS order)
MODE = "median-mad"

# First match wins. FLAIR before T2 ("T2_FLAIR" is FLAIR); CT1 before T1 (t1 is a substring
# of ct1). The CT1 guard is a lookbehind, not \bct1\b -- '_' is a word character, so \b never
# fires between the '_' and the 'c' in 'NYU0001_CT1.nii.gz'.
RULES = (
    ("FLAIR", r"flair"),
    ("T1ce",  r"(?<![a-z0-9])c[\W_]?t1(?!\d)"),
    ("T1",    r"(?<![a-z0-9])t1(?!\d)"),
    ("T2",    r"(?<![a-z0-9])t2(?!\d)"),
)
RULES = tuple((lab, re.compile(p, re.I)) for lab, p in RULES)
SKIP_RE = re.compile(r"(seg|label|lesion|mask|roi|contour)", re.I)
DATE_RE = re.compile(r"((?:19|20)\d{2})[-_.]?(\d{2})[-_.]?(\d{2})")


def find_sessions(root):
    """-> {(patient, session): {contrast: path}}, keeping only complete quartets.

    LAYOUT: patientId/<PID>/studyId/<STUDY_ID>/{FLAIR,T1,CT1,T2}.nii -- `studyId` is a fixed
    literal directory and the session is one level BELOW it.

    The session key is therefore the IMMEDIATE PARENT DIRECTORY of the file: contrasts
    acquired together live together, which is the only rule that survives this layout. It
    replaces a date-regex-then-first-path-component fallback that returned the literal string
    "studyId" for every file, collapsing all of a patient's studies into one pseudo-session
    whose contrasts were then taken from DIFFERENT dates -- which is what produced both the
    mass "shapes differ" failures and the apparent in-plane disagreements between contrasts.

    Matching is on the BASENAME, not the whole relative path, so a directory name can never
    be mistaken for a contrast tag. RTSTRUCT_segmentation.nii is dropped by SKIP_RE;
    RTSTRUCT_MRI.nii simply matches no contrast rule.
    """
    out = {}
    for pid in sorted(e.name for e in os.scandir(root) if e.is_dir()):
        for path in glob.glob(os.path.join(root, pid, "**", "*.nii*"), recursive=True):
            fname = os.path.basename(path)
            if SKIP_RE.search(fname):
                continue
            lab = next((l for l, rx in RULES if rx.search(fname)), None)
            if lab is None:
                continue
            ses = os.path.basename(os.path.dirname(path)) or "single"
            out.setdefault((pid, ses), {}).setdefault(lab, path)
    return {k: v for k, v in sorted(out.items()) if all(c in v for c in CONTRASTS)}


def load_ras(path):
    """(H, W, D) float32 in canonical RAS, plus the affine. No resampling.

    REFUSES complex input rather than casting it. `np.asarray(z, dtype=np.float32)` on a
    complex array discards the imaginary part behind a ComplexWarning -- it would silently
    throw away the phase and leave the real part (NOT the magnitude) in its place, which is
    worse than either intended behaviour. If NYUMets ever turns out to carry complex volumes,
    decide explicitly here whether to store magnitude+phase as separate channels.
    """
    img = nib.as_closest_canonical(nib.load(path))
    v = np.asanyarray(img.dataobj)
    while v.ndim > 3:
        v = v[..., 0]
    if np.iscomplexobj(v):
        raise NotImplementedError(
            f"{path} holds complex data ({v.dtype}); this builder stores real channels only. "
            f"Decide how to carry phase (e.g. magnitude and phase as separate channels) "
            f"before proceeding -- casting here would drop the imaginary part silently.")
    return np.asarray(v, dtype=np.float32), img.affine


def center_crop_or_pad(a, size, h_axis, w_axis):
    """Center-crop to `size` on the two spatial axes, ZERO-PADDING any axis shorter than it.

    cmap.center_crop_spatial only crops. Given H < size it computes a NEGATIVE start index,
    which Python reads as an offset from the end, so it silently returns an array of length
    (size - H) // 2 instead of failing -- a 168-row volume with --crop 224 came out 28 rows,
    and only surfaced much later as a RandomCrop error inside the dataloader. BraTS never hit
    it because every subject is 240x240; NYUMets matrix sizes are NOT uniform, so the pad
    branch is load-bearing here.

    Padding rather than resampling keeps the voxel spacing honest, and padded voxels are zero
    and fall outside the support mask, so nothing downstream mistakes them for anatomy.
    """
    is_bool = a.dtype == torch.bool
    if is_bool:
        a = a.to(torch.uint8)
    for axis in (h_axis, w_axis):
        n = a.shape[axis]
        if n > size:
            a = a.narrow(axis, (n - size) // 2, size)
        elif n < size:
            before = (size - n) // 2
            pad = [0] * (2 * a.dim())
            j = (a.dim() - 1 - axis) * 2          # F.pad fills from the LAST dim backwards
            pad[j], pad[j + 1] = before, size - n - before
            a = F.pad(a, pad)
    return a.to(torch.bool) if is_bool else a


class Volume:
    """One session's full-depth arrays, before any slice filtering.

    raw/norm are (D, C, H, W); fg/fov are (D, H, W). Registration resamples all four together.

    `reg` is filled in by `register_patient` -- the transform that was applied, for write_h5 to
    record. It is not a constructor argument; nothing builds a Volume already registered.
    """

    __slots__ = ("raw", "norm", "fg", "fov", "stats", "orig_depth", "affine", "lost",
                 "native_hw", "reg")

    def __init__(self, **kw):
        self.reg = None
        for k in self.__slots__:
            if k != "reg":
                setattr(self, k, kw[k])


def lowpass_inplane(v, frac):
    """Blur a (D, H, W) real volume by keeping the central `frac` of k-space IN-PLANE only.

    Follows generate_longibrain.jl, which correlates `rss(F'(F(c) .* hamming .* acs_mask))`:
    a Hamming-tapered, ACS-cropped (so heavily low-passed) image. Registering on coarse
    structure is what keeps the peak from being set by noise and fine detail.

    IN-PLANE only, deliberately: the through-plane offset is the one worth having here, and
    blurring along z would smear exactly the axis we are trying to localise. Slice spacing is
    also several times the in-plane spacing, so z has far less detail to reject.

    `operators.fourier.fftc` is already centred (fftshift o fft o ifftshift), so the window
    below is built about the array centre and needs no shifting of its own -- unlike
    `operators.hpf.HighPassFilter`, which builds its window from an UNSHIFTED `fftfreq` grid.
    """
    if not frac or frac >= 1.0:
        return v
    # on v's device: a CPU window against a CUDA volume is an error, not a silent copy
    kw = dict(dtype=v.dtype, device=v.device)
    win = torch.ones(v.shape[-2:], **kw)
    for ax, n in enumerate(v.shape[-2:]):
        k = max(2, int(round(n * float(frac))))
        w = torch.zeros(n, **kw)
        lo = n // 2 - k // 2
        w[lo:lo + k] = torch.hamming_window(k, periodic=False, **kw)
        win = win * (w.view(-1, 1) if ax == 0 else w.view(1, -1))
    return ifftc(fftc(v, dim=(-2, -1)) * win, dim=(-2, -1)).abs()


def stored_to_world(affine, native_hw, crop):
    """4x4 mapping a STORED (h, w, z, 1) index to world mm.

    INDEX ORDER IS (h, w, z), matching the NIfTI (i, j, k) the affine was written for -- NOT
    the (z, h, w) the stored arrays are shaped as. `load_ras` transposes the pixels to put z
    first; the affine is left alone, so anything multiplying by it has to use (h, w, z).

    `center_crop_or_pad` shifts the in-plane indices and the builder never folded that into the
    saved affine, so reconstruct it here: an axis of native length n rendered at `crop` moves by
    `(crop - n) // 2` -- positive when padded, negative when cropped, one expression for both.
    The z index is untouched (slice_index IS the native z), so only rows 0 and 1 shift.
    """
    A = np.asarray(affine, dtype=np.float64).copy()
    if not crop or int(crop) <= 0:        # write_h5 stores crop_size = -1 for "no crop"
        return A
    # stored = native + off  ->  native = stored - off
    off = [int((crop - int(n)) // 2) for n in native_hw]
    S = np.eye(4)
    S[0, 3], S[1, 3] = -off[0], -off[1]
    return A @ S


def affine_matrix(ref, other, crop):
    """-> (M, oblique). M is the 4x4 taking a REF stored index (h, w, z, 1) to the OTHER's.

    Each argument is (affine, native_hw, orig_depth); `orig_depth` is unused here and accepted
    only so the signature matches `affine_offset`.

    This is the whole transform relating the two studies, rotation included, and it is EXACT
    for a rigid pair on a common voxel grid -- which the --geometry survey says NYUMets is
    (1.0 mm isotropic, one stored FOV, 4478 studies). It is the quantity a cross-correlation
    was estimating, badly: the correlation can only return a translation, and the 2026-09-24
    cohort survey put the median inter-session obliqueness at 0.113 (~6.5 deg), five times the
    0.02 above which no translation aligns anything. `M` is what `register_patient` resamples
    with, and it needs no pixels to compute.

    M maps an OUTPUT (reference-grid) index to the SOURCE index it should be sampled from,
    which is the direction `torch.nn.functional.grid_sample` wants.
    """
    Ar = stored_to_world(ref[0], ref[1], crop)
    Ao = stored_to_world(other[0], other[1], crop)
    M = np.linalg.inv(Ao) @ Ar
    R = M[:3, :3]
    return M, float(np.abs(R - np.diag(np.diag(R))).max())


def affine_offset(ref, other, crop):
    """Integer (dz, dh, dw) putting `other` onto `ref`, from their affines alone.

    Each argument is (affine, native_hw, orig_depth). Exact whenever the two volumes share a
    voxel size and differ by a translation -- which the --geometry survey says is this cohort:
    1.0 mm isotropic everywhere, only the FOV and the slice count differ. Cross-correlation is
    estimating a quantity the headers already state exactly.

    The translation is taken AT THE VOLUME CENTRE (see below), which is the best single shift
    when the two are also slightly rotated -- as this cohort is.

    Also returns `oblique`, the largest off-axis component of the rotation relating them: 0 for a
    pure translation, ~0.0175 per degree. Above ~0.02 the studies are genuinely rotated and NO
    integer shift aligns them, whatever produced it -- the periphery of the head is displaced by
    roughly sin(angle) * 110 voxels even after the best possible shift.
    """
    hw = (crop, crop) if crop and int(crop) > 0 else tuple(int(v) for v in ref[1])
    M, oblique = affine_matrix(ref, other, crop)
    # M maps a REF index to the OTHER index holding the same anatomy, so ref row h is other
    # row h + t. `translate(a, c)` sets out[i] = a[i + c], so c = +t is exactly the shift that
    # brings other onto ref -- NOT -t. (The negation belongs in `translation_offset`, whose
    # correlation peak points the other way.)
    #
    # EVALUATED AT THE VOLUME CENTRE, not at index (0, 0, 0). `M[:3, 3]` is the displacement at
    # the CORNER, and the two only agree when the studies are related by a pure translation.
    # This cohort is oblique -- ~5 deg between sessions is normal for clinically angled brain
    # MRI -- and at 5 deg the corner and the centre of a 224 volume disagree by
    # sin(5 deg) * 112 ~ 10 voxels. The head sits at the centre, so that is where the best
    # single translation is measured. Whatever rotation remains is what `oblique` reports and
    # no integer shift can remove.
    c = np.array([(hw[0] - 1) / 2.0, (hw[1] - 1) / 2.0, (ref[2] - 1) / 2.0, 1.0])
    t = (M @ c)[:3] - c[:3]
    return (int(round(t[2])), int(round(t[0])), int(round(t[1]))), oblique


def translation_offset(x, y, phase=True):
    """Integer (dz, dh, dw) to pass to `translate(y, t)` so that y lands on x.

    3-D FFT cross-correlation, PHASE-normalised by default (see `phase`).

    corr = IFFT( FFT(x) * conj(FFT(y)) ), peak -> lag. The transform is UNCENTERED here and
    the peak index is folded into [-n/2, n/2), which is the same answer generate_longibrain.jl
    gets from its centred `Fourier{3}` and `argmax - size/2 - 1`, without the fftshift dance.

    PLAIN cross-correlation, not phase correlation: the cross-power spectrum is NOT normalised
    by its magnitude. That matches the Julia script. Two consequences worth knowing:
      * bright regions dominate the peak, which is usually what you want on magnitude MR;
      * with zero-padded volumes the overlap shrinks as |lag| grows, so the estimate is biased
        TOWARD SMALL SHIFTS. Fine when the studies are nearly aligned already, which is the
        regime here; it would under-estimate a genuinely large offset.
    """
    if x.shape != y.shape:
        raise ValueError(f"cross-correlation needs one grid: {tuple(x.shape)} vs {tuple(y.shape)}")
    dim = (-3, -2, -1)
    X = fftc(x.to(torch.float32), dim=dim)
    Y = fftc(y.to(torch.float32), dim=dim)
    C = X * Y.conj()
    if phase:
        # PHASE correlation: divide the cross-power spectrum by its magnitude. Every frequency
        # then contributes equally, so the peak is a near-delta instead of a broad blob, the
        # estimate stops depending on per-scan intensity scaling, and -- the part that matters
        # here -- the DC / overlap-area envelope that biases plain correlation toward zero lag
        # is gone. generate_longibrain.jl does NOT do this; it registers one protocol at three
        # visits on identical grids, which is a far easier problem than two NYUMets studies.
        C = C / C.abs().clamp_min(1e-12)
    corr = ifftc(C, dim=dim).abs().cpu()                # one small transfer, then plain numpy
    # `ifftc` is centred, so zero lag sits at the array centre -- the same convention
    # generate_longibrain.jl reads with `argmax .- size .÷ 2 .- 1`. NEGATED, because the peak
    # says where y sits relative to x and moving y onto x is the opposite direction.
    peak = np.unravel_index(int(torch.argmax(corr)), tuple(corr.shape))
    return tuple(int(n // 2 - p) for p, n in zip(peak, corr.shape))


def translate(a, t):
    """Shift `a` by integer `t` along its FIRST len(t) axes: crop, then zero-pad the other end.

    `Sljiva.translate`'s semantics (src/utils.jl), NOT a circular shift: anatomy pushed out of
    the volume is dropped rather than wrapping around to the far side. `valid_after` says which
    slices are then real data instead of the zeros that came in.
    """
    for axis, c in enumerate(t):
        if c == 0:
            continue
        n = a.shape[axis]
        if abs(c) >= n:
            return torch.zeros_like(a)
        src = slice(c, n) if c > 0 else slice(0, n + c)
        piece = a[(slice(None),) * axis + (src,)]
        pad = torch.zeros(
            a.shape[:axis] + (n - piece.shape[axis],) + a.shape[axis + 1:],
            dtype=a.dtype, device=a.device)
        a = torch.cat([piece, pad] if c > 0 else [pad, piece], dim=axis)
    return a


def _source_index(M, out_dhw, dev):
    """-> (src_h, src_w, src_z), each (D, H, W): where each OUTPUT voxel reads from.

    `M` is `affine_matrix`'s, so it consumes (h, w, z, 1) -- see `stored_to_world` on why the
    index order is not the array's.
    """
    D, H, W = out_dhw
    M = torch.as_tensor(np.asarray(M, dtype=np.float32), device=dev)
    z = torch.arange(D, dtype=torch.float32, device=dev).view(D, 1, 1)
    h = torch.arange(H, dtype=torch.float32, device=dev).view(1, H, 1)
    w = torch.arange(W, dtype=torch.float32, device=dev).view(1, 1, W)
    # each term broadcasts to (D, H, W)
    return tuple(M[r, 0] * h + M[r, 1] * w + M[r, 2] * z + M[r, 3] for r in range(3))


def resample_plan(M, out_dhw, src_dhw, real, dev):
    """-> (grid, cover) for a resample by `M` from a `src_dhw` volume onto an `out_dhw` grid.

    `grid` is what `resample_array` feeds to grid_sample. `cover` is a (D, H, W) bool saying
    which output voxels are drawn from real acquired source data rather than invented at the
    edge -- the replacement for `translate`'s `valid_after`, which could be a per-slice flag
    only because a translation along z moves whole slices. Under a rotation validity varies
    within a slice, so it is tracked per voxel and the caller reduces it.

    `cover` comes from the SOURCE INDEX BOUNDS, exactly, never from the sampled values: after
    padding, an invented voxel and a genuinely dark one are indistinguishable.

    Geometry only -- no intensities are touched, so a survey can compute the kept slice range
    from the headers and one mask without reading any image data.
    """
    Ds, Hs, Ws = (int(n) for n in src_dhw)
    sh, sw, sz = _source_index(M, out_dhw, dev)
    inside = ((sh >= 0) & (sh <= Hs - 1) & (sw >= 0) & (sw <= Ws - 1)
              & (sz >= 0) & (sz <= Ds - 1))

    # ACQUIRED slices of the source. `real` is a function of the source z alone, so interpolate
    # it along z rather than grid_sample a whole volume of it. 0.999 rather than 0.5: a voxel
    # half-interpolated from a slice the study never acquired is not real data.
    zc = sz.clamp(0, max(Ds - 1, 0))
    z0 = zc.floor().long()
    z1 = (z0 + 1).clamp(max=max(Ds - 1, 0))
    f = zc - z0.to(zc.dtype)
    rf = torch.as_tensor(np.asarray(real, dtype=np.float32), device=dev)
    cover = inside & ((rf[z0] * (1.0 - f) + rf[z1] * f) >= 0.999)

    unit = lambda a, n: 2.0 * a / max(n - 1, 1) - 1.0
    grid = torch.stack([unit(sw, Ws), unit(sh, Hs), unit(sz, Ds)], dim=-1)[None]
    return grid, cover


def resample_array(x, grid, dev, pad="border"):
    """Apply a `resample_plan` grid to a (D, C, H, W) or (D, H, W) tensor."""
    flat = x.dim() == 3
    t = (x[:, None] if flat else x).permute(1, 0, 2, 3)[None].to(dev, torch.float32)
    o = F.grid_sample(t, grid, mode="bilinear", padding_mode=pad, align_corners=True)
    o = o[0].permute(1, 0, 2, 3)
    return o[:, 0] if flat else o


def resample_to_ref(v, M, out_dhw, real, dev):
    """Rigidly resample `v`'s raw / norm / fg / fov onto the reference grid. Mutates `v`.

    -> the plan's `cover`; see `resample_plan`.
    """
    dtypes = (v.raw.dtype, v.norm.dtype)
    src = (v.raw.shape[0], v.raw.shape[2], v.raw.shape[3])
    grid, cover = resample_plan(M, out_dhw, src, real, dev)
    # `border`, NOT `zeros`, for the intensities. The uncovered sliver sits at the frame
    # periphery, which is air in both studies, so border replication fills it with air. Zero
    # would be wrong for `norm`: median/MAD on an unmasked volume puts background at
    # -median/MAD, so 0 is a mid-tissue value there and zero-padding would paint a bright rim
    # exactly where the old masking artefacts were. `cover`, not the padding, records where the
    # data is real.
    v.raw = resample_array(v.raw, grid, dev, "border").to(dtypes[0]).cpu()
    v.norm = resample_array(v.norm, grid, dev, "border").to(dtypes[1]).cpu()
    # masks: zero-pad and threshold. Outside the source IS "not brain", and a mask has no
    # background offset to get wrong.
    v.fg = (resample_array(v.fg.to(torch.float32), grid, dev, "zeros") > 0.5).cpu()
    v.fov = (resample_array(v.fov.to(torch.float32), grid, dev, "zeros") > 0.5).cpu()
    return cover


def _register_affine(vols, cfg, ref, given):
    """Read each study's rigid transform off its header and RESAMPLE onto vols[ref]'s grid.

    Rotation and translation in one trilinear pass. No correlation, no peak to trust, and
    nothing is zero-padded into the middle of a volume: every study comes out on the
    reference's grid, so `slice_index` names one anatomical level across the patient -- the
    invariant the loader's guide_slice="index" needs, now actually delivered rather than
    approximated by an integer shift.

    Studies are NOT padded to a common depth first. That padding only existed because
    `translate` needs matching shapes; resampling reads the source at fractional indices and
    handles the bounds itself, so each study keeps its own depth as the source and the output
    is the reference's.
    """
    crop = int(getattr(cfg, "crop", 0) or 0)
    dev = torch.device(getattr(cfg, "reg_device", None) or "cpu")
    min_cover = float(getattr(cfg, "reg_min_cover", 0.99))
    out_dhw = (vols[ref].raw.shape[0],) + tuple(int(n) for n in vols[ref].raw.shape[2:])
    key = lambda v: (v.affine, v.native_hw, v.orig_depth)

    real = []
    for v, i in zip(vols, range(len(vols))):
        d0 = v.raw.shape[0]
        if given is None:
            ok = torch.ones(d0, dtype=torch.bool)
        else:
            ok = torch.as_tensor(np.asarray(given[i]), dtype=torch.bool)
            if ok.shape[0] < d0:
                ok = torch.cat([ok, torch.zeros(d0 - ok.shape[0], dtype=torch.bool)])
            ok = ok[:d0]
        real.append(ok)

    # THE REFERENCE'S BRAIN is the region a slice has to cover to count as usable. A frame-wide
    # coverage rule would fail every study for a reason that does not matter: rotating a square
    # by 6 deg empties its corners, which are air. Read before anything is resampled -- the
    # reference itself never is.
    roi = vols[ref].fg.to(dev)

    # The grid every study ends up on. write_h5 stores this as `affine` / `native_size`, because
    # after resampling a study's own header no longer addresses its pixels.
    grid = {"grid_affine": np.asarray(vols[ref].affine, dtype=np.float32),
            "grid_native_hw": tuple(int(n) for n in vols[ref].native_hw)}

    offsets, valid = [], []
    for i, v in enumerate(vols):
        if i == ref:
            v.reg = dict(grid, method="affine", matrix=np.eye(4, dtype=np.float32),
                         oblique=0.0, cover=1.0)
            offsets.append((0, 0, 0))
            valid.append(real[i].numpy())
            continue
        M, ob = affine_matrix(key(vols[ref]), key(v), crop)
        cover = resample_to_ref(v, M, out_dhw, real[i], dev)
        num = (cover & roi).flatten(1).sum(1).to(torch.float32)
        den = roi.flatten(1).sum(1).to(torch.float32)
        frac = torch.where(den > 0, num / den.clamp_min(1.0),
                           cover.flatten(1).to(torch.float32).mean(1))
        valid.append((frac >= min_cover).cpu().numpy())
        # reporting only: the single translation closest to this transform, so the manifest and
        # the summary stay readable. The DATA was moved by `matrix`, not by this.
        t, _ = affine_offset(key(vols[ref]), key(v), crop)
        v.reg = dict(grid, method="affine", matrix=np.asarray(M, dtype=np.float32), oblique=ob,
                     cover=float(cover.to(torch.float32).mean()))
        offsets.append(t)
        del cover
    return offsets, valid


def register_patient(vols, cfg, ref=0, real=None):
    """-> (offsets, valid) for one patient's studies, all put onto vols[ref]'s grid.

    `cfg.reg_method` picks how:

      centroid (default) dz by masked correlation, then one in-plane (dh, dw) that centres the
                        brain mask on the reference's. See `_register_centroid`.
      search            dz, then in-plane (dh, dw) -- one per volume, then per slice -- found by
                        searching the masked correlation of the images. See `_register_search`.
      affine            read the rigid transform off the headers and resample. Exact for a
                        rigid pair on a common grid, and it corrects ROTATION, which the
                        cohort has ~6.5 deg of between sessions.
      xcorr             the old 3-D FFT cross-correlation, integer translation only. Kept so
                        the two can be compared on real data, not argued about.

    `offsets` is (dz, dh, dw) per study either way -- under `affine` it is the equivalent
    centre translation and is REPORTING ONLY; the transform actually applied is on
    `vols[i].reg["matrix"]`, which `write_h5` stores.

    `valid[i]` marks the output slices of study i that hold real data: acquired in the source
    AND covered after the transform. Intersecting `valid` across the patient is what "truncate
    to the same slice range" is built on.
    """
    hw = {tuple(v.raw.shape[2:]) for v in vols}
    if len(hw) != 1:
        raise ValueError(f"studies are on different in-plane grids {sorted(hw)}; set --crop so "
                         f"they share one before registering")
    method = getattr(cfg, "reg_method", "centroid")
    if method == "centroid":
        return _register_centroid(vols, cfg, ref, real)
    if method == "search":
        return _register_search(vols, cfg, ref, real)
    if method == "affine":
        return _register_affine(vols, cfg, ref, real)
    return _register_xcorr(vols, cfg, ref, real)


def _register_xcorr(vols, cfg, ref=0, real=None):
    """TRANSLATION ONLY, by 3-D FFT cross-correlation. The pre-2026-09-24 default; now the
    `--reg-method xcorr` fallback, kept for comparison against `_register_affine`.

    It cannot correct rotation, and the cohort survey measured a median inter-session
    obliqueness of 0.113 (~6.5 deg, p90 worse), which displaces the edge of a 224 volume by
    ~13 voxels after the best possible shift. That is what this leaves on the table, and it is
    why `affine` is the default.

    Every study is first zero-padded at the END to the deepest one, so index 0 stays original
    slice 0 for all of them and the measured dz is a true inter-study offset rather than an
    artefact of differing depth. The offset is measured on the low-passed `reg_contrast` of the
    NORMALISED volume -- raw intensities are not comparable between scans, median/MAD ones are
    -- and the resulting (dz, dh, dw) is applied to raw / norm / fg / fov alike.

    `valid[i]` marks the slices of study i that hold REAL data afterwards: the study's own
    acquired extent, carried through the same shift. The padding a shallow study needed, and
    the zeros a shift pulled in, are both False -- so intersecting `valid` across the patient
    is what "truncate to the same slice range" has to be built on.

    The reference is not shifted, so afterwards every study of the patient lives on the
    reference's index axis and `slice_index` means one anatomical level across the whole
    patient -- exactly what the loader's guide_slice="index" assumes.
    """
    depth = max(v.raw.shape[0] for v in vols)
    ci = CONTRASTS.index(cfg.reg_contrast)

    # `real[i]` marks study i's genuinely acquired slices. Straight from the volume when the
    # caller says nothing (a builder volume is dense from 0), but scripts/register_nyumets.py
    # rebuilds volumes from h5s that were already slice-filtered, so their acquired set has
    # gaps and it passes the masks in explicitly.
    given, real = real, []
    for i, v in enumerate(vols):
        d0 = v.raw.shape[0]
        if given is None:
            ok = torch.zeros(depth, dtype=torch.bool)
            ok[:d0] = True
        else:
            ok = torch.as_tensor(given[i], dtype=torch.bool)
            if ok.shape[0] < depth:
                ok = torch.cat([ok, torch.zeros(depth - ok.shape[0], dtype=torch.bool)])
        real.append(ok)
        if depth > d0:
            for name in ("raw", "norm", "fg", "fov"):
                a = getattr(v, name)
                z = torch.zeros((depth - d0,) + a.shape[1:], dtype=a.dtype,
                                device=a.device)
                setattr(v, name, torch.cat([a, z], dim=0))

    # Only the PROBE moves to the accelerator. The bulk arrays stay put: they are only
    # sliced and concatenated by `translate`, which is memory-bound either way, while the
    # 3-D FFTs are what actually pay for a GPU.
    dev = torch.device(getattr(cfg, "reg_device", None) or "cpu")
    phase = bool(getattr(cfg, "reg_phase", True))
    # WHICH ARRAY. `norm` (img_median_mad) is brain-masked -- cmap.normalize_masked ends with
    # `out = out * fg`, so it is EXACTLY zero outside the mask, and the mask is brain & fov, so
    # it carries each study's own FOV cut. Correlating it aligns mask shapes as much as
    # anatomy. `raw` is the whole unmasked head but carries a large DC, which only plain
    # correlation cares about (phase correlation whitens it away). Measure both with
    # scripts/register_nyumets.py --compare rather than guessing.
    src = getattr(cfg, "reg_source", "norm")
    pick = (lambda v: v.raw) if src == "raw" else (lambda v: v.norm)
    probe = lambda v: lowpass_inplane(pick(v)[:, ci].to(dev), cfg.reg_lowpass)
    fixed = probe(vols[ref])

    offsets, valid = [], []
    for i, v in enumerate(vols):
        t = (0, 0, 0) if i == ref else translation_offset(fixed, probe(v), phase=phase)
        if i != ref and cfg.reg_max_shift and max(abs(c) for c in t) > cfg.reg_max_shift:
            print(f"  ?? offset {t} exceeds --reg-max-shift {cfg.reg_max_shift}; "
                  f"leaving this study unshifted")
            t = (0, 0, 0)
        if any(t):
            # The shifts themselves go to the accelerator too when there is one. A crop and a
            # zero-pad is memory-bound, so this is worth the transfer only because the arrays
            # are large (4 contrasts at full depth); on CPU `to`/`back` are both no-ops.
            to = (lambda a: a.to(dev)) if dev.type == "cuda" else (lambda a: a)
            back = (lambda a: a.cpu()) if dev.type == "cuda" else (lambda a: a)
            v.raw = back(translate(to(v.raw), (t[0], 0, t[1], t[2])))
            v.norm = back(translate(to(v.norm), (t[0], 0, t[1], t[2])))
            v.fg = back(translate(to(v.fg), t))
            v.fov = back(translate(to(v.fov), t))
            real[i] = translate(real[i], (t[0],))
        # the same record `_register_affine` leaves, so write_h5 stores one shape of attrs
        # whichever method ran. A translation IS a 4x4, in (h, w, z) order like `affine_matrix`'s.
        T = np.eye(4, dtype=np.float32)
        T[0, 3], T[1, 3], T[2, 3] = t[1], t[2], t[0]
        v.reg = {"method": "xcorr", "matrix": T, "oblique": 0.0}
        offsets.append(t)
        valid.append(real[i].numpy())
    return offsets, valid


def masked_ncc(a, b, ma, mb, t=(0, 0, 0), stride=1, min_px=1000):
    """Correlation of `a` against `translate(b, t)`, inside BOTH brain masks. NaN if they overlap
    in fewer than `min_px` (strided) voxels.

    (D, H, W) tensors; `t` is (dz, dh, dw) with `translate`'s semantics, so the `t` that maximises
    this is exactly the shift that puts b onto a. `stride` subsamples in-plane only -- a speed knob,
    never along z, which is the axis the search most needs to resolve.
    """
    if a.shape != b.shape:
        raise ValueError(f"masked_ncc needs one grid: {tuple(a.shape)} vs {tuple(b.shape)}")
    # a[i] against b[i + t] over their OVERLAP, as views: identical to translating b (whose
    # zero-filled edge the mask excludes anyway) without copying the volume for every shift
    sa, sb = [], []
    for ax, (n, d) in enumerate(zip(a.shape, t)):
        st = 1 if ax == 0 else stride
        sa.append(slice(max(0, -d), n - max(0, d), st))
        sb.append(slice(max(0, d), n - max(0, -d), st))
    sa, sb = tuple(sa), tuple(sb)
    j = ma[sa] & mb[sb]
    if int(j.sum()) < min_px:
        return float("nan")
    x, y = a[sa][j].float(), b[sb][j].float()
    x, y = x - x.mean(), y - y.mean()
    return float((x * y).sum() / (x.norm() * y.norm()).clamp_min(1e-8))


def _inplane_search(a, ma, b, mb, dz, shifts, stride, min_px):
    """Score every (dh, dw) in `shifts` at a fixed dz, by masked NCC.

    -> (global (dh, dw), per-slice best (D, 2) long, per-slice ok (D,) bool). One pass serves both
    answers: per-slice sums give each slice its own best, and their totals give the volume's, so the
    global and per-slice shifts are scored identically. A slice gets its own shift only where the
    two brains overlap in >= `min_px` strided pixels.
    """
    S = max(max(abs(dh), abs(dw)) for dh, dw in shifts)
    D = a.shape[0]
    full_t = torch.zeros((D, 2), dtype=torch.long)
    full_ok = torch.zeros(D, dtype=torch.bool)
    # Only the REFERENCE brain's bounding box can contribute (the joint mask requires it), so the
    # search runs on that box alone -- exact, and roughly half the voxels of the full frame.
    nz = lambda m: torch.nonzero(m).flatten()
    zs, hs, ws = nz(ma.any(2).any(1)), nz(ma.any(2).any(0)), nz(ma.any(1).any(0))
    if zs.numel() == 0:
        return (0, 0), full_t, full_ok
    z0, z1 = int(zs[0]), int(zs[-1]) + 1
    h0, h1 = int(hs[0]), int(hs[-1]) + 1
    w0, w1 = int(ws[0]), int(ws[-1]) + 1
    pad = lambda v: F.pad(translate(v, (dz, 0, 0))[z0:z1].float()[None], (S, S, S, S))[0]
    bp, mp = pad(b), pad(mb) > 0.5
    A = a[z0:z1, h0:h1:stride, w0:w1:stride].float()
    MA = ma[z0:z1, h0:h1:stride, w0:w1:stride]
    best_s = torch.full((z1 - z0,), -2.0, device=a.device)
    best_t = torch.zeros((z1 - z0, 2), dtype=torch.long, device=a.device)
    ok_any = torch.zeros(z1 - z0, dtype=torch.bool, device=a.device)
    g_best, g_t = -2.0, (0, 0)
    for dh, dw in shifts:
        # strided VIEW of translate(b, (dz, dh, dw)) over the box -- no copy per shift
        hsl = slice(S + h0 + dh, S + h1 + dh, stride)
        wsl = slice(S + w0 + dw, S + w1 + dw, stride)
        B = bp[:, hsl, wsl]
        J = (MA & mp[:, hsl, wsl]).float()
        n = J.sum((1, 2))
        sx, sy = (A * J).sum((1, 2)), (B * J).sum((1, 2))
        sxx, syy, sxy = (A * A * J).sum((1, 2)), (B * B * J).sum((1, 2)), (A * B * J).sum((1, 2))
        ok = n >= min_px
        nn = n.clamp_min(1)
        cov = sxy - sx * sy / nn
        var = ((sxx - sx ** 2 / nn) * (syy - sy ** 2 / nn)).clamp_min(1e-12)
        c = torch.where(ok, cov / var.sqrt(), torch.full_like(cov, -2.0))
        better = c > best_s
        best_s = torch.where(better, c, best_s)
        best_t[better] = torch.tensor([dh, dw], device=a.device)
        ok_any |= ok
        N = float(n.sum())
        if N >= 1000:
            cv = float(sxy.sum() - sx.sum() * sy.sum() / N)
            vv = float((sxx.sum() - sx.sum() ** 2 / N) * (syy.sum() - sy.sum() ** 2 / N))
            gs = cv / max(vv, 1e-12) ** 0.5
            if gs > g_best:
                g_best, g_t = gs, (dh, dw)
    full_t[z0:z1] = best_t.cpu()
    full_ok[z0:z1] = ok_any.cpu()
    return g_t, full_t, full_ok


def _blur_inplane(v, r):
    """(D, H, W) -> the same, box-blurred twice in-plane with a (2r+1)^2 window (~triangular,
    sigma ~ 0.8 r). Signed-safe, unlike lowpass_inplane's magnitude, which would fold the negative
    half of a median/MAD image onto the positive. Only the coarse search stages see this."""
    x = v[:, None].float()
    for _ in range(2):
        x = F.avg_pool2d(x, 2 * r + 1, 1, r, count_include_pad=False)
    return x[:, 0]


def _peaks(scores, k, sep):
    """Indices of the k highest local maxima of a 1-D score list, at least `sep` apart."""
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    out = []
    for i in order:
        if all(abs(i - j) >= sep for j in out):
            out.append(i)
        if len(out) == k:
            break
    return out


def _smooth_slice_shifts(t, ok, window):
    """Median-filter per-slice shifts along z over the slices that had a shift of their own; the
    rest take the nearest smoothed value. -> (D, 2) long."""
    idx = torch.nonzero(ok).flatten()
    v = t[idx].float()
    h = max(int(window), 1) // 2
    sm = torch.stack([v[max(0, i - h):i + h + 1].median(0).values for i in range(len(v))])
    near = (torch.arange(t.shape[0])[:, None] - idx[None]).abs().argmin(1)
    return sm[near].round().long()


def _shift_slices(a, dz, shifts):
    """translate `a` by dz along its first axis, then each slice z in-plane by shifts[z].

    `a` is (D, ..., H, W): the in-plane shift lands on the LAST two axes whatever sits between
    (the contrast axis of raw / norm, nothing for fg / fov).
    """
    az = translate(a, (dz,)) if dz else a
    lead = (0,) * (a.dim() - 3)
    out = torch.empty_like(az)
    for z in range(az.shape[0]):
        dh, dw = (int(v) for v in shifts[z])
        out[z] = translate(az[z], lead + (dh, dw)) if (dh or dw) else az[z]
    return out


def _pad_studies(vols, given=None):
    """Zero-pad every study at the END to the deepest, IN PLACE (raw / norm / fg / fov), so index 0
    stays original slice 0 for all of them. -> per-study (depth,) bool of genuinely acquired slices:
    `given[i]` when the caller knows them (register_nyumets rebuilds volumes with gaps), else each
    study's own extent."""
    depth = max(v.raw.shape[0] for v in vols)
    real = []
    for i, v in enumerate(vols):
        d0 = v.raw.shape[0]
        if given is None:
            ok = torch.zeros(depth, dtype=torch.bool)
            ok[:d0] = True
        else:
            ok = torch.as_tensor(np.asarray(given[i]), dtype=torch.bool)
            if ok.shape[0] < depth:
                ok = torch.cat([ok, torch.zeros(depth - ok.shape[0], dtype=torch.bool)])
        real.append(ok)
        if depth > d0:
            for name in ("raw", "norm", "fg", "fov"):
                a = getattr(v, name)
                setattr(v, name, torch.cat(
                    [a, torch.zeros((depth - d0,) + a.shape[1:], dtype=a.dtype)], dim=0))
    return real


def find_dz(a, ma, b, mb, max_dz, stride=2, blur_r=4):
    """Through-plane shift of b onto a, in-plane held at 0: the masked-correlation maximum over
    [-max_dz, max_dz] on in-plane-BLURRED images, refined +-2 at full resolution.

    Blurred because an unknown in-plane offset of a few px wrecks full-resolution correlation and
    can put the maximum on the wrong level (a 6 px offset did in the tests); blurred, the profile
    barely cares where the brain sits in-plane.
    """
    ab, bb = _blur_inplane(a, blur_r), _blur_inplane(b, blur_r)
    sc = lambda x, y, d: np.nan_to_num(masked_ncc(x, y, ma, mb, (d, 0, 0), stride), nan=-2.0)
    d0 = max(range(-max_dz, max_dz + 1), key=lambda d: sc(ab, bb, d))
    return max(range(d0 - 2, d0 + 3), key=lambda d: sc(a, b, d))


def centroid_shift(ma, mb, dz):
    """(dh, dw) that puts b's brain-mask centroid on a's, over the slices both hold after dz.

    `translate(b, t)` sets out[i] = b[i + t], which moves b's centroid by -t, so t = c_b - c_a.
    Cheap and search-free; biased wherever the two masks genuinely differ (coverage, a FOV or
    skull-strip cut, a large lesion), which the 2026-09-25 viewer comparison found close enough
    to a correlation search to prefer it.
    """
    mbz = translate(mb, (dz,))
    both = ma.flatten(1).any(1) & mbz.flatten(1).any(1)
    if not bool(both.any()):
        return 0, 0
    H, W = ma.shape[1:]
    hh = torch.arange(H, dtype=torch.float64, device=ma.device)[:, None]
    ww = torch.arange(W, dtype=torch.float64, device=ma.device)[None, :]

    def c(m):
        m = m[both].double()
        n = m.sum().clamp_min(1)
        return float((m * hh).sum() / n), float((m * ww).sum() / n)

    (ah, aw), (bh, bw) = c(ma), c(mbz)
    return int(round(bh - ah)), int(round(bw - aw))


def _register_centroid(vols, cfg, ref=0, real=None):
    """dz by masked correlation, then ONE in-plane (dh, dw) per study that centres its brain mask
    on the reference's. Translation only; no in-plane search.

    Chosen 2026-09-25 after comparing, on the source NIfTIs, dz alone (visibly off in-plane on
    about half the patients), dz + brain-centroid shift, and dz + a correlation search for (dh, dw):
    centroid and search landed close, and centroid costs one mask reduction instead of a few
    hundred correlation evaluations. `--reg-method search` remains for the searched version,
    optionally per-slice.

      1. dz: `find_dz` -- blurred masked-correlation scan over +-reg_max_dz, refined +-2
      2. (dh, dw): `centroid_shift` over the slices both brains occupy after dz
      3. dz refined +-2 once more with that in-plane shift in place (cheap; the scan in 1 was
         blind to the in-plane offset)

    Same bookkeeping as `_register_search`: studies padded at the END, shifts applied to raw /
    norm / fg / fov alike, `slice_shift` (uniform here) and the final masked `ncc` in v.reg.
    """
    ci = CONTRASTS.index(getattr(cfg, "reg_contrast", "T1"))
    dev = torch.device(getattr(cfg, "reg_device", None) or "cpu")
    stride = int(getattr(cfg, "reg_stride", 2))
    max_dz = int(getattr(cfg, "reg_max_dz", 60))
    real = _pad_studies(vols, real)
    depth = vols[ref].raw.shape[0]

    A = vols[ref].norm[:, ci].to(dev, torch.float32)
    MA = vols[ref].fg.to(dev).bool()
    offsets, valid = [], []
    for i, v in enumerate(vols):
        if i == ref:
            v.reg = {"method": "centroid", "matrix": np.eye(4, dtype=np.float32),
                     "oblique": 0.0, "ncc": 1.0,
                     "slice_shift": np.zeros((depth, 2), dtype=np.int64)}
            offsets.append((0, 0, 0))
            valid.append(real[i].numpy())
            continue
        B = v.norm[:, ci].to(dev, torch.float32)
        MB = v.fg.to(dev).bool()
        dz = find_dz(A, MA, B, MB, max_dz, stride)
        dh, dw = centroid_shift(MA, MB, dz)
        dz = max(range(dz - 2, dz + 3), key=lambda d: np.nan_to_num(
            masked_ncc(A, B, MA, MB, (d, dh, dw), stride), nan=-2.0))

        v.raw = translate(v.raw, (dz, 0, dh, dw))
        v.norm = translate(v.norm, (dz, 0, dh, dw))
        v.fg = translate(v.fg, (dz, dh, dw))
        v.fov = translate(v.fov, (dz, dh, dw))
        real[i] = translate(real[i], (dz,))

        T = np.eye(4, dtype=np.float32)
        T[0, 3], T[1, 3], T[2, 3] = dh, dw, dz              # (h, w, z) order, like affine_matrix
        v.reg = {"method": "centroid", "matrix": T, "oblique": 0.0,
                 "ncc": masked_ncc(A, v.norm[:, ci].to(dev, torch.float32), MA,
                                   v.fg.to(dev).bool(), (0, 0, 0), stride),
                 "slice_shift": np.tile(np.asarray([[dh, dw]], dtype=np.int64), (depth, 1))}
        offsets.append((dz, dh, dw))
        valid.append(real[i].numpy())
    return offsets, valid


def _register_search(vols, cfg, ref=0, real=None):
    """dz, then in-plane (dh, dw), found by SEARCHING the masked correlation. Translation only.

    Chosen over `affine` and `xcorr` on the 2026-09-25 comparison (27 study pairs, 6 patients,
    source NIfTIs, scored by masked CT1 correlation, which the search never sees):

        none 0.060 | xcorr 3-D 0.353 | dz + global 0.465 | dz + per-slice (smoothed) 0.485

    Header affines had ~0 correlation after resampling: scanner coordinates are not anatomical
    across sessions. The FFT correlation peak occasionally locks onto the wrong lag (dh = 22 on one
    pair, scoring 0.06 where the search found 0.44). A search cannot do that: it maximises the same
    score that judges the result.

    STEPS, all on the normalised `reg_contrast` inside the brain masks, in-plane strided by
    `reg_stride`:
      1. dz in [-reg_max_dz, reg_max_dz], in-plane held at 0, on in-plane-BLURRED images, keeping
         the 3 best-separated peaks. At full resolution an unknown in-plane offset of a few px
         wrecks the correlation and can put the maximum on the wrong level (a 6 px offset did in
         the tests); blurred, the profile barely cares about in-plane position
      2. one (dh, dw) per candidate, +-reg_search px: a step-4 grid on the blurred images, then
         +-2 at full resolution. The candidate with the best full-resolution score wins
      3. dz again, +-4 around the winner, at its (dh, dw)
      4. joint +-1 hill-climb on (dz, dh, dw): sequential searches stop a voxel short where the
         axes trade off
      5. (reg_slicewise) every reference slice gets its own (dh, dw) within +-reg_slice_radius of
         the global shift, median-filtered along z over reg_smooth slices. It follows the in-plane
         drift a between-session head TILT produces, which no single shift can; it cannot undo a
         rotation WITHIN the axial plane. Smoothing is what keeps it from fitting noise -- on the
         comparison it changed CT1 by < 0.01, so the per-slice shifts were tracking a real trend.

    The shifts are applied to raw / norm / fg / fov alike. As with xcorr, every study is first
    zero-padded at the END to the deepest, so index 0 stays original slice 0 and `valid` (the
    study's acquired slices carried through dz) is what the common slice range is built on.
    `v.reg["slice_shift"]` is the (D, 2) per-OUTPUT-slice in-plane shift actually applied;
    `v.reg["ncc"]` the final masked correlation, for write_h5 to record and the caller to gate on.
    """
    depth = max(v.raw.shape[0] for v in vols)
    ci = CONTRASTS.index(getattr(cfg, "reg_contrast", "T1"))
    dev = torch.device(getattr(cfg, "reg_device", None) or "cpu")
    stride = int(getattr(cfg, "reg_stride", 2))
    max_dz = int(getattr(cfg, "reg_max_dz", 60))
    S = int(getattr(cfg, "reg_search", 16))
    R = int(getattr(cfg, "reg_slice_radius", 6))
    min_px = int(getattr(cfg, "reg_slice_min_px", 400))
    slicewise = bool(getattr(cfg, "reg_slicewise", True))
    window = int(getattr(cfg, "reg_smooth", 7))
    real = _pad_studies(vols, real)

    blur_r = max(2, S // 4)
    n_cand = 3
    coarse = [(dh, dw) for dh in range(-S, S + 1, 4) for dw in range(-S, S + 1, 4)]
    A = vols[ref].norm[:, ci].to(dev, torch.float32)
    MA = vols[ref].fg.to(dev).bool()
    Ab = _blur_inplane(A, blur_r)
    zero = torch.zeros((depth, 2), dtype=torch.long)
    offsets, valid = [], []
    for i, v in enumerate(vols):
        if i == ref:
            v.reg = {"method": "search", "matrix": np.eye(4, dtype=np.float32), "oblique": 0.0,
                     "ncc": 1.0, "slice_shift": zero.numpy()}
            offsets.append((0, 0, 0))
            valid.append(real[i].numpy())
            continue
        B = v.norm[:, ci].to(dev, torch.float32)
        MB = v.fg.to(dev).bool()
        Bb = _blur_inplane(B, blur_r)
        score = lambda t: np.nan_to_num(masked_ncc(A, B, MA, MB, t, stride), nan=-2.0)
        score_b = lambda t: np.nan_to_num(masked_ncc(Ab, Bb, MA, MB, t, stride), nan=-2.0)

        def global_at(d):
            """Whole-volume (dh, dw) at dz = d: coarse grid on the blurred images, then +-2 at
            full resolution around the winner."""
            g0, _, _ = _inplane_search(Ab, MA, Bb, MB, d, coarse, stride, min_px)
            fine = [(g0[0] + a, g0[1] + b) for a in range(-2, 3) for b in range(-2, 3)]
            return _inplane_search(A, MA, B, MB, d, fine, stride, min_px)[0]

        # 1. dz with in-plane held at 0, on the BLURRED images: an unknown in-plane offset of
        #    several px wrecks full-resolution correlation and can put the maximum on the wrong
        #    level entirely. Keep the top candidates and let the in-plane search judge them.
        dzs = list(range(-max_dz, max_dz + 1))
        prof = [score_b((d, 0, 0)) for d in dzs]
        cands = [dzs[i] for i in _peaks(prof, n_cand, 3)]
        # 2. an in-plane shift for each candidate; the best full-resolution score wins
        best = max(((d, global_at(d)) for d in cands), key=lambda dg: score((dg[0],) + dg[1]))
        dz, g = best
        # 3. dz again around the winner, at its in-plane shift
        dz = max(range(dz - 4, dz + 5), key=lambda d: score((d,) + tuple(g)))
        g = global_at(dz)
        cur = (dz, g[0], g[1])
        cur_s = score(cur)
        while True:
            nb = max(((cur[0] + a, cur[1] + b, cur[2] + c) for a in (-1, 0, 1)
                      for b in (-1, 0, 1) for c in (-1, 0, 1)), key=score)
            if score(nb) <= cur_s:
                break
            cur, cur_s = nb, score(nb)
        dz, g = cur[0], (cur[1], cur[2])

        shifts = torch.tensor([g], dtype=torch.long).expand(depth, 2).clone()
        if slicewise:
            local = [(g[0] + a, g[1] + b) for a in range(-R, R + 1) for b in range(-R, R + 1)]
            _, per, ok = _inplane_search(A, MA, B, MB, dz, local, stride, min_px)
            if bool(ok.any()):
                shifts = _smooth_slice_shifts(per, ok, window)

        for name in ("raw", "norm", "fg", "fov"):
            setattr(v, name, _shift_slices(getattr(v, name), dz, shifts))
        real[i] = translate(real[i], (dz,))

        T = np.eye(4, dtype=np.float32)
        T[0, 3], T[1, 3], T[2, 3] = g[0], g[1], dz          # (h, w, z) order, like affine_matrix
        v.reg = {"method": "search", "matrix": T, "oblique": 0.0,
                 "ncc": masked_ncc(A, v.norm[:, ci].to(dev, torch.float32), MA,
                                   v.fg.to(dev).bool(), (0, 0, 0), stride),
                 "slice_shift": shifts.numpy()}
        offsets.append((dz, g[0], g[1]))
        valid.append(real[i].numpy())
    return offsets, valid


def _morph(m, r, op):
    """2D per-slice dilate/erode of a bool mask with a (2r+1)^2 box."""
    x = m[:, None].float()
    if op == "dilate":
        x = F.max_pool2d(x, 2 * r + 1, 1, r)
    else:
        x = -F.max_pool2d(-x, 2 * r + 1, 1, r)
    return x[:, 0] > 0.5


def support_mask(img, frac, r):
    """(D,C,H,W) -> (D,C,H,W) bool: where each contrast actually has ACQUIRED data.

    This is a different question from "is this tissue", and needs a different rule. The
    threshold is low -- just above the air noise floor -- so that dark-but-acquired voxels
    (CSF on T1) stay inside the support; an opening then drops isolated noise speckle and a
    closing fills interior holes, so the result is a field-of-view region rather than a
    tissue segmentation.
    """
    out = torch.zeros(img.shape, dtype=torch.bool)
    for c in range(img.shape[1]):
        v = img[:, c].float()
        flat = v.reshape(-1)
        step = max(1, flat.numel() // 2_000_000)          # torch.quantile caps around 16M
        hi = torch.quantile(flat[::step], 0.995).clamp_min(1e-6)
        m = v > frac * hi
        if r > 0:
            m = _morph(_morph(m, r, "erode"), r, "dilate")     # opening: drop speckle
            m = _morph(_morph(m, r, "dilate"), r, "erode")     # closing: fill holes
        out[:, c] = m
    return out


def brain_mask(img, bg_frac):
    """(D,C,H,W) -> (D,H,W) bool.

    NYUMets is not skull-stripped, so there is no `> 0` rule to lean on. Scale each
    contrast by its own p99.5, AVERAGE across contrasts, then threshold. Averaging rather
    than intersecting keeps CSF, which is dark on T1 but bright on T2/FLAIR and would be
    cut by an all-contrasts-must-agree rule. This is a proxy -- swap in HD-BET or
    SynthStrip masks later if the tissue statistics need to be exact.
    """
    acc = torch.zeros((img.shape[0],) + tuple(img.shape[2:]), dtype=torch.float32)
    for c in range(img.shape[1]):
        v = img[:, c].float()
        flat = v.reshape(-1)
        step = max(1, flat.numel() // 2_000_000)          # torch.quantile caps around 16M
        hi = torch.quantile(flat[::step], 0.995).clamp_min(1e-6)
        acc += v / hi
    return (acc / img.shape[1]) > bg_frac


def build_volume(files, cfg):
    """One session -> a FULL-DEPTH `Volume`, in-plane cropped, not yet slice-filtered.

    Registration across a patient's studies runs on these full volumes (that is the whole
    reason the slice filtering is a separate step): a study whose h5 had already been cut to
    its own brain-bearing slices could only be aligned to another such cut, and the through-
    plane offset would be confounded with the difference in what each cut kept.
    """
    vols, affines = zip(*(load_ras(files[c]) for c in CONTRASTS))
    if len({v.shape for v in vols}) != 1:
        raise ValueError(f"shapes differ: {[v.shape for v in vols]} -- not on a common grid")
    if cfg.require_affine and not all(np.allclose(a, affines[0], atol=1e-3) for a in affines):
        raise ValueError("affines differ -- contrasts are not co-registered")

    img = torch.from_numpy(np.stack([v.transpose(2, 0, 1) for v in vols], axis=1))  # (D,C,H,W)

    # COMMON SUPPORT. Contrasts of the same session do not always cover the same anatomy --
    # a T1 whose FOV includes the eyes paired with a CT1 whose FOV does not is the usual
    # case. The intersection of the per-contrast supports is the region all four share.
    #
    # This cannot come from brain_mask: that AVERAGES the contrasts, so an eye voxel carrying
    # signal in one of four contrasts still lands at ~1/4 of its normalised value, well above
    # bg_frac, and survives.
    #
    # `store` (default) records it and leaves the pixels alone; `apply` multiplies it in,
    # which bakes the ragged FOV boundary into the data. See the module docstring.
    lost = {}
    if cfg.support != "none":
        sup = support_mask(img, cfg.support_frac, cfg.support_close)
        fov = sup.all(dim=1)                                   # (D,H,W)
        for c, name in enumerate(CONTRASTS):
            own = sup[:, c].sum()
            lost[name] = float(1.0 - (fov & sup[:, c]).sum() / own.clamp_min(1))
        if cfg.support == "apply":
            img = img * fov.unsqueeze(1).to(img.dtype)
    else:
        fov = torch.ones(img.shape[:1] + img.shape[2:], dtype=torch.bool)

    # `& fov` even under `--support store`: the pixels outside the common FOV are kept, but a
    # median/MAD taken over voxels only some contrasts acquired would not be comparable across
    # channels. So the STATISTICS (and the stored brain mask) stay on the shared region.
    fg = brain_mask(img, cfg.bg_frac) & fov
    clipped = (channelwise_percentile_clip(img, fg, cfg.clip[0], cfg.clip[1])
               if cfg.clip else img)
    # mask_output=False: NOTHING is masked into the stored pixels. `fg` sets the statistics
    # only. See the module docstring -- a masked array cannot be used to judge alignment, and it
    # puts a hard mask edge in every training target.
    norm, stats = normalize_masked(clipped, fg, mode=MODE, mask_output=False)

    raw, orig_depth = img, img.shape[0]
    native_hw = tuple(int(v) for v in img.shape[2:4])
    if cfg.crop:
        raw = center_crop_or_pad(raw, cfg.crop, h_axis=2, w_axis=3)
        norm = center_crop_or_pad(norm, cfg.crop, h_axis=2, w_axis=3)
        fg = center_crop_or_pad(fg, cfg.crop, h_axis=1, w_axis=2)
        fov = center_crop_or_pad(fov, cfg.crop, h_axis=1, w_axis=2)

    return Volume(raw=raw, norm=norm, fg=fg, fov=fov, stats=stats, orig_depth=orig_depth,
                  affine=affines[0], lost=lost, native_hw=native_hw)


def keep_mask(vol, cfg):
    """Per-slice boolean over the FULL depth: enough brain, and inside --start/--end."""
    frac = vol.fg.reshape(vol.fg.shape[0], -1).float().mean(dim=1).numpy()
    keep = frac >= cfg.min_brain_frac
    if cfg.start is not None or cfg.end is not None:
        z = np.arange(vol.fg.shape[0])
        hi = cfg.end if cfg.end is not None else vol.fg.shape[0]
        keep &= (z >= (cfg.start or 0)) & (z < hi)
    return keep


def select_slices(vol, idx):
    """A registered, full-depth Volume + the kept indices -> the arrays write_h5 stores."""
    to_hwc = lambda t: np.transpose(t.numpy()[idx], (0, 2, 3, 1))
    return (to_hwc(vol.raw), to_hwc(vol.norm), vol.fg.numpy()[idx][..., None],
            vol.fov.numpy()[idx][..., None])


def write_h5(path, raw, norm, mask, support, idx, stats, orig_depth, affine, key,
             files, cfg, lost, native_hw, reg=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    comp = "gzip" if cfg.compress else None
    chunk = lambda a: (1,) + a.shape[1:]
    with h5py.File(path, "w") as f:
        f.create_dataset("img_raw", data=raw.astype(np.float32),
                         chunks=chunk(raw), compression=comp)
        f.create_dataset(norm_key(MODE), data=norm.astype(np.float32),
                         chunks=chunk(norm), compression=comp)
        # `img` is the name every existing loader reaches for; a soft link keeps that working
        # without a second copy of the pixels on disk.
        f["img"] = h5py.SoftLink("/" + norm_key(MODE))
        f.create_dataset("mask", data=mask.astype(np.uint8),
                         chunks=chunk(mask), compression=comp)
        # the common acquisition support: under the default `--support store` the pixels are
        # NOT multiplied by it, so this is the region to mask WITH downstream if you want to
        # -- check `support_applied` before assuming either way
        f.create_dataset("support", data=support.astype(np.uint8),
                         chunks=chunk(support), compression=comp)
        f.create_dataset("slice_index", data=idx.astype(np.int32))
        f.attrs["subject_id"] = f"{key[0]}_{key[1]}"
        f.attrs["patient"], f.attrs["session"] = key
        f.attrs["contrasts"] = ",".join(CONTRASTS)
        f.attrs["normalize"] = MODE
        f.attrs[f"norm_stats_{MODE.replace('-', '_')}"] = stats      # (C, 2) centre/scale
        f.attrs["norm_stats"] = stats
        f.attrs["clip_percentiles"] = np.asarray(cfg.clip or (-1, -1), dtype=np.float32)
        f.attrs["crop_size"] = int(cfg.crop) if cfg.crop else -1
        # `affine` and `native_size` describe THE GRID THE STORED PIXELS ARE ON. Under
        # reg_method="affine" that is the patient's REFERENCE study's grid, not this session's
        # own -- the pixels were resampled onto it, so its own affine no longer addresses them,
        # and anything recomputing geometry from these attrs (scripts/register_nyumets.py
        # --geometry, stored_to_world, affine_offset) has to see the grid the data is actually
        # on or it will report aligned data as misaligned. The session's own values are kept
        # alongside as provenance.
        _r = reg or {}
        _grid_aff = _r.get("grid_affine")
        _grid_hw = _r.get("grid_native_hw")
        # native in-plane size BEFORE the crop/pad, so a padded session is identifiable
        f.attrs["native_size"] = np.asarray(
            _grid_hw if _grid_hw is not None else native_hw, dtype=np.int32)
        f.attrs["native_size_source"] = np.asarray(native_hw, dtype=np.int32)
        f.attrs["orig_depth"] = int(orig_depth)
        f.attrs["min_brain_frac"] = float(cfg.min_brain_frac)
        f.attrs["mask_rule"] = f"mean_c(v/p99.5_c) > {cfg.bg_frac}"
        # NO mask is multiplied into any stored array: `mask` and `support` are recorded for
        # downstream use, never applied. Background is therefore NOT 0 in img_median_mad.
        f.attrs["background_masked"] = False
        f.attrs["support_mode"] = cfg.support                     # store | apply | none
        f.attrs["support_applied"] = bool(cfg.support == "apply")  # were the pixels zeroed?
        # kept under its old name so anything written against the pre-2026-09-22 h5s still
        # reads something sensible -- but it says COMPUTED, which is no longer the same as
        # APPLIED; `support_applied` is the one that describes the pixels.
        f.attrs["intersect_support"] = bool(cfg.support != "none")
        f.attrs["support_rule"] = (f"intersect_c(open/close(v_c > {cfg.support_frac}*p99.5_c), "
                                   f"r={cfg.support_close})" if cfg.support != "none" else "none")
        # fraction of each contrast's OWN support dropped by the intersection: a large value
        # on one contrast is the FOV mismatch (eyes in T1, absent in CT1) this guards against
        for _c in CONTRASTS:
            f.attrs[f"support_lost_{_c}"] = float(lost.get(_c, 0.0))
        f.attrs["affine"] = np.asarray(
            _grid_aff if _grid_aff is not None else affine, dtype=np.float32)
        f.attrs["affine_source"] = np.asarray(affine, dtype=np.float32)
        f.attrs["source_files"] = ",".join(os.path.basename(files[c]) for c in CONTRASTS)
        # Cross-study registration. After it, all studies of a patient share one index axis and
        # one slice set, so `slice_index` names the same anatomical level in each.
        #
        # `reg_matrix` is the authoritative record: the 4x4 that took an OUTPUT (reference-grid)
        # index (h, w, z, 1) to the SOURCE index this study was sampled at. `reg_offset` is only
        # the closest single translation to it, kept because a manifest column has to be a
        # number -- under reg_method="affine" it does NOT describe what was done to the pixels,
        # because a rotation was corrected too.
        r = _r
        f.attrs["reg_applied"] = bool(r.get("applied", False))
        f.attrs["reg_method"] = str(r.get("method", "none"))
        f.attrs["reg_offset"] = np.asarray(r.get("offset", (0, 0, 0)), dtype=np.int32)
        f.attrs["reg_matrix"] = np.asarray(r.get("matrix", np.eye(4)), dtype=np.float32)
        # obliqueness this study was rotated by relative to the reference, ~0.0175 per degree.
        # Under "affine" it was CORRECTED; under "xcorr" it was left in the data.
        f.attrs["reg_oblique"] = float(r.get("oblique", 0.0))
        # fraction of the output volume drawn from real acquired source data; NaN when the
        # method does not track it (xcorr zero-fills instead)
        f.attrs["reg_cover"] = float(r.get("cover", float("nan")))
        f.attrs["reg_reference"] = str(r.get("reference", ""))
        # masked correlation of reg_contrast with the reference AFTER registration (search only;
        # NaN otherwise). The quality gate: a low value is a study that did not align.
        f.attrs["reg_ncc"] = float(r.get("ncc", float("nan")))
        if r.get("slice_shift") is not None:
            # (n, 2) in-plane (dh, dw) applied to each STORED slice, after the whole-volume dz.
            # Under reg_slicewise these differ slice to slice; reg_offset holds the global one.
            f.create_dataset("reg_slice_shift",
                             data=np.asarray(r["slice_shift"])[idx].astype(np.int32))
        f.attrs["reg_rule"] = (
            "none" if not r.get("applied") else
            ("rigid transform read off the NIfTI affines (inv(A_other) @ A_ref), applied by "
             "trilinear grid_sample onto the reference's grid: rotation AND translation, "
             "border padding for intensities, nearest-threshold for the masks"
             if r.get("method") == "affine" else
             f"integer translation found by searching the masked correlation of the normalised "
             f"{cfg.reg_contrast}: dz (+-{cfg.reg_max_dz}), then one (dh, dw) (+-{cfg.reg_search})"
             + (f", then per-slice (dh, dw) within +-{cfg.reg_slice_radius} of it, median-"
                f"filtered over {cfg.reg_smooth} slices (see reg_slice_shift)"
                if cfg.reg_slicewise else "")
             + " -- rotation NOT corrected"
             if r.get("method") == "search" else
             f"integer translation: dz by the masked correlation of the normalised "
             f"{cfg.reg_contrast} (blurred scan over +-{cfg.reg_max_dz}, refined +-2), then one "
             f"(dh, dw) centring the brain mask on the reference's over the shared slices "
             f"-- rotation NOT corrected"
             if r.get("method") == "centroid" else
             "3-D FFT cross-correlation of the low-passed normalised "
             f"{cfg.reg_contrast} (lowpass={cfg.reg_lowpass}), integer voxel translation only "
             "-- rotation NOT corrected"))
        f.attrs["axis_order"] = ("N,H,W,C ; H,W are the canonical-RAS (R, A) axes and "
                                 "slice_index maps N back to the canonical-RAS z")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="patient-ID level, e.g. .../imaging/patientId")
    ap.add_argument("--out", required=True)
    ap.add_argument("--crop", type=int, default=0, help="center crop in-plane; 0 = none")
    ap.add_argument("--clip", type=float, nargs=2, default=(0.5, 99.5),
                    help="foreground percentile clip before the stats; '--clip 0 0' disables")
    ap.add_argument("--bg-frac", type=float, default=0.05, dest="bg_frac")
    ap.add_argument("--support", choices=("store", "apply", "none"), default="store",
                    help="what to do with the common acquisition support. store (default): "
                         "write it to `support`, leave the pixels alone. apply: also zero "
                         "every channel outside it -- the pre-2026-09-22 behaviour, which "
                         "bakes a ragged FOV boundary into the data. none: do not compute it")
    ap.add_argument("--no-intersect-support", action="store_const", const="none",
                    dest="support", help="deprecated alias for --support none")
    ap.add_argument("--support-frac", type=float, default=0.02, dest="support_frac",
                    help="acquisition-support threshold as a fraction of each contrast's "
                         "p99.5; low on purpose, so dark-but-acquired CSF stays inside")
    ap.add_argument("--support-close", type=int, default=2, dest="support_close",
                    help="radius of the opening/closing on the support mask; 0 disables")
    ap.add_argument("--support-warn", type=float, default=0.02, dest="support_warn",
                    help="warn when the intersection drops more than this fraction of a "
                         "contrast's own support")
    ap.add_argument("--min-brain-frac", type=float, default=0.02, dest="min_brain_frac")
    ap.add_argument("--no-register", action="store_false", dest="register",
                    help="do NOT align a patient's studies to one another (they are then "
                         "written on their own index axes, as before 2026-09-23)")
    ap.add_argument("--reg-method", default="centroid",
                    choices=("centroid", "search", "affine", "xcorr"), dest="reg_method",
                    help="centroid (default): dz by masked correlation, then one in-plane "
                         "(dh, dw) centring the brain mask on the reference's -- see "
                         "_register_centroid. "
                         "search: dz, then in-plane (dh, dw) -- one per volume, then "
                         "per slice -- found by searching the masked correlation of the images; "
                         "see _register_search for the comparison that chose it. "
                         "affine: read the rigid transform off the NIfTI headers and "
                         "resample onto the patient's reference grid, correcting ROTATION as "
                         "well as translation. Exact for a rigid pair on a common voxel grid, "
                         "which --geometry says this cohort is. xcorr: the old 3-D FFT "
                         "cross-correlation, integer translation only -- it cannot remove the "
                         "~6.5 deg of median inter-session obliqueness the 2026-09-24 survey "
                         "measured, which leaves ~13 voxels of error at the edge of the head")
    ap.add_argument("--reg-min-cover", type=float, default=0.99, dest="reg_min_cover",
                    help="--reg-method affine only: keep an output slice when at least this "
                         "fraction of the REFERENCE'S BRAIN on it was drawn from real acquired "
                         "source data. Not the whole frame -- rotating a square empties its "
                         "corners, and those corners are air")
    ap.add_argument("--reg-contrast", default="T1", choices=CONTRASTS, dest="reg_contrast",
                    help="which contrast the cross-correlation runs on; the same contrast is "
                         "compared across studies, so a structural one is the right choice")
    ap.add_argument("--reg-lowpass", type=float, default=0.25, dest="reg_lowpass",
                    help="fraction of k-space kept IN-PLANE before correlating; small = "
                         "coarse structure drives the peak. 1.0 disables the low-pass")
    ap.add_argument("--reg-source", default="norm", choices=("norm", "raw"),
                    dest="reg_source",
                    help="measure the offset on img_median_mad ('norm', brain-MASKED, so its "
                         "strongest edge is a mask boundary that differs between studies) or "
                         "on img_raw ('raw', the whole head, large DC)")
    ap.add_argument("--no-reg-phase", action="store_false", dest="reg_phase",
                    help="plain cross-correlation instead of phase correlation -- exactly what "
                         "generate_longibrain.jl does. Phase is the default because it removes "
                         "the zero-lag bias that plain correlation has on unmasked data")
    ap.add_argument("--reg-device", default="cpu", dest="reg_device",
                    help="where the registration FFTs run. 'cpu' by default because this "
                         "builder is IO- and NIfTI-bound and normally runs on a cpu_ partition; "
                         "pass cuda on a GPU node. Only the low-passed probe moves")
    ap.add_argument("--reg-max-shift", type=int, default=0, dest="reg_max_shift",
                    help="reject any offset larger than this in any axis. 0 (the default) "
                         "accepts whatever the correlation finds: the 40 this used to default "
                         "to was arbitrary, and the 2026-09-23 survey caught it rejecting dz of "
                         "43-49 that looked genuine -- large through-plane, small and consistent "
                         "in-plane. --min-overlap is the principled guard")
    ap.add_argument("--reg-max-dz", type=int, default=60, dest="reg_max_dz",
                    help="centroid / search: through-plane search range, slices. The 2026-09-25 comparison "
                         "saw |dz| up to ~30; 60 leaves headroom")
    ap.add_argument("--reg-search", type=int, default=16, dest="reg_search",
                    help="search: in-plane range for the whole-volume (dh, dw), px")
    ap.add_argument("--no-reg-slicewise", action="store_false", dest="reg_slicewise",
                    help="search: one in-plane shift per volume only, no per-slice refinement")
    ap.add_argument("--reg-slice-radius", type=int, default=6, dest="reg_slice_radius",
                    help="search: per-slice (dh, dw) range around the whole-volume shift, px")
    ap.add_argument("--reg-slice-min-px", type=int, default=400, dest="reg_slice_min_px",
                    help="search: joint-brain pixels (after the in-plane stride) a slice needs "
                         "for a shift of its own; thinner slices take their neighbours'")
    ap.add_argument("--reg-smooth", type=int, default=7, dest="reg_smooth",
                    help="search: median window along z for the per-slice shifts")
    ap.add_argument("--reg-stride", type=int, default=2, dest="reg_stride",
                    help="search: in-plane subsampling while scoring (speed only; the shifts "
                         "themselves stay in full-resolution pixels)")
    ap.add_argument("--reg-min-ncc", type=float, default=0.2, dest="reg_min_ncc",
                    help="search: flag studies whose masked correlation with the reference is "
                         "below this after registration (recorded as reg_ncc; nothing dropped)")
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--require-affine", action="store_true", dest="require_affine",
                    help="skip sessions whose contrasts are not on one affine")
    ap.add_argument("--compress", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", dest="dry_run",
                    help="report the native in-plane size distribution from HEADERS ONLY and "
                         "exit -- use it to choose --crop before building anything")
    cfg = ap.parse_args()
    if cfg.clip and float(cfg.clip[1]) <= 0:
        cfg.clip = None

    sessions = find_sessions(cfg.root)
    if cfg.limit:
        sessions = dict(list(sessions.items())[:cfg.limit])
    print(f"{len(sessions)} complete {'/'.join(CONTRASTS)} sessions under {cfg.root}")
    print(f"support    : {cfg.support}"
          + ("  (pixels outside the common FOV are ZEROED)" if cfg.support == "apply" else
             "  (pixels are left alone)" if cfg.support == "store" else "  (not computed)"))

    if cfg.dry_run:
        # nib.load is lazy and as_closest_canonical only rewrites the affine, so .shape costs
        # no voxel reads -- this is seconds over the whole cohort.
        hist, per_contrast, bad, dtypes = {}, {c: {} for c in CONTRASTS}, [], {}
        for key, files in sessions.items():
            try:
                imgs = {c: nib.as_closest_canonical(nib.load(files[c])) for c in CONTRASTS}
                shp = {c: im.shape[:2] for c, im in imgs.items()}
            except Exception as err:
                bad.append((key, str(err)))
                continue
            # The NIfTI datatype is the ONLY authoritative answer to "is this complex".
            # NIfTI can hold complex64/128/256; if the files were magnitude-only DICOM
            # exports they will be int16/uint16/float32 and the phase is simply not there.
            for c, im in imgs.items():
                dt = str(im.header.get_data_dtype())
                dtypes[(c, dt)] = dtypes.get((c, dt), 0) + 1
            for c, hw in shp.items():
                per_contrast[c][hw] = per_contrast[c].get(hw, 0) + 1
            if len(set(shp.values())) != 1:
                bad.append((key, f"contrasts disagree in-plane: {shp}"))
            hw = shp[CONTRASTS[0]]
            hist[hw] = hist.get(hw, 0) + 1
        print("\nstored NIfTI datatype, by contrast:")
        for c in CONTRASTS:
            row = {dt: n for (cc, dt), n in dtypes.items() if cc == c}
            print(f"  {c:<6} " + "  ".join(f"{dt} x{n}" for dt, n in sorted(row.items())))
        cplx = sorted({dt for (_, dt) in dtypes if "complex" in dt.lower()})
        if cplx:
            print(f"  ** COMPLEX data present ({', '.join(cplx)}) -- phase is available and "
                  f"load_ras will refuse rather than silently drop it")
        else:
            print("  -> all real: these are MAGNITUDE images and the phase is not in the")
            print("     files. No amount of preprocessing recovers it; it would have to come")
            print("     from the source DICOM (or the scanner) if it was ever saved at all.")

        print("\nnative in-plane sizes (H, W), by session:")
        for hw, n in sorted(hist.items(), key=lambda kv: -kv[1]):
            print(f"  {str(hw):<14} {n:>5}")
        dims = [d for hw in hist for d in hw]
        if dims:
            print(f"\nsmallest single in-plane dimension seen: {min(dims)}")
            print(f"largest  single in-plane dimension seen: {max(dims)}")
            print("\n--crop N pads any session smaller than N and crops any larger, so:")
            print(f"  --crop {max(dims)}  keeps every voxel (pads the small ones)")
            print(f"  --crop {min(dims)}  crops everything, never pads (loses FOV on the big ones)")
            print("The loader's crop_size must be <= --crop, and a RandomCrop much smaller")
            print("than the padded region will sample mostly zeros on the small sessions.")
        if bad:
            print(f"\n{len(bad)} of {len(sessions)} sessions have contrasts that DISAGREE "
                  f"in-plane ({100*len(bad)/max(len(sessions),1):.1f}%) -- these raise "
                  f"'shapes differ' and are skipped by the build:")
            for k, why in bad[:10]:
                print(f"  !! {k[0]}/{k[1]}: {why}")
            if len(bad) > 10:
                print(f"  ... and {len(bad) - 10} more")
        return

    rows, shapes, skipped, support_lost = [], {}, [], []
    natives, padded, reg_rows = {}, [], []

    # PER PATIENT, not per session: registration is a cross-study operation, and the common
    # slice range it produces is a property of the patient. Sessions of one patient are built
    # together, aligned, truncated to their shared range, and only then written.
    by_patient = {}
    for key, files in sessions.items():
        by_patient.setdefault(key[0], []).append((key, files))

    def drop_stale(key):
        """Delete a session's EXISTING h5 when this run skips it. Without this a rebuild with
        --overwrite leaves the previous build's file in place for every session it could not
        write, and a patient ends up mixing old (differently registered) files with new ones."""
        case = f"{key[0]}_{key[1]}"
        path = os.path.join(cfg.out, case, f"{case}_img.h5")
        if cfg.overwrite and os.path.exists(path):
            os.remove(path)
            print(f"     removed stale {path}")

    done = 0
    for pid, items in by_patient.items():
        paths = [os.path.join(cfg.out, f"{k[0]}_{k[1]}", f"{k[0]}_{k[1]}_img.h5")
                 for k, _ in items]
        if all(os.path.exists(q) for q in paths) and not cfg.overwrite:
            done += len(items)
            continue

        vols, keys = [], []
        for (key, files), _ in zip(items, paths):
            try:
                vols.append(build_volume(files, cfg))
                keys.append(key)
            except Exception as err:
                skipped.append((key, str(err)))
                print(f"  !! {key[0]}/{key[1]}: {err}")
                drop_stale(key)
        done += len(items)
        if not vols:
            continue

        # ---- register every study of this patient onto one of them --------------------------
        # Reference = the DEEPEST study (ties broken by session id, so a rebuild is
        # reproducible): the most z coverage to align the others into, which keeps the common
        # range as large as it can be.
        offsets = [(0, 0, 0)] * len(vols)
        valid = [np.ones(v.raw.shape[0], dtype=bool) for v in vols]
        ref = 0
        reg_ok = False       # recorded as reg_applied: True only if the pixels were really moved
        if cfg.register and len(vols) > 1:
            ref = max(range(len(vols)), key=lambda i: (vols[i].raw.shape[0], -i))
            try:
                offsets, valid = register_patient(vols, cfg, ref=ref)
                reg_ok = True
            except ValueError as err:
                print(f"  !! {pid}: registration skipped -- {err}")
                offsets = [(0, 0, 0)] * len(vols)
                valid = [np.ones(v.raw.shape[0], dtype=bool) for v in vols]

        # ---- ONE slice range for the whole patient ------------------------------------------
        # Intersect each study's own keep rule with every study's real-data extent, so the
        # written slice set is IDENTICAL across the patient and `slice_index` means the same
        # anatomical level in all of them. That is the invariant guide_slice="index" needs.
        depth = max(v.raw.shape[0] for v in vols)
        common = np.ones(depth, dtype=bool)
        for v, ok in zip(vols, valid):
            k = keep_mask(v, cfg)
            # register_patient pads every study to `depth`; without it (--no-register, or a
            # single-study patient) a shallow study still has its own shorter masks, and the
            # slices it simply does not have must count as unavailable rather than broadcast.
            if k.shape[0] < depth:
                k = np.pad(k, (0, depth - k.shape[0]))
            if ok.shape[0] < depth:
                ok = np.pad(ok, (0, depth - ok.shape[0]))
            common &= k & ok
        idx = np.where(common)[0]
        if idx.size == 0:
            for key in keys:
                skipped.append((key, "no slice survived the patient's common range"))
                print(f"  !! {key[0]}/{key[1]}: no slice survived the patient's common range")
                drop_stale(key)
            continue

        for i, (v, key) in enumerate(zip(vols, keys)):
            case = f"{key[0]}_{key[1]}"
            path = os.path.join(cfg.out, case, f"{case}_img.h5")
            raw, norm, mask, support = select_slices(v, idx)
            write_h5(path, raw, norm, mask, support, idx, v.stats, v.orig_depth, v.affine,
                     key, sessions[key], cfg, v.lost, v.native_hw,
                     reg=dict(v.reg or {}, offset=offsets[i],
                              reference=f"{keys[ref][0]}_{keys[ref][1]}",
                              applied=reg_ok))
            ncc = float((v.reg or {}).get("ncc", float("nan")))
            if reg_ok and i != ref and ncc == ncc and ncc < cfg.reg_min_ncc:
                print(f"  ?? {key[0]}/{key[1]}: correlation with the reference after registration "
                      f"is {ncc:.3f} < --reg-min-ncc {cfg.reg_min_ncc} -- check this study")
            natives[v.native_hw] = natives.get(v.native_hw, 0) + 1
            if cfg.crop and (v.native_hw[0] < cfg.crop or v.native_hw[1] < cfg.crop):
                padded.append((key, v.native_hw))
            if v.lost:
                worst = max(v.lost, key=v.lost.get)
                if v.lost[worst] > cfg.support_warn:
                    print(f"  ?? {key[0]}/{key[1]}: {v.lost[worst]:.1%} of {worst}'s support is "
                          f"outside the intersection -- FOV mismatch between contrasts"
                          + (" (ZEROED)" if cfg.support == "apply" else " (kept; see `support`)"))
                support_lost.append((key, dict(v.lost)))
            shapes[raw.shape[1:3]] = shapes.get(raw.shape[1:3], 0) + 1
            rows.append({"patient": key[0], "session": key[1], "path": path,
                         "n_slices": len(idx), "orig_depth": v.orig_depth,
                         "H": raw.shape[1], "W": raw.shape[2],
                         "dz": offsets[i][0], "dh": offsets[i][1], "dw": offsets[i][2]})
            if i != ref:
                reg_rows.append((key, offsets[i], v.reg))
        del vols
        if done % 25 < len(items):
            print(f"  {done}/{len(sessions)}  {len(rows)} written, {len(skipped)} skipped")

    os.makedirs(cfg.out, exist_ok=True)
    with open(os.path.join(cfg.out, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["patient", "session", "path", "n_slices",
                                          "orig_depth", "H", "W", "dz", "dh", "dw"])
        w.writeheader()
        w.writerows(rows)

    print(f"\n{len(rows)} written, {len(skipped)} skipped -> {cfg.out}")
    print("native in-plane sizes:", dict(natives))
    print("stored in-plane sizes:", dict(shapes))
    if support_lost:
        import statistics
        print("\ncommon-support intersection, fraction of each contrast's own support that "
              "falls outside it")
        print(f"  (support={cfg.support}: "
              + ("those voxels were ZEROED in every channel):"
                 if cfg.support == "apply" else "the pixels were KEPT; this is diagnostic):"))
        for c in CONTRASTS:
            vals = [d[c] for _, d in support_lost]
            print(f"  {c:<6} median {statistics.median(vals):6.2%}   max {max(vals):6.2%}")
        worst = max(support_lost, key=lambda kv: max(kv[1].values()))
        print(f"  worst session: {worst[0][0]}/{worst[0][1]}  "
              + "  ".join(f"{k} {v:.1%}" for k, v in worst[1].items()))
    if reg_rows:
        import math
        import statistics
        print(f"\ncross-study registration ({cfg.reg_method}): {len(reg_rows)} non-reference "
              f"study(ies) put onto their patient's reference grid")
        for ax, name in enumerate(("dz (slice)", "dh (R)", "dw (A)")):
            v = [abs(t[ax]) for _, t, _ in reg_rows]
            nz = sum(x > 0 for x in v)
            print(f"  {name:<11} |offset| median {statistics.median(v):5.1f}  max {max(v):3d}  "
                  f"nonzero on {nz}/{len(v)}")
        worst = max(reg_rows, key=lambda kt: max(abs(c) for c in kt[1]))
        print(f"  largest: {worst[0][0]}/{worst[0][1]} -> {worst[1]}")
        if cfg.reg_method == "affine":
            deg = lambda x: math.degrees(math.asin(min(abs(x), 1.0)))
            ob = [float((r or {}).get("oblique", 0.0)) for _, _, r in reg_rows]
            cv = [float((r or {}).get("cover", float("nan"))) for _, _, r in reg_rows]
            cv = [c for c in cv if c == c]
            print(f"  obliqueness CORRECTED: median {statistics.median(ob):.4f} "
                  f"({deg(statistics.median(ob)):.1f} deg)  max {max(ob):.4f} "
                  f"({deg(max(ob)):.1f} deg)")
            print("  The offset above is only the closest single translation, for readability; "
                  "the\n  transform applied is the full 4x4 in each file's `reg_matrix` attr.")
            if cv:
                print(f"  coverage (output voxels drawn from real source data): median "
                      f"{statistics.median(cv):.3f}  min {min(cv):.3f}")
                print(f"  slices are kept only where >= {cfg.reg_min_cover:.0%} of the "
                      f"REFERENCE'S BRAIN is covered")
        elif cfg.reg_method in ("search", "centroid"):
            nc = [float((r or {}).get("ncc", float("nan"))) for _, _, r in reg_rows]
            nc = [c for c in nc if c == c]
            if nc:
                low = sum(c < cfg.reg_min_ncc for c in nc)
                print(f"  masked correlation with the reference after registration: median "
                      f"{statistics.median(nc):.3f}  min {min(nc):.3f}  "
                      f"{low}/{len(nc)} below --reg-min-ncc {cfg.reg_min_ncc} (reg_ncc attr)")
            print("  Offsets are (dz, dh, dw) of the whole-volume shift"
                  + ("; per-slice in-plane shifts are in each file's `reg_slice_shift`."
                     if cfg.reg_method == "search" and cfg.reg_slicewise else "."))
        else:
            print("  A large offset is either a real displacement or a correlation failure; "
                  "--reg-max-shift rejects the obvious failures. Rotation is NOT corrected by "
                  "this method.")
    if padded:
        print(f"\n{len(padded)} session(s) were smaller than --crop {cfg.crop} and were "
              f"ZERO-PADDED up to it:")
        for k, n in padded[:8]:
            print(f"    {k[0]}/{k[1]}  native {n[0]}x{n[1]}")
        if len(padded) > 8:
            print(f"    ... and {len(padded) - 8} more")
    if len(shapes) > 1:
        # The failure this guards against is not cosmetic: I2SBDataset applies a fixed
        # RandomCrop, which throws the moment it meets a stored image smaller than crop_size,
        # and only after the loader has already yielded a few good batches.
        print("!! STORED SHAPES ARE RAGGED -- a loader with a fixed crop_size will fail on "
              "the odd one out. Set --crop, or raise it above the largest native size.")


if __name__ == "__main__":
    main()
