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

    raw/norm are (D, C, H, W); fg/fov are (D, H, W). Registration shifts all four together.
    """

    __slots__ = ("raw", "norm", "fg", "fov", "stats", "orig_depth", "affine", "lost",
                 "native_hw")

    def __init__(self, **kw):
        for k in self.__slots__:
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


def register_patient(vols, cfg, ref=0, real=None):
    """-> (offsets, valid) for one patient's studies, all aligned onto `vols[ref]`.

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
    hw = {tuple(v.raw.shape[2:]) for v in vols}
    if len(hw) != 1:
        raise ValueError(f"studies are on different in-plane grids {sorted(hw)}; set --crop so "
                         f"they share one before registering")
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
            v.raw = translate(v.raw, (t[0], 0, t[1], t[2]))
            v.norm = translate(v.norm, (t[0], 0, t[1], t[2]))
            v.fg = translate(v.fg, t)
            v.fov = translate(v.fov, t)
            real[i] = translate(real[i], (t[0],))
        offsets.append(t)
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
    norm, stats = normalize_masked(clipped, fg, mode=MODE)

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
        # native in-plane size BEFORE the crop/pad, so a padded session is identifiable
        f.attrs["native_size"] = np.asarray(native_hw, dtype=np.int32)
        f.attrs["orig_depth"] = int(orig_depth)
        f.attrs["min_brain_frac"] = float(cfg.min_brain_frac)
        f.attrs["mask_rule"] = f"mean_c(v/p99.5_c) > {cfg.bg_frac}"
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
        f.attrs["affine"] = np.asarray(affine, dtype=np.float32)
        f.attrs["source_files"] = ",".join(os.path.basename(files[c]) for c in CONTRASTS)
        # Cross-study registration. `reg_offset` is the (dz, dh, dw) THIS study was shifted by
        # to land on `reg_reference`; (0,0,0) on the reference itself, and on every study when
        # `reg_applied` is False. After registration all studies of a patient share one index
        # axis and one slice set, so `slice_index` names the same anatomical level in each.
        r = reg or {}
        f.attrs["reg_applied"] = bool(r.get("applied", False))
        f.attrs["reg_offset"] = np.asarray(r.get("offset", (0, 0, 0)), dtype=np.int32)
        f.attrs["reg_reference"] = str(r.get("reference", ""))
        f.attrs["reg_rule"] = ("3-D FFT cross-correlation of the low-passed normalised "
                               f"{cfg.reg_contrast} (lowpass={cfg.reg_lowpass}), integer "
                               "voxel translation only" if r.get("applied") else "none")
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
        if cfg.register and len(vols) > 1:
            ref = max(range(len(vols)), key=lambda i: (vols[i].raw.shape[0], -i))
            try:
                offsets, valid = register_patient(vols, cfg, ref=ref)
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
            continue

        for i, (v, key) in enumerate(zip(vols, keys)):
            case = f"{key[0]}_{key[1]}"
            path = os.path.join(cfg.out, case, f"{case}_img.h5")
            raw, norm, mask, support = select_slices(v, idx)
            write_h5(path, raw, norm, mask, support, idx, v.stats, v.orig_depth, v.affine,
                     key, sessions[key], cfg, v.lost, v.native_hw,
                     reg=dict(offset=offsets[i], reference=f"{keys[ref][0]}_{keys[ref][1]}",
                              applied=bool(cfg.register and len(vols) > 1)))
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
                reg_rows.append((key, offsets[i]))
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
        import statistics
        print(f"\ncross-study registration: {len(reg_rows)} non-reference study(ies) shifted "
              f"onto their patient's reference")
        for ax, name in enumerate(("dz (slice)", "dh (R)", "dw (A)")):
            v = [abs(t[ax]) for _, t in reg_rows]
            nz = sum(x > 0 for x in v)
            print(f"  {name:<11} |offset| median {statistics.median(v):5.1f}  max {max(v):3d}  "
                  f"nonzero on {nz}/{len(v)}")
        worst = max(reg_rows, key=lambda kt: max(abs(c) for c in kt[1]))
        print(f"  largest: {worst[0][0]}/{worst[0][1]} -> {worst[1]}")
        print("  A large offset is either a real displacement or a correlation failure; "
              "--reg-max-shift rejects the obvious failures.")
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
