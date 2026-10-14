# -*- coding: utf-8 -*-
"""
Cross-study registration of the released NYUMets NIfTIs with SimpleITK -- before any of the
builder's processing (no crop, no median/MAD normalisation, no heuristic masks).

The standard intensity-based pipeline, as SimpleITK's own examples write it:

    initialise   centres of mass aligned (CenteredTransformInitializer, MOMENTS)
    rigid        6 dof; then optionally a similarity (7) or affine (12) stage started from it
    metric       Mattes mutual information (or plain correlation), sampled inside the
                 reference's head mask
    optimiser    regular-step gradient descent, steps scaled from physical shift. NOT plain
                 gradient descent: with a fixed rate its convergence check stopped after ~30
                 iterations on the synthetic test and could end worse than it started
    pyramid      shrink 4 / 2 / 1, smoothing 2 / 1 / 0 mm

Everything here is in PHYSICAL space, so the result does not depend on how each file orders its
voxels. One transform, fitted on T1, moves a whole session: the release co-registered each
session's contrasts itself.

`to_builder_space` maps an array on the reference NIfTI's grid into the builder's (D, H, W) --
canonical RAS (`load_ras`) and `--crop` (`center_crop_or_pad`) -- so a SimpleITK result lines up
index for index with a built h5. Used by notebooks/sitk_registration_test.ipynb and
scripts/compare_sitk_fit.py.

Needs `pip install SimpleITK`.
"""

import glob
import math
import os
import time
from collections import defaultdict

import numpy as np
import SimpleITK as sitk
import torch
import nibabel as nib

from preprocessing.nyumets_h5 import RULES, SKIP_RE, center_crop_or_pad

# Named settings for scripts/compare_sitk_fit.py. Each is a dict of `register` keywords.
VARIANTS = {
    "rigid":       dict(stages=("rigid",)),
    "similarity":  dict(stages=("rigid", "similarity")),
    "affine":      dict(stages=("rigid", "affine")),
    "affine_corr": dict(stages=("rigid", "affine"), metric="corr"),
    "affine_s25":  dict(stages=("rigid", "affine"), sampling=0.25),
}


def sessions_of(nifti_root, pid, need=("T1", "T1ce")):
    """{session: {contrast: path}} of one patient, with at least `need` -- the builder's naming
    rules (RULES / SKIP_RE), the session being the file's parent directory."""
    out = defaultdict(dict)
    for f in glob.glob(os.path.join(nifti_root, pid, "**", "*.nii*"), recursive=True):
        b = os.path.basename(f)
        if SKIP_RE.search(b):
            continue
        lab = next((l for l, rx in RULES if rx.search(b)), None)
        if lab:
            out[os.path.basename(os.path.dirname(f)) or "single"].setdefault(lab, f)
    return {s: d for s, d in sorted(out.items()) if all(c in d for c in need)}


def read(path, lps=True):
    """float32 image; reoriented to LPS index order unless lps=False (the file's own order)."""
    img = sitk.ReadImage(path, sitk.sitkFloat32)
    return sitk.DICOMOrient(img, "LPS") if lps else img


def head_mask(img):
    """Foreground by Otsu, closed and hole-filled. The released volumes are skull-stripped, so
    this is essentially the brain."""
    m = sitk.OtsuThreshold(img, 0, 1, 128)
    m = sitk.BinaryMorphologicalClosing(m, [2, 2, 2])
    return sitk.BinaryFillhole(m)


def _method(fmask, metric, bins, sampling, shrink, smooth, iters, seed):
    R = sitk.ImageRegistrationMethod()
    if metric == "mi":
        R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=bins)
    elif metric == "corr":
        R.SetMetricAsCorrelation()
    else:
        raise ValueError(f"metric must be 'mi' or 'corr', got {metric!r}")
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(sampling, seed)
    if fmask is not None:
        R.SetMetricFixedMask(fmask)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsRegularStepGradientDescent(learningRate=2.0, minStep=1e-4,
                                               numberOfIterations=iters, relaxationFactor=0.5,
                                               gradientMagnitudeTolerance=1e-8)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetShrinkFactorsPerLevel(list(shrink))
    R.SetSmoothingSigmasPerLevel(list(smooth))
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    return R


def register(fixed, moving, fmask=None, stages=("rigid", "affine"), metric="mi", bins=50,
             sampling=0.10, shrink=(4, 2, 1), smooth=(2.0, 1.0, 0.0), iters=300, seed=0):
    """Register `moving` onto `fixed` (both sitk images, any voxel order).

    -> dict(tx=the final transform, rigid=the rigid stage's, metric=its final value, secs, stop).
    Transforms map a REFERENCE point to the moving image's point -- what sitk.Resample wants.
    Every later stage starts from the previous one's result.
    """
    t0 = time.time()
    kw = dict(metric=metric, bins=bins, sampling=sampling, shrink=shrink, smooth=smooth,
              iters=iters, seed=seed)
    rigid = sitk.Euler3DTransform(sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS))
    R = _method(fmask, **kw)
    R.SetInitialTransform(rigid, inPlace=True)
    R.Execute(fixed, moving)
    tx, value, stop = rigid, R.GetMetricValue(), R.GetOptimizerStopConditionDescription()
    for stage in stages[1:]:
        if stage == "similarity":
            nxt = sitk.Similarity3DTransform()
            nxt.SetCenter(rigid.GetCenter())
            nxt.SetMatrix(rigid.GetMatrix())
            nxt.SetTranslation(rigid.GetTranslation())
        elif stage == "affine":
            nxt = sitk.AffineTransform(3)
            nxt.SetCenter(tx.GetCenter())
            nxt.SetMatrix(tx.GetMatrix())
            nxt.SetTranslation(tx.GetTranslation())
        else:
            raise ValueError(f"unknown stage {stage!r}")
        R = _method(fmask, **kw)
        R.SetInitialTransform(nxt, inPlace=True)
        R.Execute(fixed, moving)
        tx, value, stop = nxt, R.GetMetricValue(), R.GetOptimizerStopConditionDescription()
    return dict(tx=tx, rigid=rigid, metric=value, secs=time.time() - t0, stop=stop)


def describe(tx):
    """(rotation in deg, sorted scales, |translation| in mm) of a transform's 3x3 + shift."""
    A = np.asarray(tx.GetMatrix()).reshape(3, 3)
    U, s, Vt = np.linalg.svd(A)
    R = U @ Vt
    ang = math.degrees(math.acos(min(1.0, max(-1.0, (np.trace(R) - 1) / 2))))
    return ang, tuple(float(v) for v in sorted(s)), float(np.linalg.norm(tx.GetTranslation()))


def resample_onto(img, ref, tx, nearest=False):
    """`img` on `ref`'s grid through `tx`, as a numpy array in `ref`'s (k, j, i) order. Zero
    outside the moving image."""
    interp = sitk.sitkNearestNeighbor if nearest else sitk.sitkLinear
    return sitk.GetArrayFromImage(sitk.Resample(img, ref, tx, interp, 0.0, img.GetPixelID()))


def ref_grid_to_canonical(arr_kji, ref_path):
    """An array on the reference NIfTI's OWN grid (SimpleITK's k, j, i order, i.e. the reference
    read with lps=False) -> (H, W, D) float32 in canonical RAS, plus that canonical affine --
    exactly what `load_ras` returns for a file, so the builder can take it in place of one."""
    ref = nib.load(ref_path)
    img = nib.as_closest_canonical(
        nib.Nifti1Image(np.ascontiguousarray(np.asarray(arr_kji).transpose(2, 1, 0)), ref.affine))
    v = np.asarray(img.dataobj)
    while v.ndim > 3:
        v = v[..., 0]
    return np.asarray(v, dtype=np.float32), img.affine


def register_sessions(items, variant="affine", contrasts=("FLAIR", "T1", "T1ce", "T2"), seed=0,
                      log=print):
    """Register every session of ONE patient onto its deepest, on the source NIfTIs.

    items  [(key, {contrast: path}), ...] as the builder groups them
    -> (ref_key, out) with out[key] for every NON-reference session that registered:
         arrays   {contrast: (H, W, D) canonical array, affine}: this session's contrasts
                  resampled onto the REFERENCE's grid through one transform -- what
                  `build_volume(files, cfg, arrays=...)` takes in place of reading the files
         cover    (H, W, D) bool, canonical: where the resampled data came from inside this
                  session's own field of view (zero-filled elsewhere)
         reg      {method, variant, matrix (4x4, reference LPS mm -> this session's LPS mm),
                   oblique, rot, scales, secs, stop}
    A session whose registration raises is left out of `out` and reported through `log`.

    The reference is the session with the most axial slices, ties to the EARLIEST in `items` --
    the builder's own rule, so the same study is the reference whichever method runs.
    """
    t1 = {key: read(files["T1"]) for key, files in items}           # LPS: size[2] = axial count
    order = [key for key, _ in items]
    ref_key = max(order, key=lambda k: (t1[k].GetSize()[2], -order.index(k)))
    ref_path = dict(items)[ref_key]["T1"]
    fixed_native = read(ref_path, lps=False)
    fmask = head_mask(t1[ref_key])
    out = {}
    for key, files in items:
        if key == ref_key:
            continue
        try:
            reg = register(t1[ref_key], t1[key], fmask, seed=seed, **VARIANTS[variant])
            tx = reg["tx"]
            arrays = {}
            for c in contrasts:
                mov = read(files[c], lps=False)
                arrays[c] = ref_grid_to_canonical(resample_onto(mov, fixed_native, tx), ref_path)
            ones = sitk.Image(t1[key].GetSize(), sitk.sitkUInt8) + 1
            ones.CopyInformation(t1[key])
            cover = ref_grid_to_canonical(resample_onto(ones, fixed_native, tx, nearest=True),
                                          ref_path)[0] > 0.5
            A = np.asarray(tx.GetMatrix()).reshape(3, 3)
            c0 = np.asarray(tx.GetCenter())
            M = np.eye(4)
            M[:3, :3] = A
            M[:3, 3] = c0 + np.asarray(tx.GetTranslation()) - A @ c0
            rot, scales, _ = describe(tx)
            out[key] = dict(arrays=arrays, cover=cover,
                            reg=dict(method="sitk", variant=variant, matrix=M,
                                     oblique=float(np.abs(A - np.diag(np.diag(A))).max()),
                                     rot=rot, scales=scales, secs=reg["secs"],
                                     stop=reg["stop"]))
        except Exception as err:                       # noqa: BLE001 -- one study, not the run
            log(f"  !! {key[0]}/{key[1]}: SimpleITK registration failed -- {err}")
    return ref_key, out


def to_builder_space(arr_kji, ref_path, crop, mask=False):
    """An array on the reference NIfTI's OWN grid (SimpleITK's k, j, i order, i.e. the reference
    read with lps=False) -> the builder's (D, H, W): canonical RAS via nibabel, exactly as
    load_ras does, then the in-plane --crop as build_volume does."""
    v, _ = ref_grid_to_canonical(arr_kji, ref_path)
    t = torch.from_numpy(np.ascontiguousarray(v.transpose(2, 0, 1)))[:, None]
    if crop and crop > 0:
        t = center_crop_or_pad(t, crop, h_axis=2, w_axis=3)
    out = t[:, 0].numpy()
    return out > 0.5 if mask else out
