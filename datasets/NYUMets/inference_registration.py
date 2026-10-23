"""
Inference-time registration of a patient's two studies, WITHOUT the current study's CT1.

The fit-registered h5 build aligned every study of a patient onto one reference grid, but its
alignment is not something an inference pipeline could have produced: its masks (brain_mask
averages all four contrasts, the common FOV intersects all four) and its centroid start both saw
the CT1 being predicted. At inference there is no current CT1. This module re-aligns the two
studies of one sample using only the contrasts that ARE available -- by default FLAIR, T1, T2 of
both studies -- and applies the transform to every contrast, CT1 included: the contrasts within
one study are co-registered, so the anatomy the T1/T2/FLAIR fit aligns is the anatomy the CT1
shows.

    direction "prior_to_current" (default)  the prior study is resampled onto the current one;
                                             the current study, and so the target, is untouched
    direction "current_to_prior"            the current study is resampled onto the prior one;
                                             target, inputs and mask are all the warped current

THE FIT is the build's own `fit_affine` (preprocessing/nyumets_h5.py) -- rotation, per-axis scale
and shift by masked correlation, coarse to fine -- run on several contrasts at once (the score is
the mean of the per-contrast correlations) and inside a brain mask made by the build's
`brain_mask` rule from those contrasts alone. It starts from the identity: the stored studies
already share a grid, so this is a refinement whose optimum is set by the CT1-free metric. A fit
that does not beat the identity, loses the overlap, or leaves sane bounds is rejected and the
identity kept (`fit_ok` False), exactly as in the build -- and so is one whose final correlation
stays below `min_ncc`, which the build does not check: two studies with no shared visible anatomy
always "improve" a little, and that improvement is noise.

NOT REPEATED. One transform per (fixed study, moving study, settings): kept in a process-wide
dict -- so the two loaders a mixed-scaling evaluation builds share it; the fit does not depend on
intensity scaling -- and, with `cache_dir`, on disk, so a later job reuses it. Each dataset keeps
the last few WARPED volumes, so the slices of one volume cost one resample.

DEVICE. `device: "cuda"` makes a fit take about a second instead of tens of seconds on a CPU;
use it only with num_workers=0 (a forked worker cannot use the parent's CUDA context).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

DIRECTIONS = ("prior_to_current", "current_to_prior")
DEFAULTS = dict(direction="prior_to_current", contrasts=[0, 1, 3],    # FLAIR, T1, T2 (stored order)
                dof=9, levels=[4, 2], iters=[120, 60], bg_frac=0.05,
                max_rot=25.0, max_scale=0.25, min_ncc=0.3, device="cpu", cache_dir=None)
_MATRICES = {}                    # process-wide: key -> (M (4, 4) float64, info dict)


def parse_register(spec, x0_idx):
    """A dataset's `register` option -> a full settings dict, or None (off).

    Accepts None / False / "none" (off), a direction string, or a dict overriding DEFAULTS.
    The target contrast may not be among the registration contrasts: that is the point."""
    if spec in (None, False, "none", ""):
        return None
    if isinstance(spec, str):
        spec = {"direction": spec}
    unknown = set(spec) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"register: unknown key(s) {sorted(unknown)}; known: {sorted(DEFAULTS)}")
    out = dict(DEFAULTS, **spec)
    if out["direction"] not in DIRECTIONS:
        raise ValueError(f"register.direction must be one of {DIRECTIONS}, got {out['direction']!r}")
    out["contrasts"] = [int(c) for c in out["contrasts"]]
    if int(x0_idx) in out["contrasts"]:
        raise ValueError(f"register.contrasts {out['contrasts']} include the target contrast "
                         f"{x0_idx}: inference-time registration may not use the scan it predicts")
    out["levels"] = [int(f) for f in out["levels"]]
    out["iters"] = [int(n) for n in out["iters"]]
    if len(out["levels"]) != len(out["iters"]):
        raise ValueError("register.iters needs one count per register.levels entry")
    return out


def describe(spec):
    """One line for logs and dump attrs: what was registered, how."""
    if spec is None:
        return "stored"
    names = {0: "FLAIR", 1: "T1", 2: "CT1", 3: "T2"}
    return (f"{spec['direction']} fit[{','.join(names.get(c, str(c)) for c in spec['contrasts'])}] "
            f"dof{spec['dof']}")


def _key(fixed_path, moving_path, image_key, spec):
    k = dict(fixed=os.path.abspath(fixed_path), moving=os.path.abspath(moving_path),
             image_key=image_key, **{n: spec[n] for n in ("contrasts", "dof", "levels", "iters",
                                                        "bg_frac", "max_rot", "max_scale",
                                                        "min_ncc")})
    return hashlib.sha1(json.dumps(k, sort_keys=True).encode()).hexdigest()


def _brain(vol, present, bg_frac):
    """(C, Z, H, W) -> (Z, H, W) bool: the build's brain_mask rule on these contrasts only,
    restricted to the slices the study holds."""
    from preprocessing.nyumets_h5 import brain_mask
    return brain_mask(vol.permute(1, 0, 2, 3), bg_frac) & present.view(-1, 1, 1)


def fit_pair(fixed, fixed_present, moving, moving_present, spec, key=None):
    """Transform taking a FIXED-grid index to the MOVING index it reads from. -> (M, info).

    fixed / moving   (Z, H, W, C) arrays on the original-slice axis (absent slices zero), any
                     scaling (the metric is a correlation per contrast); only spec["contrasts"]
                     are read
    *_present        (Z,) bool: which original slices the study holds
    """
    if key is not None and key in _MATRICES:
        return _MATRICES[key]
    path = None
    if key is not None and spec.get("cache_dir"):
        path = os.path.join(spec["cache_dir"], key + ".npz")
        if os.path.exists(path):
            z = np.load(path, allow_pickle=False)
            out = (z["M"].astype(np.float64), json.loads(str(z["info"])))
            _MATRICES[key] = out
            return out

    from preprocessing.nyumets_h5 import fit_affine
    dev = torch.device(spec["device"])
    cs = spec["contrasts"]
    a = torch.from_numpy(np.ascontiguousarray(fixed[..., cs])).permute(3, 0, 1, 2).float()
    b = torch.from_numpy(np.ascontiguousarray(moving[..., cs])).permute(3, 0, 1, 2).float()
    ma = _brain(a, torch.as_tensor(fixed_present), spec["bg_frac"])
    mb = _brain(b, torch.as_tensor(moving_present), spec["bg_frac"])
    eye = np.eye(4)
    info = dict(fit_ok=False, ncc_init=float("nan"), ncc=float("nan"), why="")
    try:
        # gradients ON whatever the caller is doing: an evaluation loop runs under no_grad, and
        # the fit is gradient descent on the transform parameters
        with torch.enable_grad():
            M, fi = fit_affine(a, ma, b, mb, (0, 0, 0), dof=spec["dof"], levels=spec["levels"],
                               iters=spec["iters"], dev=dev)
        info.update(ncc_init=fi["ncc_init"], ncc=fi["ncc"], overlap=fi["overlap"])
        sv = np.linalg.svd(M[:3, :3], compute_uv=False)
        angle = math.degrees(math.acos(min(1.0, max(-1.0, (np.trace(M[:3, :3]) / max(sv.mean(), 1e-6)
                                                           - 1.0) / 2.0))))
        if not fi["ncc"] > fi["ncc_init"]:
            info["why"] = "did not beat the identity"
        elif fi["ncc"] < spec["min_ncc"]:
            # "better" is not enough: on two studies that share no visible anatomy any transform
            # nudges a near-zero correlation up, and warping by it would be fitting noise
            info["why"] = f"correlation {fi['ncc']:.2f} below min_ncc {spec['min_ncc']}"
        elif fi["overlap"] < 0.5:
            info["why"] = f"brains overlap only {fi['overlap']:.0%}"
        elif sv.max() > 1 + spec["max_scale"] or sv.min() < 1 - spec["max_scale"]:
            info["why"] = f"scale {sv.min():.2f}..{sv.max():.2f} out of bounds"
        elif angle > spec["max_rot"]:
            info["why"] = f"rotation {angle:.0f} deg above max_rot"
        else:
            info.update(fit_ok=True, rot_deg=angle, scale=[float(s) for s in sorted(sv)])
            eye = M
    except ValueError as err:                     # empty mask, volume too small
        info["why"] = str(err)
    out = (np.asarray(eye, dtype=np.float64), info)
    if key is not None:
        _MATRICES[key] = out
    if path is not None:
        os.makedirs(spec["cache_dir"], exist_ok=True)
        tmp = path + f".{os.getpid()}.tmp.npz"
        np.savez(tmp, M=out[0], info=json.dumps(info))
        os.replace(tmp, path)                     # atomic: concurrent jobs never read half a file
    return out


def warp(moving, M, out_dhw, device="cpu"):
    """(Zm, H, W, C) moving array -> (Z, H, W, C) on the fixed grid; zero where it reads outside."""
    from preprocessing.nyumets_h5 import _level_grid
    dev = torch.device(device)
    src = torch.from_numpy(np.ascontiguousarray(moving)).to(dev, torch.float32)
    src = src.permute(3, 0, 1, 2)[None]                                    # (1, C, Zm, H, W)
    Mt = torch.as_tensor(M, dtype=torch.float32, device=dev)
    grid = _level_grid(Mt, tuple(out_dhw), tuple(src.shape[2:]), 1)
    out = F.grid_sample(src, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return out[0].permute(1, 2, 3, 0).cpu().numpy()


class WarpCache:
    """The last `size` warped volumes of one dataset, keyed by (fixed file, moving file)."""

    def __init__(self, size=2):
        self.size, self.d = size, OrderedDict()

    def get(self, key, make):
        if key in self.d:
            self.d.move_to_end(key)
            return self.d[key]
        v = make()
        self.d[key] = v
        while len(self.d) > self.size:
            self.d.popitem(last=False)
        return v
