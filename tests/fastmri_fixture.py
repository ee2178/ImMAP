"""
One real fastMRI slice for the test suite, or a clean skip.

Every other test in `tests/` is synthetic, which is right for the algebra but
leaves two things unexercised that only real data has:

  * the actual matrix sizes -- whether `pad_multiple` ever fires at all is a
    property of the dataset, not of the code, and synthetic 60x60 phantoms
    prove nothing about it;
  * real coil maps and real anatomy placement, which is what the image-domain
    embedding's premise rests on ("the anatomy is already inside the FOV, so
    the added border carries no signal").

Resolution order for the data roots, first hit wins:

  1. `$FASTMRI_KSPACE_ROOT` / `$FASTMRI_SMAP_ROOT`
  2. a generated config under `config/<anatomy>/mg/`
  3. `datasets.fastmri.loader.FASTMRI_PATHS`

Relative roots are resolved against the repo root, not the cwd, so the tests
work from anywhere.

`load_slice()` returns None rather than raising when nothing is reachable. A
caller MUST report that it skipped -- a suite that silently passes because it
tested nothing is worse than one that fails.
"""

from __future__ import annotations

import glob
import os

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _abs(p):
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(ROOT, p))


def _roots_from_config(anatomy):
    import json
    for path in sorted(glob.glob(os.path.join(ROOT, "config", anatomy, "mg", "*.json"))):
        try:
            with open(path) as f:
                d = json.load(f)["data"]["train"]
            if d.get("kspace_root") and d.get("smap_root"):
                return d["kspace_root"], d["smap_root"], d.get("scale_fac", 1.0)
        except Exception:                                     # noqa: BLE001
            continue
    return None


def resolve_roots(anatomy="brain"):
    """`(kspace_root, smap_root, scale_fac)` as absolute paths, or None."""
    env_k, env_s = os.environ.get("FASTMRI_KSPACE_ROOT"), os.environ.get("FASTMRI_SMAP_ROOT")
    cands = []
    if env_k and env_s:
        cands.append((env_k, env_s, float(os.environ.get("FASTMRI_SCALE_FAC", 1.0))))
    cfg = _roots_from_config(anatomy)
    if cfg:
        cands.append(cfg)
    try:
        from datasets.fastmri.loader import FASTMRI_PATHS
        p = FASTMRI_PATHS[anatomy]
        cands.append((p["kspace_root"], p["smap_root"], p["scale_fac"]))
        cands.append((p["kspace_root"].replace("../../", "../"),
                      p["smap_root"].replace("../../", "../"), p["scale_fac"]))
    except Exception:                                         # noqa: BLE001
        pass

    for k, s, fac in cands:
        ka, sa = _abs(k), _abs(s)
        if os.path.isdir(ka) and os.path.isdir(sa):
            return ka, sa, float(fac)
    return None


def _index_needing_pad(files, sroot, pad_multiple):
    """Index of the first volume whose grid is NOT a multiple of `pad_multiple`.

    Without this the fixture takes file 0, and on a split that is mostly
    320x320 the real-data test would exercise an IDENTITY Truncate every time
    -- passing while never touching the code path it exists to cover.
    """
    import h5py
    for i, name in enumerate(files):
        try:
            with h5py.File(os.path.join(sroot, name), "r") as f:
                key = "image" if "image" in f else "smaps"
                H, W = f[key].shape[-2], f[key].shape[-1]
        except Exception:                                     # noqa: BLE001
            continue
        if H % pad_multiple or W % pad_multiple:
            return i
    return None


def load_slice(anatomy="brain", pad_multiple=8, index=0, slice_idx=None,
               prefer_padded=True):
    """One recon sample straight out of `FastMRIDataset`, or None.

    Goes through the real dataset rather than reading h5 directly, so the
    5-tuple contract and the `pad_multiple` plumbing are themselves under test.

    `prefer_padded` picks a volume that actually needs the embedding when one
    exists, so the interesting path is the one measured.
    """
    roots = resolve_roots(anatomy)
    if roots is None:
        return None
    kroot, sroot, scale_fac = roots
    try:
        from datasets.fastmri.loader import FastMRIDataset
        ds = FastMRIDataset(task="recon", anatomy=anatomy,
                            kspace_root=kroot, smap_root=sroot,
                            scale_fac=scale_fac, pad_multiple=pad_multiple,
                            start_slice=0, end_slice=1, random_flips=False)
        if len(ds) == 0:
            return None
        if slice_idx is not None:
            ds.start_slice, ds.end_slice = slice_idx, slice_idx + 1
        pick = index
        if prefer_padded:
            j = _index_needing_pad(ds.files, sroot, pad_multiple)
            if j is not None:
                pick = j
        kspace, smaps, image, organ_mask, pad_hw = ds[pick % len(ds)]
    except Exception as e:                                    # noqa: BLE001
        print(f"  [fixture] dataset construction failed: {type(e).__name__}: {e}")
        return None

    # add the batch axis the training loop's collate would
    def b(t):
        return t if t.dim() == 4 else t.unsqueeze(0)

    return dict(kspace=b(kspace), smaps=b(smaps), image=b(image),
                organ_mask=b(organ_mask), pad_hw=pad_hw.reshape(1, 2),
                n_files=len(ds), kspace_root=kroot, smap_root=sroot,
                anatomy=anatomy, scale_fac=scale_fac, file=ds.files[pick % len(ds)])


def adjoint_tol(numel, factor=1.0):
    """fp32 accumulation bound for an inner product over `numel` terms.

    An adjointness residual is a difference of two sums, so it grows like
    sqrt(N) * eps -- a fixed threshold calibrated on a 60x60 phantom fails at
    320x320 for no reason but arithmetic. Measured ratios to this bound run
    0.03-0.17 across grids and coil counts, so it holds with room to spare.
    """
    return factor * float(torch.finfo(torch.float32).eps) * (float(numel) ** 0.5)


def survey_sizes(anatomy="brain", pad_multiple=8, limit=None):
    """`{(H, W): count}` over the split, from HDF5 headers only. None if absent."""
    import h5py
    roots = resolve_roots(anatomy)
    if roots is None:
        return None
    _, sroot, _ = roots
    from collections import Counter
    sizes = Counter()
    files = sorted(glob.glob(os.path.join(sroot, "*.h5")))
    for path in files[:limit] if limit else files:
        try:
            with h5py.File(path, "r") as f:
                key = "image" if "image" in f else "smaps"
                shape = f[key].shape
        except Exception:                                     # noqa: BLE001
            continue
        sizes[(int(shape[-2]), int(shape[-1]))] += 1
    return sizes


def banner(sample):
    """One line naming what the real-data tests actually ran on."""
    if sample is None:
        return ("  [fixture] no fastMRI data reachable -- real-data checks SKIPPED.\n"
                "            set FASTMRI_KSPACE_ROOT and FASTMRI_SMAP_ROOT to enable.")
    H, W = sample["image"].shape[-2:]
    big = (int(sample['pad_hw'][0, 0]), int(sample['pad_hw'][0, 1]))
    fires = "" if big == (H, W) else f"  ->  embeds to {big[0]}x{big[1]}"
    return (f"  [fixture] {sample['anatomy']} {H}x{W}{fires}, "
            f"{sample['smaps'].shape[1]} coils, {sample['n_files']} volumes, "
            f"scale_fac={sample['scale_fac']:g}\n"
            f"            {sample['file']} in {sample['smap_root']}")
