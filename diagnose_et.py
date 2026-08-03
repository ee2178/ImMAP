#!/usr/bin/env python3
"""One-shot diagnosis of why 'et' is missing from the training h5 files.

Run on the cluster, from anywhere:
    python diagnose_et.py /home/ee2178/scratch/ee2178/datasets/BraTS/BraTS2021_DataSet_train

Answers, in order, the things that can each produce the KeyError you saw:
  1. is the repo checkout actually carrying the backfill code?
  2. does the path the TRAINING config reads resolve to the same files the backfill wrote?
  3. do the case folders even contain a *_seg.nii.gz?
  4. can h5py open these files read-write at all (HDF5 locking / permissions)?
  5. what keys are present, and does seg geometry line up with the stored images?
"""
import os
import sys
import glob
import subprocess

root = sys.argv[1] if len(sys.argv) > 1 else "."
root = os.path.abspath(os.path.expanduser(root))
print(f"root : {root}")
print(f"real : {os.path.realpath(root)}")
if os.path.realpath(root) != root:
    print("       ^ this path is a symlink; the backfill must have targeted either this or")
    print("         its real location -- both write the same files, so either is fine.")

# 1 -- is the checked-out code new enough?
try:
    import preprocessing.cmap as cm
    print(f"\ncmap module : {cm.__file__}")
    print(f"  has add_seg_to_existing : {hasattr(cm, 'add_seg_to_existing')}")
    print(f"  has load_seg            : {hasattr(cm, 'load_seg')}")
except Exception as e:
    print(f"\ncmap module : NOT importable from here ({type(e).__name__}: {e})")
    print("  (run from the repo root, or this check is just inconclusive)")
try:
    head = subprocess.run(["git", "log", "--oneline", "-1"], capture_output=True, text=True,
                          cwd=os.path.dirname(os.path.abspath(__file__)))
    if head.returncode == 0:
        print(f"  repo HEAD               : {head.stdout.strip()}")
except Exception:
    pass

cases = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
print(f"\n{len(cases)} case folder(s)")

import h5py
import numpy as np

tally = {"has_et": 0, "no_et": 0, "no_h5": 0, "no_seg": 0, "unopenable": 0, "misaligned": 0}
shown = 0
for case in cases:
    cdir = os.path.join(root, case)
    imgs = sorted(glob.glob(os.path.join(cdir, "*_img.h5")))
    segs = sorted(glob.glob(os.path.join(cdir, "*_seg.nii.gz")))
    if not imgs:
        tally["no_h5"] += 1
        continue
    if not segs:
        tally["no_seg"] += 1
    p = imgs[0]
    # read-write openability is the check that catches HDF5 file locking on Lustre/GPFS
    try:
        with h5py.File(p, "r+") as f:
            keys = list(f.keys())
            has_et = "et" in f
            tally["has_et" if has_et else "no_et"] += 1
            if shown < 3:
                shown += 1
                n = f["mask"].shape
                crop = int(f.attrs.get("crop_size", -1))
                sl = np.asarray(f["slice_index"])
                print(f"\n  {case}")
                print(f"    keys        : {keys}")
                print(f"    mask shape  : {n}   crop_size attr: {crop}")
                print(f"    slice_index : n={sl.size} range [{sl.min()}, {sl.max()}]")
                print(f"    seg nifti   : {os.path.basename(segs[0]) if segs else 'MISSING'}")
                print(f"    writable    : yes")
                if segs and not has_et:
                    import nibabel as nib
                    s = np.rint(np.asarray(nib.load(segs[0]).get_fdata())).astype(np.int16)
                    s = np.transpose(s, (2, 0, 1))
                    if crop > 0:
                        H, W = s.shape[1], s.shape[2]
                        h0, w0 = (H - crop) // 2, (W - crop) // 2
                        s = s[:, h0:h0 + crop, w0:w0 + crop]
                    s = s[sl][..., None]
                    ok = s.shape == tuple(n)
                    tally["misaligned"] += 0 if ok else 1
                    print(f"    seg labels  : {sorted(np.unique(s).tolist())}")
                    print(f"    ET(==4) frac: {float((s == 4).mean()):.4f}")
                    print(f"    would align : {ok}  (seg {s.shape} vs mask {tuple(n)})")
    except OSError as e:
        tally["unopenable"] += 1
        if tally["unopenable"] == 1:
            print(f"\n  {case}: CANNOT OPEN READ-WRITE -- {type(e).__name__}: {e}")
            print("    ^ if this is 'unable to lock file', re-run the backfill with")
            print("      HDF5_USE_FILE_LOCKING=FALSE set in the environment.")

print("\n" + "=" * 60)
for k, v in tally.items():
    print(f"  {k:12s} {v}")
print("=" * 60)
if tally["has_et"] == len(cases) and cases:
    print("VERDICT: every case has 'et'. The training error must be reading a DIFFERENT root")
    print("         than the one passed here -- compare against data.train.root in the config.")
elif tally["no_seg"]:
    print("VERDICT: case folders are missing *_seg.nii.gz. Nothing to derive ET from; the")
    print("         segmentations need to be present alongside the contrast volumes.")
elif tally["unopenable"]:
    print("VERDICT: h5py cannot open these read-write -- HDF5 locking or file permissions.")
elif tally["misaligned"]:
    print("VERDICT: seg does not line up with the stored images; crop_size/start_slice in the")
    print("         config differ from the run that generated these h5 files.")
elif tally["no_et"]:
    print("VERDICT: files are writable and seg is present, but 'et' was never written here.")
    print("         The backfill ran against a different directory. Re-run with:")
    print(f"           python preprocessing/cmap.py --config config/BraTS/cmap.yaml \\")
    print(f"               --add-seg-only --root {root}")
