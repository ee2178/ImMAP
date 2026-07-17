# -*- coding: utf-8 -*-
"""
Precompute the end-to-end contrast-synthesis output yhat = net(img[..., input_idx]) for every
slice of every subject *_img.h5, and WRITE IT BACK into that same h5 as a new dataset (default
key 'yhat'). This is the I2SB bridge prior x1 for the yhat -> true-T1ce bridge -- a much shorter,
better-posed bridge than the raw T1 -> T1ce one (the prior already lives in the T1ce domain).

Representation: yhat is stored in the SAME space as the h5 "img" contrasts (z-scored; the synth
net's own input/output space -- SynthesisDataset applies NO intensity scaling, so we feed "img"
directly and store the raw output). The I2SB dataset (x1_source="synth") then divides yhat by
scales[x0_idx] exactly as it divides x0, so x1 lands on x0's scale automatically -- no scale
bookkeeping here.

In-place: this ADDS one dataset ('yhat') per h5; it does not touch 'img'/'mask'. Existing 'yhat'
is skipped unless --overwrite.

Point arg1 at the TRAINED run's SAVED config.json (it carries paths.ckpt). Run from the repo root:

    python datasets/BraTS/synth_precompute.py \
        trained_nets/brats/Synth_T1ce_Pretrain_VGG_CosLR/config.json \
        /home/ee2178/scratch/ee2178/datasets/BraTS/BraTS2021_DataSet_train  [--key yhat] [--overwrite]

Re-run once per split (train, val).
"""

import os
import sys
import json
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import h5py

# repo root on sys.path so `python datasets/BraTS/synth_precompute.py ...` works from anywhere
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from training.common import load_model                      # noqa: E402
from datasets.BraTS.i2sb_dataset import index_img_from_root  # noqa: E402


@torch.no_grad()
def infer_volume(net, img, input_idx, mult, device, batch):
    """img: (n, H, W, C) z-scored. Returns yhat (n, H, W) float32 in the same z-scored space.
    Pads H, W up to a multiple of `mult` (= 2**num_pool_layers) for the Unet2D pooling, then crops
    back, so full (non-32-divisible) slices work."""
    n, H, W, _ = img.shape
    x = np.transpose(img[..., input_idx], (0, 3, 1, 2))               # (n, Cin, H, W)
    x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    pad_h, pad_w = (-H) % mult, (-W) % mult
    out = torch.empty(n, H, W, dtype=torch.float32)
    for i in range(0, n, batch):
        xb = x[i:i + batch].to(device)
        if pad_h or pad_w:
            xb = F.pad(xb, (0, pad_w, 0, pad_h), mode="replicate")
        yb = net(xb)                                                  # (b, 1, Hpad, Wpad)
        out[i:i + batch] = yb[..., :H, :W][:, 0].float().cpu()        # crop back, drop channel
    return out.numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="SAVED synth run config.json (with paths.ckpt set)")
    ap.add_argument("root", help="dataset root: subject folders each holding a *_img.h5")
    ap.add_argument("--key", default="yhat", help="h5 dataset name to write (default: yhat)")
    ap.add_argument("--batch", type=int, default=16, help="slices per forward pass")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing 'yhat' dataset")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.config) as f:
        cfg = json.load(f)
    input_idx = list(cfg["data"]["train"]["input_idx"])              # EXACT training input channels/order
    mult = 2 ** int(cfg["model"]["params"].get("num_pool_layers", 5))
    net = load_model(args.config, device=device)                    # build_model + load ckpt + eval()
    net.eval()
    print(f"[precompute] net={cfg['model']['type']} input_idx={input_idx} pad_mult={mult} "
          f"key={args.key!r} device={device}\n[precompute] root={args.root}")

    img_paths, _ = index_img_from_root(args.root)
    n_done = n_skip = 0
    for p in img_paths:
        with h5py.File(p, "r+") as h:
            if "img" not in h:
                print(f"  [warn] no 'img' in {p} -- skipping"); continue
            if args.key in h:
                if not args.overwrite:
                    print(f"  skip (exists): {os.path.basename(p)}"); n_skip += 1; continue
                del h[args.key]
            img = np.asarray(h["img"])                              # (n, H, W, C) z-scored
            yhat = infer_volume(net, img, input_idx, mult, device, args.batch)
            h.create_dataset(args.key, data=yhat.astype(np.float32))
            n_done += 1
            print(f"  wrote {args.key} {yhat.shape} -> {os.path.basename(p)}")
    print(f"[precompute] done: wrote {n_done}, skipped {n_skip} (of {len(img_paths)}).")


if __name__ == "__main__":
    main()
