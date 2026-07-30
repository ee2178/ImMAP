# -*- coding: utf-8 -*-
"""
Precompute the end-to-end contrast-synthesis output yhat = net(img[..., input_idx]) for every
slice of every subject *_img.h5, and WRITE IT BACK into that same h5 as a new dataset (default
key 'yhat'). This is the I2SB bridge prior x1 for the yhat -> true-T1ce bridge -- a much shorter,
better-posed bridge than the raw T1 -> T1ce one (the prior already lives in the T1ce domain).

Representation: yhat is stored in the SAME space as the h5 "img" contrasts (z-scored). If the
training config set SynthesisDataset `scales`, the net lives in the SCALED space, so this script
divides the input by those scales and multiplies the output back by scales[target_idx] -- the
stored yhat is z-scored either way. The I2SB dataset (x1_source="synth") then divides yhat by
scales[x0_idx] exactly as it divides x0, so x1 lands on x0's scale automatically -- no scale
bookkeeping downstream.

RESIDUAL RUNS (`training.residual_mode: true`) are supported: those nets predict
target - X[:, residual_src_idx] (e.g. T1ce - T1), so the anchor channel is added back here,
exactly as training/synthesis.py lifts its metrics into the T1ce domain:

    yhat = net(X) + X[:, residual_src_idx]

`residual_src_idx` indexes the INPUT STACK X = img[..., input_idx] (not the stored contrasts),
matching training.synthesis.anchor_channel -- so with input_idx [0,1,3] and residual_src_idx 1
the anchor is stored channel 1 (T1). The addition happens in the SCALED space the net trained
in, before the scales[target_idx] multiply back to z-scored, which reproduces the training-time
T1ce-domain estimate term for term.

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
def infer_volume(net, img, input_idx, mult, device, batch, scales=None, target_idx=2,
                 residual_src_idx=None):
    """img: (n, H, W, C) z-scored. Returns yhat (n, H, W) float32 in the same z-scored space.
    Pads H, W up to a multiple of `mult` (= 2**num_pool_layers) for the Unet2D pooling, then crops
    back, so full (non-32-divisible) slices work.

    `scales` must be the training run's SynthesisDataset scales. The net was trained on
    img/scales, so it has to be fed img/scales; the output is then multiplied back by
    scales[target_idx] so yhat stays stored in the unscaled z-scored space this file
    documents (and that I2SBDataset expects, since it divides yhat by scales[x0_idx]
    itself). Feeding unscaled data to a scale-trained net is a silent domain shift.

    `residual_src_idx` (None = plain run) turns on residual lifting: the net's output is the
    residual target - X[:, residual_src_idx], so the anchor channel is added back before the
    crop. The anchor is taken from the SAME padded batch the net saw, so pad/crop stays
    consistent between the two terms."""
    n, H, W, _ = img.shape
    if scales is not None:
        img = img / np.asarray(scales, dtype=np.float32)[None, None, :]
    x = np.transpose(img[..., input_idx], (0, 3, 1, 2))               # (n, Cin, H, W)
    x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    pad_h, pad_w = (-H) % mult, (-W) % mult
    out = torch.empty(n, H, W, dtype=torch.float32)
    for i in range(0, n, batch):
        xb = x[i:i + batch].to(device)
        if pad_h or pad_w:
            xb = F.pad(xb, (0, pad_w, 0, pad_h), mode="replicate")
        yb = net(xb)                                                  # (b, 1, Hpad, Wpad)
        if isinstance(yb, (tuple, list)):        # CDLNet-family: (x_hat, ...)
            yb = yb[0]
        if residual_src_idx is not None:         # residual run: net predicts target - anchor
            yb = yb + xb[:, residual_src_idx:residual_src_idx + 1]
        out[i:i + batch] = yb[..., :H, :W][:, 0].float().cpu()        # crop back, drop channel
    out = out.numpy()
    if scales is not None:
        out = out * float(np.asarray(scales, dtype=np.float32)[target_idx])   # back to z-scored
    return out


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
    train_cfg = cfg["data"]["train"]
    input_idx = list(train_cfg["input_idx"])                         # EXACT training input channels/order
    scales = train_cfg.get("scales", None)                           # EXACT training intensity scaling
    target_idx = int(train_cfg.get("target_idx", [2])[0])
    mult = 2 ** int(cfg["model"]["params"].get("num_pool_layers", 5))

    # residual runs predict target - X[:, residual_src_idx]; infer_volume adds the anchor back.
    # The index addresses the INPUT STACK (len == len(input_idx)), as in synthesis.anchor_channel:
    # an out-of-range slice would silently return an EMPTY tensor, so check it here.
    residual_src_idx = None
    if cfg["training"].get("residual_mode"):
        residual_src_idx = int(cfg["training"].get("residual_src_idx", 0))
        if not 0 <= residual_src_idx < len(input_idx):
            raise SystemExit(
                f"residual_src_idx={residual_src_idx} is out of range for input_idx={input_idx} "
                f"({len(input_idx)} channels). It indexes the input stack, not the stored contrasts.")

    net = load_model(args.config, device=device)                    # build_model + load ckpt + eval()
    net.eval()
    residual_note = ("off" if residual_src_idx is None else
                     f"on (anchor = input ch {residual_src_idx} = stored ch {input_idx[residual_src_idx]})")
    print(f"[precompute] net={cfg['model']['type']} input_idx={input_idx} scales={scales} "
          f"pad_mult={mult} key={args.key!r} residual={residual_note} device={device}\n"
          f"[precompute] root={args.root}")

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
            yhat = infer_volume(net, img, input_idx, mult, device, args.batch,
                                scales=scales, target_idx=target_idx,
                                residual_src_idx=residual_src_idx)
            h.create_dataset(args.key, data=yhat.astype(np.float32))
            n_done += 1
            print(f"  wrote {args.key} {yhat.shape} -> {os.path.basename(p)}")
    print(f"[precompute] done: wrote {n_done}, skipped {n_skip} (of {len(img_paths)}).")


if __name__ == "__main__":
    main()
