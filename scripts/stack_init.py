#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build a K_outer-deep MGLPDSNet checkpoint from a pretrained single-V-cycle one.

    python scripts/stack_init.py \
        --src   trained_nets/stack/pretrain_K1/net.ckpt \
        --config config/brain/mg/mglpds_R8.json \
        --out   trained_nets/stack/damped/init.ckpt \
        --step-scale 0.01

The output is a checkpoint `train.py` can consume through `paths.ckpt`.

WEIGHTS ONLY, NOT A RESUME
--------------------------
`train.py` treats `paths.ckpt` as a full resume: it restores the optimizer, the scheduler and
`start_step = ckpt["step"] + 1`. That is wrong here -- a transferred init wants a fresh Adam
state and the WHOLE cosine schedule, not the tail of the pretrain's.

So this writes `optimizer_state_dict = None`, `scheduler_state_dict = None` (which `load_ckpt`
skips, since it only loads a key that is present AND non-None) and `step = -1`, so
`start_step = -1 + 1 = 0`. Verified by --self-test.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models.stacking import describe, stack_state_dict


def build_from_config(cfg, K_outer=None):
    """Instantiate the config's model, optionally overriding K_outer."""
    from models.mg_lpds import MGLPDSNet

    mtype = cfg["model"]["type"]
    if mtype != "MGLPDSNet":
        raise SystemExit(
            f"[stack] model.type={mtype!r}; this transfer is defined for MGLPDSNet's outer "
            f"stack. Other models have no `net.layers.<j>` to replicate into.")
    params = dict(cfg["model"]["params"])
    K = params.get("K")
    if K is None or isinstance(K, int):
        raise SystemExit(
            f"[stack] K={K!r} -- a bare int is a FLAT stack with no V-cycle. This needs "
            f"K=[K_outer, [i0, i1, ...]].")
    K_outer_cfg, iters = int(K[0]), list(K[1])
    if K_outer is not None:
        params["K"] = [int(K_outer), iters]
    return MGLPDSNet(**params), K_outer_cfg, iters


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", help="pretrained single-V-cycle checkpoint (net.ckpt)")
    ap.add_argument("--config", help="target config; supplies the model params and K_outer")
    ap.add_argument("--out", help="checkpoint to write")
    ap.add_argument("--step-scale", type=float, default=0.01,
                    help="primal step size multiplier in outer slots 1.. "
                         "(1.0 = naive replication, 0.01 = damped; default 0.01)")
    ap.add_argument("--thresh-scale", type=float, default=1.0,
                    help="prox threshold multiplier in the tail. Leave at 1.0 -- see "
                         "models/stacking.py for why zeroing thresholds destroys the pretrain.")
    ap.add_argument("--slot", type=int, default=0, help="source outer slot (default 0)")
    ap.add_argument("--self-test", action="store_true",
                    help="run the built-in checks and exit; needs no files")
    args = ap.parse_args(argv)

    if args.self_test:
        return _self_test()

    for req in ("src", "config", "out"):
        if getattr(args, req) is None:
            ap.error(f"--{req} is required (or pass --self-test)")

    with open(args.config) as f:
        cfg = json.load(f)

    dst, K_outer, iters = build_from_config(cfg)
    print(f"[stack] target: K_outer={K_outer} iters={iters} "
          f"M={cfg['model']['params'].get('M')}")

    ckpt = torch.load(args.src, map_location="cpu", weights_only=False)
    src_state = ckpt.get("model_state_dict")
    if src_state is None:
        raise SystemExit(f"[stack] {args.src} has no model_state_dict.")

    new_state, stats = stack_state_dict(
        src_state, dst.state_dict(), K_outer,
        step_scale=args.step_scale, thresh_scale=args.thresh_scale, slot=args.slot)
    dst.load_state_dict(new_state)          # strict: proves nothing was left behind

    print(f"[stack] prototype tensors={stats['proto_tensors']}  copied={stats['copied']}  "
          f"skipped={stats['skipped']}")
    print(f"[stack] step_scale={args.step_scale} thresh_scale={args.thresh_scale}")
    d = describe(dst.state_dict())
    print(f"[stack] step size per slot : {d['step_size_per_slot']}")
    print(f"[stack] threshold per slot : {d['threshold_per_slot']}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    # step=-1 so train.py's `start_step = step + 1` lands on 0; optimizer / scheduler left None
    # so load_ckpt skips them and training starts with a fresh Adam and the full LR schedule.
    torch.save({"step": -1,
                "model_state_dict": dst.state_dict(),
                "optimizer_state_dict": None,
                "scheduler_state_dict": None}, args.out)
    print(f"[stack] wrote {args.out}")
    return 0


def _self_test():
    """Round-trip the transfer on a tiny model, with no files and no data."""
    from models.mg_lpds import MGLPDSNet
    try:
        from training.common import load_ckpt
    except ImportError:
        # training/__init__ pulls in torchvision, which a bare checkout may not have.
        # The self-test only needs one function, so load the module by path.
        import importlib.util as _u
        _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _sp = _u.spec_from_file_location(
            "_stack_common", os.path.join(_root, "training", "common.py"))
        _m = _u.module_from_spec(_sp); _sp.loader.exec_module(_m)
        load_ckpt = _m.load_ckpt

    P = dict(M=8, C=1, P=3, s=2, widen=1, degrees=1, lam0=1e-3, tau0=0.5, theta0=0.0,
             alpha0=1.0, is_complex=True, preproc="kspace", resize_noise=True)
    iters, K = [2, 2, 2], 6
    torch.manual_seed(0); src = MGLPDSNet(K=[1, iters], **P)
    with torch.no_grad():                     # make the prototype distinguishable
        for n, p in src.named_parameters():
            p.add_(0.05 * torch.randn_like(p))

    ok = True
    for tag, ss in (("naive", 1.0), ("damped", 0.01)):
        torch.manual_seed(1); dst = MGLPDSNet(K=[K, iters], **P)
        new, stats = stack_state_dict(src.state_dict(), dst.state_dict(), K, step_scale=ss)
        dst.load_state_dict(new)
        d = describe(dst.state_dict())
        steps, thr = d["step_size_per_slot"], d["threshold_per_slot"]
        slot0_matches = all(
            torch.equal(v, dst.state_dict()[k])
            for k, v in src.state_dict().items() if k.startswith("net.layers.0."))
        tail_ratio = steps[1] / steps[0] if steps[0] else float("nan")
        thr_flat = max(thr) - min(thr) < 1e-9
        good = slot0_matches and abs(tail_ratio - ss) < 1e-4 and thr_flat
        ok &= good
        print(f"  [{'ok  ' if good else 'FAIL'}] {tag:6s} slot0 verbatim={slot0_matches}  "
              f"tail/slot0 step={tail_ratio:.4f} (want {ss})  thresholds untouched={thr_flat}")

    # the written checkpoint must NOT resume: fresh optimizer, step 0
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "init.ckpt")
        torch.save({"step": -1, "model_state_dict": dst.state_dict(),
                    "optimizer_state_dict": None, "scheduler_state_dict": None}, p)
        torch.manual_seed(2); m2 = MGLPDSNet(K=[K, iters], **P)
        opt = torch.optim.Adam(m2.parameters(), lr=1e-3)
        m2, opt2, sch2, start = load_ckpt(p, model=m2, optimizer=opt, scheduler=None)
        fresh = len(opt2.state) == 0
        good = (start == 0) and fresh
        ok &= good
        print(f"  [{'ok  ' if good else 'FAIL'}] checkpoint is weights-only: start_step="
              f"{start} (want 0), optimizer state empty={fresh}")

    print("SELF-TEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
