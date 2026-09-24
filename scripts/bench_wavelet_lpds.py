#!/usr/bin/env python3
"""
GPU timing for WaveletLPDSNet: how K and K^H are computed, and what it costs.

Three implementations of the same operator, for each carry mode:

  gauss         the ORIGINAL path: every level is a `components.Conv2d` /
                `ConvTranspose2d` running the Gauss 3-multiply trick on complex
                maps, then a complex einsum for Q.  With carry="conv" this is
                the model exactly as it was before the carry / interleaved
                changes, so `conv / gauss` is the baseline every speedup is
                quoted against.
  interleaved   `models/wavelet_lpds.py` as it now stands: Re/Im interleaved
                in channels, ONE real conv per level with the block weight
                [[wr, -wi], [wi, wr]] (4 real multiplies per complex MAC, vs
                Gauss's 3), complex <-> real once per operator.
  compile       interleaved + `net.compile_operator()` (Inductor).

The question it answers: with the dense conv carries, does the interleaved
layout's 1 launch per level beat Gauss's 3, or does its +33% conv arithmetic
lose on a GPU where T3 is compute-bound?  If `conv / interleaved` is not
faster than `conv / gauss` in the op-only table, a hybrid (Gauss at T3 only)
is the next thing to try.

Every variant of a carry mode runs the SAME weights (one state dict), and the
last column is its max relative error against `gauss` of that mode -- expect
~1e-3 with TF32 convs (cuDNN's default), ~1e-6 without.

Synthetic SENSE problem (`profile_mg.build_problem`): only shapes matter.

Usage
-----
    python scripts/bench_wavelet_lpds.py                       # wlpds16 defaults
    python scripts/bench_wavelet_lpds.py --config config/brain/mg/wlpds16_R16.json
    python scripts/bench_wavelet_lpds.py --carry conv --impl gauss interleaved

Timing uses CUDA events around each rep; peak memory is per variant.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import types

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from models.wavelet_lpds import NTREES, WaveletLPDSNet     # noqa: E402
from profile_mg import build_problem                       # noqa: E402

# wlpds16 as `make_mg_recon_configs.py` generates it (ML_LPDS_COMMON, M=16)
WLPDS16 = dict(K=30, M=16, L=3, C=1, P=7, s=2, lam0=1.0e-3, tau0=5.0e-1,
               theta0=0.0, degrees=1, is_complex=True, preproc="kspace")

IMPLS = ("gauss", "interleaved", "compile")


# ===========================================================================
#  the original Gauss-trick K / K^H, patched back onto a layer
# ===========================================================================
def _mix(lay, v, Q):
    B, _, h, w = v.shape
    v = v.reshape(B, NTREES, lay.Cg, h, w)
    return torch.einsum("cij,bjchw->bichw", Q, v).reshape(B, -1, h, w)


def _unshuffle(x):
    B, T, n, H, W = x.shape
    x = x.reshape(B, T, n, H // 2, 2, W // 2, 2)
    return x.permute(0, 1, 2, 4, 6, 3, 5).reshape(B, T, 4 * n, H // 2, W // 2)


def _shuffle(x):
    B, T, c, h, w = x.shape
    x = x.reshape(B, T, c // 4, 2, 2, h, w)
    return x.permute(0, 1, 2, 5, 3, 6, 4).reshape(B, T, c // 4, 2 * h, 2 * w)


def _gauss_analyse(self, x):
    if self.carry == "conv":
        for a in self.analysis:
            x = a(x)
        return _mix(self, x, self.Q)
    x = self.analysis[0](x)
    for a in self.analysis[1:]:
        B, _, H, W = x.shape
        x = x.reshape(B, NTREES, -1, H, W)
        ll = a(x[:, :, 0]).reshape(B, NTREES, 4, H // 2, W // 2)
        x = torch.cat([ll, _unshuffle(x[:, :, 1:])], 2).reshape(B, -1, H // 2, W // 2)
    return _mix(self, x, self.Q)


def _gauss_adjoint(self, z):
    z = _mix(self, z, self.QH)
    if self.carry == "conv":
        for b in reversed(self.synthesis):
            z = b(z)
        return z
    for b in reversed(self.synthesis[1:]):
        B, _, h, w = z.shape
        z = z.reshape(B, NTREES, -1, h, w)
        ll = b(z[:, :, :4].reshape(B, 4 * NTREES, h, w))
        z = torch.cat([ll.unsqueeze(2), _shuffle(z[:, :, 4:])], 2)
        z = z.reshape(B, -1, 2 * h, 2 * w)
    return self.synthesis[0](z)


def use_gauss(net):
    for lay in net.net.layers:
        lay.analyse = types.MethodType(_gauss_analyse, lay)
        lay.adjoint = types.MethodType(_gauss_adjoint, lay)


# ===========================================================================
#  timing
# ===========================================================================
def timed(fn, reps, warmup, dev):
    """Per-rep ms (CUDA events on GPU), and peak GB over the timed reps."""
    cuda = dev.type == "cuda"
    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(reps):
        if cuda:
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            fn()
            e.record()
            e.synchronize()
            samples.append(s.elapsed_time(e))
        else:
            t = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - t) * 1e3)
    peak = torch.cuda.max_memory_allocated() / 1e9 if cuda else float("nan")
    return dict(med=statistics.median(samples), lo=min(samples),
                hi=max(samples), peak=peak)


def make(params, carry, impl, state, dev, compile_mode):
    net = WaveletLPDSNet(**params, carry=carry).to(dev)
    net.load_state_dict(state)
    if impl == "gauss":
        use_gauss(net)
    elif impl == "compile":
        torch._dynamo.reset()          # _analyse/_adjoint share one code object
        net.compile_operator(**({"mode": compile_mode} if compile_mode else {}))
    return net


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="model config JSON (params + mri.R / acs_lines)")
    ap.add_argument("--size", type=int, nargs=2, default=(320, 320), metavar=("H", "W"))
    ap.add_argument("--coils", type=int, default=16)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--R", type=int, default=16)
    ap.add_argument("--acs", type=int, default=24)
    ap.add_argument("--K", type=int, help="override the layer count")
    ap.add_argument("--carry", nargs="+", default=["conv", "unshuffle"],
                    choices=["conv", "unshuffle"])
    ap.add_argument("--impl", nargs="+", default=list(IMPLS), choices=IMPLS)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--compile-mode", default=None,
                    help="torch.compile mode, e.g. max-autotune-no-cudagraphs")
    ap.add_argument("--no-train", action="store_true", help="skip fwd+bwd")
    ap.add_argument("--no-eval", action="store_true", help="skip no_grad fwd")
    ap.add_argument("--no-op", action="store_true", help="skip the K^H K microbench")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    dev = torch.device(args.device)
    torch.backends.cudnn.benchmark = True          # train.py's default

    params = dict(WLPDS16)
    if args.config:
        cfg = json.load(open(args.config))
        if cfg["model"]["type"] != "WaveletLPDSNet":
            raise SystemExit(f"{args.config} is a {cfg['model']['type']}")
        params.update(cfg["model"]["params"])
        args.R = cfg.get("mri", {}).get("R", args.R)
        args.acs = cfg.get("mri", {}).get("acs_lines", args.acs)
    for k in ("carry", "compile_operator"):
        params.pop(k, None)
    if args.K:
        params["K"] = args.K

    H, W = args.size
    y, E, sigma = build_problem((H, W), args.coils, args.batch, args.R, args.acs,
                                0.05, dev)
    sig = torch.full((args.batch, 1, 1, 1), float(sigma), device=dev)

    print(f"[bench] {torch.cuda.get_device_name(dev) if dev.type == 'cuda' else 'cpu'}"
          f"  torch {torch.__version__}")
    print(f"[bench] {H}x{W}, {args.coils} coils, batch {args.batch}, R={args.R}, "
          f"K={params['K']}, P={params['P']}, preproc={params['preproc']}")
    print(f"[bench] cudnn.benchmark=True  cudnn.allow_tf32="
          f"{torch.backends.cudnn.allow_tf32}  reps={args.reps} warmup={args.warmup}")

    # one weight draw per carry mode, shared by all its implementations
    states = {}
    for carry in args.carry:
        torch.manual_seed(0)
        states[carry] = WaveletLPDSNet(**params, carry=carry).state_dict()

    def train_step(net):
        def f():
            net.zero_grad(set_to_none=True)
            x_hat, _ = net(y, E=E, sigma=sig)
            x_hat.abs().pow(2).mean().backward()
        return f

    def eval_step(net):
        def f():
            with torch.no_grad():
                net(y, E=E, sigma=sig)
        return f

    def op_step(net):
        lay = net.layer(0)
        x = torch.randn(args.batch, 1, H, W, dtype=torch.complex64, device=dev,
                        requires_grad=True)
        def f():
            lay.zero_grad(set_to_none=True)
            lay.adjoint(lay.analyse(x)).abs().pow(2).mean().backward()
        return f

    sections = []
    if not args.no_train:
        sections.append(("full net, fwd+bwd (train step)", train_step))
    if not args.no_eval:
        sections.append(("full net, fwd only (no_grad)", eval_step))
    if not args.no_op:
        sections.append(("one layer's K^H K, fwd+bwd (operator only)", op_step))

    for title, step in sections:
        print(f"\n=== {title} ===")
        print(f"{'carry':10s}{'impl':13s}{'median ms':>11s}{'min':>9s}{'max':>9s}"
              f"{'peak GB':>9s}{'vs conv/gauss':>15s}{'rel err':>10s}")
        base, ref = None, {}
        for carry in args.carry:
            for impl in args.impl:
                try:
                    net = make(params, carry, impl, states[carry], dev,
                               args.compile_mode)
                    r = timed(step(net), args.reps, args.warmup, dev)
                    with torch.no_grad():
                        out = net(y, E=E, sigma=sig)[0]
                    if impl == "gauss":
                        ref[carry] = out
                    err = ((out - ref[carry]).abs().max() / ref[carry].abs().max()
                           ).item() if carry in ref else float("nan")
                    if carry == "conv" and impl == "gauss":
                        base = r["med"]
                    sp = f"{base / r['med']:.2f}x" if base else "--"
                    print(f"{carry:10s}{impl:13s}{r['med']:>11.2f}{r['lo']:>9.2f}"
                          f"{r['hi']:>9.2f}{r['peak']:>9.2f}{sp:>15s}{err:>10.1e}",
                          flush=True)
                    del net, out
                except Exception as ex:                        # keep going
                    print(f"{carry:10s}{impl:13s}   {type(ex).__name__}: "
                          f"{str(ex).splitlines()[0][:70]}", flush=True)
                if dev.type == "cuda":
                    torch.cuda.empty_cache()

    print("\n[bench] done.")


if __name__ == "__main__":
    main()
