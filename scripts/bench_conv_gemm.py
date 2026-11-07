"""
The LGGS dictionary convs at knee size: cuDNN vs unfold + GEMM (models/conv_gemm.py).

    python scripts/bench_conv_gemm.py                       # knee: 640 x 368, M = 169, P = 7
    python scripts/bench_conv_gemm.py --iters 50 --batch 2

Times, per backend, forward + backward (input AND weight gradients) of

    analysis    conv2d            planar image (B, 2, H, W)    -> latent (B, 2M, H/2, W/2)
    synthesis   conv_transpose2d  latent (B, 2M, H/2, W/2)     -> planar image (B, 2, H, W)
    guide       conv2d            planar image (B, 2, H, W)    -> (B, 4 Mh, H/2, W/2)
                                  (ProjectedGuide's block-weight conv on a refresh layer)

with the planar-state shapes LGGS trains on, and scores each fp32 backend against a float64
reference (cuDNN in double) -- output, dx / dz and dW -- so the speed comes with its accuracy:

    cudnn        F.conv2d / F.conv_transpose2d, cudnn.benchmark on (as train.py)
    gemm-ieee    IMMAP_CONV_GEMM=ieee: unfold + IEEE fp32 GEMMs
    gemm-tf32    IMMAP_CONV_GEMM=tf32: unfold + TF32 GEMMs

Then the whole step:  IMMAP_CONV_GEMM=ieee python scripts/bench_lggs_materialize.py --stages profile
"""

import argparse
import os
import statistics
import sys

import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from models import conv_gemm                                         # noqa: E402


def rel(a, b):
    return float((a.double() - b).norm() / b.norm().clamp_min(1e-300))


def timed(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--H", type=int, default=640)
    ap.add_argument("--W", type=int, default=368)
    ap.add_argument("--M", type=int, default=169)
    ap.add_argument("--Mh", type=int, default=64)
    ap.add_argument("--P", type=int, default=7)
    ap.add_argument("--s", type=int, default=2)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("needs CUDA")
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")       # what models/groupcdl.py leaves in force
    print(f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}, cudnn.benchmark True, "
          f"cudnn.allow_tf32 {torch.backends.cudnn.allow_tf32}, matmul precision high")

    B, P, s = a.batch, a.P, a.s
    p, op = (P - 1) // 2, s - 1
    Hl, Wl = (a.H + 2 * p - P) // s + 1, (a.W + 2 * p - P) // s + 1
    g = torch.Generator(device=dev).manual_seed(0)
    rn = lambda *sh: torch.randn(*sh, device=dev, generator=g, dtype=torch.float64)  # noqa: E731
    convs = {
        "analysis": dict(kind="conv", x=rn(B, 2, a.H, a.W), W=rn(2 * a.M, 2, P, P) * 0.05),
        "synthesis": dict(kind="convT", x=rn(B, 2 * a.M, Hl, Wl), W=rn(2 * a.M, 2, P, P) * 0.05),
        "guide": dict(kind="conv", x=rn(B, 2, a.H, a.W), W=rn(4 * a.Mh, 2, P, P) * 0.05),
    }
    for k, c in convs.items():
        c["grad_x"] = k != "guide"                 # the guide image is data: no dx
        c["f32"] = {"x": c["x"].float(), "W": c["W"].float()}
        out = (F.conv2d(c["x"], c["W"], stride=s, padding=p) if c["kind"] == "conv" else
               F.conv_transpose2d(c["x"], c["W"], stride=s, padding=p, output_padding=op))
        c["g"] = rn(*out.shape)
        c["f32"]["g"] = c["g"].float()

    def run(c, dtype, fwd_only=False):
        src = c if dtype == torch.float64 else c["f32"]   # fp32 copies made once, not timed
        x = src["x"].detach().requires_grad_(c["grad_x"])
        W = src["W"].detach().requires_grad_()
        if c["kind"] == "conv":
            out = conv_gemm.conv2d(x, W, stride=s, padding=p)
        else:
            out = conv_gemm.conv_transpose2d(x, W, stride=s, padding=p, output_padding=op)
        if fwd_only:
            return out
        ins = [t for t in (x, W) if t.requires_grad]
        return (out,) + torch.autograd.grad(out, ins, src["g"])

    # float64 references (cuDNN in double)
    conv_gemm.set_mode(None)
    refs = {k: [t.detach() for t in run(c, torch.float64)] for k, c in convs.items()}

    backends = [("cudnn", None), ("gemm-ieee", "ieee"), ("gemm-tf32", "tf32")]
    res = {}
    for name, mode in backends:
        conv_gemm.set_mode(mode)
        res[name] = {}
        for k, c in convs.items():
            got = run(c, torch.float32)
            errs = [rel(t.detach(), r) for t, r in zip(got, refs[k])]
            with torch.no_grad():
                fwd = timed(lambda: run(c, torch.float32, fwd_only=True), a.warmup, a.iters)
            both = timed(lambda: run(c, torch.float32), a.warmup, a.iters)
            res[name][k] = (fwd, both, errs)
    conv_gemm.set_mode(None)

    print(f"\nB={B}, image {a.H}x{a.W} (2 planar ch), latent {Hl}x{Wl} ({2 * a.M} planar ch), "
          f"P={P}, s={s}; median of {a.iters}")
    print(f"  {'conv':10s}{'backend':11s}{'fwd ms':>9s}{'fwd+bwd ms':>12s}{'speed-up':>10s}"
          f"   rel. error vs float64: out / dx / dW (guide: out / dW)")
    for k in convs:
        base = res["cudnn"][k][1]
        for name, _ in backends:
            fwd, both, errs = res[name][k]
            print(f"  {k:10s}{name:11s}{fwd:9.2f}{both:12.2f}{base / both:9.2f}x   "
                  + " / ".join(f"{e:.1e}" for e in errs))
    per_layer = {name: sum(res[name][k][1] for k in ("analysis", "synthesis"))
                 for name, _ in backends}
    print("\n  one LGGS layer's analysis + synthesis, fwd+bwd: "
          + ", ".join(f"{n} {t:.2f} ms" for n, t in per_layer.items()))
    print(f"  x 30 layers: " + ", ".join(f"{n} {30 * t:.0f} ms" for n, t in per_layer.items()))


if __name__ == "__main__":
    main()
