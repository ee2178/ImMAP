"""
The RAM port (models/ram/): vendored upstream + the ImMAP adapter.

Run with `python -m tests.test_ram`. Needs deepinv. CPU, about a minute.

1. VENDORED = UPSTREAM: models/ram/_upstream/ram.py is github.com/matthieutrs/ram's
   ram/models/ram.py with only the documented header and the moved
   huggingface import. Checked when a clone is found (RAM_UPSTREAM, or
   ../ram next to this repo); skipped, and said so, otherwise.
2. THE OPERATOR WRAPPER IS AN ADJOINT PAIR: <A x, y> = <x, A^H y> in the real
   inner product deepinv's CG uses -- if not, RAM's realignment prox solves
   the wrong system.
3. THE ADAPTER'S FORWARD = UPSTREAM'S `RAM.forward`, bit for bit, at B = 1 with
   a scalar sigma -- the only place the adapter re-states upstream code.
4. PER-SAMPLE SIGMA: a batch of two with different levels gives what each
   sample gives on its own, to deepinv's CG tolerance (the reason that
   method is re-stated at all).
5. THE NOISE CONVERSION: the model is conditioned on sigma / sqrt(2), the
   per-real-component std of ImMAP's complex noise.
6. SIZES: an image whose sides are not multiples of 8 comes back the same size.
7. GRADIENTS: the learned realignment weight is trained, and the ONLY
   parameters without a gradient are the six kinds upstream defines but never
   uses in forward.
10. DRUNet TRUNK INIT (the `ramdrunet` cell): the installed deepinv's DRUNet
   copies into RAM's trunk conv for conv, leaves the rest alone, and a
   mismatched DRUNet is refused; plus the cached pretrained weights, if present.
"""

from __future__ import annotations

import math
import os
import re
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models import build_model                                       # noqa: E402
from models.ram import adapter as ad                                  # noqa: E402
from models.ram._upstream import ram as up                            # noqa: E402
from operators import FFT2D, Mask, Sense                              # noqa: E402

FAIL = []
UPSTREAM_UNUSED = {"PhysicsBlock.gain", "PhysicsBlock.gain_gradx",
                   "PhysicsBlock.gain_grady", "PhysicsBlock.gain_pinvx",
                   "PhysicsBlock.gain_pinvy",
                   "PhysicsBlock.decoding_conv.tail0.zero_conv_skip.weight"}


def check(name, ok, detail=""):
    print(f"[{'ok ' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def problem(B=1, C=4, H=52, W=60, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, 1, H, W, dtype=torch.complex64, generator=g)
    s = torch.randn(B, C, H, W, dtype=torch.complex64, generator=g)
    s = s / s.abs().pow(2).sum(1, keepdim=True).sqrt()
    cols = (torch.rand(1, 1, 1, W, generator=g) > 0.6).float()
    cols[..., W // 2 - 3: W // 2 + 3] = 1.0                       # a centre block
    E = Mask(cols.expand(B, 1, H, W)) @ FFT2D() @ Sense(s)
    return x, E, E.forward(x)


def upstream_file():
    for c in (os.environ.get("RAM_UPSTREAM"),
              os.path.join(os.path.dirname(ROOT), "ram")):
        if c and os.path.exists(os.path.join(c, "ram", "models", "ram.py")):
            return os.path.join(c, "ram", "models", "ram.py")
    return None


def main():
    # Few threads: the workload is tiny (~7 s on a laptop), and on a cluster
    # LOGIN node torch otherwise starts one thread per visible core while the
    # session is capped to a couple -- the convolutions then crawl (seen
    # 2026-10-20: stalled at the first RAM forward on torch-login-b-3).
    # OMP_NUM_THREADS, if set, still wins.
    if "OMP_NUM_THREADS" not in os.environ:
        torch.set_num_threads(min(2, torch.get_num_threads()))

    # 1 ------------------------------------------------------------------
    src = upstream_file()
    if src is None:
        print("[skip] no upstream clone found (set RAM_UPSTREAM) -- vendoring "
              "not re-checked")
    else:
        mine = open(os.path.join(ROOT, "models", "ram", "_upstream", "ram.py"),
                    encoding="utf-8").read()
        theirs = open(src, encoding="utf-8").read()
        body = re.sub(r"\A(#[^\n]*\n)+", "", mine)                     # header
        body = body.replace(
            "            from huggingface_hub import hf_hub_download  "
            "# vendoring change: was module-level\n", "")
        theirs_wo = theirs.replace("from huggingface_hub import hf_hub_download\n", "", 1)
        check("vendored ram.py = upstream, but for the header and the moved import",
              body == theirs_wo, src)

    # 2 ------------------------------------------------------------------
    torch.manual_seed(0)
    x, E, y = problem()
    phys = ad._make_physics(E, torch.tensor([0.01]))
    u = torch.randn(1, 2, 52, 60)
    v = torch.randn(1, 2, 4, 52, 60)
    lhs = float((phys.A(u) * v).sum())
    rhs = float((u * phys.A_adjoint(v)).sum())
    check("the operator wrapper is an adjoint pair", abs(lhs - rhs) < 1e-4 * abs(lhs),
          f"<Au,v>={lhs:.6f}  <u,A^H v>={rhs:.6f}")

    # 3 ------------------------------------------------------------------
    net = build_model({"model": {"type": "RAM", "params": {"in_channels": [2]}}}).eval()
    s_c = 0.015                                         # ImMAP's complex std
    s_r = torch.tensor([s_c / math.sqrt(2)])            # what RAM is given
    y_r = torch.stack((y.real, y.imag), dim=1)
    with torch.no_grad():
        theirs = net.net(y_r, physics=ad._make_physics(E, s_r))   # upstream forward
        mine = ad._ram_forward(net.net, y_r, ad._make_physics(E, s_r), s_r)
        via_model, _ = net(y, E=E, sigma=torch.full((1, 1, 1, 1), s_c))
    check("adapter forward = upstream RAM.forward, bit for bit (B=1)",
          torch.equal(theirs, mine), f"max |diff| {float((theirs - mine).abs().max()):.3g}")
    check("...and the model's output is that tensor as a complex image",
          torch.equal(via_model, torch.complex(mine[:, 0:1], mine[:, 1:2])))

    # 5 ------------------------------------------------------------------
    with torch.no_grad():
        wrong = ad._ram_forward(net.net, y_r, ad._make_physics(E, torch.tensor([s_c])),
                                torch.tensor([s_c]))
    check("conditioned on sigma/sqrt(2), not on ImMAP's complex sigma",
          not torch.equal(wrong, mine) and torch.equal(
              via_model, torch.complex(mine[:, 0:1], mine[:, 1:2])))

    # 4 ------------------------------------------------------------------
    x2, E2, y2 = problem(B=2, seed=3)
    sig = torch.tensor([0.01, 0.02]).view(2, 1, 1, 1)
    with torch.no_grad():
        both, _ = net(y2, E=E2, sigma=sig)
        each = []
        for b in range(2):
            Eb = Mask(E2.ops[0].mask[b:b + 1]) @ FFT2D() @ Sense(E2.ops[2].smaps[b:b + 1])
            each.append(net(y2[b:b + 1], E=Eb, sigma=sig[b:b + 1])[0])
        each = torch.cat(each)
    # Not bit-exact, and not because of the adapter: deepinv's CG prox (the
    # realignment) stops on a residual over the WHOLE batch at tol 1e-4, so a
    # batch and its samples stop at slightly different iterates. Measured
    # 2026-10-20: 6e-5 relative at deepinv's tol, 5e-7 (float32 roundoff) with
    # tol 1e-10. Every generated config uses batch_size 1.
    rel = float((both - each).norm() / each.norm())
    check("a batch of two levels = each sample on its own (to the CG tolerance)",
          rel < 5e-4, f"relative difference {rel:.2g}")

    # 6 ------------------------------------------------------------------
    x3, E3, y3 = problem(H=45, W=61, seed=5)
    with torch.no_grad():
        o3, _ = net(y3, E=E3, sigma=torch.full((1, 1, 1, 1), 0.015))
    check("45x61 in, 45x61 out (RAM pads to 8 internally)",
          tuple(o3.shape) == (1, 1, 45, 61) and bool(torch.isfinite(o3).all()),
          str(tuple(o3.shape)))

    # 7 ------------------------------------------------------------------
    net.train()
    out, _ = net(y, E=E, sigma=torch.full((1, 1, 1, 1), s_c))
    (out.abs() - x.abs()).abs().mean().backward()
    unused = {re.sub(r"^net\.m_\w+\.enc\.\d+\.", "", n)
              for n, p in net.named_parameters() if p.grad is None}
    check("the realignment weight fact_realign gets a gradient",
          net.net.fact_realign.grad is not None
          and bool(torch.isfinite(net.net.fact_realign.grad).all()))
    check("only upstream's never-used parameters lack a gradient",
          unused == UPSTREAM_UNUSED, f"{sorted(unused)}")

    # 8 ------------------------------------------------------------------
    # THE NOISE MODEL STAYS PUT. deepinv 0.4's LinearPhysics.__init__ calls
    # self.to("cpu") by default, and RAM's Pad / MultiScalePhysics wrap the
    # physics without a device and share its noise model -- on a GPU that moved
    # sigma to the CPU mid-forward (RAM_knee_R12, 2026-10-20). The mechanism is
    # a parent's `.to()` recursing into the noise model, which a dtype move
    # exercises on any machine; the control shows the check can fail.
    from deepinv.physics import GaussianNoise
    sig = torch.tensor([0.01])
    phys = ad._make_physics(E, sig)
    before = phys.noise_model.sigma
    wrapped = up.MultiScaleLinearPhysics(up.Pad(phys, (0, 0)), (2, 52, 60))
    wrapped.to(torch.float64)
    after = wrapped.noise_model.sigma
    check("a parent's .to() leaves RAM's noise model (sigma) where it was",
          after is before and after.dtype == torch.float32,
          f"sigma {before.dtype} -> {after.dtype}")
    plain = up.Pad(type(phys)(E, sig), (0, 0))
    plain.noise_model = GaussianNoise(sigma=torch.tensor([0.01]))
    plain.to(torch.float64)
    check("  control: an unpinned GaussianNoise IS moved by the same call",
          plain.noise_model.sigma.dtype == torch.float64)

    # 9 ------------------------------------------------------------------
    # The real thing, wherever there is a GPU (a compute node; skipped here).
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        gnet = build_model({"model": {"type": "RAM",
                                      "params": {"in_channels": [2]}}}).to(dev)
        xg, Eg, yg = problem()
        Eg = Mask(Eg.ops[0].mask.to(dev)) @ FFT2D() @ Sense(Eg.ops[2].smaps.to(dev))
        yg = Eg.forward(xg.to(dev))
        out, _ = gnet(yg, E=Eg, sigma=torch.full((1, 1, 1, 1), 0.015, device=dev))
        (out.abs() - xg.to(dev).abs()).abs().mean().backward()
        check("on CUDA: forward + backward run, output on the GPU",
              out.is_cuda and bool(torch.isfinite(out).all())
              and gnet.net.fact_realign.grad is not None)
    else:
        print("[skip] no CUDA here -- run on a GPU node to exercise the device path")

    # 10 -----------------------------------------------------------------
    # DRUNet TRUNK INIT (`ramdrunet`), against the INSTALLED deepinv's DRUNet:
    # every trunk conv copied, nothing else touched, and a DRUNet that does
    # not fit refused rather than half-copied.
    from deepinv.models import DRUNet
    torch.manual_seed(0)
    d = DRUNet(in_channels=1, out_channels=1, pretrained=None)
    r = ad.RAM(in_channels=[2])
    head0 = r.net.m_head.conv0.weight.clone()
    cond0 = r.net.m_body.enc[0].PhysicsBlock.encoding_conv.state_dict()
    n = ad.init_trunk_from_drunet(r.net, d)
    pairs = ad.drunet_trunk_pairs(r.net, d)
    check("DRUNet init: 62 trunk convolutions copied, each exactly",
          n == 62 and all(torch.equal(a.weight, b.weight) for a, b in pairs), f"n={n}")
    check("DRUNet init: complex head and measurement blocks keep upstream's init",
          torch.equal(r.net.m_head.conv0.weight, head0)
          and all(torch.equal(v, r.net.m_body.enc[0].PhysicsBlock.encoding_conv
                              .state_dict()[k]) for k, v in cond0.items()))
    try:
        ad.init_trunk_from_drunet(ad.RAM(in_channels=[2]).net,
                                  DRUNet(in_channels=1, out_channels=1, nb=5,
                                         pretrained=None))
        check("DRUNet init: a DRUNet of another depth is refused", False)
    except ValueError:
        check("DRUNet init: a DRUNet of another depth is refused", True)
    # The pretrained weights themselves, when cached (a compute node cannot
    # download them; torch/exp13_ram.sbatch says how to fetch them).
    cached = os.path.join(torch.hub.get_dir(), "checkpoints",
                          "drunet_deepinv_gray_finetune_26k.pth")
    if os.path.exists(cached):
        rp = build_model({"model": {"type": "RAM", "params": {
            "in_channels": [2], "init_from": "drunet_gray"}}})
        ref = ad.load_drunet("drunet_gray")
        check("DRUNet init: the cached PRETRAINED gray DRUNet lands in the trunk",
              all(torch.equal(a.weight, b.weight)
                  for a, b in ad.drunet_trunk_pairs(rp.net, ref)))
    else:
        print(f"[skip] no cached pretrained DRUNet at {cached}")

    print(f"\n{'FAILED: ' + ', '.join(FAIL) if FAIL else 'all checks passed'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
