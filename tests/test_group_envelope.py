"""
`GroupThreshold.FUSED_CHAIN`: the group prox's envelope, fused around the
attention (models/clip_triton.py::planar_envelope).

Run with `python -m tests.test_group_envelope`.

    za -> |za|^2 -> [attention] -> sqrt(. + eps)

The squared modulus and the square root are each one kernel that also changes
between the image layout (B, C, H, W) and the attention's sequence layout
(B, 1, H*W, C), with hand-written backwards. Nothing about the model changes.

What is checked, and where:

1. THE KERNEL SOURCE ITSELF, on CPU. The four `@triton.jit` bodies are run
   through a stand-in for `triton.language` that implements load / store /
   arange / program_id with torch indexing. So their index arithmetic, their
   masks and their arithmetic are executed as written -- for a channel count
   that is not a power of two and a pixel count that is not a multiple of the
   tile -- and compared with the torch formulas. Every unmasked access must be
   in bounds and every output element written exactly once.
2. the hand-written BACKWARDS against autograd on the forward formulas;
3. the WIRING: a group prox with the chain on (emulated) against the eager
   chain, output and every gradient, clip and shrink; and that the chain was
   really taken;
4. NEGATIVE CONTROLS: a backward missing its factor of 2, and a kernel reading
   the wrong half, must both be caught by the same comparisons;
5. on a GPU with triton: the real kernels against the formulas, and the whole
   prox on the flex backend, chain on against chain off.

The stand-in cannot see what only the Triton compiler would reject (a type it
will not broadcast, say). That is what (5) and the first-use self-check in
`clip_triton._chain_run` are for: a kernel that disagrees with its formula on
its first call is disabled for the process, with a warning.
"""

from __future__ import annotations

import torch

import models.clip_triton as ct
from models.prox import GroupThreshold, SoftThreshold

FAIL = []


def check(name, ok, detail=""):
    print(f"[{'ok ' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def rel(a, b):
    a, b = a.detach(), b.detach()
    return float((a - b).abs().max() / (b.abs().max() + 1e-30))


# ===========================================================================
#  a stand-in for triton.language, enough to RUN the kernel bodies
# ===========================================================================
class _Ptr:
    """A flat buffer plus an offset expression: what `BUF + offsets` builds."""

    def __init__(self, flat, off=0, writes=None):
        self.flat, self.off = flat, off
        self.writes = writes if writes is not None else torch.zeros(flat.numel(), dtype=torch.long)

    def __add__(self, o):
        return _Ptr(self.flat, self.off + o, self.writes)


class _FakeTL:
    constexpr = int

    def __init__(self):
        self.pid = (0, 0)
        self.oob = 0

    def program_id(self, axis):
        return self.pid[axis]

    @staticmethod
    def arange(a, b):
        return torch.arange(a, b)

    @staticmethod
    def sqrt(x):
        return torch.sqrt(x)

    def _resolve(self, ptr, mask):
        off = torch.as_tensor(ptr.off)
        mask = torch.as_tensor(True) if mask is None else mask
        off, mask = torch.broadcast_tensors(off, mask)
        bad = mask & ((off < 0) | (off >= ptr.flat.numel()))
        self.oob += int(bad.sum())
        return off, mask & ~bad

    def load(self, ptr, mask=None, other=0.0):
        off, mask = self._resolve(ptr, mask)
        val = ptr.flat[off.clamp(0, ptr.flat.numel() - 1)]
        return torch.where(mask, val, torch.as_tensor(other, dtype=val.dtype))

    def store(self, ptr, val, mask=None):
        off, mask = self._resolve(ptr, mask)
        val = torch.as_tensor(val, dtype=ptr.flat.dtype).expand(off.shape)
        ptr.flat[off[mask]] = val[mask]
        ptr.writes.index_add_(0, off[mask].reshape(-1), torch.ones(int(mask.sum()), dtype=torch.long))


def run_kernel(kernel, grid, out_shape, *args, **consts):
    """Execute a kernel body on CPU. -> (output tensor, out-of-bounds count,
    whether every output element was written exactly once).

    Tensor arguments become flat `_Ptr`s; the one named by the string "OUT" in
    `args` is the freshly allocated output.
    """
    fn = getattr(kernel, "fn", kernel)            # triton.JITFunction keeps the source in .fn
    fake, real = _FakeTL(), ct.tl
    out = torch.full(out_shape, float("nan"))
    out_ptr = _Ptr(out.reshape(-1))
    call = [out_ptr if (isinstance(a, str) and a == "OUT") else
            (_Ptr(a.contiguous().reshape(-1)) if torch.is_tensor(a) else a) for a in args]
    ct.tl = fake
    try:
        for p0 in range(grid[0]):
            for p1 in range(grid[1]):
                fake.pid = (p0, p1)
                fn(*call, **consts)
    finally:
        ct.tl = real
    once = bool((out_ptr.writes == 1).all())
    return out, fake.oob, once


SHAPES = [(2, 5, 6, 7), (1, 64, 9, 11), (3, 81, 4, 5), (1, 3, 1, 37)]   # (B, C, H, W)


def test_kernel_source_on_cpu():
    torch.manual_seed(0)
    eps = 1e-8
    for B, C, H, W in SHAPES:
        S = H * W
        bc, bs = ct._tile(C)
        grid = (B, -(-S // bs))
        tag = f"B={B} C={C} S={S} tile={bc}x{bs}"
        za = torch.randn(B, 2 * C, H, W)
        g_seq = torch.randn(B, 1, S, C)
        o = torch.rand(B, 1, S, C)
        g_img = torch.randn(B, C, H, W)
        xi = ct._eager_sqrt_img(o, eps, H, W)

        got, oob, once = run_kernel(ct._abs2_seq_kernel, grid, (B, 1, S, C),
                                    za, "OUT", S, C, C * S, BLOCK_C=bc, BLOCK_S=bs)
        check(f"_abs2_seq_kernel      {tag}",
              oob == 0 and once and rel(got, ct._eager_abs2_seq(za)) < 1e-6,
              f"rel {rel(got, ct._eager_abs2_seq(za)):.1e}, oob {oob}, once {once}")

        got, oob, once = run_kernel(ct._abs2_seq_backward, grid, (B, 2 * C, H, W),
                                    za, g_seq, "OUT", S, C, C * S, BLOCK_C=bc, BLOCK_S=bs)
        ref = ct._eager_abs2_seq_backward(za, g_seq)
        check(f"_abs2_seq_backward    {tag}",
              oob == 0 and once and rel(got, ref) < 1e-6, f"rel {rel(got, ref):.1e}")

        got, oob, once = run_kernel(ct._sqrt_img_kernel, grid, (B, C, H, W),
                                    o, "OUT", S, C, C * S, eps, BLOCK_C=bc, BLOCK_S=bs)
        check(f"_sqrt_img_kernel      {tag}",
              oob == 0 and once and rel(got, xi) < 1e-6, f"rel {rel(got, xi):.1e}")

        got, oob, once = run_kernel(ct._sqrt_img_backward, grid, (B, 1, S, C),
                                    xi, g_img, "OUT", S, C, C * S, BLOCK_C=bc, BLOCK_S=bs)
        ref = ct._eager_sqrt_img_backward(xi, g_img)
        check(f"_sqrt_img_backward    {tag}",
              oob == 0 and once and rel(got, ref) < 1e-6, f"rel {rel(got, ref):.1e}")


def test_backward_formulas():
    torch.manual_seed(1)
    B, C, H, W, eps = 2, 5, 6, 7, 1e-8
    za = torch.randn(B, 2 * C, H, W, dtype=torch.float64, requires_grad=True)
    g = torch.randn(B, 1, H * W, C, dtype=torch.float64)
    (auto,) = torch.autograd.grad((ct._eager_abs2_seq(za) * g).sum(), za)
    check("|za|^2 backward == autograd on its forward",
          rel(ct._eager_abs2_seq_backward(za.detach(), g), auto) < 1e-12)
    o = torch.rand(B, 1, H * W, C, dtype=torch.float64, requires_grad=True)
    gi = torch.randn(B, C, H, W, dtype=torch.float64)
    xi = ct._eager_sqrt_img(o, eps, H, W)
    (auto,) = torch.autograd.grad((xi * gi).sum(), o)
    check("sqrt backward == autograd on its forward",
          rel(ct._eager_sqrt_img_backward(xi.detach(), gi), auto) < 1e-12)


def _prox_run(chain, dual, backend="gather", device="cpu", emulate=True):
    old = (GroupThreshold.FUSED_CHAIN, ct.EMULATE)
    GroupThreshold.FUSED_CHAIN, ct.EMULATE = chain, emulate
    try:
        torch.manual_seed(1)
        M, Mh, H, W = 12, 6, 12, 10
        gt = GroupThreshold(M, Mh=Mh, window=5, tau0=0.05, sim_fun="distance",
                            attn_backend=backend).to(device)
        if backend == "flex" and device != "cpu":
            gt.compile_flex()
        z = torch.randn(1, 2 * M, H, W, generator=torch.Generator().manual_seed(2))
        z = z.to(device).requires_grad_(True)
        out, _ = (gt.fenchel if dual else gt)(z, 0.01, {})
        out.pow(2).sum().backward()
        return out.detach(), [z.grad.clone()] + [p.grad.clone() for p in gt.parameters()
                                                 if p.grad is not None]
    finally:
        GroupThreshold.FUSED_CHAIN, ct.EMULATE = old


def test_wiring():
    calls = {"n": 0}
    orig = ct._PlanarAbs2Seq.forward

    def counted(ctx, za):
        calls["n"] += 1
        return orig(ctx, za)

    ct._PlanarAbs2Seq.forward = staticmethod(counted)
    try:
        for dual in (True, False):
            calls["n"] = 0
            o1, g1 = _prox_run(True, dual)
            taken = calls["n"]
            o0, g0 = _prox_run(False, dual)
            worst = max(rel(a, b) for a, b in zip(g1, g0))
            check(f"group {'clip' if dual else 'shrink'}: chain on == chain off "
                  f"(output and {len(g0)} gradients)",
                  taken > 0 and calls["n"] == taken and rel(o1, o0) < 1e-6 and worst < 1e-5,
                  f"output rel {rel(o1, o0):.1e}, worst grad rel {worst:.1e}, "
                  f"fused calls {taken}")
    finally:
        ct._PlanarAbs2Seq.forward = staticmethod(orig)

    # the kill switches
    old = SoftThreshold.FUSED_GRAD
    SoftThreshold.FUSED_GRAD = False
    try:
        o2, _ = _prox_run(True, True)
    finally:
        SoftThreshold.FUSED_GRAD = old
    o0, _ = _prox_run(False, True)
    check("FUSED_GRAD off: the chain stands down under autograd", rel(o2, o0) == 0.0)


def test_negative_controls():
    # (a) a backward missing its factor of 2
    orig = ct._eager_abs2_seq_backward
    ct._eager_abs2_seq_backward = lambda za, g: 0.5 * orig(za, g)
    try:
        _, bad = _prox_run(True, True)
    finally:
        ct._eager_abs2_seq_backward = orig
    _, good = _prox_run(False, True)
    worst = max(rel(a, b) for a, b in zip(bad, good))
    check("negative control: a wrong |.|^2 backward is detected",
          worst > 1e-2, f"worst grad rel {worst:.2e} (must be large)")

    # (b) a kernel that reads the wrong half: the same comparison as test 1
    B, C, H, W = 2, 5, 6, 7
    S = H * W
    bc, bs = ct._tile(C)
    za = torch.randn(B, 2 * C, H, W)
    wrong, _, _ = run_kernel(ct._abs2_seq_kernel, (B, -(-S // bs)), (B, 1, S, C),
                             za, "OUT", S, C, C * (S + 1), BLOCK_C=bc, BLOCK_S=bs)
    r = rel(torch.nan_to_num(wrong), ct._eager_abs2_seq(za))
    check("negative control: a wrong half-offset in the kernel is detected",
          r > 1e-2, f"rel {r:.2e} (must be large)")


def test_gpu():
    if not (torch.cuda.is_available() and ct.HAVE_TRITON):
        print("[skip] no CUDA + triton here: the real kernels are checked on the "
              "cluster (and by the first-use self-check at run time)")
        return
    dev, eps = "cuda", 1e-8
    torch.manual_seed(0)
    for B, C, H, W in ((1, 64, 40, 23), (2, 81, 17, 9)):
        za = torch.randn(B, 2 * C, H, W, device=dev)
        g_seq = torch.randn(B, 1, H * W, C, device=dev)
        o = torch.rand(B, 1, H * W, C, device=dev)
        g_img = torch.randn(B, C, H, W, device=dev)
        xi = ct._eager_sqrt_img(o, eps, H, W)
        pairs = (("abs2 forward", ct._abs2_impl(za), ct._eager_abs2_seq(za)),
                 ("abs2 backward", ct._abs2_backward_impl(za, g_seq),
                  ct._eager_abs2_seq_backward(za, g_seq)),
                 ("sqrt forward", ct._sqrt_impl(o, eps, H, W), xi),
                 ("sqrt backward", ct._sqrt_backward_impl(xi, g_img),
                  ct._eager_sqrt_img_backward(xi, g_img)))
        for name, got, ref in pairs:
            check(f"GPU kernel, {name} (C={C}, {H}x{W})", rel(got, ref) < 1e-5,
                  f"rel {rel(got, ref):.1e}")
    state = {k[0]: v for k, v in ct._STATE.items() if k[0].startswith("envelope")}
    check("GPU: all four envelope kernels passed their first-use check",
          len(state) == 4 and all(state.values()), str(state))
    for dual in (True, False):
        o1, g1 = _prox_run(True, dual, backend="flex", device=dev, emulate=False)
        o0, g0 = _prox_run(False, dual, backend="flex", device=dev, emulate=False)
        worst = max(rel(a, b) for a, b in zip(g1, g0))
        check(f"GPU, flex: group {'clip' if dual else 'shrink'} chain on == off",
              rel(o1, o0) < 1e-5 and worst < 1e-4,
              f"output rel {rel(o1, o0):.1e}, worst grad rel {worst:.1e}")


def main():
    for fn in (test_kernel_source_on_cpu, test_backward_formulas, test_wiring,
               test_negative_controls, test_gpu):
        print(f"\n--- {fn.__name__}")
        fn()
    print(f"\n{'FAILED: ' + ', '.join(FAIL) if FAIL else 'all checks passed'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
