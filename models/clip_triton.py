# -*- coding: utf-8 -*-
"""
The Fenchel clip `z -> z * min(1, t / |z|)` as a single fused kernel.

This is `prox_{g*}` for `g = t ||.||_1`, i.e. the dual step of every LPDS
sweep, and it runs once per layer -- 84 times per forward at `K=[6,[4,4,6]]`.

Why it is worth a kernel
------------------------
`SoftThreshold.fenchel` already writes the map in its shortest eager form,

    a = z.abs().clamp_min(eps)
    z * (t / a).clamp_max(1.0)

but that is still four kernels and ~12 passes over the M-channel latent
(counting a read and a write separately, in units of one real half):

    abs         read 2, write 1
    clamp_min   read 1, write 1
    div         read 1, write 1
    clamp_max   read 1, write 1
    mul         read 3, write 2

At M=169 on a 320x160 latent one such unit is 33 MB, so the eager chain moves
~396 MB for an operation whose irreducible traffic is one read and one write of
z: 4 units, ~132 MB.  Nothing here is compute-bound -- an L40S retires ~400
FLOPs per element in the time it takes to read that element -- so pass count IS
the runtime, and this is a 3x.

Eager cannot fuse it, and `torch.compile` cannot either while the latent is
complex64 (inductor falls back on complex elementwise chains).  Hence Triton,
on the `view_as_real` float32 view.

Limits
------
FORWARD ONLY.  `clip_modulus` is used when no autograd graph is being built;
with grad enabled the caller keeps the eager chain, which differentiates.  The
map is not holomorphic, so a hand-written backward would have to reproduce
torch's conjugate-Wirtinger convention for `min(1, t/|z|)` exactly -- worth
doing to speed up TRAINING, not worth guessing at.  This speeds up inference,
evaluation and `scripts/profile_mg.py`.

Returns None rather than raising for anything it cannot express (no triton, not
CUDA, not complex64, non-contiguous, a full-resolution threshold map, int32
pointer overflow), so the caller falls back silently and correctly.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:                                          # pragma: no cover
    HAVE_TRITON = False
    triton = None

    class _TLStub:
        constexpr = int

        def __getattr__(self, name):
            raise RuntimeError("triton is not available")

    tl = _TLStub()


def _jit(fn):
    """`triton.jit` where triton exists, else leave the function alone.

    Same guard as `models/circulant_triton.py`: keeps the module importable on
    the CPU dev box.
    """
    return triton.jit(fn) if HAVE_TRITON else fn


@_jit
def _clip_kernel(Z, T, OUT, n_elem, HW, M, EPS, BLOCK: tl.constexpr):
    """`out = z * min(1, t_c / |z|)`, one complex element per lane.

    `Z` / `OUT` are the float32 `view_as_real` buffers, so element `i` lives at
    float offsets `2i` (real) and `2i + 1` (imag).  The two strided loads hit
    the same cache lines, so the second is served from L1 and the pair streams
    at full bandwidth.

    `t` is per-channel: `c = (i // HW) % M` is the channel of element `i` in a
    contiguous NCHW layout.  A scalar threshold is passed as `M = 1, HW = 1`,
    which makes `c` identically 0.
    """
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < n_elem

    re = tl.load(Z + 2 * i, mask=m, other=0.0)
    im = tl.load(Z + 2 * i + 1, mask=m, other=0.0)
    t = tl.load(T + (i // HW) % M, mask=m, other=0.0)

    # clamp the modulus, keep the phase. `maximum(a, EPS)` only guards the
    # division: where |z| <= t the scale saturates at 1 and returns z exactly,
    # which is the correct value at z = 0 as well.
    a = tl.sqrt(re * re + im * im)
    s = tl.minimum(t / tl.maximum(a, EPS), 1.0)

    tl.store(OUT + 2 * i, re * s, mask=m)
    tl.store(OUT + 2 * i + 1, im * s, mask=m)


# Pointer arithmetic is int32: `2 * i` must stay representable.
_MAX_ELEMS = (2 ** 31 - 1) // 2


def clip_modulus(z, t, eps, block=1024):
    """`z * min(1, t/|z|)` in one pass, or None if this path cannot be taken.

    `z`  complex64, contiguous, (B, C, H, W)
    `t`  real, broadcastable against z, with either one element per CHANNEL
         (the `Polynomial(degrees=d)(sigma)` shape `(1, C, 1, 1)`) or exactly
         one.  A full `(B, C, H, W)` threshold map -- what a spatial `sigma`
         with `degrees > 0` produces -- is not handled and returns None.
    """
    if not (HAVE_TRITON and z.is_cuda and z.dtype == torch.complex64):
        return None
    if z.dim() != 4 or not z.is_contiguous():
        return None
    n = z.numel()
    if n == 0 or n > _MAX_ELEMS:
        return None

    C, HW = z.shape[1], z.shape[2] * z.shape[3]
    nt = t.numel()
    if nt == C:
        tf, M, hw = t.reshape(-1), C, HW
    elif nt == 1:
        tf, M, hw = t.reshape(-1), 1, 1
    else:
        return None
    tf = tf.to(device=z.device, dtype=torch.float32).contiguous()

    out = torch.empty_like(z)
    _clip_kernel[(triton.cdiv(n, block),)](
        torch.view_as_real(z), tf, torch.view_as_real(out),
        n, hw, M, float(eps), BLOCK=block)
    return out
