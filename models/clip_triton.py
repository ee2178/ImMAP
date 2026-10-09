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
The COMPLEX kernel (`clip_modulus`) is FORWARD ONLY: it is used when no
autograd graph is being built, and with grad enabled the caller keeps the eager
chain, which differentiates.  This speeds up inference, evaluation and
`scripts/profile_mg.py`.

The PLANAR kernels (`prox_planar`, `prox_planar_grad`) cover training too.  A
planar code is a real tensor, so the map is an ordinary real function of its
two halves and its backward is the transpose of a 2x2 Jacobian per pixel -- no
Wirtinger convention to reproduce.  `prox_planar_grad` is a
`torch.autograd.Function`: one kernel forward, one kernel backward (see "The
backward" below).  Both the clip (dual) and the shrink (primal) are covered.

Returns None rather than raising for anything it cannot express (no triton, not
CUDA, wrong dtype, non-contiguous, a full-resolution threshold map, int32
pointer overflow), so the caller falls back silently and correctly.

The backward
------------
Per pixel, with z = (re, im), a = |z|, ac = max(a, eps), q = t / ac and the
incoming gradient g = (g_re, g_im):

    clip   (dual)     s = min(q, 1)        active where q <= 1,  ds/dq = +1
    shrink (primal)   s = relu(1 - q)      active where q <  1,  ds/dq = -1
    out = s z

    dL/dt = ds/dq (g . z) / ac                              (0 where inactive)
    dL/dz = s g  -  ds/dq  q (g . z) / ac^2  z              (2nd term: active, a >= eps)

For the clip that is `q (g - z (g . z) / a^2)` where it clips -- q times the
part of g tangent to the circle |z| = t, the radial part being annihilated --
and `g` where it does not.  The activity tests and the `a >= eps` test are the
ones `clamp_max`, `relu` and `clamp_min` use in the eager chain, so the two
agree on the measure-zero boundaries as well.  At z = 0 the gradient is `s g`:
finite, where `hypot` would give 0/0.

dL/dt is a REDUCTION (one number per channel).  The kernel's grid is therefore
(batch x channel rows, blocks within a row), so that every program lies inside
one channel; each writes its block's partial sum to its own slot and torch adds
the slots up -- a few thousand floats.  No atomics, and deterministic.

What is checked where
---------------------
The CPU dev box cannot run a Triton kernel, so:

* `_eager_forward` / `_eager_backward` are the same arithmetic in torch.
  tests/test_planar_state.py checks them against autograd through the eager
  chain, emulates the kernels' index arithmetic, and (with `EMULATE`) runs the
  whole network through the autograd Function on these formulas.
* the GROUP prox (models/prox.py::GroupThreshold) ends in the same kind of
  map with the modulus replaced by an envelope `xi` that the windowed
  attention supplies: `out = z * s(t / (xi + eps))`. `scale_planar` /
  `scale_planar_grad` are that step as one kernel each way, returning the
  gradient to `xi` as well so it can continue into the attention. Same
  switches, same first-use check.
* each kernel's FIRST use in a process is compared against those formulas on
  the very tensors it was given; a mismatch, or a kernel that fails to build,
  disables it with a warning and the formulas (then the eager chain) take over.
  `planar_kernel_report()` says which state each one is in.
"""

from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable

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


@_jit
def _prox_kernel_planar(Z, T, OUT, n_elem, HW, TN, MHW, EPS,
                        DUAL: tl.constexpr, BLOCK: tl.constexpr):
    """The clip (`DUAL`) or the shrink for a PLANAR code: real (B, 2M, H, W),
    `[re; im]` stacked on the channel axis.

    Complex element `i` of batch `b = i // MHW` has its real part at float
    offset `i + b * MHW` (each batch holds 2 * MHW floats) and its imaginary
    part `MHW` further on. `i // HW` is the (batch, channel) row `b * M + c`,
    and `T` holds `TN` thresholds addressed by `row % TN`: `TN = M` is one per
    channel, `TN = B * M` one per batch and channel, `TN = 1` a scalar.
    """
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < n_elem
    off = i + (i // MHW) * MHW

    re = tl.load(Z + off, mask=m, other=0.0)
    im = tl.load(Z + off + MHW, mask=m, other=0.0)
    t = tl.load(T + (i // HW) % TN, mask=m, other=0.0)

    a = tl.sqrt(re * re + im * im)
    q = t / tl.maximum(a, EPS)
    if DUAL:
        s = tl.minimum(q, 1.0)
    else:
        s = tl.maximum(1.0 - q, 0.0)

    tl.store(OUT + off, re * s, mask=m)
    tl.store(OUT + off + MHW, im * s, mask=m)


@_jit
def _prox_backward_planar(Z, T, G, GZ, PART, HW, M, TN, MHW, NB, EPS,
                          DUAL: tl.constexpr, BLOCK: tl.constexpr):
    """The backward of `_prox_kernel_planar` (module docstring, "The backward").

    Grid `(B * M, NB)`: program `(r, pb)` owns block `pb` of the (batch,
    channel) row `r = b * M + c`, so all its lanes share one threshold and its
    contribution to dL/dt is a single number, stored at `PART[r * NB + pb]`.
    Row `r`'s real half starts at float row `b * 2M + c = r + (r // M) * M`.
    """
    r = tl.program_id(0)
    pb = tl.program_id(1)
    j = pb * BLOCK + tl.arange(0, BLOCK)
    m = j < HW
    off = (r + (r // M) * M) * HW + j

    re = tl.load(Z + off, mask=m, other=0.0)
    im = tl.load(Z + off + MHW, mask=m, other=0.0)
    gr = tl.load(G + off, mask=m, other=0.0)
    gi = tl.load(G + off + MHW, mask=m, other=0.0)
    t = tl.load(T + r % TN)

    a = tl.sqrt(re * re + im * im)
    ac = tl.maximum(a, EPS)
    q = t / ac
    d = (gr * re + gi * im) / ac
    if DUAL:
        act = q <= 1.0
        s = tl.where(act, q, 1.0)
        w = tl.where(act, d, 0.0)
    else:
        act = q < 1.0
        s = tl.where(act, 1.0 - q, 0.0)
        w = tl.where(act, -d, 0.0)
    k = tl.where(a >= EPS, w * q / ac, 0.0)

    tl.store(GZ + off, gr * s - k * re, mask=m)
    tl.store(GZ + off + MHW, gi * s - k * im, mask=m)
    tl.store(PART + r * NB + pb, tl.sum(tl.where(m, w, 0.0), 0))


@_jit
def _scale_kernel_planar(Z, XI, T, OUT, n_elem, HW, TN, MHW, EPS,
                         DUAL: tl.constexpr, BLOCK: tl.constexpr):
    """The GROUP prox's last step on a planar code: `out = z * s(t / (xi + eps))`.

    As `_prox_kernel_planar`, with the code's own modulus replaced by an
    envelope `XI` computed elsewhere (the windowed attention): real,
    (B, M, H, W), so complex element `i` reads `XI[i]`. `s` is `min(q, 1)` for
    the group clip (`DUAL`) and `relu(1 - q)` for the group shrink.
    """
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < n_elem
    off = i + (i // MHW) * MHW

    re = tl.load(Z + off, mask=m, other=0.0)
    im = tl.load(Z + off + MHW, mask=m, other=0.0)
    x = tl.load(XI + i, mask=m, other=1.0)
    t = tl.load(T + (i // HW) % TN, mask=m, other=0.0)

    q = t / (x + EPS)
    if DUAL:
        s = tl.minimum(q, 1.0)
    else:
        s = tl.maximum(1.0 - q, 0.0)

    tl.store(OUT + off, re * s, mask=m)
    tl.store(OUT + off + MHW, im * s, mask=m)


@_jit
def _scale_backward_planar(Z, XI, T, G, GZ, GXI, PART, HW, M, TN, MHW, NB, EPS,
                           DUAL: tl.constexpr, BLOCK: tl.constexpr):
    """The backward of `_scale_kernel_planar`, on `_prox_backward_planar`'s grid.

    With den = xi + eps, q = t / den and ds/dq = +1 (clip, where q <= 1) or -1
    (shrink, where q < 1):

        dL/dz  = s g
        dL/dt  = ds/dq (g . z) / den                (summed per channel: PART)
        dL/dxi = -ds/dq (g . z) q / den             (per pixel: GXI, shaped like XI)

    There is no radial term: `s` does not depend on this pixel's own `z` here,
    only through `xi`, whose gradient goes back out to the attention.
    """
    r = tl.program_id(0)
    pb = tl.program_id(1)
    j = pb * BLOCK + tl.arange(0, BLOCK)
    m = j < HW
    off = (r + (r // M) * M) * HW + j
    xo = r * HW + j

    re = tl.load(Z + off, mask=m, other=0.0)
    im = tl.load(Z + off + MHW, mask=m, other=0.0)
    gr = tl.load(G + off, mask=m, other=0.0)
    gi = tl.load(G + off + MHW, mask=m, other=0.0)
    x = tl.load(XI + xo, mask=m, other=1.0)
    t = tl.load(T + r % TN)

    den = x + EPS
    q = t / den
    d = (gr * re + gi * im) / den
    if DUAL:
        act = q <= 1.0
        s = tl.where(act, q, 1.0)
        w = tl.where(act, d, 0.0)
    else:
        act = q < 1.0
        s = tl.where(act, 1.0 - q, 0.0)
        w = tl.where(act, -d, 0.0)

    tl.store(GZ + off, gr * s, mask=m)
    tl.store(GZ + off + MHW, gi * s, mask=m)
    tl.store(GXI + xo, -w * q, mask=m)
    tl.store(PART + r * NB + pb, tl.sum(tl.where(m, w, 0.0), 0))


# Pointer arithmetic is int32: `2 * i` must stay representable.
_MAX_ELEMS = (2 ** 31 - 1) // 2

# Run the planar prox on the torch formulas below instead of the kernels, on
# any device. For tests: it puts the autograd Function and everything around
# it under test on the CPU dev box. Never faster than the eager chain.
EMULATE = False

# (kind, dual) -> True (checked, in use) / False (disabled); absent = not used
# yet in this process. kind is "forward" or "backward".
_STATE = {}


def planar_kernel_report():
    """One line on the planar kernels, for the timing tools."""
    def one(kind):
        # the local prox's kernels and the group prox's ("group forward", ...)
        vals = [_STATE[k] for k in _STATE if k[0].endswith(kind)]
        if not vals:
            return "not used"
        return "active" if all(vals) else "DISABLED"
    return f"fwd {one('forward')}, bwd {one('backward')}"


def _disable(key, why):
    import warnings
    _STATE[key] = False
    warnings.warn(f"planar {'clip' if key[1] else 'shrink'} kernel ({key[0]}) "
                  f"disabled, using the eager formula instead: {why}")


def _halves(x):
    B, C2, H, W = x.shape
    pairs = x.reshape(B, 2, C2 // 2, H, W)
    return pairs, pairs[:, 0], pairs[:, 1]


def _eager_forward(z, t4, eps, dual):
    """`_prox_kernel_planar` in torch. `t4` broadcasts against (B, M, H, W)."""
    pairs, re, im = _halves(z)
    q = t4 / torch.hypot(re, im).clamp_min(eps)
    s = q.clamp_max(1.0) if dual else (1.0 - q).clamp_min(0.0)
    return (pairs * s.unsqueeze(1)).reshape(z.shape)


def _eager_backward(z, t4, g, eps, dual):
    """`_prox_backward_planar` in torch -> `(dL/dz, dL/dt per (batch, channel))`,
    the second of shape (B, M, 1, 1)."""
    _, re, im = _halves(z)
    _, gr, gi = _halves(g)
    a = torch.hypot(re, im)
    ac = a.clamp_min(eps)
    q = t4 / ac
    d = (gr * re + gi * im) / ac
    zero = torch.zeros((), dtype=z.dtype, device=z.device)
    if dual:
        act = q <= 1.0
        s = torch.where(act, q, torch.ones_like(q))
        w = torch.where(act, d, zero)
    else:
        act = q < 1.0
        s = torch.where(act, 1.0 - q, zero)
        w = torch.where(act, -d, zero)
    k = torch.where(a >= eps, w * q / ac, zero)
    gz = torch.stack((gr * s - k * re, gi * s - k * im), dim=1).reshape(z.shape)
    return gz, w.sum(dim=(2, 3), keepdim=True)


def _planar_args(z, t):
    """`(t4, tf, TN)` for an expressible call, else None.

    `t4` is the threshold as a 4-D tensor that broadcasts against the complex
    (B, M, H, W) -- still attached to `t`'s graph -- and `tf` / `TN` the flat
    float32 buffer and count the kernels address it by.
    """
    if not (torch.is_tensor(z) and z.dim() == 4 and z.dtype == torch.float32
            and z.shape[1] % 2 == 0 and z.is_contiguous()):
        return None
    if not (EMULATE or (HAVE_TRITON and z.is_cuda)):
        return None
    B, M = z.shape[0], z.shape[1] // 2
    n = z.numel() // 2
    if n == 0 or n > _MAX_ELEMS or not torch.is_tensor(t):
        return None
    if t.numel() == 1:
        t4 = t.reshape(1, 1, 1, 1)
    elif (t.dim() == 4 and tuple(t.shape[2:]) == (1, 1)
          and t.shape[1] in (1, M) and t.shape[0] in (1, B)):
        # a per-batch scalar has no flat addressing of its own: one per row
        t4 = t.expand(B, M, 1, 1) if t.shape[1] == 1 else t
    elif t.numel() == M:
        t4 = t.reshape(1, M, 1, 1)
    else:
        return None                       # e.g. a full-resolution threshold map
    t4 = t4.to(device=z.device, dtype=torch.float32)
    tf = t4.detach().reshape(-1).contiguous()
    return t4, tf, tf.numel()


def _rel_err(got, ref, scale=None):
    scale = ref.abs().max() if scale is None else scale
    return float((got - ref).abs().max() / scale.clamp_min(1e-30))


def _forward_impl(z, t4, tf, TN, eps, dual, block=1024):
    """The prox of `z`: the kernel where it is in use, else the torch formula."""
    key = ("forward", bool(dual))
    if EMULATE or _STATE.get(key) is False:
        return _eager_forward(z, t4.detach(), eps, dual)
    M, HW = z.shape[1] // 2, z.shape[2] * z.shape[3]
    n = z.numel() // 2
    out = torch.empty_like(z)
    try:
        _prox_kernel_planar[(triton.cdiv(n, block),)](
            z, tf, out, n, HW, TN, M * HW, float(eps), DUAL=bool(dual), BLOCK=block)
    except Exception as e:                                    # noqa: BLE001
        _disable(key, f"{type(e).__name__}: {e}")
        return _eager_forward(z, t4.detach(), eps, dual)
    if key not in _STATE:
        ref = _eager_forward(z, t4.detach(), eps, dual)
        err = _rel_err(out, ref, z.abs().max())
        if not err < 1e-5:
            _disable(key, f"max rel err {err:.2e} against the eager formula")
            return ref
        _STATE[key] = True
    return out


def _backward_impl(z, t4, tf, TN, g, eps, dual, block=1024):
    """`(dL/dz, dL/dt as (B, M, 1, 1))` for the incoming gradient `g`."""
    key = ("backward", bool(dual))
    if EMULATE or _STATE.get(key) is False:
        return _eager_backward(z, t4, g, eps, dual)
    B, M, HW = z.shape[0], z.shape[1] // 2, z.shape[2] * z.shape[3]
    nb = triton.cdiv(HW, block)
    gz = torch.empty_like(z)
    part = torch.empty(B * M * nb, device=z.device, dtype=torch.float32)
    try:
        _prox_backward_planar[(B * M, nb)](
            z, tf, g, gz, part, HW, M, TN, M * HW, nb, float(eps),
            DUAL=bool(dual), BLOCK=block)
    except Exception as e:                                    # noqa: BLE001
        _disable(key, f"{type(e).__name__}: {e}")
        return _eager_backward(z, t4, g, eps, dual)
    gt = part.view(B, M, nb).sum(dim=2).view(B, M, 1, 1)
    if key not in _STATE:
        rz, rt = _eager_backward(z, t4, g, eps, dual)
        # dL/dt is a sum of signed terms: measure its error against the size of
        # what was summed, not against a total that may nearly cancel
        _, re, im = _halves(z)
        _, gr, gi = _halves(g)
        mass = ((gr * re + gi * im).abs() / torch.hypot(re, im).clamp_min(eps)
                ).sum(dim=(2, 3)).max()
        ez, et = _rel_err(gz, rz, g.abs().max()), _rel_err(gt, rt, mass)
        if not (ez < 1e-4 and et < 1e-4):
            _disable(key, f"max rel err dz {ez:.2e}, dt {et:.2e} against the eager formula")
            return rz, rt
        _STATE[key] = True
    return gz, gt


class _PlanarProx(torch.autograd.Function):
    """`_forward_impl` with `_backward_impl` as its derivative.

    Saves its two inputs and nothing else -- the eager chain keeps the modulus,
    the quotient, the clamp and the scale alive for the backward pass.
    """

    @staticmethod
    def forward(ctx, z, t4, eps, dual):
        tf = t4.reshape(-1).contiguous()
        ctx.save_for_backward(z, t4)
        ctx.eps, ctx.dual = eps, dual
        return _forward_impl(z, t4, tf, tf.numel(), eps, dual)

    @staticmethod
    @once_differentiable
    def backward(ctx, g):
        z, t4 = ctx.saved_tensors
        tf = t4.reshape(-1).contiguous()
        gz, gt = _backward_impl(z, t4, tf, tf.numel(), g.contiguous(), ctx.eps, ctx.dual)
        gt = gt.sum_to_size(t4.shape) if ctx.needs_input_grad[1] else None
        return (gz if ctx.needs_input_grad[0] else None), gt, None, None


def prox_planar(z, t, eps, dual=True):
    """The planar clip (`dual`) or shrink in one pass, NO autograd, or None if
    this path cannot be taken.

    `z`  float32, contiguous, (B, 2M, H, W) -- `[re; im]` on the channel axis
    `t`  real: one element, one per complex CHANNEL (`(1, M, 1, 1)`), or one
         per batch and channel (`(B, M, 1, 1)` / `(B, 1, 1, 1)`)
    """
    if _STATE.get(("forward", bool(dual))) is False:
        return None
    args = _planar_args(z, t)
    if args is None:
        return None
    return _forward_impl(z, *args, eps, dual)


def prox_planar_grad(z, t, eps, dual=True):
    """`prox_planar` as a differentiable op (gradients to `z` and to `t`), or
    None if it cannot be taken -- including when either of its two kernels has
    been disabled, since the torch formulas are no faster than the eager chain.
    """
    if (_STATE.get(("forward", bool(dual))) is False
            or _STATE.get(("backward", bool(dual))) is False):
        return None
    args = _planar_args(z, t)
    if args is None:
        return None
    return _PlanarProx.apply(z, args[0], float(eps), bool(dual))


def clip_modulus_planar(z, t, eps):
    """`clip_modulus` for a planar code (no autograd): `prox_planar(dual=True)`."""
    return prox_planar(z, t, eps, dual=True)


# ---------------------------------------------------------------------------
#  the GROUP prox's scale step:  out = z * s(t / (xi + eps))
# ---------------------------------------------------------------------------
def _eager_scale_forward(z, xi, t4, eps, dual):
    """`_scale_kernel_planar` in torch."""
    pairs, _, _ = _halves(z)
    q = t4 / (xi + eps)
    s = q.clamp_max(1.0) if dual else (1.0 - q).clamp_min(0.0)
    return (pairs * s.unsqueeze(1)).reshape(z.shape)


def _eager_scale_backward(z, xi, t4, g, eps, dual):
    """`_scale_backward_planar` in torch
    -> `(dL/dz, dL/dxi, dL/dt per (batch, channel) as (B, M, 1, 1))`."""
    _, re, im = _halves(z)
    _, gr, gi = _halves(g)
    den = xi + eps
    q = t4 / den
    d = (gr * re + gi * im) / den
    zero = torch.zeros((), dtype=z.dtype, device=z.device)
    if dual:
        act = q <= 1.0
        s = torch.where(act, q, torch.ones_like(q))
        w = torch.where(act, d, zero)
    else:
        act = q < 1.0
        s = torch.where(act, 1.0 - q, zero)
        w = torch.where(act, -d, zero)
    gz = torch.stack((gr * s, gi * s), dim=1).reshape(z.shape)
    return gz, -w * q, w.sum(dim=(2, 3), keepdim=True)


def _scale_args(z, xi, t):
    """`_planar_args`, plus the envelope: real float32, one value per complex
    element of `z`."""
    args = _planar_args(z, t)
    if args is None:
        return None
    B, C2, H, W = z.shape
    if not (torch.is_tensor(xi) and xi.dtype == torch.float32
            and tuple(xi.shape) == (B, C2 // 2, H, W) and xi.device == z.device):
        return None
    return args


def _scale_forward_impl(z, xi, t4, tf, TN, eps, dual, block=1024):
    key = ("group forward", bool(dual))
    if EMULATE or _STATE.get(key) is False:
        return _eager_scale_forward(z, xi, t4.detach(), eps, dual)
    M, HW = z.shape[1] // 2, z.shape[2] * z.shape[3]
    n = z.numel() // 2
    out = torch.empty_like(z)
    try:
        _scale_kernel_planar[(triton.cdiv(n, block),)](
            z, xi, tf, out, n, HW, TN, M * HW, float(eps), DUAL=bool(dual), BLOCK=block)
    except Exception as e:                                    # noqa: BLE001
        _disable(key, f"{type(e).__name__}: {e}")
        return _eager_scale_forward(z, xi, t4.detach(), eps, dual)
    if key not in _STATE:
        ref = _eager_scale_forward(z, xi, t4.detach(), eps, dual)
        err = _rel_err(out, ref, z.abs().max())
        if not err < 1e-5:
            _disable(key, f"max rel err {err:.2e} against the eager formula")
            return ref
        _STATE[key] = True
    return out


def _scale_backward_impl(z, xi, t4, tf, TN, g, eps, dual, block=1024):
    key = ("group backward", bool(dual))
    if EMULATE or _STATE.get(key) is False:
        return _eager_scale_backward(z, xi, t4, g, eps, dual)
    B, M, HW = z.shape[0], z.shape[1] // 2, z.shape[2] * z.shape[3]
    nb = triton.cdiv(HW, block)
    gz, gxi = torch.empty_like(z), torch.empty_like(xi)
    part = torch.empty(B * M * nb, device=z.device, dtype=torch.float32)
    try:
        _scale_backward_planar[(B * M, nb)](
            z, xi, tf, g, gz, gxi, part, HW, M, TN, M * HW, nb, float(eps),
            DUAL=bool(dual), BLOCK=block)
    except Exception as e:                                    # noqa: BLE001
        _disable(key, f"{type(e).__name__}: {e}")
        return _eager_scale_backward(z, xi, t4, g, eps, dual)
    gt = part.view(B, M, nb).sum(dim=2).view(B, M, 1, 1)
    if key not in _STATE:
        rz, rx, rt = _eager_scale_backward(z, xi, t4, g, eps, dual)
        _, re, im = _halves(z)
        _, gr, gi = _halves(g)
        mass = ((gr * re + gi * im).abs() / (xi + eps).abs()).sum(dim=(2, 3)).max()
        ez = _rel_err(gz, rz, g.abs().max())
        ex = _rel_err(gxi, rx, rx.abs().max())
        et = _rel_err(gt, rt, mass)
        if not (ez < 1e-4 and ex < 1e-4 and et < 1e-4):
            _disable(key, f"max rel err dz {ez:.2e}, dxi {ex:.2e}, dt {et:.2e} "
                          f"against the eager formula")
            return rz, rx, rt
        _STATE[key] = True
    return gz, gxi, gt


class _PlanarScale(torch.autograd.Function):
    """`_scale_forward_impl` with `_scale_backward_impl` as its derivative:
    gradients to the code, to the envelope and to the threshold."""

    @staticmethod
    def forward(ctx, z, xi, t4, eps, dual):
        tf = t4.reshape(-1).contiguous()
        ctx.save_for_backward(z, xi, t4)
        ctx.eps, ctx.dual = eps, dual
        return _scale_forward_impl(z, xi, t4, tf, tf.numel(), eps, dual)

    @staticmethod
    @once_differentiable
    def backward(ctx, g):
        z, xi, t4 = ctx.saved_tensors
        tf = t4.reshape(-1).contiguous()
        gz, gxi, gt = _scale_backward_impl(z, xi, t4, tf, tf.numel(), g.contiguous(),
                                           ctx.eps, ctx.dual)
        need = ctx.needs_input_grad
        return (gz if need[0] else None, gxi if need[1] else None,
                gt.sum_to_size(t4.shape) if need[2] else None, None, None)


def scale_planar(z, xi, t, eps, dual=True):
    """`z * s(t / (xi + eps))` on a planar code in one pass, NO autograd, or
    None if this path cannot be taken. `xi` real, (B, M, H, W); `z`, `t` as in
    `prox_planar`."""
    if _STATE.get(("group forward", bool(dual))) is False:
        return None
    args = _scale_args(z, xi, t)
    if args is None:
        return None
    return _scale_forward_impl(z, xi.contiguous(), *args, eps, dual)


def scale_planar_grad(z, xi, t, eps, dual=True):
    """`scale_planar` as a differentiable op (gradients to `z`, `xi` and `t`),
    or None if it cannot be taken or either of its kernels has been disabled."""
    if (_STATE.get(("group forward", bool(dual))) is False
            or _STATE.get(("group backward", bool(dual))) is False):
        return None
    args = _scale_args(z, xi, t)
    if args is None:
        return None
    return _PlanarScale.apply(z, xi.contiguous(), args[0], float(eps), bool(dual))


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
