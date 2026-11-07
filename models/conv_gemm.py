"""
Convolutions with a NARROW side, as unfold + one GEMM -- for the dictionary convs of LGGS/LPDS.

Why
---
The analysis / synthesis pair of an unrolled primal-dual net maps a C = 1 complex image (2
planar channels) to an M = 169 complex latent (338 planar channels) with a 7 x 7 stride-2
kernel. Profiled on an L40S (scripts/bench_lggs_materialize.py --stages profile), cuDNN's
weight gradient for these shapes ran at ~12 TFLOP/s, ~13 % of the card's fp32 peak, even with
`cudnn.benchmark` -- the single most expensive kernel of an LGGS training step. Nothing in its
algorithm table suits a conv whose input has two channels.

But such a conv IS a small matrix product. With K = Cin * P^2 (= 98 here) values per output
pixel,

    conv2d(x, W)            = W[N, K]   @ unfold(x)[K, L]                 analysis
    conv_transpose2d(z, W)  = fold( W[N, K]^T @ z[N, L] )                 synthesis

and every gradient is a GEMM too: dW = g @ unfold(x)^T, a long-K reduction cuBLAS does well;
dx = fold(W^T @ g). `unfold(x)` is only K * L floats (~23 MB for a 640 x 368 image), and it
is RECOMPUTED in the backward rather than kept.

Switch: `IMMAP_CONV_GEMM` = 0 (off, default) | ieee (exact fp32 GEMMs) | tf32 (TF32 GEMMs,
faster, ~1e-3 relative per product). `ieee` / `tf32` are applied INSIDE these Functions, in
the forward and the backward alike, so the process-wide matmul precision (models/groupcdl.py
sets "high") does not leak in. Taken only where expressible -- groups = 1, no bias, and the
narrow side's K <= `IMMAP_CONV_GEMM_MAX_K` (default 256) -- otherwise the cuDNN conv runs.
"""

from __future__ import annotations

import contextlib
import os

import torch
import torch.nn.functional as F

_MODES = {"0": None, "": None, "off": None, "1": "ieee", "ieee": "ieee", "tf32": "tf32"}


def _env_mode():
    v = os.environ.get("IMMAP_CONV_GEMM", "0").strip().lower()
    if v not in _MODES:
        raise ValueError(f"IMMAP_CONV_GEMM must be one of 0 / ieee / tf32, got {v!r}")
    return _MODES[v]


MODE = _env_mode()                     # None | "ieee" | "tf32"; tests and tools set it directly
MAX_K = int(os.environ.get("IMMAP_CONV_GEMM_MAX_K", "256"))


def set_mode(mode):
    """Set the switch for the process (None / "ieee" / "tf32") -> the previous value."""
    global MODE
    if mode not in (None, "ieee", "tf32"):
        raise ValueError(f"conv GEMM mode must be None, 'ieee' or 'tf32'; got {mode!r}")
    old, MODE = MODE, mode
    return old


@contextlib.contextmanager
def _precision(mode):
    """fp32 matmul precision for the GEMMs of one call (forward or backward)."""
    keep = torch.get_float32_matmul_precision()
    want = "highest" if mode == "ieee" else "high"
    if keep == want:
        yield
        return
    torch.set_float32_matmul_precision(want)
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(keep)


def _out_len(n, P, s, p):
    return (n + 2 * p - P) // s + 1


def _mm_sum(a, b):
    """sum_b a[b] @ b[b]^T for (B, N, L), (B, K, L) -> (N, K)."""
    if a.shape[0] == 1:
        return a[0] @ b[0].transpose(0, 1)
    return torch.bmm(a, b.transpose(1, 2)).sum(0)


class _Conv(torch.autograd.Function):
    """conv2d(x, W; stride s, padding p), W (N, Cin, P, P), as W @ unfold(x)."""

    @staticmethod
    def forward(ctx, x, W, s, p, mode):
        B, Cin, H, Wd = x.shape
        N, _, P, _ = W.shape
        Ho, Wo = _out_len(H, P, s, p), _out_len(Wd, P, s, p)
        with _precision(mode):
            cols = F.unfold(x, P, padding=p, stride=s)                      # (B, K, L)
            out = torch.matmul(W.reshape(N, -1), cols)                      # (B, N, L)
        ctx.save_for_backward(x, W)
        ctx.cfg = (s, p, mode)
        return out.reshape(B, N, Ho, Wo)

    @staticmethod
    def backward(ctx, g):
        x, W = ctx.saved_tensors
        s, p, mode = ctx.cfg
        B, Cin, H, Wd = x.shape
        N, _, P, _ = W.shape
        g = g.reshape(B, N, -1)
        W2 = W.reshape(N, -1)
        dx = dW = None
        with _precision(mode):
            if ctx.needs_input_grad[0]:
                dx = F.fold(torch.matmul(W2.t(), g), (H, Wd), P, padding=p, stride=s)
            if ctx.needs_input_grad[1]:
                dW = _mm_sum(g, F.unfold(x, P, padding=p, stride=s)).reshape(W.shape)
        return dx, dW, None, None, None


class _ConvT(torch.autograd.Function):
    """conv_transpose2d(z, W; stride s, padding p) onto an (H, W) grid, W (N, Cout, P, P) in
    torch's (in, out) layout, as fold(W^T @ z) -- the adjoint of `_Conv` with the same W."""

    @staticmethod
    def forward(ctx, z, W, s, p, out_hw, mode):
        B, N, Hi, Wi = z.shape
        P = W.shape[-1]
        with _precision(mode):
            cols = torch.matmul(W.reshape(N, -1).t(), z.reshape(B, N, -1))  # (B, K, L)
            y = F.fold(cols, out_hw, P, padding=p, stride=s)
        ctx.save_for_backward(z, W)
        ctx.cfg = (s, p, mode)
        return y

    @staticmethod
    def backward(ctx, gy):
        z, W = ctx.saved_tensors
        s, p, mode = ctx.cfg
        B, N, Hi, Wi = z.shape
        P = W.shape[-1]
        dz = dW = None
        with _precision(mode):
            gcols = F.unfold(gy, P, padding=p, stride=s)                     # (B, K, L)
            if ctx.needs_input_grad[0]:
                dz = torch.matmul(W.reshape(N, -1), gcols).reshape(z.shape)
            if ctx.needs_input_grad[1]:
                dW = _mm_sum(z.reshape(B, N, -1), gcols).reshape(W.shape)
        return dz, dW, None, None, None, None


def _usable(x, W, bias, groups, K):
    return (MODE is not None and bias is None and groups == 1 and K <= MAX_K
            and x.dim() == 4 and not x.is_complex() and not W.is_complex()
            and x.dtype == W.dtype)


def conv2d(x, W, bias=None, stride=1, padding=0, groups=1):
    """`F.conv2d`, through unfold + GEMM when the switch is on and the input side is narrow."""
    s, p = int(stride), int(padding)
    if _usable(x, W, bias, groups, W.shape[1] * W.shape[2] * W.shape[3]) \
            and W.shape[2] == W.shape[3]:
        return _Conv.apply(x, W, s, p, MODE)
    return F.conv2d(x, W, bias=bias, stride=stride, padding=padding, groups=groups)


def conv_transpose2d(z, W, bias=None, stride=1, padding=0, output_padding=0, groups=1):
    """`F.conv_transpose2d`, through GEMM + fold when the switch is on and the OUTPUT side is
    narrow (W is (in, out, P, P); K = out * P^2)."""
    s, p, op = int(stride), int(padding), int(output_padding)
    P = W.shape[-1]
    if _usable(z, W, bias, groups, W.shape[1] * P * P) and W.shape[2] == W.shape[3]:
        Hi, Wi = z.shape[-2:]
        out_hw = ((Hi - 1) * s - 2 * p + P + op, (Wi - 1) * s - 2 * p + P + op)
        # fold must see exactly one block per input pixel on that grid
        if (_out_len(out_hw[0], P, s, p), _out_len(out_hw[1], P, s, p)) == (Hi, Wi):
            return _ConvT.apply(z, W, s, p, out_hw, MODE)
    return F.conv_transpose2d(z, W, bias=bias, stride=stride, padding=padding,
                              output_padding=output_padding, groups=groups)
