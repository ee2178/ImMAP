"""
A circulant-window adjacency whose softmax weights are MATERIALISED once per refresh.

Why
---
`TritonAdjacency` holds an adjacency implicitly: every `apply` re-runs the whole fused
attention -- all W^2 phase-invariant scores (two C-wide reductions each, C = Mh complex) and
the online softmax -- before it touches the values. The guided prox reuses an adjacency for
`dK` layers (LGGS: dK = 5) and applies it once per layer, so 4 applies in 5 recompute a
softmax that has not changed. The weights themselves are small: (B*heads, W^2, H*W) floats,
~53 MB per guide per sample for a 15 x 15 window on the 320 x 184 knee latent -- a size
independent of the channel count. Held explicitly, a cached apply is one weighted window sum.

What is NOT reusable: `Circulant.apply` (models/circulant_attention.py) materialises the
(B, C, Q, K) gathered neighbourhood, ~3.4 GB per apply at Mh = 64, K = 225 on that latent. So
both halves are kernels here, in the offset-loop style of models/circulant_triton.py:

    window_scores(q, k)  -> S (B, K, HW)     s_jo = |<q_j, k_{j+o}>| - 1/2 ||k_{j+o}||^2
    window_apply(P, v)   -> (B, HW, DV)      out_j = sum_o P_oj v_{j+o}      (transpose: P^T)

and `WindowAdjacency` composes them with a torch softmax / log-sum-exp over the K axis. The
scores follow the triton kernel's convention exactly (`-1/2||q||^2` left out, SIM_EPS inside
the modulus), so a branch built here merges with `PointwiseAdjacency` / `TritonAdjacency`
branches through `merge_joint` and is a drop-in for `TritonAdjacency` in the guided prox: the
same map, computed differently. Offsets are ordered o = i * W + j, (dr, dc) = (i - p, j - p).

Layout: P is (B, K, HW), offset-major, so each kernel's per-offset load / store of P is
contiguous along the query axis.

Backends: the triton kernels on CUDA fp32, otherwise a pure-torch reference (an offset loop
over `torch.roll`, differentiated by autograd -- exact, and memory-hungry: tests and small
problems only). `IMMAP_WINDOW_REF=1` forces the reference, which is how the kernels are
checked against it (scripts/bench_lggs_materialize.py).

Backward
--------
apply      out = A v                 dv = A^T dout       (the same kernel, TRANSPOSE)
                                     dP_oj = <dout_j, v_{j+o}>
           out = A^T x               dx = A dout
                                     dP_oj = <x_j, dout_{j+o}>
scores     dS arrives from the torch softmax; dq is accumulated per QUERY over k = j + o,
           dk per KEY over j = k - o (the window is symmetric), exactly as `_bwd_dq` /
           `_bwd_dkdv` do with `ds = gamma (dp - delta)` replaced by the given dS.
"""

from __future__ import annotations

import os

import torch

from models.circulant_triton import (HAVE_TRITON, SIM_EPS, _img_to_seq, _jit,
                                     _next_pow2, _seq_to_img, tl, triton)

FORCE_REFERENCE = os.environ.get("IMMAP_WINDOW_REF", "0") == "1"


def _use_kernel(t):
    return HAVE_TRITON and not FORCE_REFERENCE and t.is_cuda and t.dtype == torch.float32


def _offsets(win):
    p = (win - 1) // 2
    return [(i - p, j - p) for i in range(win) for j in range(win)]


# ===========================================================================
#  torch reference
# ===========================================================================
def _nb(x, dr, dc, H, W):
    """(B, HW, C) -> the value at (r + dr, c + dc), circularly."""
    B, HW, C = x.shape
    return torch.roll(x.reshape(B, H, W, C), shifts=(-dr, -dc), dims=(1, 2)).reshape(B, HW, C)


def window_scores_ref(qr, qi, kr, ki, H, W, win, use_knorm=True, eps=SIM_EPS):
    knorm = (kr * kr).sum(-1, keepdim=True)
    if ki is not None:
        knorm = knorm + (ki * ki).sum(-1, keepdim=True)
    out = []
    for dr, dc in _offsets(win):
        krs = _nb(kr, dr, dc, H, W)
        re = (qr * krs).sum(-1)
        if ki is not None:
            kis = _nb(ki, dr, dc, H, W)
            re = re + (qi * kis).sum(-1)
            im = (qi * krs).sum(-1) - (qr * kis).sum(-1)
            s = torch.sqrt(re * re + im * im + eps)
        else:
            s = torch.sqrt(re * re + eps)
        if use_knorm:
            s = s - 0.5 * _nb(knorm, dr, dc, H, W)[..., 0]
        out.append(s)
    return torch.stack(out, dim=1)                                        # (B, K, HW)


def window_apply_ref(P, v, H, W, win, transpose=False):
    out = torch.zeros_like(v)
    for o, (dr, dc) in enumerate(_offsets(win)):
        w = P[:, o, :, None]
        if transpose:
            out = out + _nb(w * v, -dr, -dc, H, W)       # out_k += P_{o, k-o} v_{k-o}
        else:
            out = out + w * _nb(v, dr, dc, H, W)         # out_j += P_{o, j} v_{j+o}
    return out


# ===========================================================================
#  triton kernels
# ===========================================================================
@_jit
def _wrap(x, n):
    x = tl.where(x < 0, x + n, x)
    return tl.where(x >= n, x - n, x)


@_jit
def _wscores_fwd(
    QR, QI, KR, KI, KNORM, S,
    stride_qb, stride_qn, stride_sb, stride_so, stride_nb,
    H, W, HW, WIN, PAD, EPS,
    HAS_IMAG: tl.constexpr, USE_KNORM: tl.constexpr, C: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1).to(tl.int64)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < HW
    safe_m = tl.where(mask_m, offs_m, 0)
    row = safe_m // W
    col = safe_m % W
    offs_c = tl.arange(0, BLOCK_C)
    qmask = mask_m[:, None] & (offs_c < C)[None, :]
    base = pid_b * stride_qb
    q_off = base + safe_m[:, None] * stride_qn + offs_c[None, :]
    qr = tl.load(QR + q_off, mask=qmask, other=0.0)
    qi = tl.load(QI + q_off, mask=qmask, other=0.0) if HAS_IMAG else qr
    for i in range(WIN):
        nr = _wrap(row + (i - PAD), H)
        for j in range(WIN):
            nc = _wrap(col + (j - PAD), W)
            nn = nr * W + nc
            k_off = base + nn[:, None] * stride_qn + offs_c[None, :]
            kr = tl.load(KR + k_off, mask=qmask, other=0.0)
            re = tl.sum(qr * kr, 1)
            if HAS_IMAG:
                ki = tl.load(KI + k_off, mask=qmask, other=0.0)
                re += tl.sum(qi * ki, 1)
                im = tl.sum(qi * kr, 1) - tl.sum(qr * ki, 1)
                s = tl.sqrt(re * re + im * im + EPS)
            else:
                s = tl.sqrt(re * re + EPS)
            if USE_KNORM:
                s = s - 0.5 * tl.load(KNORM + pid_b * stride_nb + nn, mask=mask_m, other=0.0)
            tl.store(S + pid_b * stride_sb + (i * WIN + j) * stride_so + safe_m, s,
                     mask=mask_m)


@_jit
def _wscores_bwd_dq(
    QR, QI, KR, KI, DS, DQR, DQI,
    stride_qb, stride_qn, stride_sb, stride_so,
    H, W, HW, WIN, PAD, EPS,
    HAS_IMAG: tl.constexpr, C: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr,
):
    """Per QUERY j over its keys k = j + o. The knorm term has no q dependence."""
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1).to(tl.int64)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < HW
    safe_m = tl.where(mask_m, offs_m, 0)
    row = safe_m // W
    col = safe_m % W
    offs_c = tl.arange(0, BLOCK_C)
    qmask = mask_m[:, None] & (offs_c < C)[None, :]
    base = pid_b * stride_qb
    q_off = base + safe_m[:, None] * stride_qn + offs_c[None, :]
    qr = tl.load(QR + q_off, mask=qmask, other=0.0)
    qi = tl.load(QI + q_off, mask=qmask, other=0.0) if HAS_IMAG else qr
    dqr = tl.zeros([BLOCK_M, BLOCK_C], tl.float32)
    dqi = tl.zeros([BLOCK_M, BLOCK_C], tl.float32)
    for i in range(WIN):
        nr = _wrap(row + (i - PAD), H)
        for j in range(WIN):
            nc = _wrap(col + (j - PAD), W)
            nn = nr * W + nc
            k_off = base + nn[:, None] * stride_qn + offs_c[None, :]
            kr = tl.load(KR + k_off, mask=qmask, other=0.0)
            re = tl.sum(qr * kr, 1)
            if HAS_IMAG:
                ki = tl.load(KI + k_off, mask=qmask, other=0.0)
                re += tl.sum(qi * ki, 1)
                im = tl.sum(qi * kr, 1) - tl.sum(qr * ki, 1)
            else:
                ki = kr
                im = re * 0.0
            A = tl.sqrt(re * re + im * im + EPS)
            ds = tl.load(DS + pid_b * stride_sb + (i * WIN + j) * stride_so + safe_m,
                         mask=mask_m, other=0.0)
            c_re = ds * re / A
            c_im = ds * im / A
            dqr += c_re[:, None] * kr - c_im[:, None] * ki
            if HAS_IMAG:
                dqi += c_re[:, None] * ki + c_im[:, None] * kr
    tl.store(DQR + q_off, dqr, mask=qmask)
    if HAS_IMAG:
        tl.store(DQI + q_off, dqi, mask=qmask)


@_jit
def _wscores_bwd_dk(
    QR, QI, KR, KI, DS, DKR, DKI,
    stride_qb, stride_qn, stride_sb, stride_so,
    H, W, HW, WIN, PAD, EPS,
    HAS_IMAG: tl.constexpr, USE_KNORM: tl.constexpr, C: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr,
):
    """Per KEY k over the queries j = k - o that see it (slot o of query j)."""
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1).to(tl.int64)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < HW
    safe_m = tl.where(mask_m, offs_m, 0)
    row = safe_m // W
    col = safe_m % W
    offs_c = tl.arange(0, BLOCK_C)
    kmask = mask_m[:, None] & (offs_c < C)[None, :]
    base = pid_b * stride_qb
    k_off = base + safe_m[:, None] * stride_qn + offs_c[None, :]
    kr = tl.load(KR + k_off, mask=kmask, other=0.0)
    ki = tl.load(KI + k_off, mask=kmask, other=0.0) if HAS_IMAG else kr
    dkr = tl.zeros([BLOCK_M, BLOCK_C], tl.float32)
    dki = tl.zeros([BLOCK_M, BLOCK_C], tl.float32)
    dperkey = tl.zeros([BLOCK_M], tl.float32)
    for i in range(WIN):
        jr = _wrap(row - (i - PAD), H)
        for j in range(WIN):
            jc = _wrap(col - (j - PAD), W)
            jj = jr * W + jc
            q_off = base + jj[:, None] * stride_qn + offs_c[None, :]
            qr = tl.load(QR + q_off, mask=kmask, other=0.0)
            re = tl.sum(qr * kr, 1)
            if HAS_IMAG:
                qi = tl.load(QI + q_off, mask=kmask, other=0.0)
                re += tl.sum(qi * ki, 1)
                im = tl.sum(qi * kr, 1) - tl.sum(qr * ki, 1)
            else:
                qi = qr
                im = re * 0.0
            A = tl.sqrt(re * re + im * im + EPS)
            ds = tl.load(DS + pid_b * stride_sb + (i * WIN + j) * stride_so + jj,
                         mask=mask_m, other=0.0)
            dperkey += ds
            c_re = ds * re / A
            c_im = ds * im / A
            dkr += c_re[:, None] * qr + c_im[:, None] * qi
            if HAS_IMAG:
                dki += c_re[:, None] * qi - c_im[:, None] * qr
    if USE_KNORM:                       # s -= 1/2 ||k||^2  ->  d/dk -= ds k
        dkr -= dperkey[:, None] * kr
        if HAS_IMAG:
            dki -= dperkey[:, None] * ki
    tl.store(DKR + k_off, dkr, mask=kmask)
    if HAS_IMAG:
        tl.store(DKI + k_off, dki, mask=kmask)


@_jit
def _wapply(
    PW, V, OUT,
    stride_pb, stride_po, stride_vb, stride_vn,
    H, W, HW, WIN, PAD,
    TRANSPOSE: tl.constexpr, DV: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_V: tl.constexpr,
):
    """out_j = sum_o P_{o,j} v_{j+o};  TRANSPOSE: out_k = sum_o P_{o,k-o} v_{k-o}."""
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1).to(tl.int64)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < HW
    safe_m = tl.where(mask_m, offs_m, 0)
    row = safe_m // W
    col = safe_m % W
    offs_v = tl.arange(0, BLOCK_V)
    vmask = mask_m[:, None] & (offs_v < DV)[None, :]
    acc = tl.zeros([BLOCK_M, BLOCK_V], tl.float32)
    for i in range(WIN):
        if TRANSPOSE:
            nr = _wrap(row - (i - PAD), H)
        else:
            nr = _wrap(row + (i - PAD), H)
        for j in range(WIN):
            if TRANSPOSE:
                nc = _wrap(col - (j - PAD), W)
            else:
                nc = _wrap(col + (j - PAD), W)
            nn = nr * W + nc
            if TRANSPOSE:
                p_idx = nn                      # weight of query k - o, slot o
            else:
                p_idx = safe_m
            w = tl.load(PW + pid_b * stride_pb + (i * WIN + j) * stride_po + p_idx,
                        mask=mask_m, other=0.0)
            v = tl.load(V + pid_b * stride_vb + nn[:, None] * stride_vn + offs_v[None, :],
                        mask=vmask, other=0.0)
            acc += w[:, None] * v
    tl.store(OUT + pid_b * stride_vb + safe_m[:, None] * stride_vn + offs_v[None, :], acc,
             mask=vmask)


@_jit
def _wdot(
    X, Y, DP,
    stride_pb, stride_po, stride_vb, stride_vn,
    H, W, HW, WIN, PAD,
    DV: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_V: tl.constexpr,
):
    """DP_{o,j} = <x_j, y_{j+o}>."""
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1).to(tl.int64)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < HW
    safe_m = tl.where(mask_m, offs_m, 0)
    row = safe_m // W
    col = safe_m % W
    offs_v = tl.arange(0, BLOCK_V)
    vmask = mask_m[:, None] & (offs_v < DV)[None, :]
    x = tl.load(X + pid_b * stride_vb + safe_m[:, None] * stride_vn + offs_v[None, :],
                mask=vmask, other=0.0)
    for i in range(WIN):
        nr = _wrap(row + (i - PAD), H)
        for j in range(WIN):
            nc = _wrap(col + (j - PAD), W)
            nn = nr * W + nc
            y = tl.load(Y + pid_b * stride_vb + nn[:, None] * stride_vn + offs_v[None, :],
                        mask=vmask, other=0.0)
            tl.store(DP + pid_b * stride_pb + (i * WIN + j) * stride_po + safe_m,
                     tl.sum(x * y, 1), mask=mask_m)


# ===========================================================================
#  autograd wrappers
# ===========================================================================
# The fixed launch (IMMAP_WINDOW_AUTOTUNE=0), and the space `triton.autotune` searches: query
# tile x warps. Every kernel here writes each output element exactly once (plain stores into a
# fresh tensor), so the autotuner's repeated trial launches are harmless, and the tile changes
# only how queries are grouped, never the arithmetic of one output. Tuned once per key -- the
# grid, window and channel widths -- and cached for the process.
_CFG = dict(block_m=64, num_warps=4)
AUTOTUNE = os.environ.get("IMMAP_WINDOW_AUTOTUNE", "1") != "0"
_TILES = [(16, 2), (32, 2), (32, 4), (64, 2), (64, 4), (64, 8), (128, 4), (128, 8)]
_KEYS = {
    "_wscores_fwd": ["HW", "WIN", "C", "HAS_IMAG", "USE_KNORM"],
    "_wscores_bwd_dq": ["HW", "WIN", "C", "HAS_IMAG"],
    "_wscores_bwd_dk": ["HW", "WIN", "C", "HAS_IMAG", "USE_KNORM"],
    "_wapply": ["HW", "WIN", "DV", "TRANSPOSE"],
    "_wdot": ["HW", "WIN", "DV"],
}
_TUNED = {}


def _tuned(name):
    if name not in _TUNED:
        _TUNED[name] = triton.autotune(
            configs=[triton.Config({"BLOCK_M": m}, num_warps=w) for m, w in _TILES],
            key=_KEYS[name])(globals()[name])
    return _TUNED[name]


def _launch(name, HW, B, *args, **meta):
    """One kernel launch over (query tiles, batch): autotuned, or the fixed _CFG tile."""
    if AUTOTUNE:
        _tuned(name)[lambda M: (triton.cdiv(HW, M["BLOCK_M"]), B)](*args, **meta)
    else:
        bm = _CFG["block_m"]
        globals()[name][(triton.cdiv(HW, bm), B)](*args, BLOCK_M=bm,
                                                  num_warps=_CFG["num_warps"], **meta)


def tuned_configs():
    """{kernel: [chosen config per key]} -- what the autotuner picked, for the timing tools."""
    out = {}
    for name, t in _TUNED.items():
        cache = getattr(t, "cache", {}) or {}
        out[name] = [str(c) for c in cache.values()]
    return out


def _check(H, W, win, HW):
    if HW != H * W:
        raise ValueError(f"HW={HW} does not match {H}x{W}")
    if win % 2 == 0 or win > min(H, W):
        raise ValueError(f"window {win} must be odd and fit the {H}x{W} grid")


class _WindowScores(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qr, qi, kr, ki, H, W, win, use_knorm, eps):
        B, HW, C = qr.shape
        _check(H, W, win, HW)
        BLOCK_C = _next_pow2(C)
        if BLOCK_C > 128:
            raise ValueError(f"C={C} per head must round up to <= 128")
        has_imag = qi is not None
        qi_, ki_ = (qi, ki) if has_imag else (qr, kr)
        knorm = (kr * kr).sum(-1)
        if has_imag:
            knorm = knorm + (ki * ki).sum(-1)
        K = win * win
        S = torch.empty((B, K, HW), dtype=qr.dtype, device=qr.device)
        _launch("_wscores_fwd", HW, B, qr, qi_, kr, ki_, knorm.contiguous(), S,
                qr.stride(0), qr.stride(1), S.stride(0), S.stride(1), HW,
                H, W, HW, win, (win - 1) // 2, eps,
                HAS_IMAG=has_imag, USE_KNORM=use_knorm, C=C, BLOCK_C=BLOCK_C)
        ctx.save_for_backward(qr, qi_, kr, ki_)
        ctx.cfg = (H, W, win, use_knorm, eps, has_imag, C, BLOCK_C)
        return S

    @staticmethod
    def backward(ctx, dS):
        qr, qi_, kr, ki_ = ctx.saved_tensors
        H, W, win, use_knorm, eps, has_imag, C, BLOCK_C = ctx.cfg
        dS = dS.contiguous()
        B, HW, _ = qr.shape
        dqr, dqi = torch.empty_like(qr), torch.empty_like(qr)
        dkr, dki = torch.empty_like(kr), torch.empty_like(kr)
        args = (qr.stride(0), qr.stride(1), dS.stride(0), dS.stride(1),
                H, W, HW, win, (win - 1) // 2, eps)
        _launch("_wscores_bwd_dq", HW, B, qr, qi_, kr, ki_, dS, dqr, dqi, *args,
                HAS_IMAG=has_imag, C=C, BLOCK_C=BLOCK_C)
        _launch("_wscores_bwd_dk", HW, B, qr, qi_, kr, ki_, dS, dkr, dki, *args,
                HAS_IMAG=has_imag, USE_KNORM=use_knorm, C=C, BLOCK_C=BLOCK_C)
        return (dqr, dqi if has_imag else None, dkr, dki if has_imag else None,
                None, None, None, None, None)


def _launch_apply(P, v, H, W, win, transpose):
    B, HW, DV = v.shape
    BLOCK_V = _next_pow2(DV)
    if BLOCK_V > 128:
        raise ValueError(f"DV={DV} per head must round up to <= 128")
    out = torch.empty_like(v)
    _launch("_wapply", HW, B, P, v, out, P.stride(0), P.stride(1), v.stride(0), v.stride(1),
            H, W, HW, win, (win - 1) // 2, TRANSPOSE=transpose, DV=DV, BLOCK_V=BLOCK_V)
    return out


def _launch_dot(x, y, H, W, win):
    B, HW, DV = x.shape
    K = win * win
    dP = torch.empty((B, K, HW), dtype=x.dtype, device=x.device)
    _launch("_wdot", HW, B, x, y, dP, dP.stride(0), dP.stride(1), x.stride(0), x.stride(1),
            H, W, HW, win, (win - 1) // 2, DV=DV, BLOCK_V=_next_pow2(DV))
    return dP


class _WindowApply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, P, v, H, W, win, transpose):
        _check(H, W, win, v.shape[1])
        ctx.save_for_backward(P, v)
        ctx.cfg = (H, W, win, transpose)
        return _launch_apply(P, v, H, W, win, transpose)

    @staticmethod
    def backward(ctx, dout):
        P, v = ctx.saved_tensors
        H, W, win, transpose = ctx.cfg
        dout = dout.contiguous()
        dP = dv = None
        if ctx.needs_input_grad[1]:
            dv = _launch_apply(P, dout, H, W, win, not transpose)
        if ctx.needs_input_grad[0]:
            dP = (_launch_dot(v, dout, H, W, win) if transpose
                  else _launch_dot(dout, v, H, W, win))
        return dP, dv, None, None, None, None


def window_scores(qr, qi, kr, ki, H, W, win, use_knorm=True, eps=SIM_EPS):
    """(B, HW, C) channel-last q / k (imag parts may be None) -> scores S (B, W^2, HW)."""
    if _use_kernel(qr):
        return _WindowScores.apply(qr.contiguous(), None if qi is None else qi.contiguous(),
                                   kr.contiguous(), None if ki is None else ki.contiguous(),
                                   H, W, win, use_knorm, eps)
    return window_scores_ref(qr, qi, kr, ki, H, W, win, use_knorm, eps)


def window_apply(P, v, H, W, win, transpose=False):
    """P (B, W^2, HW) weights, v (B, HW, DV) real -> (B, HW, DV)."""
    if _use_kernel(v):
        return _WindowApply.apply(P.contiguous(), v.contiguous(), H, W, win, transpose)
    return window_apply_ref(P, v, H, W, win, transpose)


# ===========================================================================
#  the adjacency
# ===========================================================================
class WindowAdjacency:
    """`row-sm(CircSim(q, k; W))` with its weights held: a drop-in for `TritonAdjacency`.

    Built once (scores kernel + softmax over the window axis); every `apply` is then one
    weighted window sum. `_lse` is the triton kernel's log-sum-exp, (B*heads, HW), so a
    joint softmax merges it with the other branches unchanged.
    """

    __slots__ = ("P", "win", "sim", "heads", "B", "H", "W", "_lse")

    def __init__(self, q_img, k_img, win, sim="pidistance", heads=1, eps=SIM_EPS):
        if sim not in ("pidistance", "pidot"):
            raise ValueError(f"WindowAdjacency implements 'pidistance' / 'pidot'; got {sim!r}")
        B, C, H, W = q_img.shape
        self.B, self.H, self.W, self.win, self.sim, self.heads = B, H, W, win, sim, heads
        if q_img.is_complex():
            qr, qi = _img_to_seq(q_img.real, heads), _img_to_seq(q_img.imag, heads)
            kr, ki = _img_to_seq(k_img.real, heads), _img_to_seq(k_img.imag, heads)
        else:
            qr, qi, kr, ki = _img_to_seq(q_img, heads), None, _img_to_seq(k_img, heads), None
        S = window_scores(qr, qi, kr, ki, H, W, win, use_knorm=sim == "pidistance", eps=eps)
        self._lse = torch.logsumexp(S, dim=1)                             # (B*h, HW)
        self.P = torch.exp(S - self._lse.unsqueeze(1))                    # (B*h, K, HW)

    @property
    def values(self):
        """The weights in `Circulant`'s (B*h, Q, K) layout (tests / inspection; a copy)."""
        return self.P.transpose(1, 2)

    def apply_seq(self, v, transpose=False):
        """`apply` on values already in the sequence layout, (B * heads, H * W, C / heads) ->
        the same layout: no image <-> sequence copies (the caller fuses them elsewhere)."""
        return window_apply(self.P, v, self.H, self.W, self.win, transpose)

    def apply(self, x, transpose=False):
        if x.is_complex():
            raise TypeError("WindowAdjacency.apply needs a real value tensor")
        v = _img_to_seq(x, self.heads)
        out = window_apply(self.P, v, self.H, self.W, self.win, transpose)
        return _seq_to_img(out, self.B, self.heads, self.H, self.W)

    def matvec(self, x):
        return self.apply(x)

    def rmatvec(self, x):
        return self.apply(x, transpose=True)
