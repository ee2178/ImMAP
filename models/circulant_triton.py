"""
Fused circulant-window attention with the PHASE-INVARIANT similarity, in Triton.

This is the kernel FlexAttention cannot express. A `score_mod` sees one number
per (query, key) pair -- the raw dot product -- so it can carry `distance`
(`q.k - 1/2||k||^2`) and, for REAL features, `pidistance` (`|q.k| - 1/2||k||^2`).
For COMPLEX features the stacked score is only `Re<q,k>`, and

    pidistance(q, k) = -1/2||q||^2 + |<q,k>| - 1/2||k||^2

needs `Im<q,k>` as well: a second bilinear form. A hand-written kernel can carry
both accumulators, which is what this does (and what Julia's
`CirculantAttention.jl` flash path does).

    re_jk = sum_c ( qr_jc kr_kc + qi_jc ki_kc )       = Re<q_j, k_k>
    im_jk = sum_c ( qi_jc kr_kc - qr_jc ki_kc )       = Im<q_j, k_k>
    s_jk  = sqrt(re^2 + im^2 + eps) - 1/2 knorm_k + bias_k
    Gamma = softmax_k(s_jk)   over the W x W circular window around j
    out_j = sum_k Gamma_jk v_k

`-1/2||q_j||^2` is constant across `k` and cancels in the row softmax, so it is
never formed. Passing `qi = ki = None` gives the real-feature case
(`sqrt(re^2) = |q.k|`); `use_knorm=False` gives `pidot`.

Why an offset loop and not a matmul
-----------------------------------
Tiling this as dense `tl.dot` blocks wastes most of the work: a query attends
`W^2 = 81` keys, but any tile big enough to feed a matmul spans ~600, so ~14% of
the scores computed survive masking. Instead each of the `W^2` offsets is one
*reduction over channels* -- `tl.sum(q * k_shifted, 1)` -- which is entirely
useful work. Shifted key tiles overlap almost completely between adjacent
offsets, so they come from L1/L2 rather than DRAM. Nothing of size
`(B, C, Q, K)` is ever allocated, which is the point: the gather backend
materialises 506 MiB per apply at `Mh=64, W=9` on a 160x160 latent, and a
V-cycle holds ~10 GiB of them.

Kernels
-------
`_fwd`            one pass over offsets, streaming (online) softmax, writes out + lse
`_bwd_preprocess` delta_j = <dout_j, out_j> - dlse_j
`_bwd_dq`         per-QUERY accumulation, offsets k = j + o
`_bwd_dkdv`       per-KEY accumulation, offsets j = k - o. The circular window is
                  symmetric, so that is the same offset set negated -- which is
                  what lets this be a second clean pass rather than atomics
                  scattered out of `_bwd_dq`. Gradients are deterministic.

Limitations
-----------
* fp32, CUDA only. The models here are fp32/complex64; a bf16 path would need
  accumulator plumbing that buys nothing yet.
* `C` and `DV` must each fit one block (<= 128). `Mh` is 64-96 in every config.
* `v` must be REAL. Not a restriction in practice: all four `apply_gamma` call
  sites in `models/prox.py` pass real tensors (`_abs2(...)` and `1/xi_a`), and
  `PixelConv` has real weights. `Gamma` is real, so a caller that genuinely
  needs a complex `v` can stack Re/Im on the channel axis itself.
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


# Finite sentinel rather than -inf: a fully masked block would otherwise hit
# (-inf) - (-inf) = NaN in the streaming-softmax rescale.
NEG = -1.0e30


def _jit(fn):
    """`triton.jit` where triton exists, else leave the function alone.

    Keeps this module importable on a machine without triton (the CPU dev box,
    any non-CUDA install) so the rest of the repo still works.
    """
    return triton.jit(fn) if HAVE_TRITON else fn


# ===========================================================================
#  forward
# ===========================================================================
@_jit
def _fwd(
    QR, QI, KR, KI, V, KNORM, BIAS, OUT, LSE,
    stride_qb, stride_qn, stride_qc,
    stride_vb, stride_vn, stride_vc,
    stride_ob, stride_on, stride_oc,
    stride_sb,                      # per-pixel side tensors are (B, HW) contiguous
    H, W, HW, WIN, P, EPS,
    HAS_IMAG: tl.constexpr, USE_KNORM: tl.constexpr, HAS_BIAS: tl.constexpr,
    C: tl.constexpr, DV: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_V: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1).to(tl.int64)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < HW
    safe_m = tl.where(mask_m, offs_m, 0)          # keep derived indices in range
    row = safe_m // W
    col = safe_m % W

    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C
    offs_v = tl.arange(0, BLOCK_V)
    mask_v = offs_v < DV

    qk_base = pid_b * stride_qb
    q_off = qk_base + safe_m[:, None] * stride_qn + offs_c[None, :] * stride_qc
    qmask = mask_m[:, None] & mask_c[None, :]

    qr = tl.load(QR + q_off, mask=qmask, other=0.0)
    qi = tl.load(QI + q_off, mask=qmask, other=0.0) if HAS_IMAG else qr

    m_i = tl.full([BLOCK_M], NEG, tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_V], tl.float32)

    # Runtime loops: W=25 would unroll to 625 bodies and blow up compile time.
    for i in range(WIN):
        dr = i - P
        nr = row + dr
        nr = tl.where(nr < 0, nr + H, nr)
        nr = tl.where(nr >= H, nr - H, nr)
        for j in range(WIN):
            dc = j - P
            nc = col + dc
            nc = tl.where(nc < 0, nc + W, nc)
            nc = tl.where(nc >= W, nc - W, nc)
            nn = nr * W + nc                             # circular wrap

            k_off = qk_base + nn[:, None] * stride_qn + offs_c[None, :] * stride_qc
            kr = tl.load(KR + k_off, mask=qmask, other=0.0)

            re = tl.sum(qr * kr, 1)
            if HAS_IMAG:
                ki = tl.load(KI + k_off, mask=qmask, other=0.0)
                re += tl.sum(qi * ki, 1)
                im = tl.sum(qi * kr, 1) - tl.sum(qr * ki, 1)
                s = tl.sqrt(re * re + im * im + EPS)
            else:
                s = tl.sqrt(re * re + EPS)               # = |q.k|

            if USE_KNORM:
                s = s - 0.5 * tl.load(KNORM + pid_b * stride_sb + nn,
                                      mask=mask_m, other=0.0)
            if HAS_BIAS:
                s = s + tl.load(BIAS + pid_b * stride_sb + nn,
                                mask=mask_m, other=0.0)
            s = tl.where(mask_m, s, NEG)

            m_new = tl.maximum(m_i, s)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new)

            v_off = (pid_b * stride_vb + nn[:, None] * stride_vn
                     + offs_v[None, :] * stride_vc)
            v = tl.load(V + v_off, mask=mask_m[:, None] & mask_v[None, :], other=0.0)

            l_i = l_i * alpha + p
            acc = acc * alpha[:, None] + p[:, None] * v
            m_i = m_new

    out = acc / l_i[:, None]
    o_off = pid_b * stride_ob + safe_m[:, None] * stride_on + offs_v[None, :] * stride_oc
    tl.store(OUT + o_off, out, mask=mask_m[:, None] & mask_v[None, :])
    tl.store(LSE + pid_b * stride_sb + safe_m, m_i + tl.log(l_i), mask=mask_m)


# ===========================================================================
#  backward
# ===========================================================================
@_jit
def _bwd_preprocess(
    DOUT, OUT, DLSE, DELTA,
    stride_ob, stride_on, stride_oc, stride_sb,
    HW, HAS_DLSE: tl.constexpr, DV: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_V: tl.constexpr,
):
    """delta_j = <dout_j, out_j> - dlse_j.

    The softmax backward needs `<dout, out>`. The `-dlse` folds in the gradient
    of the returned log-sum-exp, since `d lse_j / d s_jk` is just `Gamma_jk`.
    Without it the transposed apply -- which feeds the forward's lse back in as
    a per-key bias -- would silently drop a gradient path.
    """
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1).to(tl.int64)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < HW
    safe_m = tl.where(mask_m, offs_m, 0)
    offs_v = tl.arange(0, BLOCK_V)
    mask_v = offs_v < DV

    off = pid_b * stride_ob + safe_m[:, None] * stride_on + offs_v[None, :] * stride_oc
    m2 = mask_m[:, None] & mask_v[None, :]
    do = tl.load(DOUT + off, mask=m2, other=0.0)
    o = tl.load(OUT + off, mask=m2, other=0.0)

    delta = tl.sum(do * o, 1)
    if HAS_DLSE:
        delta -= tl.load(DLSE + pid_b * stride_sb + safe_m, mask=mask_m, other=0.0)
    tl.store(DELTA + pid_b * stride_sb + safe_m, delta, mask=mask_m)


@_jit
def _bwd_dq(
    QR, QI, KR, KI, V, KNORM, BIAS, DOUT, LSE, DELTA, DQR, DQI,
    stride_qb, stride_qn, stride_qc,
    stride_vb, stride_vn, stride_vc,
    stride_ob, stride_on, stride_oc,
    stride_sb,
    H, W, HW, WIN, P, EPS,
    HAS_IMAG: tl.constexpr, USE_KNORM: tl.constexpr, HAS_BIAS: tl.constexpr,
    C: tl.constexpr, DV: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_V: tl.constexpr,
):
    """Per-QUERY accumulation: for query j, walk its neighbours k = j + o."""
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1).to(tl.int64)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < HW
    safe_m = tl.where(mask_m, offs_m, 0)
    row = safe_m // W
    col = safe_m % W

    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C
    offs_v = tl.arange(0, BLOCK_V)
    mask_v = offs_v < DV

    qk_base = pid_b * stride_qb
    q_off = qk_base + safe_m[:, None] * stride_qn + offs_c[None, :] * stride_qc
    qmask = mask_m[:, None] & mask_c[None, :]
    vmask = mask_m[:, None] & mask_v[None, :]

    qr = tl.load(QR + q_off, mask=qmask, other=0.0)
    qi = tl.load(QI + q_off, mask=qmask, other=0.0) if HAS_IMAG else qr

    do_off = pid_b * stride_ob + safe_m[:, None] * stride_on + offs_v[None, :] * stride_oc
    do = tl.load(DOUT + do_off, mask=vmask, other=0.0)

    lse = tl.load(LSE + pid_b * stride_sb + safe_m, mask=mask_m, other=0.0)
    delta = tl.load(DELTA + pid_b * stride_sb + safe_m, mask=mask_m, other=0.0)

    dqr = tl.zeros([BLOCK_M, BLOCK_C], tl.float32)
    dqi = tl.zeros([BLOCK_M, BLOCK_C], tl.float32)

    for i in range(WIN):
        dr = i - P
        nr = row + dr
        nr = tl.where(nr < 0, nr + H, nr)
        nr = tl.where(nr >= H, nr - H, nr)
        for j in range(WIN):
            dc = j - P
            nc = col + dc
            nc = tl.where(nc < 0, nc + W, nc)
            nc = tl.where(nc >= W, nc - W, nc)
            nn = nr * W + nc

            k_off = qk_base + nn[:, None] * stride_qn + offs_c[None, :] * stride_qc
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

            s = A
            if USE_KNORM:
                s = s - 0.5 * tl.load(KNORM + pid_b * stride_sb + nn,
                                      mask=mask_m, other=0.0)
            if HAS_BIAS:
                s = s + tl.load(BIAS + pid_b * stride_sb + nn,
                                mask=mask_m, other=0.0)

            gamma = tl.where(mask_m, tl.exp(s - lse), 0.0)

            v_off = (pid_b * stride_vb + nn[:, None] * stride_vn
                     + offs_v[None, :] * stride_vc)
            v = tl.load(V + v_off, mask=vmask, other=0.0)

            ds = gamma * (tl.sum(do * v, 1) - delta)
            c_re = ds * re / A
            c_im = ds * im / A

            # re = sum_c (qr kr + qi ki);  im = sum_c (qi kr - qr ki)
            dqr += c_re[:, None] * kr - c_im[:, None] * ki
            if HAS_IMAG:
                dqi += c_re[:, None] * ki + c_im[:, None] * kr

    tl.store(DQR + q_off, dqr, mask=qmask)
    if HAS_IMAG:
        tl.store(DQI + q_off, dqi, mask=qmask)


@_jit
def _bwd_dkdv(
    QR, QI, KR, KI, V, KNORM, BIAS, DOUT, LSE, DELTA, DKR, DKI, DV_, DBIAS,
    stride_qb, stride_qn, stride_qc,
    stride_vb, stride_vn, stride_vc,
    stride_ob, stride_on, stride_oc,
    stride_sb,
    H, W, HW, WIN, P, EPS,
    HAS_IMAG: tl.constexpr, USE_KNORM: tl.constexpr, HAS_BIAS: tl.constexpr,
    C: tl.constexpr, DV: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_V: tl.constexpr,
):
    """Per-KEY accumulation: for key k, walk the queries j = k - o that see it."""
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1).to(tl.int64)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)         # KEY indices here
    mask_m = offs_m < HW
    safe_m = tl.where(mask_m, offs_m, 0)
    row = safe_m // W
    col = safe_m % W

    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C
    offs_v = tl.arange(0, BLOCK_V)
    mask_v = offs_v < DV

    qk_base = pid_b * stride_qb
    k_off = qk_base + safe_m[:, None] * stride_qn + offs_c[None, :] * stride_qc
    kmask = mask_m[:, None] & mask_c[None, :]
    vmask = mask_m[:, None] & mask_v[None, :]

    kr = tl.load(KR + k_off, mask=kmask, other=0.0)
    ki = tl.load(KI + k_off, mask=kmask, other=0.0) if HAS_IMAG else kr

    v_self_off = (pid_b * stride_vb + safe_m[:, None] * stride_vn
                  + offs_v[None, :] * stride_vc)
    v_self = tl.load(V + v_self_off, mask=vmask, other=0.0)

    # knorm / bias belong to THIS key block, so they load once
    knorm = tl.load(KNORM + pid_b * stride_sb + safe_m, mask=mask_m,
                    other=0.0) if USE_KNORM else 0.0
    bias = tl.load(BIAS + pid_b * stride_sb + safe_m, mask=mask_m,
                   other=0.0) if HAS_BIAS else 0.0

    dkr = tl.zeros([BLOCK_M, BLOCK_C], tl.float32)
    dki = tl.zeros([BLOCK_M, BLOCK_C], tl.float32)
    dv = tl.zeros([BLOCK_M, BLOCK_V], tl.float32)
    dperkey = tl.zeros([BLOCK_M], tl.float32)

    for i in range(WIN):
        dr = i - P
        jr = row - dr                                # query j = k - o
        jr = tl.where(jr < 0, jr + H, jr)
        jr = tl.where(jr >= H, jr - H, jr)
        for j in range(WIN):
            dc = j - P
            jc = col - dc
            jc = tl.where(jc < 0, jc + W, jc)
            jc = tl.where(jc >= W, jc - W, jc)
            jj = jr * W + jc

            q_off = qk_base + jj[:, None] * stride_qn + offs_c[None, :] * stride_qc
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

            s = A
            if USE_KNORM:
                s = s - 0.5 * knorm
            if HAS_BIAS:
                s = s + bias

            lse = tl.load(LSE + pid_b * stride_sb + jj, mask=mask_m, other=0.0)
            delta = tl.load(DELTA + pid_b * stride_sb + jj, mask=mask_m, other=0.0)
            gamma = tl.where(mask_m, tl.exp(s - lse), 0.0)

            do_off = (pid_b * stride_ob + jj[:, None] * stride_on
                      + offs_v[None, :] * stride_oc)
            do = tl.load(DOUT + do_off, mask=vmask, other=0.0)

            dv += gamma[:, None] * do

            ds = gamma * (tl.sum(do * v_self, 1) - delta)
            dperkey += ds

            c_re = ds * re / A
            c_im = ds * im / A

            # d re/d kr = qr,  d re/d ki = qi;  d im/d kr = qi,  d im/d ki = -qr
            dkr += c_re[:, None] * qr + c_im[:, None] * qi
            if HAS_IMAG:
                dki += c_re[:, None] * qi - c_im[:, None] * qr

    # knorm_k = sum_c (kr^2 + ki^2) enters as -1/2 knorm_k, so
    #   d/d kr += (-1/2 dperkey) * 2 kr = -dperkey * kr
    if USE_KNORM:
        dkr -= dperkey[:, None] * kr
        if HAS_IMAG:
            dki -= dperkey[:, None] * ki

    tl.store(DKR + k_off, dkr, mask=kmask)
    if HAS_IMAG:
        tl.store(DKI + k_off, dki, mask=kmask)
    tl.store(DV_ + v_self_off, dv, mask=vmask)
    if HAS_BIAS:
        tl.store(DBIAS + pid_b * stride_sb + safe_m, dperkey, mask=mask_m)


# ===========================================================================
#  autograd wrapper
# ===========================================================================
def _next_pow2(n):
    return 1 << (int(n) - 1).bit_length()


class _CircPIAttention(torch.autograd.Function):
    """`out, lse = softmax_window(pidistance(q, k)) @ v`, fused.

    Tensors are (B, HW, C) contiguous -- channel-last, so the reduction axis is
    the fast one. `qi` / `ki` may be None (real features).
    """

    @staticmethod
    def forward(ctx, qr, qi, kr, ki, v, bias, H, W, win, use_knorm, eps,
                block_m, num_warps, num_stages):
        if not HAVE_TRITON:
            raise RuntimeError(
                "the triton attention backend needs triton, which ships with "
                "PyTorch's Linux CUDA wheels. Use attn_backend='gather' or "
                "'flex' on this machine.")
        if not qr.is_cuda:
            raise RuntimeError("the triton backend is CUDA-only")
        if qr.dtype != torch.float32 or v.dtype != torch.float32:
            raise TypeError(f"fp32 only; got q={qr.dtype}, v={v.dtype}")

        B, HW, C = qr.shape
        DV = v.shape[-1]
        if HW != H * W:
            raise ValueError(f"HW={HW} does not match {H}x{W}")
        if win % 2 == 0:
            raise ValueError(f"window side must be odd; got {win}")
        if win > min(H, W):
            raise ValueError(f"window {win} exceeds the {H}x{W} grid")

        BLOCK_C, BLOCK_V = _next_pow2(C), _next_pow2(DV)
        if BLOCK_C > 128 or BLOCK_V > 128:
            raise ValueError(
                f"C={C}, DV={DV}: this kernel holds one channel block in "
                f"registers, so both must round up to <= 128")

        has_imag = qi is not None
        has_bias = bias is not None
        # triton still needs a real pointer for a compiled-out branch
        qi_ = qi if has_imag else qr
        ki_ = ki if has_imag else kr
        bias_ = bias if has_bias else qr

        knorm = (kr * kr).sum(-1)
        if has_imag:
            knorm = knorm + (ki * ki).sum(-1)
        knorm = knorm.contiguous()

        out = torch.empty((B, HW, DV), dtype=qr.dtype, device=qr.device)
        lse = torch.empty((B, HW), dtype=torch.float32, device=qr.device)

        grid = (triton.cdiv(HW, block_m), B)
        _fwd[grid](
            qr, qi_, kr, ki_, v, knorm, bias_, out, lse,
            qr.stride(0), qr.stride(1), qr.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            HW,
            H, W, HW, win, (win - 1) // 2, eps,
            HAS_IMAG=has_imag, USE_KNORM=use_knorm, HAS_BIAS=has_bias,
            C=C, DV=DV,
            BLOCK_M=block_m, BLOCK_C=BLOCK_C, BLOCK_V=BLOCK_V,
            num_warps=num_warps, num_stages=num_stages,
        )

        ctx.save_for_backward(qr, qi_, kr, ki_, v, knorm, bias_, out, lse)
        ctx.cfg = (H, W, win, use_knorm, eps, has_imag, has_bias,
                   block_m, num_warps, num_stages, C, DV, BLOCK_C, BLOCK_V)
        return out, lse

    @staticmethod
    def backward(ctx, dout, dlse):
        qr, qi_, kr, ki_, v, knorm, bias_, out, lse = ctx.saved_tensors
        (H, W, win, use_knorm, eps, has_imag, has_bias,
         block_m, num_warps, num_stages, C, DV, BLOCK_C, BLOCK_V) = ctx.cfg

        B, HW, _ = qr.shape
        dout = dout.contiguous()
        # No `.any()` here -- that would sync the device every backward. An
        # unused lse arrives as zeros, and the extra loads cost nothing.
        has_dlse = dlse is not None
        dlse_ = dlse.contiguous() if has_dlse else lse

        delta = torch.empty((B, HW), dtype=torch.float32, device=qr.device)
        grid = (triton.cdiv(HW, block_m), B)

        _bwd_preprocess[grid](
            dout, out, dlse_, delta,
            out.stride(0), out.stride(1), out.stride(2), HW,
            HW, HAS_DLSE=has_dlse, DV=DV,
            BLOCK_M=block_m, BLOCK_V=BLOCK_V, num_warps=num_warps,
        )

        dqr = torch.zeros_like(qr)
        dqi = torch.zeros_like(qr)
        dkr = torch.zeros_like(kr)
        dki = torch.zeros_like(kr)
        dv = torch.zeros_like(v)
        dbias = torch.zeros((B, HW), dtype=torch.float32, device=qr.device)

        strides = (
            qr.stride(0), qr.stride(1), qr.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            HW,
        )
        geom = (H, W, HW, win, (win - 1) // 2, eps)
        common = dict(
            HAS_IMAG=has_imag, USE_KNORM=use_knorm, HAS_BIAS=has_bias,
            C=C, DV=DV, BLOCK_M=block_m, BLOCK_C=BLOCK_C, BLOCK_V=BLOCK_V,
            num_warps=num_warps, num_stages=num_stages,
        )

        _bwd_dq[grid](qr, qi_, kr, ki_, v, knorm, bias_, dout, lse, delta,
                      dqr, dqi, *strides, *geom, **common)
        _bwd_dkdv[grid](qr, qi_, kr, ki_, v, knorm, bias_, dout, lse, delta,
                        dkr, dki, dv, dbias, *strides, *geom, **common)

        return (dqr,
                dqi if has_imag else None,
                dkr,
                dki if has_imag else None,
                dv,
                dbias if has_bias else None,
                None, None, None, None, None, None, None, None)


def circulant_pi_attention(qr, kr, v, H, W, win, qi=None, ki=None, bias=None,
                           use_knorm=True, eps=1e-8, block_m=64, num_warps=4,
                           num_stages=2):
    """Fused windowed attention with the phase-invariant similarity.

    qr, kr : (B, HW, C)   real parts (or the whole tensor, for real features)
    qi, ki : (B, HW, C) or None
    v      : (B, HW, DV)  real
    bias   : (B, HW) per-KEY additive term, or None
    use_knorm : include `-1/2 ||k||^2` (pidistance); False gives pidot

    Returns (out, lse): out (B, HW, DV), lse (B, HW).
    """
    return _CircPIAttention.apply(
        qr.contiguous(), None if qi is None else qi.contiguous(),
        kr.contiguous(), None if ki is None else ki.contiguous(),
        v.contiguous(), None if bias is None else bias.contiguous(),
        H, W, win, use_knorm, eps, block_m, num_warps, num_stages)


# ===========================================================================
#  Circulant-compatible adjacency
# ===========================================================================
def _img_to_seq(x, heads):
    """(B, C, H, W) -> (B*heads, H*W, C//heads), channel-last and contiguous."""
    B, C, H, W = x.shape
    if C % heads:
        raise ValueError(f"channels {C} must divide heads {heads}")
    return x.reshape(B * heads, C // heads, H, W).permute(0, 2, 3, 1).contiguous()


def _seq_to_img(s, B, heads, H, W):
    """(B*heads, H*W, D) -> (B, heads*D, H, W)."""
    D = s.shape[-1]
    return (s.reshape(B * heads, H, W, D).permute(0, 3, 1, 2)
             .reshape(B, heads * D, H, W))


class TritonAdjacency:
    """`row-sm(CircSim(q, k; W))` held implicitly, with a transposed apply.

    Drop-in for `circulant_attention.Circulant` / `circulant_flex.FlexAdjacency`
    as far as `.apply(x, transpose=)` is concerned. Unlike `FlexAdjacency` this
    carries COMPLEX queries and keys and the true `pidistance`.

    The transpose reuses the FORWARD kernel. Writing the score as `Stilde_jk`
    and `Z_j = sum_k exp(Stilde_jk)`,

        (Gamma^T h)_k = sum_j exp(Stilde_jk - log Z_j) h_j

    is a windowed attention with q and k swapped and a per-KEY bias `-log Z_j`;
    its own softmax denominator is divided back out with the returned lse. That
    swap is only legal because the similarity is symmetric, and it is:
    `|<k,q>| = |conj(<q,k>)| = |<q,k>|`. The `-1/2||k||^2` term becomes a
    per-QUERY constant, absorbed by the `colsum` factor -- the same bookkeeping
    `FlexAdjacency` already does.
    """

    __slots__ = ("qr", "qi", "kr", "ki", "ksq", "win", "sim", "heads",
                 "H", "W", "B", "eps", "block_m", "num_warps", "num_stages", "_lse")

    def __init__(self, q_img, k_img, win, sim="pidistance", heads=1, eps=1e-8,
                 block_m=64, num_warps=4, num_stages=2):
        if sim not in ("pidistance", "pidot"):
            raise ValueError(
                f"TritonAdjacency implements the phase-invariant similarities "
                f"('pidistance', 'pidot'); got {sim!r}. distance / dot / "
                f"realdot are cheaper on the flex backend.")
        B, C, H, W = q_img.shape
        self.B, self.H, self.W = B, H, W
        self.win, self.sim, self.heads = win, sim, heads
        self.eps = eps
        self.block_m, self.num_warps = block_m, num_warps
        self.num_stages = num_stages

        if q_img.is_complex():
            self.qr = _img_to_seq(q_img.real, heads)
            self.qi = _img_to_seq(q_img.imag, heads)
            self.kr = _img_to_seq(k_img.real, heads)
            self.ki = _img_to_seq(k_img.imag, heads)
        else:
            self.qr, self.qi = _img_to_seq(q_img, heads), None
            self.kr, self.ki = _img_to_seq(k_img, heads), None

        ksq = (self.kr * self.kr).sum(-1)
        if self.ki is not None:
            ksq = ksq + (self.ki * self.ki).sum(-1)
        self.ksq = ksq
        self._lse = None

    @property
    def _use_knorm(self):
        return self.sim == "pidistance"

    def _run(self, qr, qi, kr, ki, v, bias, use_knorm):
        return circulant_pi_attention(
            qr, kr, v, self.H, self.W, self.win, qi=qi, ki=ki, bias=bias,
            use_knorm=use_knorm, eps=self.eps, block_m=self.block_m,
            num_warps=self.num_warps, num_stages=self.num_stages)

    def apply(self, x, transpose=False):
        """`Gamma x` (or `Gamma^T x`) applied channel-wise; x is (B, C, H, W).

        `x` must be REAL -- see the module docstring. Every `apply_gamma` call
        site in models/prox.py already satisfies that.
        """
        if x.is_complex():
            raise TypeError(
                "TritonAdjacency.apply needs a real value tensor; Gamma is "
                "real, so stack Re/Im on the channel axis and call once.")

        v = _img_to_seq(x, self.heads)

        if not transpose:
            out, self._lse = self._run(self.qr, self.qi, self.kr, self.ki, v,
                                       None, self._use_knorm)
        else:
            if self._lse is None:                        # forward not seen yet
                _, self._lse = self._run(self.qr, self.qi, self.kr, self.ki, v,
                                         None, self._use_knorm)
            # swap q <-> k, per-key bias -log Z_j, and NO knorm: that term is
            # per-query in this direction and comes back through colsum.
            out, lse_t = self._run(self.kr, self.ki, self.qr, self.qi, v,
                                   -self._lse, use_knorm=False)
            qbias = -0.5 * self.ksq if self._use_knorm else 0.0
            colsum = torch.exp(lse_t + qbias)            # = sum_j Gamma_jk, O(1)
            out = out * colsum.unsqueeze(-1)

        return _seq_to_img(out, self.B, self.heads, self.H, self.W)

    def matvec(self, x):
        return self.apply(x, transpose=False)

    def rmatvec(self, x):
        return self.apply(x, transpose=True)
