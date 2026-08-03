"""
FlexAttention implementation of circulant-sparse (sliding-window) attention for
images, the fused analogue of circulant_similarity.py / circulant_attention.py.

Idea
----
GroupCDL-style circulant attention restricts each query pixel to a W x W window
of keys (circular boundary => BCCB sparsity). FlexAttention expresses exactly
this: the window is a `BlockMask` (so empty key-blocks are skipped, never
materialised), and the similarity choice is a `score_mod`:

  * dot / realdot  : S_ij = q_i . k_j                      -> score_mod = identity
  * distance       : S_ij = -1/2 ||q_i - k_j||^2 .
       Inside row-softmax the -1/2||q_i||^2 term is constant in j and cancels, so
         softmax_j(-1/2||q_i-k_j||^2) = softmax_j(q_i.k_j - 1/2||k_j||^2),
       i.e. ordinary QK attention with a per-key bias -1/2||k_j||^2. That bias is
       a captured tensor indexed by kv_idx in the score_mod. scale = 1 (no 1/sqrt d).

FlexAttention returns the *fused* softmax(S)·V, i.e. row-sm(CircSim)·V in one
kernel -- it never forms the (B, N, K) window tensor. This is the big win on
large images. (Trade-off: you get the applied output, not the explicit adjacency
matrix; see notes on the GroupCDL convex blend at the bottom.)

Layout: images are (B, C, H, W). We flatten spatial row-major to a sequence of
length S = H*W (seq index s = r*W + c) and run heads over the channel dim.
Real-valued only (FlexAttention has no complex support) -> denoising / real
features; complex CS-MRI must use the gather/sparse path instead.
"""

from __future__ import annotations
import functools
import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask


# ---------------------------------------------------------------------------
# image <-> (B, heads, S, D) sequence layout
# ---------------------------------------------------------------------------
def image_to_seq(x, heads=1):
    """(B, C, H, W) -> (B, heads, H*W, C//heads).  C must be divisible by heads."""
    B, C, H, W = x.shape
    assert C % heads == 0, "channels must divide heads"
    s = x.flatten(2).transpose(1, 2)                      # (B, S=H*W, C), s = r*W + c
    return s.view(B, H * W, heads, C // heads).transpose(1, 2).contiguous()

def seq_to_image(o, H, W, heads=1):
    """(B, heads, H*W, Dv) -> (B, heads*Dv, H, W)."""
    B, _, S, Dv = o.shape
    o = o.transpose(1, 2).reshape(B, S, heads * Dv).transpose(1, 2)   # (B, C, S)
    return o.view(B, heads * Dv, H, W)


# ---------------------------------------------------------------------------
# circulant (circular sliding-window) BlockMask over a flattened 2D image
# ---------------------------------------------------------------------------
def circular_window_mask_mod(H, W, win):
    """mask_mod: query s=(qr,qc) attends key t=(kr,kc) iff the *circular*
    Chebyshev distance <= p = (win-1)//2 in both axes (the BCCB pattern)."""
    p = (win - 1) // 2

    def mask_mod(b, h, q_idx, kv_idx):
        qr, qc = q_idx // W, q_idx % W
        kr, kc = kv_idx // W, kv_idx % W
        dr = (qr - kr).abs(); dr = torch.minimum(dr, H - dr)   # circular row dist
        dc = (qc - kc).abs(); dc = torch.minimum(dc, W - dc)   # circular col dist
        return (dr <= p) & (dc <= p)

    return mask_mod


def window_mask_mod(H, W, win):
    """Non-circular (clamped-boundary) sliding window, common in plain image
    local-attention. Same Chebyshev test without the wraparound minimum."""
    p = (win - 1) // 2

    def mask_mod(b, h, q_idx, kv_idx):
        qr, qc = q_idx // W, q_idx % W
        kr, kc = kv_idx // W, kv_idx % W
        return ((qr - kr).abs() <= p) & ((qc - kc).abs() <= p)

    return mask_mod


def build_block_mask(H, W, win, device, circular=True, BLOCK_SIZE=128, compile=True):
    """Precompute the BlockMask once per (H, W, win). Reuse across layers/steps."""
    mm = (circular_window_mask_mod if circular else window_mask_mod)(H, W, win)
    return create_block_mask(mm, B=None, H=None, Q_LEN=H * W, KV_LEN=H * W,
                             device=device, BLOCK_SIZE=BLOCK_SIZE, _compile=compile)


# The mask depends only on (grid, window, device) -- never on weights or data --
# so it is shared PROCESS-WIDE rather than per module.  An unrolled network has
# one prox per layer per level and they all want the same mask; building it is
# expensive (create_block_mask torch.compiles itself), and the object is small
# (block-level index tensors), so memoising is a clear win.
_BLOCK_MASK_CACHE = {}


def get_block_mask(H, W, win, device, circular=True, BLOCK_SIZE=128, compile=None):
    """Memoised `build_block_mask`. Pass `compile=None` to auto-enable on CUDA."""
    if compile is None:
        compile = torch.cuda.is_available()
    BLOCK_SIZE = min(BLOCK_SIZE, H * W)
    key = (H, W, win, bool(circular), BLOCK_SIZE, str(device), bool(compile))
    if key not in _BLOCK_MASK_CACHE:
        _BLOCK_MASK_CACHE[key] = build_block_mask(
            H, W, win, device, circular=circular,
            BLOCK_SIZE=BLOCK_SIZE, compile=compile)
    return _BLOCK_MASK_CACHE[key]


# ---------------------------------------------------------------------------
# score_mod for distance similarity (per-key bias -1/2 ||k_j||^2)
# ---------------------------------------------------------------------------
def _distance_score_mod(k_sqnorm):
    """k_sqnorm : (B, heads, S) = sum over head_dim of k^2."""
    def score_mod(score, b, h, q_idx, kv_idx):
        return score - 0.5 * k_sqnorm[b, h, kv_idx]
    return score_mod


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------
def circulant_flex_attention(q, k, v, H, W, win, sim="distance",
                             block_mask=None, circular=True, compiled=None):
    """
    Fused circulant attention  out = row-sm(CircSim(q, k; W)) @ v.

    q, k, v : (B, heads, S, D) with S = H*W (see image_to_seq). v's head_dim may
              differ from q/k's. Real dtype only.
    sim     : 'distance' | 'dot' | 'realdot'.
    block_mask : reuse a precomputed mask (recommended); else built here.
    returns : (B, heads, S, Dv).
    """
    if block_mask is None:
        block_mask = build_block_mask(H, W, win, q.device, circular=circular)

    if sim == "distance":
        score_mod = _distance_score_mod((k * k).sum(dim=-1))
    elif sim in ("dot", "realdot"):
        score_mod = None
    else:
        raise ValueError(f"unsupported sim {sim!r} for FlexAttention (real-valued)")

    fa = compiled if compiled is not None else flex_attention
    return fa(q, k, v, score_mod=score_mod, block_mask=block_mask, scale=1.0)


def circulant_flex_attention_image(xq, xk, xv, win, sim="distance",
                                   heads=1, block_mask=None, circular=True, compiled=None):
    """Convenience wrapper taking image tensors (B, C, H, W) directly."""
    B, C, H, W = xq.shape
    q = image_to_seq(xq, heads); k = image_to_seq(xk, heads); v = image_to_seq(xv, heads)
    o = circulant_flex_attention(q, k, v, H, W, win, sim=sim,
                                 block_mask=block_mask, circular=circular, compiled=compiled)
    return seq_to_image(o, H, W, heads)


# ---------------------------------------------------------------------------
# FlexAdjacency: a Circulant-compatible adjacency that is never materialised
# ---------------------------------------------------------------------------
def _flex_with_lse(q, k, v, score_mod, block_mask, compiled=None):
    """Call flex_attention and also return the row log-sum-exp.

    torch renamed `return_lse` to `return_aux=AuxRequest(lse=True)`; try the new
    spelling first and fall back so this works across versions.
    """
    fa = compiled if compiled is not None else flex_attention
    try:
        from torch.nn.attention.flex_attention import AuxRequest
        out, aux = fa(q, k, v, score_mod=score_mod, block_mask=block_mask,
                      scale=1.0, return_aux=AuxRequest(lse=True))
        return out, aux.lse
    except (ImportError, TypeError):
        return fa(q, k, v, score_mod=score_mod, block_mask=block_mask,
                  scale=1.0, return_lse=True)


def _per_key_bias(bias):
    """score_mod adding a per-KEY term, `bias : (B, heads, S)`."""
    def score_mod(score, b, h, q_idx, kv_idx):
        return score + bias[b, h, kv_idx]
    return score_mod


class FlexAdjacency:
    """`row-sm(CircSim(q, k; W))` held implicitly, with a transposed apply.

    Drop-in for `circulant_attention.Circulant` as far as `.apply(x, transpose)`
    is concerned, but nothing of size (B, Q, K) is ever allocated -- which is
    what makes a W=35 window tractable on a full-resolution latent.

    The transpose is the part that is not obvious.  Writing the (masked, biased)
    score as `Stilde_jk` and `Z_j = sum_k exp(Stilde_jk)`,

        (Gamma^T h)_k = sum_j exp(Stilde_jk - log Z_j) h_j

    is itself a windowed attention with the roles of q and k swapped and a
    PER-KEY bias of `-log Z_j` -- which is exactly what a `score_mod` indexed by
    `kv_idx` expresses.  Its own softmax denominator is then divided back out
    using the returned lse:

        (Gamma^T h)_k = colsum_k * flex(q<-k, k<-q, v=h ; -log Z)_k,
        colsum_k      = exp(lse'_k + b_k) = sum_j Gamma_jk

    and `colsum` is a column sum of a row-stochastic matrix, hence O(1) -- no
    large exponentials anywhere.  The window relation is symmetric (circular
    Chebyshev distance), so the SAME BlockMask serves both directions.

    Validated against the dense gather path in tests/test_multigrid.py.
    """

    __slots__ = ("q", "k", "ksq", "win", "sim", "heads", "H", "W",
                 "block_mask", "compiled", "_lse")

    def __init__(self, q_img, k_img, win, sim="distance", heads=1,
                 block_mask=None, circular=True, compiled=None,
                 block_size=128, compile_mask=None):
        if sim not in ("distance", "dot", "realdot"):
            raise ValueError(f"unsupported sim {sim!r} for FlexAttention")
        B, C, H, W = q_img.shape
        self.H, self.W, self.win, self.sim, self.heads = H, W, win, sim, heads
        self.q = image_to_seq(q_img, heads)
        self.k = image_to_seq(k_img, heads)
        self.ksq = (self.k * self.k).sum(dim=-1)          # (B, heads, S)
        if block_mask is None:
            block_mask = get_block_mask(H, W, win, q_img.device, circular=circular,
                                        BLOCK_SIZE=block_size, compile=compile_mask)
        self.block_mask = block_mask
        self.compiled = compiled
        self._lse = None

    # -- score_mods ---------------------------------------------------------
    def _fwd_mod(self):
        # distance: softmax_k(-1/2||q_j - k_k||^2) == softmax_k(q.k - 1/2||k||^2)
        return _distance_score_mod(self.ksq) if self.sim == "distance" else None

    def _query_bias(self):
        # the -1/2||k||^2 term becomes a per-QUERY constant in the transpose
        return -0.5 * self.ksq if self.sim == "distance" else 0.0

    # -- application --------------------------------------------------------
    def apply(self, x, transpose=False):
        """`Gamma x` (or `Gamma^T x`) applied channel-wise; x is (B, C, H, W)."""
        v = image_to_seq(x, self.heads)
        if not transpose:
            out, lse = _flex_with_lse(self.q, self.k, v, self._fwd_mod(),
                                      self.block_mask, self.compiled)
            self._lse = lse                               # reused by the transpose
            return seq_to_image(out, self.H, self.W, self.heads)

        if self._lse is None:                             # forward not seen yet
            _, self._lse = _flex_with_lse(
                self.q, self.k, v, self._fwd_mod(), self.block_mask, self.compiled)
        out, lseT = _flex_with_lse(self.k, self.q, v, _per_key_bias(-self._lse),
                                   self.block_mask, self.compiled)
        colsum = torch.exp(lseT + self._query_bias())     # = sum_j Gamma_jk, O(1)
        return seq_to_image(out * colsum.unsqueeze(-1), self.H, self.W, self.heads)

    def matvec(self, x):
        return self.apply(x, transpose=False)

    def rmatvec(self, x):
        return self.apply(x, transpose=True)
