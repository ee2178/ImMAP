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
  * pidot          : S_ij = |<q_i, k_j>|                   -> score_mod = |score|
  * pidistance     : S_ij = -1/2||q_i||^2 + |<q_i,k_j>| - 1/2||k_j||^2
       -> |score| plus the same per-key bias as distance. For REAL features
       <q,k> = q.k, so |<q,k>| is |score| and this is exact. See SIM_ABS below
       for why the complex case is not expressible.

FlexAttention returns the *fused* softmax(S)·V, i.e. row-sm(CircSim)·V in one
kernel -- it never forms the (B, N, K) window tensor. This is the big win on
large images. (Trade-off: you get the applied output, not the explicit adjacency
matrix; see notes on the GroupCDL convex blend at the bottom.)

Layout: images are (B, C, H, W). We flatten spatial row-major to a sequence of
length S = H*W (seq index s = r*W + c) and run heads over the channel dim.
Real-valued only (FlexAttention has no complex support). Complex features are
stacked as [Re; Im] by the caller, which leaves dot/realdot/distance unchanged
because q.k over the stacked vectors IS Re<q,k>; the phase-invariant pi- family
is exact for genuinely real features only.
"""

from __future__ import annotations
import functools
import os
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
# score_mods
# ---------------------------------------------------------------------------
# A score_mod sees ONE number per (q, k) pair -- the raw dot product -- plus
# anything indexable by b / h / q_idx / kv_idx. So a similarity is fusible iff
# it is a pointwise function of `q . k` plus per-query and per-key terms:
#
#   dot / realdot : q . k                                      identity
#   distance      : -1/2||q-k||^2 = q.k - 1/2||k||^2 (+ const) per-key bias
#   pidot         : |<q,k>|                                    |score|
#   pidistance    : -1/2||q||^2 + |<q,k>| - 1/2||k||^2         |score| + per-key bias
#
# The -1/2||q||^2 term is constant across k, so it cancels inside the row
# softmax and never has to be formed.
#
# REAL FEATURES ONLY for the pi- family: with real q, k the inner product is
# real, so |<q,k>| is |score| and the fusion is exact. With COMPLEX features
# (stacked here as [Re; Im]) the raw score is Re<q,k>, and recovering
# |<q,k>| = sqrt(Re^2 + Im^2) needs a SECOND bilinear form, Im<q,k> = [b;-a].[c;d].
# One flex_attention call produces one score, and score_mods cannot see across
# heads or calls, so complex pidistance is not expressible here -- it needs a
# bespoke kernel that accumulates both parts, which is what the Julia flash
# path is. `GroupThreshold` rejects that combination up front.
SIM_ABS = ("pidot", "pidistance")            # need |score|
SIM_KNORM = ("distance", "pidistance")       # need the -1/2||k||^2 per-key bias
FLEX_SIMS = ("distance", "dot", "realdot", "pidot", "pidistance")


def _score_mod(sim, k_sqnorm=None, bias=None):
    """Build the score_mod for `sim`, optionally with an extra per-key `bias`.

    `k_sqnorm` / `bias` : (B, heads, S), indexed by kv_idx.
    `bias` is what the transposed apply adds (-log Z_j); it composes with the
    similarity's own per-key term rather than replacing it.
    """
    use_abs = sim in SIM_ABS
    knorm = k_sqnorm if sim in SIM_KNORM else None

    if knorm is None and bias is None and not use_abs:
        return None                                   # plain QK attention

    if knorm is not None and bias is not None:
        total = bias - 0.5 * knorm
    elif knorm is not None:
        total = -0.5 * knorm
    else:
        total = bias

    if total is None:
        def score_mod(score, b, h, q_idx, kv_idx):
            return torch.abs(score)
    elif use_abs:
        def score_mod(score, b, h, q_idx, kv_idx):
            return torch.abs(score) + total[b, h, kv_idx]
    else:
        def score_mod(score, b, h, q_idx, kv_idx):
            return score + total[b, h, kv_idx]

    return score_mod


def _distance_score_mod(k_sqnorm):
    """`distance` similarity: per-key bias -1/2 ||k_j||^2.

    k_sqnorm : (B, heads, S) = sum over head_dim of k^2.
    """
    return _score_mod("distance", k_sqnorm=k_sqnorm)


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

    if sim not in FLEX_SIMS:
        raise ValueError(
            f"unsupported sim {sim!r} for FlexAttention; expected one of "
            f"{FLEX_SIMS} (all real-valued)")

    score_mod = _score_mod(sim, k_sqnorm=(k * k).sum(dim=-1))

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
def _kernel_options(block_mask):
    """Tile sizes for a sparse block smaller than torch's default, else None.

    The compiled kernel computes one dense tile per non-empty (query block, key
    block) pair, and requires the SPARSE block size to be divisible by its
    internal tiles (`BLOCK_M`, `BLOCK_N`): "Q and KV block size must be
    divisible by BLOCK_M and BLOCK_N". torch picks those per GPU -- (128, 32)
    on A100, (32, 64) on H100 for fp32 at this head size -- so a sparse block
    of 32 or 16 fails to compile on some cards unless the tiles are given.

    Why a smaller sparse block is worth having: a 2-D window laid out row-major
    touches `win` separate row-bands of keys, so at BLOCK_SIZE=128 a query on a
    320x184 latent has ~3000 scores computed for the 225 its window uses. At 32
    that is ~1070, at 16 ~600 (tests/test_flex_block_size.py counts them). Same
    mask, same map -- only less masked work.

    Smaller tiles also need less per-block SHARED memory, and that is a hard
    limit: on an L40S (99 KB) the default tiles at this head size do not
    compile at all -- "out of resource: triton_tem_fused_flex_attention_0
    Required: 114688 Hardware limit: 101376" -- while B=16 does. That error,
    not the card's 48 GB, is why the group net "did not fit" an L40S.

    At the default 128 this returns None and the call is exactly what it was.
    """
    bs = getattr(block_mask, "BLOCK_SIZE", None)
    if bs is None:
        return None
    b = int(min(bs)) if isinstance(bs, (tuple, list)) else int(bs)
    if b >= 128 or b & (b - 1):          # default, or not a power of two
        return None
    return {"BLOCK_M": min(b, 64), "BLOCK_N": min(b, 32)}


def _flex_with_lse(q, k, v, score_mod, block_mask, compiled=None):
    """Call flex_attention and also return the row log-sum-exp.

    torch renamed `return_lse` to `return_aux=AuxRequest(lse=True)`; try the new
    spelling first and fall back so this works across versions.
    """
    fa = compiled if compiled is not None else flex_attention
    kw = {}
    ko = _kernel_options(block_mask)
    if ko is not None:
        kw["kernel_options"] = ko
    try:
        from torch.nn.attention.flex_attention import AuxRequest
        out, aux = fa(q, k, v, score_mod=score_mod, block_mask=block_mask,
                      scale=1.0, return_aux=AuxRequest(lse=True), **kw)
        return out, aux.lse
    except (ImportError, TypeError):
        return fa(q, k, v, score_mod=score_mod, block_mask=block_mask,
                  scale=1.0, return_lse=True, **kw)


def _per_key_bias(bias):
    """score_mod adding a per-KEY term, `bias : (B, heads, S)`."""
    def score_mod(score, b, h, q_idx, kv_idx):
        return score + bias[b, h, kv_idx]
    return score_mod


class PointwiseAdjacency:
    """The adjacency of a window of ONE neighbour (each pixel sees only itself), in closed form.

    A softmax over a single score is 1, so Gamma = I: `apply` returns its argument, in either
    direction. What a joint softmax still needs from this branch is its log-sum-exp, which for
    one neighbour is the score itself, in FlexAdjacency's convention (`_fwd_mod`):

        distance     q.k - 1/2 ||k||^2          dot / realdot   q.k
        pidistance  |q.k| - 1/2 ||k||^2         pidot          |q.k|

    at the same pixel -- two reductions over the head dimension. The guided nets run
    `window: 1` beside a 15 x 15 guide window, and a FlexAdjacency there launches the whole
    attention kernel (forward and backward, every layer) to compute exactly this.
    IMMAP_POINTWISE_SELF=0 restores the kernel."""

    __slots__ = ("H", "W", "win", "sim", "heads", "_lse")

    def __init__(self, q_img, k_img, sim="distance", heads=1):
        if sim not in FLEX_SIMS:
            raise ValueError(f"unsupported sim {sim!r}; expected one of {FLEX_SIMS}")
        if q_img.is_complex() or k_img.is_complex():
            raise ValueError("PointwiseAdjacency needs real query/key tensors; stack complex "
                             "features as [Re; Im] first (GroupThreshold._stack_ri)")
        B, C, H, W = q_img.shape
        self.H, self.W, self.win, self.sim, self.heads = H, W, 1, sim, heads
        d = C // heads                                    # heads are contiguous channel runs
        score = (q_img * k_img).reshape(B, heads, d, H * W).sum(dim=2)        # (B, heads, S)
        if sim in SIM_ABS:
            score = score.abs()
        if sim in SIM_KNORM:
            score = score - 0.5 * (k_img * k_img).reshape(B, heads, d, H * W).sum(dim=2)
        self._lse = score

    def apply(self, x, transpose=False):
        return x

    def apply_seq(self, v):
        return v

    def matvec(self, x):
        return x

    def rmatvec(self, x):
        return x


POINTWISE_SELF = os.environ.get("IMMAP_POINTWISE_SELF", "1") != "0"


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
        if sim not in FLEX_SIMS:
            raise ValueError(
                f"unsupported sim {sim!r} for FlexAttention; expected one of "
                f"{FLEX_SIMS}")
        if q_img.is_complex() or k_img.is_complex():
            raise ValueError(
                "FlexAdjacency needs real query/key tensors; stack complex "
                "features as [Re; Im] first (GroupThreshold._stack_ri)")
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
        # distance:   softmax_k(-1/2||q_j-k_k||^2) == softmax_k(q.k - 1/2||k||^2)
        # pidistance: the same with |q.k| in place of q.k
        return _score_mod(self.sim, k_sqnorm=self.ksq)

    def _rev_mod(self):
        """Transposed apply: the SAME nonlinearity, per-key bias -log Z_j.

        The transposed call swaps the roles of q and k, so its raw score at
        (row k, col j) is still `q_j . k_k` -- which is why `|.|` has to be
        applied here too, and why forgetting it would silently give the
        transpose of a DIFFERENT matrix.  The similarity's own -1/2||k_k||^2
        term is now indexed by the query, so it moves to `_query_bias`.
        """
        use_abs = self.sim in SIM_ABS
        bias = -self._lse
        if use_abs:
            def score_mod(score, b, h, q_idx, kv_idx):
                return torch.abs(score) + bias[b, h, kv_idx]
            return score_mod
        return _per_key_bias(bias)

    def _query_bias(self):
        # the -1/2||k||^2 term becomes a per-QUERY constant in the transpose
        return -0.5 * self.ksq if self.sim in SIM_KNORM else 0.0

    # -- application --------------------------------------------------------
    def apply_seq(self, v):
        """`Gamma v` with `v` and the result already in the kernel's own layout,
        (B, heads, H*W, C // heads) -- `apply` without its two layout copies.
        For a caller that produces and consumes that layout directly
        (models/clip_triton.py::planar_envelope)."""
        out, lse = _flex_with_lse(self.q, self.k, v, self._fwd_mod(),
                                  self.block_mask, self.compiled)
        self._lse = lse                                   # reused by the transpose
        return out

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
        out, lseT = _flex_with_lse(self.k, self.q, v, self._rev_mod(),
                                   self.block_mask, self.compiled)
        colsum = torch.exp(lseT + self._query_bias())     # = sum_j Gamma_jk, O(1)
        return seq_to_image(out * colsum.unsqueeze(-1), self.H, self.W, self.heads)

    def matvec(self, x):
        return self.apply(x, transpose=False)

    def rmatvec(self, x):
        return self.apply(x, transpose=True)
