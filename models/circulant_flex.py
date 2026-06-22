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
