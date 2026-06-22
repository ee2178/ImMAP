"""
PyTorch translation of the Julia CirculantAttention `circulant_similarity` family.

The Julia code builds a sparse CSR matrix S of shape (N, N) per batch element,
where N = prod(spatial_dims). For every query position r, only the M^d positions
inside a circulant window of side W = M (odd) around r are stored, with

    S[r, c] = simval(x_r, y_c),                c in window(r)  (circular wrap)

`softmax` is then applied across the W^d neighbours of each row (`dims=1` over the
window view in Julia).

Strategy here (no custom CUDA kernels):
  * The windowed pattern is equivalent to a set of K = W^d integer offsets.
  * For each offset o, S[r, r+o] = simval(x[r], y[r+o]); over all r this is a
    single elementwise expression between x and `roll(y, -o)`, reduced over the
    channel dim. So the dense "window-space" tensor (B, N, K) is built with K
    `torch.roll`s -- fully vectorised, autograd-friendly, GPU-friendly.
  * Column indices come from rolling an index grid by the same offsets, which
    reproduces Julia's `mod((r + o), N)` wrap exactly.
  * The result is assembled into a (batched) torch.sparse_csr_tensor.

Conventions
  * Tensors are (B, C, *spatial), C = channel/feature dim summed over by simval,
    B = batch. (Julia uses (*spatial, C, B); same semantics, transposed layout.)
  * Complex dtypes are supported.
"""

from __future__ import annotations
import itertools
import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Similarity functions  (mirror similarity.jl)
# ----------------------------------------------------------------------------
# Each takes the query feature map `x` (B, C, *spatial) and an already-rolled
# key map `yr` (= y shifted so that yr[..., r] == y[..., r + offset]) and reduces
# over the channel dim (dim=1), returning (B, *spatial).

def _dot(x, yr):                      # DotSimilarity:  sum_c x * conj(y)
    return (x * yr.conj()).sum(dim=1)

def _realdot(x, yr):                  # RealDotSimilarity: Real(sum_c x * conj(y))
    return (x * yr.conj()).real.sum(dim=1)

def _distance(x, yr):                 # DistanceSimilarity: -1/2 sum_c |x - y|^2
    return -0.5 * (x - yr).abs().pow(2).sum(dim=1)

def _pidot(x, yr):                    # PIDotSimilarity: |sum_c x * conj(y)|
    return (x * yr.conj()).sum(dim=1).abs()

def _pidistance(x, yr):               # PIDistanceSimilarity
    s_xx = x.abs().pow(2).sum(dim=1)
    s_xy = (x * yr.conj()).sum(dim=1).abs()
    s_yy = yr.abs().pow(2).sum(dim=1)
    return -0.5 * s_xx + s_xy - 0.5 * s_yy

SIMILARITIES = {
    "dot": _dot,
    "realdot": _realdot,
    "distance": _distance,
    "pidot": _pidot,
    "pidistance": _pidistance,
}

# Which similarities yield a complex-valued S (rest are real even for complex x,y)
_COMPLEX_VALUED = {"dot"}


# ----------------------------------------------------------------------------
# Sparsity pattern  (mirror cartesian_circulant)
# ----------------------------------------------------------------------------
def _offsets(win: int, ndim: int):
    """All W^d offset tuples for an odd window of side `win` over `ndim` dims."""
    assert win % 2 == 1, "window side W must be odd"
    p = (win - 1) // 2
    rng = range(-p, p + 1)
    return list(itertools.product(*([rng] * ndim)))


def circulant_indices(spatial, win: int, device=None, dtype=torch.int64):
    """
    Build the circulant CSR index arrays for spatial shape `spatial` (tuple).

    Returns
        crow   : (N + 1,)  row pointer
        col    : (N, K)    column index of each window slot (K = win^ndim),
                           rows are sorted ascending in column index
        offsets: list[tuple]  the K offsets, aligned with `col`'s last axis
                           *before* sorting (see note)
    The per-row column ordering is sorted so the CSR is canonical; this differs
    from Julia's internal slot ordering but the matrix (and its row-softmax) is
    identical, since softmax over a row is order-invariant.
    """
    ndim = len(spatial)
    N = 1
    for s in spatial:
        N *= s
    K = win ** ndim
    offs = _offsets(win, ndim)

    # index grid laid out row-major (matches flatten of (B, *spatial) -> (B, N))
    grid = torch.arange(N, device=device, dtype=dtype).reshape(spatial)
    cols = []
    for o in offs:
        rolled = torch.roll(grid, shifts=tuple(-d for d in o),
                            dims=tuple(range(ndim)))   # rolled[r] = grid[r + o]
        cols.append(rolled.reshape(N))
    col = torch.stack(cols, dim=1)                     # (N, K)

    # sort columns within each row -> canonical CSR
    col, _ = torch.sort(col, dim=1)

    crow = torch.arange(0, N * K + 1, K, device=device, dtype=dtype)
    return crow, col, offs


# ----------------------------------------------------------------------------
# Window-space similarity (dense): (B, N, K)
# ----------------------------------------------------------------------------
def circulant_similarity_window(simfun, x, y, win: int):
    """
    Dense window-space similarity, the analogue of Julia's `windowview(S)`.

    x, y : (B, C, *spatial)   (complex or real)
    returns
        vals : (B, N, K)  similarity of each query to its K window neighbours
        col  : (N, K)     column (flattened spatial) index of each neighbour
        crow : (N + 1,)   CSR row pointer
    The K axis is aligned with `circulant_indices(...)`'s offset order, i.e.
    unsorted; `to_sparse_csr` below re-sorts to canonical form.
    """
    if isinstance(simfun, str):
        simfun = SIMILARITIES[simfun]
    assert x.shape == y.shape, "x and y must share shape (B, C, *spatial)"
    B, C, *spatial = x.shape
    ndim = len(spatial)
    N = 1
    for s in spatial:
        N *= s
    offs = _offsets(win, ndim)

    out = []
    for o in offs:
        yr = torch.roll(y, shifts=tuple(-d for d in o),
                       dims=tuple(2 + i for i in range(ndim)))  # yr[..,r]=y[..,r+o]
        out.append(simfun(x, yr).reshape(B, N))               # (B, N)
    vals = torch.stack(out, dim=2)                            # (B, N, K)

    # matching (unsorted) column indices
    grid = torch.arange(N, device=x.device, dtype=torch.int64).reshape(spatial)
    col = torch.stack(
        [torch.roll(grid, shifts=tuple(-d for d in o),
                    dims=tuple(range(ndim))).reshape(N) for o in offs],
        dim=1,
    )                                                          # (N, K)
    crow = torch.arange(0, N * (win ** ndim) + 1, win ** ndim,
                        device=x.device, dtype=torch.int64)
    return vals, col, crow


def _window_to_csr(vals, col, crow, N):
    """
    Pack window-space (B, N, K) values + (N, K) cols into a batched CSR of
    shape (B, N, N). Sorts each row's columns to canonical order.
    """
    B, _, K = vals.shape
    col_sorted, perm = torch.sort(col, dim=1)                 # (N, K)
    vals_sorted = torch.gather(
        vals, 2, perm.unsqueeze(0).expand(B, -1, -1)
    )                                                          # (B, N, K)

    col_flat = col_sorted.reshape(N * K)                      # (nnz,)
    vals_flat = vals_sorted.reshape(B, N * K)                 # (B, nnz)

    crow_b = crow.unsqueeze(0).expand(B, -1)
    col_b = col_flat.unsqueeze(0).expand(B, -1)
    return torch.sparse_csr_tensor(crow_b, col_b, vals_flat, size=(B, N, N))


# ----------------------------------------------------------------------------
# Public API  (mirror circulant_similarity / circulant_adjacency)
# ----------------------------------------------------------------------------
def circulant_similarity(simfun, x, y, win: int):
    """
    Sparse circulant similarity S, shape (B, N, N), N = prod(spatial).
    S[b, r, c] = simval(x[b,:,r], y[b,:,c]) for c in the circulant window of r.
    Returned as a batched torch.sparse_csr_tensor.
    """
    B, C, *spatial = x.shape
    N = 1
    for s in spatial:
        N *= s
    vals, col, crow = circulant_similarity_window(simfun, x, y, win)
    return _window_to_csr(vals, col, crow, N)


def circulant_adjacency(simfun, x, y, win: int):
    """
    softmax(circulant_similarity(...)) over each row's window neighbours.
    The softmax is done densely in window space (over K) -- exact and avoids
    sparse-softmax limitations -- then re-packed as a (B, N, N) sparse CSR.
    """
    B, C, *spatial = x.shape
    N = 1
    for s in spatial:
        N *= s
    vals, col, crow = circulant_similarity_window(simfun, x, y, win)
    if vals.is_complex():
        raise ValueError(
            "circulant_adjacency needs a real-valued similarity "
            "(realdot, distance, pidot, pidistance); 'dot' is complex."
        )
    vals = F.softmax(vals, dim=2)                             # over K neighbours
    return _window_to_csr(vals, col, crow, N)


def circulant_apply_window(weights, value, col):
    """
    Apply window-space weights to a value map -- the sparse (S @ value) product,
    done as gather + weighted sum so it is fully differentiable (no batched-CSR
    autograd needed).

    weights : (B, N, K)        e.g. from circulant_adjacency's window form
    value   : (B, Cv, *spatial)
    col     : (N, K)           neighbour column indices (from circulant_indices)
    returns : (B, Cv, *spatial)
    """
    B, Cv, *spatial = value.shape
    N = 1
    for s in spatial:
        N *= s
    vflat = value.reshape(B, Cv, N)
    # gather K neighbours per position: (B, Cv, N, K)
    idx = col.view(1, 1, N, -1).expand(B, Cv, -1, -1)
    neigh = torch.gather(vflat.unsqueeze(-1).expand(-1, -1, -1, idx.shape[-1]), 2, idx)
    out = (weights.unsqueeze(1) * neigh).sum(dim=-1)         # (B, Cv, N)
    return out.reshape(B, Cv, *spatial)


def circulant_attention(simfun, q, k, v, win: int):
    """
    End-to-end local circulant attention, fully differentiable, no custom kernels
    and no reliance on batched-sparse autograd:
        A = softmax_neighbours( simval(q, k) );   out = A @ v   (windowed)
    q, k, v : (B, C, *spatial)   (q,k may be complex; simfun must be real-valued)
    returns : (B, Cv, *spatial)
    """
    w, col, crow = circulant_similarity_window(simfun, q, k, win)
    if w.is_complex():
        raise ValueError("attention needs a real-valued similarity for softmax")
    w = F.softmax(w, dim=2)
    return circulant_apply_window(w, v, col)


def joint_softmax_window(*window_vals):
    """
    Analogue of Julia `joint_softmax`: softmax jointly across the concatenated
    neighbour (K) axes of several window-space tensors, returning the split
    pieces. Each input is (B, N, K_i); softmax is over sum(K_i).
    """
    sizes = [v.shape[2] for v in window_vals]
    cat = torch.cat(window_vals, dim=2)
    sm = F.softmax(cat, dim=2)
    return torch.split(sm, sizes, dim=2)

