"""
Circulant-Sparse Attention for GroupCDL  (PyTorch, no custom kernels).

Builds on circulant_similarity.py and adds what the GroupCDL architecture needs
beyond raw similarity:

  * a composable `Circulant` adjacency object stored in WINDOW SPACE (B, Q, K),
    so every operation below is autograd-friendly end-to-end (the batched sparse
    CSR layout is forward-only for autograd in torch 2.12, so we never rely on it
    for the training graph -- only as an optional materialised export);
  * scalar multiplication  (gamma * Gamma)        -- Alg. 4
  * addition of two adjacencies sharing the BCCB pattern  (A + B) -- Alg. 4
  * the convex combination  Gamma = g*row-sm(S) + (1-g)*Gamma_prev  -- Alg. 4
  * apply  y = Gamma x   (CircAtt, Alg. 2) and transpose  y = Gamma^T x;
  * row-softmax producing a valid (row-stochastic) adjacency;
  * AdjUpdate (Alg. 4) and learned group-thresholding GT (Eq. 11).

Because the two adjacency matrices blended in Alg. 4 share an identical sparsity
pattern, scalar-mult / addition reduce to elementwise ops on the (B, Q, K) window
values -- trivially differentiable, and the convex blend stays row-stochastic
(rows of both operands sum to 1, gamma + (1-gamma) = 1).

Layout convention: feature tensors are (B, C, *spatial); the adjacency acts on the
flattened spatial index Q = prod(spatial), channel-wise (the I_{Mh} (x) Gamma of
Eq. 11), broadcasting the same Gamma across channels.
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from models.circulant_similarity import (
    SIMILARITIES, _offsets, circulant_similarity_window, _window_to_csr,
)


# ----------------------------------------------------------------------------
# Composable adjacency / similarity matrix, stored in window space
# ----------------------------------------------------------------------------
class Circulant:
    """
    BCCB-sparse matrix held as window-space values.

      values : (B, Q, K)  the K = W^d nonzeros of each row  (real or complex)
      col    : (Q, K) long neighbour (flattened-spatial) column index per slot
      crow   : (Q+1,) long CSR row pointer
      spatial: tuple  the Q-grid shape
      win    : int    window side W (odd)

    `col`/`crow` are the shared sparsity pattern; the *same* offset ordering is
    used for every Circulant built from the same (spatial, win), so values across
    two Circulants are slot-aligned and can be added directly.
    """

    __slots__ = ("values", "col", "crow", "spatial", "win")

    def __init__(self, values, col, crow, spatial, win):
        self.values = values
        self.col = col
        self.crow = crow
        self.spatial = tuple(spatial)
        self.win = int(win)

    # -- bookkeeping ---------------------------------------------------------
    @property
    def Q(self):
        q = 1
        for s in self.spatial:
            q *= s
        return q

    @property
    def K(self):
        return self.values.shape[-1]

    def _like(self, new_values):
        return Circulant(new_values, self.col, self.crow, self.spatial, self.win)

    def _check(self, other):
        assert isinstance(other, Circulant), "operand must be a Circulant"
        assert other.spatial == self.spatial and other.win == self.win, \
            "Circulants must share the BCCB sparsity pattern"

    # -- adjacency algebra (Alg. 4) -----------------------------------------
    def __mul__(self, s):
        # scalar (python number or broadcastable tensor) * matrix
        return self._like(self.values * s)

    __rmul__ = __mul__

    def __add__(self, other):
        self._check(other)
        return self._like(self.values + other.values)

    def __sub__(self, other):
        self._check(other)
        return self._like(self.values - other.values)

    def convex(self, other, gamma):
        """gamma * self + (1 - gamma) * other  (gamma scalar or learnable tensor)."""
        self._check(other)
        return self._like(gamma * self.values + (1.0 - gamma) * other.values)

    # -- normalisation -------------------------------------------------------
    def row_softmax(self):
        """row-sm: softmax across each row's K neighbours -> row-stochastic adj."""
        if self.values.is_complex():
            raise ValueError("row-softmax needs a real-valued similarity "
                             "(distance / realdot / pidot / pidistance).")
        return self._like(F.softmax(self.values, dim=-1))

    def row_sums(self):
        """Sum of each row (== 1 for a valid adjacency). Shape (B, Q)."""
        return self.values.sum(dim=-1)

    # -- application: CircAtt  y = Gamma x  (Alg. 2) -------------------------
    def apply(self, x, transpose=False):
        """
        Channel-wise matvec.  x : (B, C, *spatial)  ->  (B, C, *spatial).
        forward:    y[r] = sum_k  w[r,k] * x[col[r,k]]          (gather)
        transpose:  y[c] = sum_{r,k: col[r,k]=c} w[r,k] * x[r]  (scatter)
        Both are fully differentiable (no sparse-autograd dependency).
        """
        B, C, *sp = x.shape
        Q, K = self.col.shape
        assert tuple(sp) == self.spatial, "feature spatial shape must match adjacency"
        xflat = x.reshape(B, C, Q)
        w = self.values                                   # (B, Q, K)
        if not transpose:
            idx = self.col.view(1, 1, Q, K).expand(B, C, -1, -1)
            neigh = torch.gather(
                xflat.unsqueeze(-1).expand(-1, -1, -1, K), 2, idx)   # (B,C,Q,K)
            y = (w.unsqueeze(1) * neigh).sum(dim=-1)                 # (B,C,Q)
        else:
            contrib = w.unsqueeze(1) * xflat.unsqueeze(-1)           # (B,C,Q,K)
            idx = self.col.reshape(1, 1, Q * K).expand(B, C, -1)     # (B,C,Q*K)
            y = torch.zeros(B, C, Q, dtype=contrib.dtype, device=x.device)
            y = y.scatter_add(2, idx, contrib.reshape(B, C, Q * K))  # (B,C,Q)
        return y.reshape(B, C, *sp)

    def matvec(self, x):
        return self.apply(x, transpose=False)

    def rmatvec(self, x):
        return self.apply(x, transpose=True)

    # -- interop / inspection ------------------------------------------------
    def to_sparse_csr(self):
        """Materialise as batched (B, Q, Q) sparse CSR. Forward use / interop only:
        backward through batched sparse ops is unsupported in torch 2.12."""
        return _window_to_csr(self.values, self.col, self.crow, self.Q)

    def to_dense(self):
        """Dense (B, Q, Q) -- for tests / small problems only."""
        B = self.values.shape[0]
        Q, K = self.col.shape
        out = torch.zeros(B, Q, Q, dtype=self.values.dtype, device=self.values.device)
        rows = torch.arange(Q, device=self.values.device).view(1, Q, 1).expand(B, Q, K)
        cols = self.col.view(1, Q, K).expand(B, Q, K)
        out.index_put_((torch.arange(B).view(B, 1, 1).expand(B, Q, K), rows, cols),
                       self.values, accumulate=True)
        return out


# ----------------------------------------------------------------------------
# Builders  (CircSim -> Circulant ;  row-sm o CircSim -> adjacency)
# ----------------------------------------------------------------------------
def circ_similarity(simfun, x, y, win):
    """CircDistSim / CircDotSim -> Circulant (un-normalised similarity)."""
    vals, col, crow = circulant_similarity_window(simfun, x, y, win)
    B, C, *spatial = x.shape
    return Circulant(vals, col, crow, spatial, win)


def circ_adjacency(simfun, x, y, win):
    """row-sm(CircSim(x, y; W)) -> row-stochastic Circulant adjacency."""
    return circ_similarity(simfun, x, y, win).row_softmax()


# ----------------------------------------------------------------------------
# GroupCDL Algorithm 4 -- Adjacency Matrix Update
# ----------------------------------------------------------------------------
def adjacency_update(Gamma_prev, z, W_theta, W_phi, gamma, win,
                     k, dK, simfun="distance"):
    """
    Alg. 4.  z : (B, M, *spatial) latent.  W_theta, W_phi : (Mh, M) pixel-wise
    transforms (shared across layers).  gamma : scalar/tensor in [0,1].

        S       = CircSim(W_theta z, W_phi z; W)
        k == 1                  -> Gamma = row-sm(S)
        (k+1) % dK == 0         -> Gamma = gamma row-sm(S) + (1-gamma) Gamma_prev
        otherwise               -> Gamma = Gamma_prev
    """
    if k != 1 and (k + 1) % dK != 0:
        return Gamma_prev                              # no update this layer
    zt = _pixelwise(W_theta, z)                        # (B, Mh, *spatial)
    zp = _pixelwise(W_phi, z)
    new = circ_adjacency(simfun, zt, zp, win)          # row-sm(S)
    if k == 1:
        return new
    return new.convex(Gamma_prev, gamma)               # scalar-mult + addition


# ----------------------------------------------------------------------------
# GroupCDL Eq. 11 -- learned group-thresholding  (uses CircAtt apply)
# ----------------------------------------------------------------------------
def _pixelwise(Wmat, z):
    """Apply (out,in) matrix per pixel: z (B,in,*spatial) -> (B,out,*spatial)."""
    B, Cin, *sp = z.shape
    Q = 1
    for s in sp:
        Q *= s
    out = torch.einsum("oi,biq->boq", Wmat.to(z.dtype), z.reshape(B, Cin, Q))
    return out.reshape(B, Wmat.shape[0], *sp)


def _abs2(a):
    return (a.real ** 2 + a.imag ** 2) if a.is_complex() else a ** 2


def group_threshold(z, Gamma, tau, W_alpha, W_beta, eps=1e-8):
    """
    Eq. 11:
        xi = W_beta sqrt( (I_{Mh} (x) Gamma) (W_alpha^T z)^2 )
        GT = z * relu(1 - tau / xi)
    z : (B, M, *spatial) (real or complex).  W_alpha : (M, Mh), W_beta : (M, Mh)>=0.
    tau : (M,) >=0 noise-adaptive threshold (broadcast over space).
    The adjacency Gamma (Q x Q, real) is applied channel-wise in the compressed
    Mh-domain via CircAtt -- the differentiable apply above.
    """
    a = _pixelwise(W_alpha.transpose(-2, -1), z)       # (B, Mh, *spatial) = W_alpha^T z
    energy = Gamma.apply(_abs2(a))                     # (I (x) Gamma)(.)^2, (B,Mh,*sp)
    xi = _pixelwise(W_beta, torch.sqrt(energy + eps))  # (B, M, *spatial), >= 0
    shape = [1, z.shape[1]] + [1] * (z.dim() - 2)
    factor = torch.clamp(1.0 - tau.view(shape) / (xi + eps), min=0.0)
    return z * factor                                  # complex z * real factor ok
