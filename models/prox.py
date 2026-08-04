"""
Proximal operators for the unrolled networks, with subgradients.

PyTorch port of Sljiva's `src/networks/layers.jl` (Polynomial, SoftThreshold,
FenchelProx), the `GroupThreshold` half of `src/networks/group.jl`, and all of
`src/networks/mg_group.jl` (the closed-form subgradients d g).

Every prox here exposes the same three-method interface, which is what makes
the multigrid V-cycle possible:

    prox(z, sigma, cache)          ->  (z_new, cache)     the prox itself
    prox.subgradient(z, sigma, c)  ->  (g, cache)         an element of dg(z)
    prox.project_()                                        constraint projection

`cache` is an explicit dict threaded through a level's layers.  It replaces
Lux's `st` NamedTuple: it holds the group-attention adjacency Gamma so that a
level computes it once and reuses / blends it across its iterations, and it is
what lets the V-cycle's post-smoother inherit the pre-smoother's adjacency for
free (Julia: `V.listaB(..., ps.listaB, st_A)`).

On dg
-----
The generic subgradient is the gradient of the Moreau envelope,

    dg(z) = z - prox_g(z),

which is exactly lambda * sign(z) outside the shrinkage region and 0 inside it
for the l1 case -- correct support, correct direction, bounded everywhere.  It
costs one extra prox evaluation and needs no new code per prox.

`GroupThreshold` overrides it with the rigorous chain rule through
Walpha / Gamma / Wbeta (mg_group.jl, "Rigorous"), and also offers the cheap
envelope approximation ("Simple") and the Moreau fallback.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.circulant_attention import Circulant, circ_adjacency, _abs2
from models.circulant_flex import (FLEX_SIMS, SIM_ABS, FlexAdjacency,
                                   get_block_mask)
from models.circulant_triton import HAVE_TRITON, TritonAdjacency

TRITON_SIMS = ("pidistance", "pidot")

EPS = 1e-8


# ===========================================================================
#  noise-level bookkeeping
# ===========================================================================
def as_noise_map(sigma, ref):
    """Normalise `sigma` to a real tensor broadcastable against (B, C, H, W).

    Accepts None, a python float, a 0-d tensor, a (B,) vector, a (B,1,1,1)
    per-image level, or a full (B,1,H,W) noise map.
    """
    rdtype = ref.real.dtype if torch.is_complex(ref) else ref.dtype
    if sigma is None:
        sigma = 0.0
    if not torch.is_tensor(sigma):
        return torch.as_tensor(float(sigma), device=ref.device, dtype=rdtype).view(1, 1, 1, 1)
    sigma = sigma.to(device=ref.device, dtype=rdtype)
    if sigma.dim() == 0:
        return sigma.view(1, 1, 1, 1)
    if sigma.dim() == 1:                       # (B,) -> per-image scalar
        return sigma.view(-1, 1, 1, 1)
    while sigma.dim() < 4:
        sigma = sigma.unsqueeze(-1)
    return sigma


def resize_noise(sigma, size):
    """Resample a spatial noise map onto a `size = (H, W)` grid (no-op if flat)."""
    if not torch.is_tensor(sigma) or sigma.dim() != 4:
        return sigma
    if sigma.shape[-2:] == tuple(size) or (sigma.shape[-1] == 1 and sigma.shape[-2] == 1):
        return sigma
    return F.interpolate(sigma, size=tuple(size), mode="bilinear", align_corners=False)


# ===========================================================================
#  Polynomial -- noise-adaptive, per-channel scalar field
# ===========================================================================
class Polynomial(nn.Module):
    """`f(sigma) = sum_d w_d sigma^d`, one coefficient vector per channel.

    Weight is stored (degrees+1, channels) and evaluated by Horner's rule, the
    same as Julia's `tensor_evalpoly`.  `degrees=0` is the common case and
    short-circuits to a constant, so no sigma is even required.
    """

    def __init__(self, channels, degrees=0, tau0=1e-2):
        super().__init__()
        self.channels, self.degrees = int(channels), int(degrees)
        w = torch.zeros(self.degrees + 1, self.channels)
        w[0] = float(tau0)
        self.weight = nn.Parameter(w)

    def forward(self, sigma=None, ref=None):
        w = self.weight
        if self.degrees == 0:
            return w[0].view(1, -1, 1, 1)
        s = as_noise_map(sigma, w if ref is None else ref)
        y = w[self.degrees].view(1, -1, 1, 1) * torch.ones_like(s)
        for i in range(self.degrees - 1, -1, -1):
            y = y * s + w[i].view(1, -1, 1, 1)
        return y

    @torch.no_grad()
    def project_(self, lo=0.0, hi=None):
        self.weight.clamp_(min=lo, max=hi)

    def extra_repr(self):
        return f"channels={self.channels}, degrees={self.degrees}"


# ===========================================================================
#  real-weight pixel-wise (1x1) transforms, complex-safe, group-aware
# ===========================================================================
class PixelConv(nn.Module):
    """1x1 convolution with REAL weights applied to a possibly-complex input.

    A real matrix W acting on a complex feature map is fully realisable as
    `W Re(z) + i W Im(z)`; keeping W real matches the paper's
    W_theta, W_phi, W_alpha in R^{Mh x M} and halves the parameter count.

    `transpose_apply` gives W^T (cout -> cin) using the *same* parameter -- the
    trick mg_group.jl uses to get W_alpha^T without a second weight
    ("apply the Wbeta ConvTranspose with Walpha's params").
    """

    def __init__(self, cin, cout, groups=1, bias=False):
        super().__init__()
        self.cin, self.cout, self.groups = int(cin), int(cout), int(groups)
        self.conv = nn.Conv2d(cin, cout, 1, bias=bias, groups=groups)

    @property
    def weight(self):
        return self.conv.weight            # (cout, cin // groups, 1, 1)

    def forward(self, x):
        if torch.is_complex(x):
            return torch.complex(self.conv(x.real), self.conv(x.imag))
        return self.conv(x)

    def transpose_apply(self, x):
        w = self.conv.weight
        if torch.is_complex(x):
            return torch.complex(F.conv_transpose2d(x.real, w, groups=self.groups),
                                 F.conv_transpose2d(x.imag, w, groups=self.groups))
        return F.conv_transpose2d(x, w, groups=self.groups)


# ===========================================================================
#  Soft thresholding
# ===========================================================================
def soft_threshold(x, t):
    """`sgn(x) relu(|x| - t)`; correct for complex x (sgn = x / |x|)."""
    return torch.sgn(x) * F.relu(x.abs() - t)


class SoftThreshold(nn.Module):
    """Elementwise prox of `tau(sigma) ||z||_1`, tau per-channel and >= 0."""

    def __init__(self, channels, tau0=1e-2, degrees=0):
        super().__init__()
        self.tau = Polynomial(channels, degrees=degrees, tau0=tau0)
        self.channels = channels

    def threshold(self, z, sigma):
        return self.tau(sigma, ref=z)

    def forward(self, z, sigma=None, cache=None):
        return soft_threshold(z, self.threshold(z, sigma)), cache

    def subgradient(self, z, sigma=None, cache=None):
        # Moreau: z - ST(z, t) = sgn(z) min(|z|, t) -- exact and cheap.
        zt, cache = self.forward(z, sigma, cache)
        return z - zt, cache

    @torch.no_grad()
    def project_(self):
        self.tau.project_(lo=0.0)


# ===========================================================================
#  Group thresholding (GroupCDL Eq. 11) + its subgradients
# ===========================================================================
class GroupThreshold(nn.Module):
    """Learned group-sparse thresholding with a circulant nonlocal adjacency.

        xi = W_beta sqrt( (I (x) Gamma) (W_alpha z)^2 )
        GT(z) = z * relu(1 - tau / xi)

    Gamma is built from a windowed similarity of `q = W_theta z`, `k = W_phi z`
    (each scaled by a learned, noise-adaptive rho), row-softmaxed, and blended
    with the previous adjacency every `dK` layers -- Alg. 4 of the GroupCDL
    paper, matching `group.jl`.

    Two backends, both exposing `.apply(x, transpose=)`:

    * `gather` -- materialises the (B, Q, K) window values as a `Circulant`.
      Exact, complex-capable, differentiable on CPU, and supports the convex
      adjacency blend of Alg. 4.  Memory is O(B Mh Q W^2), which at W=35 on a
      full-resolution latent is gigabytes.
    * `flex`   -- `FlexAdjacency`, fused, nothing of size (B, Q, K) allocated.
      This is what makes a realistic window tractable.  Real-valued queries and
      keys only (complex ones are stacked as [Re; Im], which leaves distance and
      realdot similarities unchanged), no adjacency blend, and torch's
      FlexAttention has no CPU backward -- so it trains on GPU only.

    Multi-head: `gather` folds the head axis into the batch axis so the
    single-head `Circulant` machinery is reused verbatim; `flex` carries heads
    natively.
    """

    def __init__(self, M, Mh=None, nheads=1, window=15, tau0=1e-3, degrees=0,
                 gamma0=0.8, gamma_degrees=0, rho0=1.0, rho_degrees=0,
                 rho_inv=True, sim_fun="distance", dK=1,
                 init_strategy="spectral_norm", subgrad_mode="rigorous",
                 attn_backend="gather", flex_block_size=128, flex_compile_mask=None,
                 triton_block_m=64, eps=EPS, **_ignored):
        super().__init__()
        self.M, self.Mh = int(M), (None if Mh is None else int(Mh))
        self.nheads, self.window, self.dK = int(nheads), int(window), max(int(dK), 1)
        self.sim_fun, self.rho_inv, self.eps = sim_fun, bool(rho_inv), float(eps)
        self.subgrad_mode = subgrad_mode
        assert self.window % 2 == 1, "window side must be odd"

        assert attn_backend in ("gather", "flex", "triton")
        self.attn_backend = attn_backend
        self.flex_block_size = int(flex_block_size)
        self.flex_compile_mask = flex_compile_mask
        # Query-block size for the triton kernel. Changes only the tiling and
        # the order of the softmax reduction, never the result in exact
        # arithmetic -- which makes it the handle for measuring how much of a
        # backend-vs-backend gap is just fp32 non-associativity.
        self.triton_block_m = int(triton_block_m)
        self._flex_fn = None          # set by compile_flex()
        if attn_backend == "flex" and sim_fun not in FLEX_SIMS:
            raise ValueError(
                f"FlexAttention cannot fuse sim_fun={sim_fun!r}; expected one "
                f"of {FLEX_SIMS}")
        if attn_backend == "triton":
            # The triton kernel exists FOR the phase-invariant similarities --
            # the ones flex cannot carry on complex features. Anything else is
            # cheaper on flex, so point the config there rather than running a
            # hand-written kernel for no reason.
            if sim_fun not in TRITON_SIMS:
                raise ValueError(
                    f"attn_backend='triton' implements {TRITON_SIMS}; "
                    f"sim_fun={sim_fun!r} is faster on attn_backend='flex'")
            if not HAVE_TRITON:
                raise RuntimeError(
                    "attn_backend='triton' but triton is not importable. It "
                    "ships with PyTorch's Linux CUDA wheels; on this machine "
                    "use 'gather' (exact, slow) or 'flex' with "
                    "sim_fun='distance'.")

        self.grouped = self.Mh is not None
        if self.grouped:
            assert M % nheads == 0 and self.Mh % nheads == 0, \
                "M and Mh must both be divisible by nheads"
            self.Wtheta = PixelConv(M, self.Mh, groups=nheads)
            self.Wphi = PixelConv(M, self.Mh, groups=nheads)
            self.Walpha = PixelConv(M, self.Mh, groups=nheads)
            # W_beta maps Mh -> M but is STORED with W_alpha's (M -> Mh) shape
            # and applied transposed, exactly as Lux's
            # `ConvTranspose((1,1), Mh=>M; groups=h)` does.  Same shape is what
            # lets `_init_alpha_beta` seed W_beta from W_alpha, and what lets
            # the widening helper treat all four transforms identically.
            self.Wbeta = PixelConv(M, self.Mh, groups=nheads)
            self._init_alpha_beta(init_strategy)
            with torch.no_grad():                    # tie W_phi = W_theta at init
                self.Wphi.weight.copy_(self.Wtheta.weight)
        else:
            self.Wtheta = self.Wphi = self.Walpha = self.Wbeta = None

        self.tau = Polynomial(M, degrees=degrees, tau0=tau0)
        self.gamma = Polynomial(nheads, degrees=gamma_degrees, tau0=gamma0)
        self.rho = Polynomial(self.Mh if self.grouped else M,
                              degrees=rho_degrees, tau0=rho0)

    # -- initialisation -----------------------------------------------------
    @torch.no_grad()
    def _init_alpha_beta(self, strategy):
        """Per-head init of (W_alpha, W_beta); ports `_init_*!` from group.jl.

        `semi_orthogonal` : each head's block has orthonormal columns and
            W_beta = |W_alpha|, so W_beta is already non-negative and the first
            `project_()` does not zero half its entries.
        `spectral_norm`   : uniform draw, per-head operator-norm normalisation,
            W_beta = W_alpha (the original recipe -- keeps the known clamp
            pathology, kept for checkpoint compatibility).
        """
        W = self.Walpha.weight                       # (Mh, M // h, 1, 1)
        Mh_h = self.Mh // self.nheads
        if strategy == "semi_orthogonal":
            M_h = self.M // self.nheads
            for h in range(self.nheads):
                blk = torch.linalg.qr(torch.randn(M_h, Mh_h, dtype=W.dtype))[0]
                W[h * Mh_h:(h + 1) * Mh_h, :, 0, 0] = blk[:, :Mh_h].t()
            self.Wbeta.weight.copy_(W.abs())
        elif strategy == "spectral_norm":
            W.uniform_(0.0, 1.0)
            for h in range(self.nheads):
                blk = W[h * Mh_h:(h + 1) * Mh_h, :, 0, 0]
                blk /= torch.linalg.matrix_norm(blk, ord=2) + EPS
            self.Wbeta.weight.copy_(W)
        else:
            raise ValueError(f"unknown init_strategy '{strategy}'")

    # -- head <-> batch folding --------------------------------------------
    def _to_heads(self, x):
        B, C, H, W = x.shape
        return x.reshape(B * self.nheads, C // self.nheads, H, W)

    def _from_heads(self, x, B):
        _, Ch, H, W = x.shape
        return x.reshape(B, Ch * self.nheads, H, W)

    def beta_apply(self, x):
        """`W_beta x`: Mh -> M, the transpose of the stored (M -> Mh) weight."""
        return self.Wbeta.transpose_apply(x)

    def apply_gamma(self, Gamma, x, transpose=False):
        """`(I (x) Gamma) x` channel-wise, complex-safe, head-aware."""
        if isinstance(Gamma, (FlexAdjacency, TritonAdjacency)):
            if torch.is_complex(x):
                raise TypeError(
                    f"the {self.attn_backend} backend applies Gamma to real "
                    f"values only; use attn_backend='gather'")
            return Gamma.apply(x, transpose=transpose)     # heads handled inside
        B = x.shape[0]
        xh = self._to_heads(x)
        if torch.is_complex(xh):
            out = torch.complex(Gamma.apply(xh.real, transpose=transpose),
                                Gamma.apply(xh.imag, transpose=transpose))
        else:
            out = Gamma.apply(xh, transpose=transpose)
        return self._from_heads(out, B)

    # -- flex plumbing ------------------------------------------------------
    def _stack_ri(self, x):
        """complex (B, C, H, W) -> real (B, 2C, H, W), keeping heads contiguous.

        `Re(q^H k)` and `||k||^2` are both preserved by [Re; Im] stacking, so
        the distance and realdot similarities are unchanged.
        """
        if not torch.is_complex(x):
            return x
        B, C, H, W = x.shape
        xr = x.reshape(B, self.nheads, C // self.nheads, H, W)
        return torch.cat([xr.real, xr.imag], dim=2).reshape(B, 2 * C, H, W)

    def _flex_block_mask(self, ref):
        """The BlockMask for this grid.

        Memoised process-wide by `get_block_mask`, NOT per module: an unrolled
        network has one prox per layer per level, every one of them wants the
        same mask for a given grid, and building it torch.compiles.  A 3-level
        MG-GroupCDL ends up with one mask per level instead of one per prox.
        """
        H, W = ref.shape[-2], ref.shape[-1]
        return get_block_mask(H, W, self.window, ref.device, circular=True,
                              BLOCK_SIZE=self.flex_block_size,
                              compile=self.flex_compile_mask)

    def compile_flex(self):
        """torch.compile the fused kernel (call once, after moving to device)."""
        from torch.nn.attention.flex_attention import flex_attention
        self._flex_fn = torch.compile(flex_attention, dynamic=None)
        return self

    # -- adjacency ----------------------------------------------------------
    def _scaled_qk(self, z, sigma):
        # Ungrouped (Mh=None) keeps the query/key as z itself -- the Lux
        # `NoOpLayer` branch of `GroupThreshold`.  The adjacency is still built.
        q = self.Wtheta(z) if self.grouped else z
        k = self.Wphi(z) if self.grouped else z
        rho = self.rho(sigma, ref=z)
        sq = torch.sqrt(rho + self.eps)
        if self.rho_inv:
            return q / sq, k / sq
        return q * sq, k * sq

    def _build_gamma(self, z, sigma):
        q, k = self._scaled_qk(z, sigma)
        if self.attn_backend == "triton":
            # Complex q/k are the point of this backend: it accumulates BOTH
            # Re<q,k> and Im<q,k> so |<q,k>| is available, which no score_mod
            # can reach. Real q/k work too (im is compiled out).
            return TritonAdjacency(q, k, self.window, sim=self.sim_fun,
                                   heads=self.nheads, eps=self.eps,
                                   block_m=self.triton_block_m)
        if self.attn_backend == "flex":
            # The pi- (phase-invariant) similarities need |<q,k>|. For REAL
            # features that is |q.k|, a pointwise function of the score, so the
            # fusion is exact. For COMPLEX ones the stacked score is only
            # Re<q,k>; recovering the modulus needs Im<q,k> as well, a second
            # bilinear form that a score_mod cannot see. Fail here rather than
            # silently attending on the wrong similarity.
            if self.sim_fun in SIM_ABS and q.is_complex():
                raise ValueError(
                    f"sim_fun={self.sim_fun!r} needs |<q,k>|, which "
                    f"FlexAttention cannot form for COMPLEX features: its "
                    f"score is one bilinear form (Re<q,k>) and the modulus "
                    f"needs two. Options: attn_backend='gather' (exact, "
                    f"materialises the window), or sim_fun='distance' with "
                    f"flex (fused, drops phase invariance). Real-valued "
                    f"models can use {self.sim_fun!r} with flex directly.")
            q, k = self._stack_ri(q), self._stack_ri(k)
            return FlexAdjacency(q, k, self.window, sim=self.sim_fun,
                                 heads=self.nheads,
                                 block_mask=self._flex_block_mask(q),
                                 compiled=self._flex_fn)
        return circ_adjacency(self.sim_fun, self._to_heads(q), self._to_heads(k),
                              self.window)

    def gamma_of(self, z, sigma, cache):
        """Fetch / build / blend Gamma, mirroring `GroupThreshold`'s state logic.

        First call builds it; afterwards it is rebuilt every `dK` layers and
        simply reused in between.  On the gather backend the rebuild is a convex
        blend with the previous adjacency (Alg. 4); the flex backend has no
        materialised values to blend, so it re-caches the query/key pair
        outright -- the same trade `groupcdl.py` makes.
        """
        if cache is None:
            cache = {}
        G_prev = cache.get("Gamma")
        if G_prev is None:
            cache["Gamma"], cache["dupdate"] = self._build_gamma(z, sigma), 1
        elif cache.get("dupdate", 0) % self.dK == 0:
            G_new = self._build_gamma(z, sigma)
            if isinstance(G_new, (FlexAdjacency, TritonAdjacency)):
                # No materialised values to blend (Alg. 4's convex combination),
                # so the query/key pair is re-cached outright instead -- the
                # same trade the flex backend makes.
                cache["Gamma"] = G_new
            else:
                g = self.gamma(sigma, ref=z).reshape(-1)          # (nheads,)
                g = g.repeat(z.shape[0]).view(-1, 1, 1)           # (B*h, 1, 1)
                cache["Gamma"] = G_prev._like(
                    G_prev.values + g * (G_new.values - G_prev.values))
        cache["dupdate"] = (cache.get("dupdate", 0) + 1) % self.dK
        return cache["Gamma"], cache

    # -- prox ---------------------------------------------------------------
    def forward(self, z, sigma=None, cache=None):
        tau = self.tau(sigma, ref=z)
        Gamma, cache = self.gamma_of(z, sigma, cache)
        za = self.Walpha(z) if self.grouped else z
        xi_a = torch.sqrt(self.apply_gamma(Gamma, _abs2(za)) + self.eps)
        xi = self.beta_apply(xi_a) if self.grouped else xi_a
        return z * F.relu(1.0 - tau / (xi + self.eps)), cache

    # -- subgradients (mg_group.jl) ----------------------------------------
    def subgradient(self, z, sigma=None, cache=None, mode=None):
        mode = mode or self.subgrad_mode
        if not self.grouped or mode == "moreau":
            zt, cache = self.forward(z, sigma, cache)
            return z - zt, cache
        if mode == "simple":
            return self._subgrad_simple(z, sigma, cache)
        if mode == "rigorous":
            return self._subgrad_rigorous(z, sigma, cache)
        raise ValueError(f"unknown subgradient mode '{mode}'")

    def _subgrad_simple(self, z, sigma, cache):
        """`dg(z) = lambda z / xi`, xi treated as a frozen envelope.

        Agrees with the true subgradient on support and direction and differs
        only in the relative per-pixel weighting; the learned step size alpha
        absorbs the discrepancy.  Half the cost of the rigorous form (no
        transposed attention).
        """
        Gamma, cache = self.gamma_of(z, sigma, cache)
        xi2a = self.apply_gamma(Gamma, self.Walpha(_abs2(z)))
        xi = torch.sqrt(self.beta_apply(xi2a) + self.eps)
        lam = self.tau(sigma, ref=z)
        return lam * z / (xi + self.eps), cache

    def _subgrad_rigorous(self, z, sigma, cache):
        """Exact gradient of the group energy, by the chain rule.

            g(z)  = tau^T W_beta sqrt( Gamma (W_alpha z)^2 )
            u     = W_alpha z ,   xi_a,j = sqrt( sum_k Gamma_jk |u_k|^2 )
            d/du_k  sum_j c_j xi_a,j  =  c . u_k . ( Gamma^T (1 / xi_a) )_k
            grad_z g = W_alpha^T [ c . u . Gamma^T (1 / xi_a) ] ,  c = W_beta^T tau

        Two deliberate departures from `mg_group.jl`'s "rigorous" branch:

        * it forms `Gamma^T ( c u / xi_a )`, which is not the derivative --
          u carries the *inner* index k while 1/xi_a carries the *outer*
          index j, so u does not commute through Gamma^T.  The two agree only
          when Gamma = I.  `tests/test_multigrid.py` checks this version
          against autograd on the energy itself.
        * `c` is `W_beta^T tau` rather than `(W_beta^T 1)` with tau applied
          after `W_alpha^T`.  These coincide for a channel-uniform tau, and
          only this form is right when tau varies per subband (the usual case,
          since tau is a per-channel polynomial in sigma).

        Cost is the same: one forward and one transposed adjacency apply.
        """
        Gamma, cache = self.gamma_of(z, sigma, cache)
        u = self.Walpha(z)                                     # (B, Mh, H, W)
        xi_a = torch.sqrt(self.apply_gamma(Gamma, _abs2(u)) + self.eps)
        c = self.Wbeta(self.tau(sigma, ref=z))                 # W_beta^T tau
        gt = self.apply_gamma(Gamma, 1.0 / (xi_a + self.eps), transpose=True)
        return self.Walpha.transpose_apply(c * u * gt), cache

    # -- constraints --------------------------------------------------------
    @torch.no_grad()
    def project_(self):
        self.tau.project_(lo=0.0)
        self.gamma.project_(lo=0.05, hi=0.95)
        self.rho.project_(lo=0.1)
        if self.grouped:
            self.Wbeta.weight.clamp_(min=0.0)

    def extra_repr(self):
        return (f"M={self.M}, Mh={self.Mh}, nheads={self.nheads}, "
                f"window={self.window}, dK={self.dK}, sim_fun={self.sim_fun}")


# ===========================================================================
#  Fenchel (dual) wrapper -- Moreau's identity
# ===========================================================================
class FenchelProx(nn.Module):
    """`prox_{g*}(z) = z - prox_g(z)`.

    Turns soft-thresholding into clipping and group-thresholding into group
    clipping, which is what the LPDS / dual family of unrollings needs.
    """

    def __init__(self, prox):
        super().__init__()
        self.prox = prox

    def forward(self, z, sigma=None, cache=None):
        zt, cache = self.prox(z, sigma, cache)
        return z - zt, cache

    def subgradient(self, z, sigma=None, cache=None):
        # dg* of the conjugate: fall back to the Moreau envelope of *this* map.
        zt, cache = self.forward(z, sigma, cache)
        return z - zt, cache

    @torch.no_grad()
    def project_(self):
        self.prox.project_()


# ===========================================================================
#  factory
# ===========================================================================
def build_prox(M, window=1, Mh=None, dual=False, **kws):
    """`window > 1` selects the group prox, otherwise plain soft-thresholding.

    Mirrors `LISTALayer`'s prox selection in `lista.jl`.
    """
    if window and window > 1:
        prox = GroupThreshold(M, Mh=Mh, window=window, **kws)
    else:
        prox = SoftThreshold(M, tau0=kws.get("tau0", 1e-2),
                             degrees=kws.get("degrees", 0))
    return FenchelProx(prox) if dual else prox
