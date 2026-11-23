"""
NCSR-LPDS -- guided LPDS with a non-local CENTRALIZED sparse prior (prototype, 2026-11-23).

LGGS lets the guide decide WHICH codes are pooled into a group ENERGY (a threshold on |z|).
This net instead lets the target's neighbours and the guide PREDICT the code's VALUE and puts
the sparsity on what is left over, as NCSR (Dong et al. 2013) does -- the "shared value" row of
DualCDL/group/priors.tex (eq. ncsr / ncsrdual), with the guide added as a second key/value set:

    g(x) = || beta (.) ( A x - [Gamma_zz  Gamma_zw] [A x ; v] ) + (1 - beta) (.) A x ||_{1,lam}
         = || (I - beta Gamma_zz) A x  -  beta Gamma_zw v ||_{1,lam},      v = A_g w

beta in [0, 1] per channel (beta = 0: plain LPDS), w the guide image, A_g the guide's own
analysis conv (initialised as a copy of A). Gamma = [Gamma_zz Gamma_zw] is one windowed
attention under ONE softmax over the self window and every guide window, and it is COMPLEX:

    Gamma_ri = P_ri * omega_ri,   P = joint-softmax( |k_i^H q_r| - 1/2 |k_i|^2 ),
                                  omega_ri = sign(k_i^H q_r)                    (pidistance + transport)

so a matched neighbour's code is turned into the query's phase before it is averaged (complex
codes of a shifted / differently-phased pattern agree up to a phase per pixel, which LGGS never
had to care about because it pools |.|^2). The weights act on every channel of a head alike.

The sweep: with Gamma held fixed, g is lam ||K x - b||_1, K = (I - beta Gamma_zz) A, b =
beta Gamma_zw v, and LPDS's dual prox of g* (f*(z) = <z, b> + box) is a SHIFTED clip:

    x+ = x - tau (E^H E x - y~ + B (I - beta Gamma_zz)^H z)
    xb = x+ + theta (x+ - x)
    z+ = clip_lam( z + (I - beta Gamma_zz) A xb - beta Gamma_zw v )

Gamma is built at the start of a layer from the current estimate's code A x (the pilot; NCSR
builds its means from the previous estimate), and reused for `dK` layers. The guide enters
ONLY through b and the weights -- an offset of the dual variable, never added to x.

Leave-one-out (`include_center=False`, default): the self window omits the pixel itself.
With it, a matched pixel predicts itself and the penalty on it can vanish (beta P_rr -> 1).

COST / MEMORY. Prototype: everything is gathered offset by offset with `torch.roll` (periodic
windows, as the circulant kernels), no materialised (B, M, W^2, H, W) tensor:
  * `_WindowApply` applies Gamma (and its adjoint Gamma^H, the transposed pass the notes call
    for) with a hand-written backward that saves only (Gamma, values);
  * the weights are built under activation checkpointing (recomputed in the backward).
Per layer that is O((W_s^2 + G W_g^2) M H W) work in a Python loop over offsets; fine for a
prototype, a fused kernel later (circatt.py already has the transport forward and the dv pass).
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from models.base import batched_projection, project_conv, set_weight
from models.guided_prox import as_guide_list
from models.lpds import gram
from models.prox import Polynomial
from operators.identity import Identity
from operators.padding import calc_pad_2d
from preprocessing.image import post_process, pre_process
from preprocessing.kspace import _pad_complex, kspace_post_process, kspace_pre_process
from solvers.eigen import power_method
from models.components import Conv2d, ConvTranspose2d


# ---------------------------------------------------------------------------------------------
# windowed (periodic) application of a complex adjacency
# ---------------------------------------------------------------------------------------------
def window_offsets(win, include_center=True):
    """[(dy, dx)] of a win x win window, row-major; the centre dropped unless `include_center`."""
    p = int(win) // 2
    return tuple((dy, dx) for dy in range(-p, p + 1) for dx in range(-p, p + 1)
                 if include_center or (dy, dx) != (0, 0))


def _shift(t, o):
    """(S_o t)_r = t_{r + o}, periodic in the last two axes."""
    return torch.roll(t, shifts=(-o[0], -o[1]), dims=(-2, -1)) if o != (0, 0) else t


def _unshift(t, o):
    """S_o^{-1} = S_{-o} = S_o^H."""
    return torch.roll(t, shifts=(o[0], o[1]), dims=(-2, -1)) if o != (0, 0) else t


def _apply(G, x, offsets):
    """y_r = sum_o G[:, :, o, r] x_{r+o}.  G (B, h, K, H, W), x (B, h, c, H, W)."""
    y = torch.zeros_like(x)
    for j, o in enumerate(offsets):
        y = y + G[:, :, j, None] * _shift(x, o)
    return y


def _apply_adjoint(G, x, offsets):
    """y = Gamma^H x:  y_a = sum_o conj(G[:, :, o, a - o]) x_{a - o}."""
    y = torch.zeros_like(x)
    for j, o in enumerate(offsets):
        y = y + _unshift(G[:, :, j, None].conj() * x, o)
    return y


class _WindowApply(torch.autograd.Function):
    """Gamma x (adjoint=False) or Gamma^H x (adjoint=True), saving only (G, x).

    PyTorch's complex convention (grads are dL/d conj): for y = G S_o x,
    dx = Gamma^H dy, dG_o = sum_c dy conj(S_o x); for y = Gamma^H x, dx = Gamma dy,
    dG_o = sum_c conj(S_o dy) x."""

    @staticmethod
    def forward(ctx, G, x, offsets, adjoint):
        ctx.save_for_backward(G, x)
        ctx.offsets, ctx.adjoint = offsets, adjoint
        with torch.no_grad():
            return (_apply_adjoint if adjoint else _apply)(G, x, offsets)

    @staticmethod
    def backward(ctx, gy):
        G, x = ctx.saved_tensors
        offs, adj = ctx.offsets, ctx.adjoint
        gG = gx = None
        with torch.no_grad():
            if ctx.needs_input_grad[1]:
                gx = (_apply if adj else _apply_adjoint)(G, gy, offs)
            if ctx.needs_input_grad[0]:
                if adj:
                    parts = [(_shift(gy, o).conj() * x).sum(dim=2) for o in offs]
                else:
                    parts = [(gy * _shift(x, o).conj()).sum(dim=2) for o in offs]
                gG = torch.stack(parts, dim=2)
        return gG, gx, None, None


def window_apply(G, x, offsets, adjoint=False):
    return _WindowApply.apply(G, x, tuple(offsets), bool(adjoint))


# ---------------------------------------------------------------------------------------------
# the complex joint adjacency [Gamma_zz Gamma_zw]
# ---------------------------------------------------------------------------------------------
def _branch_scores(q, k, offsets, eps):
    """For every offset: pidistance score |k^H q| - 1/2 |k|^2 and the transport phasor
    sign(k^H q). q, k (B, h, d, H, W) complex -> two (B, h, K, H, W)."""
    s, w = [], []
    for o in offsets:
        ks = _shift(k, o)
        ip = (ks.conj() * q).sum(dim=2)                       # k^H q
        mag = torch.sqrt(ip.real ** 2 + ip.imag ** 2 + eps)   # smooth |k^H q|
        s.append(mag - 0.5 * (ks.real ** 2 + ks.imag ** 2).sum(dim=2))
        w.append(ip / mag)
    return torch.stack(s, dim=2), torch.stack(w, dim=2)


def joint_adjacency(q, k_self, k_guides, self_offsets, guide_offsets, eps=1e-8):
    """[G_self, G_guide_1, ...]: complex weights P * omega under one softmax over all of them."""
    branches = [(k_self, self_offsets)] if len(self_offsets) else []
    branches += [(k, guide_offsets) for k in k_guides]
    scores, phases = zip(*[_branch_scores(q, k, offs, eps) for k, offs in branches])
    P = torch.softmax(torch.cat(scores, dim=2), dim=2)
    G = P.to(phases[0].dtype) * torch.cat(phases, dim=2)
    return torch.split(G, [len(o) for _, o in branches], dim=2)


class NCSRAttention(nn.Module):
    """q = W_theta(A x), k_self = W_phi(A x), k_g = W_phi^g(A_g w), all divided by sqrt(rho),
    shared by every layer; rho noise-adaptive per head.

    The W's are COMPLEX 1x1 transforms M -> Mh (LGGS's are real): a channel of the features can
    then be any phase-rotated mix of the code's subbands, so the similarity and the transport
    phase omega = sign(k^H q) are learned phase relations, not only the codes' own phases.
    All start as the same semi-orthogonal matrix (W W^H = I).

    separate_guide_w: the guide's keys get their OWN W_phi^g (untied per contrast, as LGGS's
    option of the same name); otherwise the guide's codes go through the target's W_phi. The guide
    has no other transform here -- its codes are the values as they are. W_theta (the query)
    is the target's either way."""

    def __init__(self, M, Mh=16, nheads=1, window=7, guide_window=11, include_center=False,
                 rho0=1.0, degrees=0, checkpoint=True, separate_guide_w=False):
        super().__init__()
        if M % nheads or Mh % nheads:
            raise ValueError(f"M={M} and Mh={Mh} must both divide into nheads={nheads}")
        self.M, self.Mh, self.nheads = int(M), int(Mh), int(nheads)
        self.window, self.guide_window = int(window), int(guide_window)
        self.include_center = bool(include_center)
        self.separate_guide_w = bool(separate_guide_w)
        self.self_offsets = window_offsets(self.window, self.include_center)
        self.guide_offsets = window_offsets(self.guide_window, True)
        pix = lambda: Conv2d(M, Mh, 1, bias=False, complex=True)        # noqa: E731
        self.Wtheta, self.Wphi = pix(), pix()
        self.Wphi_g = pix() if self.separate_guide_w else None
        with torch.no_grad():                                  # semi-orthogonal, all equal
            Q, _ = torch.linalg.qr(torch.randn(M, Mh, dtype=torch.complex64))
            W0 = Q.conj().T.reshape(Mh, M, 1, 1).contiguous()  # rows orthonormal: W W^H = I
            for W in (self.Wtheta, self.Wphi, self.Wphi_g):
                if W is not None:
                    set_weight(W, W0.clone())
        self.rho = Polynomial(nheads, degrees=degrees, tau0=rho0)
        self.checkpoint = bool(checkpoint)
        self.eps = 1e-8

    @property
    def guide_Wphi(self):
        """The transform the guide's keys go through: W_phi^g when untied, else W_phi."""
        return self.Wphi if self.Wphi_g is None else self.Wphi_g

    def heads(self, t):
        B, C, H, W = t.shape
        return t.reshape(B, self.nheads, C // self.nheads, H, W)

    def _features(self, u, v_guides, sigma):
        sq = torch.sqrt(self.rho(sigma, ref=u.real).clamp_min(0) + self.eps)     # (., h, ., .)
        sq = sq.repeat_interleave(self.Mh // self.nheads, dim=1)
        q = self.heads(self.Wtheta(u) / sq)
        k = self.heads(self.Wphi(u) / sq)
        kg = [self.heads(self.guide_Wphi(v) / sq) for v in v_guides]
        return q, k, kg

    def forward(self, u, v_guides, sigma=None):
        """The weights [G_self, G_1, ...] for the code u (B, M, H, W) and guide codes v_g."""
        q, k, kg = self._features(u, v_guides, sigma)
        so, go, eps = self.self_offsets, self.guide_offsets, self.eps

        def build(q, k, *kg):
            return joint_adjacency(q, k, list(kg), so, go, eps)

        if self.checkpoint and torch.is_grad_enabled():
            return checkpoint(build, q, k, *kg, use_reentrant=False)
        return build(q, k, *kg)

    @torch.no_grad()
    def project_(self):
        self.rho.project_(lo=0.0)

    def extra_repr(self):
        return f"complex W, separate_guide_w={self.separate_guide_w}"


# ---------------------------------------------------------------------------------------------
class NCSRLPDSLayer(nn.Module):
    """One primal-dual sweep with the centralized prior. Per layer: A, B, A_g, tau, theta, lam,
    beta. The attention transforms live in the net (shared), passed in at call time."""

    def __init__(self, C, M, P=7, stride=2, lam0=1e-3, tau0=0.5, theta0=0.0, beta0=0.5,
                 degrees=0, prox_degrees=None):
        super().__init__()
        self.C, self.M, self.P, self.stride = int(C), int(M), int(P), int(stride)
        self.analysis = Conv2d(C, M, P, stride=stride, bias=False, complex=True)
        self.synthesis = ConvTranspose2d(M, C, P, stride=stride, bias=False, complex=True)
        self.guide_analysis = Conv2d(C, M, P, stride=stride, bias=False, complex=True)
        self.tau = Polynomial(C, degrees=degrees, tau0=tau0)
        self.theta = Polynomial(C, degrees=degrees, tau0=theta0)
        self.lam = Polynomial(M, degrees=degrees if prox_degrees is None else prox_degrees,
                              tau0=lam0)
        self.beta = Polynomial(M, degrees=0, tau0=beta0)

    # -- init / constraints ----------------------------------------------------------------
    @torch.no_grad()
    def init_filters(self):
        w = torch.randn(self.M, self.C, self.P, self.P, dtype=torch.complex64)
        set_weight(self.analysis, w)
        set_weight(self.synthesis, w.conj())

    @torch.no_grad()
    def spectral_normalize(self, size=64, num_iter=100):
        w = self.analysis.weight
        x0 = torch.rand(1, self.C, size, size, dtype=w.dtype)
        L = power_method(lambda x: self.synthesis(self.analysis(x)), x0, num_iter=num_iter)[0]
        s = float(math.sqrt(abs(L)))
        set_weight(self.analysis, self.analysis.weight / s)
        set_weight(self.synthesis, self.synthesis.weight / s)

    @torch.no_grad()
    def copy_guide_conv_(self):
        set_weight(self.guide_analysis, self.analysis.weight.clone())

    @torch.no_grad()
    def project_(self):
        for conv in (self.analysis, self.synthesis, self.guide_analysis):
            project_conv(conv)
        self.tau.project_(lo=0.0)
        self.theta.project_(lo=0.0, hi=1.0)
        self.lam.project_(lo=0.0)
        self.beta.project_(lo=0.0, hi=1.0)

    # -- pieces ------------------------------------------------------------------------------
    @staticmethod
    def clip(z, lam, eps=1e-12):
        """Projection onto |z_m| <= lam_m, elementwise (complex modulus)."""
        return z * torch.clamp(lam / (z.abs() + eps), max=1.0)

    def _K(self, u, Gs, attn, beta):
        """(I - beta Gamma_zz) u, on the code u."""
        if not attn.self_offsets:
            return u
        Gu = window_apply(Gs, attn.heads(u), attn.self_offsets).reshape(u.shape)
        return u - beta * Gu

    def _KH(self, z, Gs, attn, beta):
        """(I - beta Gamma_zz)^H z = z - Gamma_zz^H (beta z)."""
        if not attn.self_offsets:
            return z
        Gz = window_apply(Gs, attn.heads(beta * z), attn.self_offsets, adjoint=True)
        return z - Gz.reshape(z.shape)

    def _offset(self, Gg, v, attn, beta):
        """b = beta sum_g Gamma_zw,g v_g."""
        b = 0
        for G, vg in zip(Gg, v):
            b = b + window_apply(G, attn.heads(vg), attn.guide_offsets).reshape(vg.shape)
        return beta * b

    def adjacency(self, x, guides, attn, sigma, cache, dK):
        """(G_self, [G_guide], v): rebuilt from this layer's code of x every dK layers."""
        v = [self.guide_analysis(w) for w in guides]          # this layer's A_g, every layer
        n = cache.get("n", 0)
        cache["n"] = n + 1
        if "G" not in cache or n % dK == 0:
            Gs = attn(self.analysis(x), v, sigma)
            cache["G"] = Gs if attn.self_offsets else (None, *Gs)
        G = cache["G"]
        return G[0], list(G[1:]), v

    # -- forward -------------------------------------------------------------------------
    def forward(self, state, y_tilde, guides, attn, E=None, sigma=None, cache=None, dK=1):
        cache = {} if cache is None else cache
        beta = self.beta()
        if state is None:                                      # cold start: x = y~
            x = y_tilde
            Gs, Gg, v = self.adjacency(x, guides, attn, sigma, cache, dK)
            u = self._K(self.analysis(x), Gs, attn, beta) - self._offset(Gg, v, attn, beta)
            return (x, self.clip(u, self.lam(sigma, ref=u.real))), cache

        x, z = state
        Gs, Gg, v = self.adjacency(x, guides, attn, sigma, cache, dK)
        tau, theta = self.tau(sigma, ref=x.real), self.theta(sigma, ref=x.real)
        residual = gram(E, x) - y_tilde + self.synthesis(self._KH(z, Gs, attn, beta))
        x_new = x - tau * residual
        x_bar = x_new + theta * (x_new - x)
        u = z + self._K(self.analysis(x_bar), Gs, attn, beta) - self._offset(Gg, v, attn, beta)
        return (x_new, self.clip(u, self.lam(sigma, ref=u.real))), cache


# ---------------------------------------------------------------------------------------------
class NCSRLPDSNet(nn.Module):
    """K untied NCSR-LPDS sweeps, guided by one or more fully-sampled guide images.

    Takes and returns what LGGSNet does: forward(y, guide, E, sigma) -> (x_hat, (x, z)),
    complex, `preproc="kspace"` for reconstruction. Every layer's code is complex (no planar
    state), the attention transforms are shared, A / B / A_g / steps / lam / beta per layer.

    beta0 = 0 starts as (untied) LPDSNet with a dual offset of zero: the guide's influence is
    then learned from nothing, but the attention gets no gradient until beta moves.

    separate_guide_w: the guide's keys through their own complex W_phi^g (per contrast), not
    the target's W_phi; starts equal, so an untrained net is the tied one (NCSRAttention).
    """

    guided = True
    guide_domain = "image"

    def __init__(self, K=30, M=64, C=1, P=7, s=2, Mh=16, nheads=1, window=7, guide_window=11,
                 include_center=False, lam0=1e-3, tau0=0.5, theta0=0.0, beta0=0.5, rho0=1.0,
                 degrees=0, prox_degrees=None, dK=5, preproc="kspace", spectral_init=True,
                 checkpoint=True, guided=True, separate_guide_w=False):
        super().__init__()
        if C != 1:
            raise ValueError("NCSRLPDSNet: C = 1 (a coil-combined complex image) only")
        if preproc not in ("kspace", "image"):
            raise ValueError(f"preproc must be 'kspace' or 'image', got {preproc!r}")
        self.K, self.M, self.C, self.P, self.s = int(K), int(M), int(C), int(P), int(s)
        self.dK, self.preproc, self.pad_stride = int(dK), preproc, int(s)
        self.guided = bool(guided)
        self.attn = NCSRAttention(M, Mh=Mh, nheads=nheads, window=window,
                                  guide_window=guide_window, include_center=include_center,
                                  rho0=rho0, degrees=degrees if prox_degrees is None
                                  else prox_degrees, checkpoint=checkpoint,
                                  separate_guide_w=separate_guide_w)
        proto = NCSRLPDSLayer(C, M, P=P, stride=s, lam0=lam0, tau0=tau0, theta0=theta0,
                              beta0=beta0, degrees=degrees, prox_degrees=prox_degrees)
        proto.init_filters()
        if spectral_init:
            proto.spectral_normalize()
        proto.copy_guide_conv_()
        self.layers = nn.ModuleList([proto] + [copy.deepcopy(proto) for _ in range(self.K - 1)])

    def _prep_guides(self, guide, ref):
        """Each guide mean-subtracted on its own and padded onto y~'s grid (as LGGSNet)."""
        out = []
        for w in as_guide_list(guide):
            w = w.to(ref.dtype)
            w = w - w.mean(dim=(-3, -2, -1), keepdim=True)
            pad = calc_pad_2d(*w.shape[-2:], self.pad_stride)
            if any(pad):
                w = _pad_complex(w, pad)
            if w.shape[-2:] != ref.shape[-2:]:
                raise ValueError(f"guide grid {tuple(w.shape[-2:])} != measurement grid "
                                 f"{tuple(ref.shape[-2:])}")
            out.append(w)
        return out

    def forward(self, y, guide=None, E=None, sigma=None, state=None):
        if E is None:
            E = Identity()
        if self.preproc == "kspace":
            y_tilde, E, params = kspace_pre_process(y, E, self.pad_stride)
            post = kspace_post_process
        else:
            x_adj = y if isinstance(E, Identity) else E.adjoint(y)
            y_tilde, params = pre_process(x_adj, self.pad_stride)
            post = lambda x, p: post_process(x, list(p))     # noqa: E731
        guides = self._prep_guides(guide, y_tilde) if self.guided else []
        cache = {}
        for layer in self.layers:
            state, cache = layer(state, y_tilde, guides, self.attn, E=E, sigma=sigma,
                                 cache=cache, dK=self.dK)
        x, z = state
        return post(x, params), (x, z)

    @torch.no_grad()
    def project(self):
        with batched_projection():
            for layer in self.layers:
                layer.project_()
            self.attn.project_()

    def extra_repr(self):
        a = self.attn
        return (f"K={self.K}, M={self.M}, s={self.s}, Mh={a.Mh}, heads={a.nheads}, "
                f"window={a.window}{'' if a.include_center else ' (LOO)'}, "
                f"guide_window={a.guide_window}, dK={self.dK}")
